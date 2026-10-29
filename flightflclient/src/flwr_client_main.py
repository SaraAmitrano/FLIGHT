from flight_common.logging import build_logger

logger = build_logger()

import argparse
import os
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
import zmq
from grpc import Call, RpcError, StatusCode

from fl_app.client import FlowerClient
from fl_app.client_app import start_client
from fl_app.data import OutDataParams, get_data_path, load_data
from fl_app.exceptions import StopEvaluating, StopTraining
from fl_app.info_data import ClientDetailedStatus, ClientInfoData
from fl_app.model_builder import build_custom_model, build_model
from flight_common.communication.fl_get_data import FLGetDataDetails
from flight_common.fl.subprocess.zmq_fl_utils import (
    ErrorDetail,
    handle_stop_signal,
    send_status_update,
)
from flight_common.job.dataset import Dataset
from flight_common.job.learning_algorithm import CustomLearningAlgorithm
from flight_common.response_return import ResponseReturn
from flight_common.monitoring import MonitoringRecorder
from flight_common.value_container import ValueContainer
from flight_common.zmq_client import ZmqClient
from flight_common.zmq_poller import Poller
from flight_common.zmq_socket import Socket


class FLClientData:
    def __init__(
        self,
        context: zmq.Context[Any],
        req_client: ZmqClient,
        rep_socket: Socket,
        poller: Poller,
        server_address: str,
        details: FLGetDataDetails,
    ) -> None:
        self.context = context
        self.req_client = req_client
        self.rep_socket = rep_socket
        self.poller = poller
        self.server_address = server_address
        self.details = details

    def send_status_update(
        self, job_id: str, client_id: str, status: str, detail: Any = None
    ) -> ResponseReturn:
        """Send a status update to the server."""
        return send_status_update(
            job_id, client_id, self.req_client, status, detail=detail
        )

    def cleanup(self) -> None:
        """Clean up the resources."""
        self.rep_socket.close()
        self.req_client.close()
        self.context.term()


class FLClientEnvironment:
    def __init__(
        self, ipc_addr: str, server_address: str, timeout: float = 5.0
    ) -> None:
        self.ipc_addr = ipc_addr
        self.server_address = server_address
        self.context: zmq.Context[Any] = zmq.Context()
        self.req_socket: Socket = Socket.from_context(self.context, "SEND")
        self.req_client = ZmqClient(self.req_socket, owns_socket=True)
        self.timeout = timeout

    def load(self) -> FLClientData:
        logger.debug(f"Connecting to main app at {self.ipc_addr}...")
        self.req_socket.connect(self.ipc_addr)

        logger.debug("Requesting data from main app...")
        data = self.req_client.request(dict(command="get_data"), timeout=self.timeout)

        details = FLGetDataDetails.model_validate(data["detail"])

        rep_addr = details.rep_addr

        logger.debug(f"Connecting to REP socket at {rep_addr}...")
        rep_socket = Socket.from_context(self.context, "RECV")
        rep_socket.bind(rep_addr)

        logger.debug("Connected. Creating polling ...")
        poller = Poller()
        poller.register(rep_socket, zmq.POLLIN)

        logger.debug("Polling created successfully.")

        return FLClientData(
            context=self.context,
            req_client=self.req_client,
            rep_socket=rep_socket,
            poller=poller,
            server_address=self.server_address,
            details=details,
        )


stop_running: ValueContainer[bool] = ValueContainer()
job_id: ValueContainer[str] = ValueContainer()

CLIENT_ID = int(os.environ.get("CLIENT_ID", "-1"))  # Default to 0

if CLIENT_ID == -1:
    raise ValueError("CLIENT_ID environment variable must be set to a valid integer")

def _is_expected_grpc_shutdown(error: Exception, status: str | None) -> bool:
    """Return whether gRPC closed because an accepted stop is already in progress."""
    if (
        status != "stopping"
        or not isinstance(error, RpcError)
        or not isinstance(error, Call)
    ):
        return False

    try:
        if error.code() != StatusCode.UNAVAILABLE:
            return False
        details = error.details()
    except Exception:
        return False

    if not isinstance(details, str):
        return False
    normalized_details = " ".join(details.casefold().split())
    return "closed" in normalized_details and any(
        transport in normalized_details for transport in ("socket", "channel")
    )


def set_seed(seed: int) -> None:
    """Seed the random generators used by the client."""
    # Called before building the model: every client of the job starts from
    # the same initial weights.
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  # global RNG used by libraries
    torch.manual_seed(seed)  # also seeds CUDA on all devices


def perform_training(
    job_id: str, data: FLClientData, train_data: OutDataParams, device: torch.device,
    monitoring: MonitoringRecorder | None = None,
) -> int:
    """Run federated learning training process."""
    training_params = data.details.params
    logger.debug(f"Received training data: {training_params}")

    status_code = 0

    logger.info(f"Starting FL process for job_id {job_id}")

    num_rounds = training_params.num_rounds
    epochs = training_params.epochs_per_round

    metrics_names = training_params.evaluation_metrics
    loss_name = training_params.loss_function
    optimizer = training_params.optimizer
    lr = training_params.learning_rate

    proximal_mu = training_params.aggregation_model.params.get("proximal_mu")

    dataset = training_params.dataset
    task = training_params.task
    model_name = training_params.learning_algorithm
    logger.debug(f"Building model {model_name} for dataset {dataset.value}...")

    input_dim = train_data["input_dim"]

    if isinstance(model_name, CustomLearningAlgorithm):
        model = build_custom_model(dataset, model_name, input_dim)
    else:
        model = build_model(dataset, model_name, input_dim)

    model.to(device)

    train_loader = train_data["train_loader"]
    train_eval_loader = train_data["train_eval_loader"]
    val_loader = train_data["val_loader"]
    test_loader = train_data["test_loader"]

    info_data = ClientInfoData(
        client_id=CLIENT_ID,
        cur_round=0,
        num_rounds=num_rounds,
        job_id=job_id,
        rep_socket=data.rep_socket,
        poller=data.poller,
        req_client=data.req_client,
        status="running",
        detailed_status=ClientDetailedStatus(
            msg="startup",
            subset=None,
            round=0,
            rounds=num_rounds,
            epoch=0,
            epochs=epochs,
        ),
    )

    failure_data: ErrorDetail | None = None

    try:
        start_client(
            grpc_max_message_length=538_145_477,
            server_address=data.server_address,
            client=FlowerClient(
                info_data=info_data,
                model=model,
                epochs=epochs,
                train_loader=train_loader,
                train_eval_loader=train_eval_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                loss_name=loss_name,
                metrics_names=metrics_names,
                optimizer=optimizer,
                lr=lr,
                device=device,
                dataset=dataset,
                task=task,
                proximal_mu=proximal_mu,
                model_input_dim=input_dim,
                monitoring=monitoring,
            ).to_client(),
        )

        if info_data.status is not None and info_data.status != "stopping":
            logger.info(
                f"Previous status: {info_data.status}. "
                f"Completed FL process for job_id {job_id}."
            )
            info_data.status = "completed"
    except (StopTraining, StopEvaluating) as e:
        logger.info(f"Received stop command: {e}")
        info_data.update_detailed_status(msg="stopping")
        info_data.status = "stopping"
    except Exception as e:
        if _is_expected_grpc_shutdown(e, info_data.status):
            logger.info(
                "gRPC transport closed while the client was stopping; "
                "completing the requested stop."
            )
        else:
            logger.exception("Training failed")
            info_data.update_detailed_status(msg="failed")
            failure_data = ErrorDetail(msg="failed", error=str(e))
            info_data.status = "failed"
            status_code = 1
    finally:
        logger.debug("Run FL cleanup")
        if info_data.status is not None:
            if info_data.status == "stopping":
                info_data.status = "stopped"

            # send status
            logger.debug(f"Sending final status update: {info_data.status}")
            info_data.update_detailed_status(msg=info_data.status)
            _ = data.send_status_update(
                job_id,
                f"client-{CLIENT_ID}",
                info_data.status,
                detail=failure_data
                if failure_data is not None
                else info_data.detailed_status,
            )
            logger.debug("Status update sent.")
        # clean the resources
        data.cleanup()

    return status_code


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="FL Client")
        parser.add_argument(
            "--ipc-addr", type=str, required=True, help="IPC address for ZeroMQ"
        )
        parser.add_argument(
            "--server-address",
            type=str,
            required=True,
            help="Server address for Flower",
        )

        args = parser.parse_args()
    except Exception:
        logger.exception("Error while starting")
        sys.exit(1)

    # Register signal handler for stop command
    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGINT, handle_stop_signal)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # data = retrieve_data(args)
        client_data = FLClientEnvironment(
            ipc_addr=args.ipc_addr, server_address=args.server_address
        ).load()
    except Exception:
        logger.exception("Error while retrieving data")
        sys.exit(1)
    params = client_data.details.params

    job_id.set(client_data.details.job_id)
    os.environ["FLIGHT_MONITOR_JOB_ID"] = job_id()
    _ = client_data.send_status_update(
        job_id(), f"client-{CLIENT_ID}", "running", dict(remove_data_ok=True)
    )

    monitoring = MonitoringRecorder(job_id(), f"client-{CLIENT_ID}")
    monitoring.start()
    try:
        t_preprocessing = time.perf_counter()
        learning_algorithm = (
            params.learning_algorithm.params.base_model
            if isinstance(params.learning_algorithm, CustomLearningAlgorithm)
            else params.learning_algorithm
        )

        logger.info(f"Using seed {params.seed}")
        set_seed(params.seed)

        partition_path = get_data_path(CLIENT_ID, params.dataset, job_id())
        train_data = load_data(
            params.dataset,
            params.task,
            learning_algorithm,
            params.batch_size,
            partition_path,
            seed=params.seed,
        )
        monitoring.event("phase", phase="preprocessing", duration_s=time.perf_counter() - t_preprocessing,
                         partition_path=str(partition_path),
                         sample_counts={split: len(train_data[key].dataset) for split, key in
                                        (("train", "train_loader"), ("validation", "val_loader"), ("test", "test_loader"))})
        status_code = perform_training(job_id(), client_data, train_data, device, monitoring)
        sys.exit(status_code)
    except Exception:
        logger.exception("Error during training")
        sys.exit(1)
    finally:
        monitoring.stop()
