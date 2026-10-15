from flight_common.logging import build_logger

logger = build_logger()

import argparse
import base64
import os
import signal
import sys
from typing import Any

import torch
import zmq
from grpc import Call, RpcError, StatusCode
from torch import nn

from fl_app import learning_algorithms
from fl_app.client import FlowerClient
from fl_app.client_app import start_client
from fl_app.custom_model_executor import build_model as build_customized_model
from fl_app.data import OutDataParams, get_data_path, load_data
from fl_app.exceptions import StopEvaluating, StopTraining
from fl_app.info_data import ClientDetailedStatus, ClientInfoData
from flight_common.communication.fl_get_data import FLGetDataDetails
from flight_common.fl.subprocess.zmq_fl_utils import (
    ErrorDetail,
    handle_stop_signal,
    send_status_update,
)
from flight_common.job.dataset import Dataset
from flight_common.job.learning_algorithm import (
    CustomLearningAlgorithm,
    LearningAlgorithm,
)
from flight_common.response_return import ResponseReturn
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

BUILD_MODEL_REGISTRY: dict[Dataset, dict[LearningAlgorithm, dict[str, Any]]] = {
    Dataset.gefcom_2012: {
        "GEFCom2012Dense": {
            "input_dim_required": True,
        },
        "GEFCom2012LSTM": {
            "input_dim_required": True,
        },
    },
    Dataset.ashrae: {
        "ASHRAEDense": {
            "input_dim_required": True,
        },
        "ASHRAELSTM": {
            "input_dim_required": True,
        },
    },
    Dataset.plant_village: {
        "PlantVillageVGG16": {
            "input_dim_required": False,
        },
    },
}


def build_custom_model(
    dataset: Dataset,
    model_name: CustomLearningAlgorithm,
    input_dim: int | None,
) -> nn.Module:
    dataset_registry = BUILD_MODEL_REGISTRY.get(dataset)
    if dataset_registry is None:
        raise ValueError(f"Dataset {dataset} is not supported.")

    custom_params = model_name.params
    base_model = custom_params.base_model
    model_code = base64.b64decode(custom_params.model_code).decode("utf-8")

    model_registry = dataset_registry.get(base_model)

    if model_registry is None:
        raise ValueError(f"Model {base_model} is not supported for dataset {dataset}.")

    model_args: dict[str, Any] = dict()
    if model_registry["input_dim_required"]:
        if input_dim is None:
            raise ValueError(
                f"Model {base_model} requires input_dim to be specified "
                f"for dataset {dataset}."
            )
        model_args.update(input_dim=input_dim)

    return build_customized_model(model_code)(**model_args)


# Create your model function which constructs a fresh model
def build_model(
    dataset: Dataset, model_name: LearningAlgorithm, input_dim: int | None
) -> nn.Module:
    """Build a model based on the given parameters."""
    dataset_registry = BUILD_MODEL_REGISTRY.get(dataset)
    if dataset_registry is None:
        raise ValueError(f"Dataset {dataset} is not supported.")

    model_registry = dataset_registry.get(model_name)
    if model_registry is None:
        raise ValueError(f"Model {model_name} is not supported for dataset {dataset}.")

    model_args: dict[str, Any] = dict()

    if model_registry["input_dim_required"]:
        if input_dim is None:
            raise ValueError(
                f"Model {model_name} requires input_dim to be specified "
                f"for dataset {dataset}."
            )
        model_args.update(input_dim=input_dim)

    return getattr(learning_algorithms, model_name)(**model_args)


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


def perform_training(
    job_id: str, data: FLClientData, train_data: OutDataParams, device: torch.device
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
    _ = client_data.send_status_update(
        job_id(), f"client-{CLIENT_ID}", "running", dict(remove_data_ok=True)
    )

    try:
        learning_algorithm = (
            params.learning_algorithm.params.base_model
            if isinstance(params.learning_algorithm, CustomLearningAlgorithm)
            else params.learning_algorithm
        )

        train_data = load_data(
            params.dataset,
            params.task,
            learning_algorithm,
            params.batch_size,
            get_data_path(CLIENT_ID, params.dataset, job_id()),
        )
        status_code = perform_training(job_id(), client_data, train_data, device)
        sys.exit(status_code)
    except Exception:
        logger.exception("Error during training")
        sys.exit(1)
