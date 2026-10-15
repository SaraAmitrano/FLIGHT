from flight_common.logging import build_logger

logger = build_logger()

import argparse
import base64
import signal
import sys
from typing import Any

import zmq
from flwr.common import parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import Strategy

import fl_app.strategy
from fl_app.custom_aggregation_executor import build_function
from fl_app.info_data import ServerInfoData
from fl_app.server import Server
from fl_app.server_app import start_server
from fl_app.server_config import ServerConfig
from flight_common.communication.fl_get_data import FLGetDataDetails
from flight_common.fl.subprocess.zmq_fl_utils import (
    ErrorDetail,
    ServerDetailedStatus,
    handle_stop_signal,
    send_status_update,
)
from flight_common.global_model import DEFAULT_MODELS_DIR, save_completed_global_model
from flight_common.job.aggregation_model import AggregationModel, AggregationModelData
from flight_common.response_return import ResponseReturn
from flight_common.value_container import ValueContainer
from flight_common.zmq_client import ZmqClient
from flight_common.zmq_poller import Poller
from flight_common.zmq_socket import Socket


class FLServerData:
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
        self, job_id: str, status: str, detail: Any = None
    ) -> ResponseReturn:
        """Send a status update to the server."""
        return send_status_update(
            job_id, "server", self.req_client, status, detail=detail
        )

    def cleanup(self) -> None:
        """Clean up the resources."""
        self.rep_socket.close()
        self.req_client.close()
        self.context.term()


class FLServerEnvironment:
    def __init__(
        self, ipc_addr: str, server_address: str, timeout: float = 5.0
    ) -> None:
        self.ipc_addr = ipc_addr
        self.server_address = server_address
        self.context: zmq.Context[Any] = zmq.Context()
        self.req_socket: Socket = Socket.from_context(self.context, "SEND")
        self.req_client = ZmqClient(self.req_socket, owns_socket=True)
        self.timeout = timeout

    def load(self) -> FLServerData:
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

        return FLServerData(
            context=self.context,
            req_client=self.req_client,
            rep_socket=rep_socket,
            poller=poller,
            server_address=self.server_address,
            details=details,
        )


job_id: ValueContainer[str] = ValueContainer()


def get_strategy(aggregation_model: AggregationModelData, num_clients: int) -> Strategy:
    if aggregation_model.type == AggregationModel.custom:
        # logger.info(f"Using aggregation model: {aggregation_model.type.value}")
        func = aggregation_model.params.pop("func")

        func_str = base64.b64decode(func).decode("utf-8")

        logger.info("Using aggregation function:")
        logger.info(func_str)

        aggregation_model.params["aggregate_fn"] = build_function(func_str)

    strategy = getattr(fl_app.strategy, aggregation_model.type.value)(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        **aggregation_model.params,
    )
    logger.debug(f"Using strategy: {strategy}")
    return strategy


def perform_training(job_id: str, data: FLServerData) -> int:
    """Run federated learning training process."""
    training_params = data.details.params
    logger.debug(f"Received training data: {training_params}")

    status_code = 0

    logger.info(f"Starting FL process for job_id {job_id}")

    num_rounds = training_params.num_rounds
    model_dump = getattr(training_params, "model_dump", None)
    configuration = model_dump(mode="json") if callable(model_dump) else None

    info_data = ServerInfoData(
        cur_round=0,
        num_rounds=num_rounds,
        job_id=job_id,
        rep_socket=data.rep_socket,
        poller=data.poller,
        req_client=data.req_client,
        status="running",
        detailed_status=ServerDetailedStatus(msg="startup", round=0, rounds=num_rounds),
    )

    failure_data: ErrorDetail | None = None

    try:
        server = Server(
            client_manager=SimpleClientManager(),
            strategy=get_strategy(
                training_params.aggregation_model, training_params.num_clients
            ),
            info_data=info_data,
        )

        start_server(
            grpc_max_message_length=538_145_477,
            server_address=data.server_address,
            config=ServerConfig(num_rounds=num_rounds, check_timeout=30.0),
            server=server,
        )

        if info_data.status == "running":
            if server.parameters is None or server.model_schema is None:
                raise RuntimeError("Final model parameters or schema are unavailable")
            schema_tensors = server.model_schema.get("tensors")
            model_class = server.model_schema.get("class")
            model_input_dim = server.model_schema.get("input_dim")
            if not isinstance(schema_tensors, list) or not isinstance(model_class, str):
                raise RuntimeError("Client returned an incomplete model schema")
            if not isinstance(configuration, dict):
                raise RuntimeError("Training configuration cannot be serialized")
            parameters = parameters_to_ndarrays(server.parameters)
            artifact_path = save_completed_global_model(
                status=info_data.status,
                job_id=job_id,
                parameters=parameters,
                tensor_schema=schema_tensors,
                configuration=configuration,
                rounds_completed=server.completed_rounds,
                rounds_expected=num_rounds,
                model_class=model_class,
                model_input_dim=(
                    model_input_dim if isinstance(model_input_dim, int) else None
                ),
                models_dir=DEFAULT_MODELS_DIR,
            )
            if artifact_path is None:
                raise RuntimeError("Final model artifact was not published")
            logger.info(
                f"Previous status: {info_data.status}. "
                f"Completed FL process for job_id {job_id}."
            )
            info_data.status = "completed"
    except Exception as e:
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
            info_data.update_detailed_status(
                msg=info_data.status, round=None, rounds=num_rounds
            )
            _ = info_data.maybe_send_status_update_info(detail=failure_data)
            logger.debug("Status update sent.")
        # clean the resources
        data.cleanup()

    return status_code


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="FL Server")
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

    try:
        # data = retrieve_data(args)
        server_data = FLServerEnvironment(
            ipc_addr=args.ipc_addr, server_address=args.server_address
        ).load()
    except Exception:
        logger.exception("Error while retrieving data")
        sys.exit(1)

    job_id.set(server_data.details.job_id)
    _ = server_data.send_status_update(job_id(), "running")

    try:
        status_code = perform_training(job_id(), server_data)
        sys.exit(status_code)
    except Exception:
        logger.exception("Error during training")
        sys.exit(1)
