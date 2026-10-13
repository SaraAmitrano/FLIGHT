from flight_common.logging import build_logger

logger = build_logger()

import asyncio
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Any

import zmq
import zmq.asyncio

from flight_common.network import get_service_addresses, read_network
from flight_common.response_return import ResponseReturn, create_error_response
from flight_common.startup import startup
from flight_common.utils import get_full_class_name
from flight_common.value_container import ValueContainer
from flight_common.zmq_asyncio_client import AsyncZmqClient
from flight_common.zmq_asyncio_poller import Poller
from flight_common.zmq_asyncio_socket import Envelope, Socket

from . import file_manager as data_manager
from .asyncio_lock import GLOBAL_LOCK
from .command_handler_mgr import get_command_handlers
from .config import (
    CLIENTS_ADDRESSES,
    NUM_CLIENTS,
    SERVER_MAX_PORT,
    SERVER_MIN_PORT,
    get_data_path,
)
from .data_load.module import init_dataset_data
from .data_reset import reset_clients, reset_jobs, reset_server
from .flwr_port import free_flwr_port, init_flwr_ports, is_flwr_port_used
from .service import (
    FLService,
    Service,
    get_fl_client_id,
    get_fl_server_service,
    is_fl_client_service,
)
from .update_status import (
    handle_client_status_update,
    handle_server_status_update,
    remove_job_temp_data,
    update_job_status,
)
from .zmq_orchestrator_utils import send_zeromq_request

if TYPE_CHECKING:
    from flight_common.job_id import JobID

ALL_SERVICES = ["api", "flserver"] + [f"flclient-{i}" for i in range(NUM_CLIENTS)]


SERVICE_TO_ADDRESS = (
    {f"flclient-{i}": CLIENTS_ADDRESSES[i] for i in range(NUM_CLIENTS)}
    if CLIENTS_ADDRESSES is not None
    else None
)

NETWORK_DATA = read_network()

ZMQ_ADDRESSES = get_service_addresses(
    NETWORK_DATA, "orchestrator", ALL_SERVICES, "tcp", SERVICE_TO_ADDRESS
)

AUTO_CHECK_INTERVAL = 5  # seconds

SLEEP_TIME = 0.1

check_time = dict(last=time.time())


def is_check_interval() -> bool:
    cur_time = time.time()

    if cur_time - check_time["last"] >= AUTO_CHECK_INTERVAL:
        check_time["last"] = cur_time
        return True

    return False


def get_service(service: str) -> Service:
    """Get the service name as a Service type."""
    if service not in ALL_SERVICES:
        raise ValueError(
            f"Invalid service name: {service}. Must be one of {ALL_SERVICES}."
        )
    return Service(service)


def get_fl_service(service: str) -> FLService:
    """Get the service name as a FLService type."""
    service = get_service(service)

    if service != get_fl_server_service() and not is_fl_client_service(service):
        raise ValueError(
            f"Invalid FLService name: {service}. Must be 'flserver' or 'flclient-<id>'."
        )
    return FLService(service)


stop_running: ValueContainer[bool] = ValueContainer()


async def handle_request(
    message_envelope: tuple[dict[str, Any], Envelope],
    service: Service,
    socket: Socket,
    req_clients: dict[FLService, AsyncZmqClient],
    lock: asyncio.Lock,
) -> None:
    message, envelope = message_envelope
    command = message["command"]
    address = ZMQ_ADDRESSES.RECV[service]

    logger.debug(f"Received command from {service}: {command}")

    try:
        pre_cmd_handler, post_cmd_handler = get_command_handlers(command, service)
        logger.info(f"Processing command {command}")
        response = await pre_cmd_handler(message)
        # logger.info(
        #     f"Command {command} processed with response: {response}. Sending..."
        # )
        await socket.send_data(response.to_dict(), envelope)
        lock.release()
        # logger.info(
        #     f"Response for command {command} sent successfully. Running post..."
        # )
        if response.detail["status_code"] >= 400:
            return

        _ = await post_cmd_handler(message, req_clients)
        # logger.info(f"Post processing for command {command} completed.")
    except Exception as e:
        logger.exception(f"Error processing message {message} from {service}")
        await socket.send_data(
            create_error_response(
                500,
                exception=get_full_class_name(e),
                address=address,
                recv_payload=message,
                error=str(e),
            ).to_dict(),
            envelope,
        )


async def handle_status_response(
    service: Service, response: ResponseReturn
) -> ResponseReturn:
    response_detail = response.detail
    response_status = response.status

    async with GLOBAL_LOCK:
        with data_manager.lock(verbose=False):
            if service == get_fl_server_service():
                server = data_manager.load_server()
                if not server["online"]:
                    # Since we got a response, the server is online
                    server["online"] = True
                    logger.info("FL server is online.")
                    data_manager.save_server(server)
            else:
                client_id = get_fl_client_id(service)
                client = data_manager.load_client(client_id)
                if not client["online"]:
                    # Since we got a response, the client is online
                    client["online"] = True
                    logger.info(f"FL client {client_id} is online.")
                    data_manager.save_client(client_id, client)

    if response_status == "error":
        logger.error(f"Error status in response from {service}: {response_detail}")
        return create_error_response(
            404, message=f"Error status in response from {service}"
        )

    if "jobs_datas" not in response_detail:
        logger.error(f"Invalid response from {service}: {response_detail}")
        return create_error_response(
            500,
            message=f"Invalid response from {service}",
            detail=response_detail,
        )
    jobs_datas = response_detail["jobs_datas"]

    if len(jobs_datas) == 0:
        # logger.info(f"No updates from {service}.")
        return ResponseReturn(
            status="success",
            detail=dict(message=f"No updates from {service}"),
        )

    responses_returns: dict[JobID, ResponseReturn] = {}
    # logger.info("Received FL server status update")
    async with GLOBAL_LOCK:
        with data_manager.lock():
            for job_id, job_datas in jobs_datas.items():
                logger.debug(
                    f"Processing job_id {job_id} with {len(job_datas)} data entries "
                    f"from {service}..."
                )
                for job_data in job_datas:
                    logger.debug(f"Job data: {job_data}")
                    status = job_data["status"]
                    message = job_data.get("message")
                    results = job_data["results"]
                    detail = job_data["detail"]
                    if service == get_fl_server_service():
                        upd_res = await handle_server_status_update(
                            job_id, status, message, results, detail
                        )
                    else:
                        client_id = get_fl_client_id(service)
                        upd_res = await handle_client_status_update(
                            client_id, job_id, status, message, results, detail
                        )

                    job = upd_res.detail.get("job")
                    if job is not None:
                        res = update_job_status(job, job_id)

                        if res is not None and not isinstance(res, str):
                            job = res

                        data_manager.save_job(job_id, job)
                        data_manager.export_completed_job_results(job_id, job)

                        if job["status"] in [
                            "stopped",
                            "failed",
                            "completed",
                        ]:
                            if is_flwr_port_used(job_id):
                                logger.debug(f"Freeing FLWR port for job {job_id}")
                                free_flwr_port(job_id)

                            # Ensure temporary data is removed
                            remove_job_temp_data(job, job_id)
                    responses_returns[job_id] = upd_res

                # jobs = {
                #     job_id: data_manager.load_job(job_id) for job_id in jobs_datas
                # }
            if service == get_fl_server_service():
                server = data_manager.load_server()
                upd_server = response_detail["server"]
                write_server = False
                # two special cases to avoid going back to idle:
                # 1. job is just created (before the true starting)
                # 2. job failed while starting (in this case)
                if not (
                    server["status"] == "starting" and upd_server["status"] == "idle"
                    # and job_status in ["starting", "failed"]
                ):
                    server["status"] = upd_server["status"]
                    write_server = True
                if server["job_ids"] != upd_server["job_ids"]:
                    server["job_ids"] = upd_server["job_ids"]
                    write_server = True
                if write_server:
                    data_manager.save_server(server)
            else:
                client_id = get_fl_client_id(service)
                client = data_manager.load_client(client_id)
                upd_client = response_detail["client"]
                write_client = False
                if not (
                    client["status"] == "starting" and upd_client["status"] == "idle"
                ):
                    client["status"] = upd_client["status"]
                    write_client = True
                if write_client:
                    data_manager.save_client(client_id, client)

    success_res_details: dict[JobID, dict[str, Any]] = {}
    error_res_details: dict[JobID, dict[str, Any]] = {}
    for job_id, res in responses_returns.items():
        if res.status == "error":
            logger.error(
                f"Error while updating status for {service} {job_id}: {res.detail}"
            )
            error_res_details[job_id] = res.detail
        else:
            logger.debug(
                f"Status update for {service} {job_id} processed successfully."
            )
            success_res_details[job_id] = res.detail

    if len(error_res_details) > 0:
        res = create_error_response(
            500,
            message=f"Errors occurred while processing status updates for {service}",
            detail=error_res_details,
        )
    else:
        res = ResponseReturn(
            status="success",
            detail=dict(
                message=f"Status updates for {service} processed successfully",
                success_details=success_res_details,
            ),
        )
    return res


async def handle_status_response_failure(
    service: FLService,
    req_retries: dict[FLService, int],
    failure_details: dict[str, Any],
    retry_count: int = 10,
) -> None:
    async with GLOBAL_LOCK:
        with data_manager.lock(verbose=False):
            if service == get_fl_server_service():
                server = data_manager.load_server()
                if server["online"]:
                    logger.info(
                        f"Sending request to {service} failed: {failure_details}"
                    )
                    req_retries[service] += 1
                    if req_retries[service] >= retry_count:
                        server["online"] = False
                        logger.error(
                            f"Failed to connect to {service} after "
                            f"{req_retries[service]} retries. Marking as offline."
                        )
                    data_manager.save_server(server)
                return

            client_id = get_fl_client_id(service)
            client = data_manager.load_client(client_id)
            if client["online"]:
                logger.info(f"Sending request to {service} failed: {failure_details}")
                req_retries[service] += 1
                if req_retries[service] >= retry_count:
                    client["online"] = False
                    logger.error(
                        f"Failed to connect to {service} after "
                        f"{req_retries[service]} retries. Marking as offline."
                    )
                    data_manager.save_client(client_id, client)


# Define a global semaphore with N concurrent tasks
SEMAPHORE = asyncio.Semaphore(30)


async def limited_handle_request(
    message_envelope: tuple[dict[str, Any], Envelope],
    service: Service,
    socket: Socket,
    req_clients: dict[FLService, AsyncZmqClient],
    lock: asyncio.Lock,
) -> None:
    async with SEMAPHORE:
        await handle_request(message_envelope, service, socket, req_clients, lock)


async def do_check_updates(
    service: FLService,
    client: AsyncZmqClient,
    req_retries: dict[FLService, int],
) -> None:
    response = await send_zeromq_request(
        service, client, {"command": "status"}, timeout=1.0
    )
    if (
        response.status == "error"
        and response.detail.get("exception") == "TimeoutError"
    ):
        await handle_status_response_failure(service, req_retries, response.detail)
    else:
        if req_retries[service] > 0:
            logger.info(
                f"Reconnected to {service} after {req_retries[service]} retries."
            )
            req_retries[service] = 0
        res = await handle_status_response(service, response)
        if res.status == "error":
            logger.error(f"Error in status response from {service}: {res.detail}")
        elif not res.detail["message"].startswith("No updates from") and not res.detail[
            "message"
        ].startswith("Status updates for"):
            logger.info(f"Status response from {service} processed successfully.")


async def check_updates(
    req_clients: dict[FLService, AsyncZmqClient], req_retries: dict[FLService, int]
) -> None:
    tasks: list[tuple[FLService, Awaitable[None]]] = []
    for service, client in req_clients.items():
        tasks.append((service, do_check_updates(service, client, req_retries)))

    await asyncio.gather(*(task for _, task in tasks))


async def main_task() -> int:
    """Subscribe to FL training commands from the API."""
    status_code = 0

    context = zmq.asyncio.Context()

    req_clients: dict[FLService, AsyncZmqClient] = {}
    rep_sockets: dict[Service, Socket] = {}

    req_retries: dict[FLService, int] = {}

    tasks: set[asyncio.Task[None]] = set()

    try:
        poller = Poller()

        for service, address in ZMQ_ADDRESSES.SEND.items():
            socket = Socket.from_context(context, "SEND")
            socket.connect(address)

            client = AsyncZmqClient(socket, owns_socket=False)
            await client.start()
            req_clients[get_fl_service(service)] = client
            req_retries[get_fl_service(service)] = 0

        for service, address in ZMQ_ADDRESSES.RECV.items():
            socket = Socket.from_context(context, "RECV")
            socket.bind(address)
            poller.register(socket, zmq.POLLIN)
            rep_sockets[get_service(service)] = socket

        listened_services = list(ZMQ_ADDRESSES.RECV.keys())
        logger.info(
            f"Listening for commands from the following services: "
            f"{listened_services}..."
        )

        # Initial check of all services
        # logger.info("Checking initial status of all services...")
        await check_updates(req_clients, req_retries)

        while status_code == 0 and not stop_running:
            try:
                # logger.debug("Waiting for commands...")
                events = dict(await poller.poll(50))

                for service, socket in rep_sockets.items():
                    if socket not in events:
                        continue

                    lock = asyncio.Lock()
                    await lock.acquire()
                    message_envelope = await socket.recv_data()

                    task = asyncio.create_task(
                        limited_handle_request(
                            message_envelope, service, socket, req_clients, lock
                        )
                    )
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

                # similar to heartbeat, check if we need to send status requests
                # to the clients and server
                if is_check_interval():
                    # logger.info("Checking for new updates...")
                    await check_updates(req_clients, req_retries)
            except zmq.ZMQError:
                logger.exception("Error receiving command")

            except Exception:
                logger.exception("Unexpected error")
                status_code = 1

            await asyncio.sleep(SLEEP_TIME)
    except zmq.ZMQError:
        logger.exception("Error instantiating ZeroMQ socket")
        status_code = 1
    except Exception:
        logger.exception("Unexpected error")
        status_code = 2
    finally:
        if stop_running:
            _ = await reset_clients()
            _ = await reset_server()
            _ = await reset_jobs()
        if req_clients:
            logger.debug(f"Cleaning up REQ sockets: {list(req_clients.keys())}...")
            for client in req_clients.values():
                await client.close()
        if rep_sockets:
            logger.debug(f"Cleaning up REP sockets: {list(rep_sockets.keys())}...")
            for socket in rep_sockets.values():
                socket.close()
        context.term()
    return status_code


def init_main_data() -> None:
    """Initialize the files and folders."""
    data_manager.init()

    flwr_min = SERVER_MIN_PORT
    if flwr_min < 0:
        raise ValueError(f"Invalid flwr_min value: {flwr_min}")

    flwr_max = SERVER_MAX_PORT
    if flwr_max < flwr_min:
        raise ValueError(f"Invalid flwr_max value: {flwr_max}")

    init_flwr_ports(flwr_min, flwr_max)

    init_dataset_data(get_data_path(), "Preprocessed")


def main() -> None:
    # Initial setup
    init_main_data()

    startup(main_task)
