from flight_common.logging import build_logger

logger = build_logger()

import asyncio
import contextlib
import os
import signal
import sys
from asyncio.subprocess import Process
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol

import zmq
import zmq.asyncio

from app.process_data import JobData, ProcessData
from flight_common.asyncio_utils import is_running
from flight_common.fl.utils import get_job_id_key
from flight_common.fl.zmq import send_command_to_subprocess
from flight_common.network import get_service_addresses, read_network
from flight_common.response_return import ResponseReturn, Status
from flight_common.startup import startup
from flight_common.utils import get_full_class_name
from flight_common.value_container import ValueContainer
from flight_common.zmq_asyncio_client import AsyncZmqClient
from flight_common.zmq_asyncio_poller import Poller
from flight_common.zmq_asyncio_socket import Envelope, Socket


class CommandPreHandler(Protocol):
    async def __call__(
        self,
        context: zmq.asyncio.Context,
        service: str,
        message: dict[str, Any],
    ) -> ResponseReturn: ...


class CommandPostHandler(Protocol):
    async def __call__(
        self,
        context: zmq.asyncio.Context,
        service: str,
        message: dict[str, Any],
        req_clients: dict[str, AsyncZmqClient],
    ) -> ResponseReturn: ...


class ErrorCode:
    SUCCESS = 0
    PROCESS_EXISTS = 1
    UNKNOWN_COMMAND = 2
    START_ERROR = 3
    PROCESS_NOT_FOUND = 4
    SOCKET_NOT_FOUND = 5
    UNKNOWN_ERROR = 6


SERVICE = "flserver"

NETWORK_DATA = read_network()

ZMQ_ADDRESSES = get_service_addresses(NETWORK_DATA, SERVICE, ["orchestrator"], "tcp")

SLEEP_TIME = 0.1

GLOBAL_LOCK = asyncio.Lock()


def create_response(status: Status, status_code: int, **kwargs: Any) -> ResponseReturn:
    return ResponseReturn(status=status, detail=dict(status_code=status_code, **kwargs))


def from_exec_to_global_status(status: str) -> str:
    """Maps execution status to global status."""
    if status in ["running", "stopping"]:
        return status
    if status in ["starting"]:
        return "running"
    return "idle"


processes_datas: dict[str, ProcessData] = {}
params_data: dict[str, dict[str, Any]] = {}

stop_running: ValueContainer[bool] = ValueContainer(value=False)


async def find_and_kill_flwr_subprocess(
    job_id: str, *, wait_for_termination: bool = False, timeout: float = 10.0
) -> None:
    """
    Terminate the running FL server process for the given job_id.

    When wait_for_termination is True, waits for the process to exit.
    When False, only sends the termination signal.
    """
    process_data = processes_datas.get(job_id)

    if process_data is None:
        logger.warning(f"No active process found with job_id: {job_id}")
        return

    process = process_data.process
    status = process_data.last_status
    job_datas = process_data.job_datas

    if process is None:
        logger.info("No active process found.")
        if process_data.last_status not in ["stopped", "completed", "failed"]:
            process_data.last_status = "stopped"
            job_datas.append(JobData(status="stopped"))
        return

    if not await is_running(process):
        logger.info(f"Training process {job_id} already stopped.")
        if process_data.last_status not in ["stopped", "completed", "failed"]:
            process_data.last_status = "stopped"
            job_datas.append(JobData(status="stopped"))
        return

    if status != "stopped":
        process_data.last_status = "stopping"
        job_datas.append(JobData(status="stopping"))

        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            # already dead
            pass
        except Exception as e:
            logger.warning(
                f"Failed to send SIGTERM to process group: {e}; "
                f"falling back to terminating."
            )
            # Fallback to terminating the process directly
            with contextlib.suppress(ProcessLookupError):
                process.terminate()

        logger.info(
            f"Sent stopping training command for job_id: {job_id}, PID: {process.pid}"
        )

    if wait_for_termination:
        try:
            await asyncio.sleep(0.1)  # small grace
            await asyncio.wait_for(process.wait(), timeout)
            logger.info(f"Training process {job_id} successfully stopped.")
        except TimeoutError:
            logger.warning(
                f"Timeout waiting for process {job_id}, killing it forcefully."
            )
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already terminated
            except Exception:
                logger.exception("Force kill PG failed")
                with contextlib.suppress(Exception):
                    process.kill()
            finally:
                with contextlib.suppress(Exception):
                    await process.wait()
        finally:
            process_data.last_status = "stopped"
            job_datas.append(JobData(status="stopped"))


async def handle_command_health(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],  # noqa: ARG001
) -> ResponseReturn:
    return ResponseReturn(
        status="success", detail=dict(code=ErrorCode.SUCCESS, message="Healthy")
    )


async def handle_command_start(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
) -> ResponseReturn:
    # await asyncio.sleep(3)  # Simulate some delay
    job_id: str = message["job_id"]

    if job_id in processes_datas:
        logger.warning(f"Process with job_id {job_id} already running.")
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.PROCESS_EXISTS,
                message=f"Process with job_id {job_id} already running.",
            ),
        )

    return ResponseReturn(
        status="success", detail=dict(code=ErrorCode.SUCCESS, message="FL starting")
    )


async def handle_command_post_start(
    context: zmq.asyncio.Context,
    service: str,  # noqa: ARG001
    message: dict[str, Any],
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    job_id: str = message["job_id"]
    params: dict[str, Any] = message["params"]
    flwr_port: int = message["flwr_port"]

    process: Process | None = None

    try:
        ipc_addr = f"ipc:///tmp/flwr_REQ_{job_id}.ipc"
        ipc_path = Path(ipc_addr.replace("ipc://", "")).resolve()
        if ipc_path.exists():
            logger.debug(f"Removed existing IPC file: {ipc_path}")
            try:
                ipc_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove IPC address {ipc_path}: {e}")

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-uB",
            "flwr_server_main.py",
            "--ipc-addr",
            ipc_addr,
            "--server-address",
            f"{SERVICE}:{flwr_port}",
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
            creationflags=0,
        )

    except (OSError, ValueError):
        message_s = "Failed to start FL process"
        logger.exception(f"Error while starting FL process for job_id {job_id}")

        processes_datas["job_id"] = ProcessData(job_id, process, "failed", message_s)
        return ResponseReturn(
            status="error",
            detail=dict(code=ErrorCode.START_ERROR, message=message_s),
        )
    except Exception:
        message_s = "Error while starting FL process"
        processes_datas["job_id"] = ProcessData(job_id, process, "failed", message_s)
        logger.exception(message_s)
        return ResponseReturn(
            status="error",
            detail=dict(code=ErrorCode.UNKNOWN_ERROR, message=message_s),
        )

    processes_datas[job_id] = ProcessData(job_id, process, "starting")
    logger.info(f"FL job started with job_id {job_id}, PID: {process.pid}")

    logger.debug(f"Building new socket for {job_id}")
    socket_ = Socket.from_context(context, "RECV")
    logger.debug(f"Binding to IPC address: {ipc_addr}")
    socket_.bind(ipc_addr)
    params_data[job_id] = params
    return ResponseReturn(
        status="success",
        detail=dict(
            code=ErrorCode.SUCCESS,
            message="FL started",
            add_rep_socket={get_job_id_key(job_id): socket_},
        ),
    )


async def handle_command_stop(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
) -> ResponseReturn:
    job_id: str = message["job_id"]
    if job_id not in processes_datas:
        logger.info(
            f"Attempt to stop job with job_id {job_id},"
            f" but no active training process to stop"
        )
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.PROCESS_NOT_FOUND,
                message="No active training process to stop",
            ),
        )

    return ResponseReturn(
        status="success",
        detail=dict(code=ErrorCode.SUCCESS, message="Stopping command to be sent."),
    )


async def handle_command_post_stop(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
    req_clients: dict[str, AsyncZmqClient],
) -> ResponseReturn:
    job_id: str = message["job_id"]
    client = req_clients.get(get_job_id_key(job_id))

    if not client:
        if stop_running():
            logger.error(
                f"Attempt to stop job with job_id {job_id},"
                f" but no corresponding socket found."
            )
            return ResponseReturn(
                status="error",
                detail=dict(
                    code=ErrorCode.SOCKET_NOT_FOUND,
                    message="No active training process to stop",
                ),
            )
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.SUCCESS,
                message="No active socket found, assuming process already stopped",
            ),
        )
    process_data = processes_datas[job_id]
    process_data.last_status = "stopping"
    process_data.job_datas.append(JobData(status="stopping"))
    # The server is terminated in case the clients are terminated, so there is no need
    # to send the command at all.

    if stop_running():
        logger.debug(f"Sending stop command to FL job {job_id}")
        response = await send_command_to_subprocess(
            client, command="stop", job_id=job_id
        )

        if response.status == "error":
            detail_error = response.detail.get("error")
            # if detail_error is None or detail_error != "TimeoutError":
            #     logger.error(
            #         f"Error sending stop command to FL server: {detail_error}. "
            #         f"Forcing termination."
            #     )
            #     await find_and_kill_flwr_subprocess()
            #     return ResponseReturn(
            #         status="error",
            #         detail=dict(
            #             code=ErrorCode.UNKNOWN_ERROR,
            #             message=f"Error sending stop command: {detail_error}",
            #         ),
            #     )
            # # If timeout, it could be due to the subprocess already terminating itself by FLWR
            # # still, call the find_and_kill_flwr_subprocess function to ensure cleanup
            # await find_and_kill_flwr_subprocess()
            logger.error(
                f"Error sending stop command to FL server: {detail_error}. "
                f"Forcing termination."
            )
            await find_and_kill_flwr_subprocess(job_id)
            return ResponseReturn(
                status="error",
                detail=dict(
                    code=ErrorCode.UNKNOWN_ERROR,
                    message=f"Error sending stop command: {detail_error}",
                ),
            )

        logger.info(f"FL job {job_id} stopping in progress")

    return ResponseReturn(
        status="success",
        detail=dict(code=ErrorCode.SUCCESS, message="Stopping command sent correctly"),
    )


async def handle_command_get_status(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],  # noqa: ARG001
) -> ResponseReturn:
    response = ResponseReturn(
        status="success",
        detail=dict(code=ErrorCode.SUCCESS, message="Status command received"),
    )

    # logger.info(f"Received status command from {service}: {message}")

    jobs_datas = {
        job_id: deepcopy(process_data.get_job_datas())
        for job_id, process_data in processes_datas.items()
    }
    jobs_datas = {
        job_id: job_datas
        for job_id, job_datas in jobs_datas.items()
        if len(job_datas) > 0
    }
    job_ids = [
        job_id
        for job_id, process_data in processes_datas.items()
        if process_data.last_status not in ["stopped", "completed", "failed"]
    ]

    if job_ids:
        if any(
            process_data.last_status == "stopping"
            for process_data in processes_datas.values()
        ):
            status = "stopping"
        else:
            status = "running"
    else:
        status = "idle"

    response = ResponseReturn(
        status="success",
        detail=dict(
            code=ErrorCode.SUCCESS,
            server=dict(status=status, job_ids=job_ids),
            jobs_datas=jobs_datas,
        ),
    )

    # now that we are going to send the response, we can clean up the job data
    for job_id, process_data in list(processes_datas.items()):
        process_data.job_datas.clear()

        if process_data.last_status in ["stopped", "completed", "failed"]:
            # If the process is stopped, completed, or failed, we can remove it
            processes_datas.pop(job_id)

    return response


async def handle_command_get_data(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,
    message: dict[str, Any],  # noqa: ARG001
) -> ResponseReturn:
    job_id = service.split("_")[-1]
    params = params_data.pop(job_id)
    req_addr = f"ipc:///tmp/flwr_REP_{job_id}.ipc"
    return ResponseReturn(
        status="success", detail=dict(job_id=job_id, params=params, rep_addr=req_addr)
    )


async def handle_command_post_get_data(
    context: zmq.asyncio.Context,
    service: str,
    message: dict[str, Any],  # noqa: ARG001
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    job_id = service.split("_")[-1]
    req_addr = f"ipc:///tmp/flwr_REP_{job_id}.ipc"
    socket_ = Socket.from_context(context, "SEND")
    socket_.connect(req_addr)
    client_ = AsyncZmqClient(socket_, owns_socket=False)
    await client_.start()
    logger.debug(f"Connecting to IPC address: {req_addr}")
    return ResponseReturn(
        status="success",
        detail=dict(
            add_req_client={get_job_id_key(job_id): client_},
        ),
    )


async def handle_command_update_status(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,
    message: dict[str, Any],
) -> ResponseReturn:
    job_id: str = message["job_id"]

    logger.debug(f"Received update_status command from {service}: {message}")

    if job_id not in processes_datas:
        return create_response(
            "error",
            ErrorCode.PROCESS_NOT_FOUND,
            message=f"No active training process found with job_id {job_id}",
        )

    return create_response("success", ErrorCode.SUCCESS, message="Ok")


async def handle_command_post_update_status(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    job_id: str = message["job_id"]
    status: str = message["status"]
    results = message.get("results")
    detail = message.get("detail")

    process_data = processes_datas[job_id]
    process_data.last_status = status
    process_data.job_datas.append(
        JobData(status=status, results=results, detail=detail)
    )

    return create_response("success", ErrorCode.SUCCESS, message="Ok")


async def handle_command_post_default(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],  # noqa: ARG001
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    return ResponseReturn(
        status="success", detail=dict(code=ErrorCode.SUCCESS, message="Ok")
    )


COMMAND_HANDLERS: dict[str, tuple[CommandPreHandler, CommandPostHandler]] = {
    "health": (handle_command_health, handle_command_post_default),
    "start": (handle_command_start, handle_command_post_start),
    "stop": (handle_command_stop, handle_command_post_stop),
    "status": (handle_command_get_status, handle_command_post_default),
    "get_data": (handle_command_get_data, handle_command_post_get_data),
    "update_status": (handle_command_update_status, handle_command_post_update_status),
}


async def handle_unknown_command(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
) -> ResponseReturn:
    command = message.get("command")
    available_commands = ", ".join(COMMAND_HANDLERS.keys())
    return create_response(
        "error",
        500,
        message=f"Unknown command: {command}. Available commands: {available_commands}",
    )


async def handle_request(
    context: zmq.asyncio.Context,
    poller: Poller,
    message_envelope: tuple[dict[str, Any], Envelope],
    service: str,
    socket: Socket,
    req_clients: dict[str, AsyncZmqClient],
    rep_sockets: dict[str, Socket],
    lock: asyncio.Lock,
) -> None:
    message, envelope = message_envelope
    command = message["command"]

    report_logging = command != "status"

    if report_logging:
        logger.debug(f"Received command {command} from {service}.")

    response = create_response(
        "error", 500, message="Unknown error", recv_payload=message
    )

    send_reply = True
    try:
        if command not in COMMAND_HANDLERS:
            response = await handle_unknown_command(
                context=context, service=service, message=message
            )
        else:
            if command != "status":
                logger.debug(
                    f"Running command handler for {command} from {service} "
                    f"with message: {message}"
                )

            async with lock:
                pre_cmd, post_cmd = COMMAND_HANDLERS[command]
                response = await pre_cmd(
                    context=context, service=service, message=message
                )
                jobs_datas: dict[str, list[dict[str, Any]]] = (
                    response.detail.get("jobs_datas") or {}
                )
                report_logging = report_logging or (
                    command != "status"
                    or response.status != "success"
                    or len(jobs_datas) > 0
                )
                if report_logging:
                    logger.debug(
                        f"Finishing command handler for {command} from {service} "
                        f"with message: {message}"
                    )
                if report_logging:
                    logger.debug(f"Sending response to {service} {command}: {response}")
                await socket.send_data(response.to_dict(), envelope)
                send_reply = False
                response = await post_cmd(context, service, message, req_clients)

                add_rep_socket_data = response.detail.pop("add_rep_socket", None)
                if add_rep_socket_data is not None:
                    socket_name: str
                    socket_: Socket
                    socket_name, socket_ = next(iter(add_rep_socket_data.items()))
                    poller.register(socket_, zmq.POLLIN)
                    rep_sockets[socket_name] = socket_
                    logger.debug(f"Added REP socket for {socket_name} to poller")

                add_req_client_data = response.detail.pop("add_req_client", None)
                if add_req_client_data is not None:
                    socket_name: str
                    client_: AsyncZmqClient
                    client_name, client_ = next(iter(add_req_client_data.items()))
                    req_clients[client_name] = client_
                    logger.debug(f"Added REQ socket for {client_name} to poller")

                rm_rep_socket_data = response.detail.pop("rm_rep_socket", None)
                if rm_rep_socket_data is not None:
                    socket_name: str
                    socket_: Socket
                    socket_name = rm_rep_socket_data["socket_name"]
                    socket_ = rep_sockets.pop(socket_name)
                    poller.unregister(socket_)
                    socket_.close()
                    logger.debug(f"Removed REP socket for {socket_name} from poller")

                rm_req_client_data = response.detail.pop("rm_req_socket", None)
                if rm_req_client_data is not None:
                    socket_name: str
                    client_: AsyncZmqClient
                    client_name = rm_req_client_data["socket_name"]
                    client_ = req_clients.pop(client_name)
                    await client_.close()
                    logger.debug(f"Removed REQ socket for {client_name} from poller")

    # except asyncio.CancelledError:
    #     logger.exception(f"Task cancelled: {service}", exc_info=True, stack_info=True)
    #     response = create_response(
    #         "error",
    #         500,
    #         message="Task cancelled",
    #         recv_payload=message,
    #     )
    except Exception as e:
        logger.exception(f"Error processing message {message} from {service}")
        if send_reply:
            response = create_response(
                "error",
                500,
                exception=get_full_class_name(e),
                recv_payload=message,
                error=str(e),
            )
            await socket.send_data(response.to_dict(), envelope)


async def main_task() -> int:
    """Subscribe to FL training commands from the API."""
    logger.debug(f"FL server {SERVICE} starting up...")

    status_code = 0
    context = zmq.asyncio.Context()
    poller = Poller()

    tasks: set[asyncio.Task[None]] = set()

    req_clients: dict[str, AsyncZmqClient] = {}
    rep_sockets: dict[str, Socket] = {}
    try:
        for service, address in ZMQ_ADDRESSES.SEND.items():
            socket = Socket.from_context(context, "SEND")
            socket.connect(address)

            client = AsyncZmqClient(socket, owns_socket=False)
            await client.start()
            req_clients[service] = client

        for service, address in ZMQ_ADDRESSES.RECV.items():
            socket = Socket.from_context(context, "RECV")
            socket.bind(address)
            poller.register(socket, zmq.POLLIN)
            rep_sockets[service] = socket

        logger.info("Server started.")
        listened_services = list(ZMQ_ADDRESSES.RECV.keys())
        logger.debug(
            f"Listening for commands from the following services: "
            f"{listened_services}..."
        )

        while status_code == 0 and not stop_running():
            try:
                # logger.debug("Waiting for commands...")
                events = dict(await poller.poll(50))

                # add_rep_sockets: dict[str, Socket] = {}
                # remove_rep_sockets: dict[str, Socket] = {}
                # add_req_sockets: dict[str, Socket] = {}
                # remove_req_sockets: dict[str, Socket] = {}

                services_sockets = list(rep_sockets.items())

                for service, socket in services_sockets:
                    if socket not in events:
                        continue

                    message_envelope = await socket.recv_data()

                    task = asyncio.create_task(
                        handle_request(
                            context,
                            poller,
                            message_envelope,
                            service,
                            socket,
                            req_clients,
                            rep_sockets,
                            GLOBAL_LOCK,
                        )
                    )
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

            except zmq.ZMQError:
                logger.exception("Error receiving command")

            except Exception:
                logger.exception("Unexpected error")
                status_code = 1

            async with GLOBAL_LOCK:
                removable_keys: list[str] = []
                for job_id, process_data in list(processes_datas.items()):
                    if (
                        get_job_id_key(job_id) not in req_clients
                        or get_job_id_key(job_id) not in rep_sockets
                    ):
                        continue

                    process = process_data.process
                    status = process_data.last_status
                    job_datas = process_data.job_datas

                    status_running = status not in ["stopped", "completed", "failed"]

                    process_running = process is not None and await is_running(process)

                    if status_running and process_running:
                        # If the process is still running and not idle, skip it
                        continue

                    if (
                        not status_running
                        and not process_running
                        and len(job_datas) == 0
                    ):
                        # If the process is idle and not running,
                        # and we are sure no more data is to be sent to orchestrator,
                        # we can remove it
                        removable_keys.append(job_id)
                        continue

                    if process_running:
                        # If the process is idle but is still running, skip it
                        # as it will be close to completion
                        continue

                    # If we reach here, the process is not running and is not idle
                    # if process is not None:
                    #     message = f"Process exited with code {process.returncode}"
                    # else:
                    #     message = "Process did not start"

                    # logger.error(
                    #     message
                    # ) if process is None or process.returncode != 0 else logger.warning(
                    #     message
                    # )
                    # process_data.last_status = "failed"
                    # process_data.job_datas.append(
                    #     JobData(
                    #         status="failed",
                    #         detail=dict(status="error", message=message),
                    #     )
                    # )

                for job_id in removable_keys:
                    key_name = get_job_id_key(job_id)
                    # Remove the sockets associated with the job_id
                    if key_name in req_clients:
                        client_ = req_clients.pop(key_name)
                        await client_.close()
                        logger.debug(f"Removed REQ socket for job_id {job_id}")
                    if key_name in rep_sockets:
                        socket_ = rep_sockets.pop(key_name)
                        poller.unregister(socket_)
                        socket_.close()
                        logger.debug(f"Removed REP socket for job_id {job_id}")

            await asyncio.sleep(SLEEP_TIME)
    except zmq.ZMQError:
        logger.exception("Error instantiating ZeroMQ socket")
        status_code = 1
    except Exception:
        logger.exception("Unexpected error")
        status_code = 2
    finally:
        if stop_running():
            logger.info("Shutting down the flwr servers")
            logger.debug(f"Currently running processes: {list(processes_datas.keys())}")
            for job_id in list(processes_datas.keys()):
                process_data = processes_datas[job_id]
                process = process_data.process
                if get_job_id_key(job_id) in req_clients:
                    message = dict(command="stop", job_id=job_id)
                    _ = await handle_command_post_stop(
                        context=context,
                        service=get_job_id_key(job_id),
                        message=message,
                        req_clients=req_clients,
                    )
                    if process is not None:
                        await asyncio.wait_for(process.wait(), 4)
                        if await is_running(process):
                            logger.warning(
                                f"Process {job_id} did not terminate. "
                                f"Killing it forcefully."
                            )
                            process.kill()
                            await process.wait()
                        logger.info(f"Process {job_id} terminated.")
                else:
                    await find_and_kill_flwr_subprocess(
                        job_id, wait_for_termination=True
                    )
        if req_clients:
            logger.debug(f"Cleaning up REQ sockets: {list(req_clients.keys())}...")
            for client in req_clients.values():
                await client.close()
        if rep_sockets:
            logger.debug(f"Cleaning up REP sockets: {list(rep_sockets.keys())}...")
            for socket in rep_sockets.values():
                poller.unregister(socket)
                socket.close()
        logger.debug("Closing context...")
        context.term()
    return status_code


def main() -> None:
    # # Initial setup
    # init_data()

    startup(main_task)


if __name__ == "__main__":
    main()
