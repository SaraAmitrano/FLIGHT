from flight_common.logging import build_logger

logger = build_logger()

import asyncio
import contextlib
import os
import signal
import sys
from asyncio.subprocess import Process
from pathlib import Path
from time import monotonic
from typing import Any, Protocol
from uuid import UUID

import torch
import zmq
import zmq.asyncio

from app.custom_model_validation import (
    ModelValidationConfig,
    SandboxError,
    validate_custom_model,
)
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
        req_clients: dict[str, AsyncZmqClient],
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


CLIENT_ID = int(os.environ.get("CLIENT_ID", "-1"))  # Default to 0

if CLIENT_ID == -1:
    raise ValueError("CLIENT_ID environment variable must be set to a valid integer")


SERVICE = f"flclient-{CLIENT_ID}"

NETWORK_DATA = read_network()

CENTRALIZED = os.getenv("CENTRALIZED", "0") == "1"
ZMQ_ADDRESSES = get_service_addresses(
    NETWORK_DATA,
    SERVICE,
    ["orchestrator"],
    "tcp",
    {"orchestrator": "ip"} if not CENTRALIZED else None,
)

SLEEP_TIME = 0.1

SAMPLE_DATA_PATH = Path("/") / "sample_data"

GLOBAL_LOCK = asyncio.Lock()

TERMINAL_STATUSES = {"stopped", "completed", "failed"}
STOP_GRACE_PERIOD_SECONDS = 10.0


def create_response(status: Status, status_code: int, **kwargs: Any) -> ResponseReturn:
    return ResponseReturn(status=status, detail=dict(status_code=status_code, **kwargs))


def from_exec_to_global_status(status: str) -> str:
    """Maps execution status to global status."""
    if status in ["running", "stopping"]:
        return status
    if status in ["starting"]:
        return "running"
    return "idle"


# We do not expect multiple processes to run at the same time, but this
# structure is more idiomatic
processes_datas: dict[str, ProcessData] = {}
params_data: dict[str, dict[str, Any]] = {}

stop_running: ValueContainer[bool] = ValueContainer(value=False)


def check_untracked_training_process(
    proc_path: Path = Path("/proc"),
) -> tuple[str, str | None]:
    """Return found/job ID, absent, or inconclusive for untracked workers."""
    inconclusive_reason = None
    try:
        for entry in proc_path.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                argv = [
                    os.fsdecode(arg)
                    for arg in (entry / "cmdline").read_bytes().split(b"\0")
                    if arg
                ]
            except FileNotFoundError:
                continue  # Process exited while /proc was being scanned.
            except OSError as exc:
                inconclusive_reason = f"Cannot inspect PID {entry.name}: {exc}"
                continue
            if not any(Path(arg).name == "flwr_client_main.py" for arg in argv):
                continue
            try:
                ipc_addr = argv[argv.index("--ipc-addr") + 1]
                if "--server-address" not in argv:
                    continue
            except (ValueError, IndexError):
                continue
            prefix, suffix = "ipc:///tmp/flwr_REQ_", ".ipc"
            if not ipc_addr.startswith(prefix) or not ipc_addr.endswith(suffix):
                continue
            try:
                job_id = str(UUID(ipc_addr[len(prefix) : -len(suffix)]))
            except ValueError:
                continue
            return "found", job_id
    except OSError as exc:
        return "inconclusive", f"Cannot scan {proc_path}: {exc}"
    if inconclusive_reason is not None:
        return "inconclusive", inconclusive_reason
    return "absent", None


async def find_and_kill_flwr_subprocess(
    *, wait_for_termination: bool = False, timeout: float = 10.0
) -> bool:
    """
    Terminate the running FL client process.

    When wait_for_termination is True, waits for the process to exit.
    When False, only sends the termination signal.
    """
    process_data = processes_datas.get("job_id")

    if process_data is None:
        logger.warning("No active process found.")
        return True

    process = process_data.process
    job_id = process_data.job_id
    status = process_data.last_status
    job_datas = process_data.job_datas

    if process is None:
        logger.info("No active process found.")
        if process_data.last_status not in TERMINAL_STATUSES:
            process_data.last_status = "stopped"
            job_datas.append(JobData(status="stopped"))
        return True

    if not await is_running(process):
        logger.info(f"Training process {job_id} already stopped.")
        if process_data.last_status not in TERMINAL_STATUSES:
            process_data.last_status = "stopped"
            job_datas.append(JobData(status="stopped"))
        return True

    if status not in TERMINAL_STATUSES:
        if status != "stopping":
            process_data.last_status = "stopping"
            job_datas.append(JobData(status="stopping"))

        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already dead
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

    if not wait_for_termination:
        return False

    try:
        logger.info(f"Training process {job_id} - wait for termination.")
        await asyncio.sleep(0.1)  # small grace
        await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        logger.warning(f"Timeout waiting for process {job_id}, killing it forcefully.")
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already terminated
        except Exception:
            logger.exception("Force kill PG failed")
            with contextlib.suppress(Exception):
                process.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout)

    if not await is_running(process):
        if process_data.last_status not in TERMINAL_STATUSES:
            process_data.last_status = "stopped"
            job_datas.append(JobData(status="stopped"))
        logger.info(f"Training process {job_id} successfully stopped.")
        return True

    logger.error(f"Training process {job_id} is still running after forced stop.")
    return False


def record_flwr_process_exit(process_data: ProcessData) -> bool:
    """Record an exited Flower subprocess without changing terminal jobs."""
    status = process_data.last_status
    if status in TERMINAL_STATUSES:
        return False

    process = process_data.process
    returncode = process.returncode if process is not None else None

    if status == "stopping":
        process_data.last_status = "stopped"
        process_data.job_datas.append(JobData(status="stopped"))
        logger.info(
            f"Training process {process_data.job_id} stopped with exit code "
            f"{returncode}."
        )
        return True

    if process is not None and returncode is None:
        return False

    if process is None:
        message = (
            f"Flower subprocess for job_id {process_data.job_id} is not available"
        )
    else:
        message = (
            f"Flower subprocess for job_id {process_data.job_id} "
            f"exited unexpectedly with code {returncode}"
        )

    detail: dict[str, Any] = {
        "status": "error",
        "message": message,
        "job_id": process_data.job_id,
    }
    if returncode is not None:
        detail["returncode"] = returncode

    logger.error(message)
    process_data.last_status = "failed"
    process_data.job_datas.append(
        JobData(status="failed", message=message, detail=detail)
    )
    return True


async def handle_command_health(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],  # noqa: ARG001
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    return ResponseReturn(
        status="success", detail=dict(code=ErrorCode.SUCCESS, message="Healthy")
    )


async def handle_command_start(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    # await asyncio.sleep(3)  # Simulate some delay
    job_id: str = message["job_id"]

    if "job_id" in processes_datas:
        other_job_id = processes_datas["job_id"].job_id
        if other_job_id == job_id:
            logger.warning(f"Process with job_id {job_id} already running.")
            return ResponseReturn(
                status="success",
                detail=dict(
                    code=ErrorCode.PROCESS_EXISTS,
                    message=f"Process with job_id {job_id} already running.",
                ),
            )
        logger.error(
            f"Attempt to run job_id {job_id} "
            f"but another process with job_id {other_job_id} already running."
        )
        return ResponseReturn(
            status="error",
            detail=dict(
                code=ErrorCode.PROCESS_EXISTS,
                message=f"Attempt to run job_id {job_id} "
                f"but process with job_id {other_job_id} already running.",
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
    server_address: str = message["server_address"]

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
            "flwr_client_main.py",
            "--ipc-addr",
            ipc_addr,
            "--server-address",
            f"{server_address}:{flwr_port}",
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

    processes_datas["job_id"] = ProcessData(job_id, process, "starting")
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
    req_clients: dict[str, AsyncZmqClient],
) -> ResponseReturn:
    job_id: str = message["job_id"]
    client = req_clients.get(get_job_id_key(job_id))
    if not client:
        logger.info(
            f"Attempt to stop job with job_id {job_id},"
            f" but no corresponding socket found."
        )
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.SOCKET_NOT_FOUND,
                message="No active training process to stop",
            ),
        )

    if "job_id" not in processes_datas:
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

    process_data = processes_datas["job_id"]
    if process_data.job_id != job_id:
        logger.error(f"Process mismatch: expected {job_id}, got {process_data.job_id}.")
        return ResponseReturn(
            status="error",
            detail={
                "code": ErrorCode.PROCESS_NOT_FOUND,
                "message": f"Process with job_id {job_id} not found.",
            },
        )

    if process_data.last_status in ["stopped", "completed", "failed"]:
        logger.info(
            f"Process with job_id {job_id} already ended "
            f"with status {process_data.last_status}."
        )
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.SUCCESS,
                message=f"Process with job_id {job_id} already ended "
                f"with status {process_data.last_status}.",
            ),
        )

    if process_data.last_status == "stopping":
        logger.warning(f"Process with job_id {job_id} already stopping.")
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.SUCCESS,
                message=f"Process with job_id {job_id} already stopping.",
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
    process_data = processes_datas.get("job_id")

    if process_data is None or process_data.job_id != job_id:
        logger.info(f"Training process {job_id} already ended.")
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.SUCCESS,
                message="No active training process to stop",
            ),
        )

    if process_data.last_status in TERMINAL_STATUSES:
        logger.info(
            f"Training process {job_id} already ended "
            f"with status {process_data.last_status}."
        )
        return ResponseReturn(
            status="success",
            detail=dict(
                code=ErrorCode.SUCCESS,
                message=f"Process already ended with status "
                f"{process_data.last_status}",
            ),
        )

    if process_data.last_status == "stopping":
        process = process_data.process
        if process is None or not await is_running(process):
            terminated = await find_and_kill_flwr_subprocess(
                wait_for_termination=True
            )
            return ResponseReturn(
                status="success" if terminated else "error",
                detail=dict(
                    code=(
                        ErrorCode.SUCCESS
                        if terminated
                        else ErrorCode.UNKNOWN_ERROR
                    ),
                    message=(
                        "Training process already stopped"
                        if terminated
                        else "Training process did not stop"
                    ),
                ),
            )
        if not process_data.stop_attempt_failed:
            now = monotonic()
            if process_data.stop_requested_at is None:
                process_data.stop_requested_at = now
            if now - process_data.stop_requested_at < STOP_GRACE_PERIOD_SECONDS:
                logger.info(f"Training process {job_id} is already stopping.")
                return ResponseReturn(
                    status="success",
                    detail=dict(
                        code=ErrorCode.SUCCESS,
                        message="Training process already stopping",
                    ),
                )
            logger.warning(
                f"Training process {job_id} did not stop within the grace "
                "period; forcing termination."
            )
            terminated = await find_and_kill_flwr_subprocess(
                wait_for_termination=True
            )
            process_data.stop_attempt_failed = not terminated
            return ResponseReturn(
                status="success" if terminated else "error",
                detail=dict(
                    code=(
                        ErrorCode.SUCCESS
                        if terminated
                        else ErrorCode.UNKNOWN_ERROR
                    ),
                    message=(
                        "Training process stopped after grace period"
                        if terminated
                        else "Training process did not stop"
                    ),
                ),
            )
        logger.warning(f"Retrying failed stop for training process {job_id}.")
        process_data.stop_attempt_failed = False

    client = req_clients.get(get_job_id_key(job_id))

    if not client:
        logger.warning(
            f"No subprocess socket found for job_id {job_id}; forcing termination."
        )
        terminated = await find_and_kill_flwr_subprocess(wait_for_termination=True)
        process_data.stop_attempt_failed = not terminated
        return ResponseReturn(
            status="success" if terminated else "error",
            detail=dict(
                code=ErrorCode.SUCCESS if terminated else ErrorCode.UNKNOWN_ERROR,
                message=(
                    "Training process stopped without subprocess socket"
                    if terminated
                    else "Training process did not stop"
                ),
            ),
        )

    if process_data.last_status != "stopping":
        process_data.last_status = "stopping"
        process_data.job_datas.append(JobData(status="stopping"))
    process_data.stop_attempt_failed = False
    process_data.stop_requested_at = monotonic()
    logger.debug(f"Sending stop command to FL job {job_id}")

    response = await send_command_to_subprocess(client, command="stop", job_id=job_id)

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
        # # If timeout, it could be due to the subprocess already terminating itself by
        # # FLWR. Still, call the find_and_kill_flwr_subprocess function to ensure
        # # cleanup
        # await find_and_kill_flwr_subprocess()
        logger.error(
            f"Error sending stop command to FL server: {detail_error}. "
            f"Forcing termination."
        )
        terminated = await find_and_kill_flwr_subprocess(wait_for_termination=True)
        process_data.stop_attempt_failed = not terminated
        return ResponseReturn(
            status="success" if terminated else "error",
            detail=dict(
                code=ErrorCode.SUCCESS if terminated else ErrorCode.UNKNOWN_ERROR,
                message=(
                    f"Training process stopped after command error: {detail_error}"
                    if terminated
                    else f"Error sending stop command: {detail_error}"
                ),
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
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    response = ResponseReturn(
        status="success",
        detail=dict(code=ErrorCode.SUCCESS, message="Status command received"),
    )

    # logger.info(f"Received status command from {service}: {message}")
    supports_status_ack = "status_batch_acks" in message
    status_batch_acks = message.get("status_batch_acks", {})
    terminal_processes_running: set[str] = set()
    for process_key, process_data in list(processes_datas.items()):
        batch_ack = status_batch_acks.get(process_data.job_id)
        if isinstance(batch_ack, str):
            process_data.acknowledge_status_batch(batch_ack)
        process_running = process_data.process is not None and await is_running(
            process_data.process
        )
        if process_data.last_status in TERMINAL_STATUSES and process_running:
            terminal_processes_running.add(process_key)
        if (
            process_data.last_status in TERMINAL_STATUSES
            and not process_data.job_datas
            and not process_running
        ):
            processes_datas.pop(process_key)

    process_data = processes_datas.get("job_id")
    jobs_datas: dict[str, list[dict[str, Any]]] = {}
    status_batch_ids: dict[str, str] = {}
    if process_data is not None and "job_id" not in terminal_processes_running:
        if supports_status_ack:
            batch_id, job_datas = process_data.get_status_batch()
            if batch_id is not None and job_datas:
                jobs_datas[process_data.job_id] = job_datas
                status_batch_ids[process_data.job_id] = batch_id
        elif process_data.job_datas:
            jobs_datas[process_data.job_id] = process_data.get_job_datas()

    if process_data is not None and "job_id" in terminal_processes_running:
        status = "stopping" if process_data.last_status == "stopped" else "running"
    elif process_data is None:
        check, reason = await asyncio.to_thread(check_untracked_training_process)
        if check == "found":
            logger.warning(
                f"Untracked Flower process for job {reason}; client is busy."
            )
        elif check == "inconclusive":
            logger.warning(f"Cannot prove client is idle: {reason}")
        status = "idle" if check == "absent" else "running"
    else:
        status = from_exec_to_global_status(process_data.last_status)

    # if len(jobs_datas) > 0:
    #     for job_id, job_datas in jobs_datas.items():
    #         logger.info(f"Job ID: {job_id}")
    #         for job_data in job_datas:
    #             logger.info(f"\tStatus: {job_data['status']}, "
    #                         f"Results: {job_data['results']}, "
    #                         f"Detail: {job_data['detail']}")

    response = ResponseReturn(
        status="success",
        detail=dict(
            code=ErrorCode.SUCCESS,
            client=dict(status=status),
            jobs_datas=jobs_datas,
            status_batch_ids=status_batch_ids,
        ),
    )

    if not supports_status_ack and process_data is not None:
        process_data.job_datas.clear()
        if (
            process_data.last_status in TERMINAL_STATUSES
            and "job_id" not in terminal_processes_running
        ):
            processes_datas.pop("job_id")

    return response


async def handle_command_get_data(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,
    message: dict[str, Any],  # noqa: ARG001
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
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
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    job_id: str = message["job_id"]

    logger.debug(f"Received update_status command from {service}: {message}")

    if "job_id" not in processes_datas:
        return create_response(
            "error",
            ErrorCode.PROCESS_NOT_FOUND,
            message=f"No active training process found with job_id {job_id}",
        )

    process_data = processes_datas["job_id"]
    if process_data.job_id != job_id:
        raise ValueError(
            f"Process mismatch: expected {job_id}, got {process_data.job_id}."
        )

    return create_response("success", ErrorCode.SUCCESS, message="Ok")


async def handle_command_post_update_status(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    status: str = message["status"]
    results = message.get("results")
    detail = message.get("detail")

    if "job_id" not in processes_datas:
        logger.info(f"Received update status for unknown job_id. message: {message}")
    else:
        process_data = processes_datas["job_id"]
        if process_data.last_status in TERMINAL_STATUSES:
            if results is not None or detail is not None:
                logger.info(
                    f"Preserving late data for terminal job "
                    f"{process_data.job_id} ({process_data.last_status})."
                )
                process_data.job_datas.append(
                    JobData(status=status, results=results, detail=detail)
                )
            else:
                logger.info(
                    f"Ignoring obsolete status {status} for terminal job "
                    f"{process_data.job_id} ({process_data.last_status})."
                )
            return create_response("success", ErrorCode.SUCCESS, message="Ok")
        process_data.last_status = status
        process_data.job_datas.append(
            JobData(status=status, results=results, detail=detail)
        )

    return create_response("success", ErrorCode.SUCCESS, message="Ok")


def load_data(data_path: Path) -> tuple[Any, Any]:
    import torch  # noqa: PLC0415

    # read .pth data
    X_torch_data = torch.load(  # type: ignore # noqa: N806
        data_path / "X_train.pth", weights_only=True
    )
    y_torch_data = torch.load(  # type: ignore
        data_path / "y_train.pth", weights_only=True
    )
    return X_torch_data, y_torch_data


async def handle_command_validate_custom_model(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
) -> ResponseReturn:
    model_str: str = message["model_str"]
    base_model: str = message["base_model"]
    dataset: str = message["dataset"]
    batch_size: int = message["batch_size"]

    model_kinds = {
        "GEFCom2012Dense": "tabular",
        "GEFCom2012LSTM": "timeseries",
        "ASHRAEDense": "tabular",
        "ASHRAELSTM": "timeseries",
        # "PlantVillageVGG16": "image",
    }

    tasks = {
        "GEFCom2012": "forecasting",
        "ASHRAE": "forecasting",
        "PlantVillage": "classification",
    }

    require_input_dims = {
        ("GEFCom2012", "tabular"): True,
        ("GEFCom2012", "timeseries"): True,
        ("ASHRAE", "tabular"): True,
        ("ASHRAE", "timeseries"): True,
        # ("PlantVillage", "image"): False,
    }

    output_dims = {
        "GEFCom2012": 1,
        "ASHRAE": 1,
        # "PlantVillage": 10,
    }

    model_kind = model_kinds[base_model]

    data_path = SAMPLE_DATA_PATH / dataset / model_kind

    X_train, y_train = load_data(data_path)  # noqa: N806

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Mismatched number of samples between X_train and y_train")
    if X_train.shape[0] != 1:
        raise ValueError("Expected sample data with a single sample for validation")

    task = tasks[dataset]
    require_input_dim = require_input_dims[dataset, model_kind]
    output_dim = output_dims[dataset]

    X_train_batch = torch.concat([X_train] * batch_size)  # noqa: N806

    # one hot encode y_train
    if task == "classification":
        y_train_batch = torch.zeros((batch_size, output_dim), dtype=torch.float32)
        y_value = y_train[0].item()
        if not isinstance(y_value, int):
            raise TypeError("Expected integer class label for classification task")
        if not (0 <= y_value < output_dim):
            raise ValueError(
                f"Class label {y_value} out of range for output dimension {output_dim}"
            )
        for i in range(batch_size):
            y_train_batch[i, y_value] = 1.0
    else:
        y_train_batch = torch.concat([y_train] * batch_size)

    import os  # noqa: PLC0415

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    max_params = 10_000_000  # 10M params cap
    min_memory_size = 6 * 1024 * 1024 * 1024  # 6GB
    cumulative_memory_size = 2 * 1024 * 1024 * 1024  # 2GB

    config = ModelValidationConfig(
        min_memory_size=min_memory_size,
        max_memory_size=cumulative_memory_size,
        max_params=max_params,
        # max_nprocs=4096 * 2,
        device_str=("cuda:0" if CENTRALIZED else "cuda")
        if torch.cuda.is_available()
        else "cpu",
    )

    logger.info("Validating custom model...")
    out = validate_custom_model(
        model_str,
        X=X_train_batch,
        y=y_train_batch,
        require_input_dim=require_input_dim,
        config=config,
        timeout_seconds=60,
    )
    if "exc" in out:
        exc = out["exc"]
        if not isinstance(exc, Exception):
            raise TypeError("Expected exc to be an Exception instance")
        if isinstance(exc, TimeoutError):
            logger.info("Custom model validation timed out")
            return ResponseReturn(
                status="error",
                detail=dict(
                    code=ErrorCode.UNKNOWN_ERROR,
                    message="Custom model validation timed out",
                ),
            )

        if isinstance(exc, SandboxError):
            logger.error("Custom model validation sandbox error")
            logger.error(f"Original exception: {str(exc.original_exc)}")
        else:
            logger.error(f"Custom model validation exception: {str(exc)}")
        traceback = out.get("traceback")
        if traceback is not None:
            logger.error(f"Traceback: {traceback}")
        if isinstance(exc, SandboxError):
            return ResponseReturn(
                status="error",
                detail=dict(
                    code=ErrorCode.UNKNOWN_ERROR,
                    message=f"Custom model validation sandbox error: "
                    f"{str(exc.original_exc)}",
                ),
            )
        return ResponseReturn(
            status="error",
            detail=dict(
                code=ErrorCode.UNKNOWN_ERROR,
                message=f"Custom model validation failed: {str(exc)}",
            ),
        )
    model_shapes = out["model_shapes"]
    logger.info("Done.")

    return ResponseReturn(
        status="success",
        detail=dict(
            code=ErrorCode.SUCCESS,
            message="Validate custom model received",
            model_shapes=model_shapes,
        ),
    )


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
    "validate_custom_model": (
        handle_command_validate_custom_model,
        handle_command_post_default,
    ),
}


async def handle_unknown_command(
    context: zmq.asyncio.Context,  # noqa: ARG001
    service: str,  # noqa: ARG001
    message: dict[str, Any],
    req_clients: dict[str, AsyncZmqClient],  # noqa: ARG001
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
                context=context,
                service=service,
                message=message,
                req_clients=req_clients,
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
                    context=context,
                    service=service,
                    message=message,
                    req_clients=req_clients,
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
    logger.debug(f"FL client {SERVICE} starting up...")

    status_code = 0
    context = zmq.asyncio.Context()
    poller = Poller()

    tasks: set[asyncio.Task[None]] = set()

    req_clients: dict[str, AsyncZmqClient] = {}
    rep_sockets: dict[str, Socket] = {}

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available for FL client")

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

        logger.info("Client started.")
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
                for _, process_data in list(processes_datas.items()):
                    job_id = process_data.job_id
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

                    # The process exited while the job was still active. Record the
                    # terminal event and let the existing ACK flow deliver it.
                    record_flwr_process_exit(process_data)

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
            logger.info("Shutting down the flwr client")
            logger.debug(f"Currently running processes: {list(processes_datas.keys())}")
            process_data = processes_datas.get("job_id")
            if process_data is not None:
                job_id = process_data.job_id
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
                    await find_and_kill_flwr_subprocess(wait_for_termination=True)
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
