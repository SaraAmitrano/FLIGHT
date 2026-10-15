import asyncio
import logging
from time import monotonic
from typing import Any, cast

from flight_common.response_return import ResponseReturn
from flight_common.zmq_asyncio_client import AsyncZmqClient

from . import file_manager as data_manager
from .asyncio_lock import GLOBAL_LOCK
from .file_manager.data_types import CommonStopPayload, Job, JobID
from .service import (
    FLService,
    get_fl_client_service,
    get_fl_server_service,
)
from .update_status import TERMINAL
from .zmq_orchestrator_utils import send_zeromq_request

logger = logging.getLogger(__name__)

MAX_STOP_ATTEMPTS = 3
STOP_REQUEST_TIMEOUT_SECONDS = 5.0
STOP_RETRY_DELAY_SECONDS = 5.0
STOP_COMPLETION_TIMEOUT_SECONDS = 30.0
STOP_ACCEPTED_CODE = 0

_stop_attempts: dict[tuple[str, FLService], int] = {}
_stop_acknowledged: dict[tuple[str, FLService], float] = {}
_stop_retry_after: dict[tuple[str, FLService], float] = {}
_stop_tasks: dict[tuple[str, FLService], asyncio.Task[None]] = {}
_cleanup_tasks: set[asyncio.Task[None]] = set()
_exhausted_logged: set[tuple[str, FLService]] = set()


def _active_services(job: Job) -> list[FLService]:
    services: list[FLService] = []
    if job["server_status"]["status"] not in TERMINAL:
        services.append(get_fl_server_service())
    services.extend(
        get_fl_client_service(int(client_id))
        for client_id, status in job["clients_status"].items()
        if status["status"] not in TERMINAL
    )
    return services


async def _send_failed_job_stop(
    key: tuple[str, FLService],
    client: AsyncZmqClient,
    attempt: int,
) -> None:
    job_id, service = key
    payload = CommonStopPayload(command="stop", job_id=JobID(job_id))

    logger.info(
        f"Sending coordinated stop for failed job {job_id} to {service} "
        f"(attempt {attempt}/{MAX_STOP_ATTEMPTS})."
    )
    try:
        response: ResponseReturn = await send_zeromq_request(
            service,
            client,
            cast("dict[str, Any]", payload),
            timeout=STOP_REQUEST_TIMEOUT_SECONDS,
        )
        if _stop_tasks.get(key) is not asyncio.current_task():
            return

        response_code = response.detail.get("code")
        if response.status != "success" or response_code != STOP_ACCEPTED_CODE:
            logger.error(
                f"Coordinated stop for failed job {job_id} to {service} was not "
                f"accepted on attempt {attempt}/{MAX_STOP_ATTEMPTS}: "
                f"status={response.status}, code={response_code!r}, "
                f"detail={response.detail}"
            )
            _stop_retry_after[key] = monotonic() + STOP_RETRY_DELAY_SECONDS
            if attempt >= MAX_STOP_ATTEMPTS:
                _log_exhausted(key)
            return

        # The gateway replies from its pre-handler, before its post-handler stops
        # Flower.  Remember acceptance only until a terminal status is observed.
        _stop_acknowledged[key] = monotonic()
        _stop_retry_after.pop(key, None)
        logger.info(
            f"Coordinated stop for failed job {job_id} accepted by {service}; "
            "waiting for a terminal status."
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        # send_zeromq_request normally converts errors to ResponseReturn, but keep
        # this task self-contained if a caller/test double raises unexpectedly.
        logger.exception(
            f"Unexpected error sending coordinated stop for failed job {job_id} "
            f"to {service} on attempt {attempt}/{MAX_STOP_ATTEMPTS}."
        )
        if _stop_tasks.get(key) is asyncio.current_task():
            _stop_retry_after[key] = monotonic() + STOP_RETRY_DELAY_SECONDS
            if attempt >= MAX_STOP_ATTEMPTS:
                _log_exhausted(key)
    finally:
        if _stop_tasks.get(key) is asyncio.current_task():
            _stop_tasks.pop(key, None)


def _log_exhausted(key: tuple[str, FLService]) -> None:
    if key in _exhausted_logged:
        return
    logger.error(
        f"Giving up coordinated stop for failed job {key[0]} to {key[1]} "
        f"after {MAX_STOP_ATTEMPTS} attempts; the component still has no "
        "terminal status and its state will not be changed artificially."
    )
    _exhausted_logged.add(key)


def _clear_key(key: tuple[str, FLService]) -> None:
    task = _stop_tasks.pop(key, None)
    if task is not None and not task.done():
        task.cancel()
    _stop_attempts.pop(key, None)
    _stop_acknowledged.pop(key, None)
    _stop_retry_after.pop(key, None)
    _exhausted_logged.discard(key)


async def coordinate_failed_jobs(
    req_clients: dict[FLService, AsyncZmqClient],
) -> list[asyncio.Task[None]]:
    """Schedule bounded, idempotent stop requests for active failed-job peers."""
    failed_jobs: list[tuple[str, list[FLService]]] = []
    async with GLOBAL_LOCK:
        with data_manager.lock(verbose=False):
            for job_id, job in data_manager.get_current_jobs():
                if job["status"] == "failed":
                    failed_jobs.append((str(job_id), _active_services(job)))

    active_keys = {
        (job_id, service)
        for job_id, services in failed_jobs
        for service in services
    }
    known_keys = (
        set(_stop_attempts)
        | set(_stop_acknowledged)
        | set(_stop_retry_after)
        | set(_stop_tasks)
        | _exhausted_logged
    )
    for key in known_keys - active_keys:
        _clear_key(key)

    scheduled: list[asyncio.Task[None]] = []
    now = monotonic()
    for key in active_keys:
        task = _stop_tasks.get(key)
        if task is not None and not task.done():
            continue

        accepted_at = _stop_acknowledged.get(key)
        if accepted_at is not None:
            if now - accepted_at < STOP_COMPLETION_TIMEOUT_SECONDS:
                continue
            _stop_acknowledged.pop(key, None)
            logger.warning(
                f"Coordinated stop for failed job {key[0]} to {key[1]} was "
                f"accepted, but no terminal status was observed within "
                f"{STOP_COMPLETION_TIMEOUT_SECONDS:g} seconds."
            )

        attempts = _stop_attempts.get(key, 0)
        if attempts >= MAX_STOP_ATTEMPTS:
            _log_exhausted(key)
            continue
        if now < _stop_retry_after.get(key, 0.0):
            continue

        client = req_clients.get(key[1])
        if client is None:
            _stop_attempts[key] = attempts + 1
            _stop_retry_after[key] = now + STOP_RETRY_DELAY_SECONDS
            logger.error(
                f"Cannot send coordinated stop for failed job {key[0]} to "
                f"{key[1]}: ZeroMQ client is unavailable "
                f"(attempt {_stop_attempts[key]}/{MAX_STOP_ATTEMPTS})."
            )
            if _stop_attempts[key] >= MAX_STOP_ATTEMPTS:
                _log_exhausted(key)
            continue

        attempt = attempts + 1
        _stop_attempts[key] = attempt
        task = asyncio.create_task(_send_failed_job_stop(key, client, attempt))
        _stop_tasks[key] = task
        _cleanup_tasks.add(task)
        task.add_done_callback(_cleanup_tasks.discard)
        scheduled.append(task)

    return scheduled


async def shutdown_failed_job_cleanup() -> None:
    """Cancel outstanding stop requests before ZeroMQ clients are closed."""
    tasks = list(_cleanup_tasks)
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    for key in list(
        set(_stop_attempts)
        | set(_stop_acknowledged)
        | set(_stop_retry_after)
        | set(_stop_tasks)
        | _exhausted_logged
    ):
        _clear_key(key)
    _cleanup_tasks.clear()
