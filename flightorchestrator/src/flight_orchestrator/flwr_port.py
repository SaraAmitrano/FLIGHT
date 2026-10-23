import logging

from flight_common.exceptions import FLIGHTException
from flight_common.job_id import JobID
from flight_common.value_container import ValueContainer

from .address import is_port_in_use
from .service import get_fl_server_service

logger = logging.getLogger(__name__)

MIN_FLWR_PORT: ValueContainer[int] = ValueContainer()
MAX_FLWR_PORT: ValueContainer[int] = ValueContainer()


AVAILABLE_PORTS: list[int] = []


USED_PORTS: dict[JobID, int] = {}


def _reclaim_stale_port_reservations() -> None:
    """Recover leases belonging to finished jobs whose Flower port is closed.

    Caller must hold the orchestrator's GLOBAL_LOCK and data_manager.lock().
    Never reclaim an active job or a port with an accepting Flower listener.
    The cleanup affects only the *in-memory port lease*, not job data/results.
    """
    import socket

    from . import file_manager as data_manager

    terminal = {"completed", "failed", "stopped"}
    server_host = str(get_fl_server_service())

    # Refuse reclamation if service discovery is unavailable; a failed DNS
    # lookup must not be mistaken for a free port.
    try:
        socket.getaddrinfo(server_host, None, type=socket.SOCK_STREAM)
    except OSError:
        logger.exception(
            "Cannot resolve Flower server %s: keeping existing port leases",
            server_host,
        )
        return

    for previous_job_id, port in list(USED_PORTS.items()):
        try:
            job = data_manager.load_job(previous_job_id)
        except Exception:
            logger.exception("Cannot inspect job %s; keeping port %s", previous_job_id, port)
            continue

        if not isinstance(job, dict):
            # Missing or corrupted job data is NOT enough evidence to recycle.
            continue
        if job.get("status") not in terminal:
            continue
        if job.get("server_status", {}).get("status") not in terminal:
            continue

        address = f"{server_host}:{port}"
        if is_port_in_use(address):
            continue
        # Guard against a transient gap between socket close/open.
        if is_port_in_use(address):
            continue

        logger.warning(
            "Recovering stale Flower lease: job=%s port=%s job_status=%s "
            "server_status=%s; no listener present",
            previous_job_id,
            port,
            job["status"],
            job["server_status"]["status"],
        )
        free_flwr_port(previous_job_id)


def get_flwr_port(job_id: JobID) -> int:
    # Normal release still happens in the status handler. This reconciliation
    # catches terminal jobs missed by that path (e.g. dropped client updates).
    _reclaim_stale_port_reservations()

    for port in AVAILABLE_PORTS:
        if port not in USED_PORTS.values():
            flserver_address = f"{get_fl_server_service()}:{port}"
            if is_port_in_use(flserver_address):
                logger.warning(
                    f"Port {port} from address {flserver_address} is already in use, "
                    f"skipping."
                )
                continue
            USED_PORTS[job_id] = port
            logger.info(
                "Reserved Flower port %s for job %s (%s/%s active leases)",
                port, job_id, len(USED_PORTS), len(AVAILABLE_PORTS),
            )
            return port

    logger.error(
        "No Flower port available: %s leases reserved; pool=%s-%s; jobs=%s",
        len(USED_PORTS),
        MIN_FLWR_PORT(),
        MAX_FLWR_PORT(),
        {str(k): v for k, v in USED_PORTS.items()},
    )
    raise FLIGHTException("No available FL server ports")


def is_flwr_port_used(job_id: JobID) -> bool:
    return job_id in USED_PORTS


def free_flwr_port(job_id: JobID) -> None:
    if job_id in USED_PORTS:
        USED_PORTS.pop(job_id)
    else:
        logger.warning(f"Job id {job_id} had no port.")


def init_flwr_ports(min_port: int, max_port: int) -> None:
    MIN_FLWR_PORT.set(min_port)
    MAX_FLWR_PORT.set(max_port)
    AVAILABLE_PORTS.extend(list(range(MIN_FLWR_PORT(), MAX_FLWR_PORT() + 1)))
    logger.info(f"Initialized FLWR ports from {MIN_FLWR_PORT()} to {MAX_FLWR_PORT()}.")
