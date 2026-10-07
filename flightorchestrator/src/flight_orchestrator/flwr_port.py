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


def get_flwr_port(job_id: JobID) -> int:
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
            return port
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
