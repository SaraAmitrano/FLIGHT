import logging
from typing import Any

from flight_common.job_id import validate_job_id
from flight_common.response_return import ResponseReturn, create_success_response
from flight_orchestrator import file_manager as data_manager
from flight_orchestrator.asyncio_lock import GLOBAL_LOCK

logger = logging.getLogger(__name__)


async def handle_command_get_status(message: dict[str, Any]) -> ResponseReturn:
    job_id_ = message["job_id"]

    job_id = validate_job_id(job_id_)
    if job_id is None:
        logger.info(f"Invalid job id: {job_id_}. Returning 404.")
        return create_success_response(400, message="job_id invalid")

    async with GLOBAL_LOCK:
        with data_manager.lock():
            response = data_manager.load_job_response(job_id)

            if isinstance(response, tuple):
                return create_success_response(response[0], message=response[1])

            job = response
            status = job["status"]

    return create_success_response(200, status=status)
