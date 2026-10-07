import logging
from typing import Any

from flight_common.job_id import validate_job_id
from flight_common.models.zmq.requests.api_orchestrator import GetJobRequest
from flight_common.response_return import ResponseReturn, create_success_response
from flight_orchestrator import file_manager as data_manager
from flight_orchestrator.asyncio_lock import GLOBAL_LOCK

logger = logging.getLogger(__name__)


PART_KEYS_MAP = {
    "all": "all",
    "status": ["status"],
    "results": ["status", "message", "server_results", "clients_results"],
    "details": ["status", "message", "server_details", "clients_details"],
}


async def handle_command_get_job(message: dict[str, Any]) -> ResponseReturn:
    return await handle_command_get_job_v2(GetJobRequest.model_validate(message))


async def handle_command_get_job_v2(message: GetJobRequest) -> ResponseReturn:
    job_id_ = message.job_id

    part = message.part

    job_id = validate_job_id(job_id_)
    if job_id is None:
        logger.info(f"Invalid job id: {job_id_}. Returning 404.")
        return create_success_response(404, message="job_id invalid")

    async with GLOBAL_LOCK:
        with data_manager.lock():
            response = data_manager.load_job_response(job_id)
            if isinstance(response, tuple):
                # If the response indicates an error, return it
                return create_success_response(response[0], message=response[1])
            job = response

            part_keys = PART_KEYS_MAP[part]
            job_keys = list(job.keys()) if part_keys == "all" else part_keys

            out: dict[str, Any] = {k: job[k] for k in job_keys if k in job}

    # logger.info(f"Job data found: {out}")
    return create_success_response(200, **out)
