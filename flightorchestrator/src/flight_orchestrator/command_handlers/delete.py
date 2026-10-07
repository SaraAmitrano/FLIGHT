import logging
from typing import Any

from flight_common.job_id import validate_job_id
from flight_common.models.zmq.requests.api_orchestrator import DeleteJobRequest
from flight_common.response_return import ResponseReturn, create_success_response
from flight_common.zmq_asyncio_client import AsyncZmqClient
from flight_orchestrator import file_manager as data_manager
from flight_orchestrator.asyncio_lock import GLOBAL_LOCK
from flight_orchestrator.service import FLService

logger = logging.getLogger(__name__)


async def handle_command_pre_delete(message: dict[str, Any]) -> ResponseReturn:
    return await handle_command_pre_delete_v2(DeleteJobRequest.model_validate(message))


async def handle_command_pre_delete_v2(message: DeleteJobRequest) -> ResponseReturn:
    from typing import cast  # noqa: PLC0415

    job_id_ = message.job_id

    job_id = validate_job_id(job_id_)
    if job_id is None:
        logger.debug(f"Invalid job_id {job_id_}. Returning 404.")
        return create_success_response(404, message="Invalid job_id")

    async with GLOBAL_LOCK:
        with data_manager.lock():
            response = data_manager.load_job_response(job_id)
            if isinstance(response, tuple) and response[0] != 500:
                # If the response indicates an error, return it
                # (except when it's 500, meaning data loss,
                #  but in that case we can still delete the job)
                return create_success_response(response[0], message=response[1])

            # if the job is still running, do not delete it
            job = cast("data_manager.Job", response)
            if job["status"] in ["running", "starting", "stopping"]:
                logger.debug(f"Job {job_id} is still running. Returning 400.")
                return create_success_response(
                    400, message="Cannot delete a running job"
                )
    return create_success_response(200, status="deleted")


async def handle_command_post_delete(
    message: dict[str, Any], _req_clients: dict[FLService, AsyncZmqClient]
) -> ResponseReturn:
    return await handle_command_post_delete_v2(
        DeleteJobRequest.model_validate(message), _req_clients
    )


async def handle_command_post_delete_v2(
    message: DeleteJobRequest, _req_clients: dict[FLService, AsyncZmqClient]
) -> ResponseReturn:
    job_id_ = message.job_id

    job_id = validate_job_id(job_id_)
    if job_id is None:
        logger.debug(f"Invalid job_id {job_id_}. Returning 404.")
        return create_success_response(404, message="Invalid job_id")

    async with GLOBAL_LOCK:
        with data_manager.lock():
            data_manager.delete_job(job_id)

    logger.info(f"Job {job_id} deleted successfully.")
    return create_success_response(200, message=f"Job {job_id} deleted successfully")
