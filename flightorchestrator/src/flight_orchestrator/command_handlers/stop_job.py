import asyncio
import logging
from typing import Any, cast

from flight_common.job_id import validate_job_id
from flight_common.models.zmq.requests.api_orchestrator import StopJobRequest
from flight_common.response_return import (
    ResponseReturn,
    create_error_response,
    create_success_response,
)
from flight_common.zmq_asyncio_client import AsyncZmqClient
from flight_orchestrator import file_manager as data_manager
from flight_orchestrator.asyncio_lock import GLOBAL_LOCK
from flight_orchestrator.file_manager.data_types import CommonStopPayload, JobID
from flight_orchestrator.service import FLService, get_fl_client_service
from flight_orchestrator.zmq_orchestrator_utils import send_zeromq_request

logger = logging.getLogger(__name__)


async def handle_command_pre_stop_job(message: dict[str, Any]) -> ResponseReturn:
    return await handle_command_pre_stop_job_v2(StopJobRequest.model_validate(message))


async def handle_command_pre_stop_job_v2(message: StopJobRequest) -> ResponseReturn:
    """Handles the stop command from the API."""
    job_id_ = message.job_id

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
            if job["status"] != "running":
                logger.info(f"Job {job_id_} not running. Returning 400.")
                return create_success_response(
                    400, message=f"Job {job_id_} not running"
                )

    return create_success_response(200, status="stopping")


async def handle_command_post_stop_job(
    message: dict[str, Any], req_clients: dict[FLService, AsyncZmqClient]
) -> ResponseReturn:
    return await handle_command_post_stop_job_v2(
        StopJobRequest.model_validate(message), req_clients
    )


async def handle_command_post_stop_job_v2(
    message: StopJobRequest, req_clients: dict[FLService, AsyncZmqClient]
) -> ResponseReturn:
    job_id_ = message.job_id

    job_id = validate_job_id(job_id_)
    if job_id is None:
        # it was already checked in pre-stop, so this should not happen
        logger.warning(f"Invalid job_id {job_id_}.")
        return create_error_response(500, message="Invalid job_id")

    payload = CommonStopPayload(command="stop", job_id=JobID(str(job_id)))

    async with GLOBAL_LOCK:
        with data_manager.lock():
            response = data_manager.load_job_response(job_id)
            if isinstance(response, tuple):
                # If the response indicates an error, return it
                logger.warning(f"Error loading job {job_id_}: {response[1]}")
                return create_success_response(response[0], message=response[1])

            job = response

    clients_status = job["clients_status"]
    running_clients = [
        int(client_id)
        for client_id, status in clients_status.items()
        if status not in ["stopping", "stopped", "failed", "completed"]
    ]

    clients_responses: dict[int, ResponseReturn] = {}
    if running_clients:

        async def stop_one(client_id: int) -> tuple[int, ResponseReturn]:
            logger.info(f"Sending stop command to client {client_id}")
            client_service = get_fl_client_service(client_id)
            client = req_clients[client_service]
            resp = await send_zeromq_request(
                client_service, client, cast("dict[str, Any]", payload)
            )
            return client_id, resp

        results = await asyncio.gather(
            *[stop_one(client_id) for client_id in running_clients],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                # Should not happen because send_zeromq_request handles errors,
                # but guard anyway:
                logger.exception("Unexpected error stopping a client", exc_info=result)
                continue
            if isinstance(result, tuple):
                client_id, response = result
                clients_responses[client_id] = response

    # logger.info("Sending stop command to server")
    # server_service = get_fl_server_service()

    # server_response = await send_zeromq_request(
    #     server_service, req_clients[server_service], cast(dict[str, Any], payload)
    # )

    # if server_response.status == "error":
    #     logger.error(
    #         f"Error while sending stop command to server: {server_response.detail}"
    #     )
    #     return create_error_response(
    #         500,
    #         message="Error while sending stop command to server",
    #         server_detail=server_response.detail,
    #     )

    if any(
        client_response.status == "error"
        for client_response in clients_responses.values()
    ):
        errored_clients = {
            client_id: client_response.detail
            for client_id, client_response in clients_responses.items()
            if client_response.status == "error"
        }
        logger.error(f"Error while sending stop command to clients: {errored_clients}")
        return create_error_response(
            500,
            message="Error while sending stop command to clients",
            client_detail={k: v.to_dict() for k, v in clients_responses.items()},
        )

    return create_success_response(200, status="stopping")
