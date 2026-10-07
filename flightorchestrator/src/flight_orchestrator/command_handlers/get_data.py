import logging
from typing import Any

from flight_common.models.zmq.requests.api_orchestrator import (
    GetClients,
    GetDataRequest,
    GetJobIDs,
    GetServer,
)
from flight_common.response_return import ResponseReturn, create_success_response
from flight_orchestrator import file_manager as data_manager
from flight_orchestrator.asyncio_lock import GLOBAL_LOCK
from flight_orchestrator.config import NUM_CLIENTS

logger = logging.getLogger(__name__)


async def get_server_data() -> ResponseReturn:
    async with GLOBAL_LOCK:
        with data_manager.lock():
            server = data_manager.load_server()
    logger.debug(f"Server data: {server}")
    return create_success_response(200, **server)


async def get_clients_data(sel_client_id: int | None = None) -> ResponseReturn:
    if sel_client_id is None:
        async with GLOBAL_LOCK:
            with data_manager.lock():
                clients = {
                    str(client_id): data_manager.load_client(client_id)
                    for client_id in range(NUM_CLIENTS)
                }
                logger.debug(f"Clients data: {clients}")
                return create_success_response(200, **clients)

    if sel_client_id < 0 or sel_client_id >= NUM_CLIENTS:
        logger.info(f"Invalid client ID: {sel_client_id}.")
        return create_success_response(
            404, message=f"Invalid client ID: {sel_client_id}"
        )

    async with GLOBAL_LOCK:
        with data_manager.lock():
            client = data_manager.load_client(sel_client_id)
    logger.debug(f"Client {sel_client_id} data: {client}")
    return create_success_response(200, **client)


async def get_job_ids_data() -> ResponseReturn:
    async with GLOBAL_LOCK:
        with data_manager.lock():
            job_ids = data_manager.get_current_jobs_ids()
    logger.debug(f"Job ids: {job_ids}")
    return create_success_response(200, job_ids=[str(job_id) for job_id in job_ids])


async def handle_command_get_data_v2(message: GetDataRequest) -> ResponseReturn:
    detail = message.detail

    match detail:
        case GetServer():
            return await get_server_data()

        case GetClients():
            return await get_clients_data(sel_client_id=detail.client_id)

        case GetJobIDs():
            return await get_job_ids_data()

        case _:
            raise TypeError("Unknown 'detail' type in GetDataRequest")


async def handle_command_get_data(message: dict[str, Any]) -> ResponseReturn:
    return await handle_command_get_data_v2(GetDataRequest.model_validate(message))
