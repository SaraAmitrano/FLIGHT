import logging
from typing import Any

from flight_common.models.zmq.requests.api_orchestrator import (
    ResetClients,
    ResetJobs,
    ResetRequest,
    ResetServer,
)
from flight_common.response_return import ResponseReturn
from flight_orchestrator.data_reset import reset_clients, reset_jobs, reset_server

logger = logging.getLogger(__name__)


async def handle_command_reset(message: dict[str, Any]) -> ResponseReturn:
    return await handle_command_reset_v2(ResetRequest.model_validate(message))


async def handle_command_reset_v2(message: ResetRequest) -> ResponseReturn:
    detail = message.detail
    match detail:
        case ResetClients():
            response = await reset_clients(detail.client_id)
        case ResetServer():
            response = await reset_server()
        case ResetJobs():
            response = await reset_jobs(detail.job_id)
        case _:
            raise TypeError("Unknown 'detail' type in ResetRequest")

    logger.info(f"Command reset {detail} executed successfully.")
    return response
