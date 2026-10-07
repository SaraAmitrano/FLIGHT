import logging

from fastapi.responses import JSONResponse

from flight_common.models.zmq.requests.api_orchestrator import BaseAPIZMQRequest

from .orchestrator_client import OrchestratorClient

logger = logging.getLogger(__name__)


async def send_command(
    command_data: BaseAPIZMQRequest,
    orchestrator_client: OrchestratorClient,
    success_status_code: int = 200,
) -> JSONResponse:
    response = await orchestrator_client.send(
        command_data.model_dump(mode="json"),
        success_status_code=success_status_code,
        timeout=100.0,
    )

    logger.debug(f"Response: {response}, status code: {success_status_code}")
    return JSONResponse(content=response, status_code=success_status_code)
