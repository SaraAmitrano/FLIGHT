import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from flight_api.orchestrator_client import OrchestratorClient
from flight_api.send_commands import send_command
from flight_api.zmq_orchestrator_client import get_orchestrator_client
from flight_common.models.zmq.requests.api_orchestrator import (
    GetDataRequest,
    GetServer,
    ResetRequest,
    ResetServer,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/server", tags=["Server"])


@router.get("/")
async def get_server(
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    """
    Return the status of the server.
    """
    return await send_command(
        GetDataRequest(detail=GetServer()), orchestrator_client=orchestrator_client
    )


@router.post("/reset")
async def reset_server(
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    """
    Reset the server.
    """
    return await send_command(
        ResetRequest(detail=ResetServer()), orchestrator_client=orchestrator_client
    )
