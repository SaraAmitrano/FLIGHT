import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from flight_api.orchestrator_client import OrchestratorClient
from flight_api.send_commands import send_command
from flight_api.zmq_orchestrator_client import get_orchestrator_client
from flight_common.models.zmq.requests.api_orchestrator import (
    GetClients,
    GetDataRequest,
    ResetClients,
    ResetRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clients", tags=["Clients"])


@router.get("/")
async def get_clients(
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    """
    Return a list of clients with their status.
    """
    return await send_command(
        GetDataRequest(detail=GetClients(client_id=None)),
        orchestrator_client=orchestrator_client,
    )


@router.get("/{client_id}/")
async def get_client(
    client_id: int,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    """
    Return a list of clients with their status.
    """
    return await send_command(
        GetDataRequest(detail=GetClients(client_id=client_id)),
        orchestrator_client=orchestrator_client,
    )


@router.post("/reset")
async def reset_clients(
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    """
    Reset all the clients.
    """
    return await send_command(
        ResetRequest(detail=ResetClients(client_id=None)),
        orchestrator_client=orchestrator_client,
    )


@router.post("/{client_id}/reset")
async def reset_client(
    client_id: int,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    """
    Reset the client with the specified client id.
    """
    return await send_command(
        ResetRequest(detail=ResetClients(client_id=client_id)),
        orchestrator_client=orchestrator_client,
    )
