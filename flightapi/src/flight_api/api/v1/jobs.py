import logging
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import SkipValidation

from flight_api.config import app_data
from flight_api.core.flparams_schema_validation import (
    extract_error_info,
    validate_flparams,
)
from flight_api.orchestrator_client import OrchestratorClient
from flight_api.send_commands import send_command
from flight_api.zmq_orchestrator_client import get_orchestrator_client

# from flight_api.models.v1.jobs import (
#     ClientResultResponse,
#     GlobalEvaluationResponse,
#     JobResultResponse,
#     TrainDataResponse,
# )
from flight_common.job.params import FLParams
from flight_common.models.zmq.requests.api_orchestrator import (
    DeleteJobRequest,
    GetDataRequest,
    GetJobIDs,
    GetJobRequest,
    ResetJobs,
    ResetRequest,
    StartJobRequest,
    StopJobRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/ids")
async def get_job_ids(
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        GetDataRequest(detail=GetJobIDs()), orchestrator_client=orchestrator_client
    )


@router.post("/")
async def start_job(
    params: SkipValidation[FLParams],
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
    client_ids: list[int] | None = None,
) -> JSONResponse:
    logger.debug("Received params: %s", params)

    # Validate incoming request with JSON schema.
    errors = validate_flparams(
        cast("dict[str, Any]", params), app_data.flparams_schema_validator
    )

    if errors:
        error_info = extract_error_info(errors)

        errors_details = [v.model_dump(mode="json") for v in error_info]

        logger.info(
            "Validation errors for params: %s: %s", str(params), str(errors_details)
        )
        raise HTTPException(
            status_code=400,
            detail=dict(
                message="Invalid params",
                errors=errors_details,
            ),
        )

    params_ = FLParams.model_validate(params)

    if client_ids is not None:
        if len(client_ids) != params_.num_clients:
            raise HTTPException(
                status_code=400,
                detail=dict(
                    message="Number of client IDs does not match num_clients in params",
                ),
            )

        if any(
            client_id < 0 or client_id >= app_data.NUM_CLIENTS
            for client_id in client_ids
        ):
            raise HTTPException(
                status_code=400,
                detail=dict(
                    message="Client IDs must be in the range of available clients",
                ),
            )

    return await send_command(
        StartJobRequest(
            params=params_,  # type: ignore
            client_ids=client_ids,
        ),
        success_status_code=201,
        orchestrator_client=orchestrator_client,
    )


@router.get("/{job_id}/")
async def get_job(
    job_id: str,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        GetJobRequest(job_id=job_id, part="all"),
        orchestrator_client=orchestrator_client,
    )


@router.get("/{job_id}/status")
async def get_job_status(
    job_id: str,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        GetJobRequest(job_id=job_id, part="status"),
        orchestrator_client=orchestrator_client,
    )


@router.get("/{job_id}/results")
async def get_job_results(
    job_id: str,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        GetJobRequest(job_id=job_id, part="results"),
        orchestrator_client=orchestrator_client,
    )


@router.get("/{job_id}/details")
async def get_job_details(
    job_id: str,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        GetJobRequest(job_id=job_id, part="details"),
        orchestrator_client=orchestrator_client,
    )


@router.post("/{job_id}/stop")
async def stop_job(
    job_id: str,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        StopJobRequest(job_id=job_id),
        orchestrator_client=orchestrator_client,
    )


@router.post("/reset")
async def reset_all_jobs(
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        ResetRequest(detail=ResetJobs(job_id=None)),
        orchestrator_client=orchestrator_client,
    )


@router.post("/{job_id}/reset")
async def reset_job(
    job_id: str,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        ResetRequest(detail=ResetJobs(job_id=job_id)),
        orchestrator_client=orchestrator_client,
    )


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    orchestrator_client: Annotated[
        OrchestratorClient, Depends(get_orchestrator_client)
    ],
) -> JSONResponse:
    return await send_command(
        DeleteJobRequest(job_id=job_id),
        orchestrator_client=orchestrator_client,
    )
