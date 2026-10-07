# import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from flight_common.utils import get_full_class_name

from .api.v1 import clients, health, jobs, logs, server
from .config import app_data
from .exceptions import APIException

# from .monitoring import monitor_subsystems, subsystems_online
from .monitoring import subsystems_online
from .startup import wait_for_subsystems

logger = logging.getLogger(__name__)

INTERNAL_ERROR_CODE = 500

# background_tasks: set[asyncio.Task[None]] = set()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """
    Context manager to handle the lifespan of the FastAPI application.
    """
    try:
        out = await wait_for_subsystems()
        if not out:
            logger.error("Failed to connect to all subsystems. Shutting down.")
            raise RuntimeError("Failed to connect to all subsystems.")
        subsystems_online.set(out)
        # monitor_task = asyncio.create_task(monitor_subsystems())
        # background_tasks.add(monitor_task)
        # monitor_task.add_done_callback(background_tasks.discard)
        logger.info("Ready to accept requests.")
        yield
    finally:
        app_data.terminate()


app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)

# TODO: add prefix /api/v1
app.include_router(health.router, prefix="")
app.include_router(jobs.router, prefix="")
app.include_router(clients.router, prefix="")
app.include_router(server.router, prefix="")
app.include_router(logs.router, prefix="")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code == INTERNAL_ERROR_CODE:
        logger.exception(
            "Unhandled HTTP exception",
            exc_info=True,  # noqa: LOG014
            extra={
                "url": str(request.url),
                "status_code": exc.status_code,
                "detail": exc.detail,
            },
        )
    return JSONResponse(content=dict(detail=exc.detail), status_code=exc.status_code)


@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled API exception",
        exc_info=True,  # noqa: LOG014
        extra={"url": str(request.url), "exception": str(exc)},
    )
    return JSONResponse(
        content=dict(detail=dict(exception=get_full_class_name(exc), info=str(exc))),
        status_code=INTERNAL_ERROR_CODE,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled error",
        exc_info=True,  # noqa: LOG014
        extra={"url": str(request.url), "exception": str(exc)},
    )
    return JSONResponse(
        content=dict(detail=dict(exception=get_full_class_name(exc), info=str(exc))),
        status_code=INTERNAL_ERROR_CODE,
    )
