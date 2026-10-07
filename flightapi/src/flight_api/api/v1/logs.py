import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from anyio import Path as AsyncPath
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from zipstream import AioZipStream  # type: ignore

from flight_api.config import app_data
from flight_api.log_extractor import get_logfiles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logs", tags=["Logs"])


async def get_log_generator(log_file: Path) -> AsyncGenerator[bytes]:
    log_file_async = AsyncPath(log_file)
    yield await log_file_async.read_bytes()


@router.get("/")
async def get_logs(days: int | None = None) -> StreamingResponse:
    log_files = await get_logfiles(app_data.LOGS_DIR, days=days)

    files = [
        {"stream": get_log_generator(f), "name": str(f.relative_to(app_data.LOGS_DIR))}
        for f in log_files
    ]

    zip_stream = AioZipStream(files, chunksize=32768)
    return StreamingResponse(
        zip_stream.stream(),  # type: ignore
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="logs.zip"'},
    )
