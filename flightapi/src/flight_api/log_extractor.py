from datetime import date
from pathlib import Path

from anyio import Path as ASyncPath


async def _is_log_file_valid(file: ASyncPath, days: int | None = None) -> bool:
    if not await file.exists():
        return False

    if not await file.is_file():
        return False

    if days is None:
        return True

    today = date.today()  # noqa: DTZ011
    first_day = today.fromordinal(today.toordinal() - days)
    return file.name.split(".log")[-1] == "" or file.name.split(".log.")[-1] >= str(
        first_day
    )


async def get_logfiles(logs_dir: Path, days: int | None = None) -> list[Path]:
    if not logs_dir.exists() or not logs_dir.is_dir():
        return []
    logs_dir_async = ASyncPath(logs_dir)
    return [
        Path(f)
        async for f in logs_dir_async.rglob("*.log*")
        if await _is_log_file_valid(f, days)
    ]
