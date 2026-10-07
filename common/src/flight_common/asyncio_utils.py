import asyncio
import contextlib


async def is_running(proc: asyncio.subprocess.Process) -> bool:
    if proc.returncode is not None:
        return False

    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), 1e-3)
    return proc.returncode is None
