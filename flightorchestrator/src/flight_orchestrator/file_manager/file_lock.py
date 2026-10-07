import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

logger = logging.getLogger(__name__)


def lock_file_name(filepath: Path) -> Path:
    """
    Generate a lock file name based on the original file name.
    The lock file will have the same name as the original file
    with an added ".lock" suffix.
    """
    return filepath.with_suffix(filepath.suffix + ".lock")


@contextmanager
def lock(
    filepath: Path,
    timeout: float = 10,
    poll_interval: float = 0.1,
    *,
    verbose: bool = True,
) -> Iterator[TextIO]:
    """
    A simple exclusive lock using a .lock file.
    Blocks until the lock can be acquired or the timeout is reached.
    """
    lock_path = lock_file_name(filepath)
    start_time = time.time()

    # Wait until the lock file is gone or timeout
    while lock_path.exists():
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for lock on {filepath}")
        time.sleep(poll_interval)

    # Acquire lock by creating the lock file
    try:
        if verbose:
            logger.debug("Start locking...")
        with lock_path.open("w") as lockfile:
            lockfile.write(str(os.getpid()))
        yield lockfile
    finally:
        lock_path.unlink(missing_ok=True)
        if verbose:
            logger.debug("Locking done.")
