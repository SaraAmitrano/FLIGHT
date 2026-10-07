import os
import tempfile as tmp
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, cast


@contextmanager
def tempfile(
    suffix: str = "", temp_dir: str | None = None
) -> Generator[Path, None, None]:
    """
    Context for temporary file.

    Will find a free temporary filename upon entering
    and will try to delete the file on leaving, even in case of an exception.

    Parameters
    ----------
    suffix : string
        optional file suffix
    temp_dir : string
        optional directory to save temporary file in

    """
    with tmp.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=temp_dir, delete_on_close=False
    ) as tf:
        pass
    tfnamepath = Path(tf.name)
    try:
        yield tfnamepath
    finally:
        tfnamepath.unlink(missing_ok=True)


@contextmanager
def open_atomic(
    filepath: Path, *args: Any, **kwargs: Any
) -> Generator[IO[Any], None, None]:
    """
    Open temporary file object that atomically moves to destination upon
    exiting.

    Allows reading and writing to and from the same filename.

    The file will not be moved to destination in case of an exception.

    Parameters
    ----------
    filepath : string
        the file path to be opened
    fsync : bool
        whether to force write the file to disk
    *args : mixed
        Any valid arguments for :code:`open`
    **kwargs : mixed
        Any valid keyword arguments for :code:`open`

    """
    fsync = kwargs.pop("fsync", False)

    with tempfile(temp_dir=str(filepath.absolute().parent)) as tmppath:
        try:
            with tmppath.open(*args, **kwargs) as file:  # type: ignore
                file = cast("IO[Any]", file)
                yield file
                if fsync:
                    file.flush()
                    os.fsync(file.fileno())
        except Exception:
            tmppath.unlink()
            raise
        else:
            tmppath.rename(filepath)
            filepath.chmod(0o644)
