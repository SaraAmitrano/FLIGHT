import json
import logging
import uuid
from collections.abc import AsyncIterable, Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, TextIO, cast, overload

from flight_common.config import MAIN_PATH
from flight_common.job_id import JobID, get_job_id, validate_job_id
from flight_orchestrator.config import NUM_CLIENTS

from .atomic_file_io import open_atomic
from .data_types import Client, Job, Server
from .file_lock import lock, lock_file_name

logger = logging.getLogger(__name__)

# Storage paths
STORAGE_PATH = MAIN_PATH.parent / "storage"

JOBS_FOLDER = STORAGE_PATH / "jobs"
SERVER_FILE = STORAGE_PATH / "server.json"
CLIENTS_FOLDER = STORAGE_PATH / "clients"
LOCK_FILE = STORAGE_PATH / "lockfile"
RESULTS_FOLDER = MAIN_PATH.parent / "results"

JOB_RESULTS_KEYS = ("status", "message", "server_results", "clients_results")


def _get_job_file_path(job_id: JobID) -> Path:
    """Get the file path for a job."""
    return JOBS_FOLDER / f"{str(job_id)}.json"


def file_lock(*, verbose: bool = True) -> AbstractContextManager[TextIO]:
    """
    Context manager for file locking.
    This will lock the file until the context is exited.
    """
    return lock(LOCK_FILE, verbose=verbose)


def gen_new_job_id() -> JobID:
    while True:
        job_id = get_job_id(uuid.uuid4())
        if job_id not in get_current_jobs_ids():
            return job_id
        logger.warning(f"Generated duplicate job_id {job_id}. Generating a new one.")


@overload
def load_json_file(
    filepath: Path, default_data: None = None
) -> dict[str, Any] | None: ...


@overload
def load_json_file(filepath: Path, default_data: dict[str, Any]) -> dict[str, Any]: ...


def load_json_file(
    filepath: Path, default_data: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Load a JSON file safely. If the file does not exist, return default data."""
    if not filepath.exists():
        return default_data
    try:
        with filepath.open() as file:
            data = json.load(file)
    except json.JSONDecodeError:
        logger.warning(f"JSON decode error reading {filepath}, returning default data.")
        return default_data
    else:
        if not isinstance(data, dict):
            return default_data
        return cast("dict[str, Any]", data)


def save_json_file(filepath: Path, data: dict[str, Any]) -> None:
    """Save data to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open_atomic(filepath, mode="w") as file:
        json.dump(data, file, indent=4)


def get_job_results(job: Job) -> dict[str, Any]:
    """Return the same job data exposed by the results endpoint."""
    return {key: job[key] for key in JOB_RESULTS_KEYS if key in job}


def export_completed_job_results(job_id: JobID, job: Job) -> None:
    """Atomically export completed job results without affecting the job itself."""
    if job["status"] != "completed":
        return

    filepath = RESULTS_FOLDER / str(job_id) / "results.json"
    try:
        save_json_file(filepath, get_job_results(job))
    except Exception:
        logger.exception(f"Failed to export results for job {job_id} to {filepath}")


def load_server_json_file() -> Server:
    default_server: Server = {"online": False, "status": "idle", "job_ids": []}

    server = load_json_file(SERVER_FILE, cast("dict[str, Any]", default_server))

    if not server or not SERVER_FILE.exists():
        server = default_server
        save_json_file(SERVER_FILE, cast("dict[str, Any]", server))

    return cast("Server", server)


def save_server_json_file(server: Server) -> None:
    server_file = SERVER_FILE
    save_json_file(server_file, cast("dict[str, Any]", server))


def load_client_json_file(client_id: int) -> Client:
    client_file = CLIENTS_FOLDER / f"{client_id}.json"
    default_client: Client = {"online": False, "status": "idle"}
    return cast(
        "Client", load_json_file(client_file, cast("dict[str, Any]", default_client))
    )


def save_client_json_file(client_id: int, client: Client) -> None:
    client_file = CLIENTS_FOLDER / f"{client_id}.json"
    save_json_file(client_file, cast("dict[str, Any]", client))


def load_job_json_file(job_id: JobID) -> Job | None:
    data = load_json_file(_get_job_file_path(job_id))

    if data is None:
        return data

    return cast("Job", data)


def load_job_json_file_response(job_id: JobID) -> Job | tuple[int, str]:
    if not is_job(job_id):
        logger.info(f"Job id {job_id} not found. Returning 404.")
        return 404, "job_id not found"

    job = load_job_json_file(job_id)
    if job is None:
        logger.warning(f"Job id {job_id} data loss. Returning 500.")
        return 500, "job_id data loss"

    return job


def save_job_json_file(job_id: JobID, job: Job) -> None:
    save_json_file(_get_job_file_path(job_id), cast("dict[str, Any]", job))


def delete_job_json_file(job_id: JobID) -> None:
    """Remove an job JSON file."""
    if not is_job(job_id):
        raise FileNotFoundError(f"Job {job_id} does not exist.")

    filepath = _get_job_file_path(job_id)
    filepath.unlink()


def get_current_jobs_ids() -> list[JobID]:
    """Get the current jobs ids. Assume the lock is already acquired."""
    out: list[JobID] = []
    for f in JOBS_FOLDER.iterdir():
        if not f.is_file():
            continue

        if f.suffix != ".json":
            continue

        job_id_ = validate_job_id(f.stem)

        if job_id_ is not None:
            out.append(job_id_)

    return out


def get_current_jobs() -> Iterable[tuple[JobID, Job]]:
    for job_id in get_current_jobs_ids():
        job = load_job_json_file(job_id)
        # The job is supposed to be a dict,
        # but it can be None if the file is corrupted
        if job:
            yield job_id, job


async def get_current_jobs_ids_async() -> AsyncIterable[JobID]:
    """Get the current jobs ids. Assume the lock is already acquired."""
    for f in JOBS_FOLDER.iterdir():
        if not f.is_file():
            continue

        if f.suffix != ".json":
            continue

        job_id_ = validate_job_id(f.stem)
        if job_id_ is not None:
            yield job_id_


async def get_current_jobs_async() -> AsyncIterable[tuple[JobID, Job]]:
    async for job_id in get_current_jobs_ids_async():
        job = load_job_json_file(job_id)
        # The job is supposed to be a dict,
        # but it can be None if the file is corrupted
        if job:
            yield job_id, job


def is_job(job_id: JobID) -> bool:
    return _get_job_file_path(job_id).exists()


def init() -> None:
    """Initialize the files and folders."""
    if not JOBS_FOLDER.exists():
        JOBS_FOLDER.mkdir(parents=True, exist_ok=True)
    if not CLIENTS_FOLDER.exists():
        CLIENTS_FOLDER.mkdir(parents=True, exist_ok=True)
    if lock_file_name(LOCK_FILE).exists():
        logger.warning(
            f"Lock file {lock_file_name(LOCK_FILE)} exists. This may indicate a crash."
        )
        lock_file_name(LOCK_FILE).unlink()
    if not SERVER_FILE.exists():
        save_server_json_file(load_server_json_file())
    for filepath in CLIENTS_FOLDER.iterdir():
        if filepath.is_file() and filepath.suffix == ".json":
            client_id_s = filepath.stem
            if not client_id_s.isdigit():
                logger.warning(f"Invalid client id {client_id_s}. Deleting file.")
                filepath.unlink()
                continue

            client_id = int(client_id_s)
            if client_id >= NUM_CLIENTS:
                logger.info(
                    f"Client id {client_id} exceeds NUM_CLIENTS ({NUM_CLIENTS}). "
                    f"Deleting file."
                )
                filepath.unlink()
                continue

            save_client_json_file(client_id, load_client_json_file(client_id))
    for client_id in range(NUM_CLIENTS):
        client_file = CLIENTS_FOLDER / f"{client_id}.json"
        if not client_file.exists():
            save_client_json_file(client_id, load_client_json_file(client_id))

    for job_id in get_current_jobs_ids():
        job = load_job_json_file(job_id)
        if job is None:
            logger.warning(f"Job {job_id} is invalid. Removing it")
            (JOBS_FOLDER / f"{job_id}.json").unlink()
