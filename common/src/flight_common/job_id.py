from typing import Any, NewType
from uuid import UUID

JobID = NewType("JobID", UUID)


def get_job_id(job_id: UUID) -> JobID:
    """
    Convert a job ID from string or UUID to JobID type.
    """
    if isinstance(job_id, str):
        return JobID(job_id)
    return JobID(job_id)


def validate_job_id(id_obj: Any) -> JobID | None:
    if not isinstance(id_obj, str):
        return None

    try:
        return get_job_id(UUID(id_obj))
    except ValueError:
        return None


def sanitize_job_id(id_obj: Any, length_limit: int = 20) -> str:
    job_id = validate_job_id(id_obj)
    if job_id is None:
        id_s = str(id_obj)
        return id_s[:length_limit] + ("..." if len(id_s) > length_limit else "")
    return str(job_id)
