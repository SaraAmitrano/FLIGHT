from asyncio.subprocess import Process
from dataclasses import dataclass


@dataclass
class ProcessData:
    job_id: str
    process: Process
    status: str


def get_job_id_key(job_id: str) -> str:
    """Generate a unique key for the job_id."""
    return f"flwr_{job_id}"
