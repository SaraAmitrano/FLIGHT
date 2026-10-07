from asyncio.subprocess import Process
from typing import Any

from pydantic import BaseModel, Field


class JobData(BaseModel):
    status: str = Field(..., description="Current status of the job")
    message: str | None = Field(
        default=None, description="Optional message regarding the job status"
    )
    results: dict[str, Any] | None = Field(
        default=None, description="Results of the job, if available"
    )
    detail: dict[str, Any] | None = Field(
        default=None, description="Detail of the job, if available"
    )


class ProcessData:
    def __init__(
        self,
        job_id: str,
        process: Process | None,
        status: str,
        message: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.process = process
        self.last_status = status
        self.job_datas = [JobData(status=status, message=message)]

    def get_job_datas(self) -> list[dict[str, Any]]:
        return [job_data.model_dump() for job_data in self.job_datas]
