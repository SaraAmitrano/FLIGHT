from asyncio.subprocess import Process
import os
from typing import Any

from pydantic import BaseModel, Field
from flight_common.monitoring import SpoolBatch


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
        self._status_batch_counter = 0
        self._status_batch_id: str | None = None
        self._status_batch_size = 0
        self.stop_attempt_failed = False
        self.stop_requested_at: float | None = None
        self.monitoring = SpoolBatch(job_id, f"client-{os.environ.get('CLIENT_ID', '0')}")
        self._monitor_batch: list[JobData] = []

    def get_job_datas(self) -> list[dict[str, Any]]:
        return [job_data.model_dump() for job_data in self.job_datas]

    def get_status_batch(self) -> tuple[str | None, list[dict[str, Any]]]:
        """Return a stable batch until the orchestrator acknowledges it."""
        if self._status_batch_id is None:
            records = self.monitoring.acquire()
            self._monitor_batch = [JobData(status=self.last_status,
                                           results={"flight_monitoring": records})] if records else []
        if self._status_batch_id is None and (self.job_datas or self._monitor_batch):
            self._status_batch_counter += 1
            self._status_batch_id = (
                f"{self.job_id}:{self._status_batch_counter}"
            )
            self._status_batch_size = len(self.job_datas)

        return (
            self._status_batch_id,
            [
                job_data.model_dump()
                for job_data in self.job_datas[: self._status_batch_size] + self._monitor_batch
            ],
        )

    def acknowledge_status_batch(self, batch_id: str) -> bool:
        """Discard only the batch confirmed as acquired by the orchestrator."""
        if batch_id != self._status_batch_id:
            return False

        del self.job_datas[: self._status_batch_size]
        self.monitoring.acknowledge()
        self._monitor_batch = []
        self._status_batch_id = None
        self._status_batch_size = 0
        return True
