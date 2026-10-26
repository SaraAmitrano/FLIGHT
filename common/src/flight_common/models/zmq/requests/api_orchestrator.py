from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from flight_common.gen_request_id import generate_request_id
from flight_common.job.params import FLParams

RequestID = Annotated[
    str,
    StringConstraints(
        pattern=r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{6}-[a-f0-9]{8}$"
    ),
]


class BaseAPIZMQRequest(BaseModel):
    request_id: RequestID = Field(default_factory=generate_request_id)
    model_config = ConfigDict(extra="forbid", frozen=True)


# get_data
class GetJobIDs(BaseModel):
    which: Literal["job_ids"] = "job_ids"


class GetServer(BaseModel):
    which: Literal["server"] = "server"


class GetClients(BaseModel):
    which: Literal["clients"] = "clients"
    client_id: int | None


class GetDataRequest(BaseAPIZMQRequest):
    command: Literal["get_data"] = "get_data"
    detail: GetJobIDs | GetServer | GetClients = Field(discriminator="which")


# get_job
class GetJobRequest(BaseAPIZMQRequest):
    command: Literal["get_job"] = "get_job"
    job_id: str = Field(...)
    part: Literal["all", "status", "results", "details", "monitoring", "monitoring_samples"] = Field(...)
    source: str | None = None
    kind: str | None = None
    offset: int = 0
    limit: int = 100


# start_job
class StartJobRequest(BaseAPIZMQRequest):
    command: Literal["start_job"] = "start_job"
    params: FLParams = Field(...)
    client_ids: list[int] | None = Field(default=None)


# stop_job
class StopJobRequest(BaseAPIZMQRequest):
    command: Literal["stop_job"] = "stop_job"
    job_id: str = Field(...)


# reset
class ResetClients(BaseModel):
    which: Literal["clients"] = "clients"
    client_id: int | None = None


class ResetJobs(BaseModel):
    which: Literal["jobs"] = "jobs"
    job_id: str | None = Field(...)


class ResetServer(BaseModel):
    which: Literal["server"] = "server"


class ResetRequest(BaseAPIZMQRequest):
    command: Literal["reset"] = "reset"
    detail: ResetClients | ResetJobs | ResetServer = Field(discriminator="which")


# delete
class DeleteJobRequest(BaseAPIZMQRequest):
    command: Literal["delete"] = "delete"
    job_id: str = Field(...)
