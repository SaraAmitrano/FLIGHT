from pydantic import BaseModel, Field

from flight_common.job.params import FLParams  # type: ignore[reportMissingTypeStubs]


class FLGetDataDetails(BaseModel):
    job_id: str = Field(..., description="Unique FL job identifier")
    params: FLParams = Field(..., description="FL job parameters")
    rep_addr: str = Field(..., description="ZMQ REP socket address")
