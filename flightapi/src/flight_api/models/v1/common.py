from pydantic import BaseModel


class OrchestratorErrorDetail(BaseModel):
    status_code: int
    message: str


class OrchestratorErrorResponse(BaseModel):
    status: str
    detail: OrchestratorErrorDetail
