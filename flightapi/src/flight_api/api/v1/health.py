from fastapi import APIRouter

from flight_api.models.v1.health import HealthStatus
from flight_api.monitoring import subsystems_online

router = APIRouter(prefix="", tags=["Health"])


@router.get("/")
async def health_check() -> HealthStatus:
    if subsystems_online() is False:
        return HealthStatus(status="degraded")
    return HealthStatus(status="ok")
