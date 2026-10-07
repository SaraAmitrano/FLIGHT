import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException

from flight_common.response_return import ResponseReturn

logger = logging.getLogger(__name__)


class OrchestratorClient:
    def __init__(
        self, zmq_sender: Callable[[dict[str, Any], float], Awaitable[ResponseReturn]]
    ) -> None:
        self.zmq_sender = zmq_sender

    async def send(
        self,
        command_payload: dict[str, Any],
        success_status_code: int = 200,
        success_content: dict[str, Any] | None = None,
        timeout: float = 100.0,
    ) -> dict[str, Any]:
        response = await self.zmq_sender(command_payload, timeout)

        if response.status == "error":
            logger.error(f"Error status in response: {response.detail}")
            raise HTTPException(status_code=500, detail=response.detail)

        status_code = response.detail.pop("status_code", success_status_code)
        if status_code != success_status_code:
            logger.error(
                f"Erroneous status code"
                f"[expected: {success_status_code}, got: {status_code}] "
                f"in response: {response.detail}"
            )
            raise HTTPException(status_code=status_code, detail=response.detail)

        return success_content if success_content is not None else response.detail
