import asyncio
import logging

from flight_common.response_return import ResponseReturn
from flight_common.zmq_asyncio_client import AsyncZmqClient

logger = logging.getLogger(__name__)

ZMQ_RESPONSE_LEN = 2


async def send_command_to_subprocess(
    client: AsyncZmqClient, command: str, job_id: str, timeout: float = 60.0
) -> ResponseReturn:
    response = ResponseReturn()
    try:
        async with asyncio.Lock():
            response = await client.request(dict(command=command), timeout=timeout)

        if "status" not in response:
            raise ValueError("Missing status from response")
        if "detail" not in response:
            raise ValueError("Missing detail from response")
        if len(response) != ZMQ_RESPONSE_LEN:
            raise ValueError("Invalid response format")

        return ResponseReturn(**response)
    except Exception as e:
        logger.exception(
            f"Request error sending command {command} to FL subprocess "
            f"for job_id {job_id}",
        )
        return ResponseReturn(status="error", detail=dict(error=str(e)))
