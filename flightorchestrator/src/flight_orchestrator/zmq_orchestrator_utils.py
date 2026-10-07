import asyncio
import logging
from typing import Any

from flight_common.response_return import ResponseReturn
from flight_common.utils import get_full_class_name
from flight_common.zmq_asyncio_client import AsyncZmqClient

logger = logging.getLogger(__name__)

LEN_RESPONSE = 2


async def send_zeromq_request(
    service: str,
    client: AsyncZmqClient,
    payload: dict[str, Any],
    *,
    timeout: float = 5.0,
) -> ResponseReturn:
    try:
        async with asyncio.Lock():
            response = await client.request(payload, timeout=timeout)

        if "status" not in response:
            raise ValueError("Missing status from response")
        if "detail" not in response:
            raise ValueError("Missing detail from response")
        if len(response) != LEN_RESPONSE:
            raise ValueError("Invalid response format")

        return ResponseReturn(**response)

    except Exception as e:
        # Any error (timeout, ZMQ, decode, etc.) is mapped to your error return
        details = dict(
            exception=get_full_class_name(e),
            service=str(service),
            sent_payload=payload,
            error=str(e),
        )

        if details["exception"] != "TimeoutError":
            logger.exception(
                f"Request error for service {service}.\nPayload:\n{payload}\n"
                f"Details:\n{details}"
            )
        return ResponseReturn(
            status="error",
            detail=details,
        )
