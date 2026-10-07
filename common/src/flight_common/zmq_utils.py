import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from zmq.asyncio import Context

from .response_return import ResponseReturn
from .utils import get_full_class_name
from .zmq_asyncio_client import AsyncZmqClient
from .zmq_asyncio_socket import Socket
from .zmq_connection_types import ConnectionType

logger = logging.getLogger(__name__)

ZMQ_RESPONSE_LEN = 2


@asynccontextmanager
async def zmq_socket(
    zmq_context: Context, conn_type: ConnectionType, address: str
) -> AsyncGenerator[Socket]:
    socket = Socket.from_context(zmq_context, conn_type)
    try:
        socket.connect(address)
        yield socket
    finally:
        socket.close()


async def send_zeromq_request(
    zmq_context: Context, address: str, payload: dict[str, Any], timeout: float = 100.0
) -> ResponseReturn:
    try:
        logger.debug(
            "Sending command %s to address %s - (timeout=%s)", payload, address, timeout
        )
        async with (
            zmq_socket(zmq_context, "SEND", address) as socket,
            AsyncZmqClient(socket, owns_socket=False) as client,
        ):
            response = await client.request(payload, timeout=timeout)

        if "status" not in response:
            raise ValueError("Missing status from response")

        if "detail" not in response:
            raise ValueError("Missing detail from response")

        if len(response) != ZMQ_RESPONSE_LEN:
            raise ValueError("Invalid response format")
    except TimeoutError as e:
        # logger.error(
        #     f"TimeoutError for address {address}.\nPayload:\n{payload}\nDetails: {e}"
        # )
        return ResponseReturn(
            status="error",
            detail=dict(
                exception=get_full_class_name(e),
                address=address,
                sent_payload=payload,
                error="Request timed out",
            ),
        )
    except Exception as e:
        # logger.error(
        #     f"Unknown error (type: {type(e)}) for address {address}.\n"
        #     f"Payload:\n{payload}\nDetails: {e}"
        # )
        return ResponseReturn(
            status="error",
            detail=dict(
                exception=get_full_class_name(e),
                address=address,
                sent_payload=payload,
                error=str(e),
            ),
        )
    else:
        return ResponseReturn(**response)
