import logging
from typing import Any

from flight_common.response_return import ResponseReturn
from flight_common.zmq_utils import send_zeromq_request

from .config import app_data

logger = logging.getLogger(__name__)


async def send_zeromq_request_to_orchestrator(
    payload: dict[str, Any], timeout: float = 100.0
) -> ResponseReturn:
    return await send_zeromq_request(
        app_data.zmq_context, app_data.zmq_address, payload, timeout=timeout
    )
