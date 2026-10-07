import logging
from typing import Any

from flight_common.response_return import ResponseReturn, create_success_response

logger = logging.getLogger(__name__)


async def handle_command_get_health(
    message: dict[str, Any],  # noqa: ARG001
) -> ResponseReturn:
    return create_success_response(200)
