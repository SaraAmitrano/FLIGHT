from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, TypedDict

from flight_common.response_return import ResponseReturn, create_error_response
from flight_common.zmq_asyncio_client import AsyncZmqClient

from .command_handlers import (
    handle_command_get_data,
    handle_command_get_health,
    handle_command_get_job,
    handle_command_get_status,
    handle_command_post_delete,
    handle_command_post_start_job,
    handle_command_post_stop_job,
    handle_command_pre_delete,
    handle_command_pre_start_job,
    handle_command_pre_stop_job,
    handle_command_reset,
)
from .service import FLService

PreHandlerFunc = Callable[[dict[str, Any]], Awaitable[ResponseReturn]]

PostHandlerFunc = Callable[
    [dict[str, Any], dict[FLService, AsyncZmqClient]], Awaitable[ResponseReturn]
]


class CommandData(TypedDict):
    pre_func: PreHandlerFunc
    post_func: PostHandlerFunc
    receiver: str


async def handle_command_post_default(
    message: dict[str, Any],  # noqa: ARG001
    _: dict[FLService, AsyncZmqClient],
) -> ResponseReturn:
    return create_error_response(200, message="")


command_data: dict[str, CommandData] = {
    "start_job": CommandData(
        pre_func=handle_command_pre_start_job,
        post_func=handle_command_post_start_job,
        receiver="api",
    ),
    "stop_job": CommandData(
        pre_func=handle_command_pre_stop_job,
        post_func=handle_command_post_stop_job,
        receiver="api",
    ),
    "status": CommandData(
        pre_func=handle_command_get_status,
        post_func=handle_command_post_default,
        receiver="api",
    ),
    "health": CommandData(
        pre_func=handle_command_get_health,
        post_func=handle_command_post_default,
        receiver="api",
    ),
    "get_data": CommandData(
        pre_func=handle_command_get_data,
        post_func=handle_command_post_default,
        receiver="api",
    ),
    "get_job": CommandData(
        pre_func=handle_command_get_job,
        post_func=handle_command_post_default,
        receiver="api",
    ),
    "reset": CommandData(
        pre_func=handle_command_reset,
        post_func=handle_command_post_default,
        receiver="api",
    ),
    "delete": CommandData(
        pre_func=handle_command_pre_delete,
        post_func=handle_command_post_delete,
        receiver="api",
    ),
}


async def handle_unknown_command(message: dict[str, Any]) -> ResponseReturn:
    command = message.get("command")
    available_commands = ", ".join(command_data.keys())
    return create_error_response(
        500,
        message=f"Unknown command: {command}. Available commands: {available_commands}",
    )


async def handle_invalid_receiver(
    message: dict[str, Any], receiver: str
) -> ResponseReturn:
    command = message["command"]
    valid_receiver = command_data[command]["receiver"]

    return create_error_response(
        500,
        message=f"Command {command} received from {receiver} "
        f"but expected from {valid_receiver}",
    )


def get_command_handlers(
    command: str, receiver: str
) -> tuple[PreHandlerFunc, PostHandlerFunc]:
    """
    Get the command handler function for a given command.

    Args:
        command (str): The command string,
        receiver (str): The receiver string.

    Returns:
        HandlerFunc: The handler function for the command.

    """
    if command not in command_data:
        return handle_unknown_command, handle_command_post_default

    cmd_data = command_data[command]
    if not receiver.startswith(cmd_data["receiver"]):
        return partial(
            handle_invalid_receiver, receiver=receiver
        ), handle_command_post_default

    return command_data[command]["pre_func"], command_data[command]["post_func"]
