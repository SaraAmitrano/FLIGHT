import logging
from types import FrameType
from typing import Any

import zmq
from pydantic import BaseModel, Field

from flight_common.response_return import (  # type: ignore[reportMissingTypeStubs]
    ResponseReturn,
)
from flight_common.value_container import (  # type: ignore[reportMissingTypeStubs]
    ValueContainer,
)
from flight_common.zmq_client import ZmqClient  # type: ignore[reportMissingTypeStubs]
from flight_common.zmq_poller import Poller  # type: ignore[reportMissingTypeStubs]
from flight_common.zmq_socket import Socket  # type: ignore[reportMissingTypeStubs]

stop_running: ValueContainer[bool] = ValueContainer()


logger = logging.getLogger(__name__)


def handle_stop_signal(
    signum: int,  # noqa: ARG001
    frame: FrameType | None,  # noqa: ARG001
) -> None:
    """Gracefully handle stop signal."""
    logger.info("Received stop signal, stopping training...")
    stop_running.set(True)


class ClientDetailedStatus(BaseModel):
    msg: str = Field(..., description="Status message")
    subset: str | None = Field(None, description="Subset of data being processed")
    round: int = Field(..., description="Current training round")
    rounds: int = Field(..., description="Total number of training rounds")
    epoch: int = Field(..., description="Current training epoch")
    epochs: int = Field(..., description="Total number of training epochs")

    def update(self, msg: str | None = None, **kwargs: Any) -> None:
        """Update the status with new values."""
        if msg is not None:
            self.msg = msg
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown field {key} in ClientDetailedStatus update.")


class ServerDetailedStatus(BaseModel):
    msg: str = Field(..., description="Status message")
    round: int = Field(..., description="Current training round")
    rounds: int = Field(..., description="Total number of training rounds")

    def update(self, msg: str | None = None, **kwargs: Any) -> None:
        """Update the status with new values."""
        if msg is not None:
            self.msg = msg
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown field {key} in ClientDetailedStatus update.")


class ErrorDetail(BaseModel):
    msg: str = Field(..., description="Error message")
    error: str = Field(..., description="Error details")


def send_status_update(
    job_id: str,
    which: str,
    client: ZmqClient,
    status: str,
    results: dict[str, Any] | None = None,
    detail: ClientDetailedStatus
    | ServerDetailedStatus
    | ErrorDetail
    | dict[str, Any]
    | None = None,
    timeout: float = 5.0,
) -> ResponseReturn:
    """Send a status update to main process via ZeroMQ with a timeout."""
    response = ResponseReturn()

    payload: dict[str, Any] = {
        "command": "update_status",
        "which": which,
        "job_id": job_id,
        "status": status,
    }

    if results:
        payload["results"] = results

    if detail:
        payload["detail"] = (
            detail.model_dump() if not isinstance(detail, dict) else detail
        )

    try:
        logger.debug(f"Sending status update to main process: {status}")
        response_data = client.request(payload, timeout=timeout)
        response = ResponseReturn(**response_data)
        logger.debug(f"Received response from main process: {response}")
    except zmq.ZMQError as e:
        logger.exception("Error sending status update to main process")
        response = ResponseReturn(
            status="error",
            detail=dict(
                error=str(e), message="Failed to send status update to main process."
            ),
        )
    except Exception as e:
        logger.exception("Unexpected error")
        response = ResponseReturn(
            status="error",
            detail=dict(
                error=str(e),
                message="Unknown error occurred while sending status update.",
            ),
        )
    return response


def check_receiving(
    job_id: str,
    rep_socket: Socket,
    poller: Poller,
    req_client: ZmqClient,  # noqa: ARG001
    status: str | None,
    detailed_status: ServerDetailedStatus | ClientDetailedStatus,
) -> str | None:
    # poll

    events = dict(poller.poll(1))
    if rep_socket in events:
        message, envelope = rep_socket.recv_data()
        command = message["command"]
        detail: dict[str, Any] = {}
        if command == "stop":
            logger.debug(f"Received {command} command for job_id {job_id}")
            status = "stopping"
        elif command == "status":
            logger.debug(f"Received {command} command for job_id {job_id}")
            detail.update(detailed_status)
        else:
            logger.warning(f"Unknown command: {command}")
        rep_socket.send_data(dict(status="success", detail=detail), envelope=envelope)

    if stop_running:
        status = None

    return status
