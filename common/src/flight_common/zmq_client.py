import logging
import time
import uuid
from collections.abc import Callable
from types import TracebackType
from typing import Any

from .zmq_socket import Envelope, Socket  # your blocking wrapper

logger = logging.getLogger(__name__)


class ZmqClient:
    """
    Sync client without threads/futures:
      - exactly one request in flight
      - on_push handled inline while waiting for your reply
    """

    def __init__(
        self,
        socket: Socket,
        *,
        owns_socket: bool = False,
        on_push: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._sock = socket
        self._owns_socket = owns_socket
        self._on_push = on_push
        self._closed = False

    def start(self) -> None:
        pass  # nothing to do

    def close(self) -> None:
        self._closed = True
        if self._owns_socket:
            try:
                self._sock.close()
            except Exception:
                logger.debug("Socket.close raised", exc_info=True)

    def __enter__(self) -> "ZmqClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def request(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Client is closed")

        msg_id = uuid.uuid4().hex
        env = Envelope(message_id=msg_id)

        # Send
        self._sock.send_data(payload, env)

        # Receive until our reply arrives
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            # Compute remaining time for this recv
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            if deadline is not None and remaining == 0.0:
                raise TimeoutError("Request timed out")

            # Assumes your Socket supports an optional timeout argument; adapt if not.
            # Alternatives: set ZMQ_RCVTIMEO or use a poller in your Socket wrapper.
            payload_in, env_in = self._sock.recv_data()

            # If the envelope matches our request, return it
            if getattr(env_in, "message_id", None) == msg_id:
                return payload_in

            # Otherwise treat as push (unsolicited / other), then continue waiting
            if self._on_push is not None:
                try:
                    self._on_push(payload_in)
                except Exception:
                    logger.debug("on_push handler raised", exc_info=True)
