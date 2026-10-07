import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import suppress
from types import TracebackType
from typing import Any

from .zmq_asyncio_socket import Envelope, Socket

logger = logging.getLogger(__name__)


class AsyncZmqClient:
    """
    High-level DEALER client that:
      - is initialized with a preconfigured Socket (dependency-injected)
      - does request/response via correlation IDs (Envelope.message_id)
      - runs a background recv loop
      - supports timeouts, backpressure, and clean shutdown
    """

    def __init__(
        self,
        socket: Socket,
        *,
        owns_socket: bool = False,
        on_push: Callable[[dict[str, Any]], None] | None = None,
        max_in_flight: int = 1000,
    ) -> None:
        """
        Args:
            socket: A preconfigured Socket (should wrap a DEALER).
                    Callers are responsible for setting identity and connect().
            owns_socket: If True, close() will also close the injected socket.
            on_push: Optional callback for unsolicited server -> client messages.
            max_in_flight: Upper bound on outstanding requests.

        """
        self._sock = socket
        self._owns_socket = owns_socket
        self._on_push = on_push
        self._max_in_flight = max_in_flight

        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._pending_lock = asyncio.Lock()
        self._recv_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._recv_task is None:
            self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self) -> None:
        self._closed = True
        if self._recv_task:
            self._recv_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None

        # Fail any outstanding requests
        async with self._pending_lock:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(asyncio.CancelledError("Client closed"))
            self._pending.clear()

        if self._owns_socket:
            self._sock.close()

    async def __aenter__(self) -> "AsyncZmqClient":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        await self.close()

    async def request(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Send a request and await its reply. Replies are matched via Envelope.message_id.
        If message_id is None, the Socket.send_data implementation will assign one
        (assuming your send_data auto-fills when missing).
        """
        if self._closed:
            raise RuntimeError("Client is closed")

        # Backpressure
        async with self._pending_lock:
            if len(self._pending) >= self._max_in_flight:
                raise RuntimeError("Too many in-flight requests")

        msg_id = uuid.uuid4().hex
        env = Envelope(message_id=msg_id)

        # Register future *after* send (common pattern; you can also pre-register)
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        async with self._pending_lock:
            # Guard against race: if a very fast reply arrives before we register
            # we still add and resolve below in _recv_loop.
            # If already resolved, we'll pop it there.
            self._pending[msg_id] = fut

        await self._sock.send_data(payload, env)

        try:
            if timeout is None:
                return await fut
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            # Cleanup in case of timeout/cancel
            async with self._pending_lock:
                self._pending.pop(msg_id, None)

    async def _recv_loop(self) -> None:
        try:
            while True:
                payload, env = await self._sock.recv_data()
                msg_id = env.message_id

                if msg_id is not None:
                    async with self._pending_lock:
                        fut = self._pending.pop(msg_id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(payload)
                        continue

                # Uncorrelated message -> push callback (if any)
                if self._on_push is not None:
                    with suppress(Exception):
                        self._on_push(payload)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            # Propagate error to all pending
            async with self._pending_lock:
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_exception(e)
                self._pending.clear()
            raise
