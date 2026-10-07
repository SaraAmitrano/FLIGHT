from typing import Any

import zmq
import zmq.asyncio

from .zmq_connection_types import ConnectionType
from .zmq_envelope import Envelope
from .zmq_json import compress_json, decompress_json


class Socket:
    def __init__(self, socket: zmq.asyncio.Socket) -> None:
        self.socket = socket

    @classmethod
    def from_context(
        cls, context: zmq.asyncio.Context, conn_type: ConnectionType
    ) -> "Socket":
        return cls(context.socket(zmq.DEALER if conn_type == "SEND" else zmq.ROUTER))

    def connect(self, address: str) -> None:
        self.socket.connect(address)

    def bind(self, address: str) -> None:
        self.socket.bind(address)

    def setsockopt(self, option: int, value: Any) -> None:
        self.socket.setsockopt(option, value)

    async def send_data(
        self,
        payload: dict[str, Any],
        envelope: Envelope | None = None,
    ) -> None:
        env = envelope or Envelope()

        headers: dict[str, Any] = {}
        if env.message_id is not None:
            headers["message_id"] = env.message_id

        frames = [
            env.empty,
            compress_json(headers) if headers else b"",
            compress_json(payload),
        ]
        if env.identity is not None:
            frames = [env.identity, *frames]

        await self.socket.send_multipart(frames)  # type: ignore

    async def recv_data(self) -> tuple[dict[str, Any], Envelope]:
        data_with_compressed = await self.socket.recv_multipart()

        min_frames = 3  # empty, headers, data
        max_frames = 4  # identity, empty, headers, data

        if len(data_with_compressed) == min_frames:
            # No identity, just empty, headers and data
            identity = None
            empty, headers_raw, data_raw = data_with_compressed
        elif len(data_with_compressed) == max_frames:
            # Identity, empty, headers and data
            identity, empty, headers_raw, data_raw = data_with_compressed
        else:
            # Malformed message
            raise ValueError(
                f"Malformed message with length {len(data_with_compressed)}."
            )

        headers = decompress_json(headers_raw) if headers_raw else {}
        message_id = headers.get("message_id")
        payload = decompress_json(data_raw)
        env = Envelope(identity=identity, empty=empty, message_id=message_id)
        return payload, env

    def close(self) -> None:
        self.socket.close()
