from dataclasses import dataclass


@dataclass(frozen=True)
class Envelope:
    identity: bytes | None = None
    empty: bytes = b""
    message_id: str | None = None
