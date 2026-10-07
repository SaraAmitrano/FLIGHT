# ---------- Exceptions ----------


class ParseError(ValueError):
    """Raised when code fails structural/syntactic validation."""


class InvalidOperationError(ValueError):
    """Raised when user code attempts a forbidden operation."""


class SandboxError(Exception):
    """Base exception for sandboxing errors."""

    def __init__(
        self,
        message: str,
        original_exc: Exception | None,
        debug_messages: list[str] | None,
    ) -> None:
        super().__init__(message)
        self.original_exc = original_exc
        self.debug_messages = debug_messages


# ---------- Exports ----------

__all__ = [
    "InvalidOperationError",
    "ParseError",
]
