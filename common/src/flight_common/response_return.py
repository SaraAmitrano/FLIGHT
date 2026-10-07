from dataclasses import dataclass, field
from typing import Any, Literal, cast

Status = Literal["error", "success"]


@dataclass
class ResponseReturn:
    status: Status = field(default="error")
    detail: dict[str, Any] = field(default_factory=lambda: cast("dict[str, Any]", {}))

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail}

    @classmethod
    def success(cls, detail: dict[str, Any]) -> "ResponseReturn":
        return cls(status="success", detail=detail)

    @classmethod
    def error(cls, detail: dict[str, Any]) -> "ResponseReturn":
        return cls(status="error", detail=detail)


def create_success_response(status_code: int, **kwargs: Any) -> ResponseReturn:
    return ResponseReturn.success(detail=dict(status_code=status_code, **kwargs))


def create_error_response(status_code: int, **kwargs: Any) -> ResponseReturn:
    return ResponseReturn.error(detail=dict(status_code=status_code, **kwargs))
