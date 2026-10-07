# logging_common_utils.py
from __future__ import annotations

import logging
import sysconfig
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

DEFAULT_FMT = "[{asctime}][{levelname:<8}] {stack} | {message}"


def get_default_log_level() -> str:
    import os  # noqa: PLC0415

    return os.getenv("LOG_LEVEL", "INFO").upper()


class RFC3339Formatter(logging.Formatter):
    def formatTime(  # noqa: N802
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,  # noqa: ARG002
    ) -> str:
        # Always RFC3339 with microseconds, localized
        return (
            datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat(timespec="microseconds")
        )


class StackInfoFilter(logging.Filter):
    def __init__(
        self,
        *,
        excluded_stack_modules: tuple[str, ...] = (
            "/asyncio/",
            "/logging/",
            "<frozen runpy>",
            "/contextlib.py",
            "/__main__.py",
        ),
        trim_amount: int = -1,
    ) -> None:
        super().__init__()
        self.excluded_modules = excluded_stack_modules
        self.trim_amount = trim_amount

        # stdlib is for Python library code, purelib is for 3rd-party packages
        # (e.g., site-packages). Purelib may be a subdir of stdlib (e.g., venv), hence
        # it should be processed first.
        self.stdlib_paths = {f: sysconfig.get_path(f) for f in ("purelib", "stdlib")}

    def filter(self, record: logging.LogRecord) -> bool:
        stack_summary = traceback.extract_stack()[:-1]
        filtered_frames: list[str] = []

        for frame in stack_summary:
            if any(subf in frame.filename for subf in self.excluded_modules):
                continue
            try:
                rel_path = Path(frame.filename).relative_to(Path().absolute())
            except Exception:
                rel_path = frame.filename
                for stdlib, path in self.stdlib_paths.items():
                    if path and rel_path.startswith(path):
                        rel_path = rel_path.replace(path, f"${{{stdlib}}}")

            filtered_frames.append(f"{rel_path}:{frame.lineno}({frame.name})")

        if self.trim_amount == 0:
            filtered_frames = []
        elif self.trim_amount > 0:
            filtered_frames = filtered_frames[-self.trim_amount :]

        record.stack = " -> ".join(filtered_frames)
        return True


def make_stream_handler(*, stream: IO[str]) -> logging.Handler:
    """
    Tiny factory so dictConfig can bind a StreamHandler to an arbitrary stream
    (e.g., a StringIO for tests).
    """
    return logging.StreamHandler(stream)


@dataclass(frozen=True)
class RotatingParams:
    when: str = "midnight"
    interval: int = 1
    backupCount: int = 5  # noqa: N815
    encoding: str = "utf-8"
    delay: bool = True


DEFAULT_ROTATING_PARAMS = RotatingParams()


def make_formatters_dict(
    *, main_fmt: str, main_formatter_cls: type = RFC3339Formatter
) -> dict[str, Any]:
    """Common main formatter dict (FastAPI can add its own access formatter)."""
    return {
        "rfc3339_main": {
            "()": main_formatter_cls,
            "style": "{",
            "format": main_fmt,
        }
    }


def make_filters_dict(
    *,
    trim_amount: int,
    excluded_stack_modules: tuple[str, ...],
    filter_name: str = "stack_info",
) -> dict[str, Any]:
    """Common stack filter dict (FastAPI can add an extra access filter)."""
    return {
        filter_name: {
            "()": StackInfoFilter,
            "trim_amount": trim_amount,
            "excluded_stack_modules": excluded_stack_modules,
        }
    }


def make_console_handler(
    *,
    level: str,
    formatter: str,
    filters: list[str] | None = None,
    stream: str = "ext://sys.stdout",
) -> dict[str, Any]:
    h: dict[str, Any] = {
        "class": "logging.StreamHandler",
        "level": level,
        "formatter": formatter,
        "stream": stream,
    }
    if filters:
        h["filters"] = filters
    return h


def make_rotating_file_handler(
    *,
    level: str,
    formatter: str,
    filename: Path | str,
    filters: list[str] | None = None,
    params: RotatingParams = DEFAULT_ROTATING_PARAMS,
) -> dict[str, Any]:
    h: dict[str, Any] = {
        "class": "logging.handlers.TimedRotatingFileHandler",
        "level": level,
        "formatter": formatter,
        "filename": str(filename),
        "when": params.when,
        "interval": params.interval,
        "backupCount": params.backupCount,
        "encoding": params.encoding,
        "delay": params.delay,
    }
    if filters:
        h["filters"] = filters
    return h


def validate_level(level: str | None) -> str:
    lvl = (level or get_default_log_level()).upper()
    if lvl not in _VALID_LEVELS:
        raise ValueError(f"Invalid log level: {lvl}")
    return lvl


def tag_logger_name(name: str) -> str:
    """Convert dots to underscores for logger names used in dictConfig."""
    return name.replace(".", "_")


def assemble_config(
    *,
    formatters: dict[str, Any],
    filters: dict[str, Any],
    handlers: dict[str, Any],
    loggers: dict[str, Any] | None,
    root: dict[str, Any],
    version: int = 1,
    disable_existing_loggers: bool = False,
) -> dict[str, Any]:
    """Shared dictConfig skeleton."""
    cfg: dict[str, Any] = {
        "version": version,
        "disable_existing_loggers": disable_existing_loggers,
        "formatters": formatters,
        "filters": filters,
        "handlers": handlers,
        "root": root,
    }
    if loggers:
        cfg["loggers"] = loggers
    return cfg
