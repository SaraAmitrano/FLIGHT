import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, override

from uvicorn.logging import AccessFormatter  # uvicorn-only import lives here

from flight_common.logging_config import (
    DEFAULT_FMT,
    RFC3339Formatter,
    assemble_config,
    make_console_handler,
    make_filters_dict,
    make_formatters_dict,
    make_rotating_file_handler,
    tag_logger_name,
    validate_level,
)

ACCESS_FMT = (
    "[{asctime}][{levelname:<8}] {stack} | "
    '{client_addr} - "{request_line}" {status_code} {body_length}'
)


class RFC3339AccessFormatter(AccessFormatter):
    @override
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        # defer to the same RFC3339 style as the main formatter

        return (
            datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat(timespec="microseconds")
        )

    def format(self, record: logging.LogRecord) -> str:
        # Make sure attributes exist even when called outside Uvicorn
        for key, default in {
            "client_addr": "-",
            "request_line": "-",
            "status_code": "-",
            "body_length": "-",
        }.items():
            if not hasattr(record, key):
                setattr(record, key, default)
        return super().format(record)


def make_logging_config(
    *,
    log_level: str | None = None,
    main_fmt: str = DEFAULT_FMT,
    access_fmt: str = ACCESS_FMT,
    trim_amount: int = 5,
    excluded_stack_modules: tuple[str, ...] = (
        "/asyncio/",
        "/logging/",
        "/site-packages/click",
        "/site-packages/fastapi",
        "/site-packages/starlette",
        "/site-packages/uvicorn",
        "<frozen runpy>",
        "/contextlib.py",
        "/__main__.py",
    ),
    excluded_access_stack_modules: tuple[str, ...] = (
        "/asyncio/",
        "/logging/",
        "/site-packages/click",
        "/site-packages/fastapi",
        "/site-packages/starlette",
        "/site-packages/uvicorn",
        "<frozen runpy>",
        "/contextlib.py",
        "/__main__.py",
    ),
    logger_names: Iterable[str] = (
        "flight_api",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.asgi",
        "uvicorn.access",
    ),
    logs_dir: Path | str = "/app/logs/api",
    version: int = 1,
    disable_existing_loggers: bool = False,
) -> dict[str, Any]:
    """
    FastAPI/Uvicorn config with per-logger console + timed-rotating file handlers,
    and a dedicated access formatter.
    """
    level = validate_level(log_level)
    logs_dir = Path(logs_dir)

    formatters = make_formatters_dict(
        main_fmt=main_fmt, main_formatter_cls=RFC3339Formatter
    )
    formatters["rfc3339_access"] = {
        "()": RFC3339AccessFormatter,
        "style": "{",
        "format": access_fmt,
    }

    filters = make_filters_dict(
        trim_amount=trim_amount, excluded_stack_modules=excluded_stack_modules
    )
    filters.update(
        make_filters_dict(
            trim_amount=trim_amount,
            excluded_stack_modules=excluded_access_stack_modules,
            filter_name="access_stack_info",
        )
    )

    handlers: dict[str, Any] = {}
    tags = {name: tag_logger_name(name) for name in logger_names}

    for name in logger_names:
        t = tags[name]
        formatter = "rfc3339_access" if name == "uvicorn.access" else "rfc3339_main"
        filt = ["access_stack_info"] if name == "uvicorn.access" else ["stack_info"]

        handlers[f"{t}_file"] = make_rotating_file_handler(
            level=level,
            formatter=formatter,
            filename=logs_dir / f"{t}.log",
            filters=filt,
        )
        handlers[f"{t}_console"] = make_console_handler(
            level=level, formatter=formatter, filters=filt
        )

    # Root fallback
    handlers["default"] = make_console_handler(
        level="WARNING", formatter="rfc3339_main", filters=["stack_info"]
    )

    loggers: dict[str, Any] = {}
    for name in logger_names:
        t = tags[name]
        loggers[name] = {
            "handlers": [f"{t}_console", f"{t}_file"],
            "level": level,
            "propagate": False,
        }

    root = {"handlers": ["default"], "level": "WARNING"}
    return assemble_config(
        formatters=formatters,
        filters=filters,
        handlers=handlers,
        loggers=loggers,
        root=root,
        version=version,
        disable_existing_loggers=disable_existing_loggers,
    )
