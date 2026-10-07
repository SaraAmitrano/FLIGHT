import logging
import logging.config
from io import StringIO
from pathlib import Path
from typing import Any

from .logging_config import (
    DEFAULT_FMT,
    RFC3339Formatter,
    assemble_config,
    make_console_handler,
    make_filters_dict,
    make_formatters_dict,
    make_rotating_file_handler,
    make_stream_handler,
    validate_level,
)


def make_logging_config(
    *,
    log_level: str | None = None,
    fmt: str = DEFAULT_FMT,
    trim_amount: int = 5,
    excluded_stack_modules: tuple[str, ...] = (
        "/asyncio/",
        "/logging/",
        "<frozen runpy>",
        "/contextlib.py",
        "/__main__.py",
    ),
    filepath: Path | None = None,
    log_stream: StringIO | None = None,
    logger_name: str | None = None,
    version: int = 1,
    disable_existing_loggers: bool = False,
) -> dict[str, Any]:
    """
    Generic/service config:
      - console handler (stdout)
      - optional TimedRotatingFileHandler
      - optional in-memory stream (StringIO)
      - root or a named logger
    """
    level = validate_level(log_level)

    formatters = make_formatters_dict(main_fmt=fmt, main_formatter_cls=RFC3339Formatter)
    filters = make_filters_dict(
        trim_amount=trim_amount,
        excluded_stack_modules=excluded_stack_modules,
        filter_name="stack_info",
    )

    handlers = {
        "console": make_console_handler(
            level=level, formatter="rfc3339_main", filters=["stack_info"]
        )
    }

    if filepath is not None:
        handlers["file"] = make_rotating_file_handler(
            level=level,
            formatter="rfc3339_main",
            filename=filepath,
            filters=["stack_info"],
        )

    if log_stream is not None:
        handlers["memstream"] = {
            "()": make_stream_handler,
            "level": level,
            "formatter": "rfc3339_main",
            "filters": ["stack_info"],
            "stream": log_stream,
        }

    attached = list(handlers.keys())
    root: dict[str, Any] = (
        {"level": level, "handlers": attached}
        if logger_name is None
        else {"level": "WARNING", "handlers": []}
    )
    loggers = None
    if logger_name is not None:
        loggers = {
            logger_name: {"level": level, "handlers": attached, "propagate": False}
        }

    return assemble_config(
        formatters=formatters,
        filters=filters,
        handlers=handlers,
        loggers=loggers,
        root=root,
        version=version,
        disable_existing_loggers=disable_existing_loggers,
    )


def build_logger(
    cfg: dict[str, Any] | None = None, filepath: Path | None = None
) -> logging.Logger:
    if cfg is None:
        if filepath is None:
            from .config import LOGS_PATH  # noqa: PLC0415

            filepath = LOGS_PATH / "app.log"
        cfg = make_logging_config(filepath=filepath)
    logging.config.dictConfig(cfg)
    return logging.getLogger()


__all__ = ["build_logger"]
