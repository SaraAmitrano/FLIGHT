import argparse
from dataclasses import dataclass

import uvicorn

from . import logging_uvicorn
from .api_main import app


@dataclass
class FLIGHTApiArgs:
    host: str
    port: int
    log_level: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FLIGHT API application with Uvicorn."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = FLIGHTApiArgs(**vars(parser.parse_args()))

    uvicorn.run(
        app,
        log_config=logging_uvicorn.make_logging_config(log_level=args.log_level),
        host=args.host,
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
