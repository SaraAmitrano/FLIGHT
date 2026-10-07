import os
from dataclasses import dataclass, field
from pathlib import Path

import zmq.asyncio as zmq
from jsonschema import Draft7Validator

from flight_api.core.flparams_schema_reader import (
    get_schema_validator,
    load_all_schemas,
)
from flight_common.config import MAIN_PATH
from flight_common.network import get_service_addresses, read_network


@dataclass
class AppData:
    NUM_CLIENTS: int = int(os.getenv("NUM_CLIENTS", "4"))
    MAIN_DIR = MAIN_PATH
    LOGS_DIR = Path("/app/logs").resolve()
    flparams_schema_validator: Draft7Validator = field(
        default_factory=lambda: Draft7Validator({})
    )
    # NETWORK_DATA: dict[str, dict[str, int]] = field(default_factory=dict)
    zmq_address: str = field(default_factory=str)
    zmq_context: zmq.Context = field(default_factory=zmq.Context)

    def __post_init__(self) -> None:
        self.flparams_schema_validator = get_schema_validator(
            *load_all_schemas("flparams")
        )

        self.zmq_address = get_service_addresses(
            read_network(),
            "api",
            ["orchestrator"],
            "tcp",
        ).SEND["orchestrator"]

    def terminate(self) -> None:
        self.zmq_context.term()


app_data = AppData()
