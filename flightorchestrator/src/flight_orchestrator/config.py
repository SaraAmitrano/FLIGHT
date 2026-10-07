import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

NUM_CLIENTS = int(os.environ["NUM_CLIENTS"])

MIN_NUM_CLIENTS = 2
MAX_NUM_CLIENTS = 15

if NUM_CLIENTS < MIN_NUM_CLIENTS:
    raise ValueError(
        f"NUM_CLIENTS is set to {NUM_CLIENTS}, which is less than the minimum of 2."
    )
if NUM_CLIENTS > MAX_NUM_CLIENTS:
    raise ValueError(
        f"NUM_CLIENTS is set to {NUM_CLIENTS}, which is greater than the maximum of 15."
    )

MAX_CONCURRENT_JOBS = NUM_CLIENTS // 2

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "flserver")
if not SERVER_ADDRESS:
    raise ValueError("SERVER_ADDRESS environment variable is not set.")

CLIENTS_ADDRESSES_ENV = os.getenv("CLIENTS_ADDRESSES")
CLIENTS_ADDRESSES = (
    CLIENTS_ADDRESSES_ENV.split(",") if CLIENTS_ADDRESSES_ENV is not None else None
)

if CLIENTS_ADDRESSES is not None and len(CLIENTS_ADDRESSES) != NUM_CLIENTS:
    raise ValueError(
        f"NUM_CLIENTS is set to {NUM_CLIENTS}, "
        f"but CLIENTS_ADDRESSES has {len(CLIENTS_ADDRESSES)} addresses."
    )

FLSERVER_PORTRANGE_ENV = os.getenv("FLSERVER_PORTRANGE")
if FLSERVER_PORTRANGE_ENV is None:
    raise ValueError("FLSERVER_PORTRANGE environment variable is not set.")
SERVER_MIN_PORT, SERVER_MAX_PORT = map(int, FLSERVER_PORTRANGE_ENV.split("-"))


def get_data_path() -> Path:
    return Path(os.getenv("DATA_PATH", "/opt/data")).resolve()


def get_datasplit_path() -> Path:
    return Path(os.getenv("DATA_SPLIT_PATH", "/opt/datasplit")).resolve()


def get_model_weights_path() -> Path:
    return Path(os.getenv("MODEL_WEIGHTS_PATH", "/opt/model_weights")).resolve()
