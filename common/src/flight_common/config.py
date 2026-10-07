import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MAIN_PATH = Path(os.getenv("MAIN_DIR", ".")).resolve(strict=True)
LOGS_PATH = Path(os.getenv("LOGS_DIR", "/app/logs")).resolve()
