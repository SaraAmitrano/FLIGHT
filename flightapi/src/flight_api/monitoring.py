import logging

from flight_common.value_container import ValueContainer

logger = logging.getLogger(__name__)


subsystems_online = ValueContainer[bool]()


async def monitor_subsystems() -> None:
    logger.info("Monitoring subsystems...")
