import asyncio
import logging

from flight_api.zmq_orchestrator_client import get_orchestrator_client

logger = logging.getLogger(__name__)


async def wait_for_subsystems() -> bool:
    logger.info("Starting application...")

    orchestrator_online = False
    orchestrator_client = get_orchestrator_client()
    while True:
        response = await orchestrator_client.zmq_sender(
            {"command": "health", "request_id": "N/A"}, 4.0
        )
        if response.status == "success":
            orchestrator_online = True
            break
        if response.detail.get("exception") != "TimeoutError":
            raise RuntimeError(f"Orchestrator error: {response.detail}")
        logger.info("Orchestrator not ready yet, waiting...")

    await asyncio.sleep(0.1)
    logger.info("All subsystems are operational.")
    return orchestrator_online
