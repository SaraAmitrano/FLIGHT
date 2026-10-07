import asyncio
import logging
import signal

logger = logging.getLogger(__name__)


async def shutdown(
    loop: asyncio.AbstractEventLoop,
    main_task_name: str,
    signal: signal.Signals = signal.SIGTERM,
) -> None:
    logger.info(f"Received exit signal {signal.name}...")

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    subtasks = [task for task in tasks if task.get_name() != main_task_name]

    for task in subtasks:
        task.cancel()

    logger.debug(f"Cancelling {len(subtasks)} outstanding subtasks...")
    await asyncio.gather(*subtasks, return_exceptions=True)

    main_tasks = [task for task in tasks if task.get_name() == main_task_name]

    for task in main_tasks:
        task.cancel()

    logger.debug(f"Cancelling {len(main_tasks)} main outstanding tasks...")
    await asyncio.gather(*main_tasks, return_exceptions=True)

    loop.stop()
