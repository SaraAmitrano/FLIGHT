import asyncio
import signal
from collections.abc import Callable, Coroutine
from typing import Any

from .shutdown import shutdown


def startup(main_task: Callable[[], Coroutine[Any, Any, int]]) -> None:
    main_task_name = "main_task"
    # db = Database(Path("data.db"))

    loop = asyncio.new_event_loop()
    # loop.set_debug(True)

    tasks: set[asyncio.Task[int]] = set()

    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(loop, main_task_name, s))
        )

    try:
        # loop.run_until_complete(db.init())
        # loop.create_task(main_task(db), name=MAIN_TASK_NAME)
        task = loop.create_task(main_task(), name=main_task_name)
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        loop.run_forever()
    finally:
        # loop.run_until_complete(db.close())
        loop.close()
