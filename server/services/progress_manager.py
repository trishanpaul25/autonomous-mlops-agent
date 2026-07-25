import asyncio
from collections import defaultdict

from server.services.events import ProgressEvent


class ProgressManager:

    def __init__(self):
        self._queues = defaultdict(asyncio.Queue)
        # The loop the FastAPI app (and therefore every SSE subscriber) is
        # running on. Set once at startup via bind_loop(). Needed because
        # publish() can be called from a worker thread (the pipeline runs
        # via run_in_threadpool), and asyncio.Queue is not thread-safe —
        # any put() has to be scheduled back onto this loop.
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def publish(self, event: ProgressEvent):
        await self._queues[event.run_id].put(event)

    def publish_threadsafe(self, event: ProgressEvent) -> None:
        """Safe to call from any thread, including a run_in_threadpool worker."""
        if self._loop is None:
            # No loop bound yet (e.g. called before startup) — best effort,
            # only safe if we happen to already be on an event loop thread.
            asyncio.get_event_loop().create_task(self.publish(event))
            return

        asyncio.run_coroutine_threadsafe(self.publish(event), self._loop)

    async def subscribe(self, run_id: str):
        queue = self._queues[run_id]

        while True:
            event = await queue.get()
            yield event

    def cleanup(self, run_id: str):
        self._queues.pop(run_id, None)


progress_manager = ProgressManager()