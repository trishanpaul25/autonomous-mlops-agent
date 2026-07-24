import asyncio
from collections import defaultdict

from server.services.events import ProgressEvent


class ProgressManager:

    def __init__(self):
        self._queues = defaultdict(asyncio.Queue)

    async def publish(self, event: ProgressEvent):
        await self._queues[event.run_id].put(event)

    async def subscribe(self, run_id: str):
        queue = self._queues[run_id]

        while True:
            event = await queue.get()
            yield event

    def cleanup(self, run_id: str):
        self._queues.pop(run_id, None)


progress_manager = ProgressManager()