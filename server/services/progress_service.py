import asyncio
import logging
from uuid import UUID

from server.services.events import ProgressEvent
from server.services.progress_manager import progress_manager
from server.services.progress_types import ProgressEventType

logger = logging.getLogger(__name__)


class ProgressService:

    @staticmethod
    def emit(
        run_id: UUID | str,
        message: str,
        event_type: ProgressEventType = ProgressEventType.INFO,
    ):

        event = ProgressEvent(
            run_id=str(run_id),
            type=event_type,
            message=message,
        )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(progress_manager.publish(event))
        except RuntimeError:
            try:
                asyncio.run(progress_manager.publish(event))
            except Exception:
                logger.exception("Failed to publish progress event")