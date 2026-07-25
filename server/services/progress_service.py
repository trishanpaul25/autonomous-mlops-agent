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
            # Thread-safe: works whether this is called from the main
            # event loop or from the worker thread the pipeline runs on
            # (via run_in_threadpool in the /chat route).
            progress_manager.publish_threadsafe(event)
        except Exception:
            logger.exception("Failed to publish progress event")