from datetime import datetime
from pydantic import BaseModel

from server.services.progress_types import ProgressEventType


class ProgressEvent(BaseModel):
    run_id: str
    type: ProgressEventType
    message: str
    timestamp: datetime = datetime.utcnow()