from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class PipelineRunSummary(BaseModel):

    id: UUID

    dataset_id: UUID | None

    status: str

    execution_time: float | None

    started_at: datetime

    completed_at: datetime | None

    class Config:
        from_attributes = True