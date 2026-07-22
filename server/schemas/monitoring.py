from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class MonitoringResponse(BaseModel):
    id: UUID
    deployment_id: UUID | None
    prediction_count: int | None
    average_latency: float | None
    drift_score: float | None
    accuracy: float | None
    alert_status: str | None
    last_checked: datetime | None

    class Config:
        from_attributes = True
