from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DeploymentResponse(BaseModel):
    id: UUID
    model_id: UUID | None
    endpoint: str | None
    status: str | None
    deployed_at: datetime | None

    class Config:
        from_attributes = True


class PredictRequest(BaseModel):
    """
    Raw input rows shaped like the original (pre-feature-engineering)
    dataset — the bundled model applies feature engineering itself.
    Each dict in `records` is one row: {"column_name": value, ...}.
    """

    records: list[dict[str, Any]] = Field(
        ..., min_length=1, description="One or more raw input rows to score."
    )


class PredictResponse(BaseModel):
    deployment_id: str
    predictions: list[dict[str, Any]]
