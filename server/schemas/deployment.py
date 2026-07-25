from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from server.schemas.monitoring import MonitoringResponse


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


class DeploymentSummary(BaseModel):
    """
    One row on the frontend's Deployments page. Assembled by
    DeploymentService (not a plain from_attributes dump of the
    Deployment row) since it stitches together model metrics from
    TrainedModel, the dataset name from Dataset, and live serving
    status from the in-process ModelServerRegistry.
    """

    id: UUID
    # The value that actually works in POST /predict/{deployment_id} —
    # this is the pipeline run_id, not `id` above. Named deployment_id
    # here (rather than run_id) because that's the term the UI and
    # judges should see; "Run ID" is an internal implementation detail.
    deployment_id: str

    model_name: str | None
    dataset_name: str | None
    status: str | None

    # True if the model is currently loaded in the in-memory
    # ModelServerRegistry and will answer predictions immediately.
    # False means the deployment still exists (Postgres + MLflow have
    # it) but a restart happened since and it hasn't been reloaded —
    # only ever the case for a deployment that ISN'T the single most
    # recently completed one, since server/main.py always reloads that
    # one on startup.
    is_active: bool

    accuracy: float | None
    precision: float | None
    recall: float | None
    f1_score: float | None

    endpoint: str | None
    deployed_at: datetime | None


class DeploymentDetail(DeploymentSummary):
    """Single-deployment view — adds the latest monitoring snapshot."""

    monitoring: MonitoringResponse | None = None
