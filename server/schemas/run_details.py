from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from server.schemas.pipeline_run import PipelineRunSummary
from server.schemas.deployment import DeploymentResponse
class DatasetResponse(BaseModel):

    id: UUID

    dataset_name: str

    dataset_path: str

    source_type: str

    filename: str

    file_size: int | None

    uploaded_at: datetime

    class Config:
        from_attributes = True


class PipelineLogResponse(BaseModel):
    id: UUID
    log_message: str
    created_at: datetime

    class Config:
        from_attributes = True


class TrainedModelResponse(BaseModel):
    id: UUID
    model_name: str
    model_path: str
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1_score: float | None

    class Config:
        from_attributes = True


class ModelRegistryResponse(BaseModel):
    id: UUID
    model_name: str
    model_path: str
    version: int
    status: str

    class Config:
        from_attributes = True


class RunDetailsResponse(BaseModel):
    run: PipelineRunSummary
    dataset: DatasetResponse | None
    logs: list[PipelineLogResponse]
    trained_models: list[TrainedModelResponse]
    registry: ModelRegistryResponse | None
    deployment: DeploymentResponse | None = None