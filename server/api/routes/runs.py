from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from server.db.session import get_db

from server.repositories.pipeline_run_repository import (
    PipelineRunRepository,
)

from server.repositories.pipeline_log_repository import PipelineLogRepository
from server.repositories.trained_model_repository import TrainedModelRepository
from server.repositories.model_registry_repository import ModelRegistryRepository
from server.repositories.dataset_repository import DatasetRepository

from server.schemas import RunDetailsResponse
from server.schemas import PipelineRunSummary


router = APIRouter(
    prefix="/runs",
    tags=["Pipeline Runs"],
)


@router.get(
    "/",
    response_model=list[PipelineRunSummary],
)
def get_pipeline_runs(
    db: Session = Depends(get_db),
):

    repository = PipelineRunRepository(db)

    return repository.get_all()


@router.get(
    "/{run_id}",
    response_model=RunDetailsResponse,
)
def get_pipeline_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    pipeline_repository = PipelineRunRepository(db)
    dataset_repository = DatasetRepository(db)
    log_repository = PipelineLogRepository(db)
    trained_repository = TrainedModelRepository(db)
    registry_repository = ModelRegistryRepository(db)

    run = pipeline_repository.get_by_id(run_id)

    dataset = None

    if run.dataset_id:
        dataset = dataset_repository.get_by_id(
            run.dataset_id
        )

    if run is None:
        raise HTTPException(
            status_code=404,
            detail="Pipeline run not found."
        )

    return RunDetailsResponse(
        run=run,
        dataset=dataset,
        logs=log_repository.get_by_run_id(run_id),
        trained_models=trained_repository.get_by_run_id(run_id),
        registry=registry_repository.get_by_run_id(run_id),
    )