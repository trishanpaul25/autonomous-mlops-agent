from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID
from server.models.user import User

from server.auth.dependencies import get_current_user

from server.db.session import get_db

from server.repositories.pipeline_run_repository import (
    PipelineRunRepository,
)

from server.repositories.pipeline_log_repository import PipelineLogRepository
from server.repositories.trained_model_repository import TrainedModelRepository
from server.repositories.model_registry_repository import ModelRegistryRepository
from server.repositories.dataset_repository import DatasetRepository
from server.repositories.deployment_repository import DeploymentRepository

from server.schemas import RunDetailsResponse
from server.schemas import PipelineRunSummary


from sse_starlette.sse import EventSourceResponse
from server.services.progress_manager import progress_manager
from server.services.events import ProgressEvent
from server.services.progress_types import ProgressEventType

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
    current_user: User = Depends(get_current_user),
):

    repository = PipelineRunRepository(db)

    return repository.get_by_user_id(current_user.id)


@router.get(
    "/{run_id}",
    response_model=RunDetailsResponse,
)
def get_pipeline_run(
    run_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    pipeline_repository = PipelineRunRepository(db)
    dataset_repository = DatasetRepository(db)
    log_repository = PipelineLogRepository(db)
    trained_repository = TrainedModelRepository(db)
    registry_repository = ModelRegistryRepository(db)
    deployment_repository = DeploymentRepository(db)

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

    if run.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="You are not allowed to access this run."
        )


    return RunDetailsResponse(
        run=run,
        dataset=dataset,
        logs=log_repository.get_by_run_id(run_id),
        trained_models=trained_repository.get_by_run_id(run_id),
        registry=registry_repository.get_by_run_id(run_id),
        deployment=deployment_repository.get_by_run_id(run_id),
    )


@router.get("/{run_id}/events")
async def stream_run_events(run_id: str):

    async def event_generator():
        try:
            async for event in progress_manager.subscribe(run_id):
                yield {
                    "event": event.type.value,
                    "data": event.model_dump_json(),
                }
        finally:
            progress_manager.cleanup(run_id)

    return EventSourceResponse(event_generator())


"""@router.post("/test-stream/{run_id}")
async def test_stream(run_id: str):

    await progress_manager.publish(
        ProgressEvent(
            run_id=run_id,
            type=ProgressEventType.INFO,
            message="Hello from backend!"
        )
    )

    return {"status": "sent"}"""