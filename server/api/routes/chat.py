from fastapi import UploadFile
from fastapi import File
from fastapi import Form

from fastapi import APIRouter, HTTPException, Depends
from server.auth.dependencies import get_current_user
from server.models.user import User


from server.dependencies import orchestration_service
from server.schemas import ChatResponse
from server.schemas.chat import (
    HyperparameterOptimizationResult,
    ModelEvaluationResult,
    ModelSelectionResult,
    ModelTrainingResult,
)
from fastapi import Depends
from sqlalchemy.orm import Session

from server.db.session import get_db
from server.repositories.dataset_repository import DatasetRepository

from state.pipeline_state import PipelineState

from server.services.upload_service import upload_service
from server.services.dataset_service import DatasetService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post("/", response_model=ChatResponse)
async def chat(
    prompt: str = Form(...),
    file: UploadFile | None = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
    ):

    """dataset = None

    if request.dataset_id:
        repository = DatasetRepository(db)

        dataset = repository.get_by_id(request.dataset_id)

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found."
            )"""

    state = PipelineState(
        user_prompt=prompt,
        user_id=str(current_user.id)
    )

    """if dataset:
        state.dataset.dataset_id = dataset.id      # <-- ADD THIS
        state.dataset.dataset_path = dataset.dataset_path
        state.dataset.dataset_name = dataset.dataset_name
        state.dataset.source_type = dataset.source_type

        state.logs.append("Uploaded dataset detected.")"""
    
    if file:

        upload_info = await upload_service.save_file(file)

        dataset = DatasetService(db).create_uploaded_dataset(
            current_user,
            upload_info,
        )

        state.dataset.dataset_id = str(dataset.id)
        state.dataset.dataset_name = dataset.dataset_name
        state.dataset.dataset_path = dataset.dataset_path
        state.dataset.source_type = dataset.source_type

        state.logs.append(
            "Uploaded dataset detected."
        )

    result = orchestration_service.run(state)

    model_selection = None
    if result.model_selection.is_completed:
        model_selection = ModelSelectionResult(
            task_type=result.model_selection.task_type,
            primary_model_name=result.model_selection.primary_model_name,
            primary_model_library=result.model_selection.primary_model_library,
            ranking=result.model_selection.ranking,
            reasoning=result.model_selection.reasoning,
            confidence=result.model_selection.confidence,
            summary=result.model_selection.summary,
        )

    model_training = None
    if result.model_training.is_completed:
        model_training = ModelTrainingResult(
            training_status=result.model_training.training_status,
            train_samples=result.model_training.train_samples,
            test_samples=result.model_training.test_samples,
            trained_models=result.model_training.trained_models,
            failed_models=result.model_training.failed_models,
            total_execution_time_seconds=result.model_training.total_execution_time_seconds,
            summary=result.model_training.summary,
        )

    hyperparameter_optimization = None
    if result.hyperparameter_optimization.is_completed:
        hyperparameter_optimization = HyperparameterOptimizationResult(
            optimization_status=result.hyperparameter_optimization.optimization_status,
            scoring_metric=result.hyperparameter_optimization.scoring_metric,
            best_overall_model_name=result.hyperparameter_optimization.best_overall_model_name,
            best_overall_score=result.hyperparameter_optimization.best_overall_score,
            optimized_models=result.hyperparameter_optimization.optimized_models,
            failed_models=result.hyperparameter_optimization.failed_models,
            total_execution_time_seconds=result.hyperparameter_optimization.total_execution_time_seconds,
            summary=result.hyperparameter_optimization.summary,
        )

    model_evaluation = None
    if result.model_evaluation.is_completed:
        model_evaluation = ModelEvaluationResult(
            evaluation_status=result.model_evaluation.evaluation_status,
            primary_metric=result.model_evaluation.primary_metric,
            best_model_name=result.model_evaluation.best_model_name,
            best_model_metrics=result.model_evaluation.best_model_metrics,
            comparison_table=result.model_evaluation.comparison_table,
            summary=result.model_evaluation.summary,
        )

    return ChatResponse(
        user_prompt=result.user_prompt,
        assistant_message=result.assistant_message,
        run_id=result.run_id,
        status=result.status,
        execution_time=result.execution_time,
        completed_steps=result.completed_steps,
        logs=result.logs,
        dataset_name=result.dataset.dataset_name,
        rows=result.dataset.num_rows,
        columns=result.dataset.num_columns,
        problem_type=result.validation.problem_type,
        target_column=result.validation.target_column,
        model_selection=model_selection,
        model_training=model_training,
        hyperparameter_optimization=hyperparameter_optimization,
        model_evaluation=model_evaluation,
        mlflow_run_id=result.mlflow_run_id,
        model_name=result.model_name,
        model_path=result.model_path,
        metrics=result.metrics,
    )