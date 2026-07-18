import time
from datetime import datetime

from graph import workflow
from state import PipelineState
from server.core.constants import PipelineStatus

from sqlalchemy.orm import Session

from server.db.database import SessionLocal
from server.models.pipeline_run import PipelineRun
from server.models.pipeline_log import PipelineLog
from server.models.trained_model import TrainedModel

from server.repositories.pipeline_run_repository import PipelineRunRepository
from server.repositories.pipeline_log_repository import PipelineLogRepository
from server.repositories.trained_model_repository import TrainedModelRepository

class OrchestrationService:
    """
    Service responsible for executing the LangGraph workflow.
    """

    def run(self, state: PipelineState) -> PipelineState:
        #pipeline started
        state.status = PipelineStatus.RUNNING
        start = time.perf_counter()

        db: Session = SessionLocal()
        pipeline_run_repository = PipelineRunRepository(db)
        pipeline_log_repository = PipelineLogRepository(db)
        trained_model_repository = TrainedModelRepository(db)

        pipeline_run = PipelineRun(
            id=state.run_id,
            dataset_id=state.dataset.dataset_id,
            user_prompt=state.user_prompt,
            assistant_message=None,
            status=PipelineStatus.RUNNING.value,
            started_at=datetime.now(),
        )

        pipeline_run_repository.create(pipeline_run)
        try:
            result = workflow.invoke(state)

            result = PipelineState.model_validate(result)

            result.status = PipelineStatus.SUCCESS

        except Exception as exc:

            state.status = PipelineStatus.FAILED

            pipeline_run.status = PipelineStatus.FAILED.value
            pipeline_run.completed_at = datetime.now()
            pipeline_run.execution_time = round(
                time.perf_counter() - start,
                2,
            )
            pipeline_run.assistant_message = str(exc)

            pipeline_run_repository.update(pipeline_run)

            raise

        finally:
            finish = datetime.now()
            db.close()
        
        result.completed_at = finish
        result.execution_time = round(
            time.perf_counter() - start, 2
        )

        pipeline_run.status = PipelineStatus.SUCCESS.value
        pipeline_run.completed_at = result.completed_at
        pipeline_run.execution_time = result.execution_time
        pipeline_run.assistant_message = result.assistant_message

        pipeline_run_repository.update(pipeline_run)


        comparison_lookup = {
            model["model_name"]: model
            for model in result.model_evaluation.comparison_table
        }

        trained_models = []

        for trained in result.model_training.trained_models:

            metrics = comparison_lookup.get(
                trained["model_name"],
                {},
            )

            trained_models.append(
                TrainedModel(
                    run_id=result.run_id,
                    model_name=trained["model_name"],
                    model_path=trained["model_identifier"],
                    accuracy=metrics.get("accuracy"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    f1_score=metrics.get("f1"),
                )
            )

        trained_model_repository.create_many(trained_models)

        logs = [
            PipelineLog(
                run_id=result.run_id,
                log_message=message,
            )
            for message in result.logs
        ]

        pipeline_log_repository.create_many(logs)

        return result