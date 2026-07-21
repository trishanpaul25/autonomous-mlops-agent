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
from server.models.model_registry import ModelRegistry
from server.models.deployment import Deployment
from server.models.dataset_snapshot import DatasetSnapshot

from server.repositories.pipeline_run_repository import PipelineRunRepository
from server.repositories.pipeline_log_repository import PipelineLogRepository
from server.repositories.trained_model_repository import TrainedModelRepository
from server.repositories.model_registry_repository import ModelRegistryRepository
from server.repositories.deployment_repository import DeploymentRepository
from server.repositories.dataset_snapshot_repository import DatasetSnapshotRepository

from tools.monitoring.dataset_snapshot_builder import DatasetSnapshotBuilder
from utils.logger import logger

class OrchestrationService:
    """
    Service responsible for executing the LangGraph workflow.
    """

    def run(self, state: PipelineState) -> PipelineState:
        #pipeline started
        state.status = PipelineStatus.RUNNING
        start = time.perf_counter()

        db: Session = SessionLocal()
        try:
            pipeline_run_repository = PipelineRunRepository(db)
            pipeline_log_repository = PipelineLogRepository(db)
            trained_model_repository = TrainedModelRepository(db)
            model_registry_repository = ModelRegistryRepository(db)
            deployment_repository = DeploymentRepository(db)
            dataset_snapshot_repository = DatasetSnapshotRepository(db)


            pipeline_run = PipelineRun(
                id=state.run_id,
                user_id=state.user_id,
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

            if result.model_evaluation.best_model_name:
                best_identifier = None

                for trained in result.model_training.trained_models:
                    if trained["model_name"] == result.model_evaluation.best_model_name:
                        best_identifier = trained["model_identifier"]
                        break
                best_model = ModelRegistry(

                    run_id=result.run_id,

                    model_name=result.model_evaluation.best_model_name,

                    model_path=best_identifier,

                    mlflow_run_id=result.mlflow_run_id,

                    version=1,

                    status="REGISTERED",
                )

                model_registry_repository.create(best_model)

                if result.deployment.deployment_status == "completed" and best_identifier:
                    matching_trained_model = next(
                        (
                            trained_model for trained_model in trained_models
                            if trained_model.model_path == best_identifier
                        ),
                        None,
                    )

                    if matching_trained_model is not None:
                        deployment_record = Deployment(
                            model_id=matching_trained_model.id,
                            endpoint=result.deployment.endpoint,
                            status=result.deployment.deployment_status,
                            deployed_at=datetime.now(),
                        )

                        deployment_repository.create(deployment_record)

                        # Capture the training-data reference distribution
                        # now, while `deployment_record.id` (the FK
                        # Monitoring/DatasetSnapshot key off) exists and
                        # the raw dataframe is still in memory. A failure
                        # here must never take down a successful
                        # deployment — it just means drift detection has
                        # nothing to compare against later.
                        try:
                            snapshot_payload = DatasetSnapshotBuilder().build(result)
                            if snapshot_payload is not None:
                                dataset_snapshot_repository.create(
                                    DatasetSnapshot(
                                        deployment_id=deployment_record.id,
                                        num_rows=snapshot_payload["num_rows"],
                                        target_column=snapshot_payload["target_column"],
                                        feature_statistics=snapshot_payload["feature_statistics"],
                                    )
                                )
                            else:
                                logger.warning(
                                    "DatasetSnapshot skipped for deployment %s: "
                                    "no raw dataframe available.",
                                    deployment_record.id,
                                )
                        except Exception as exc:
                            logger.warning(
                                "DatasetSnapshot capture failed for deployment %s: %s",
                                deployment_record.id,
                                exc,
                            )

            logs = [
                PipelineLog(
                    run_id=result.run_id,
                    log_message=message,
                )
                for message in result.logs
            ]

            pipeline_log_repository.create_many(logs)

            db.close()

            return result
        
        finally:
            
            db.close()