"""
Deployment Service.

Read-side aggregator for the frontend's Deployments page. A `Deployment`
row on its own is nearly meaningless to a viewer — it's just a status
string and two foreign keys — so this stitches together:

  - TrainedModel (via deployment.model_id)   -> model name + metrics
  - PipelineRun -> Dataset (via deployment.run_id) -> dataset name
  - ModelServerRegistry (in-memory, not DB)  -> is it actually loaded
    right now, or only recoverable from MLflow on next restart?
  - Monitoring (latest row, detail view only) -> drift/latency snapshot

into the single shape server/schemas/deployment.py's DeploymentSummary
/ DeploymentDetail expect, so the client doesn't have to make four
separate calls (and reimplement this joining) itself.

No LLM, no PipelineState, no LangGraph — purely a DB + in-memory read,
same category as MonitoringService.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from server.models.deployment import Deployment
from server.repositories.deployment_repository import DeploymentRepository
from server.repositories.trained_model_repository import TrainedModelRepository
from server.repositories.pipeline_run_repository import PipelineRunRepository
from server.repositories.dataset_repository import DatasetRepository
from server.repositories.monitoring_repository import MonitoringRepository
from server.schemas.deployment import DeploymentSummary, DeploymentDetail
from tools.deployment.model_server_registry import ModelServerRegistry


class DeploymentService:

    def __init__(self, db: Session):
        self.db = db
        self.deployment_repository = DeploymentRepository(db)
        self.trained_model_repository = TrainedModelRepository(db)
        self.pipeline_run_repository = PipelineRunRepository(db)
        self.dataset_repository = DatasetRepository(db)
        self.monitoring_repository = MonitoringRepository(db)

    def _build_summary(self, deployment: Deployment) -> DeploymentSummary:
        trained_model = (
            self.trained_model_repository.get_by_id(deployment.model_id)
            if deployment.model_id
            else None
        )

        pipeline_run = (
            self.pipeline_run_repository.get_by_id(deployment.run_id)
            if deployment.run_id
            else None
        )

        dataset = None
        if pipeline_run is not None and pipeline_run.dataset_id:
            dataset = self.dataset_repository.get_by_id(pipeline_run.dataset_id)

        # ModelServerRegistry (and DeploymentAgent, and predict.py) all key
        # off run_id — fall back to the deployment's own id only in the
        # unexpected case a row was somehow created without a run_id.
        deployment_id = str(deployment.run_id or deployment.id)

        return DeploymentSummary(
            id=deployment.id,
            deployment_id=deployment_id,
            model_name=trained_model.model_name if trained_model else None,
            dataset_name=dataset.dataset_name if dataset else None,
            status=deployment.status,
            is_active=ModelServerRegistry.is_registered(deployment_id),
            accuracy=trained_model.accuracy if trained_model else None,
            precision=trained_model.precision if trained_model else None,
            recall=trained_model.recall if trained_model else None,
            f1_score=trained_model.f1_score if trained_model else None,
            endpoint=deployment.endpoint,
            deployed_at=deployment.deployed_at,
        )

    def list_for_user(self, user_id: UUID) -> list[DeploymentSummary]:
        """All deployments a user has ever created, newest first."""
        deployments = self.deployment_repository.get_by_user_id(user_id)
        return [self._build_summary(d) for d in deployments]

    def get_detail(self, deployment_id: str, user_id: UUID) -> DeploymentDetail:
        """
        `deployment_id` is the run_id string used in
        POST /predict/{deployment_id} — the same identifier the UI
        shows and copies — not the internal `deployments.id` primary
        key.

        Raises ValueError (-> 404) if no such deployment exists, and
        PermissionError (-> 403) if it exists but belongs to a
        different user.
        """
        try:
            run_uuid = UUID(deployment_id)
        except ValueError as exc:
            raise ValueError(f"'{deployment_id}' is not a valid deployment id.") from exc

        deployment = self.deployment_repository.get_by_run_id(run_uuid)
        if deployment is None:
            raise ValueError(f"No deployment found for id '{deployment_id}'.")

        pipeline_run = self.pipeline_run_repository.get_by_id(deployment.run_id)
        if pipeline_run is not None and pipeline_run.user_id != user_id:
            raise PermissionError("You are not allowed to access this deployment.")

        summary = self._build_summary(deployment)
        monitoring = self.monitoring_repository.get_latest_by_deployment_id(deployment.id)

        return DeploymentDetail(**summary.model_dump(), monitoring=monitoring)
