"""
Monitoring Service.

DB-facing orchestrator for a single monitoring check: resolve the
deployment, pull its reference DatasetSnapshot and PredictionLog
history, hand them to the pure MonitoringTool, persist the result.

No LLM, no PipelineState, no LangGraph — this runs independently of
the training pipeline, triggered per-deployment whenever a check is
wanted (see server/api/routes/monitoring.py), not as a graph node.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from server.models.monitoring import Monitoring
from server.repositories.deployment_repository import DeploymentRepository
from server.repositories.dataset_snapshot_repository import DatasetSnapshotRepository
from server.repositories.prediction_log_repository import PredictionLogRepository
from server.repositories.monitoring_repository import MonitoringRepository
from tools.monitoring.monitoring_tool import MonitoringTool
from utils.logger import logger


class MonitoringService:

    def __init__(self, db: Session, tool: MonitoringTool | None = None):
        self.db = db
        self.tool = tool or MonitoringTool()
        self.deployment_repository = DeploymentRepository(db)
        self.dataset_snapshot_repository = DatasetSnapshotRepository(db)
        self.prediction_log_repository = PredictionLogRepository(db)
        self.monitoring_repository = MonitoringRepository(db)

    def run_check(self, deployment_id: UUID) -> Monitoring:
        """
        Runs a check against ALL prediction logs recorded for this
        deployment so far, not just those since the last check — each
        call recomputes prediction_count/average_latency/drift_score
        over the deployment's full history. That's a deliberate
        starting point for simplicity, not a hidden default: if you
        want a rolling window instead (e.g. "since last check", or
        "last 24h"), that's a `since=` argument to add to
        prediction_log_repository.get_by_deployment_id() here.

        Raises ValueError if no deployment exists with this id — the
        route layer is expected to translate that into a 404.
        """
        deployment = self.deployment_repository.get_by_id(deployment_id)
        if deployment is None:
            raise ValueError(f"No deployment found with id '{deployment_id}'.")

        snapshot = self.dataset_snapshot_repository.get_by_deployment_id(deployment_id)
        feature_statistics = snapshot.feature_statistics if snapshot else None

        prediction_logs = self.prediction_log_repository.get_by_deployment_id(deployment_id)

        result = self.tool.compute(prediction_logs, feature_statistics)

        for warning in result["warnings"]:
            logger.info("[Monitoring] deployment=%s: %s", deployment_id, warning)

        monitoring_record = Monitoring(
            deployment_id=deployment_id,
            prediction_count=result["prediction_count"],
            average_latency=result["average_latency"],
            drift_score=result["drift_score"],
            accuracy=result["accuracy"],
            alert_status=result["alert_status"],
        )

        return self.monitoring_repository.create(monitoring_record)
