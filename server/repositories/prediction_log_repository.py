from uuid import UUID
from datetime import datetime

from sqlalchemy.orm import Session

from server.models.prediction_log import PredictionLog


class PredictionLogRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, prediction_log: PredictionLog) -> PredictionLog:
        """
        Save a prediction log row.
        """
        self.db.add(prediction_log)
        self.db.commit()
        self.db.refresh(prediction_log)

        return prediction_log

    def get_by_deployment_id(
        self,
        deployment_id: UUID,
        since: datetime | None = None,
    ) -> list[PredictionLog]:
        """
        Fetch prediction logs for a deployment, optionally only those
        created after `since`. This is what the Monitoring agent's
        tool will query to compute prediction_count, average_latency,
        and drift_score for a monitoring window.
        """
        query = self.db.query(PredictionLog).filter(
            PredictionLog.deployment_id == deployment_id
        )

        if since is not None:
            query = query.filter(PredictionLog.created_at >= since)

        return query.all()
