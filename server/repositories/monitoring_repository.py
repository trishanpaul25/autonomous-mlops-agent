from uuid import UUID

from sqlalchemy.orm import Session

from server.models.monitoring import Monitoring


class MonitoringRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        monitoring: Monitoring,
    ) -> Monitoring:

        self.db.add(monitoring)
        self.db.commit()
        self.db.refresh(monitoring)

        return monitoring

    def get_latest_by_deployment_id(self, deployment_id: UUID) -> Monitoring | None:
        """
        Most recent check for a deployment. Each check writes a new
        row (mirrors PredictionLog's append-only convention) rather
        than updating in place, so this is a history table — this
        just reads the newest entry.
        """
        return (
            self.db.query(Monitoring)
            .filter(Monitoring.deployment_id == deployment_id)
            .order_by(Monitoring.last_checked.desc())
            .first()
        )