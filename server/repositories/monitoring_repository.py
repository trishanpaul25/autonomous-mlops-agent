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