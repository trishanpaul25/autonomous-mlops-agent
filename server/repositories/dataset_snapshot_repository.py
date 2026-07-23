from uuid import UUID

from sqlalchemy.orm import Session

from server.models.dataset_snapshot import DatasetSnapshot


class DatasetSnapshotRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, snapshot: DatasetSnapshot) -> DatasetSnapshot:
        """
        Save a dataset snapshot.
        """
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)

        return snapshot

    def get_by_deployment_id(self, deployment_id: UUID) -> DatasetSnapshot | None:
        """
        Fetch the reference snapshot for a deployment. One snapshot per
        deployment is expected — this is what the Monitoring agent's
        drift calculation reads against.
        """
        return (
            self.db.query(DatasetSnapshot)
            .filter(DatasetSnapshot.deployment_id == deployment_id)
            .first()
        )
