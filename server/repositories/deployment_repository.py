from uuid import UUID

from sqlalchemy.orm import Session

from server.models.deployment import Deployment


class DeploymentRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, deployment: Deployment) -> Deployment:
        """
        Save a deployment.
        """
        self.db.add(deployment)
        self.db.commit()
        self.db.refresh(deployment)

        return deployment

    def get_by_id(self, deployment_id: UUID) -> Deployment | None:
        """
        Fetch deployment by ID.
        """
        return (
            self.db.query(Deployment)
            .filter(Deployment.id == deployment_id)
            .first()
        )

    def get_by_run_id(self, run_id: UUID) -> Deployment | None:
        """
        Fetch deployment by pipeline run_id — this is the value that
        actually appears in live /predict/{deployment_id} requests, so
        this is how request-time code (predict.py) resolves the real
        `deployments.id` row to log or monitor against.
        """
        return (
            self.db.query(Deployment)
            .filter(Deployment.run_id == run_id)
            .first()
        )

    def update(self) -> None:
        self.db.commit()