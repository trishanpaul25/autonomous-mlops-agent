from uuid import UUID

from sqlalchemy.orm import Session

from server.models.deployment import Deployment
from server.models.pipeline_run import PipelineRun


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
    def get_latest_completed(self) -> Deployment | None:
        """
        Fetch the single most recently deployed row with status
        "completed" — used on server startup to decide which one
        model (of potentially many past pipeline runs) should be
        reloaded into the in-process ModelServerRegistry. "One live
        model at a time" demo model: whichever pipeline finished
        deployment most recently is the one judges/users should hit.
        """
        return (
            self.db.query(Deployment)
            .filter(Deployment.status == "completed")
            .order_by(Deployment.deployed_at.desc())
            .first()
        )
<<<<<<< HEAD

    def get_by_user_id(self, user_id: UUID) -> list[Deployment]:
        """
        Fetch all deployments belonging to a user, newest first.

        Deployment has no direct user_id column, so this joins through
        pipeline_runs (which does) — powers the frontend's Deployments
        page listing every model a user has ever deployed, not just
        the single latest one that server/main.py preloads on startup.
        """
        return (
            self.db.query(Deployment)
            .join(PipelineRun, Deployment.run_id == PipelineRun.id)
            .filter(PipelineRun.user_id == user_id)
            .order_by(Deployment.deployed_at.desc())
            .all()
        )
=======
>>>>>>> 55c7b9dbe5a6468b75384fcd03a1a927ab39da77

    def update(self) -> None:
        self.db.commit()