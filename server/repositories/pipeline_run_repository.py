from uuid import UUID

from sqlalchemy.orm import Session

from server.models.pipeline_run import PipelineRun


class PipelineRunRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, pipeline_run: PipelineRun) -> PipelineRun:
        """
        Save a pipeline run.
        """
        self.db.add(pipeline_run)
        self.db.commit()
        self.db.refresh(pipeline_run)

        return pipeline_run

    def get_by_id(self, run_id: UUID) -> PipelineRun | None:
        """
        Fetch a pipeline run by ID.
        """
        return (
            self.db.query(PipelineRun)
            .filter(PipelineRun.id == run_id)
            .first()
        )

    def get_all(self) -> list[PipelineRun]:
        """
        Fetch all pipeline runs ordered by newest first.
        """
        return (
            self.db.query(PipelineRun)
            .order_by(PipelineRun.started_at.desc())
            .all()
        )

    def update(
        self,
        pipeline_run: PipelineRun,
    ) -> PipelineRun:
        """
        Update an existing pipeline run.
        """
        self.db.add(pipeline_run)
        self.db.commit()
        self.db.refresh(pipeline_run)

        return pipeline_run