from uuid import UUID

from sqlalchemy.orm import Session

from server.models.pipeline_log import PipelineLog
from server.models.pipeline_run import PipelineRun


class PipelineRepository:

    def __init__(self, db: Session):
        self.db = db

    def create_run(
        self,
        pipeline_run: PipelineRun,
    ) -> PipelineRun:

        self.db.add(pipeline_run)

        self.db.commit()

        self.db.refresh(pipeline_run)

        return pipeline_run

    def update_run(
        self,
        pipeline_run: PipelineRun,
    ) -> PipelineRun:

        self.db.commit()

        self.db.refresh(pipeline_run)

        return pipeline_run

    def add_log(
        self,
        log: PipelineLog,
    ) -> PipelineLog:

        self.db.add(log)

        self.db.commit()

        self.db.refresh(log)

        return log

    def get_run(
        self,
        run_id: UUID,
    ) -> PipelineRun | None:

        return (
            self.db.query(PipelineRun)
            .filter(PipelineRun.id == run_id)
            .first()
        )