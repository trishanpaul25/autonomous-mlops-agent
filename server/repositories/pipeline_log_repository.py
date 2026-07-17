from uuid import UUID

from sqlalchemy.orm import Session

from server.models.pipeline_log import PipelineLog


class PipelineLogRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        pipeline_log: PipelineLog,
    ) -> PipelineLog:

        self.db.add(pipeline_log)
        self.db.commit()
        self.db.refresh(pipeline_log)

        return pipeline_log

    def create_many(
        self,
        logs: list[PipelineLog],
    ) -> None:

        self.db.add_all(logs)
        self.db.commit()

    def get_by_run_id(
        self,
        run_id: UUID,
    ) -> list[PipelineLog]:

        return (
            self.db.query(PipelineLog)
            .filter(PipelineLog.pipeline_run_id == run_id)
            .all()
        )