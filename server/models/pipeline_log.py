from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Text
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey

from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class PipelineLog(Base):

    __tablename__ = "pipeline_logs"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_runs.id"),
    )

    log_message = Column(Text)

    created_at = Column(TIMESTAMP)