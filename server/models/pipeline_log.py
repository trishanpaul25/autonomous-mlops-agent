from sqlalchemy import Column
from sqlalchemy import Text
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey

from sqlalchemy.sql import func

from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4
from server.db.database import Base


class PipelineLog(Base):

    __tablename__ = "pipeline_logs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_runs.id"),
        nullable=False,
    )

    log_message = Column(Text, nullable=False,)

    created_at = Column(TIMESTAMP, server_default= func.now(),)