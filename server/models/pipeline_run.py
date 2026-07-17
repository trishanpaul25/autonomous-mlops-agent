from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import Float
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey

from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class PipelineRun(Base):

    __tablename__ = "pipeline_runs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
    )

    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id"),
    )

    user_prompt = Column(Text)

    assistant_message = Column(Text)

    status = Column(String)

    started_at = Column(TIMESTAMP)

    completed_at = Column(TIMESTAMP)

    execution_time = Column(Float)