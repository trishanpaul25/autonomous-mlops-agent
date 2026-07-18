from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey
from sqlalchemy.sql  import func
from uuid import uuid4
from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class TrainedModel(Base):

    __tablename__ = "trained_models"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        default=uuid4,
    )

    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_runs.id"),
        nullable=False,
    )

    model_name = Column(String, nullable=False,)

    model_path = Column(Text)

    accuracy = Column(Float)

    precision = Column(Float)

    recall = Column(Float)

    f1_score = Column(Float)

    created_at = Column(TIMESTAMP, server_default=func.now(),)