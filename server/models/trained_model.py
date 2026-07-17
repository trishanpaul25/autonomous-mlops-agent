from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey

from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class TrainedModel(Base):

    __tablename__ = "trained_models"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
    )

    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_runs.id"),
    )

    model_name = Column(String)

    model_path = Column(Text)

    accuracy = Column(Float)

    precision = Column(Float)

    recall = Column(Float)

    f1_score = Column(Float)

    created_at = Column(TIMESTAMP)