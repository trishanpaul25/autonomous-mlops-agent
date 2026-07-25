from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
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

    problem_type = Column(String)

    target_column = Column(String)

    # {"accuracy": .., "f1": ..} for classification,
    # {"r2": .., "mae": ..} for regression — whatever
    # MetricsCalculator produced for this model.
    metrics = Column(JSONB)

    created_at = Column(TIMESTAMP, server_default=func.now(),)