from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class ModelRegistry(Base):

    __tablename__ = "model_registry"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_runs.id"),
        nullable=False,
    )

    model_name = Column(
        String,
        nullable=False,
    )

    model_path = Column(
        Text,
        nullable=False,
    )

    version = Column(
        Integer,
        nullable=False,
        default=1,
    )

    status = Column(
        String,
        nullable=False,
        default="REGISTERED",
    )

    mlflow_run_id = Column(Text)

    registered_at = Column(
        TIMESTAMP,
        server_default=func.now(),
    )