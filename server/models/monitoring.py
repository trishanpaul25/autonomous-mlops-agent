from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Float
from sqlalchemy import String
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class Monitoring(Base):

    __tablename__ = "monitoring"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    deployment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deployments.id"),
    )

    prediction_count = Column(
        Integer,
        default=0,
    )

    average_latency = Column(Float)

    drift_score = Column(Float)

    accuracy = Column(Float)

    alert_status = Column(String)

    last_checked = Column(
        TIMESTAMP,
        server_default=func.now(),
    )