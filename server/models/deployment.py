from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey

from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class Deployment(Base):

    __tablename__ = "deployments"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
    )

    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trained_models.id"),
    )

    endpoint = Column(Text)

    status = Column(String)

    deployed_at = Column(TIMESTAMP)