from uuid import uuid4

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
        nullable=False,
        default=uuid4,
    )

    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trained_models.id"),
    )

    # The pipeline run_id — what ModelServerRegistry keys models by and
    # what actually appears in the live /predict/{deployment_id} URL
    # (see agents/deployment_agent.py, which sets deployment_id =
    # state.run_id). This is NOT the same value as `id` above. Without
    # this column, there was no way to go from a live request back to
    # this row — predict.py only ever knows run_id, never `id`.
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pipeline_runs.id"),
    )

    endpoint = Column(Text)

    status = Column(String)

    deployed_at = Column(TIMESTAMP)