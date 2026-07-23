from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy import Float
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.postgresql import JSONB

from server.db.database import Base


class PredictionLog(Base):
    """
    One row per live inference call against a deployed model
    (server/api/routes/predict.py). This is the raw data the
    Monitoring agent aggregates from: prediction_count and
    average_latency come straight off these rows, and drift_score
    comes from comparing `input_payload` distributions here against
    DatasetSnapshot.feature_statistics for the same deployment_id.

    `ground_truth` is nullable and unpopulated for now — there's no
    feedback/labeling endpoint yet to fill it in after the fact, so
    accuracy monitoring is a later addition once that exists.
    """

    __tablename__ = "prediction_logs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
    )

    deployment_id = Column(
        UUID(as_uuid=True),
        ForeignKey("deployments.id"),
        nullable=False,
        index=True,
    )

    # Raw request/response, one JSON array of records each (matches
    # PredictRequest.records / PredictResponse.predictions shape).
    input_payload = Column(JSONB)
    prediction = Column(JSONB)

    latency_ms = Column(Float)

    # Filled in later, if ever — no write path for this exists yet.
    ground_truth = Column(JSONB, nullable=True)

    created_at = Column(
        TIMESTAMP,
        server_default=func.now(),
    )
