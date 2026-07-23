from uuid import uuid4

from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.postgresql import JSONB

from server.db.database import Base


class DatasetSnapshot(Base):
    """
    Fixed reference distribution of the training data used by the model
    behind a deployment, captured once at deployment time.

    The Monitoring agent compares live prediction inputs (see
    PredictionLog) against `feature_statistics` here to compute a drift
    score. This is intentionally decoupled from `DatasetProfile`
    (tools/model_selection/dataset_profiler.py), which is an in-memory
    dataclass that disappears with the PipelineState once a run
    finishes — this table is the durable equivalent, scoped narrowly to
    what drift detection needs rather than the full profiling picture.
    """

    __tablename__ = "dataset_snapshots"

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
    )

    num_rows = Column(Integer)

    target_column = Column(String)

    # Per-column reference stats, keyed by column name. Numerical
    # columns store {"type": "numerical", "mean", "std", "min", "max"};
    # categorical columns store {"type": "categorical", "frequencies":
    # {value: proportion, ...}}. Shape documented in
    # tools/monitoring/dataset_snapshot_builder.py.
    feature_statistics = Column(JSONB)

    captured_at = Column(
        TIMESTAMP,
        server_default=func.now(),
    )
