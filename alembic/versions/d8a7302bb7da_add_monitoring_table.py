"""add_monitoring_table

Revision ID: d8a7302bb7da
Revises: 0b5e109607ef
Create Date: 2026-07-23 00:00:00.000000

The `monitoring` table was referenced by earlier migrations (FK-only
alter statements against it in 0181ba3835ea, 769190d5664d, and
8a3c9396fb08) but no migration ever actually created it — those
tables were assumed to pre-exist from an out-of-band setup. On a
fresh database this leaves `alembic upgrade head` without a
`monitoring` table at all, so the first call to
POST /monitoring/{deployment_id}/check fails with
"relation \"monitoring\" does not exist". This migration creates it
to match server/models/monitoring.py.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd8a7302bb7da'
down_revision: Union[str, Sequence[str], None] = '0b5e109607ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'monitoring',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('deployment_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('prediction_count', sa.Integer(), nullable=True),
        sa.Column('average_latency', sa.Float(), nullable=True),
        sa.Column('drift_score', sa.Float(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('alert_status', sa.String(), nullable=True),
        sa.Column('last_checked', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['deployment_id'], ['deployments.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    # MonitoringRepository.get_latest_by_deployment_id() filters by
    # deployment_id and orders by last_checked on every call — index
    # both, mirroring the ix_prediction_logs_deployment_id pattern.
    op.create_index(
        'ix_monitoring_deployment_id',
        'monitoring',
        ['deployment_id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_monitoring_deployment_id', table_name='monitoring')
    op.drop_table('monitoring')
