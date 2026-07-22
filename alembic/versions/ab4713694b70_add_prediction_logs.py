"""add_prediction_logs

Revision ID: ab4713694b70
Revises: 80cc8f6b521d
Create Date: 2026-07-22 00:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ab4713694b70'
down_revision: Union[str, Sequence[str], None] = '80cc8f6b521d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'prediction_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('deployment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('input_payload', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('prediction', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('ground_truth', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['deployment_id'], ['deployments.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        'ix_prediction_logs_deployment_id',
        'prediction_logs',
        ['deployment_id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_prediction_logs_deployment_id', table_name='prediction_logs')
    op.drop_table('prediction_logs')
