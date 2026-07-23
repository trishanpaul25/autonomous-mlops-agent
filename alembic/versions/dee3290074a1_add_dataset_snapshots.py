"""add_dataset_snapshots

Revision ID: dee3290074a1
Revises: 8a3c9396fb08
Create Date: 2026-07-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'dee3290074a1'
down_revision: Union[str, Sequence[str], None] = '8a3c9396fb08'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'dataset_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('deployment_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('num_rows', sa.Integer(), nullable=True),
        sa.Column('target_column', sa.String(), nullable=True),
        sa.Column('feature_statistics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('captured_at', sa.TIMESTAMP(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['deployment_id'], ['deployments.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('dataset_snapshots')
