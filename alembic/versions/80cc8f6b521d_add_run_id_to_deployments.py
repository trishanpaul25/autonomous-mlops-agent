"""add_run_id_to_deployments

Revision ID: 80cc8f6b521d
Revises: dee3290074a1
Create Date: 2026-07-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '80cc8f6b521d'
down_revision: Union[str, Sequence[str], None] = 'dee3290074a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'deployments',
        sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        'fk_deployments_run_id_pipeline_runs',
        'deployments',
        'pipeline_runs',
        ['run_id'],
        ['id'],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        'fk_deployments_run_id_pipeline_runs',
        'deployments',
        type_='foreignkey',
    )
    op.drop_column('deployments', 'run_id')
