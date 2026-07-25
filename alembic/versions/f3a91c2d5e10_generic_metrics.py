"""generic_metrics

Revision ID: f3a91c2d5e10
Revises: d8a7302bb7da
Create Date: 2026-07-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'f3a91c2d5e10'
down_revision: Union[str, Sequence[str], None] = 'd8a7302bb7da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('pipeline_runs', sa.Column('problem_type', sa.String(), nullable=True))
    op.add_column('pipeline_runs', sa.Column('target_column', sa.String(), nullable=True))

    op.add_column('trained_models', sa.Column('problem_type', sa.String(), nullable=True))
    op.add_column('trained_models', sa.Column('target_column', sa.String(), nullable=True))
    op.add_column(
        'trained_models',
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    # Backfill metrics JSON from the old fixed columns so existing
    # classification rows don't lose their numbers.
    op.execute(
        """
        UPDATE trained_models
        SET metrics = jsonb_strip_nulls(jsonb_build_object(
            'accuracy', accuracy,
            'precision', precision,
            'recall', recall,
            'f1', f1_score
        )),
        problem_type = 'classification'
        WHERE accuracy IS NOT NULL
           OR precision IS NOT NULL
           OR recall IS NOT NULL
           OR f1_score IS NOT NULL
        """
    )

    op.drop_column('trained_models', 'accuracy')
    op.drop_column('trained_models', 'precision')
    op.drop_column('trained_models', 'recall')
    op.drop_column('trained_models', 'f1_score')


def downgrade() -> None:
    op.add_column('trained_models', sa.Column('accuracy', sa.Float(), nullable=True))
    op.add_column('trained_models', sa.Column('precision', sa.Float(), nullable=True))
    op.add_column('trained_models', sa.Column('recall', sa.Float(), nullable=True))
    op.add_column('trained_models', sa.Column('f1_score', sa.Float(), nullable=True))

    op.execute(
        """
        UPDATE trained_models
        SET accuracy = (metrics->>'accuracy')::float,
            precision = (metrics->>'precision')::float,
            recall = (metrics->>'recall')::float,
            f1_score = (metrics->>'f1')::float
        WHERE metrics IS NOT NULL
        """
    )

    op.drop_column('trained_models', 'metrics')
    op.drop_column('trained_models', 'target_column')
    op.drop_column('trained_models', 'problem_type')

    op.drop_column('pipeline_runs', 'target_column')
    op.drop_column('pipeline_runs', 'problem_type')
