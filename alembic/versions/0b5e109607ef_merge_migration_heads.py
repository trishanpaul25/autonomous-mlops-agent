"""merge migration heads

Revision ID: 0b5e109607ef
Revises: 769190d5664d, ab4713694b70
Create Date: 2026-07-22 11:38:02.984130

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0b5e109607ef'
down_revision: Union[str, Sequence[str], None] = ('769190d5664d', 'ab4713694b70')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
