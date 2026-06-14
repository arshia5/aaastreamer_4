"""drop movies.released

Revision ID: 0002_drop_movies_released
Revises: 0001_initial
Create Date: 2026-06-13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_drop_movies_released"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("movies", "released")


def downgrade() -> None:
    op.add_column("movies", sa.Column("released", sa.Date(), nullable=True))
