"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-13

Creates the pgvector extension, then builds every table directly from the
SQLAlchemy model metadata so the schema always matches app/models.py.
"""
from typing import Sequence, Union

from alembic import op

from app.core.database import Base
import app.models  # noqa: F401  (registers all tables on Base.metadata)

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector must exist before any `vector` columns are created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
    op.execute("DROP TYPE IF EXISTS user_movie_state_type")
