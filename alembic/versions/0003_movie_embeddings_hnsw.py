"""HNSW index on movie_embeddings.embedding (cosine)

Speeds up nearest-neighbour queries (similar movies, recommendations, search).

Revision ID: 0003_movie_embeddings_hnsw
Revises: 0002_drop_movies_released
Create Date: 2026-06-13
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003_movie_embeddings_hnsw"
down_revision: Union[str, None] = "0002_drop_movies_released"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_movie_embeddings_hnsw "
        "ON movie_embeddings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_movie_embeddings_hnsw")
