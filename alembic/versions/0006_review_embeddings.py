"""review_embeddings: stored per-review vectors (MiniLM 384) for community clustering

Dedicated table (written via COPY) so we never re-embed old reviews and never
bloat the hot interactions table.

Revision ID: 0006_review_embeddings
Revises: 0005_v4_embeddings_and_logs
Create Date: 2026-06-14
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0006_review_embeddings"
down_revision: Union[str, None] = "0005_v4_embeddings_and_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE review_embeddings ("
        "  interaction_id integer PRIMARY KEY REFERENCES interactions(id) ON DELETE CASCADE,"
        "  movie_id integer NOT NULL REFERENCES movies(id) ON DELETE CASCADE,"
        "  embedding vector(384) NOT NULL)"
    )
    op.execute("CREATE INDEX ix_review_embeddings_movie ON review_embeddings(movie_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS review_embeddings")
