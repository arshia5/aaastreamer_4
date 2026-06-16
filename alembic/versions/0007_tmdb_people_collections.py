"""tmdb enrichment: person biographies + movie collections

Adds lazily-populated TMDB fields:
  * people: tmdb_id, biography, profile_path, birthday, tmdb_checked_at
  * movies: tmdb_id, tmdb_collection_id (FK), tmdb_checked_at
  * new `collections` table (id = TMDB collection id)

`tmdb_checked_at` is a negative-cache marker: once set we never re-query TMDB
for that row, even if the lookup found nothing.

Revision ID: 0007_tmdb_people_collections
Revises: 0006_review_embeddings
Create Date: 2026-06-16
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0007_tmdb_people_collections"
down_revision: Union[str, None] = "0006_review_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE collections ("
        "  id integer PRIMARY KEY,"
        "  name varchar NOT NULL,"
        "  overview text,"
        "  poster_path varchar,"
        "  backdrop_path varchar,"
        "  created_at timestamp NOT NULL DEFAULT now(),"
        "  updated_at timestamp NOT NULL DEFAULT now())"
    )

    op.execute("ALTER TABLE people ADD COLUMN tmdb_id integer")
    op.execute("ALTER TABLE people ADD COLUMN biography text")
    op.execute("ALTER TABLE people ADD COLUMN profile_path varchar")
    op.execute("ALTER TABLE people ADD COLUMN birthday varchar")
    op.execute("ALTER TABLE people ADD COLUMN tmdb_checked_at timestamp")

    op.execute("ALTER TABLE movies ADD COLUMN tmdb_id integer")
    op.execute(
        "ALTER TABLE movies ADD COLUMN tmdb_collection_id integer "
        "REFERENCES collections(id) ON DELETE SET NULL"
    )
    op.execute("ALTER TABLE movies ADD COLUMN tmdb_checked_at timestamp")
    op.execute("CREATE INDEX ix_movies_tmdb_id ON movies(tmdb_id)")
    op.execute(
        "CREATE INDEX ix_movies_tmdb_collection_id ON movies(tmdb_collection_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_movies_tmdb_collection_id")
    op.execute("DROP INDEX IF EXISTS ix_movies_tmdb_id")
    op.execute("ALTER TABLE movies DROP COLUMN IF EXISTS tmdb_checked_at")
    op.execute("ALTER TABLE movies DROP COLUMN IF EXISTS tmdb_collection_id")
    op.execute("ALTER TABLE movies DROP COLUMN IF EXISTS tmdb_id")
    op.execute("ALTER TABLE people DROP COLUMN IF EXISTS tmdb_checked_at")
    op.execute("ALTER TABLE people DROP COLUMN IF EXISTS birthday")
    op.execute("ALTER TABLE people DROP COLUMN IF EXISTS profile_path")
    op.execute("ALTER TABLE people DROP COLUMN IF EXISTS biography")
    op.execute("ALTER TABLE people DROP COLUMN IF EXISTS tmdb_id")
    op.execute("DROP TABLE IF EXISTS collections")
