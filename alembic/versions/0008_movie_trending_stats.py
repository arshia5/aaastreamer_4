"""precomputed trending (velocity) scores per movie

Adds `movie_trending_stats`, refreshed by the nightly training job
(app.ml.trending.recompute_trending). trend_score is genre-independent, so a
single row per movie serves both all-movies and per-genre trending; the stats
endpoint just sorts on trend_score (optionally joining movie_genres to filter).
The window is anchored to max(review_date), not wall-clock now().

Revision ID: 0008_movie_trending_stats
Revises: 0007_tmdb_people_collections
Create Date: 2026-06-20
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0008_movie_trending_stats"
down_revision: Union[str, None] = "0007_tmdb_people_collections"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE movie_trending_stats ("
        "  movie_id integer PRIMARY KEY REFERENCES movies(id) ON DELETE CASCADE,"
        "  recent_count integer NOT NULL DEFAULT 0,"
        "  prior_count integer NOT NULL DEFAULT 0,"
        "  recent_avg_preference double precision,"
        "  trend_score double precision NOT NULL DEFAULT 0,"
        "  window_days integer NOT NULL,"
        "  updated_at timestamp NOT NULL DEFAULT now())"
    )
    # Index supports ORDER BY trend_score DESC for the trending endpoint.
    op.execute(
        "CREATE INDEX ix_movie_trending_stats_score "
        "ON movie_trending_stats(trend_score DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_movie_trending_stats_score")
    op.execute("DROP TABLE IF EXISTS movie_trending_stats")
