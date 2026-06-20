"""Recompute movie_trending_stats from interactions.

trend_score = recent_count / (prior_count + k) * ln(1 + recent_count)

Movies gaining momentum (velocity), not just evergreen-popular ones. The window
is anchored to max(review_date) in the data (the dataset is historical, so
wall-clock now() would match nothing); we compare the recent `window_days`
against the equally-long window before it. The ratio rewards acceleration; the
ln(volume) term keeps a 2-review blip from topping a genuine surge of hundreds.

The score is genre-independent, so a single row per movie serves both the
all-movies and the per-genre trending endpoints (genre filtering is a join on
movie_genres at read time).
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml import config

# The min_recent filter means the qualifying set shrinks/changes between runs, so
# we rebuild wholesale rather than upsert: DELETE + INSERT in one transaction is
# atomic, so concurrent readers see either the old or the new set, never empty.
# Config constants are inlined (trusted ints) so the text carries no bind params
# and runs unchanged under both SQLAlchemy and the asyncpg training job.
_N = config.TRENDING_WINDOW_DAYS
_DELETE_SQL = text("DELETE FROM movie_trending_stats")
_INSERT_SQL = text(
    f"""
    WITH anchor AS (
        SELECT max(review_date) AS t FROM interactions WHERE review_date IS NOT NULL
    ),
    windows AS (
        SELECT i.movie_id,
               count(*) FILTER (
                   WHERE i.review_date > a.t - make_interval(days => {_N})) AS recent_count,
               count(*) FILTER (
                   WHERE i.review_date <= a.t - make_interval(days => {_N})) AS prior_count,
               avg(i.preference_score) FILTER (
                   WHERE i.review_date > a.t - make_interval(days => {_N})) AS recent_avg_pref
        FROM interactions i CROSS JOIN anchor a
        WHERE i.review_date IS NOT NULL
          AND i.review_date > a.t - make_interval(days => 2 * {_N})
        GROUP BY i.movie_id
    )
    INSERT INTO movie_trending_stats
        (movie_id, recent_count, prior_count, recent_avg_preference,
         trend_score, window_days, updated_at)
    SELECT w.movie_id, w.recent_count, w.prior_count, w.recent_avg_pref,
           (w.recent_count::float / (w.prior_count + {config.TRENDING_SMOOTHING_K}))
           * ln(1 + w.recent_count) AS trend_score,
           {_N}, now()
    FROM windows w
    WHERE w.recent_count >= {config.TRENDING_MIN_RECENT}
    """
)


async def recompute_trending(db: AsyncSession) -> int:
    await db.execute(_DELETE_SQL)
    await db.execute(_INSERT_SQL)
    await db.commit()
    return await db.scalar(text("SELECT count(*) FROM movie_trending_stats"))
