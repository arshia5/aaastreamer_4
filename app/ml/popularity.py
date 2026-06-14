"""Recompute movie_popularity_stats from interactions.

popularity_score = log1p(review_count)/log1p(max_count) * (0.5 + 0.5*quality)
where quality = avg_preference_score/10. In [0,1]; blends reach with quality so
popular-but-disliked movies are down-weighted.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml import config

# popularity_score = log-damped reach * Bayesian-shrunk quality, both in [0,1].
# Bayesian average = (PRIOR*global_mean + sum) / (PRIOR + n) shrinks low-count
# movies toward the overall mean so "10 reviews @ 10.0" can't beat "3000 @ 9.2".
# avg_preference_score stores the raw mean (an honest stat); the shrinkage is
# applied only inside popularity_score.
_RECOMPUTE_SQL = text(
    f"""
    WITH gm AS (SELECT avg(preference_score) AS m FROM interactions)
    INSERT INTO movie_popularity_stats
        (movie_id, review_count, avg_rating, avg_preference_score,
         popularity_score, updated_at)
    SELECT
        i.movie_id,
        count(*) AS review_count,
        avg(i.rating) AS avg_rating,
        avg(i.preference_score) AS avg_preference_score,
        (ln(1 + count(*))
         / nullif((SELECT ln(1 + max(c))
                   FROM (SELECT count(*) c FROM interactions GROUP BY movie_id) t), 0))
        * (0.5 + 0.5 * (
            ({config.POPULARITY_PRIOR} * (SELECT m FROM gm)
             + coalesce(sum(i.preference_score), 0))
            / ({config.POPULARITY_PRIOR} + count(i.preference_score))
          ) / 10.0) AS popularity_score,
        now()
    FROM interactions i
    GROUP BY i.movie_id
    ON CONFLICT (movie_id) DO UPDATE SET
        review_count = EXCLUDED.review_count,
        avg_rating = EXCLUDED.avg_rating,
        avg_preference_score = EXCLUDED.avg_preference_score,
        popularity_score = EXCLUDED.popularity_score,
        updated_at = now()
    """
)


async def recompute_popularity(db: AsyncSession) -> int:
    await db.execute(_RECOMPUTE_SQL)
    await db.commit()
    return await db.scalar(text("SELECT count(*) FROM movie_popularity_stats"))
