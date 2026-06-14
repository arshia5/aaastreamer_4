"""Real-time recommendation updates triggered by a new/updated review.

Fast path (synchronous): recompute the user's content embedding and nudge the
collaborative model via fit_partial. Slow path (background task): rebuild the
user's top-N hybrid recommendations. Full training stays nightly/manual.
"""
from __future__ import annotations

import logging
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.ml import config
from app.ml.hybrid import build_context, refresh_user_recommendations
from app.ml.user_embedding import compute_user_vector
from app.models import Interaction, MovieEmbedding, UserEmbedding

log = logging.getLogger("recsys.realtime")


async def recompute_user_embedding(db: AsyncSession, user_id: int) -> bool:
    rows = (
        await db.execute(
            select(
                Interaction.movie_id,
                Interaction.preference_score,
                Interaction.review_date,
            ).where(
                Interaction.user_id == user_id,
                Interaction.preference_score.is_not(None),
            )
        )
    ).all()
    if not rows:
        return False
    movie_ids = [r[0] for r in rows]
    embs = (
        await db.execute(
            select(MovieEmbedding.movie_id, MovieEmbedding.embedding).where(
                MovieEmbedding.movie_id.in_(movie_ids)
            )
        )
    ).all()
    lookup = {mid: vec for mid, vec in embs}
    vec = compute_user_vector([(r[0], r[1], r[2]) for r in rows], lookup)
    if vec is None:
        return False
    emb = await db.get(UserEmbedding, user_id)
    if emb is None:
        db.add(UserEmbedding(user_id=user_id, embedding=vec.tolist()))
    else:
        emb.embedding = vec.tolist()
    await db.commit()
    return True


async def _update_recs_bg(
    user_id: int, movie_id: int, preference_score: float | None
) -> None:
    """Background: collaborative fit_partial + rebuild this user's top-N recs.

    Building the reco context (movie matrix + model) is cached, so only the very
    first call after startup pays the load cost — never the API request itself.
    """
    try:
        async with AsyncSessionLocal() as db:
            ctx = await build_context(db)
            try:
                if (
                    ctx.collab is not None
                    and preference_score is not None
                    and preference_score >= config.CF_POS_THRESHOLD
                    and ctx.collab.has_user(user_id)
                    and ctx.collab.has_item(movie_id)
                ):
                    ok = ctx.collab.partial_fit_user(
                        user_id, [(movie_id, preference_score / 10.0)]
                    )
                    log.info("user %s collab fit_partial=%s", user_id, ok)
            except Exception:
                log.warning("collab fit_partial skipped for %s", user_id, exc_info=True)
            await refresh_user_recommendations(db, user_id, ctx)
    except Exception:
        log.exception("Background rec update failed for user %s", user_id)


async def handle_new_review(
    db: AsyncSession,
    user_id: int,
    movie_id: int,
    preference_score: float | None,
    background_tasks,
) -> None:
    """Fast synchronous user-embedding update; defer the heavy collaborative
    fit_partial + recommendation refresh to a background task."""
    t0 = time.time()
    try:
        ok = await recompute_user_embedding(db, user_id)
        log.info("user %s embedding updated=%s in %.3fs",
                 user_id, ok, time.time() - t0)
    except Exception:
        log.exception("user %s embedding update failed", user_id)

    if background_tasks is not None:
        background_tasks.add_task(
            _update_recs_bg, user_id, movie_id, preference_score
        )
