"""Real-time recommendation refresh (v4) — same pipeline as nightly.

On a new/updated review we rebuild that user's top-100 with the full v4 path
(multi-channel candidates -> XGBoost rerank -> MMR), so daytime recommendations
match the nightly ranking objective. The user's content profiles are recomputed
on the fly; the collaborative (LightGCN) user vector is read from the DB (cold
users simply have none until the next nightly run). Runs in a background task.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml import config
from app.ml.profiles import build_user_vectors
from app.ml.ranker import XgbRanker, build_features
from app.ml.reco import build_context, dsn, generate_candidates, mmr_rerank

log = logging.getLogger("recsys.realtime")

_ranker = None
_ranker_version = None
_lock = threading.Lock()


async def _active_ranker(conn):
    global _ranker, _ranker_version
    row = await conn.fetchrow(
        "SELECT version_name, artifact_path FROM model_versions "
        "WHERE model_type='xgb_ranker' AND is_active ORDER BY created_at DESC LIMIT 1")
    if row is None:
        return None
    with _lock:
        if _ranker_version != row["version_name"]:
            try:
                _ranker = XgbRanker.load(row["artifact_path"])
                _ranker_version = row["version_name"]
            except Exception:
                log.warning("Failed to load ranker %s", row["artifact_path"], exc_info=True)
                return None
    return _ranker


async def refresh_user_recs(user_id: int) -> int:
    """Recompute + persist one user's top-100 with the v4 pipeline."""
    ctx = await build_context()
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    try:
        rows = await conn.fetch(
            "SELECT movie_id, preference_score, review_date FROM interactions "
            "WHERE user_id=$1 AND preference_score IS NOT NULL", user_id)
        items = [(r["movie_id"], float(r["preference_score"]), r["review_date"]) for r in rows]
        if not items:
            return 0
        uv = build_user_vectors(ctx, items, ctx.global_mean)
        mf = await conn.fetchval("SELECT embedding FROM user_mf_embeddings WHERE user_id=$1", user_id)
        uv.mf = np.asarray(mf, dtype=np.float32) if mf is not None else None

        seen = await conn.fetch(
            "SELECT movie_id FROM interactions WHERE user_id=$1 "
            "UNION SELECT movie_id FROM user_movie_states WHERE user_id=$1", user_id)
        seen_idx = [ctx.idx[r["movie_id"]] for r in seen if r["movie_id"] in ctx.idx]
        cand = generate_candidates(ctx, uv, seen_idx)
        if len(cand) == 0:
            return 0
        ranker = await _active_ranker(conn)
        if ranker is not None:
            scores = ranker.predict(build_features(ctx, uv, cand))
        else:
            scores = ctx.pop[cand]
        div = mmr_rerank(ctx, cand, np.asarray(scores, np.float32),
                         config.TOP_N_RECOMMENDATIONS)
        async with conn.transaction():
            await conn.execute("DELETE FROM user_recommendations WHERE user_id=$1", user_id)
            await conn.copy_records_to_table(
                "user_recommendations",
                records=[(user_id, int(ctx.ids[j]), float(s), rank, None, None, float(ctx.pop[j]))
                         for rank, (j, s) in enumerate(div, 1)],
                columns=["user_id", "movie_id", "score", "rank",
                         "content_score", "collaborative_score", "popularity_score"])
        return len(div)
    finally:
        await conn.close()


async def _refresh_bg(user_id: int):
    try:
        t0 = time.time()
        k = await refresh_user_recs(user_id)
        log.info("real-time refreshed %d recs for user %s in %.2fs",
                 k, user_id, time.time() - t0)
    except Exception:
        log.exception("real-time refresh failed for user %s", user_id)


async def handle_new_review(db, user_id, movie_id, preference_score, background_tasks):
    """Schedule a background v4 recommendation refresh for the reviewer."""
    if background_tasks is not None:
        background_tasks.add_task(_refresh_bg, user_id)
    else:
        asyncio.create_task(_refresh_bg(user_id))
