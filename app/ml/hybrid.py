"""Hybrid recommendation scoring + candidate generation + diversity.

final_score (collab available) = 0.45*collab + 0.35*content + 0.20*popularity
final_score (no collab)       = 0.70*content + 0.30*popularity
Each component is min-max normalised per candidate set. Excluded: movies the
user has reviewed OR has in watched/watchlist. The top pool is diversified with
MMR (and a per-genre cap) before the top-N are persisted to user_recommendations.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml import config
from app.ml.collaborative import CollaborativeModel

log = logging.getLogger("recsys.hybrid")

_CTX_CACHE: dict = {}
_CTX_LOCK = threading.Lock()


def _parse_vec(v) -> np.ndarray:
    if isinstance(v, str):
        return np.fromstring(v.strip("[]"), sep=",", dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


class RecoContext:
    def __init__(self, ids, content_unit, idx, pop, collab, *,
                 review_count=None, avg_pref=None, genres=None, active_version=None,
                 year=None, actors=None):
        self.ids = ids
        self.content_unit = content_unit
        self.idx = idx
        self.pop = pop
        self.review_count = (review_count if review_count is not None
                             else np.zeros(len(ids), dtype=np.float32))
        self.avg_pref = (avg_pref if avg_pref is not None
                         else np.full(len(ids), 5.0, dtype=np.float32))
        self.genres = genres if genres is not None else [set() for _ in ids]
        # used by the XGBoost ranker's overlap/distance features (nightly only)
        self.year = (year if year is not None
                     else np.full(len(ids), np.nan, dtype=np.float32))
        self.actors = actors if actors is not None else [set() for _ in ids]
        self.active_version = active_version
        self.collab: CollaborativeModel | None = collab
        if collab is not None:
            cf_dim = collab.item_factors.shape[1]
            self.collab_item = np.zeros((len(ids), cf_dim), dtype=np.float32)
            self.collab_bias = np.zeros(len(ids), dtype=np.float32)
            self.collab_mask = np.zeros(len(ids), dtype=bool)
            for mid, row in collab.item_map.items():
                j = idx.get(mid)
                if j is not None:
                    self.collab_item[j] = collab.item_factors[row]
                    self.collab_bias[j] = collab.item_bias[row]
                    self.collab_mask[j] = True


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


async def _active_collab_version(db) -> str | None:
    return await db.scalar(
        text("SELECT version_name FROM model_versions "
             "WHERE model_type='collaborative_mf' AND is_active "
             "ORDER BY created_at DESC LIMIT 1")
    )


async def load_active_collab(db: AsyncSession) -> CollaborativeModel | None:
    row = (
        await db.execute(
            text("SELECT version_name, artifact_path FROM model_versions "
                 "WHERE model_type='collaborative_mf' AND is_active "
                 "ORDER BY created_at DESC LIMIT 1")
        )
    ).first()
    if row is None:
        return None
    try:
        return CollaborativeModel.load(row[1])
    except Exception as exc:
        log.warning("Failed to load collaborative model %s: %s", row[1], exc)
        return None


async def build_context(db: AsyncSession, refresh: bool = False) -> RecoContext:
    # version check (item 2): invalidate cache when the active model changes
    count = await db.scalar(text("SELECT count(*) FROM movie_embeddings"))
    active_version = await _active_collab_version(db)
    with _CTX_LOCK:
        cached = _CTX_CACHE.get("ctx")
        if (not refresh and cached is not None
                and _CTX_CACHE.get("count") == count
                and cached.active_version == active_version):
            return cached

    erows = (await db.execute(
        text("SELECT movie_id, embedding FROM movie_embeddings"))).all()
    ids = np.array([r[0] for r in erows], dtype=np.int64)
    mat = np.array([_parse_vec(r[1]) for r in erows], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    content_unit = mat / norms
    idx = {int(m): i for i, m in enumerate(ids)}

    pop = np.zeros(len(ids), dtype=np.float32)
    rc = np.zeros(len(ids), dtype=np.float32)
    ap = np.full(len(ids), 5.0, dtype=np.float32)
    for mid, p, cnt, avgp in (await db.execute(text(
        "SELECT movie_id, popularity_score, review_count, avg_preference_score "
        "FROM movie_popularity_stats"))).all():
        j = idx.get(mid)
        if j is not None:
            pop[j] = p or 0.0
            rc[j] = cnt or 0
            ap[j] = avgp if avgp is not None else 5.0

    genres = [set() for _ in ids]
    for mid, gid in (await db.execute(
            text("SELECT movie_id, genre_id FROM movie_genres"))).all():
        j = idx.get(mid)
        if j is not None:
            genres[j].add(gid)

    collab = await load_active_collab(db)
    ctx = RecoContext(ids, content_unit, idx, pop, collab,
                      review_count=rc, avg_pref=ap, genres=genres,
                      active_version=active_version)
    with _CTX_LOCK:
        _CTX_CACHE.update(count=count, ctx=ctx)
    log.info("Built reco context: %d movies, collab=%s (v=%s)",
             len(ids), "yes" if collab else "no", active_version)
    return ctx


def component_scores(ctx, user_unit, user_id):
    """Return (content_n, collab_n, pop_n) arrays (each (N,) or None)."""
    pop_n = _minmax(ctx.pop)
    content_n = _minmax(ctx.content_unit @ user_unit) if user_unit is not None else None
    collab_n = None
    if ctx.collab is not None and ctx.collab.has_user(user_id):
        uf = ctx.collab.user_factors[ctx.collab.user_map[user_id]]
        raw = ctx.collab_item @ uf + ctx.collab_bias
        collab = np.where(ctx.collab_mask, raw, np.nan)
        finite = collab[np.isfinite(collab)]
        if finite.size:
            collab = np.where(np.isfinite(collab), collab, float(finite.min()))
            collab_n = _minmax(collab)
    return content_n, collab_n, pop_n


def _combine(content_n, collab_n, pop_n):
    if collab_n is not None and content_n is not None:
        return (config.HYBRID_W_COLLAB * collab_n
                + config.HYBRID_W_CONTENT * content_n
                + config.HYBRID_W_POP * pop_n)
    if collab_n is not None:
        s = config.HYBRID_W_COLLAB + config.HYBRID_W_POP
        return (config.HYBRID_W_COLLAB * collab_n + config.HYBRID_W_POP * pop_n) / s
    if content_n is not None:
        return config.HYBRID_FB_CONTENT * content_n + config.HYBRID_FB_POP * pop_n
    return pop_n.copy()


def compute_candidates(ctx, user_unit, user_id, seen_idx, pool=config.CAND_POOL):
    """Top `pool` unseen movies by hybrid final score.
    Returns list of (movie_idx, final, content_n, collab_n, pop_n)."""
    content_n, collab_n, pop_n = component_scores(ctx, user_unit, user_id)
    final = _combine(content_n, collab_n, pop_n).astype(np.float32)
    if seen_idx:
        final[seen_idx] = -np.inf
    k = min(pool, len(final))
    top = np.argpartition(final, -k)[-k:]
    top = top[np.argsort(final[top])[::-1]]
    out = []
    for j in top:
        if not np.isfinite(final[j]):
            continue
        out.append((
            int(j), float(final[j]),
            float(content_n[j]) if content_n is not None else None,
            float(collab_n[j]) if collab_n is not None else None,
            float(pop_n[j]),
        ))
    return out


def mmr_rerank(ctx, candidates, top_n, lam=config.DIVERSITY_LAMBDA,
               max_per_genre=config.DIVERSITY_MAX_PER_GENRE):
    """Diversify a score-sorted candidate list with MMR + a per-genre cap.
    candidates: (movie_idx, base_score, ...). Returns reordered subset (len<=top_n)."""
    if not candidates or not config.DIVERSITY_ENABLED:
        return candidates[:top_n]
    rows = [c[0] for c in candidates]
    base = np.array([c[1] for c in candidates], dtype=np.float32)
    lo, hi = base.min(), base.max()
    base_n = (base - lo) / (hi - lo) if hi > lo else np.zeros_like(base)
    emb = ctx.content_unit[rows]
    sim = emb @ emb.T
    n = len(candidates)
    chosen = np.zeros(n, dtype=bool)
    max_sim = np.zeros(n, dtype=np.float32)
    gcount: dict[int, int] = {}
    selected = []
    for _ in range(min(top_n, n)):
        scores = (1 - lam) * base_n - lam * max_sim
        scores[chosen] = -np.inf
        order = np.argsort(scores)[::-1]
        pick = -1
        fallback = -1
        for ci in order:
            if chosen[ci] or not np.isfinite(scores[ci]):
                continue
            if fallback < 0:
                fallback = ci
            gs = ctx.genres[rows[ci]]
            if gs and all(gcount.get(g, 0) >= max_per_genre for g in gs):
                continue
            pick = ci
            break
        if pick < 0:
            pick = fallback
        if pick < 0:
            break
        chosen[pick] = True
        selected.append(candidates[pick])
        for g in ctx.genres[rows[pick]]:
            gcount[g] = gcount.get(g, 0) + 1
        max_sim = np.maximum(max_sim, sim[pick])
    return selected


def iter_user_candidates(ctx, user_ids, user_unit_map, seen_map,
                         pool=config.CAND_POOL, chunk=2048):
    """Batched candidate generation. Yields (user_id, candidates) where the two
    dominant matmuls (content & collaborative scores) are computed per chunk."""
    n_items = len(ctx.ids)
    dim = ctx.content_unit.shape[1]
    pop_n = _minmax(ctx.pop)
    has_collab_global = ctx.collab is not None
    cf_dim = ctx.collab_item.shape[1] if has_collab_global else 0
    for s in range(0, len(user_ids), chunk):
        batch = user_ids[s:s + chunk]
        uu = np.zeros((len(batch), dim), dtype=np.float32)
        has_content = np.zeros(len(batch), dtype=bool)
        ufb = np.zeros((len(batch), cf_dim), dtype=np.float32)
        has_cf = np.zeros(len(batch), dtype=bool)
        for bi, u in enumerate(batch):
            v = user_unit_map.get(u)
            if v is not None:
                uu[bi] = v
                has_content[bi] = True
            if has_collab_global and ctx.collab.has_user(u):
                ufb[bi] = ctx.collab.user_factors[ctx.collab.user_map[u]]
                has_cf[bi] = True
        content_scores = uu @ ctx.content_unit.T                       # (B, N)
        collab_scores = (ufb @ ctx.collab_item.T + ctx.collab_bias
                         if has_collab_global else None)
        for bi, u in enumerate(batch):
            content_n = _minmax(content_scores[bi]) if has_content[bi] else None
            collab_n = None
            if has_cf[bi]:
                raw = np.where(ctx.collab_mask, collab_scores[bi], np.nan)
                finite = raw[np.isfinite(raw)]
                if finite.size:
                    raw = np.where(np.isfinite(raw), raw, float(finite.min()))
                    collab_n = _minmax(raw)
            final = _combine(content_n, collab_n, pop_n).astype(np.float32)
            seen = seen_map.get(u)
            if seen:
                final[seen] = -np.inf
            k = min(pool, n_items)
            top = np.argpartition(final, -k)[-k:]
            top = top[np.argsort(final[top])[::-1]]
            cands = []
            for j in top:
                if not np.isfinite(final[j]):
                    continue
                cands.append((
                    int(j), float(final[j]),
                    float(content_n[j]) if content_n is not None else None,
                    float(collab_n[j]) if collab_n is not None else None,
                    float(pop_n[j]),
                ))
            yield u, cands


def compute_user_scores(ctx, user_unit, user_id, seen_idx,
                        top_n=config.TOP_N_RECOMMENDATIONS):
    """Real-time path: hybrid candidates -> diversity -> top_n.
    Returns list of (movie_id, final, content, collab, pop)."""
    cands = compute_candidates(ctx, user_unit, user_id, seen_idx)
    diversified = mmr_rerank(ctx, cands, top_n)
    return [(int(ctx.ids[j]), final, c, cf, pp)
            for (j, final, c, cf, pp) in diversified]


async def _user_unit_and_seen(db, user_id, ctx):
    row = (await db.execute(
        text("SELECT embedding FROM user_embeddings WHERE user_id=:u"),
        {"u": user_id})).first()
    user_unit = None
    if row is not None:
        v = _parse_vec(row[0])
        n = np.linalg.norm(v)
        user_unit = v / n if n > 0 else None
    # exclude reviewed AND watched/watchlist (item 6)
    seen = (await db.execute(text(
        "SELECT movie_id FROM interactions WHERE user_id=:u "
        "UNION SELECT movie_id FROM user_movie_states WHERE user_id=:u"),
        {"u": user_id})).scalars().all()
    seen_idx = [ctx.idx[m] for m in seen if m in ctx.idx]
    return user_unit, seen_idx


async def _write_recommendations(db, user_id, scored):
    await db.execute(text("DELETE FROM user_recommendations WHERE user_id=:u"),
                     {"u": user_id})
    for rank, (mid, final, c, cf, pop) in enumerate(scored, start=1):
        await db.execute(text(
            "INSERT INTO user_recommendations "
            "(user_id, movie_id, score, rank, content_score, collaborative_score, "
            " popularity_score) VALUES (:u,:m,:s,:r,:c,:cf,:p)"),
            {"u": user_id, "m": mid, "s": final, "r": rank,
             "c": c, "cf": cf, "p": pop})


async def refresh_user_recommendations(db, user_id, ctx=None):
    """Real-time: recompute + persist one user's top-N (hybrid + diversity)."""
    if ctx is None:
        ctx = await build_context(db)
    user_unit, seen_idx = await _user_unit_and_seen(db, user_id, ctx)
    scored = compute_user_scores(ctx, user_unit, user_id, seen_idx)
    await _write_recommendations(db, user_id, scored)
    await db.commit()
    log.info("Refreshed %d recommendations for user %s", len(scored), user_id)
    return len(scored)
