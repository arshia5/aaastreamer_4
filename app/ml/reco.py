"""v4 recommendation context + multi-channel candidate generation.

Loads every component vector space (metadata, plot, mf, community) from the DB
into memory, generates a candidate pool by unioning per-channel nearest-neighbour
buckets, and exposes MMR diversity. Ranking features live in ranker.py; the
training orchestration in app/jobs/training.py.
"""
from __future__ import annotations

import logging
import threading

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml import config

log = logging.getLogger("recsys.reco")
_CACHE: dict = {}
_LOCK = threading.Lock()


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


def _l2(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


class RecoContext:
    def __init__(self):
        self.ids = None
        self.idx = {}
        self.meta = None          # (N,768) unit
        self.struct = None        # (N,348) unit — v5 structured-metadata channel
        self.struct_mask = None
        self.plot = None          # (N,768) unit
        self.mf_item = None       # (N,64)
        self.mf_mask = None       # (N,)
        self.comm_mean = None     # (N,384) unit (weighted mean centroid)
        self.comm_mask = None
        self.comm_centroids = {}  # movie_idx -> (C,384) unit centroids
        self.comm_flat = None     # (M_total,384) all centroids stacked
        self.comm_off = None      # (N+1,) segment offsets: movie j -> flat[off[j]:off[j+1]]
        self.pop = None
        self.review_count = None
        self.avg_pref = None      # Bayesian-shrunk
        self.genres = None        # list[set]
        self.year = None
        self.actors = None        # list[set]
        self.global_mean = 5.5
        self.active_version = None


async def _load_vec_table(conn, table, id_col, n, idx, dim):
    mat = np.zeros((n, dim), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for r in await conn.fetch(f"SELECT {id_col}, embedding FROM {table}"):
        j = idx.get(r[id_col])
        if j is not None:
            mat[j] = r["embedding"]
            mask[j] = True
    return mat, mask


async def build_context_from_conn(conn, mf_override=None) -> RecoContext:
    """mf_override: optional (item_mf (N,64), mask) to use the freshly trained
    model instead of the DB copy (used during a training run)."""
    ctx = RecoContext()
    erows = await conn.fetch("SELECT movie_id, embedding FROM movie_metadata_embeddings ORDER BY movie_id")
    ctx.ids = np.array([r["movie_id"] for r in erows], dtype=np.int64)
    n = len(ctx.ids)
    ctx.idx = {int(m): i for i, m in enumerate(ctx.ids)}
    ctx.meta = _l2(np.array([r["embedding"] for r in erows], dtype=np.float32))

    # v5 structured-metadata channel (in-memory, from the fitted v3 pipeline)
    try:
        from app.ml.struct_embed import build_struct_matrix
        ctx.struct, ctx.struct_mask = await build_struct_matrix(conn, ctx.ids)
    except Exception:
        log.exception("structured-metadata channel unavailable; continuing without it")
        ctx.struct, ctx.struct_mask = None, None

    plot = np.zeros((n, config.TEXT_EMBED_DIM), dtype=np.float32)
    for r in await conn.fetch("SELECT movie_id, embedding FROM movie_plot_embeddings"):
        j = ctx.idx.get(r["movie_id"])
        if j is not None:
            plot[j] = r["embedding"]
    ctx.plot = _l2(plot)

    if mf_override is not None:
        ctx.mf_item, ctx.mf_mask = mf_override
    else:
        ctx.mf_item, ctx.mf_mask = await _load_vec_table(
            conn, "movie_mf_embeddings", "movie_id", n, ctx.idx, config.LIGHTGCN_DIM)

    # community: weighted mean centroid + per-movie centroid set
    comm_mean = np.zeros((n, config.REVIEW_EMBED_DIM), dtype=np.float32)
    comm_mask = np.zeros(n, dtype=bool)
    cents: dict[int, list] = {}
    for r in await conn.fetch(
        "SELECT movie_id, weight, embedding FROM movie_community_embeddings"):
        j = ctx.idx.get(r["movie_id"])
        if j is None:
            continue
        v = np.asarray(r["embedding"], dtype=np.float32)
        comm_mean[j] += r["weight"] * v
        cents.setdefault(j, []).append(v)
        comm_mask[j] = True
    ctx.comm_mean = _l2(comm_mean)
    ctx.comm_mask = comm_mask
    ctx.comm_centroids = {j: _l2(np.array(c, dtype=np.float32)) for j, c in cents.items()}
    # flat centroid array + per-movie segment offsets (vectorised max-cos)
    flat, off = [], np.zeros(n + 1, dtype=np.int64)
    for j in range(n):
        cs = ctx.comm_centroids.get(j)
        if cs is not None:
            flat.append(cs)
        off[j + 1] = off[j] + (len(cs) if cs is not None else 0)
    ctx.comm_flat = (np.vstack(flat).astype(np.float32) if flat
                     else np.zeros((0, config.REVIEW_EMBED_DIM), np.float32))
    ctx.comm_off = off

    # popularity stats
    ctx.pop = np.zeros(n, dtype=np.float32)
    ctx.review_count = np.zeros(n, dtype=np.float32)
    ctx.avg_pref = np.full(n, 5.5, dtype=np.float32)
    for mid, p, cnt, ap in await conn.fetch(
        "SELECT movie_id, popularity_score, review_count, avg_preference_score "
        "FROM movie_popularity_stats"):
        j = ctx.idx.get(mid)
        if j is not None:
            ctx.pop[j] = p or 0.0
            ctx.review_count[j] = cnt or 0
            ctx.avg_pref[j] = ap if ap is not None else 5.5

    ctx.genres = [set() for _ in range(n)]
    for mid, gid in await conn.fetch("SELECT movie_id, genre_id FROM movie_genres"):
        j = ctx.idx.get(mid)
        if j is not None:
            ctx.genres[j].add(gid)
    ctx.year = np.full(n, np.nan, dtype=np.float32)
    for mid, yr in await conn.fetch("SELECT id, year FROM movies WHERE year IS NOT NULL"):
        j = ctx.idx.get(mid)
        if j is not None:
            ctx.year[j] = yr
    ctx.actors = [set() for _ in range(n)]
    for mid, pid in await conn.fetch(
        "SELECT mp.movie_id, mp.person_id FROM movie_people mp "
        "JOIN roles r ON r.id=mp.role_id WHERE r.name='actor'"):
        j = ctx.idx.get(mid)
        if j is not None:
            ctx.actors[j].add(pid)

    ctx.global_mean = float(await conn.fetchval(
        "SELECT avg(preference_score) FROM interactions") or 5.5)
    ctx.active_version = await conn.fetchval(
        "SELECT version_name FROM model_versions WHERE model_type='lightgcn' "
        "AND is_active ORDER BY created_at DESC LIMIT 1")
    log.info("Built v4 context: %d movies (mf=%d, community=%d)",
             n, int(ctx.mf_mask.sum()), int(ctx.comm_mask.sum()))
    return ctx


async def build_context(refresh: bool = False) -> RecoContext:
    """Cached context for the serving (real-time) path."""
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    try:
        version = await conn.fetchval(
            "SELECT version_name FROM model_versions WHERE model_type='lightgcn' "
            "AND is_active ORDER BY created_at DESC LIMIT 1")
        count = await conn.fetchval("SELECT count(*) FROM movie_metadata_embeddings")
        with _LOCK:
            c = _CACHE.get("ctx")
            if (not refresh and c is not None and _CACHE.get("v") == version
                    and _CACHE.get("n") == count):
                return c
        ctx = await build_context_from_conn(conn)
        with _LOCK:
            _CACHE.update(ctx=ctx, v=version, n=count)
        return ctx
    finally:
        await conn.close()


def _topk(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, len(scores))
    top = np.argpartition(scores, -k)[-k:]
    return top[np.isfinite(scores[top])]


def generate_candidates(ctx: RecoContext, uv, seen_idx) -> np.ndarray:
    """Union of per-channel nearest-neighbour buckets, minus seen movies."""
    buckets = []
    if uv.plot_pos is not None:
        buckets.append(_topk(ctx.plot @ uv.plot_pos, config.RETR_PLOT))
    if uv.meta_pos is not None:
        buckets.append(_topk(ctx.meta @ uv.meta_pos, config.RETR_META))
    if uv.struct_pos is not None and ctx.struct is not None:
        buckets.append(_topk(ctx.struct @ uv.struct_pos, config.RETR_STRUCT, ctx.struct_mask))
    if uv.mf is not None and ctx.mf_mask.any():
        buckets.append(_topk(ctx.mf_item @ uv.mf, config.RETR_MF, ctx.mf_mask))
    if uv.comm_pos is not None and ctx.comm_mask.any():
        buckets.append(_topk(ctx.comm_mean @ uv.comm_pos, config.RETR_COMM, ctx.comm_mask))
    buckets.append(_topk(ctx.pop.copy(), config.RETR_POP))
    if not buckets:
        return np.array([], dtype=np.int64)
    cand = np.unique(np.concatenate(buckets))
    if seen_idx:
        cand = np.setdiff1d(cand, np.array(seen_idx, dtype=np.int64), assume_unique=False)
    return cand


def mmr_rerank(ctx, cand_idx, base_scores, top_n,
               lam=config.DIVERSITY_LAMBDA,
               max_per_genre=config.DIVERSITY_MAX_PER_GENRE):
    """Diversify (idx, score) by MMR over plot similarity + per-genre cap."""
    if len(cand_idx) == 0:
        return []
    order0 = np.argsort(base_scores)[::-1]
    cand_idx = cand_idx[order0]
    base = base_scores[order0]
    lo, hi = base.min(), base.max()
    base_n = (base - lo) / (hi - lo) if hi > lo else np.zeros_like(base)
    emb = ctx.plot[cand_idx]
    sim = emb @ emb.T
    n = len(cand_idx)
    chosen = np.zeros(n, bool)
    max_sim = np.zeros(n, np.float32)
    gcount: dict = {}
    out = []
    for _ in range(min(top_n, n)):
        score = (1 - lam) * base_n - lam * max_sim
        score[chosen] = -np.inf
        order = np.argsort(score)[::-1]
        pick = fallback = -1
        for ci in order:
            if chosen[ci] or not np.isfinite(score[ci]):
                continue
            if fallback < 0:
                fallback = ci
            gs = ctx.genres[cand_idx[ci]]
            if gs and all(gcount.get(g, 0) >= max_per_genre for g in gs):
                continue
            pick = ci
            break
        if pick < 0:
            pick = fallback
        if pick < 0:
            break
        chosen[pick] = True
        out.append((int(cand_idx[pick]), float(base[pick])))
        for g in ctx.genres[cand_idx[pick]]:
            gcount[g] = gcount.get(g, 0) + 1
        max_sim = np.maximum(max_sim, sim[pick])
    return out
