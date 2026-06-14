"""Offline evaluation of the embedding recommender.

Leave-out protocol (no leakage): for a sample of users, hold out their most
recent reviews, rebuild the user vector from only the *earlier* reviews, then:

- Ranking accuracy: rank all movies by similarity to that train-only vector and
  check whether the held-out *liked* movies (preference >= threshold) surface in
  the top-K -> Precision/Recall/HitRate/NDCG/MAP@K.
- Regression loss: predict each held-out movie's preference via similarity-
  weighted kNN over the user's train movies -> RMSE/MAE on the 0..10 scale.

A popularity baseline (recommend the globally most-reviewed movies) is reported
for context.
"""
from __future__ import annotations

import math
import random
import threading

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.user_embedding import compute_user_vector
from app.models import Interaction, MovieEmbedding

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


async def _get_movie_matrix(db: AsyncSession, refresh: bool):
    count = await db.scalar(select(func.count()).select_from(MovieEmbedding))
    with _CACHE_LOCK:
        if not refresh and _CACHE.get("count") == count and "mat" in _CACHE:
            return _CACHE["ids"], _CACHE["mat"], _CACHE["unit"], _CACHE["idx"]
    rows = (
        await db.execute(select(MovieEmbedding.movie_id, MovieEmbedding.embedding))
    ).all()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.array([np.asarray(r[1], dtype=np.float32) for r in rows], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    idx = {int(m): i for i, m in enumerate(ids)}
    with _CACHE_LOCK:
        _CACHE.update(count=count, ids=ids, mat=mat, unit=unit, idx=idx)
    return ids, mat, unit, idx


async def fetch_eval_data(
    db: AsyncSession, sample_users: int, min_interactions: int, seed: int, refresh: bool
) -> dict:
    ids, mat, unit, idx = await _get_movie_matrix(db, refresh)

    # candidate users with enough scored interactions
    cand = (
        await db.execute(
            select(Interaction.user_id)
            .where(Interaction.preference_score.is_not(None))
            .group_by(Interaction.user_id)
            .having(func.count() >= min_interactions)
        )
    ).scalars().all()
    rng = random.Random(seed)
    cand = list(cand)
    rng.shuffle(cand)
    sample = cand[:sample_users]

    user_items: dict[int, list] = {}
    if sample:
        rows = (
            await db.execute(
                select(
                    Interaction.user_id,
                    Interaction.movie_id,
                    Interaction.preference_score,
                    Interaction.review_date,
                )
                .where(Interaction.user_id.in_(sample))
                .order_by(Interaction.user_id, Interaction.review_date)
            )
        ).all()
        for uid, mid, pref, date in rows:
            if mid in idx and pref is not None:
                user_items.setdefault(uid, []).append((mid, float(pref), date))

    # popularity baseline: most-reviewed movies
    pop = (
        await db.execute(
            select(Interaction.movie_id, func.count().label("c"))
            .group_by(Interaction.movie_id)
            .order_by(func.count().desc())
            .limit(200)
        )
    ).all()
    pop_ids = [m for m, _ in pop if m in idx]

    return {
        "ids": ids, "mat": mat, "unit": unit, "idx": idx,
        "user_items": user_items, "candidates": len(cand), "pop_ids": pop_ids,
    }


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg(rec_relevant: list[int], n_relevant: int, k: int) -> float:
    idcg = _dcg([1] * min(n_relevant, k))
    return _dcg(rec_relevant) / idcg if idcg > 0 else 0.0


def _ap(rec_relevant: list[int], n_relevant: int, k: int) -> float:
    hits, score = 0, 0.0
    for i, rel in enumerate(rec_relevant):
        if rel:
            hits += 1
            score += hits / (i + 1)
    denom = min(n_relevant, k)
    return score / denom if denom > 0 else 0.0


def compute_metrics(
    data: dict, k: int, holdout: float, like_threshold: float
) -> dict:
    ids, mat, unit, idx = data["ids"], data["mat"], data["unit"], data["idx"]
    pop_ids = data["pop_ids"]

    n_users_rank = 0
    p = r = hr = ndcg = mapk = 0.0
    bp = br = bhr = 0.0
    sq = ab = 0.0
    n_pred = 0

    for uid, items in data["user_items"].items():
        # items already sorted by review_date asc (None dates first)
        n = len(items)
        if n < 2:
            continue
        n_test = max(1, round(n * holdout))
        n_test = min(n_test, n - 1)          # keep >=1 train
        train, test = items[: n - n_test], items[n - n_test:]
        train_idx = [idx[m] for m, _, _ in train]
        train_pref = np.array([pr for _, pr, _ in train], dtype=np.float32)

        # --- regression: predict held-out preference via kNN over train ---
        tmat = unit[train_idx]                      # (T, D) normalised
        for m, actual, _ in test:
            sims = tmat @ unit[idx[m]]               # cosine sims to train movies
            pos = sims > 0
            if pos.any():
                pred = float((sims[pos] * train_pref[pos]).sum() / sims[pos].sum())
            else:
                pred = float(train_pref.mean())
            sq += (pred - actual) ** 2
            ab += abs(pred - actual)
            n_pred += 1

        # --- ranking: relevant = liked held-out movies ---
        relevant = {m for m, pr, _ in test if pr >= like_threshold}
        if not relevant:
            continue
        uvec = compute_user_vector(train, {m: mat[idx[m]] for m, _, _ in train})
        if uvec is None:
            continue
        nrm = np.linalg.norm(uvec)
        if nrm == 0:
            continue
        scores = unit @ (uvec / nrm)
        scores[train_idx] = -np.inf
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        rec_ids = [int(ids[i]) for i in top]
        rels = [1 if mid in relevant else 0 for mid in rec_ids]
        hits = sum(rels)

        n_users_rank += 1
        p += hits / k
        r += hits / len(relevant)
        hr += 1.0 if hits else 0.0
        ndcg += _ndcg(rels, len(relevant), k)
        mapk += _ap(rels, len(relevant), k)

        # popularity baseline for the same user
        train_set = {m for m, _, _ in train}
        brec = [m for m in pop_ids if m not in train_set][:k]
        bh = sum(1 for m in brec if m in relevant)
        bp += bh / k
        br += bh / len(relevant)
        bhr += 1.0 if bh else 0.0

    def avg(x):
        return x / n_users_rank if n_users_rank else 0.0

    return {
        "users_evaluated_ranking": n_users_rank,
        "candidates": data["candidates"],
        "model": {
            "precision_at_k": avg(p), "recall_at_k": avg(r),
            "hit_rate_at_k": avg(hr), "ndcg_at_k": avg(ndcg), "map_at_k": avg(mapk),
        },
        "popularity_baseline": {
            "precision_at_k": avg(bp), "recall_at_k": avg(br),
            "hit_rate_at_k": avg(bhr), "ndcg_at_k": 0.0, "map_at_k": 0.0,
        },
        "rating_prediction": {
            "rmse": math.sqrt(sq / n_pred) if n_pred else 0.0,
            "mae": ab / n_pred if n_pred else 0.0,
            "predictions": n_pred,
        },
    }
