"""XGBoost learning-to-rank reranker.

Trained nightly on a chronological split (positives = each user's held-out liked
movies, negatives = sampled unseen candidates), grouped per user with
objective rank:ndcg. At serving time the hybrid produces a candidate pool, this
model reranks it, then diversity is applied. Features are built from the hybrid
component scores plus movie/user statistics.
"""
from __future__ import annotations

import logging
import os
import time

import joblib
import numpy as np

from app.ml import config

log = logging.getLogger("recsys.ranker")

FEATURE_NAMES = [
    "content_n", "collab_n", "pop_n", "content_raw", "collab_raw",
    "review_count_log", "movie_avg_pref", "user_avg_pref",
    "user_n_pos_log", "user_has_collab",
    "genre_overlap", "actor_overlap", "year_distance",
]


def build_user_profile(ctx, liked_idxs):
    """User taste profile from their liked-movie rows: (liked_genres set,
    liked_actors set, preferred_year). Used for ranker overlap features."""
    lg, la, years = set(), set(), []
    for j in liked_idxs:
        lg |= ctx.genres[j]
        la |= ctx.actors[j]
        if not np.isnan(ctx.year[j]):
            years.append(float(ctx.year[j]))
    pref_year = float(np.mean(years)) if years else None
    return frozenset(lg), frozenset(la), pref_year


def build_features(ctx, user_unit, user_id, candidates, user_avg_pref, user_n_pos,
                   profile=None):
    """Feature matrix (C, 13) for a user's candidate list.
    candidates: (movie_idx, final, content_n, collab_n, pop_n).
    profile: (liked_genres, liked_actors, preferred_year) or None."""
    rows = np.array([c[0] for c in candidates])
    content_n = np.array([c[2] if c[2] is not None else 0.0 for c in candidates],
                         dtype=np.float32)
    collab_n = np.array([c[3] if c[3] is not None else 0.0 for c in candidates],
                        dtype=np.float32)
    pop_n = np.array([c[4] for c in candidates], dtype=np.float32)

    if user_unit is not None:
        content_raw = ctx.content_unit[rows] @ user_unit
    else:
        content_raw = np.zeros(len(rows), dtype=np.float32)

    has_collab = ctx.collab is not None and ctx.collab.has_user(user_id)
    if has_collab:
        uf = ctx.collab.user_factors[ctx.collab.user_map[user_id]]
        collab_raw = ctx.collab_item[rows] @ uf + ctx.collab_bias[rows]
        collab_raw = np.where(ctx.collab_mask[rows], collab_raw, 0.0)
    else:
        collab_raw = np.zeros(len(rows), dtype=np.float32)

    rc_log = np.log1p(ctx.review_count[rows])
    avg_pref = ctx.avg_pref[rows]

    liked_genres, liked_actors, pref_year = profile or (frozenset(), frozenset(), None)
    genre_ov = np.array([len(ctx.genres[j] & liked_genres) for j in rows], dtype=np.float32)
    actor_ov = np.array([len(ctx.actors[j] & liked_actors) for j in rows], dtype=np.float32)
    if pref_year is not None:
        year_dist = np.array(
            [abs(ctx.year[j] - pref_year) if not np.isnan(ctx.year[j]) else np.nan
             for j in rows], dtype=np.float32)
    else:
        year_dist = np.full(len(rows), np.nan, dtype=np.float32)  # XGBoost handles NaN

    n = len(rows)
    return np.column_stack([
        content_n, collab_n, pop_n, content_raw.astype(np.float32),
        collab_raw.astype(np.float32), rc_log.astype(np.float32),
        avg_pref.astype(np.float32),
        np.full(n, user_avg_pref, dtype=np.float32),
        np.full(n, np.log1p(user_n_pos), dtype=np.float32),
        np.full(n, 1.0 if has_collab else 0.0, dtype=np.float32),
        genre_ov, actor_ov, year_dist,
    ]).astype(np.float32)


def train_ranker(X: np.ndarray, y: np.ndarray, qid: np.ndarray):
    """Train an XGBRanker. Rows must be grouped (sorted) by qid."""
    from xgboost import XGBRanker

    model = XGBRanker(
        tree_method="hist",
        eval_metric="ndcg@10",
        **config.XGB_PARAMS,
    )
    model.fit(X, y, qid=qid)
    return model


def save_ranker(model, meta: dict) -> tuple[str, str]:
    config.RANKER_DIR.mkdir(parents=True, exist_ok=True)
    version = "xgb_" + time.strftime("%Y%m%d_%H%M%S")
    path = config.RANKER_DIR / f"{version}.joblib"
    tmp = path.with_suffix(".joblib.tmp")
    joblib.dump({"model": model, "features": FEATURE_NAMES, "meta": meta}, tmp)
    os.replace(tmp, path)
    return version, str(path)


class XgbRanker:
    def __init__(self, data: dict, path: str | None = None):
        self.model = data["model"]
        self.features = data.get("features", FEATURE_NAMES)
        self.meta = data.get("meta", {})
        self.path = path

    @classmethod
    def load(cls, path: str) -> "XgbRanker":
        return cls(joblib.load(path), path=path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
