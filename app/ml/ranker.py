"""XGBoost learning-to-rank reranker (v4).

Features combine per-channel similarity to the user's positive/negative taste
profiles (metadata, plot, community), the collaborative score + its per-user
percentile, Bayesian popularity (raw + percentile), and structured cross-features
(favourite/recent-genre match, genre/actor overlap, year distance). Trained on
graded relevance labels (0–3) with objective rank:ndcg.
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
    "meta_pos", "meta_neg", "meta_gap",
    "plot_pos", "plot_neg", "plot_gap",
    "comm_pos", "comm_neg", "comm_gap",
    "mf", "mf_pct",
    "pop", "pop_pct", "review_count_log", "movie_avg_pref",
    "genre_overlap", "fav_genre_match", "recent_genre_match",
    "actor_overlap", "year_distance",
    "user_avg_pref", "user_n_pos_log", "has_mf",
]


def graded_label(pref: float | None) -> int:
    """0–3 relevance grade for rank:ndcg (critical hit / good / neutral / none)."""
    if pref is None:
        return 0
    if pref >= 9.0:
        return 3
    if pref >= 7.0:
        return 2
    if pref >= 5.5:
        return 1
    return 0


def _pct(a: np.ndarray) -> np.ndarray:
    if len(a) <= 1:
        return np.zeros_like(a)
    return a.argsort().argsort().astype(np.float32) / (len(a) - 1)


def build_features(ctx, uv, cand_idx: np.ndarray) -> np.ndarray:
    """Feature matrix (C, len(FEATURE_NAMES)) for one user's candidates."""
    C = len(cand_idx)
    z = lambda: np.zeros(C, dtype=np.float32)

    def cos(mat, prof):
        return (mat[cand_idx] @ prof).astype(np.float32) if prof is not None else z()

    meta_pos, meta_neg = cos(ctx.meta, uv.meta_pos), cos(ctx.meta, uv.meta_neg)
    plot_pos, plot_neg = cos(ctx.plot, uv.plot_pos), cos(ctx.plot, uv.plot_neg)

    comm_pos, comm_neg = z(), z()
    if uv.comm_pos is not None and ctx.comm_flat.shape[0]:
        flat_pos = ctx.comm_flat @ uv.comm_pos            # (M_total,) — one matmul
        flat_neg = ctx.comm_flat @ uv.comm_neg if uv.comm_neg is not None else None
        off = ctx.comm_off
        for i, j in enumerate(cand_idx):
            a, b = off[j], off[j + 1]
            if b > a:
                comm_pos[i] = float(flat_pos[a:b].max())
                if flat_neg is not None:
                    comm_neg[i] = float(flat_neg[a:b].max())

    if uv.mf is not None:
        mf = (ctx.mf_item[cand_idx] @ uv.mf).astype(np.float32)
        mf = np.where(ctx.mf_mask[cand_idx], mf, 0.0)
    else:
        mf = z()
    mf_pct = _pct(mf)

    pop = ctx.pop[cand_idx]
    pop_pct = _pct(pop)
    rc_log = np.log1p(ctx.review_count[cand_idx]).astype(np.float32)
    avgp = ctx.avg_pref[cand_idx]

    liked_actors = set()
    for j in uv.liked_idx:
        liked_actors |= ctx.actors[j]
    genre_ov, fav_match, recent_match, actor_ov, year_dist = z(), z(), z(), z(), z()
    for i, j in enumerate(cand_idx):
        g = ctx.genres[j]
        genre_ov[i] = len(g & uv.fav_genres)
        fav_match[i] = 1.0 if (uv.fav_genre is not None and uv.fav_genre in g) else 0.0
        recent_match[i] = 1.0 if (uv.last_liked_genres & g) else 0.0
        actor_ov[i] = len(ctx.actors[j] & liked_actors)
        yr = ctx.year[j]
        year_dist[i] = (abs(yr - uv.pref_year)
                        if (uv.pref_year is not None and not np.isnan(yr)) else np.nan)

    n = C
    return np.column_stack([
        meta_pos, meta_neg, meta_pos - meta_neg,
        plot_pos, plot_neg, plot_pos - plot_neg,
        comm_pos, comm_neg, comm_pos - comm_neg,
        mf, mf_pct,
        pop, pop_pct, rc_log, avgp,
        genre_ov, fav_match, recent_match,
        actor_ov, year_dist,
        np.full(n, uv.avg_pref, np.float32),
        np.full(n, np.log1p(uv.n_pos), np.float32),
        np.full(n, 1.0 if uv.mf is not None else 0.0, np.float32),
    ]).astype(np.float32)


def train_ranker(X, y, qid):
    from xgboost import XGBRanker
    model = XGBRanker(tree_method="hist", eval_metric="ndcg@10", **config.XGB_PARAMS)
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
