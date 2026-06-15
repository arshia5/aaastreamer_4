"""Per-user taste vectors for the v4 ranker.

For each content channel (metadata, plot, community) we build a *positive* and a
*negative* profile = preference-centered, recency-weighted, L2-normalised sum of
the user's liked / disliked movie vectors. Preference is centered on a per-user
baseline (shrunk toward the global mean) so harsh and generous raters are
comparable. The collaborative (LightGCN) user vector is carried alongside.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from app.ml import config

_MIN_DT = datetime.min
BASELINE_PRIOR = 5  # shrinkage strength for the per-user mean


@dataclass
class UserVectors:
    meta_pos: np.ndarray | None = None
    meta_neg: np.ndarray | None = None
    struct_pos: np.ndarray | None = None
    struct_neg: np.ndarray | None = None
    plot_pos: np.ndarray | None = None
    plot_neg: np.ndarray | None = None
    comm_pos: np.ndarray | None = None
    comm_neg: np.ndarray | None = None
    mf: np.ndarray | None = None
    avg_pref: float = 5.5
    n_pos: int = 0
    last_date: datetime | None = None
    fav_genres: set = field(default_factory=set)
    fav_genre: int | None = None
    pref_year: float | None = None
    last_liked_genres: set = field(default_factory=set)
    liked_idx: list = field(default_factory=list)


def _norm(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else None


def user_baseline(prefs: list[float], global_mean: float) -> float:
    n = len(prefs)
    if n == 0:
        return global_mean
    return (BASELINE_PRIOR * global_mean + float(np.sum(prefs))) / (BASELINE_PRIOR + n)


def build_user_vectors(ctx, items, global_mean: float,
                       gamma: float = config.USER_EMBED_DECAY) -> UserVectors:
    """items: [(movie_id, preference, review_date)] (train split)."""
    rows = [(m, p, d) for (m, p, d) in items if m in ctx.idx and p is not None]
    if not rows:
        return UserVectors()
    rows.sort(key=lambda x: x[2] or _MIN_DT)            # oldest -> newest
    prefs = [p for _, p, _ in rows]
    base = user_baseline(prefs, global_mean)
    n = len(rows)

    uv = UserVectors(avg_pref=float(np.mean(prefs)),
                     last_date=rows[-1][2])
    genre_counts: Counter = Counter()
    liked_years: list[float] = []
    has_struct = getattr(ctx, "struct", None) is not None
    sd = ctx.struct.shape[1] if has_struct else 0
    acc = {k: np.zeros(d, dtype=np.float32) for k, d in
           [("meta_pos", 768), ("meta_neg", 768), ("plot_pos", 768),
            ("plot_neg", 768), ("comm_pos", 384), ("comm_neg", 384),
            ("struct_pos", sd), ("struct_neg", sd)]}
    for rank_from_new, (mid, pref, _) in enumerate(reversed(rows)):
        j = ctx.idx[mid]
        decay = gamma ** rank_from_new
        c = (pref - base) * decay
        liked = c > 0
        w = abs(c)
        if liked:
            uv.n_pos += 1
            uv.liked_idx.append(j)
            uv.fav_genres |= ctx.genres[j]
            genre_counts.update(ctx.genres[j])
            if not np.isnan(ctx.year[j]):
                liked_years.append(float(ctx.year[j]))
            if not uv.last_liked_genres:           # first liked seen = most recent
                uv.last_liked_genres = set(ctx.genres[j])
        suffix = "pos" if liked else "neg"
        acc[f"meta_{suffix}"] += w * ctx.meta[j]
        acc[f"plot_{suffix}"] += w * ctx.plot[j]
        if has_struct and ctx.struct_mask[j]:
            acc[f"struct_{suffix}"] += w * ctx.struct[j]
        if ctx.comm_mask[j]:
            acc[f"comm_{suffix}"] += w * ctx.comm_mean[j]

    uv.meta_pos, uv.meta_neg = _norm(acc["meta_pos"]), _norm(acc["meta_neg"])
    if has_struct:
        uv.struct_pos, uv.struct_neg = _norm(acc["struct_pos"]), _norm(acc["struct_neg"])
    uv.plot_pos, uv.plot_neg = _norm(acc["plot_pos"]), _norm(acc["plot_neg"])
    uv.comm_pos, uv.comm_neg = _norm(acc["comm_pos"]), _norm(acc["comm_neg"])
    uv.fav_genre = genre_counts.most_common(1)[0][0] if genre_counts else None
    uv.pref_year = float(np.mean(liked_years)) if liked_years else None
    return uv
