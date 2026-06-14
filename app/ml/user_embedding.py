"""Compute a user's 390-d embedding from their reviewed movies.

vector(user) = Σ_i  (preference_i − neutral) · γ^rank_i · movie_embedding_i

where reviews are sorted oldest→newest, the newest gets rank 0 (decay 1.0), and
each older review is multiplied by γ per step (rank-based, not absolute time).
Disliked movies (preference < neutral) contribute a negative pull.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from app.ml import config
from app.ml.scoring import centered_weight, recency_decay_weights

_MIN_DT = datetime.min


def compute_user_vector(
    items: list[tuple[int, float | None, datetime | None]],
    emb_lookup,
    dim: int = config.N_COMPONENTS,
    gamma: float = config.USER_EMBED_DECAY,
) -> np.ndarray | None:
    """items: (movie_id, preference_score, review_date). emb_lookup: movie_id ->
    390-d vector (or None if absent). Returns a float32 vector or None."""
    usable = []
    for movie_id, pref, date in items:
        emb = emb_lookup(movie_id) if callable(emb_lookup) else emb_lookup.get(movie_id)
        if emb is None or pref is None:
            continue
        usable.append((date or _MIN_DT, pref, np.asarray(emb, dtype=np.float32)))
    if not usable:
        return None

    usable.sort(key=lambda x: x[0])  # oldest -> newest
    decays = recency_decay_weights(len(usable), gamma)
    acc = np.zeros(dim, dtype=np.float32)
    for (_, pref, emb), decay in zip(usable, decays):
        acc += (centered_weight(pref) * decay) * emb
    # L2-normalise so user vectors live on the unit sphere (cosine-friendly).
    norm = float(np.linalg.norm(acc))
    if norm > 0:
        acc = acc / norm
    return acc.astype(np.float32)
