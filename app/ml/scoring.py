"""Preference scoring and user-embedding weighting helpers."""
from __future__ import annotations

from app.ml import config


def clamp_0_10(x: float | None) -> float | None:
    if x is None:
        return None
    return max(0.0, min(10.0, float(x)))


def preference_score(rating: float | None, sentiment: float | None) -> float | None:
    """Blend rating and sentiment into a 0..10 preference.

    Uses 0.7*rating + 0.3*sentiment when both exist; falls back to whichever
    component is present, and returns None if neither is.
    """
    r = clamp_0_10(rating)
    s = clamp_0_10(sentiment)
    rw, sw = config.PREF_RATING_WEIGHT, config.PREF_SENTIMENT_WEIGHT
    if r is not None and s is not None:
        return rw * r + sw * s
    if r is not None:
        return r
    if s is not None:
        return s
    return None


def recency_decay_weights(n: int, gamma: float = config.USER_EMBED_DECAY) -> list[float]:
    """Rank-based decay weights for n reviews sorted oldest -> newest.

    The newest review (last index) gets weight 1.0; each step older is
    multiplied by gamma. Independent of absolute time gaps, so a user whose
    latest review was long ago still gets a well-defined embedding.
    """
    # index i (0=oldest .. n-1=newest); rank_from_newest = (n-1-i)
    return [gamma ** (n - 1 - i) for i in range(n)]


def centered_weight(pref: float | None) -> float:
    """Centre preference around the neutral midpoint so disliked movies
    (pref < neutral) contribute a negative pull."""
    if pref is None:
        return 0.0
    return pref - config.PREF_NEUTRAL
