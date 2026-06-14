"""Deterministic feature-block construction for the movie embedding pipeline.

A "record" is a dict with these keys (all optional except as noted):
    plot:     str
    genre:    list[str]
    director: list[str]
    writer:   list[str]
    actors:   list[str]
    language: list[str]
    country:  list[str]
    year:     float | None

The MultiLabelBinarizer (genre) and StandardScaler (year) are *fitted* objects;
FeatureHasher blocks and the MiniLM encoder are deterministic given config.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from app.ml import config


def clean_tokens(value) -> list[str]:
    """Normalise a raw field into a list of clean tokens.

    Accepts a list (already split) or a comma-separated string.
    """
    if value is None:
        return []
    if isinstance(value, float):
        # pandas represents missing string cells as NaN floats.
        return [] if np.isnan(value) else [str(value)]
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple, set, np.ndarray)):
        parts = list(value)
    else:
        parts = [value]
    out = []
    for p in parts:
        s = str(p).strip()
        if s and s.lower() != "nan":
            out.append(s)
    return out


def clean_text(value) -> str:
    """Coerce a possibly-missing cell into a plain string (NaN/None -> '')."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def encode_plots(st_model, plots: list[str]) -> np.ndarray:
    return st_model.encode(
        [clean_text(p) for p in plots],
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def genre_block(genre_lists: list[list[str]], mlb) -> np.ndarray:
    classes = set(mlb.classes_)
    filtered = [[g for g in lst if g in classes] for lst in genre_lists]
    return mlb.transform(filtered).astype(np.float32)


def hashed_block(token_lists: list[list[str]], n_features: int) -> np.ndarray:
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    return hasher.transform(token_lists).toarray().astype(np.float32)


def year_block(years: list, scaler, median_year: float) -> np.ndarray:
    arr = np.array(
        [median_year if (y is None or _is_nan(y)) else float(y) for y in years],
        dtype=np.float32,
    ).reshape(-1, 1)
    return scaler.transform(arr).astype(np.float32)


def _is_nan(v) -> bool:
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False


def build_raw_blocks(records: list[dict], st_model, mlb, scaler, median_year):
    """Return the dict of raw (pre-normalise) blocks in canonical names."""
    plots = [clean_text(r.get("plot")) for r in records]
    return {
        "plot": encode_plots(st_model, plots),
        "genre": genre_block([clean_tokens(r.get("genre")) for r in records], mlb),
        "director": hashed_block(
            [clean_tokens(r.get("director")) for r in records], config.DIRECTOR_DIM
        ),
        "writer": hashed_block(
            [clean_tokens(r.get("writer")) for r in records], config.WRITER_DIM
        ),
        "actor": hashed_block(
            [clean_tokens(r.get("actors")) for r in records], config.ACTOR_DIM
        ),
        "language": hashed_block(
            [clean_tokens(r.get("language")) for r in records], config.LANG_DIM
        ),
        "country": hashed_block(
            [clean_tokens(r.get("country")) for r in records], config.COUNTRY_DIM
        ),
        "numeric": year_block(
            [r.get("year") for r in records], scaler, median_year
        ),
    }


def assemble(raw_blocks: dict) -> np.ndarray:
    """Apply L2-norm (except numeric) + weights, then concatenate in order."""
    normalized = {}
    for name, mat in raw_blocks.items():
        weighted = mat * config.WEIGHTS[name]
        if name == "numeric":
            normalized[name] = weighted  # 1-d: skip L2-norm
        else:
            normalized[name] = l2_normalize(mat) * config.WEIGHTS[name]
    ordered = [normalized[name] for name in config.BLOCK_ORDER]
    return np.concatenate(ordered, axis=1).astype(np.float32)


def build_feature_matrix(records, st_model, mlb, scaler, median_year) -> np.ndarray:
    """Full 732-d feature matrix for a batch of records."""
    return assemble(build_raw_blocks(records, st_model, mlb, scaler, median_year))
