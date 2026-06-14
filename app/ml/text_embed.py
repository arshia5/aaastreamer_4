"""mpnet text embeddings for the metadata and plot channels (v4).

Stateless (no fitted artifacts) so new movies are embedded with a single encode.
- metadata channel: a normalized text template of structured fields, embedded.
- plot channel: the plot text, embedded.
Both 768-d (all-mpnet-base-v2). Lazy singleton; torch loads on first use.
"""
from __future__ import annotations

import threading

import numpy as np

from app.ml import config
from app.ml.features import clean_text, clean_tokens

_model = None
_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(config.TEXT_EMBED_MODEL)
    return _model


def metadata_text(record: dict) -> str:
    """Build a natural-language template from a movie's structured metadata.
    record keys: title, year, genre, director, writer, actors, language, country."""
    parts: list[str] = []
    title = clean_text(record.get("title") or record.get("movie_title"))
    if title:
        parts.append(f"Title: {title}.")
    year = record.get("year")
    if year:
        parts.append(f"Year: {int(year)}.")
    genres = clean_tokens(record.get("genre"))
    if genres:
        parts.append("Genres: " + ", ".join(genres) + ".")
    directors = clean_tokens(record.get("director"))
    if directors:
        parts.append("Directed by " + ", ".join(directors) + ".")
    writers = clean_tokens(record.get("writer"))
    if writers:
        parts.append("Written by " + ", ".join(writers) + ".")
    actors = clean_tokens(record.get("actors"))
    if actors:
        parts.append("Starring " + ", ".join(actors[:8]) + ".")
    langs = clean_tokens(record.get("language"))
    if langs:
        parts.append("Language: " + ", ".join(langs) + ".")
    countries = clean_tokens(record.get("country"))
    if countries:
        parts.append("Country: " + ", ".join(countries) + ".")
    return " ".join(parts)


def _encode(texts: list[str], batch_size: int = 128) -> np.ndarray:
    return get_model().encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=False,
    ).astype(np.float32)


def embed_metadata(records: list[dict]) -> np.ndarray:
    return _encode([metadata_text(r) for r in records])


def embed_plot(records: list[dict]) -> np.ndarray:
    return _encode([clean_text(r.get("plot")) for r in records])


def embed_metadata_one(record: dict) -> list[float]:
    return _encode([metadata_text(record)])[0].tolist()


def embed_plot_one(record: dict) -> list[float]:
    return _encode([clean_text(record.get("plot"))])[0].tolist()
