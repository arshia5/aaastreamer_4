"""Community embeddings: K-means centroids of a movie's review vectors.

Each movie is summarised by up to K centroids capturing distinct audience
reactions. Reviews are embedded with MiniLM (384-d, fast). Movies with < K
reviews get fewer clusters; a 1-review movie gets a single centroid.
"""
from __future__ import annotations

import threading

import numpy as np

from app.ml import config

_model = None
_lock = threading.Lock()


def _best_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_review_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(config.REVIEW_EMBED_MODEL,
                                             device=_best_device())
                # Reviews are long; the gist/sentiment is clear early. Truncating
                # to 128 tokens ~halves encode time with negligible cluster impact.
                _model.max_seq_length = 128
    return _model


def embed_reviews(texts: list[str], batch_size: int = 256) -> np.ndarray:
    clean = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    return get_review_model().encode(
        clean, batch_size=batch_size, show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)


def embed_review_one(text: str) -> list[float]:
    return embed_reviews([text])[0].tolist()


async def rebuild_dirty_community(conn, batch: int = 500) -> int:
    """Nightly community maintenance (all work happens here, not at write-time):

    1. embed any reviews that don't yet have a stored vector (new since last run),
    2. re-cluster the movies that gained vectors (or are missing community).

    Uses stored review vectors so existing reviews are never re-embedded.
    `conn` must have pgvector registered. Returns #movies reclustered.
    """
    # 1) embed new reviews
    new_rows = await conn.fetch(
        "SELECT i.id, i.movie_id, i.review_body FROM interactions i "
        "LEFT JOIN review_embeddings re ON re.interaction_id = i.id "
        "WHERE i.review_body IS NOT NULL AND re.interaction_id IS NULL")
    dirty = set()
    for start in range(0, len(new_rows), 2000):
        chunk = new_rows[start:start + 2000]
        vecs = embed_reviews([r["review_body"] for r in chunk])
        await conn.copy_records_to_table(
            "review_embeddings",
            records=[(r["id"], r["movie_id"], vecs[i]) for i, r in enumerate(chunk)],
            columns=["interaction_id", "movie_id", "embedding"])
        dirty.update(r["movie_id"] for r in chunk)

    # movies that have review vectors but no community yet
    for r in await conn.fetch(
        "SELECT DISTINCT re.movie_id FROM review_embeddings re "
        "LEFT JOIN movie_community_embeddings c ON c.movie_id = re.movie_id "
        "WHERE c.movie_id IS NULL"):
        dirty.add(r["movie_id"])

    dirty = list(dirty)
    for start in range(0, len(dirty), batch):
        chunk = dirty[start:start + batch]
        rows = await conn.fetch(
            "SELECT movie_id, embedding FROM review_embeddings WHERE movie_id = ANY($1)", chunk)
        by_movie: dict[int, list] = {}
        for r in rows:
            by_movie.setdefault(r["movie_id"], []).append(np.asarray(r["embedding"], np.float32))
        records = []
        for mid, vs in by_movie.items():
            for ci, (cent, w) in enumerate(cluster_review_vectors(np.array(vs))):
                records.append((mid, ci, float(w), cent))
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM movie_community_embeddings WHERE movie_id = ANY($1)", chunk)
            if records:
                await conn.copy_records_to_table(
                    "movie_community_embeddings", records=records,
                    columns=["movie_id", "cluster_idx", "weight", "embedding"])
    return len(dirty)


def cluster_review_vectors(
    vecs: np.ndarray, k: int = config.COMMUNITY_K, seed: int = 42
) -> list[tuple[np.ndarray, float]]:
    """Return [(centroid, weight), ...] with weight = cluster share. Handles
    fewer reviews than k (uses min(k, n) clusters)."""
    n = len(vecs)
    if n == 0:
        return []
    if n <= k:
        # one centroid per review (or fewer); each weight = 1/n
        return [(vecs[i].astype(np.float32), 1.0 / n) for i in range(n)]
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    labels = km.fit_predict(vecs)
    out = []
    for c in range(k):
        mask = labels == c
        share = float(mask.sum()) / n
        if share > 0:
            out.append((km.cluster_centers_[c].astype(np.float32), share))
    return out
