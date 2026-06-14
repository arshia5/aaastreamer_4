"""Seed community data: embed every review (MiniLM/MPS), store the vectors in
interactions.review_body_embedding, and KMeans-cluster per movie into
movie_community_embeddings.

This is the one-time seed. In production, reviews are embedded at write-time and
the nightly job re-clusters only movies with new reviews (from the stored vectors).

Usage:
    python -m scripts.build_community [--batch 600]
"""
import asyncio
import sys
import time

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml.community import cluster_review_vectors, embed_reviews


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


async def main(movie_batch: int) -> None:
    t0 = time.time()
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    movie_ids = [r["id"] for r in await conn.fetch(
        "SELECT m.id FROM movies m WHERE EXISTS (SELECT 1 FROM interactions i "
        "WHERE i.movie_id=m.id AND i.review_body IS NOT NULL) ORDER BY m.id")]
    print(f"{len(movie_ids):,} movies with reviews")
    await conn.execute("TRUNCATE movie_community_embeddings")
    await conn.execute("TRUNCATE review_embeddings")

    done = embedded = 0
    for start in range(0, len(movie_ids), movie_batch):
        batch = movie_ids[start:start + movie_batch]
        rows = await conn.fetch(
            "SELECT id, movie_id, review_body FROM interactions "
            "WHERE movie_id = ANY($1) AND review_body IS NOT NULL ORDER BY movie_id",
            batch)
        if not rows:
            done += len(batch)
            continue
        vecs = embed_reviews([r["review_body"] for r in rows])   # MPS
        embedded += len(rows)

        # 1) store each review vector (COPY into dedicated table — fast)
        await conn.copy_records_to_table(
            "review_embeddings",
            records=[(r["id"], r["movie_id"], vecs[i]) for i, r in enumerate(rows)],
            columns=["interaction_id", "movie_id", "embedding"])

        # 2) cluster per movie -> centroids
        by_movie: dict[int, list[int]] = {}
        for i, r in enumerate(rows):
            by_movie.setdefault(r["movie_id"], []).append(i)
        crecords = []
        for mid, idxs in by_movie.items():
            for ci, (cent, w) in enumerate(cluster_review_vectors(vecs[idxs])):
                crecords.append((mid, ci, float(w), cent))
        if crecords:
            await conn.copy_records_to_table(
                "movie_community_embeddings", records=crecords,
                columns=["movie_id", "cluster_idx", "weight", "embedding"])
        done += len(batch)
        print(f"  {done:,}/{len(movie_ids):,} movies | {embedded:,} reviews "
              f"| {time.time() - t0:.0f}s", end="\r", flush=True)
    print()
    total = await conn.fetchval("SELECT count(DISTINCT movie_id) FROM movie_community_embeddings")
    stored = await conn.fetchval("SELECT count(*) FROM review_embeddings")
    print(f"community for {total:,} movies; {stored:,} review vectors stored; "
          f"{time.time() - t0:.0f}s")
    await conn.close()


if __name__ == "__main__":
    b = 600
    if "--batch" in sys.argv:
        b = int(sys.argv[sys.argv.index("--batch") + 1])
    asyncio.run(main(b))
