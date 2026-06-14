"""Fill the similar_movies table: top-N nearest movies per movie.

Exact cosine similarity computed in numpy (batched), then bulk-loaded.
Use this for the initial fill or a full rebuild; single new movies are handled
incrementally by app.ml.similar.update_similar_movies_for.

Usage:
    python -m scripts.build_similar_movies [--top 10] [--batch 2048]
"""
import asyncio
import sys
import time

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml import config


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


async def main(top_n: int, batch: int) -> None:
    t0 = time.time()
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    try:
        print("Loading movie embeddings ...")
        rows = await conn.fetch(
            "SELECT movie_id, embedding FROM movie_embeddings ORDER BY movie_id"
        )
        ids = np.array([r["movie_id"] for r in rows], dtype=np.int64)
        mat = np.array(
            [np.asarray(r["embedding"], dtype=np.float32) for r in rows],
            dtype=np.float32,
        )
        n = len(ids)
        print(f"  {n:,} vectors x {mat.shape[1]} dims")

        # L2-normalise so cosine similarity == dot product.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = mat / norms

        print(f"Computing top-{top_n} per movie (batch={batch}) ...")
        records: list[tuple[int, int, float, int]] = []
        k = top_n + 1  # +1 to drop self
        for start in range(0, n, batch):
            block = unit[start:start + batch]                 # (B, D)
            sims = block @ unit.T                             # (B, N)
            for bi in range(block.shape[0]):
                gi = start + bi
                row = sims[bi]
                row[gi] = -np.inf                             # exclude self
                top = np.argpartition(row, -k)[-k:]
                top = top[np.argsort(row[top])[::-1]][:top_n]  # desc, drop extra
                for rank, j in enumerate(top, start=1):
                    records.append(
                        (int(ids[gi]), int(ids[j]), float(row[j]), rank)
                    )
            print(f"  {min(start + batch, n):,}/{n:,}", end="\r", flush=True)
        print()

        print(f"Writing {len(records):,} rows ...")
        await conn.execute("TRUNCATE similar_movies")
        await conn.copy_records_to_table(
            "similar_movies",
            records=records,
            columns=["movie_id", "similar_movie_id", "score", "rank"],
        )
        total = await conn.fetchval("SELECT count(*) FROM similar_movies")
        print(f"  similar_movies now: {total:,}")
    finally:
        await conn.close()
    print(f"Done in {time.time() - t0:.1f}s")


def _parse(argv):
    top_n, batch = config.SIMILAR_TOP_N, 2048
    i = 0
    while i < len(argv):
        if argv[i] == "--top":
            top_n = int(argv[i + 1]); i += 2
        elif argv[i] == "--batch":
            batch = int(argv[i + 1]); i += 2
        else:
            i += 1
    return top_n, batch


if __name__ == "__main__":
    t, b = _parse(sys.argv[1:])
    asyncio.run(main(t, b))
