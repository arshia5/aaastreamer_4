"""Build 390-d user embeddings from interactions (preference + recency decay).

Loads the movie embedding matrix once, streams interactions grouped by user,
computes each user's weighted/decayed vector, and upserts user_embeddings.

Usage:
    python -m scripts.build_user_embeddings
"""
import asyncio
import time

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml.user_embedding import compute_user_vector


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


async def main() -> None:
    t0 = time.time()
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    try:
        print("Loading movie embeddings ...")
        emb = {
            r["movie_id"]: np.asarray(r["embedding"], dtype=np.float32)
            for r in await conn.fetch("SELECT movie_id, embedding FROM movie_embeddings")
        }
        print(f"  {len(emb):,} movie vectors")

        print("Loading interactions ...")
        rows = await conn.fetch(
            "SELECT user_id, movie_id, preference_score, review_date "
            "FROM interactions WHERE preference_score IS NOT NULL "
            "ORDER BY user_id"
        )
        print(f"  {len(rows):,} scored interactions")

        # group consecutive rows by user_id and compute vectors
        results: list[tuple[int, np.ndarray]] = []
        cur_uid = None
        items: list[tuple[int, float, object]] = []

        def flush(uid, its):
            if uid is None:
                return
            vec = compute_user_vector(its, emb)
            if vec is not None:
                results.append((uid, vec))

        for r in rows:
            if r["user_id"] != cur_uid:
                flush(cur_uid, items)
                cur_uid, items = r["user_id"], []
            items.append((r["movie_id"], r["preference_score"], r["review_date"]))
        flush(cur_uid, items)
        print(f"  computed {len(results):,} user vectors")

        print("Upserting user_embeddings ...")
        await conn.executemany(
            "INSERT INTO user_embeddings(user_id, embedding) VALUES($1, $2) "
            "ON CONFLICT (user_id) DO UPDATE "
            "SET embedding = EXCLUDED.embedding, updated_at = now()",
            [(uid, vec) for uid, vec in results],
        )
        total = await conn.fetchval("SELECT count(*) FROM user_embeddings")
        print(f"  user_embeddings now: {total:,}")
    finally:
        await conn.close()
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
