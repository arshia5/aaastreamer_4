"""Compute and store 390-d embeddings for every movie in the database.

Reads each movie's relational metadata, runs the fitted pipeline, and upserts
movie_embeddings. Requires scripts.fit_embeddings to have been run first.

Usage:
    python -m scripts.backfill_embeddings [--batch 512] [--only-missing]
"""
import asyncio
import sys
import time

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml.embedder import MovieEmbedder

_ROLE_KEYS = {"director": "director", "writer": "writer", "actor": "actors"}


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


async def _load_all_records(conn, only_missing: bool):
    where = (
        "WHERE NOT EXISTS (SELECT 1 FROM movie_embeddings me WHERE me.movie_id = m.id)"
        if only_missing
        else ""
    )
    movies = await conn.fetch(
        f"SELECT m.id, m.plot, m.year FROM movies m {where} ORDER BY m.id"
    )
    ids = [m["id"] for m in movies]
    if not ids:
        return [], {}

    def group(rows, key_field="movie_id", val_field="name"):
        out: dict[int, list[str]] = {}
        for r in rows:
            out.setdefault(r[key_field], []).append(r[val_field])
        return out

    genres = group(await conn.fetch(
        "SELECT mg.movie_id, g.name FROM movie_genres mg "
        "JOIN genres g ON g.id = mg.genre_id"))
    langs = group(await conn.fetch(
        "SELECT ml.movie_id, l.name FROM movie_languages ml "
        "JOIN languages l ON l.id = ml.language_id"))
    countries = group(await conn.fetch(
        "SELECT mc.movie_id, c.name FROM movie_countries mc "
        "JOIN countries c ON c.id = mc.country_id"))

    people_rows = await conn.fetch(
        "SELECT mp.movie_id, r.name AS role, p.name AS person FROM movie_people mp "
        "JOIN roles r ON r.id = mp.role_id JOIN people p ON p.id = mp.person_id")
    people: dict[int, dict[str, list[str]]] = {}
    for r in people_rows:
        key = _ROLE_KEYS.get(r["role"])
        if not key:
            continue
        people.setdefault(r["movie_id"], {}).setdefault(key, []).append(r["person"])

    records = []
    for m in movies:
        mid = m["id"]
        pr = people.get(mid, {})
        records.append({
            "plot": m["plot"] or "",
            "genre": genres.get(mid, []),
            "director": pr.get("director", []),
            "writer": pr.get("writer", []),
            "actors": pr.get("actors", []),
            "language": langs.get(mid, []),
            "country": countries.get(mid, []),
            "year": m["year"],
        })
    return ids, records


async def main(batch: int, only_missing: bool) -> None:
    t0 = time.time()
    print("Loading fitted pipeline ...")
    embedder = MovieEmbedder.load()
    print(f"  PCA -> {embedder.n_components} dims "
          f"(explained var {embedder.explained_variance:.4f})")

    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    try:
        ids, records = await _load_all_records(conn, only_missing)
        total = len(ids)
        print(f"Embedding {total:,} movies (batch={batch}) ...")
        done = 0
        for start in range(0, total, batch):
            chunk_ids = ids[start:start + batch]
            chunk_recs = records[start:start + batch]
            vecs = embedder.embed_records(chunk_recs)
            await conn.executemany(
                "INSERT INTO movie_embeddings(movie_id, embedding) VALUES($1, $2) "
                "ON CONFLICT (movie_id) DO UPDATE "
                "SET embedding = EXCLUDED.embedding, updated_at = now()",
                [
                    (mid, np.asarray(v, dtype=np.float32))
                    for mid, v in zip(chunk_ids, vecs)
                ],
            )
            done += len(chunk_ids)
            print(f"  {done:,}/{total:,}", end="\r", flush=True)
        print()
    finally:
        await conn.close()
    print(f"Done in {time.time() - t0:.1f}s")


def _parse_args(argv):
    batch, only_missing = 512, False
    i = 0
    while i < len(argv):
        if argv[i] == "--batch":
            batch = int(argv[i + 1]); i += 2
        elif argv[i] == "--only-missing":
            only_missing = True; i += 1
        else:
            i += 1
    return batch, only_missing


if __name__ == "__main__":
    b, om = _parse_args(sys.argv[1:])
    asyncio.run(main(b, om))
