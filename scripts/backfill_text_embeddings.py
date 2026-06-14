"""Backfill movie_metadata_embeddings + movie_plot_embeddings (mpnet, 768-d).

Reads metadata from movies.csv (fast, matches the catalogue) and upserts the two
component vectors per movie.

Usage:
    python -m scripts.backfill_text_embeddings [path/to/movies.csv]
"""
import asyncio
import sys
import time

import asyncpg
import numpy as np
import pandas as pd
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml import config as mlconfig
from app.ml import text_embed


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


def df_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {"title": r.get("title"), "year": r.get("year"), "genre": r.get("genre"),
         "director": r.get("director"), "writer": r.get("writer"),
         "actors": r.get("actors"), "language": r.get("language"),
         "country": r.get("country"), "plot": r.get("plot")}
        for r in df.to_dict(orient="records")
    ]


async def main(csv_path: str) -> None:
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"{len(df):,} movies")
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    movie_map = {r["imdb_id"]: r["id"]
                 for r in await conn.fetch("SELECT id, imdb_id FROM movies")}

    imdb = df["imdbid"].astype(str).tolist()
    records = df_to_records(df)
    print("Encoding metadata (mpnet) ...")
    meta = text_embed.embed_metadata(records)
    print(f"  metadata {meta.shape} in {time.time() - t0:.0f}s")
    print("Encoding plots (mpnet) ...")
    plot = text_embed.embed_plot(records)
    print(f"  plot {plot.shape}")

    meta_rows, plot_rows = [], []
    for i, im in enumerate(imdb):
        mid = movie_map.get(im)
        if mid is None:
            continue
        meta_rows.append((mid, meta[i]))
        plot_rows.append((mid, plot[i]))

    for table, rows in [("movie_metadata_embeddings", meta_rows),
                        ("movie_plot_embeddings", plot_rows)]:
        await conn.execute(f"TRUNCATE {table}")
        await conn.copy_records_to_table(
            table, records=[(m, v) for m, v in rows],
            columns=["movie_id", "embedding"])
        total = await conn.fetchval(f"SELECT count(*) FROM {table}")
        print(f"  {table}: {total:,}")
    await conn.close()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(mlconfig.DEFAULT_MOVIES_CSV)
    asyncio.run(main(path))
