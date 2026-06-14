"""Automatic v4 embedding refresh for a single movie (background task).

On movie create/edit or a genres/cast/etc. change, regenerate the metadata + plot
mpnet embeddings (stateless), store them, and refresh the movie's similar list
(content channels — the movie has no MF/community vector until the nightly run).
"""
from __future__ import annotations

import logging

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.ml import text_embed
from app.ml.db_records import movie_record_from_db
from app.ml.similar import update_similar_movies_for
from app.models import Movie

log = logging.getLogger("recsys.movie_refresh")
_LOCK_NS = 42


def _embed(record):
    return text_embed.embed_metadata_one(record), text_embed.embed_plot_one(record)


async def regenerate_movie_embedding(movie_id: int) -> None:
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT pg_advisory_lock(:ns, :m)"),
                             {"ns": _LOCK_NS, "m": movie_id})
            try:
                if await db.get(Movie, movie_id) is None:
                    return
                record = await movie_record_from_db(db, movie_id)
                if record is None:
                    return
                meta_vec, plot_vec = await run_in_threadpool(_embed, record)
                for table, vec in [("movie_metadata_embeddings", meta_vec),
                                   ("movie_plot_embeddings", plot_vec)]:
                    await db.execute(
                        text(f"INSERT INTO {table}(movie_id, embedding) "
                             f"VALUES (:m, :e) ON CONFLICT (movie_id) "
                             f"DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = now()"),
                        {"m": movie_id, "e": str(vec)})
                await db.commit()
                await update_similar_movies_for(db, movie_id)
                log.info("Auto-refreshed v4 embeddings + similar for movie %s", movie_id)
            finally:
                await db.execute(text("SELECT pg_advisory_unlock(:ns, :m)"),
                                 {"ns": _LOCK_NS, "m": movie_id})
    except Exception:
        log.exception("Auto embedding refresh failed for movie %s", movie_id)


def schedule_movie_refresh(background_tasks, movie_id: int) -> None:
    if background_tasks is not None:
        background_tasks.add_task(regenerate_movie_embedding, movie_id)
