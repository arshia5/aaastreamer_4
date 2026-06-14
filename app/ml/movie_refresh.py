"""Automatic content-embedding + similar-movies refresh for a single movie.

Used as a background task so adding/editing a movie (or its genres/cast/etc.)
keeps its embedding and similar-movies list current without a manual endpoint
call. Similar movies here are content-only (the movie has no MF item factor yet —
that is blended in by the nightly job).
"""
from __future__ import annotations

import logging

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from app.core.database import AsyncSessionLocal
from app.ml.db_records import movie_record_from_db
from app.ml.embedder import MovieEmbedder, PipelineNotFitted
from app.ml.similar import update_similar_movies_for
from app.models import Movie, MovieEmbedding

log = logging.getLogger("recsys.movie_refresh")
_LOCK_NS = 42  # advisory-lock namespace for per-movie embedding refreshes


async def regenerate_movie_embedding(movie_id: int) -> None:
    try:
        embedder = MovieEmbedder.instance()
    except PipelineNotFitted:
        log.warning("Embedding pipeline not fitted; skipping movie %s", movie_id)
        return
    try:
        async with AsyncSessionLocal() as db:
            # Serialise concurrent refreshes of the same movie (e.g. create +
            # several metadata links firing at once) to avoid colliding writes.
            await db.execute(text("SELECT pg_advisory_lock(:ns, :m)"),
                             {"ns": _LOCK_NS, "m": movie_id})
            try:
                if await db.get(Movie, movie_id) is None:
                    return  # movie was deleted before this task ran
                record = await movie_record_from_db(db, movie_id)
                if record is None:
                    return
                vec = await run_in_threadpool(embedder.embed_one, record)
                emb = await db.get(MovieEmbedding, movie_id)
                if emb is None:
                    db.add(MovieEmbedding(movie_id=movie_id, embedding=vec))
                else:
                    emb.embedding = vec
                await db.commit()
                await update_similar_movies_for(db, movie_id)
                log.info("Auto-refreshed embedding + similar for movie %s", movie_id)
            finally:
                await db.execute(text("SELECT pg_advisory_unlock(:ns, :m)"),
                                 {"ns": _LOCK_NS, "m": movie_id})
    except Exception:
        log.exception("Auto embedding refresh failed for movie %s", movie_id)


def schedule_movie_refresh(background_tasks, movie_id: int) -> None:
    """Queue a background embedding+similar refresh (no-op if no tasks object)."""
    if background_tasks is not None:
        background_tasks.add_task(regenerate_movie_embedding, movie_id)
