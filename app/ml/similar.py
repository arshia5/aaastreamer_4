"""Maintain the similar_movies table (top-N nearest movies by embedding).

`score` is cosine similarity (1 - cosine_distance); rank 1 = most similar.
Used incrementally when a single movie's embedding is (re)generated.
"""
from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml import config

log = logging.getLogger("recsys.similar")

# Replace one movie's top-N list.
_REPLACE_SQL = text(
    """
    INSERT INTO similar_movies (movie_id, similar_movie_id, score, rank)
    SELECT :mid, sub.sid, sub.score, sub.rnk
    FROM (
        SELECT me.movie_id AS sid,
               (1 - (me.embedding <=> q.emb))::double precision AS score,
               row_number() OVER (ORDER BY me.embedding <=> q.emb) AS rnk
        FROM movie_embeddings me,
             (SELECT embedding AS emb FROM movie_embeddings WHERE movie_id = :mid) q
        WHERE me.movie_id <> :mid
        ORDER BY me.embedding <=> q.emb
        LIMIT :topn
    ) sub
    """
)

# Movies whose top-N should now include :mid (their weakest score is beaten,
# or they have fewer than N neighbours yet).
_AFFECTED_SQL = text(
    """
    WITH q AS (SELECT embedding AS emb FROM movie_embeddings WHERE movie_id = :mid),
    cand AS (
        SELECT me.movie_id AS x, (1 - (me.embedding <=> q.emb)) AS s
        FROM movie_embeddings me, q
        WHERE me.movie_id <> :mid
    ),
    th AS (
        SELECT movie_id AS x, MIN(score) AS s_min, COUNT(*) AS cnt
        FROM similar_movies
        GROUP BY movie_id
    )
    SELECT cand.x
    FROM cand LEFT JOIN th ON th.x = cand.x
    WHERE th.x IS NULL OR th.cnt < :topn OR cand.s > th.s_min
    """
)


async def _replace_top_n(db: AsyncSession, movie_id: int, top_n: int) -> None:
    await db.execute(
        text("DELETE FROM similar_movies WHERE movie_id = :mid"), {"mid": movie_id}
    )
    await db.execute(_REPLACE_SQL, {"mid": movie_id, "topn": top_n})


async def update_similar_movies_for(
    db: AsyncSession, movie_id: int, top_n: int = config.SIMILAR_TOP_N
) -> int:
    """Recompute the movie's own top-N, then refresh any other movie whose
    top-N this movie now belongs in. Returns the number of other movies updated.

    No-op if the movie has no embedding. Commits on success.
    """
    has_emb = await db.scalar(
        text("SELECT 1 FROM movie_embeddings WHERE movie_id = :mid"),
        {"mid": movie_id},
    )
    if not has_emb:
        return 0

    await _replace_top_n(db, movie_id, top_n)
    affected = (
        await db.execute(_AFFECTED_SQL, {"mid": movie_id, "topn": top_n})
    ).scalars().all()
    for x in affected:
        if x != movie_id:
            await _replace_top_n(db, x, top_n)
    await db.commit()
    return len(affected)


def _l2norm(mat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


async def rebuild_similar_hybrid(
    conn,
    ids: np.ndarray,
    content_unit: np.ndarray,
    idx: dict,
    collab,
    top_n: int = config.SIMILAR_TOP_N,
    w_content: float = config.SIMILAR_W_CONTENT,
    w_collab: float = config.SIMILAR_W_COLLAB,
    batch: int = 1024,
) -> int:
    """Rebuild similar_movies as a content-weighted hybrid (nightly).

    score(X, Y) = w_content * cos(content_X, content_Y)
                + w_collab  * cos(mf_X, mf_Y)         (only when both have MF)

    A movie with no MF item factor (new / too few reviews) is scored content-only,
    and a candidate without an MF factor contributes only its content term. Writes
    transactionally per batch (no global TRUNCATE) so the site stays consistent.
    """
    n = len(ids)
    has_mf = np.zeros(n, dtype=bool)
    itf_norm = None
    if collab is not None:
        cf_dim = collab.item_factors.shape[1]
        itf = np.zeros((n, cf_dim), dtype=np.float32)
        for mid, row in collab.item_map.items():
            j = idx.get(mid)
            if j is not None:
                itf[j] = collab.item_factors[row]
                has_mf[j] = True
        itf_norm = _l2norm(itf)
    log.info("Rebuilding hybrid similar_movies: %d movies, %d with MF factor",
             n, int(has_mf.sum()))

    done = 0
    for start in range(0, n, batch):
        rows = list(range(start, min(start + batch, n)))
        cs = content_unit[rows] @ content_unit.T                       # (B, N)
        xs = itf_norm[rows] @ itf_norm.T if itf_norm is not None else None
        batch_ids, records = [], []
        for bi, gi in enumerate(rows):
            if itf_norm is not None and has_mf[gi]:
                score = w_content * cs[bi] + w_collab * xs[bi]
            else:
                score = cs[bi].copy()                                  # content-only
            score[gi] = -np.inf                                        # exclude self
            top = np.argpartition(score, -(top_n))[-top_n:]
            top = top[np.argsort(score[top])[::-1]]
            batch_ids.append(int(ids[gi]))
            for rank, j in enumerate(top, start=1):
                if not np.isfinite(score[j]):
                    continue
                records.append((int(ids[gi]), int(ids[j]), float(score[j]), rank))
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM similar_movies WHERE movie_id = ANY($1::int[])", batch_ids)
            await conn.copy_records_to_table(
                "similar_movies", records=records,
                columns=["movie_id", "similar_movie_id", "score", "rank"])
        done += len(batch_ids)
    log.info("Hybrid similar_movies rebuilt for %d movies", done)
    return done
