"""similar_movies maintenance (v4, multi-channel).

Nightly: rebuild_similar_hybrid blends plot + metadata + MF + community similarity
(content-weighted), falling back to content when a movie lacks an MF/community
vector. Incremental (new/edited movie): content-only (plot+metadata) via SQL,
since the movie has no MF/community vector until the next nightly run.
"""
from __future__ import annotations

import logging

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml import config

log = logging.getLogger("recsys.similar")

# content-only blend weights (renormalised plot/meta)
_WP = config.SIMILAR_W_PLOT / (config.SIMILAR_W_PLOT + config.SIMILAR_W_META)
_WM = config.SIMILAR_W_META / (config.SIMILAR_W_PLOT + config.SIMILAR_W_META)

_REPLACE_SQL = text(
    f"""
    INSERT INTO similar_movies (movie_id, similar_movie_id, score, rank)
    SELECT :mid, sub.sid, sub.score, sub.rnk FROM (
        SELECT p.movie_id AS sid,
               ({_WP} * (1 - (p.embedding <=> qp.e))
                + {_WM} * (1 - (m.embedding <=> qm.e)))::double precision AS score,
               row_number() OVER (ORDER BY
                   {_WP} * (p.embedding <=> qp.e) + {_WM} * (m.embedding <=> qm.e)) AS rnk
        FROM movie_plot_embeddings p
        JOIN movie_metadata_embeddings m ON m.movie_id = p.movie_id,
             (SELECT embedding e FROM movie_plot_embeddings WHERE movie_id = :mid) qp,
             (SELECT embedding e FROM movie_metadata_embeddings WHERE movie_id = :mid) qm
        WHERE p.movie_id <> :mid
        ORDER BY {_WP} * (p.embedding <=> qp.e) + {_WM} * (m.embedding <=> qm.e)
        LIMIT :topn
    ) sub
    """
)


async def update_similar_movies_for(db: AsyncSession, movie_id: int,
                                    top_n: int = config.SIMILAR_TOP_N) -> int:
    """Replace a single movie's similar list (content-only). No-op without a
    plot/metadata embedding."""
    has = await db.scalar(
        text("SELECT 1 FROM movie_plot_embeddings WHERE movie_id = :m"), {"m": movie_id})
    if not has:
        return 0
    await db.execute(text("DELETE FROM similar_movies WHERE movie_id = :m"), {"m": movie_id})
    await db.execute(_REPLACE_SQL, {"mid": movie_id, "topn": top_n})
    await db.commit()
    return 1


def _l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


async def rebuild_similar_hybrid(conn, ctx, top_n: int = config.SIMILAR_TOP_N,
                                 batch: int = 1024) -> int:
    """Nightly: top-N per movie blending plot+metadata+MF+community similarity.
    `ctx` is a v4 RecoContext (already has the loaded channels)."""
    ids, idx = ctx.ids, ctx.idx
    n = len(ids)
    plot, meta = ctx.plot, ctx.meta            # already unit-normalised
    mf = _l2(ctx.mf_item.copy())
    comm = ctx.comm_mean                        # unit-normalised, 0 where missing
    wp, wm, wf, wc = (config.SIMILAR_W_PLOT, config.SIMILAR_W_META,
                      config.SIMILAR_W_MF, config.SIMILAR_W_COMM)
    log.info("Rebuilding v4 similar_movies for %d movies", n)
    done = 0
    for start in range(0, n, batch):
        rows = list(range(start, min(start + batch, n)))
        s = wp * (plot[rows] @ plot.T) + wm * (meta[rows] @ meta.T)
        # MF / community only contribute where both movies have the vector
        mf_s = mf[rows] @ mf.T
        s = s + wf * np.where((ctx.mf_mask[rows][:, None] & ctx.mf_mask[None, :]), mf_s, 0.0)
        comm_s = comm[rows] @ comm.T
        s = s + wc * np.where((ctx.comm_mask[rows][:, None] & ctx.comm_mask[None, :]), comm_s, 0.0)
        batch_ids, records = [], []
        for bi, gi in enumerate(rows):
            row = s[bi].copy()
            row[gi] = -np.inf
            top = np.argpartition(row, -top_n)[-top_n:]
            top = top[np.argsort(row[top])[::-1]]
            batch_ids.append(int(ids[gi]))
            for rank, j in enumerate(top, 1):
                if np.isfinite(row[j]):
                    records.append((int(ids[gi]), int(ids[j]), float(row[j]), rank))
        async with conn.transaction():
            await conn.execute("DELETE FROM similar_movies WHERE movie_id = ANY($1::int[])", batch_ids)
            await conn.copy_records_to_table(
                "similar_movies", records=records,
                columns=["movie_id", "similar_movie_id", "score", "rank"])
        done += len(batch_ids)
    return done
