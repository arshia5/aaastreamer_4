"""Compare the trained XGB ranker vs simple scorers on retrieved positives.

If a simple scorer (raw mf, meta cos, blend) beats XGB, the ranker is the defect.
Run: python -m scripts.diagnose_rankers
"""
import asyncio
import random

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml.profiles import build_user_vectors
from app.ml.ranker import XgbRanker, build_features
from app.ml.reco import build_context_from_conn, generate_candidates

SAMPLE = 3000
LIKE = 7.0


def dsn():
    return settings.database_url.replace("+asyncpg", "")


def hitk(recs, relset, k):
    return 1 if any(m in relset for m in recs[:k]) else 0


async def main():
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    ctx = await build_context_from_conn(conn)

    rows = await conn.fetch(
        "SELECT user_id, movie_id, preference_score, review_date FROM interactions "
        "WHERE preference_score IS NOT NULL ORDER BY user_id, review_date NULLS FIRST")
    per_user = {}
    for r in rows:
        per_user.setdefault(r["user_id"], []).append(
            (r["movie_id"], float(r["preference_score"]), r["review_date"]))
    train_items, test_items = {}, {}
    for uid, items in per_user.items():
        if len(items) >= 2:
            k = max(1, round(0.2 * len(items)))
            train_items[uid], test_items[uid] = items[:-k], items[-k:]
        else:
            train_items[uid] = items

    uids, seen, rel = [], {}, {}
    for uid, test in test_items.items():
        r = {m for m, p, _ in test if p >= LIKE and m in ctx.idx}
        if not r:
            continue
        uids.append(uid)
        seen[uid] = [ctx.idx[m] for m, _, _ in train_items.get(uid, []) if m in ctx.idx]
        rel[uid] = r
    random.Random(42).shuffle(uids)
    uids = uids[:SAMPLE]

    import os
    path = await conn.fetchval(
        "SELECT artifact_path FROM model_versions WHERE model_type='xgb_ranker' "
        "AND is_active ORDER BY created_at DESC LIMIT 1")
    ranker = XgbRanker.load(os.path.join("artifacts/ranker", os.path.basename(path)))

    mfcache = {r["user_id"]: np.asarray(r["embedding"], dtype=np.float32)
               for r in await conn.fetch("SELECT user_id, embedding FROM user_mf_embeddings")}

    scorers = ["xgb", "mf", "meta", "plot", "mf+meta", "mf+pop"]
    h10 = {s: 0 for s in scorers}
    h50 = {s: 0 for s in scorers}
    retr_n = 0

    for uid in uids:
        uv = build_user_vectors(ctx, train_items[uid], ctx.global_mean)
        uv.mf = mfcache.get(uid)
        relset = rel[uid]
        cand = generate_candidates(ctx, uv, seen[uid])
        if not len(cand):
            continue
        retrieved = relset & {int(ctx.ids[j]) for j in cand}
        if not retrieved:
            continue
        retr_n += 1

        def norm(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

        mf = (np.where(ctx.mf_mask[cand], ctx.mf_item[cand] @ uv.mf, -1e9)
              if uv.mf is not None else np.zeros(len(cand)))
        meta = ctx.meta[cand] @ uv.meta_pos if uv.meta_pos is not None else np.zeros(len(cand))
        plot = ctx.plot[cand] @ uv.plot_pos if uv.plot_pos is not None else np.zeros(len(cand))
        pop = ctx.pop[cand]
        S = {
            "xgb": ranker.predict(build_features(ctx, uv, cand)),
            "mf": mf,
            "meta": meta,
            "plot": plot,
            "mf+meta": norm(mf) + norm(meta),
            "mf+pop": norm(mf) + 0.3 * norm(pop),
        }
        for s, sc in S.items():
            order = np.argsort(sc)[::-1]
            recs = [int(ctx.ids[cand[o]]) for o in order]
            h10[s] += hitk(recs, relset, 10)
            h50[s] += hitk(recs, relset, 50)

    print(f"\nretrieved users: {retr_n}  (hit-rate is CONDITIONAL on retrieval)")
    print(f"{'scorer':10s} {'hit@10|r':>10} {'hit@50|r':>10}")
    for s in scorers:
        print(f"{s:10s} {h10[s]/retr_n:>10.4f} {h50[s]/retr_n:>10.4f}")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
