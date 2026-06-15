"""Diagnostic: per-channel retrieval recall + ranker lift, on the held-out split.

Reuses the live training context + split logic so numbers are comparable to the
stored job metrics. Run: python -m scripts.diagnose_channels
"""
import asyncio
import random

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml import config
from app.ml.profiles import build_user_vectors
from app.ml.ranker import XgbRanker, build_features
from app.ml.reco import build_context_from_conn, generate_candidates, _topk

SAMPLE = 3000
LIKE = 7.0


def dsn():
    return settings.database_url.replace("+asyncpg", "")


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

    # eval pool: users with a liked held-out item that exists in the index
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

    # load active ranker
    path = await conn.fetchval(
        "SELECT artifact_path FROM model_versions WHERE model_type='xgb_ranker' "
        "AND is_active ORDER BY created_at DESC LIMIT 1")
    import glob, os
    local = os.path.join("artifacts/ranker", os.path.basename(path))
    ranker = XgbRanker.load(local) if os.path.exists(local) else None
    # skip the stale ranker if its feature count no longer matches build_features
    if ranker is not None:
        from app.ml.ranker import FEATURE_NAMES
        try:
            nfeat = ranker.model.get_booster().num_features()
        except Exception:
            nfeat = len(FEATURE_NAMES)
        if nfeat != len(FEATURE_NAMES):
            print(f"ranker feature mismatch ({nfeat} vs {len(FEATURE_NAMES)}); "
                  "skipping ranker-conversion section")
            ranker = None
    print(f"ranker: {local} loaded={ranker is not None}")

    CH = ["plot", "meta", "struct", "mf", "comm", "pop", "union"]
    KS = {"plot": config.RETR_PLOT, "meta": config.RETR_META,
          "struct": config.RETR_STRUCT, "mf": config.RETR_MF,
          "comm": config.RETR_COMM, "pop": config.RETR_POP}
    hit = {c: 0 for c in CH}
    # ranker lift
    n = retr_n = top10 = top50 = 0

    for uid in uids:
        uv = build_user_vectors(ctx, train_items[uid], ctx.global_mean)
        uv.mf = None
        # pull mf from stored user vec
        mfrow = await conn.fetchval(
            "SELECT embedding FROM user_mf_embeddings WHERE user_id=$1", uid)
        if mfrow is not None:
            uv.mf = np.asarray(mfrow, dtype=np.float32)
        relset = rel[uid]
        seenset = set(seen[uid])

        def bucket(scores, k, mask=None):
            b = _topk(scores, k, mask)
            b = np.setdiff1d(b, np.array(seen[uid], dtype=np.int64)) if seen[uid] else b
            return {int(ctx.ids[j]) for j in b}

        if uv.plot_pos is not None:
            hit["plot"] += 1 if (bucket(ctx.plot @ uv.plot_pos, KS["plot"]) & relset) else 0
        if uv.meta_pos is not None:
            hit["meta"] += 1 if (bucket(ctx.meta @ uv.meta_pos, KS["meta"]) & relset) else 0
        if uv.struct_pos is not None and ctx.struct is not None:
            hit["struct"] += 1 if (bucket(ctx.struct @ uv.struct_pos, KS["struct"], ctx.struct_mask) & relset) else 0
        if uv.mf is not None and ctx.mf_mask.any():
            hit["mf"] += 1 if (bucket(ctx.mf_item @ uv.mf, KS["mf"], ctx.mf_mask) & relset) else 0
        if uv.comm_pos is not None and ctx.comm_mask.any():
            hit["comm"] += 1 if (bucket(ctx.comm_mean @ uv.comm_pos, KS["comm"], ctx.comm_mask) & relset) else 0
        hit["pop"] += 1 if (bucket(ctx.pop.copy(), KS["pop"]) & relset) else 0

        cand = generate_candidates(ctx, uv, seen[uid])
        n += 1
        retrieved = relset & {int(ctx.ids[j]) for j in cand}
        hit["union"] += 1 if retrieved else 0
        if retrieved and ranker is not None and len(cand):
            sc = ranker.predict(build_features(ctx, uv, cand))
            order = np.argsort(sc)[::-1]
            recs = [int(ctx.ids[cand[o]]) for o in order]
            retr_n += 1
            top10 += 1 if any(m in relset for m in recs[:10]) else 0
            top50 += 1 if any(m in relset for m in recs[:50]) else 0

    print(f"\nusers evaluated: {n}")
    print("--- per-channel retrieval recall (channel ALONE contains a held-out positive) ---")
    for c in CH:
        print(f"  {c:6s} k={KS.get(c,'-'):>4}  recall={hit[c]/n:.4f}")
    print("\n--- ranker conversion (only among users whose positive WAS retrieved) ---")
    print(f"  retrieved users: {retr_n}")
    if retr_n:
        print(f"  ranker hit@10 | retrieved = {top10/retr_n:.4f}")
        print(f"  ranker hit@50 | retrieved = {top50/retr_n:.4f}")
        print(f"  => of retrieved positives, {top50/retr_n*100:.1f}% reach top-50, "
              f"{top10/retr_n*100:.1f}% reach top-10")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
