"""Metadata representation bake-off: v3 structured blocks vs v4 mpnet template.

Builds user positive-profiles identically in each space (preference-centered,
recency-weighted) so ONLY the movie representation differs, then compares
retrieval recall @K. Run: python -m scripts.diagnose_metadata
"""
import asyncio
import random
from collections import Counter
from datetime import datetime

import asyncpg
import joblib
import numpy as np
from pgvector.asyncpg import register_vector

from app.core.config import settings
from app.ml import config
from app.ml.features import genre_block, hashed_block, year_block, l2_normalize
from app.ml.reco import build_context_from_conn

SAMPLE = 3000
LIKE = 7.0
KS = [300, 1000]
_MIN_DT = datetime.min
BASELINE_PRIOR = 5
GAMMA = config.USER_EMBED_DECAY


def dsn():
    return settings.database_url.replace("+asyncpg", "")


def unit_rows(M):
    return l2_normalize(M.astype(np.float32))


async def fetch_records(conn, ids):
    """Return per-movie dict of token lists in ctx.ids order."""
    idx = {int(m): i for i, m in enumerate(ids)}
    n = len(ids)
    rec = {k: [[] for _ in range(n)] for k in
           ["genre", "director", "writer", "actor", "language", "country"]}
    year = [None] * n
    for mid, y in await conn.fetch("SELECT id, year FROM movies WHERE year IS NOT NULL"):
        j = idx.get(mid)
        if j is not None:
            year[j] = y
    gname = {r["id"]: r["name"] for r in await conn.fetch("SELECT id, name FROM genres")}
    for mid, gid in await conn.fetch("SELECT movie_id, genre_id FROM movie_genres"):
        j = idx.get(mid)
        if j is not None:
            rec["genre"][j].append(gname.get(gid, ""))
    pname = {r["id"]: r["name"] for r in await conn.fetch("SELECT id, name FROM people")}
    role = {r["id"]: r["name"] for r in await conn.fetch("SELECT id, name FROM roles")}
    rolemap = {"director": "director", "writer": "writer", "actor": "actor"}
    for mid, pid, rid in await conn.fetch("SELECT movie_id, person_id, role_id FROM movie_people"):
        j = idx.get(mid)
        key = rolemap.get(role.get(rid, ""))
        if j is not None and key:
            rec[key][j].append(pname.get(pid, ""))
    lname = {r["id"]: r["name"] for r in await conn.fetch("SELECT id, name FROM languages")}
    for mid, lid in await conn.fetch("SELECT movie_id, language_id FROM movie_languages"):
        j = idx.get(mid)
        if j is not None:
            rec["language"][j].append(lname.get(lid, ""))
    cname = {r["id"]: r["name"] for r in await conn.fetch("SELECT id, name FROM countries")}
    for mid, cid in await conn.fetch("SELECT movie_id, country_id FROM movie_countries"):
        j = idx.get(mid)
        if j is not None:
            rec["country"][j].append(cname.get(cid, ""))
    return rec, year


def build_structured(rec, year, art):
    """v3 structured blocks WITHOUT the plot block; L2-norm per block * weight, concat."""
    mlb, scaler, med = art["genre_mlb"], art["year_scaler"], art["median_year"]
    blocks = {
        "genre": genre_block(rec["genre"], mlb),
        "director": hashed_block(rec["director"], config.DIRECTOR_DIM),
        "writer": hashed_block(rec["writer"], config.WRITER_DIM),
        "actor": hashed_block(rec["actor"], config.ACTOR_DIM),
        "language": hashed_block(rec["language"], config.LANG_DIM),
        "country": hashed_block(rec["country"], config.COUNTRY_DIM),
        "numeric": year_block(year, scaler, med),
    }
    order = ["genre", "director", "writer", "actor", "language", "country", "numeric"]
    parts = []
    for name in order:
        m = blocks[name]
        w = config.WEIGHTS[name]
        parts.append((m * w) if name == "numeric" else (l2_normalize(m) * w))
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_profiles(M, ctx, train_items, uids, global_mean):
    """positive profile per user in space M (unit rows). Mirrors profiles.py centering."""
    profs = {}
    for uid in uids:
        rows = [(m, p, d) for (m, p, d) in train_items[uid] if m in ctx.idx and p is not None]
        if not rows:
            continue
        rows.sort(key=lambda x: x[2] or _MIN_DT)
        prefs = [p for _, p, _ in rows]
        base = (BASELINE_PRIOR * global_mean + sum(prefs)) / (BASELINE_PRIOR + len(prefs))
        acc = np.zeros(M.shape[1], dtype=np.float32)
        for r, (mid, pref, _) in enumerate(reversed(rows)):
            c = (pref - base) * (GAMMA ** r)
            if c > 0:
                acc += c * M[ctx.idx[mid]]
        nrm = np.linalg.norm(acc)
        if nrm > 0:
            profs[uid] = (acc / nrm).astype(np.float32)
    return profs


def recall_at(M, profs, ctx, seen, rel, uids, ks):
    M = M  # unit rows already
    hits = {k: 0 for k in ks}
    n = 0
    maxk = max(ks)
    for uid in uids:
        p = profs.get(uid)
        if p is None:
            continue
        s = M @ p
        if seen[uid]:
            s[np.array(seen[uid])] = -1e9
        top = np.argpartition(s, -maxk)[-maxk:]
        top = top[np.argsort(s[top])[::-1]]
        recs = [int(ctx.ids[j]) for j in top]
        n += 1
        for k in ks:
            if rel[uid] & set(recs[:k]):
                hits[k] += 1
    return {k: hits[k] / n for k in ks}, n


async def main():
    conn = await asyncpg.connect(dsn())
    await register_vector(conn)
    ctx = await build_context_from_conn(conn)
    art = joblib.load(config.PIPELINE_PATH)

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

    # --- representations (unit rows) --- #
    meta_mpnet = unit_rows(ctx.meta.copy())          # already unit, but be safe
    rec, year = await fetch_records(conn, ctx.ids)
    struct = build_structured(rec, year, art)
    meta_struct = unit_rows(struct)
    # legacy 390-d (includes plot) for reference
    leg = np.zeros((len(ctx.ids), 390), dtype=np.float32)
    for mid, emb in await conn.fetch("SELECT movie_id, embedding FROM movie_embeddings"):
        j = ctx.idx.get(mid)
        if j is not None:
            leg[j] = emb
    meta_legacy = unit_rows(leg)
    # concat mpnet + struct (both unit, then renormalize)
    cat = unit_rows(np.concatenate([meta_mpnet, meta_struct], axis=1))

    reps = {
        "mpnet (v4 current)": meta_mpnet,
        "structured (v3, no plot)": meta_struct,
        "legacy 390 (v3 plot+meta)": meta_legacy,
        "mpnet + structured": cat,
    }
    gm = ctx.global_mean
    print(f"users: {len(uids)}  K={KS}\n")
    print(f"{'representation':28s} " + "  ".join(f'recall@{k}' for k in KS))
    for name, M in reps.items():
        profs = build_profiles(M, ctx, train_items, uids, gm)
        r, n = recall_at(M, profs, ctx, seen, rel, uids, KS)
        print(f"{name:28s} " + "  ".join(f'{r[k]:9.4f}' for k in KS))
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
