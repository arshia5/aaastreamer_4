"""Full recommendation training job (v4): LightGCN + multi-channel retrieval +
graded XGBoost ranker + staged evaluation + NDCG@10 rollback.

Steps: advisory lock -> popularity -> chronological split (leave-last-k) ->
LightGCN -> v4 context (with the new MF) -> per-user taste profiles -> XGBoost
(graded labels) -> staged eval -> rollback gate -> on activation: persist
artifacts + MF vectors to DB + refresh recommendations.
"""
from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from tqdm import tqdm

from app.core.config import settings
from app.core.logging_db import log_pg
from app.ml import config
from app.ml.collaborative import save_artifact
from app.ml.evaluation import _ndcg
from app.ml.lightgcn import train_lightgcn
from app.ml.popularity import _RECOMPUTE_SQL
from app.ml.profiles import build_user_vectors
from app.ml.ranker import XgbRanker, build_features, graded_label, save_ranker, train_ranker
from app.ml.reco import build_context_from_conn, generate_candidates, mmr_rerank

log = logging.getLogger("recsys.training")
_LOCK_KEY = 911001


def _dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _active(conn, mtype, field="artifact_path"):
    return await conn.fetchval(
        f"SELECT {field} FROM model_versions WHERE model_type=$1 AND is_active "
        f"ORDER BY created_at DESC LIMIT 1", mtype)


async def run_full_recommendation_training(
    *, job_id=None, triggered_by_user_id=None, like_threshold=7.0,
    max_refresh_users=None, epochs=None, retrain_ranker=True,
) -> dict:
    """retrain_ranker: when True (initial / on-demand) the XGBoost-vs-linear A/B is
    run and the winning ranker is persisted. When False (the nightly job) the
    ranker is NOT retrained — the active ranker is loaded and reused for eval +
    refresh. The ranker's features are all dot-products / cosines / structured
    overlaps, which are invariant to LightGCN's nightly re-basis, so a frozen
    ranker stays valid as the collaborative model is retrained each night."""
    t0 = time.time()
    conn = await asyncpg.connect(_dsn())
    await register_vector(conn)
    if not await conn.fetchval("SELECT pg_try_advisory_lock($1)", _LOCK_KEY):
        await conn.close()
        return {"status": "skipped", "reason": "another training already running"}
    if job_id is None:
        job_id = await conn.fetchval(
            "INSERT INTO recommendation_jobs(job_type, status, started_at, triggered_by_user_id) "
            "VALUES('full_training','running',$1,$2) RETURNING id", _now(), triggered_by_user_id)
    else:
        await conn.execute("UPDATE recommendation_jobs SET status='running', started_at=$1 WHERE id=$2",
                           _now(), job_id)
    await conn.execute(
        "UPDATE recommendation_jobs SET status='failed', finished_at=$1, "
        "error_message='stale: superseded' WHERE job_type='full_training' "
        "AND status IN ('running','queued') AND id <> $2", _now(), job_id)
    log.info("v4 training job %s started", job_id)
    await log_pg(conn, "training_started", user_id=triggered_by_user_id,
                 entity_type="job", entity_id=job_id)
    metrics: dict = {}
    try:
        ts = time.time()
        await conn.execute(_RECOMPUTE_SQL.text)
        metrics["popularity_seconds"] = round(time.time() - ts, 1)

        # refresh community clusters for movies with new reviews (uses stored
        # review vectors, no re-embedding)
        from app.ml.community import rebuild_dirty_community
        ts = time.time()
        metrics["community_movies_rebuilt"] = await rebuild_dirty_community(conn)
        metrics["community_seconds"] = round(time.time() - ts, 1)

        # --- chronological split (leave-last-k = newest 20%, min 1) ------- #
        rows = await conn.fetch(
            "SELECT user_id, movie_id, preference_score, review_date FROM interactions "
            "WHERE preference_score IS NOT NULL ORDER BY user_id, review_date NULLS FIRST")
        per_user: dict[int, list] = {}
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
        metrics["users"] = len(per_user)

        # --- LightGCN ----------------------------------------------------- #
        ts = time.time()
        user_map, item_map = {}, {}
        pu, pi = [], []
        for uid, items in train_items.items():
            for mid, pref, _ in items:
                if pref < config.CF_POS_THRESHOLD:
                    continue
                pu.append(user_map.setdefault(uid, len(user_map)))
                pi.append(item_map.setdefault(mid, len(item_map)))
        kw = {"epochs": epochs} if epochs else {}
        user_emb, item_emb, losses = train_lightgcn(
            np.array(pu), np.array(pi), len(user_map), len(item_map), **kw)
        metrics.update(cf_users=len(user_map), cf_items=len(item_map),
                       cf_edges=len(pu), cf_final_loss=round(losses[-1], 4),
                       cf_train_seconds=round(time.time() - ts, 1))

        # --- v4 context with the NEW mf ---------------------------------- #
        ctx = await build_context_from_conn(conn)
        n = len(ctx.ids)
        mf_item = np.zeros((n, config.LIGHTGCN_DIM), dtype=np.float32)
        mf_mask = np.zeros(n, dtype=bool)
        for mid, row in item_map.items():
            j = ctx.idx.get(mid)
            if j is not None:
                mf_item[j] = item_emb[row]
                mf_mask[j] = True
        ctx.mf_item, ctx.mf_mask = mf_item, mf_mask
        user_mf = {uid: user_emb[row] for uid, row in user_map.items()}

        # --- per-user taste vectors -------------------------------------- #
        ts = time.time()
        uvecs: dict[int, object] = {}
        for uid, items in tqdm(train_items.items(), total=len(train_items),
                               desc="profiles", mininterval=5):
            uv = build_user_vectors(ctx, items, ctx.global_mean)
            uv.mf = user_mf.get(uid)
            uvecs[uid] = uv
        metrics["profiles_seconds"] = round(time.time() - ts, 1)

        # --- rankers (graded) + A/B: XGBoost vs mf-centric linear -------- #
        ts = time.time()
        ranker = None
        ranker_kind = None
        X = None
        if not retrain_ranker:
            # nightly: reuse the frozen active ranker (do not retrain)
            from app.ml.ranker import load_ranker
            rpath = await _active(conn, "xgb_ranker", "artifact_path")
            if rpath:
                try:
                    ranker = load_ranker(rpath)
                    ranker_kind = getattr(ranker, "KIND", "xgboost")
                except Exception:
                    log.warning("could not load active ranker %s; eval/refresh "
                                "will use the hybrid fallback", rpath, exc_info=True)
            metrics["ranker_kind"] = ranker_kind
            metrics["ranker_reused"] = True
            metrics["xgb_train_seconds"] = 0.0
        else:
            X, y, qid = _build_xgb_data(ctx, uvecs, train_items, test_items, like_threshold)
        if retrain_ranker and X is not None and len(set(qid)) >= 50:
            from app.ml.ranker import (LinearRanker, save_linear_ranker,
                                       train_linear_ranker)
            xgb_r = XgbRanker({"model": train_ranker(X, y, qid)})
            lin_r = LinearRanker({"model": train_linear_ranker(X, y, qid)})
            metrics.update(xgb_rows=int(len(y)), xgb_groups=int(len(set(qid))))
            # cheap A/B on a small held-out sample to pick the winner
            ab_n = min(3000, config.EVAL_SAMPLE_USERS)
            ev_x = _evaluate(ctx, xgb_r, uvecs, train_items, test_items,
                             like_threshold, sample_n=ab_n)
            ev_l = _evaluate(ctx, lin_r, uvecs, train_items, test_items,
                             like_threshold, sample_n=ab_n)
            metrics["ranker_ab"] = {
                "sample_users": ab_n,
                "xgboost_ndcg_at_10": ev_x["ndcg_at_10"],
                "linear_ndcg_at_10": ev_l["ndcg_at_10"],
            }
            if ev_l["ndcg_at_10"] > ev_x["ndcg_at_10"]:
                ranker, ranker_kind = lin_r, "linear"
            else:
                ranker, ranker_kind = xgb_r, "xgboost"
        metrics["ranker_kind"] = ranker_kind
        metrics["xgb_train_seconds"] = round(time.time() - ts, 1)

        # --- full staged evaluation on the A/B winner -------------------- #
        ts = time.time()
        ev = _evaluate(ctx, ranker, uvecs, train_items, test_items, like_threshold)
        metrics["eval"] = ev
        metrics["eval_seconds"] = round(time.time() - ts, 1)

        # --- rollback vs active stored ndcg ------------------------------ #
        old_ndcg = await conn.fetchval(
            "SELECT (metrics->>'ndcg_at_10')::float FROM model_versions "
            "WHERE model_type='lightgcn' AND is_active ORDER BY created_at DESC LIMIT 1")
        new_ndcg = ev["ndcg_at_10"]
        activate = old_ndcg is None or new_ndcg >= old_ndcg + config.ROLLBACK_MARGIN
        metrics["decision"] = "activated" if activate else "rolled_back"
        metrics["new_ndcg_at_10"] = new_ndcg
        metrics["old_ndcg_at_10"] = old_ndcg
        await log_pg(conn, "model_activated" if activate else "model_rolled_back",
                     entity_type="model", details={"ndcg_at_10": new_ndcg, "old": old_ndcg})

        if activate:
            cf_version, cf_path = save_artifact(
                user_emb, item_emb, np.zeros(len(item_map), np.float32),
                user_map, item_map, {"model": "lightgcn", "dim": config.LIGHTGCN_DIM})
            # persist a newly trained ranker only when retrain_ranker=True; the
            # nightly job reuses the frozen active ranker and leaves it untouched.
            xgb_path = None
            if retrain_ranker and ranker is not None:
                if ranker_kind == "linear":
                    xgb_version, xgb_path = save_linear_ranker(
                        ranker.model, {"features": len(X[0]), "kind": "linear"})
                else:
                    xgb_version, xgb_path = save_ranker(
                        ranker.model, {"features": len(X[0]), "kind": "xgboost"})
            await conn.execute("UPDATE model_versions SET is_active=false WHERE model_type='lightgcn'")
            await conn.execute(
                "INSERT INTO model_versions(model_type, version_name, artifact_path, is_active, metrics) "
                "VALUES('lightgcn',$1,$2,true,$3)", cf_version, cf_path, json.dumps(ev))
            if xgb_path:
                await conn.execute("UPDATE model_versions SET is_active=false WHERE model_type='xgb_ranker'")
                await conn.execute(
                    "INSERT INTO model_versions(model_type, version_name, artifact_path, is_active, metrics) "
                    "VALUES('xgb_ranker',$1,$2,true,$3)", xgb_version, xgb_path, json.dumps(ev))
            await _store_mf(conn, ctx, item_map, item_emb, user_map, user_emb)
            ts = time.time()
            refreshed = await _refresh_all(conn, ctx, ranker, uvecs, train_items,
                                           test_items, max_refresh_users)
            metrics["recommendations_refreshed"] = refreshed
            metrics["refresh_seconds"] = round(time.time() - ts, 1)
            # rebuild multi-channel similar_movies with the fresh MF/community
            from app.ml.similar import rebuild_similar_hybrid
            ts = time.time()
            await rebuild_similar_hybrid(conn, ctx)
            metrics["similar_seconds"] = round(time.time() - ts, 1)
            await log_pg(conn, "similar_rebuilt", entity_type="job", entity_id=job_id)

        metrics["total_seconds"] = round(time.time() - t0, 1)
        await conn.execute("UPDATE recommendation_jobs SET status='success', finished_at=$1, metrics=$2 WHERE id=$3",
                           _now(), json.dumps(metrics), job_id)
        await log_pg(conn, "training_completed", user_id=triggered_by_user_id,
                     entity_type="job", entity_id=job_id,
                     details={"decision": metrics["decision"], "ndcg_at_10": new_ndcg})
        log.info("v4 job %s done in %.1fs (%s)", job_id, metrics["total_seconds"], metrics["decision"])
        return {"job_id": job_id, "status": "success", "metrics": metrics}
    except Exception as exc:
        log.exception("v4 training job %s failed", job_id)
        await conn.execute("UPDATE recommendation_jobs SET status='failed', finished_at=$1, "
                           "error_message=$2, metrics=$3 WHERE id=$4",
                           _now(), str(exc)[:2000], json.dumps(metrics), job_id)
        return {"job_id": job_id, "status": "failed", "error": str(exc)}
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_KEY)
        await conn.close()


def _rank(ctx, ranker, uv, cand):
    """Return (cand_idx array, score array) ranked by XGB (or hybrid fallback)."""
    if len(cand) == 0:
        return cand, np.array([])
    if ranker is not None:
        scores = ranker.predict(build_features(ctx, uv, cand))
    else:
        # fallback: plot+meta+mf+pop blend
        s = ctx.pop[cand].copy()
        if uv.plot_pos is not None:
            s = s + (ctx.plot[cand] @ uv.plot_pos)
        if uv.mf is not None:
            s = s + np.where(ctx.mf_mask[cand], ctx.mf_item[cand] @ uv.mf, 0)
        scores = s
    return cand, np.asarray(scores, dtype=np.float32)


def _eval_pool(ctx, train_items, test_items, like_threshold, sample_n):
    uids, seen, rel = [], {}, {}
    for uid, test in test_items.items():
        r = {m for m, p, _ in test if p >= like_threshold and m in ctx.idx}
        if not r:
            continue
        uids.append(uid)
        seen[uid] = [ctx.idx[m] for m, _, _ in train_items.get(uid, []) if m in ctx.idx]
        rel[uid] = r
    random.Random(42).shuffle(uids)
    return uids[:sample_n], seen, rel


def _build_xgb_data(ctx, uvecs, train_items, test_items, like_threshold):
    uids, seen, rel = _eval_pool(ctx, train_items, test_items, like_threshold,
                                 config.XGB_TRAIN_USERS)
    pref_of = {uid: {m: p for m, p, _ in test_items[uid]} for uid in uids}
    Xs, ys, qids, g = [], [], [], 0
    rng = random.Random(42)
    for uid in tqdm(uids, desc="xgb-data", mininterval=5):
        uv = uvecs.get(uid)
        if uv is None:
            continue
        cand = generate_candidates(ctx, uv, seen[uid])
        pool = cand.tolist()
        # Label candidates by their held-out preference. We DO NOT force-inject
        # held-out positives that retrieval missed: at serving the ranker only ever
        # scores retrieved candidates, so injecting unretrieved positives (which
        # look like negatives feature-wise, esp. low mf) trains on a distribution
        # that never occurs and teaches the ranker to distrust its best signal.
        # Train only on the retrieved distribution; skip users whose positive
        # wasn't retrieved (nothing to learn to rank for them).
        labels_of = {j: graded_label(pref_of[uid].get(int(ctx.ids[j]))) for j in pool}
        positives = [j for j in pool if labels_of.get(j, 0) > 0]
        if not positives:
            continue
        # cap negatives to XGB_NEG_PER_POS per positive (keeps the matrix small)
        negatives = [j for j in pool if labels_of.get(j, 0) == 0]
        keep_neg = config.XGB_NEG_PER_POS * len(positives)
        if len(negatives) > keep_neg:
            negatives = rng.sample(negatives, keep_neg)
        group = np.array(positives + negatives, dtype=np.int64)
        labels = np.array([labels_of.get(int(j), 0) for j in group])
        Xs.append(build_features(ctx, uv, group))
        ys.append(labels)
        qids.append(np.full(len(group), g))
        g += 1
    if not Xs:
        return None, None, None
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(qids)


def _evaluate(ctx, ranker, uvecs, train_items, test_items, like_threshold, k=10,
              sample_n=None):
    uids, seen, rel = _eval_pool(ctx, train_items, test_items, like_threshold,
                                 sample_n or config.EVAL_SAMPLE_USERS)
    pop_order = np.argsort(ctx.pop)[::-1]
    buckets = {"1": [0, 0], "2-4": [0, 0], "5-10": [0, 0], "10+": [0, 0]}  # [ndcg_sum, n]
    n = retr_hit = 0
    hr = r10 = r50 = ndcg = bhr = 0.0
    for uid in tqdm(uids, desc="eval", mininterval=5):
        uv = uvecs[uid]
        cand = generate_candidates(ctx, uv, seen[uid])
        if len(cand) == 0:
            continue
        retrieved = rel[uid] & {int(ctx.ids[j]) for j in cand}
        ci, sc = _rank(ctx, ranker, uv, cand)
        order = np.argsort(sc)[::-1][:50]
        recs = [int(ctx.ids[ci[o]]) for o in order]
        rels10 = [1 if m in rel[uid] else 0 for m in recs[:k]]
        h10 = sum(rels10)
        n += 1
        retr_hit += 1 if retrieved else 0
        hr += 1.0 if h10 else 0.0
        r10 += h10 / len(rel[uid])
        r50 += sum(1 for m in recs[:50] if m in rel[uid]) / len(rel[uid])
        nd = _ndcg(rels10, len(rel[uid]), k)
        ndcg += nd
        bhr += 1.0 if any(m in rel[uid] for m in
                          [int(ctx.ids[j]) for j in pop_order[:k]]) else 0.0
        nrev = len(train_items.get(uid, []))
        b = "1" if nrev <= 1 else "2-4" if nrev <= 4 else "5-10" if nrev <= 10 else "10+"
        buckets[b][0] += nd
        buckets[b][1] += 1
    a = (lambda x: round(x / n, 4) if n else 0.0)
    return {
        "eval_users": n, "hit_rate_at_10": a(hr), "recall_at_10": a(r10),
        "recall_at_50": a(r50), "ndcg_at_10": a(ndcg),
        "retrieval_recall": a(retr_hit), "baseline_pop_hit_rate_at_10": a(bhr),
        "ndcg_by_reviews": {b: round(s / c, 4) if c else 0.0 for b, (s, c) in buckets.items()},
    }


async def _store_mf(conn, ctx, item_map, item_emb, user_map, user_emb):
    await conn.execute("TRUNCATE movie_mf_embeddings")
    await conn.copy_records_to_table(
        "movie_mf_embeddings",
        records=[(mid, item_emb[row]) for mid, row in item_map.items() if mid in ctx.idx],
        columns=["movie_id", "embedding"])
    await conn.execute("TRUNCATE user_mf_embeddings")
    await conn.copy_records_to_table(
        "user_mf_embeddings",
        records=[(uid, user_emb[row]) for uid, row in user_map.items()],
        columns=["user_id", "embedding"])


async def _refresh_all(conn, ctx, ranker, uvecs, train_items, test_items, max_users):
    seen_map: dict[int, list] = {}
    for uid, mid in await conn.fetch(
        "SELECT user_id, movie_id FROM interactions "
        "UNION SELECT user_id, movie_id FROM user_movie_states"):
        if mid in ctx.idx:
            seen_map.setdefault(uid, []).append(ctx.idx[mid])
    uids = list(uvecs.keys())
    if max_users is not None:
        uids = uids[:max_users]
    batch_uids, records, done = [], [], 0
    pbar = tqdm(total=len(uids), desc="refresh", mininterval=5)

    async def flush():
        if not batch_uids:
            return
        async with conn.transaction():
            await conn.execute("DELETE FROM user_recommendations WHERE user_id = ANY($1::int[])", batch_uids)
            if records:
                await conn.copy_records_to_table(
                    "user_recommendations", records=records,
                    columns=["user_id", "movie_id", "score", "rank",
                             "content_score", "collaborative_score", "popularity_score"])

    for uid in uids:
        uv = uvecs[uid]
        cand = generate_candidates(ctx, uv, seen_map.get(uid, []))
        if len(cand) == 0:
            batch_uids.append(uid)
        else:
            ci, sc = _rank(ctx, ranker, uv, cand)
            div = mmr_rerank(ctx, ci, sc, config.TOP_N_RECOMMENDATIONS)
            for rank, (j, score) in enumerate(div, 1):
                pj = float(ctx.pop[j])
                records.append((uid, int(ctx.ids[j]), float(score), rank, None, None, pj))
            batch_uids.append(uid)
        done += 1
        pbar.update(1)
        if len(batch_uids) >= 1500:
            await flush(); batch_uids, records = [], []
    await flush()
    pbar.close()
    return done
