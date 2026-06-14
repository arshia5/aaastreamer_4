"""Full recommendation training job (nightly or admin-triggered).

Pipeline:
  1. acquire advisory lock (no concurrent trainings)
  2. recompute movie_popularity_stats
  3. chronological split (hold out each user's newest review)
  4. recompute user_embeddings from the train split
  5. train collaborative MF on train positives -> save artifact (atomic)
  6. train XGBoost ranker on the train split
  7. evaluate the NEW pipeline and the CURRENT ACTIVE pipeline on the same
     held-out reviews; activate the new models only if NDCG@10 improves (rollback)
  8. if activated, refresh user_recommendations (XGB rerank + diversity)
  9. record status + metrics in recommendation_jobs

Heavy; runs outside the request lifecycle. Real-time partial_fit only mutates the
in-memory active model and never writes artifacts, so it can't corrupt training.
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

from app.core.config import settings
from app.ml import config
from app.ml.collaborative import CollaborativeModel, save_artifact, train_bpr
from app.ml.evaluation import _ndcg
from app.ml.hybrid import RecoContext, _minmax, iter_user_candidates, mmr_rerank
from app.ml.ranker import XgbRanker, build_features, save_ranker, train_ranker
from app.ml.user_embedding import compute_user_vector

log = logging.getLogger("recsys.training")
_LOCK_KEY = 911001


def _dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _l2norm_rows(mat):
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return mat / n


async def _active_path(conn, model_type):
    return await conn.fetchval(
        "SELECT artifact_path FROM model_versions "
        "WHERE model_type=$1 AND is_active ORDER BY created_at DESC LIMIT 1",
        model_type,
    )


async def run_full_recommendation_training(
    *,
    job_id: int | None = None,
    triggered_by_user_id: int | None = None,
    like_threshold: float = 7.0,
    max_refresh_users: int | None = None,
    epochs: int = config.CF_EPOCHS,
) -> dict:
    t0 = time.time()
    conn = await asyncpg.connect(_dsn())
    await register_vector(conn)
    if not await conn.fetchval("SELECT pg_try_advisory_lock($1)", _LOCK_KEY):
        await conn.close()
        return {"status": "skipped", "reason": "another training already running"}

    if job_id is None:
        job_id = await conn.fetchval(
            "INSERT INTO recommendation_jobs(job_type, status, started_at, triggered_by_user_id) "
            "VALUES('full_training','running',$1,$2) RETURNING id",
            _now(), triggered_by_user_id,
        )
    else:
        await conn.execute(
            "UPDATE recommendation_jobs SET status='running', started_at=$1 WHERE id=$2",
            _now(), job_id,
        )
    # We hold the advisory lock, so any other running/queued training is stale
    # (its process died without cleanup). Mark those failed.
    await conn.execute(
        "UPDATE recommendation_jobs SET status='failed', finished_at=$1, "
        "error_message='stale: superseded (process died before completion)' "
        "WHERE job_type='full_training' AND status IN ('running','queued') AND id <> $2",
        _now(), job_id,
    )
    log.info("Full training job %s started", job_id)
    metrics: dict = {}
    try:
        # --- arrays ------------------------------------------------------ #
        ts = time.time()
        from app.ml.popularity import _RECOMPUTE_SQL
        await conn.execute(_RECOMPUTE_SQL.text)
        metrics["popularity_seconds"] = round(time.time() - ts, 1)

        erows = await conn.fetch("SELECT movie_id, embedding FROM movie_embeddings")
        ids = np.array([r["movie_id"] for r in erows], dtype=np.int64)
        mat = np.array([np.asarray(r["embedding"], dtype=np.float32) for r in erows],
                       dtype=np.float32)
        idx = {int(m): i for i, m in enumerate(ids)}
        emb_lookup = {int(m): mat[i] for i, m in enumerate(ids)}
        content_unit = _l2norm_rows(mat)
        pop = np.zeros(len(ids), dtype=np.float32)
        rc = np.zeros(len(ids), dtype=np.float32)
        ap = np.full(len(ids), 5.0, dtype=np.float32)
        for mid, p, cnt, avgp in await conn.fetch(
            "SELECT movie_id, popularity_score, review_count, avg_preference_score "
            "FROM movie_popularity_stats"
        ):
            if mid in idx:
                j = idx[mid]
                pop[j] = p or 0.0
                rc[j] = cnt or 0
                ap[j] = avgp if avgp is not None else 5.0
        # Bayesian-shrink avg preference for the ranker feature (matches the
        # popularity_score shrinkage): (PRIOR*global_mean + sum) / (PRIOR + n).
        m_global = float(np.nansum(ap * rc) / max(float(np.nansum(rc)), 1.0))
        C = config.POPULARITY_PRIOR
        ap = ((C * m_global + ap * rc) / (C + rc)).astype(np.float32)
        genres = [set() for _ in ids]
        for mid, gid in await conn.fetch("SELECT movie_id, genre_id FROM movie_genres"):
            if mid in idx:
                genres[idx[mid]].add(gid)
        # year + actor sets (for ranker overlap/distance features)
        year = np.full(len(ids), np.nan, dtype=np.float32)
        for mid, yr in await conn.fetch("SELECT id, year FROM movies WHERE year IS NOT NULL"):
            if mid in idx:
                year[idx[mid]] = yr
        actors = [set() for _ in ids]
        for mid, pid in await conn.fetch(
            "SELECT mp.movie_id, mp.person_id FROM movie_people mp "
            "JOIN roles r ON r.id = mp.role_id WHERE r.name = 'actor'"
        ):
            if mid in idx:
                actors[idx[mid]].add(pid)

        # --- chronological split ---------------------------------------- #
        rows = await conn.fetch(
            "SELECT user_id, movie_id, preference_score, review_date "
            "FROM interactions WHERE preference_score IS NOT NULL "
            "ORDER BY user_id, review_date NULLS FIRST"
        )
        per_user: dict[int, list] = {}
        for r in rows:
            per_user.setdefault(r["user_id"], []).append(
                (r["movie_id"], float(r["preference_score"]), r["review_date"])
            )
        train_items, test_items = {}, {}
        for uid, items in per_user.items():
            if len(items) >= 2:
                train_items[uid] = items[:-1]
                test_items[uid] = [items[-1]]
            else:
                train_items[uid] = items
        metrics["users"] = len(per_user)

        # per-user stats (for ranker features)
        user_avg_pref, user_n_pos = {}, {}
        for uid, items in train_items.items():
            prefs = [p for _, p, _ in items]
            user_avg_pref[uid] = float(np.mean(prefs)) if prefs else 5.0
            user_n_pos[uid] = sum(1 for p in prefs if p >= config.CF_POS_THRESHOLD)

        # --- user embeddings (train split) ------------------------------ #
        ts = time.time()
        user_unit: dict[int, np.ndarray] = {}
        ue_records = []
        for uid, items in train_items.items():
            vec = compute_user_vector(items, emb_lookup)
            if vec is not None and np.linalg.norm(vec) > 0:
                user_unit[uid] = vec
                ue_records.append((uid, vec))
        await conn.execute("DROP TABLE IF EXISTS _stg_ue")
        await conn.execute("CREATE TEMP TABLE _stg_ue(user_id int, embedding vector(%d))"
                           % config.N_COMPONENTS)
        await conn.copy_records_to_table("_stg_ue", records=ue_records,
                                         columns=["user_id", "embedding"])
        await conn.execute(
            "INSERT INTO user_embeddings(user_id, embedding) "
            "SELECT user_id, embedding FROM _stg_ue "
            "ON CONFLICT (user_id) DO UPDATE SET embedding=EXCLUDED.embedding, updated_at=now()")
        await conn.execute("DROP TABLE _stg_ue")
        metrics["user_embeddings"] = len(ue_records)
        metrics["user_embed_seconds"] = round(time.time() - ts, 1)

        # --- train collaborative MF ------------------------------------- #
        ts = time.time()
        user_map, item_map = {}, {}
        u_list, i_list, w_list = [], [], []
        for uid, items in train_items.items():
            for mid, pref, _ in items:
                if pref < config.CF_POS_THRESHOLD or mid not in idx:
                    continue
                ui = user_map.setdefault(uid, len(user_map))
                ii = item_map.setdefault(mid, len(item_map))
                u_list.append(ui); i_list.append(ii); w_list.append(pref / 10.0)
        uf, itf, ib, losses = train_bpr(
            np.array(u_list), np.array(i_list), np.array(w_list, dtype=np.float32),
            len(user_map), len(item_map), epochs=epochs)
        metrics.update(cf_positives=len(u_list), cf_users=len(user_map),
                       cf_items=len(item_map),
                       cf_final_loss=round(losses[-1], 4) if losses else None,
                       cf_train_seconds=round(time.time() - ts, 1))
        cf_version, cf_path = save_artifact(
            uf, itf, ib, user_map, item_map,
            {"dim": config.CF_DIM, "epochs": epochs})
        new_collab = CollaborativeModel(
            {"user_factors": uf, "item_factors": itf, "item_bias": ib,
             "user_map": user_map, "item_map": item_map, "meta": {}}, path=cf_path)
        new_ctx = RecoContext(ids, content_unit, idx, pop, new_collab,
                              review_count=rc, avg_pref=ap, genres=genres,
                              year=year, actors=actors, active_version=cf_version)

        # per-user taste profiles for ranker overlap features
        from app.ml.ranker import build_user_profile
        user_profiles = {}
        for uid, items in train_items.items():
            liked = [idx[mid] for mid, pref, _ in items
                     if pref >= config.CF_POS_THRESHOLD and mid in idx]
            user_profiles[uid] = build_user_profile(new_ctx, liked)

        # current active models (for rollback comparison) — loaded BEFORE activation
        old_cf_path = await _active_path(conn, "collaborative_mf")
        old_xgb_path = await _active_path(conn, "xgb_ranker")
        old_collab = CollaborativeModel.load(old_cf_path) if old_cf_path else None
        old_ranker = XgbRanker.load(old_xgb_path) if old_xgb_path else None
        old_ctx = (RecoContext(ids, content_unit, idx, pop, old_collab,
                               review_count=rc, avg_pref=ap, genres=genres,
                               year=year, actors=actors)
                   if old_collab is not None else None)

        # insert new model_versions rows (inactive until rollback check passes)
        cf_mv = await conn.fetchval(
            "INSERT INTO model_versions(model_type, version_name, artifact_path, is_active) "
            "VALUES('collaborative_mf',$1,$2,false) RETURNING id", cf_version, cf_path)

        # --- train XGBoost ranker --------------------------------------- #
        ts = time.time()
        X, y, qid = _build_xgb_data(new_ctx, user_unit, train_items, test_items,
                                    user_avg_pref, user_n_pos, user_profiles,
                                    like_threshold, config.XGB_TRAIN_USERS)
        new_ranker = None
        xgb_version = xgb_path = None
        xgb_mv = None
        if X is not None and len(set(qid)) >= 50:
            model = train_ranker(X, y, qid)
            new_ranker = XgbRanker({"model": model})
            xgb_version, xgb_path = save_ranker(model, {"rows": int(len(y))})
            xgb_mv = await conn.fetchval(
                "INSERT INTO model_versions(model_type, version_name, artifact_path, is_active) "
                "VALUES('xgb_ranker',$1,$2,false) RETURNING id", xgb_version, xgb_path)
            metrics["xgb_rows"] = int(len(y))
            metrics["xgb_groups"] = int(len(set(qid)))
        metrics["xgb_train_seconds"] = round(time.time() - ts, 1)

        # --- evaluate new vs old (same held-out) ------------------------ #
        ts = time.time()
        new_metrics = _evaluate(new_ctx, new_ranker, user_unit, train_items,
                                test_items, user_avg_pref, user_n_pos,
                                user_profiles, like_threshold)
        old_metrics = (_evaluate(old_ctx, old_ranker, user_unit, train_items,
                                 test_items, user_avg_pref, user_n_pos,
                                 user_profiles, like_threshold)
                       if old_ctx is not None else None)
        metrics["eval_seconds"] = round(time.time() - ts, 1)
        metrics["new_model"] = new_metrics
        metrics["old_model"] = old_metrics

        # --- rollback decision ------------------------------------------ #
        m = config.ROLLBACK_METRIC
        if old_metrics is None:
            activate = True
        else:
            activate = new_metrics[m] >= old_metrics[m] + config.ROLLBACK_MARGIN
        metrics["rollback_metric"] = m
        metrics["decision"] = "activated" if activate else "rolled_back"
        log.info("Job %s decision=%s (new %s=%.4f vs old %s)", job_id,
                 metrics["decision"], m, new_metrics[m],
                 f"{old_metrics[m]:.4f}" if old_metrics else "none")

        if activate:
            await conn.execute("UPDATE model_versions SET is_active=false "
                               "WHERE model_type='collaborative_mf'")
            await conn.execute("UPDATE model_versions SET is_active=true WHERE id=$1", cf_mv)
            await conn.execute("UPDATE model_versions SET metrics=$1 WHERE id=$2",
                               json.dumps(new_metrics), cf_mv)
            if xgb_mv is not None:
                await conn.execute("UPDATE model_versions SET is_active=false "
                                   "WHERE model_type='xgb_ranker'")
                await conn.execute("UPDATE model_versions SET is_active=true, metrics=$1 "
                                   "WHERE id=$2", json.dumps(new_metrics), xgb_mv)
            ts = time.time()
            refreshed = await _refresh_all(conn, new_ctx, new_ranker, user_unit,
                                           user_avg_pref, user_n_pos, user_profiles,
                                           max_refresh_users)
            metrics["recommendations_refreshed"] = refreshed
            metrics["refresh_seconds"] = round(time.time() - ts, 1)

            # Rebuild hybrid similar_movies (content-weighted + MF blend) now that
            # the new collaborative item factors are active.
            ts = time.time()
            from app.ml.similar import rebuild_similar_hybrid
            await rebuild_similar_hybrid(conn, ids, content_unit, idx, new_collab)
            metrics["similar_rebuild_seconds"] = round(time.time() - ts, 1)
        else:
            # keep old models active; new artifacts retained but inactive
            metrics["recommendations_refreshed"] = 0

        metrics["total_seconds"] = round(time.time() - t0, 1)
        await conn.execute(
            "UPDATE recommendation_jobs SET status='success', finished_at=$1, metrics=$2 "
            "WHERE id=$3", _now(), json.dumps(metrics), job_id)
        log.info("Job %s succeeded in %.1fs (%s)", job_id,
                 metrics["total_seconds"], metrics["decision"])
        return {"job_id": job_id, "status": "success", "metrics": metrics}
    except Exception as exc:
        log.exception("Full training job %s failed", job_id)
        await conn.execute(
            "UPDATE recommendation_jobs SET status='failed', finished_at=$1, "
            "error_message=$2, metrics=$3 WHERE id=$4",
            _now(), str(exc)[:2000], json.dumps(metrics), job_id)
        return {"job_id": job_id, "status": "failed", "error": str(exc)}
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_KEY)
        await conn.close()


def _rerank(ctx, ranker, user_unit, uid, cands, user_avg_pref, user_n_pos, profiles):
    """Apply XGB ranker to reorder candidates (descending by predicted score)."""
    if ranker is None or not cands:
        return cands
    X = build_features(ctx, user_unit.get(uid), uid, cands,
                       user_avg_pref.get(uid, 5.0), user_n_pos.get(uid, 0),
                       profile=profiles.get(uid))
    # Skip an incompatible ranker (e.g. an older model trained on fewer features);
    # that side is then evaluated as hybrid-only.
    if getattr(ranker, "features", None) and len(ranker.features) != X.shape[1]:
        return cands
    scores = ranker.predict(X)
    order = np.argsort(scores)[::-1]
    return [(*cands[i][:1], float(scores[i]), *cands[i][2:]) for i in order]


def _eval_user_pool(ctx, train_items, test_items, like_threshold, user_unit, sample_n):
    eval_uids, seen_map, relevant = [], {}, {}
    for uid, test in test_items.items():
        if uid not in user_unit:
            continue
        rel = {m for m, pref, _ in test if pref >= like_threshold and m in ctx.idx}
        if not rel:
            continue
        eval_uids.append(uid)
        seen_map[uid] = [ctx.idx[m] for m, _, _ in train_items.get(uid, []) if m in ctx.idx]
        relevant[uid] = rel
    rng = random.Random(42)
    rng.shuffle(eval_uids)
    eval_uids = eval_uids[:sample_n]
    return eval_uids, seen_map, relevant


def _evaluate(ctx, ranker, user_unit, train_items, test_items,
              user_avg_pref, user_n_pos, profiles, like_threshold, k=10):
    eval_uids, seen_map, relevant = _eval_user_pool(
        ctx, train_items, test_items, like_threshold, user_unit,
        config.EVAL_SAMPLE_USERS)
    pop_order = np.argsort(ctx.pop)[::-1]
    n = 0
    hr = prec = r10 = r50 = ndcg = bhr = 0.0
    for uid, cands in iter_user_candidates(ctx, eval_uids, user_unit, seen_map):
        rel = relevant[uid]
        cands = _rerank(ctx, ranker, user_unit, uid, cands, user_avg_pref,
                        user_n_pos, profiles)
        div = mmr_rerank(ctx, cands, 50)
        rec = [int(ctx.ids[c[0]]) for c in div]
        rels10 = [1 if m in rel else 0 for m in rec[:k]]
        hits10 = sum(rels10)
        hits50 = sum(1 for m in rec[:50] if m in rel)
        n += 1
        hr += 1.0 if hits10 else 0.0
        prec += hits10 / k
        r10 += hits10 / len(rel)
        r50 += hits50 / len(rel)
        ndcg += _ndcg(rels10, len(rel), k)
        seen_set = set(seen_map[uid])
        brec = [int(ctx.ids[j]) for j in pop_order if j not in seen_set][:k]
        bhr += 1.0 if any(m in rel for m in brec) else 0.0
    a = (lambda x: x / n if n else 0.0)
    return {"eval_users": n, "hit_rate_at_10": a(hr), "precision_at_10": a(prec),
            "recall_at_10": a(r10), "recall_at_50": a(r50), "ndcg_at_10": a(ndcg),
            "baseline_pop_hit_rate_at_10": a(bhr)}


def _build_xgb_data(ctx, user_unit, train_items, test_items,
                    user_avg_pref, user_n_pos, profiles, like_threshold, sample_n):
    uids, seen_map, relevant = _eval_user_pool(
        ctx, train_items, test_items, like_threshold, user_unit, sample_n)
    if not uids:
        return None, None, None
    X_parts, y_parts, qid_parts = [], [], []
    g = 0
    for uid, cands in iter_user_candidates(ctx, uids, user_unit, seen_map):
        rel = relevant[uid]
        labels = np.array([1 if int(ctx.ids[c[0]]) in rel else 0 for c in cands])
        if labels.sum() == 0:          # no retrieved positive -> no ranking signal
            continue
        feats = build_features(ctx, user_unit.get(uid), uid, cands,
                               user_avg_pref.get(uid, 5.0), user_n_pos.get(uid, 0),
                               profile=profiles.get(uid))
        X_parts.append(feats)
        y_parts.append(labels)
        qid_parts.append(np.full(len(cands), g))
        g += 1
    if not X_parts:
        return None, None, None
    return (np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(qid_parts))


async def _refresh_all(conn, ctx, ranker, user_unit, user_avg_pref, user_n_pos,
                       profiles, max_users):
    # production "seen" = reviewed OR watched/watchlist
    seen_map: dict[int, list] = {}
    for uid, mid in await conn.fetch(
        "SELECT user_id, movie_id FROM interactions "
        "UNION SELECT user_id, movie_id FROM user_movie_states"
    ):
        if mid in ctx.idx:
            seen_map.setdefault(uid, []).append(ctx.idx[mid])

    uids = list(user_unit.keys())
    if max_users is not None:
        uids = uids[:max_users]

    # No global TRUNCATE: replace each batch of users' recommendations inside a
    # short transaction, so the website always sees a complete set for every user
    # (either their old or their new list) while training runs.
    batch_uids: list[int] = []
    records: list[tuple] = []
    done = 0

    async def flush():
        if not batch_uids:
            return
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM user_recommendations WHERE user_id = ANY($1::int[])",
                batch_uids)
            if records:
                await conn.copy_records_to_table(
                    "user_recommendations", records=records,
                    columns=["user_id", "movie_id", "score", "rank",
                             "content_score", "collaborative_score", "popularity_score"])

    for uid, cands in iter_user_candidates(ctx, uids, user_unit, seen_map):
        cands = _rerank(ctx, ranker, user_unit, uid, cands, user_avg_pref,
                        user_n_pos, profiles)
        div = mmr_rerank(ctx, cands, config.TOP_N_RECOMMENDATIONS)
        for rank, (j, score, c, cf, pp) in enumerate(div, start=1):
            records.append((uid, int(ctx.ids[j]), float(score), rank, c, cf, pp))
        batch_uids.append(uid)
        done += 1
        if len(batch_uids) >= 2000:
            await flush()
            batch_uids.clear(); records.clear()
        if done % 10000 == 0:
            log.info("  refreshed %d/%d users", done, len(uids))
    await flush()
    return done
