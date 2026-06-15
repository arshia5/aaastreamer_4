# The aaastreamer Recommender — v5

Every signal, weight, and hyperparameter in the recommendation system, plus how
training and serving fit together. File references throughout.

---

## v5 changelog (what changed from v4 and why)

v4 regressed against v3 (lift over the popularity baseline fell from ~4.1× to
~1.5×). A data-driven diagnosis (`scripts/diagnose_*.py`) found the **ranker**,
not retrieval, was the defect — sorting by the raw LightGCN score alone beat the
trained 23-feature XGBoost ranker. Root causes & fixes:

1. **LightGCN was undertrained** — 30 full-batch epochs = 30 gradient steps.
   Bake-off: 30→**400 epochs** lifts mf retrieval recall@300 0.415→0.503 and
   mf-as-scorer hit@10 +30%. (`LIGHTGCN_EPOCHS`.)
2. **Ranker trained on a distribution it never sees at serving** — `_build_xgb_data`
   force-injected held-out positives that retrieval *missed* (low-mf, look like
   negatives), teaching the ranker to distrust its strongest feature. Now it
   trains **only on the retrieved candidate distribution**.
3. **Retrieval budgets were backwards** — plot (recall 0.10) had 400 slots, mf
   (0.40) had 300. Rebalanced: mf 500, plot 150. Union recall 0.535→0.600.
4. **New structured-metadata channel** — the v3 block embedding (genre/cast/year),
   no plot block, added *alongside* mpnet (not replacing it). mpnet+structured
   beats either alone (0.204 vs 0.180 recall@300). (`app/ml/struct_embed.py`.)
5. **Ranker A/B** — every full ranker train fits both XGBoost and a mf-centric
   linear (Ridge) scorer and ships whichever wins NDCG@10 on the held-out split.
6. **Cadence split** — the ranker's features are invariant to LightGCN's nightly
   re-basis, so it is trained **once / occasionally**; nightly runs use
   `--skip-ranker` (LightGCN + profiles + refresh only, frozen ranker reused).

*Known eval caveat:* the ranker's training users overlap the eval users (shared
`Random(42)` sample) — inherited from v4, so v5-vs-v4 deltas are valid but absolute
NDCG is mildly optimistic. A disjoint-eval split is the next honesty improvement.

---

## Measured results (20k-user held-out eval)

Active model: `lightgcn mf_20260616_001216` + `xgb_ranker xgb_20260616_001216`
(job 15, 2026-06-16). v4 = job 12; v3 = job 9 (BPR-MF). **v3 used a different,
easier protocol** (leave-**one**-out, pop baseline 0.026), so the only fair
cross-version number is **lift over the popularity baseline**; v5-vs-v4 share the
identical leave-last-20% protocol and are directly comparable.

| metric                | v3 (job 9, BPR) | v4 (job 12) | **v5 (active)** |
|-----------------------|-----------------|-------------|-----------------|
| protocol              | leave-one-out   | leave-last-20% | leave-last-20% |
| hit_rate@10           | 0.111           | 0.0657      | **0.1646**      |
| ndcg@10               | 0.0755          | 0.032       | **0.0676**      |
| recall@10             | —               | 0.0438      | **0.0981**      |
| recall@50             | 0.165           | 0.0975      | **0.2104**      |
| retrieval_recall      | —               | 0.5329      | **0.6506**      |
| pop baseline hit@10   | 0.0263          | 0.0437      | 0.0436          |
| **lift over pop**     | **4.2×**        | **1.5×**    | **3.8×**        |
| LightGCN final loss   | (BPR)           | 0.3223      | **0.0525**      |

ndcg@10 by user history (cold-start → heavy): v4 `{1:0.019, 2–4:0.040, 5–10:0.040,
10+:0.017}` → v5 `{1:0.052, 2–4:0.070, 5–10:0.069, 10+:0.064}` — **every bucket
≈3× v4**, and roughly flat across activity (no segment sacrificed).

**v5 vs v4 (apples-to-apples): 2.1–2.5× across the board. v5 recovers v3-level
lift (3.8× vs 4.2×) on the harder protocol**, with a higher retrieval ceiling and
the full multi-channel architecture intact.

**Why (decomposition = retrieval ceiling × ranker conversion):**
- retrieval_recall 0.533 → **0.651** (+22%) — 400-epoch LightGCN + structured channel + mf-weighted budget.
- ranker conversion (hit@10 ÷ retrieval_recall) 12.3% → **25.5%** (×2) — the ranker fix (retrieved-distribution training + struct features + stronger mf). This is the dominant lever: in v4, sorting by raw mf *beat* the trained ranker; in v5 the fixed XGBoost beats both raw mf and the linear A/B contender (A/B: XGBoost 0.072 vs linear 0.041).

Reproduce / re-verify with `scripts/diagnose_{channels,rankers,metadata,lightgcn}.py`.

---

## 0. Big picture

Recommendations come from a **retrieve → rerank → diversify** pipeline over
**four independent movie embedding spaces** plus popularity:

```
MOVIE vectors (all in Postgres / pgvector, HNSW indexed)
  metadata_embedding   768  mpnet("Title:… Genres:… Director:… Cast:… Year:…")   stateless
  plot_embedding       768  mpnet(plot)
  mf_embedding          64  LightGCN item vector                                 nightly
  community_embedding  384  up to 5 KMeans centroids of the movie's review texts nightly

USER vectors
  metadata/plot/community PROFILES  — positive & negative, recency+preference weighted (recomputed on use)
  mf_embedding          64  LightGCN user vector                                 in DB

  structured metadata  348  v3 blocks (genre/cast/year, no plot), in-memory       complementary to mpnet meta

RETRIEVE  union of per-channel nearest buckets (mf 500 · metadata 300 · structured 300 · community 200 · plot 150 · popular 100), deduped, seen removed  → ~1100 pool
RANK      A/B winner of {XGBoost graded rank:ndcg, mf-centric linear} on 26 features
DIVERSIFY MMR (plot-sim, per-genre cap) → top 100  → user_recommendations
```

Heavy work is nightly; new reviews trigger a fast per-user refresh through the
**same** pipeline (real-time == nightly quality).

---

## 1. The four movie embeddings

| Channel | Dim | Source | Table | When |
|---|---|---|---|---|
| **metadata** | 768 | mpnet of a text template of genres/cast/crew/year | `movie_metadata_embeddings` | on add/edit (auto) |
| **plot** | 768 | mpnet of the plot text | `movie_plot_embeddings` | on add/edit (auto) |
| **mf** | 64 | LightGCN item vector | `movie_mf_embeddings` | nightly |
| **community** | 384 | ≤5 KMeans centroids of the movie's review vectors | `movie_community_embeddings` | nightly |

- **metadata & plot** use `all-mpnet-base-v2` (`TEXT_EMBED_MODEL`, 768-d) and are
  **stateless** — a new movie is embedded with one encode, no fitted PCA/hashers
  ([app/ml/text_embed.py](../app/ml/text_embed.py)). They're built **automatically**
  in the background whenever a movie is created/edited or its genres/cast/etc.
  change ([app/ml/movie_refresh.py](../app/ml/movie_refresh.py), per-movie advisory
  lock).
- **community** ([app/ml/community.py](../app/ml/community.py)): each review is
  embedded with MiniLM (384-d, 128-token truncation, MPS); K-means (K=5) over a
  movie's review vectors yields up to 5 centroids (+ weights) capturing distinct
  *audience reactions*. < 5 reviews → fewer clusters; 0 reviews → none (falls back
  to metadata/plot). Review vectors are stored once in `review_embeddings`; the
  nightly job re-embeds only **new** reviews and re-clusters only **changed**
  movies — never re-embedding old reviews.

## 2. preference_score & sentiment

On each review, a fine-tuned DistilBERT scores the text → 0–10 sentiment, and
`preference_score = 0.7·rating + 0.3·sentiment` (clamped 0–10). This drives both
the user profiles and the LightGCN positives.

## 3. User taste vectors

[app/ml/profiles.py](../app/ml/profiles.py) — recomputed on use (real-time builds
one user; nightly builds all in the loop). For each content channel we build a
**positive** and **negative** profile = recency-weighted, preference-centered,
L2-normalised sum of the user's liked / disliked movie vectors:

```
baseline   = (5·global_mean + Σ pref) / (5 + n)          # per-user, shrunk
centered_i = pref_i − baseline                           # harsh/generous raters comparable
weight_i   = |centered_i| · 0.9^rank_from_newest         # USER_EMBED_DECAY = 0.9 (rank-based, gap-safe)
profile(pos) = L2( Σ_{centered>0} weight·movie_vec ),  profile(neg) = L2( Σ_{centered<0} … )
```

Plus the **LightGCN user vector** (`mf`, from `user_mf_embeddings`).

## 4. LightGCN (collaborative)

[app/ml/lightgcn.py](../app/ml/lightgcn.py) — replaces BPR-MF. Propagates user/item
embeddings over the normalised user-item bipartite graph (positives =
preference ≥ 6; a user's own rated movies are never sampled as negatives), 3 layers,
final = mean of layer embeddings, trained with full-batch BPR. `LIGHTGCN_DIM=64`,
`LIGHTGCN_LAYERS=3`, **`LIGHTGCN_EPOCHS=400`** (v5: 30 was undertrained — 30 grad
steps; 400 lifts mf recall@300 0.415→0.503), `LIGHTGCN_LR=0.01`, `LIGHTGCN_REG=1e-4`.
Output user/item vectors are saved as a versioned artifact **and** stored in the DB
(`user_mf_embeddings` / `movie_mf_embeddings`).

## 5. Popularity

`movie_popularity_stats`, `popularity_score = log1p(count)/log1p(maxcount) ·
(0.5 + 0.5·bayes_pref/10)`, with Bayesian-shrunk average preference
(`POPULARITY_PRIOR=50`).

## 6. Retrieval (multi-channel)

[app/ml/reco.py](../app/ml/reco.py) `generate_candidates`: take the top-k nearest
movies in **each** space to the user's positive profile / mf vector, plus a
popularity bucket, **union**, drop seen (reviewed ∪ watched ∪ watchlist) →
candidate pool (~1100). **Six channels** (v5 budgets, rebalanced to measured
per-channel recall — mf is strongest at 0.40, plot weakest at 0.10): `RETR_MF 500`,
`RETR_META 300`, `RETR_STRUCT 300`, `RETR_COMM 200`, `RETR_PLOT 150`, `RETR_POP 100`.
The **structured-metadata** channel ([app/ml/struct_embed.py](../app/ml/struct_embed.py))
is the v3 block embedding (genre/cast/crew/year, no plot), built in-memory from the
fitted v3 pipeline — complementary to the mpnet metadata channel (mpnet+structured
beats either alone). A great movie can't be linearly cancelled out of the pool, so
the ranker actually gets to see it. Union retrieval_recall ≈ 0.65.

## 7. Ranker (graded) — XGBoost vs linear A/B

[app/ml/ranker.py](../app/ml/ranker.py), `objective=rank:ndcg`. **Graded labels**:
3 (pref ≥ 9), 2 (7–8.9), 1 (5.5–6.9), 0 (rest) — so burying a 10/10 is penalised
harder than burying a 7. **26 features**:

per-channel cos to **pos & neg** profiles + gap (metadata, **structured**, plot,
community); mf score + **per-user percentile**; popularity (raw + percentile);
review_count_log; movie_avg_pref (Bayesian); genre_overlap; favourite-genre match;
recent-genre match; actor_overlap; year_distance (candidate movie's release era vs
the user's preferred era — *not* interaction timing); user_avg_pref; user_n_pos_log;
has_mf. Per-candidate raw scores **and** percentiles are both fed (calibrated + raw).
`XGB_PARAMS`: 300 trees, depth 6, lr 0.05; `XGB_NEG_PER_POS=30`.

**v5 fixes:** (a) `_build_xgb_data` trains only on the **retrieved candidate
distribution** — it no longer force-injects held-out positives that retrieval
missed (those look like low-mf negatives and taught the ranker to distrust mf; in
v4 raw mf *beat* the trained ranker). (b) Each full ranker train runs an **A/B**:
XGBoost vs a mf-centric **Ridge linear** scorer (`LinearRanker`), shipping the
NDCG@10 winner on the held-out split (`load_ranker` dispatches by saved `kind`).
Latest A/B: XGBoost 0.072 > linear 0.041 → XGBoost active.

## 8. Diversity

MMR over plot similarity + per-genre cap (`DIVERSITY_LAMBDA=0.30`,
`DIVERSITY_MAX_PER_GENRE=4`) → final top-100.

## 9. similar_movies

Nightly `rebuild_similar_hybrid` blends `0.4·plot + 0.3·metadata + 0.2·mf +
0.1·community` cosine (mf/community contribute only when both movies have them;
content-only otherwise). A new movie gets an instant **content-only** (plot+meta)
list ([app/ml/similar.py](../app/ml/similar.py)); the behavioural channels are
layered in at the next nightly run. Runs **after** the full recs refresh completes.
(Note: does *not* yet use the v5 structured-metadata channel — content here is
plot+mpnet-metadata only.)

## 10. Nightly training job

[app/jobs/training.py](../app/jobs/training.py) (`python -m
app.jobs.retrain_recommendations`, or `POST /admin/recommendations/retrain` →
detached subprocess). Steps: advisory lock → popularity → **dirty-community
rebuild** → chronological **leave-last-k** split (newest 20%) → **LightGCN** →
build context with the new MF → per-user profiles → **ranker A/B (or reuse)** →
**staged eval** → **NDCG@10 rollback** (activate only if ≥ the active model's
stored ndcg, deterministic split) → on activation: persist artifacts + MF vectors
to DB + refresh `user_recommendations` (transactional per 1500-user batch,
incremental commits, no downtime) → rebuild similar_movies. Every step is logged.

**Cadence split (v5):** the ranker's features are dot-products / cosines /
structured overlaps — invariant to LightGCN's nightly re-basis — so the ranker is
trained **once / occasionally**. Default `retrain_recommendations` runs the full
ranker A/B; the **nightly** form `--skip-ranker` retrains LightGCN + profiles +
refresh and **reuses the frozen active ranker** (loaded for eval + refresh, not
re-persisted).

## 11. Real-time

On a new review, a background task ([app/ml/realtime.py](../app/ml/realtime.py))
rebuilds that user's top-100 with the **full pipeline** (candidates → active ranker →
MMR), reading their MF vector from the DB. Same objective as nightly.

## 12. Evaluation (staged)

[app/jobs/training.py](../app/jobs/training.py) `_evaluate` (chronological
hold-out): **retrieval_recall** (was the held-out liked movie even retrieved),
hit-rate@10, recall@10/50, NDCG@10, popularity baseline, and **NDCG by review
count** (cold-user buckets 1 / 2–4 / 5–10 / 10+). Stored in
`recommendation_jobs.metrics` and `model_versions.metrics`. Latest numbers live
there (and in the admin endpoints).

## 13. Logging

Everything is recorded to `logs` ([app/core/logging_db.py](../app/core/logging_db.py)):
a request middleware logs every mutation (method/path/status/user/latency), and
domain events (`user_login`, `review_*`, `recommendations_served`,
`training_*`, `model_activated/rolled_back`, `similar_rebuilt`, …) are emitted at
their call sites. Seeded event/entity types in migration 0005.

## 14. Storage & ops

- New vector tables + `review_embeddings` (~1.85M × 384 stored so reviews are
  never re-embedded). HNSW cosine index per movie vector column.
- One-time seed: `scripts/backfill_text_embeddings.py` (metadata+plot),
  `scripts/build_community.py` (reviews + clusters). Steady state is incremental.
- macOS: torch+xgboost coexist via the `OMP_NUM_THREADS=1` Darwin guard; community
  encoding uses MPS. On Linux/VPS full multithreading; `apt-get install libgomp1`
  for xgboost.

## 15. Key hyperparameters (`app/ml/config.py`)

| Name | Value |
|---|---|
| TEXT_EMBED_MODEL / dim | all-mpnet-base-v2 / 768 |
| REVIEW_EMBED_MODEL / dim / COMMUNITY_K | MiniLM / 384 / 5 |
| LIGHTGCN_DIM / LAYERS / EPOCHS / LR / REG | 64 / 3 / **400** / 0.01 / 1e-4 |
| USER_EMBED_DECAY / PREF_NEUTRAL(baseline) | 0.9 / per-user shrunk |
| RETR_MF/META/STRUCT/COMM/PLOT/POP | **500/300/300/200/150/100** |
| graded labels | 3≥9, 2≥7, 1≥5.5, 0 else |
| ranker | XGBoost vs linear A/B (ships NDCG@10 winner) |
| XGB trees/depth/lr / NEG_PER_POS | 300/6/0.05 / **30** |
| DIVERSITY_LAMBDA / MAX_PER_GENRE | 0.30 / 4 |
| SIMILAR_W plot/meta/mf/comm | 0.4/0.3/0.2/0.1 |
| POPULARITY_PRIOR / ROLLBACK_METRIC | 50 / ndcg_at_10 |
