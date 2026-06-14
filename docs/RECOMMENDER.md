# The aaastreamer Recommender — v4 (from scratch)

Every signal, weight, and hyperparameter in the recommendation system, plus how
training and serving fit together. File references throughout.

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

RETRIEVE  union of per-channel nearest buckets (plot 400 · metadata 300 · mf 300 · community 200 · popular 100), deduped, seen removed  → ~1000 pool
RANK      XGBoost (graded labels 0–3, rank:ndcg) on 23 features
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
weight_i   = |centered_i| · 0.95^rank_from_newest        # USER_EMBED_DECAY = 0.95
profile(pos) = L2( Σ_{centered>0} weight·movie_vec ),  profile(neg) = L2( Σ_{centered<0} … )
```

Plus the **LightGCN user vector** (`mf`, from `user_mf_embeddings`).

## 4. LightGCN (collaborative)

[app/ml/lightgcn.py](../app/ml/lightgcn.py) — replaces BPR-MF. Propagates user/item
embeddings over the normalised user-item bipartite graph (positives =
preference ≥ 6; a user's own rated movies are never sampled as negatives), 3 layers,
final = mean of layer embeddings, trained with full-batch BPR. `LIGHTGCN_DIM=64`,
`LIGHTGCN_LAYERS=3`, `LIGHTGCN_EPOCHS=30`, `LIGHTGCN_LR=0.01`, `LIGHTGCN_REG=1e-4`.
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
candidate pool (~1000). Channel sizes: `RETR_PLOT 400`, `RETR_META 300`,
`RETR_MF 300`, `RETR_COMM 200`, `RETR_POP 100`. A great movie can't be linearly
cancelled out of the pool, so the ranker actually gets to see it.

## 7. XGBoost ranker (graded)

[app/ml/ranker.py](../app/ml/ranker.py), `objective=rank:ndcg`. **Graded labels**:
3 (pref ≥ 9), 2 (7–8.9), 1 (5.5–6.9), 0 (rest) — so burying a 10/10 is penalised
harder than burying a 7. **23 features**:

per-channel cos to **pos & neg** profiles + gap (metadata, plot, community); mf
score + **per-user percentile**; popularity (raw + percentile); review_count_log;
movie_avg_pref (Bayesian); genre_overlap; favourite-genre match; recent-genre
match; actor_overlap; year_distance; user_avg_pref; user_n_pos_log; has_mf.
Per-candidate raw scores **and** percentiles are both fed (calibrated + raw).
`XGB_PARAMS`: 300 trees, depth 6, lr 0.05; `XGB_NEG_PER_POS=12`.

## 8. Diversity

MMR over plot similarity + per-genre cap (`DIVERSITY_LAMBDA=0.30`,
`DIVERSITY_MAX_PER_GENRE=4`) → final top-100.

## 9. similar_movies (v4)

Nightly `rebuild_similar_hybrid` blends `0.4·plot + 0.3·metadata + 0.2·mf +
0.1·community` cosine (mf/community contribute only when both movies have them;
content-only otherwise). A new movie gets an instant **content-only** (plot+meta)
list ([app/ml/similar.py](../app/ml/similar.py)); the behavioural channels are
layered in at the next nightly run.

## 10. Nightly training job

[app/jobs/training.py](../app/jobs/training.py) (`python -m
app.jobs.retrain_recommendations`, or `POST /admin/recommendations/retrain` →
detached subprocess). Steps: advisory lock → popularity → **dirty-community
rebuild** → chronological **leave-last-k** split (newest 20%) → **LightGCN** →
build v4 context with the new MF → per-user profiles → **graded XGBoost** →
**staged eval** → **NDCG@10 rollback** (activate only if ≥ the active model's
stored ndcg, deterministic split) → on activation: persist artifacts + MF vectors
to DB + refresh `user_recommendations` (transactional per batch, no downtime) +
rebuild similar_movies. Every step is logged to the `logs` table.

## 11. Real-time

On a new review, a background task ([app/ml/realtime.py](../app/ml/realtime.py))
rebuilds that user's top-100 with the **full v4 pipeline** (candidates → XGBoost →
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
| LIGHTGCN_DIM / LAYERS / EPOCHS / LR / REG | 64 / 3 / 30 / 0.01 / 1e-4 |
| USER_EMBED_DECAY / PREF_NEUTRAL(baseline) | 0.95 / per-user shrunk |
| RETR_PLOT/META/MF/COMM/POP | 400/300/300/200/100 |
| graded labels | 3≥9, 2≥7, 1≥5.5, 0 else |
| XGB trees/depth/lr / NEG_PER_POS | 300/6/0.05 / 12 |
| DIVERSITY_LAMBDA / MAX_PER_GENRE | 0.30 / 4 |
| SIMILAR_W plot/meta/mf/comm | 0.4/0.3/0.2/0.1 |
| POPULARITY_PRIOR / ROLLBACK_METRIC | 50 / ndcg_at_10 |
