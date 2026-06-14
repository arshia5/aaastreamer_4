# The aaastreamer Recommender — From Scratch

This document explains the entire recommendation system end to end: every signal,
every weight, every hyperparameter, and how the pieces fit together at training
time and at serving time. It is meant to be readable without looking at the code,
but file references are given throughout.

---

## 0. The big picture

A user's recommendations are produced by a **retrieve → rerank → diversify**
pipeline that blends three independent signals:

1. **Content** — how similar a movie is to the user's taste, in a 390-dim
   embedding space built from movie metadata + plot text.
2. **Collaborative** — "people who liked what you liked also liked X", learned by
   matrix factorisation over the rating matrix.
3. **Popularity** — how widely and highly reviewed a movie is.

These are combined into a **hybrid score**, then a **gradient-boosted ranker
(XGBoost)** reranks the top candidates, then **MMR diversity** spreads the final
list across genres. Movies the user has already reviewed, watched, or watchlisted
are excluded.

```
movie metadata ─► content embedding (390-d) ─┐
ratings + reviews ─► user embedding (390-d) ──┤─► content_score ┐
                                              │                  │
ratings ─► collaborative MF (128-d) ──────────┴─► collab_score ──┼─► hybrid
review counts ─► popularity_score ───────────────────────────────┘   score
                                                                       │
                                          candidate pool (top 200) ◄────┘
                                                       │
                                   XGBoost reranker (rank:ndcg) ─► reordered
                                                       │
                                        MMR diversity (genre-aware) ─► top 100
                                                       │
                                              user_recommendations
```

Everything heavy is computed in a **nightly/manual training job**; new reviews
get a **fast incremental update** during the day.

---

## 1. Data

| Table | What it holds |
|-------|---------------|
| `movies` (+ `genres`, `people`, `countries`, `languages` junctions) | catalogue + metadata, ~31,388 movies |
| `interactions` | one row per (user, movie): `rating` (0–10), `sentiment` (0–10), `preference_score`, review text, `review_date` |
| `movie_embeddings` | `vector(390)` content embedding per movie |
| `user_embeddings` | `vector(390)` taste embedding per user |
| `movie_popularity_stats` | review_count, avg_rating, avg_preference, `popularity_score` |
| `similar_movies` | precomputed top-10 nearest movies per movie |
| `user_recommendations` | stored top-100 per user (+ component scores) |
| `model_versions` | trained model artifacts (collaborative_mf, xgb_ranker), `is_active` |
| `recommendation_jobs` | training job runs: status, metrics |

---

## 2. Content embeddings (movie side)

Built by `scripts/fit_embeddings.py` + `app/ml/features.py`, reproducing
`aaastreamer_3/movie_embeddings.ipynb`. Each movie becomes a **732-dim** feature
vector from eight blocks, each L2-normalised then **weighted**:

| Block | Method | Dims | Weight |
|-------|--------|-----:|-------:|
| plot | SentenceTransformer `all-MiniLM-L6-v2` | 384 | **2.0** |
| genre | MultiLabelBinarizer | 27 | **1.2** |
| actors | FeatureHasher | 128 | **1.2** |
| director | FeatureHasher | 64 | **1.0** |
| writer | FeatureHasher | 64 | **0.7** |
| year | StandardScaler (z-score) | 1 | **0.6** |
| language | FeatureHasher | 32 | **0.5** |
| country | FeatureHasher | 32 | **0.4** |

Concatenation order is fixed: plot, genre, director, writer, actor, language,
country, numeric. The 732-dim matrix is then reduced by **PCA to 390 dims**,
which preserves **90.05 % of the variance** (the "90 %" target; 95 % would be 461
dims, 98 % → 528). The fitted PCA + scaler + genre binarizer are persisted to
`artifacts/movie_embed_pipeline.joblib` so new movies land in the exact same
space (`app/ml/embedder.py`). Hyperparameters live in `app/ml/config.py`
(`WEIGHTS`, `N_COMPONENTS = 390`, `MODEL_NAME`).

**New movie** → the content embedding is built **automatically** (background task)
whenever a movie is created/edited or its genres/cast/languages/countries change
([app/ml/movie_refresh.py](../app/ml/movie_refresh.py); per-movie advisory lock
serialises concurrent refreshes). No manual endpoint call is needed (the manual
`POST /embeddings/movies/{id}/generate` still exists). The embedding projects
through the saved PCA, and `similar_movies` is refreshed for the new movie and any
list it now belongs in.

**Similar movies are a content-weighted hybrid.** The nightly job rebuilds
`similar_movies` ([app/ml/similar.py](../app/ml/similar.py) `rebuild_similar_hybrid`):

```
score(X, Y) = 0.7 · cos(content_X, content_Y) + 0.3 · cos(mf_X, mf_Y)
```

(`SIMILAR_W_CONTENT = 0.7`, `SIMILAR_W_COLLAB = 0.3`). The collaborative term adds
behavioural "people who liked X also liked Y" signal (e.g. *The Dark Knight* →
*Inception*, *The Prestige*). A movie with **no MF item factor** (just added / too
few reviews) is scored **content-only**, as is the instant refresh when a new
movie is added — so cold movies still get sensible neighbours immediately, and the
behavioural blend is layered in at the next nightly run.

---

## 3. Sentiment → preference_score

When a review is created/updated, a fine-tuned **DistilBERT** (5-class, 1–5 stars)
scores the text. The score is the softmax **expected value rescaled to 0–10**:
`score_1_10 = (Σ p_i·(i+1) − 1)/4·9 + 1` (`app/ml/sentiment.py`,
`models/sentiment-distilbert/`).

The **preference_score** combines the explicit rating and the inferred sentiment
(`app/ml/scoring.py`):

```
preference_score = 0.7 · rating + 0.3 · sentiment        (both clamped to [0,10])
```

If only one of rating/sentiment exists, that one is used. preference_score is the
core signal that drives both the user embedding and the collaborative model.

---

## 4. User embeddings (user side)

`app/ml/user_embedding.py`. A user's 390-d vector is a **recency-weighted,
preference-centered sum** of the embeddings of movies they reviewed:

```
vector(user) = Σ_i  (preference_i − 5.5) · 0.9^rank_i · movie_embedding_i
             then L2-normalised
```

- Reviews are sorted oldest → newest. The **newest review gets rank 0 (weight
  1.0)**; each older review is multiplied by `0.9` per step (`USER_EMBED_DECAY`).
  This is **rank-based**, not absolute time, so a 2-year gap doesn't erase a
  user's profile.
- **5.5 is the neutral midpoint** (`PREF_NEUTRAL`): movies a user *disliked*
  (preference < 5.5) get a **negative** weight and push the vector *away* from
  that kind of movie; liked movies pull toward it.
- The final vector is **L2-normalised** (unit sphere) for cosine comparisons.

`content_score(user, movie) = cosine(user_vector, movie_embedding)`.

---

## 5. Popularity

`app/ml/popularity.py` recomputes `movie_popularity_stats` in pure SQL:

```
bayes_pref       = (PRIOR·global_mean + Σ preference) / (PRIOR + n)   # PRIOR = 50
popularity_score = ln(1 + review_count) / ln(1 + max_review_count)
                   · (0.5 + 0.5 · bayes_pref / 10)
```

Range [0, 1]. The log term rewards reach; the **Bayesian-shrunk** quality term
pulls low-count averages toward the global mean (`POPULARITY_PRIOR = 50`), so a
"10 reviews @ 10.0" movie can't beat "3000 @ 9.2". The same shrinkage is applied
to the `movie_avg_pref` ranker feature. `avg_preference_score` itself stores the
raw mean (an honest stat). Set `POPULARITY_PRIOR = 0` to disable shrinkage.

---

## 6. Collaborative filtering — PyTorch BPR matrix factorisation

`app/ml/collaborative.py`. We use **Bayesian Personalised Ranking** matrix
factorisation (a pure-PyTorch stand-in for LightFM's collaborative core; LightFM
has no Python 3.14 wheel, and metadata is handled separately by the content
signal, so its hybrid feature wasn't needed).

- **Positives**: interactions with `preference_score ≥ 6.0` (`CF_POS_THRESHOLD`),
  weighted by `preference/10`.
- **Model**: user factors `U ∈ ℝ^{nU×128}`, item factors `V ∈ ℝ^{nI×128}`, item
  bias `b ∈ ℝ^{nI}` (`CF_DIM = 128`).
- **Loss** (per sampled triple user *u*, positive *i*, random negative *j*):
  `−w · log σ(U_u·V_i + b_i − U_u·V_j − b_j) + λ(‖U_u‖² + ‖V_i‖² + ‖V_j‖²)`.
- **Hyperparameters**: `CF_EPOCHS = 12`, `CF_LR = 0.03` (Adam),
  `CF_REG = 1e-6` (`item_alpha == user_alpha`), `CF_BATCH = 8192`, 1 negative per
  positive, seed 42.
- `collab_score(user, movie) = U_user · V_movie + b_movie`.

Artifact: `artifacts/collaborative/mf_<timestamp>.joblib` (factors + id maps),
written **atomically** (temp file + `os.replace`) so a half-written model is never
loaded. Tracked in `model_versions` (`model_type = collaborative_mf`).

**Cold start**: a user/movie not in the trained maps simply has no collaborative
score and falls back to content + popularity.

---

## 7. Hybrid scoring

`app/ml/hybrid.py`. Each component is **min-max normalised per candidate set**,
then combined:

```
collab available:   final = 0.45·collab_n + 0.35·content_n + 0.20·pop_n
no collab (cold):   final = 0.70·content_n + 0.30·pop_n
no content either:  final = pop_n
```

Weights: `HYBRID_W_COLLAB/CONTENT/POP = 0.45/0.35/0.20`, fallback
`HYBRID_FB_CONTENT/POP = 0.70/0.30`. The hybrid produces the **candidate pool**:
top `CAND_POOL = 200` unseen movies by `final`.

---

## 8. XGBoost learning-to-rank reranker

`app/ml/ranker.py`. The hybrid candidate pool is reranked by an **XGBRanker**
(`objective = rank:ndcg`) trained nightly.

**Features per (user, candidate movie)** — 13 of them:

| # | Feature | Source |
|---|---------|--------|
| 1 | content_n | normalised content score |
| 2 | collab_n | normalised collaborative score |
| 3 | pop_n | normalised popularity |
| 4 | content_raw | raw cosine(user, movie) |
| 5 | collab_raw | raw `U·V + b` |
| 6 | review_count_log | `log1p(review_count)` |
| 7 | movie_avg_pref | movie's average preference |
| 8 | user_avg_pref | user's average preference (train) |
| 9 | user_n_pos_log | `log1p` of user's #positives |
| 10 | user_has_collab | 1 if the user has a collaborative factor |
| 11 | genre_overlap | # genres shared between candidate and the user's liked movies |
| 12 | actor_overlap | # actors shared between candidate and the user's liked movies |
| 13 | year_distance | \|candidate.year − user's mean liked year\| (NaN if unknown — XGBoost handles it) |

The overlap/distance features use a per-user **taste profile** (liked-genre set,
liked-actor set, mean liked year) built from the train split. They are computed
only in the nightly job (daytime serving reads the stored list).

**Training labels** (chronological split, see §11): each training user is a
**group**; their candidate pool is labelled `1` if the movie is one of their
**held-out liked** movies (preference ≥ 7), else `0`. Only groups containing ≥1
retrieved positive are kept (otherwise there is no ranking signal). Negatives are
the rest of the candidate pool (`XGB_NEG_PER_POS = 30` cap), sampled from up to
`XGB_TRAIN_USERS = 15,000` users.

**XGBoost hyperparameters** (`XGB_PARAMS`): `n_estimators = 300`,
`max_depth = 6`, `learning_rate = 0.05`, `subsample = 0.8`,
`colsample_bytree = 0.8`, `min_child_weight = 5`, `tree_method = hist`,
`eval_metric = ndcg@10`.

Artifact: `artifacts/ranker/xgb_<timestamp>.joblib` (atomic write),
`model_versions` (`model_type = xgb_ranker`). Used **only during the nightly
refresh** to reorder candidates before they are stored; daytime serving reads the
already-reranked `user_recommendations` from the DB.

---

## 9. Diversity (MMR)

`hybrid.mmr_rerank`. The reranked candidates are diversified with **Maximal
Marginal Relevance** + a per-genre cap, so the top of the list isn't all one
franchise/genre:

```
pick = argmax_i  (1 − λ)·base_n_i  −  λ·max_{j∈selected} cosine(i, j)
```

with `DIVERSITY_LAMBDA = 0.30` and `DIVERSITY_MAX_PER_GENRE = 4` (a genre can
appear at most 4× in the stored list; the cap is relaxed only if nothing else is
available). Similarity uses the content embeddings.

---

## 10. Exclusions

Before scoring, the candidate set removes every movie the user has **reviewed
OR has in watched / watchlist** (`interactions ∪ user_movie_states`). Verified:
0 stored recommendations leak a seen movie.

---

## 11. Nightly / manual training job

`app/jobs/training.py`, run by `python -m app.jobs.retrain_recommendations` (cron)
or `POST /admin/recommendations/retrain` (spawns the same CLI as a detached
subprocess). Steps:

1. **Advisory lock** (`pg_try_advisory_lock`) — no two trainings run at once.
2. Recompute `movie_popularity_stats`.
3. **Chronological split** — hold out each user's **newest** review as the test
   item; everything older is "train".
4. Recompute all `user_embeddings` from the train split.
5. Train the collaborative MF on train positives → save artifact (inactive).
6. Train the XGBoost ranker on the train split → save artifact (inactive).
7. **Evaluate** the new pipeline AND the current active pipeline on the **same**
   held-out reviews (only the models differ — fair comparison).
8. **Rollback decision**: activate the new models only if
   `NDCG@10(new) ≥ NDCG@10(old) + ROLLBACK_MARGIN` (`ROLLBACK_METRIC = ndcg_at_10`,
   `ROLLBACK_MARGIN = 0.0`). If worse, the new artifacts are kept but stay
   inactive and the **old models remain active** (and the existing
   recommendations are untouched).
9. If activated, **refresh `user_recommendations`** for all users (hybrid →
   XGBoost rerank → diversity → top 100), and record everything in
   `recommendation_jobs.metrics` + `model_versions.metrics`.

The refresh replaces recommendations **per batch of users inside short
transactions** (no global `TRUNCATE`), so the website always sees a complete
list for every user — their old one until their new one is committed. The whole
job runs as a **separate process** (cron CLI or a detached subprocess from the
admin endpoint), so it never blocks the API event loop, and all other writes
(popularity, embeddings, model_versions) are MVCC-friendly. The site stays fully
usable during training. Admins are ordinary users here — no recommendation logic
checks `is_admin`; they get recommendations as soon as they have a rated review.

The job never blocks the API: real-time `partial_fit` only mutates the in-memory
active model and never writes artifacts, so it cannot corrupt a training run.

---

## 12. Real-time updates (during the day)

On a new/updated review (`app/ml/realtime.py`, wired into the interactions
endpoints):

- **Synchronous (fast, ~tens of ms warm)**: recompute that user's content
  embedding and `preference_score`/`sentiment`.
- **Background task**: collaborative `partial_fit_user` — a deliberately **small
  nudge**: items are frozen (the collaborative space is never distorted, only this
  user's own position moves), with a low learning rate (`CF_PARTIAL_LR = 0.01`)
  and a blend back toward the nightly factor
  (`new = 0.30·updated + 0.70·nightly`, `CF_PARTIAL_BLEND`), so one odd review
  can't re-learn the user (skipped for cold users/movies)
  + rebuild that user's top-100 with the **hybrid + diversity** scorer (no XGBoost
  — that's nightly only). The reco context (movie matrix + active model) is cached
  in-process and **invalidated when the active model version changes**, so
  real-time updates always target the active model.

---

## 13. Evaluation

Leave-out protocol: hold out each user's newest review, rebuild from older
reviews, and measure whether the held-out **liked** movie (preference ≥ 7) is
recovered. Metrics @10 + recall@50, against a popularity baseline. Stored in
`recommendation_jobs.metrics` and surfaced at
`GET /stats/recommendations/evaluation`.

**Measured progression** (20k held-out users):

| Pipeline | hit-rate@10 | NDCG@10 | recall@50 |
|----------|------------:|--------:|----------:|
| Popularity baseline | 2.6 % | — | — |
| Content only (original) | ~3 % | ~0.02 | — |
| Hybrid (collab+content+pop) | 5.7 % | 0.035 | 11.6 % |
| Hybrid + XGBoost + diversity | 10.5 % | 0.071 | 15.9 % |
| **+ overlap features + Bayesian popularity** | **11.1 %** | **0.076** | **16.6 %** |

≈ **4× the popularity baseline**.

---

## 14. All hyperparameters (quick reference)

All in `app/ml/config.py`.

| Name | Value | Meaning |
|------|------:|---------|
| `N_COMPONENTS` | 390 | content embedding dims (PCA, 90% variance) |
| plot/genre/actor/director/writer/year/lang/country weights | 2.0/1.2/1.2/1.0/0.7/0.6/0.5/0.4 | content block weights |
| `PREF_RATING_WEIGHT` / `PREF_SENTIMENT_WEIGHT` | 0.7 / 0.3 | preference_score blend |
| `PREF_NEUTRAL` | 5.5 | user-embedding centering midpoint |
| `USER_EMBED_DECAY` | 0.9 | per-rank recency decay |
| `CF_DIM` | 128 | collaborative factor dims |
| `CF_EPOCHS` / `CF_LR` / `CF_REG` / `CF_BATCH` | 12 / 0.03 / 1e-6 / 8192 | BPR-MF training |
| `CF_POS_THRESHOLD` | 6.0 | min preference to count as a positive |
| `CF_PARTIAL_EPOCHS` / `CF_PARTIAL_LR` / `CF_PARTIAL_BLEND` | 2 / 0.01 / 0.30 | real-time fit_partial (small nudge) |
| `POPULARITY_PRIOR` | 50 | Bayesian shrinkage prior for avg preference |
| `HYBRID_W_COLLAB/CONTENT/POP` | 0.45 / 0.35 / 0.20 | hybrid weights |
| `HYBRID_FB_CONTENT/POP` | 0.70 / 0.30 | fallback (no collab) |
| `CAND_POOL` | 200 | candidates fed to reranking |
| `XGB_PARAMS.n_estimators/max_depth/lr` | 300 / 6 / 0.05 | XGBoost ranker |
| `XGB_TRAIN_USERS` / `XGB_NEG_PER_POS` | 15000 / 30 | LTR training data |
| `DIVERSITY_LAMBDA` / `DIVERSITY_MAX_PER_GENRE` | 0.30 / 4 | MMR diversity |
| `TOP_N_RECOMMENDATIONS` | 100 | stored per user |
| `ROLLBACK_METRIC` / `ROLLBACK_MARGIN` | ndcg_at_10 / 0.0 | activation gate |
| `EVAL_SAMPLE_USERS` | 20000 | users scored in nightly eval |
| `SIMILAR_TOP_N` | 10 | similar movies stored per movie |
| `SIMILAR_W_CONTENT` / `SIMILAR_W_COLLAB` | 0.7 / 0.3 | hybrid similar-movies blend |

---

## 15. Operations

```bash
# one-time data prep
python -m scripts.fit_embeddings           # content PCA pipeline
python -m scripts.load_movies              # catalogue -> relational tables
python -m scripts.backfill_embeddings      # movie_embeddings
python -m scripts.load_reviews             # users + interactions
python -m scripts.build_similar_movies     # similar_movies

# nightly (cron) or manual
python -m app.jobs.retrain_recommendations [--epochs N] [--max-users N]
```

Admin API: `POST /admin/recommendations/retrain`,
`GET /admin/recommendations/jobs[/{id}]`, `GET /admin/recommendations/models`.

**macOS note**: torch and xgboost each bundle their own OpenMP; the app pins
`OMP_NUM_THREADS=1` on Darwin to avoid a dual-runtime segfault (numpy uses
Accelerate and is unaffected). On a Linux VPS a single shared `libgomp` is used,
the guard is skipped, and full multithreading is kept (`apt-get install libgomp1`
if XGBoost complains about OpenMP).
