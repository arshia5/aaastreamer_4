# aaastreamer API

Async **FastAPI** backend for a movie‑recommendation platform, backed by
**PostgreSQL + pgvector**. Includes JWT auth with refresh‑token rotation,
full CRUD over the catalogue, user ratings/reviews, watched/watchlist state,
vector similarity search, precomputed recommendations, audit logging, and
analytics/stats endpoints.

## Stack

- FastAPI + Uvicorn
- SQLAlchemy 2.0 (async) + asyncpg
- pgvector (`vector(390)` embeddings + cosine/L2/inner‑product search)
- Alembic migrations
- PyJWT + bcrypt

## Project layout

```
app/
  core/        config, async engine/session, security (hashing + JWT)
  models.py    all SQLAlchemy models (21 tables)
  schemas.py   Pydantic request/response models
  deps.py      DB session, current-user / admin guards, pagination
  routers/     auth, users, reference, movies, interactions, states,
               embeddings, recommendations, stats, logs
  main.py      app + router wiring
alembic/       migration env + initial migration
scripts/       create_admin.py bootstrap helper
```

## Setup

1. **Postgres with pgvector.** Create a database and ensure the `vector`
   extension is available (the initial migration runs
   `CREATE EXTENSION IF NOT EXISTS vector`).

   ```sql
   CREATE DATABASE aaastreamer;
   ```

2. **Configure environment.**

   ```bash
   cp .env.example .env
   # edit DATABASE_URL and set a strong JWT_SECRET:
   python -c "import secrets; print(secrets.token_urlsafe(64))"
   ```

3. **Install deps** (a `venv/` already exists here):

   ```bash
   venv/bin/pip install -r requirements.txt
   ```

4. **Run migrations:**

   ```bash
   venv/bin/alembic upgrade head
   ```

5. **Create an admin user:**

   ```bash
   venv/bin/python -m scripts.create_admin admin admin@example.com 'StrongPass123'
   ```

6. **Run the API:**

   ```bash
   venv/bin/uvicorn app.main:app --reload
   ```

   Interactive docs at **http://localhost:8000/docs**.

## Auth flow

- `POST /auth/register` – self sign‑up (always non‑admin).
- `POST /auth/login` – OAuth2 password form (`username` accepts username *or*
  email) → `{access_token, refresh_token}`. `POST /auth/login/json` is the
  JSON equivalent.
- `POST /auth/refresh` – exchange a refresh token for a new access token.
- `POST /auth/refresh/rotate` – rotate: returns a new pair and revokes the old
  refresh token.
- `POST /auth/logout` / `POST /auth/logout/all` – revoke one / all sessions.

Send the access token as `Authorization: Bearer <token>`. Admin‑only routes
(catalogue writes, user management, embeddings, recommendations, logs) require
`is_admin = true`.

## Endpoint groups

| Area | Prefix | Highlights |
|------|--------|------------|
| Auth | `/auth` | register, login, refresh/rotate, logout, `/auth/me` |
| Users | `/users` | admin CRUD + self‑service `/users/me` |
| Reference | `/people` `/roles` `/genres` `/countries` `/languages` | CRUD + search + `/count` |
| Movies | `/movies` | filter by title/year/genre/country/language, `/by-imdb/{id}`, manage genres/countries/languages/cast |
| Interactions | `/interactions` | ratings + reviews; `/interactions/me`, upsert via `PUT /interactions/me/{movie_id}` |
| Watch state | `/me/movie-states` | watched / watchlist; list movies by state |
| Embeddings | `/embeddings` | upsert movie/user vectors, `POST /embeddings/movies/search`, `/neighbors`, user‑vector recs |
| Recommendations | `/movies/{id}/similar`, `/me/recommendations`, `/users/{id}/recommendations` | precomputed similar movies & per‑user recs |
| Stats | `/stats` | `/global`, `/movies/{id}`, `/users/{id}`, `/me`, top‑rated, most‑rated, most‑watched, rating distribution, genre/country/language distribution |
| Logs | `/logs`, `/log-event-types`, `/log-entity-types` | audit log + type management (admin) |

## Vector search

Embeddings are `vector(390)` (configurable via `EMBEDDING_DIM`). Search bodies
accept a `metric` of `cosine` (default), `l2`, or `inner`:

```bash
curl -X POST localhost:8000/embeddings/movies/search \
  -H 'Content-Type: application/json' \
  -d '{"embedding": [0.0, ...390 floats...], "limit": 10, "metric": "cosine"}'
```

`GET /embeddings/movies/{id}/neighbors` finds nearest movies to an existing
movie's stored embedding.

## Embedding pipeline (new movies @ 90% variance)

Reproduces `aaastreamer_3/movie_embeddings.ipynb`: each movie becomes a 732‑d
feature vector (MiniLM plot 384 + genre multi‑hot + hashed director/writer/cast
+ hashed language/country + scaled year, each L2‑normalised and weighted), then
PCA reduces it to **390 dims = 0.9005 explained variance** — matching the
`vector(390)` column. The fitted transformers (genre `MultiLabelBinarizer`, year
`StandardScaler`, and the **PCA‑390** projection) are persisted so any *new*
movie is embedded into the exact same space.

```bash
# 1. fit + persist the pipeline  (-> artifacts/movie_embed_pipeline.joblib)
venv/bin/python -m scripts.fit_embeddings [path/to/movies.csv]

# 2. load the catalogue into the relational schema
venv/bin/python -m scripts.load_movies [path/to/movies.csv]

# 3. backfill embeddings for every movie in the DB
venv/bin/python -m scripts.backfill_embeddings --batch 1024   # or --only-missing
```

Then, for new movies:

- `POST /embeddings/movies/{id}/generate` (admin) — rebuilds the feature record
  from the movie's DB metadata and upserts its 390‑d embedding.
- `POST /embeddings/generate` (admin) — embeds raw metadata on the fly
  (no DB write); the returned vector can be fed straight into
  `POST /embeddings/movies/search`.

> Verified: a DB‑reconstructed record yields a vector identical (cosine = 1.0)
> to the CSV pipeline, and a never‑seen "Avengers‑like" description retrieves
> *The Avengers*, *Age of Ultron*, and *Civil War* as nearest neighbours.

`artifacts/` is regenerated by `fit_embeddings` and is safe to delete/retrain.
The embedding endpoints lazily import torch only on first use, so the rest of
the API stays light.

## Sentiment, reviews & user embeddings

**Sentiment model** — a fine-tuned DistilBERT (5-class → expected value rescaled
to a 0–10 score) lives in `models/sentiment-distilbert/` ([app/ml/sentiment.py](app/ml/sentiment.py)).
Whenever a review is created/updated through the interactions API, the server
computes `sentiment` from the review text and
`preference_score = 0.7·rating + 0.3·sentiment` (both clamped to 0–10). These two
fields are read-only to clients. torch loads lazily on the first review.

**Importing the reviews dataset:**

```bash
# creates a user per reviewer (email <username>@test.com, password 1234) and
# ~1.85M interactions (rating + sentiment from the data, deduped per user/movie)
venv/bin/python -m scripts.load_reviews [path/to/reviews_dir]
```

**User embeddings** (390-d, same space as movies):

```bash
venv/bin/python -m scripts.build_user_embeddings
```

Each user vector = `Σ (preference − 5.5) · γ^rank · movie_embedding`, where reviews
are sorted oldest→newest and the **newest gets rank 0 (full weight)**, each older
review × γ (=0.9) per step — rank-based, so a long gap between reviews doesn't
wipe out a user's profile. Centred weighting means disliked movies (preference <
5.5) push the vector *away*. See [app/ml/user_embedding.py](app/ml/user_embedding.py).

Endpoints:
- `POST /embeddings/users/me/build` — (re)build the caller's embedding from their reviews.
- `POST /embeddings/users/{id}/build` (admin) — build for any user.
- `GET /embeddings/users/me/recommendations` — personalised picks (excludes seen by default).
- `GET /embeddings/users/{id}/recommendations` (admin) — same for any user.

> Verified: user "#1_Gracie" (loved Ocean's Eleven & LotR: Fellowship) is
> recommended Ocean's Twelve/Thirteen and the rest of the LotR/Hobbit films.

## Similar movies (top-10 per movie)

`similar_movies` stores the **top-10 nearest movies by embedding** for every
movie (`score` = cosine similarity, rank 1 = closest). An **HNSW index** on
`movie_embeddings.embedding` (migration 0003) keeps nearest-neighbour queries
fast.

```bash
# initial fill / full rebuild (exact, numpy, ~10s for 31k movies)
venv/bin/python -m scripts.build_similar_movies [--top 10]
```

The table is **kept fresh automatically**: whenever a movie's embedding is
generated/updated (`POST /embeddings/movies/{id}/generate` or
`PUT /embeddings/movies/{id}`), [app/ml/similar.py](app/ml/similar.py)
recomputes that movie's top-10 *and* refreshes any other movie whose top-10 the
new movie now belongs in. Read it via `GET /movies/{id}/similar` (raw rows) or
`GET /movies/{id}/similar/movies` (resolved movies, by rank).

> Verified: adding a "Toy Story 5" movie produced its own Toy-Story top-10 and
> inserted it into Toy Story's list at rank 2.

## Hybrid recommendations (content + collaborative + popularity + XGBoost)

> **Full deep-dive with every weight and hyperparameter:
> [docs/RECOMMENDER.md](docs/RECOMMENDER.md).**

The serving pipeline is **retrieve → rerank → diversify**: the hybrid scorer
generates a candidate pool, an **XGBoost learning-to-rank** model (trained
nightly, `rank:ndcg`) reranks it, and **MMR diversity** (genre-capped) spreads the
final top-100. Movies the user has **reviewed, watched, or watchlisted** are
excluded. Nightly training uses **NDCG@10 rollback** (a worse model is never
activated), an **advisory lock** (no concurrent trainings), and **atomic artifact
writes**; real-time `partial_fit` only mutates the in-memory active model.

Measured: **hit-rate@10 10.5% / NDCG@10 0.071** vs 2.6% popularity (≈4× baseline),
up from 5.7% for the plain hybrid and ~tied-with-popularity for content-only.

Underlying layers — a **collaborative model**
(PyTorch BPR matrix factorisation — [app/ml/collaborative.py](app/ml/collaborative.py))
and a **hybrid scorer** ([app/ml/hybrid.py](app/ml/hybrid.py)):

```
final = 0.45·collaborative + 0.35·content + 0.20·popularity      (collab available)
final = 0.70·content + 0.30·popularity                            (cold user)
```

Each component is min-max normalised per candidate set; already-reviewed movies
are excluded; the top-100 are written to `user_recommendations` (with the
component scores). Tables added by migration `0004`: `recommendation_jobs`,
`movie_popularity_stats`, `model_versions` (+ score columns on
`user_recommendations`).

**Evaluation (chronological leave-out):** the hybrid scores **hit-rate@10 ≈ 5.6%
vs 2.5% for popularity** on 45k held-out users — ~2.2× the baseline, where the
content-only recommender was roughly tied with popularity.

### Full training (nightly / manual)

```bash
python -m app.jobs.retrain_recommendations [--epochs N] [--max-users N]
```

Recomputes popularity → user embeddings → trains the collaborative MF → evaluates
on held-out reviews → activates the new `model_versions` row → refreshes
`user_recommendations`, recording status + metrics in `recommendation_jobs`.
Wire it into cron for nightly runs.

### Admin endpoints

- `POST /admin/recommendations/retrain` — queue a full retrain (runs in a worker
  thread; returns a `job_id` immediately).
- `GET /admin/recommendations/jobs` / `…/jobs/{id}` — job status + metrics.
- `GET /admin/recommendations/models` — model versions (active flag, metrics).

### Real-time updates

On each new/updated review the API **synchronously** recomputes that user's
content embedding (~tens of ms warm), then a **background task** runs the
collaborative `fit_partial` (skipped for cold users/movies) and rebuilds that
user's top-100 — so review submission stays fast and never triggers full
training. The reco context (movie matrix + active model) is cached in-process.

## Recommendation evaluation (admin dashboard)

`GET /stats/recommendations/evaluation` (admin) runs an **offline leave-out
evaluation** of the embedding recommender ([app/ml/evaluation.py](app/ml/evaluation.py)):
for a random sample of users it hides their most recent reviews, rebuilds the
user vector from only the earlier ones, then measures how well the
recommendations recover the held-out *liked* movies.

Returns:
- **Accuracy@K** — `precision_at_k`, `recall_at_k`, `hit_rate_at_k`,
  `ndcg_at_k`, `map_at_k`, with a **popularity baseline** for comparison.
- **Loss** — `rating_prediction.rmse` / `mae` on the 0–10 preference scale
  (preference predicted by similarity-weighted kNN over the user's train movies).

Query params: `sample_users` (default 300), `k` (10), `holdout` (0.2),
`like_threshold` (7.0), `min_interactions` (5), `seed` (42), `refresh`.

> Note: strict next-review ranking@K over a 31k-movie catalogue yields small
> absolute numbers (held-out hit-rate ~2–6%), and the content recommender is
> roughly on par with a popularity baseline there — content embeddings excel at
> "more like this" rather than predicting a user's exact next pick. The
> rating-prediction RMSE (~1.8–2.1) is the more stable quality signal.

## Notes

- Timestamp columns are `TIMESTAMP WITHOUT TIME ZONE` (per the schema); the code
  stores naive UTC values accordingly.
- `models/` (sentiment weights) and `artifacts/` (embedding pipeline) are
  gitignored and regenerated by the scripts above.

- `interactions.review_body_embedding` is an unsized `vector` column for
  arbitrary review embeddings; the 390‑dim check applies only to the
  movie/user embedding tables.
- The initial Alembic migration builds the schema directly from
  `app/models.py` metadata, so models and DB stay in sync. Add further
  changes with `alembic revision --autogenerate -m "..."`.
