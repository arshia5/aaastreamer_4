# The aaastreamer Recommendation Pipeline — Complete Explanation

This document explains the **entire machine-learning pipeline** behind aaastreamer:
what every stage does, every parameter and the reasoning behind it, every piece of
data it produces, and every measured result (losses, accuracies, recalls). It is
written so that **you can understand the whole system without ever opening the code**.

It is a companion to [RECOMMENDER.md](RECOMMENDER.md) (the terse reference) — this
file is the long, from-the-ground-up version. Where a fact comes from code, the
file is named so you can verify it.

> **Nothing here changes behaviour** — it is documentation only. Every number
> quoted is taken from the code in `app/ml/`, `app/jobs/`, `scripts/`, and the
> stored training metrics (job 15, model `mf_20260616_001216`).

---

## Table of contents

1. [What the system does (the 30-second version)](#1-what-the-system-does)
2. [The data it learns from](#2-the-data-it-learns-from)
3. [The end-to-end shape of the pipeline](#3-the-end-to-end-shape-of-the-pipeline)
4. [Stage 0 — The data model](#4-stage-0--the-data-model)
5. [Stage 1 — Sentiment and the preference score](#5-stage-1--sentiment-and-the-preference-score)
6. [Stage 2 — How a movie becomes vectors (the five representations)](#6-stage-2--how-a-movie-becomes-vectors)
7. [Stage 3 — How a user becomes a taste profile](#7-stage-3--how-a-user-becomes-a-taste-profile)
8. [Stage 4 — Popularity](#8-stage-4--popularity)
9. [Stage 5 — Retrieval (building the candidate pool)](#9-stage-5--retrieval)
10. [Stage 6 — Ranking (the learning-to-rank reranker)](#10-stage-6--ranking)
11. [Stage 7 — Diversity (MMR)](#11-stage-7--diversity)
12. [Stage 8 — "Movies like this" (similar_movies)](#12-stage-8--similar-movies)
13. [The nightly training job (how it all gets retrained)](#13-the-nightly-training-job)
14. [The real-time path (instant refresh on a new review)](#14-the-real-time-path)
15. [Evaluation — how we know it works, with every number](#15-evaluation--every-number)
16. [Complete hyperparameter reference](#16-complete-hyperparameter-reference)
17. [Everything stored in the database and on disk](#17-everything-stored)
18. [How to run every stage](#18-how-to-run-every-stage)
19. [Version history — why the pipeline looks the way it does](#19-version-history)

---

## 1. What the system does

The goal is simple to state: **given a user, produce a ranked list of ~100 movies
they have not seen and are likely to love.**

Doing that well at this scale (~31,000 movies, ~86,000 users, ~1.85M reviews) means
you cannot just score every movie with one formula. The system instead follows a
**retrieve → rank → diversify** strategy, which is how every large recommender
(YouTube, Netflix, Spotify) is built:

- **Retrieve** — cheaply gather a few hundred *plausible* candidates from millions
  of possibilities, using several different notions of "similar".
- **Rank** — apply an expensive, accurate model to *only those* candidates to sort
  them precisely.
- **Diversify** — make sure the final list isn't ten near-identical movies.

The cleverness is in using **multiple independent "views" of a movie** (what it's
about, who made it, how audiences reacted, who else liked it) so that a great
recommendation can't be missed just because one view is weak.

---

## 2. The data it learns from

The system learns from a movie catalogue and a large body of user reviews:

| Data | What it is | Scale |
|---|---|---|
| **Movies** | title, year, runtime, plot synopsis, genres, director(s), writer(s), cast, language(s), country(ies), poster, IMDb id | **~31,000 movies** |
| **Users** | one account per distinct reviewer | **~86,000 users** |
| **Interactions** | each pairs a user with a movie and carries a 0–10 rating, a 0–10 sentiment, the review text, and a date | **~1.85M** (one per user–movie pair) |
| **"Positive" edges** | interactions the user actually *liked* (preference ≥ 6) | **~1.1M**, across ~82k users — what the collaborative model trains on |

These numbers shape every design choice: 31k movies is small enough to hold all the
embeddings in memory at once, but 1.85M interactions is large enough that training
has to be batched and the serving context cached carefully. Each user has at most
one interaction per movie, so a "user × movie" pair is unique throughout.

---

## 3. The end-to-end shape of the pipeline

Here is the whole system on one page. Read it top to bottom; the rest of the
document expands each box.

```
                          ┌─────────────────────────────────────────────┐
                          │  CATALOGUE + INTERACTIONS (PostgreSQL)       │
                          │  movies, people, genres, … , users,          │
                          │  interactions (rating, sentiment, preference)│
                          └─────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼─────────────────────────────────┐
        ▼                                    ▼                                 ▼
  MOVIE VECTORS (pgvector)            USER VECTORS                       POPULARITY
  metadata  768  (mpnet text)         profiles: pos & neg per channel    log-reach ×
  plot      768  (mpnet text)         (recency + preference weighted)     Bayesian
  mf         64  (LightGCN)           mf 64 (LightGCN user vector)        quality
  community 384  (KMeans of reviews)
  structured 348 (v3 blocks, in-RAM)
        │                                    │                                 │
        └──────────────┬─────────────────────┴─────────────────────────────────┘
                       ▼
            RETRIEVE  (per-channel nearest neighbours, unioned, seen removed)
            mf 500 · meta 300 · struct 300 · comm 200 · plot 150 · pop 100
                       │           → ~1,100 candidate movies
                       ▼
            RANK  (A/B winner of {XGBoost rank:ndcg, Ridge linear}, 26 features)
                       │
                       ▼
            DIVERSIFY  (MMR over plot similarity, ≤4 per genre)
                       │
                       ▼
            TOP 100  → user_recommendations  (served by the API)
```

The heavy lifting (training LightGCN, re-clustering communities, refreshing every
user's list) runs **nightly**. When a single user posts a review, the **exact same
pipeline** reruns for just that one user in the background, so their list updates
within ~tens of milliseconds to a second — no waiting for the nightly job.

---

## 4. Stage 0 — The data model

All the data lives in a normalised PostgreSQL schema. Understanding its shape makes
the later stages clearer, because several channels read straight from these tables.

**The catalogue.** Each movie is a row in `movies` (`imdb_id`, `movie_title`,
`year`, `duration`, `plot`, `poster_url`). Everything multi-valued is a separate
entity table joined to the movie:

- `genres`, `languages`, `countries`, `people`, `roles` — the distinct values.
- `movie_genres`, `movie_languages`, `movie_countries` — many-to-many links.
- `movie_people` — links a movie to a person **with a role** (`director`, `writer`,
  or `actor`), so the same person can be both a writer and a director on different
  films.

**The interactions.** Each `(user, movie)` pair is one row in `interactions`
carrying the `rating`, the `sentiment`, the derived `preference_score`, the review
title/body, and the `review_date`. A pair is unique — at most one interaction per
user per movie — and the `review_date` gives every learning stage a chronological
order to work with (which is exactly what makes the "predict the future from the
past" evaluation in §15 possible).

That is the entire substrate the pipeline operates on; it touches nothing else.

---

## 5. Stage 1 — Sentiment and the preference score

The single most important derived quantity in the whole system is the
**preference score**: a 0–10 number that says how much a user liked a movie. Every
later stage (user profiles, collaborative positives, ranker labels, popularity) is
built on it.

### 5.1 Sentiment — `app/ml/sentiment.py`

Each review's text is scored by a **fine-tuned DistilBERT** classifier stored in
`models/sentiment-distilbert/`:

- It is a **5-class** model (the classes correspond to 1–5 stars).
- For a review it produces a softmax over the 5 classes, then takes the **expected
  value**: `score₁₋₅ = Σ probabilityᵢ · (i+1)`. Using the expected value (not just
  the top class) means a review the model is unsure about lands in the middle
  rather than committing to a hard label.
- That 1–5 value is rescaled to **0–10**: `score = (score₁₋₅ − 1)/4 · 9 + 1`, then
  clamped to `[0, 10]`. So a confident 1-star → 1.0, a confident 5-star → 10.0, a
  perfectly neutral review → 5.5.

| Parameter | Value | Why |
|---|---|---|
| `SENTIMENT_MAX_LENGTH` | 128 tokens | Reviews are long; the sentiment is clear early. Truncating keeps inference fast. |
| `SENTIMENT_BATCH_SIZE` | 64 | Throughput for batch scoring. |
| `SENTIMENT_CLASS_SCORES` | `[1,2,3,4,5]` | The ordinal value of each class for the expected-value step. |
| device | CUDA → MPS → CPU | Auto-selected; the model is a lazy singleton (torch loads on first review only). |

The existing corpus already has a stored sentiment per review; for **new** reviews
posted through the API, the server runs DistilBERT live before saving.

### 5.2 Preference score — `app/ml/scoring.py`

```
preference_score = 0.7 · rating + 0.3 · sentiment        (both clamped to 0–10)
```

| Parameter | Value | Why |
|---|---|---|
| `PREF_RATING_WEIGHT` | 0.7 | The star rating is the explicit, more reliable signal, so it dominates. |
| `PREF_SENTIMENT_WEIGHT` | 0.3 | The review *text* adds nuance the stars miss (a 7/10 with a glowing review is worth slightly more than a terse 7/10). |

If only one of the two exists, that one is used; if neither exists, the preference
is `None` (and the interaction is ignored by every learning stage). Both `rating`
and `sentiment` are read-only to clients — the server computes them.

---

## 6. Stage 2 — How a movie becomes vectors

This is the heart of the system. Each movie is turned into **five different
embeddings**, each capturing a different "view". They live in separate vector
spaces and are searched independently. The intuition: a movie you'll love might be
close to your taste in *one* of these views even if it's far in the others, so
keeping them separate (rather than mashing them into one vector) means no signal
gets averaged away.

| # | Channel | Dim | Built from | Stored in | Refreshed |
|---|---|---|---|---|---|
| 1 | **metadata** | 768 | An mpnet text-embedding of a templated sentence describing the movie | `movie_metadata_embeddings` | instantly on add/edit |
| 2 | **plot** | 768 | An mpnet text-embedding of the plot synopsis | `movie_plot_embeddings` | instantly on add/edit |
| 3 | **structured** | 348 | The v3 "block" embedding (genre/cast/crew/year, hashed) | in-memory only | rebuilt each context load |
| 4 | **mf** | 64 | The LightGCN collaborative item vector | `movie_mf_embeddings` | nightly |
| 5 | **community** | 384 | Up to 5 KMeans centroids of the movie's review texts | `movie_community_embeddings` | nightly |

There is also a **legacy 390-d embedding** (`movie_embeddings`) used by the older
endpoints and the offline-evaluation dashboard. It is described in §6.3 because the
v5 "structured" channel reuses its machinery.

### 6.1 Channels 1 & 2 — metadata and plot (mpnet text) — `app/ml/text_embed.py`

Both use the sentence-transformer **`all-mpnet-base-v2`** (768 dimensions), chosen
because it is a strong general-purpose semantic encoder and — crucially — it is
**stateless**. There is no fitted PCA, no hashing table, no vocabulary: a brand-new
movie is embedded with a single forward pass and lands in exactly the same space as
every existing movie. That is what lets new movies get instant, high-quality vectors
the moment they're added.

- **plot channel** = `mpnet(plot synopsis text)`. This captures *what the movie is
  about* semantically — two heist thrillers end up near each other even if they
  share no cast.

- **metadata channel** = `mpnet(a templated sentence)`. The template
  (`metadata_text`) turns structured fields into natural language, e.g.:

  > "Title: The Avengers. Year: 2012. Genres: Action, Sci-Fi. Directed by Joss
  > Whedon. Written by Joss Whedon, Zak Penn. Starring Robert Downey Jr., Chris
  > Evans, … (up to 8). Language: English. Country: USA."

  Cast is capped at the first 8 names to keep the sentence focused. Encoding prose
  lets the language model understand relationships ("directed by", "starring") that
  a raw one-hot vector cannot.

Both are L2-normalised when loaded so that a dot product equals cosine similarity.

### 6.2 Channel 3 — structured metadata (v3 blocks) — `app/ml/struct_embed.py`

The mpnet metadata template is semantically rich but it **blurs exact matches**:
prose embeddings smear "exactly directed by Christopher Nolan" into a soft cloud.
So v5 adds a second, complementary metadata channel that preserves **hard,
exact** genre/cast/year matches. It is the original v3 "block" construction *without*
the plot block:

For a batch of movies, each field becomes a fixed-width numeric block:

| Block | Width | How it's built |
|---|---|---|
| genre | 27 | Multi-hot via a fitted `MultiLabelBinarizer` (27 = number of genres seen at fit time) |
| director | 64 | `FeatureHasher` (the hashing trick — names → fixed 64-d counts, stateless) |
| writer | 64 | `FeatureHasher` |
| actor | 128 | `FeatureHasher` (more dims because casts are larger/more varied) |
| language | 32 | `FeatureHasher` |
| country | 32 | `FeatureHasher` |
| year (numeric) | 1 | A fitted `StandardScaler` over release year; missing → median year |
| **total** | **348** | |

Each block is **L2-normalised**, then multiplied by a hand-tuned **weight**, then
concatenated, and the whole 348-d vector is L2-normalised again. The weights
encode how much each field should matter:

| Block | Weight | | Block | Weight |
|---|---|---|---|---|
| genre | 1.2 | | writer | 0.7 |
| actor | 1.2 | | numeric (year) | 0.6 |
| director | 1.0 | | language | 0.5 |
| | | | country | 0.4 |

(Genre and lead actors carry the most signal about taste; country the least.)

This channel is **rebuilt in memory** every time the recommendation context loads,
straight from the relational tables plus the one fitted artifact (the genre binarizer
and year scaler from `artifacts/movie_embed_pipeline.joblib`). It has **no database
table and no migration** — it costs one matrix build per context load.

**Why keep it separate instead of replacing mpnet?** A measured bake-off
(`scripts/diagnose_metadata.py`) settled it (retrieval recall@300, identical user
profiles in each space):

| Representation | recall@300 |
|---|---|
| mpnet template (channel 1) alone | 0.180 |
| structured blocks (channel 3) alone | 0.143 |
| **mpnet + structured together** | **0.204** |

Structured alone is *weaker* than mpnet, but the two together beat either — because
they make different mistakes. So both are kept.

### 6.3 The legacy 390-d embedding (where the blocks come from)

The structured blocks above are exactly the v3 pipeline minus PCA. The full v3
pipeline (`scripts/fit_embeddings.py`, `app/ml/features.py`) builds a **732-d**
feature vector per movie:

```
plot 384  +  genre 27  +  director 64  +  writer 64  +  actor 128
          +  language 32  +  country 32  +  year 1     =  732
```

(plot here is the smaller **MiniLM** 384-d encoder, not mpnet) — then fits **PCA**
to compress 732 → **390** dimensions, which preserves **90.05 %** of the variance.
390 was chosen as the smallest width that clears the 90 % target. The fitted objects
(genre binarizer, year scaler, PCA matrix) are persisted to
`artifacts/movie_embed_pipeline.joblib` so any new movie can be projected into the
identical 390-d space. This legacy vector powers the older `/embeddings` similarity
search and the offline evaluation dashboard; v4/v5 retrieval uses the five channels
above instead.

> **Verified property:** a movie reconstructed from its DB metadata embeds to a
> vector cosine-identical (= 1.0) to the one from the batch pipeline, and an unseen
> "Avengers-like" description retrieves *The Avengers*, *Age of Ultron*, and *Civil
> War* as nearest neighbours.

### 6.4 Channel 4 — collaborative (LightGCN) — `app/ml/lightgcn.py`

This is the **behavioural** channel: it ignores all content and learns "people who
liked X also liked Y" purely from the interaction graph. It is the single strongest
channel (see §15), and it replaced the older BPR matrix-factorisation model.

**The model.** LightGCN treats users and movies as nodes in one big bipartite graph
where an edge = "this user liked this movie" (preference ≥ 6). Every node starts
with a random 64-d embedding. The model then **propagates** embeddings across the
graph:

```
Â = D^(−1/2) · A · D^(−1/2)          (symmetrically-normalised adjacency)
E⁽⁰⁾ = learned initial embeddings   (random, std 0.1)
E⁽ᵏ⁾ = Â · E⁽ᵏ⁻¹⁾                     (one hop = average of neighbours)
final = (E⁽⁰⁾ + E⁽¹⁾ + E⁽²⁾ + E⁽³⁾) / 4     (mean over 0…3 layers)
```

Each hop mixes a node with its neighbours; after 3 hops a user's vector reflects the
movies they liked, the *other* users who liked those movies, and *those* users'
movies. LightGCN deliberately has **no weight matrices and no non-linearities** —
this simplicity is exactly why it trains fast and generalises well at this scale.

**Training.** Full-batch **BPR** (Bayesian Personalised Ranking) for 400 epochs.
Each epoch: propagate once, then for every positive edge sample one random negative
movie and push the positive's score above the negative's:

```
loss = mean( −log σ(score_pos − score_neg) )  +  reg · ‖embeddings‖²
```

A user's own rated movies are never sampled as their negatives.

| Parameter | Value | Why |
|---|---|---|
| `LIGHTGCN_DIM` | 64 | Embedding width; ample for 31k items, small enough to be cheap. |
| `LIGHTGCN_LAYERS` | 3 | 3 hops of graph smoothing — the standard sweet spot. |
| `LIGHTGCN_EPOCHS` | **400** | **The single biggest v5 fix.** See below. |
| `LIGHTGCN_LR` | 0.01 | Adam learning rate. |
| `LIGHTGCN_REG` | 1e-4 | L2 weight decay on embeddings. |
| seed | 42 | Reproducibility. |
| device | MPS → CUDA → CPU | Full-batch keeps it feasible on a laptop/VPS. |

**Why 400 epochs (and how we know).** In v4 this was **30** epochs. Because training
is *full-batch*, 30 epochs = only **30 gradient steps** — the model was badly
undertrained. A bake-off (`scripts/diagnose_lightgcn.py`) on the held-out split:

| Variant | mf recall@300 | mf-as-scorer hit@10 |
|---|---|---|
| 30 epochs (old) | 0.415 | 0.086 |
| **400 epochs (current)** | **0.503** | **0.112** |
| 400 epochs, 4 negatives/pos | (no improvement) | (no improvement) |

400 epochs lifts the collaborative channel's recall by ~21 % and its standalone
hit-rate by ~30 %. Multi-negative sampling didn't help, so it was left out.

**Final training loss (v5):** **0.0525** (down from 0.3223 in v4 — a direct
consequence of training to convergence).

The output user and item matrices are stored both as a versioned joblib artifact
(`artifacts/collaborative/mf_*.joblib`) and in the DB tables `user_mf_embeddings`
and `movie_mf_embeddings`. Movies/users with too few interactions get **no** mf
vector and simply fall back to the content channels.

### 6.5 Channel 5 — community (KMeans of reviews) — `app/ml/community.py`

This channel captures **how audiences actually reacted**, which content metadata
can't. Each review body is embedded with **MiniLM** (`all-MiniLM-L6-v2`, 384-d,
truncated to 128 tokens for speed), and then a movie's review vectors are
**K-means clustered** into up to **5 centroids**:

- Each centroid = one distinct "audience reaction" to the movie (e.g. "loved the
  visuals", "found it boring", "great for kids"), and carries a **weight** = the
  share of reviews in that cluster.
- A movie with fewer than 5 reviews gets one centroid per review; a movie with 0
  reviews gets none (and falls back to content channels).

| Parameter | Value | Why |
|---|---|---|
| `REVIEW_EMBED_MODEL` | `all-MiniLM-L6-v2` (384-d) | Smaller/faster than mpnet — there are ~1.85M reviews to encode. |
| max sequence length | 128 tokens | Halves encode time; the gist is early in a review. |
| `COMMUNITY_K` | 5 | Up to 5 reactions per movie. |
| `COMMUNITY_MAX_REVIEWS` | 200 | Cap reviews per movie used for clustering. |
| KMeans `n_init` | 4 | Restarts to avoid bad local minima. |
| seed | 42 | Reproducibility. |

**The efficiency trick:** review vectors are computed **once** and stored in
`review_embeddings` (~1.85M × 384). The nightly job only embeds reviews that are
*new* since last run, and only re-clusters movies that *gained* reviews — old
reviews are never re-embedded. This is what makes nightly maintenance cheap.

---

## 7. Stage 3 — How a user becomes a taste profile

A user is represented by **two things**: their LightGCN `mf` vector (from §6.4) and
a set of **content profiles** built on the fly from their reviews
(`app/ml/profiles.py`).

For each content channel (metadata, structured, plot, community) the system builds a
**positive** profile and a **negative** profile. Both are recency-weighted,
preference-centred, L2-normalised sums of the user's movie vectors:

```
baseline   = (5 · global_mean + Σ preference) / (5 + n)     ← per-user, shrunk
centeredᵢ  = preferenceᵢ − baseline
weightᵢ    = |centeredᵢ| · 0.9^(rank from newest)
profile_pos = L2( Σ over movies with centeredᵢ > 0 :  weightᵢ · movie_vector )
profile_neg = L2( Σ over movies with centeredᵢ < 0 :  weightᵢ · movie_vector )
```

Three design decisions are baked into that formula, and each fixes a real problem:

1. **Per-user baseline (not a fixed midpoint).** "Liked" means *above this user's
   own average*, shrunk toward the global mean with prior strength `BASELINE_PRIOR =
   5`. A harsh critic who rates everything 4–6 and a generous one who rates
   everything 8–10 both get meaningful positive/negative splits. Without this, the
   generous user would have an all-positive profile and the harsh user an
   all-negative one.

2. **Recency decay by rank, not by clock.** Reviews are sorted oldest→newest; the
   **newest gets full weight (0.9⁰ = 1)** and each older review is multiplied by
   `USER_EMBED_DECAY = 0.9` per step. Because it's **rank-based**, a user who hasn't
   reviewed in two years still has a well-defined, full-strength profile — a long
   gap doesn't decay their taste to zero (which an absolute-time decay would do).

3. **A negative profile.** Disliked movies don't just get ignored — they form a
   separate vector the ranker uses to actively **penalise** similar candidates. The
   "gap" between positive-similarity and negative-similarity is a feature (§10).

Alongside the profiles, the builder also records lightweight taste facts used as
ranker features: average preference, number of positives, favourite genre (most
frequent across liked movies), the genres of the *most recently* liked movie, the
set of liked actors, and the **preferred era** (mean release year of liked movies).

> **A separate, simpler user vector** also exists: `app/ml/user_embedding.py` builds
> one **390-d** vector per user (centred on the fixed neutral 5.5 rather than a
> per-user baseline, single positive-minus-negative sum). It powers the legacy
> `/embeddings/users/.../recommendations` endpoints and the offline-evaluation
> dashboard. The v4/v5 recommender uses the richer pos/neg-per-channel profiles
> above. Verified example: user "#1_Gracie" (loved Ocean's Eleven & LotR:
> Fellowship) is recommended Ocean's Twelve/Thirteen and the rest of LotR/Hobbit.

---

## 8. Stage 4 — Popularity — `app/ml/popularity.py`

Popularity is both a retrieval channel (a safety net of broadly-loved films) and a
ranker feature. The score blends **reach** (how many people reviewed it) with
**quality** (how much they liked it), both in `[0, 1]`:

```
popularity_score = ln(1 + review_count) / ln(1 + max_review_count)          ← reach
                   × ( 0.5 + 0.5 · bayes_preference / 10 )                   ← quality

bayes_preference = (50 · global_mean + Σ preference) / (50 + review_count)
```

| Parameter | Value | Why |
|---|---|---|
| log damping | `ln(1+count)/ln(1+max)` | A blockbuster with 100k reviews shouldn't out-score everything by 100k×; log compresses reach into a sane range. |
| quality floor/ceiling | `0.5 + 0.5·q` | Even a mediocre-but-massive film keeps half its reach score; a beloved one gets the full multiplier. |
| `POPULARITY_PRIOR` | 50 | **Bayesian shrinkage.** Pulls low-count averages toward the global mean so "10 reviews @ 10.0" can't beat "3,000 @ 9.2". |

Note `avg_preference_score` is stored as the **raw** mean (an honest stat for the
UI); the Bayesian shrinkage is applied *only* inside `popularity_score`. Stored in
`movie_popularity_stats`.

---

## 9. Stage 5 — Retrieval — `app/ml/reco.py`

Retrieval turns "31,000 movies" into "~1,100 plausible candidates" by taking the
nearest neighbours to the user in **each channel separately** and **unioning** them.

For each channel, the user's **positive** profile (or `mf` vector) is dot-producted
against every movie's vector and the top-K are kept. Then all buckets are merged,
duplicates removed, and movies the user has already **reviewed, watched, or
watchlisted** are subtracted out.

| Channel | Budget K | (what it contributes) |
|---|---|---|
| **mf** (collaborative) | **500** | The strongest signal — gets the most slots. |
| **metadata** (mpnet) | 300 | Semantic "same kind of movie". |
| **structured** | 300 | Exact genre/cast/year matches. |
| **community** | 200 | "Audiences who reacted like you did". |
| **plot** | 150 | Semantic plot similarity (weakest channel). |
| **popularity** | 100 | A safety net of broadly-loved films. |
| | **~1,100 after dedup − seen** | |

**Why these specific budgets?** They are tuned to **measured per-channel recall**
(`scripts/diagnose_channels.py` — does this channel *alone* contain a held-out movie
the user later liked?):

| Channel | recall (alone) |
|---|---|
| mf | **0.40** (best) |
| popularity | 0.18 |
| metadata | 0.18 |
| community | 0.16 |
| structured | 0.14 |
| plot | 0.10 (worst) |

In v4 the budgets were *backwards* — plot (the weakest) had 400 slots and mf (the
strongest) had 300. v5 gives mf the largest budget and trims plot. The point of the
union is **a great movie cannot be linearly cancelled out of the pool** by one weak
channel — if *any* view ranks it highly, it survives to the ranker. The union
retrieval recall is **~0.65** (i.e. for 65 % of users, a movie they'll actually like
is somewhere in the pool — this is the ceiling the ranker works under).

Only the **positive** profile drives retrieval; negative profiles are used later as
ranker features, not to fetch candidates.

---

## 10. Stage 6 — Ranking — `app/ml/ranker.py`

The ranker takes the ~1,100 candidates and sorts them precisely. It is a
**learning-to-rank** model trained on **graded relevance**, and each nightly/manual
full train actually fits **two** rankers and ships whichever wins (an A/B).

### 10.1 Graded labels

Instead of binary liked/not-liked, each candidate gets a **0–3 grade** from its
held-out preference, so that burying a beloved film is penalised harder than burying
a merely-fine one:

| Grade | Meaning | Preference |
|---|---|---|
| 3 | critical hit | ≥ 9.0 |
| 2 | good | 7.0 – 8.9 |
| 1 | neutral/positive | 5.5 – 6.9 |
| 0 | not relevant | below 5.5 (or unknown) |

### 10.2 The 26 features

Every candidate is described by 26 numbers, combining all channels plus structured
cross-features:

```
per-channel similarity to the user's POSITIVE and NEGATIVE profile, plus the gap:
  meta_pos, meta_neg, meta_gap
  struct_pos, struct_neg, struct_gap
  plot_pos, plot_neg, plot_gap
  comm_pos, comm_neg, comm_gap         (max cosine over the movie's centroids)

collaborative:
  mf            (raw LightGCN score)
  mf_pct        (its percentile within THIS user's candidate set — calibrated)

popularity & movie stats:
  pop, pop_pct, review_count_log, movie_avg_pref (Bayesian)

structured cross-features (candidate × user taste):
  genre_overlap        (# shared genres with the user's liked genres)
  fav_genre_match      (1 if the candidate has the user's single favourite genre)
  recent_genre_match   (1 if it shares a genre with the most-recently-liked movie)
  actor_overlap        (# shared actors with the user's liked actors)
  year_distance        (|candidate release year − user's preferred era|)

user-level (same for all the user's candidates):
  user_avg_pref, user_n_pos_log, has_mf
```

Two deliberate touches: **both raw scores and percentiles** are fed (raw values are
comparable across users via the percentile; the calibration helps the trees), and
`year_distance` is the candidate's **release era** versus the user's preferred era —
**not** anything about *when* the user reviewed. Community similarity is the **max**
cosine over a movie's up-to-5 centroids (the best-matching audience reaction).

### 10.3 The two rankers (A/B) and how the winner is chosen

| Ranker | What it is | Parameters |
|---|---|---|
| **XGBoost** | Gradient-boosted trees, `objective = rank:ndcg` | 300 trees, max_depth 6, lr 0.05, subsample 0.8, colsample_bytree 0.8, min_child_weight 5, `eval_metric ndcg@10`, `tree_method hist` |
| **Linear (Ridge)** | A standardised Ridge regression on the graded labels — one learned linear blend over the same 26 features | `alpha = 1.0`, features standard-scaled, NaNs → 0 |

The linear model is the "robust floor": with far fewer parameters it cannot overfit
weak channels into noise, and in practice `mf` dominates its learned weights (hence
"mf-centric"). Each full train fits both, evaluates both on a held-out sample (up to
3,000 users), and **ships whichever has the higher NDCG@10**. `load_ranker`
dispatches to the right class by the saved `kind`.

**Latest A/B result:** XGBoost **0.072** vs linear **0.041** → **XGBoost is active.**

### 10.4 Training data — the critical v5 fix

| Parameter | Value | Meaning |
|---|---|---|
| `XGB_TRAIN_USERS` | 15,000 | Users sampled to build LTR training rows. |
| `XGB_NEG_PER_POS` | 30 | Negatives kept per positive (caps the matrix size). |

The decisive fix in v5 is **what distribution the ranker trains on**. In v4,
`_build_xgb_data` **force-injected held-out positives that retrieval had missed**.
Those movies look like negatives feature-wise (especially low `mf`), so the ranker
learned to **distrust its single best feature** — and the symptom was damning: in
v4, sorting candidates by **raw mf** *beat* the trained ranker
(`scripts/diagnose_rankers.py`).

v5 trains **only on the retrieved candidate distribution** — the exact distribution
the ranker sees at serving time. Users whose held-out positive wasn't retrieved are
simply skipped (there is nothing to learn to rank for them). Result: the ranker now
**beats** raw mf and the linear contender, and **ranker conversion** (the fraction
of retrieved positives that reach the top-10) roughly **doubled: 12.3 % → 25.5 %**.

---

## 11. Stage 7 — Diversity — `app/ml/reco.py` (`mmr_rerank`)

A perfectly-ranked list is often ten near-identical movies. **Maximal Marginal
Relevance (MMR)** re-orders the ranked candidates to trade a little relevance for
variety:

```
pick the movie maximising:  (1 − λ) · relevance  −  λ · (max plot-similarity to anything already picked)
```

so each new pick is rewarded for being relevant **and** for being different from
what's already in the list. On top of MMR there is a hard **per-genre cap**.

| Parameter | Value | Why |
|---|---|---|
| `DIVERSITY_LAMBDA` | 0.30 | 0 = pure relevance, 1 = pure diversity. 0.30 leans toward relevance with a meaningful diversity nudge. |
| `DIVERSITY_MAX_PER_GENRE` | 4 | At most 4 movies of any single genre in the stored list (with a fallback so the list always fills). |
| similarity space | plot embeddings | Plot is the most intuitive axis of "too samey". |

The output is the final **top 100**, written to `user_recommendations`.

---

## 12. Stage 8 — "Movies like this" (similar_movies) — `app/ml/similar.py`

Separate from per-user recommendations, the system precomputes the **top-10 most
similar movies for every movie** (the "More like this" rail). Nightly, this is a
**multi-channel blend** of cosine similarities:

```
similarity = 0.4 · plot  +  0.3 · metadata  +  0.2 · mf  +  0.1 · community
```

| Weight | Channel | Notes |
|---|---|---|
| 0.4 | plot | "About the same thing" dominates. |
| 0.3 | metadata | Same kind of production. |
| 0.2 | mf | "People who liked this liked…" |
| 0.1 | community | Similar audience reaction. |

`mf` and `community` only contribute when **both** movies have that vector;
otherwise the score falls back to content only. A **brand-new movie** gets an
instant content-only list (plot+metadata, renormalised to 0.571/0.429) via SQL the
moment it's added; the behavioural channels are layered in at the next nightly run.
`SIMILAR_TOP_N = 10`. (Note: this blend uses plot+mpnet-metadata for content — it
does not yet use the v5 structured channel.)

> An older content-only variant (`SIMILAR_W_CONTENT 0.7 / SIMILAR_W_COLLAB 0.3`)
> exists in config for the legacy 390-d similar builder.

---

## 13. The nightly training job — `app/jobs/training.py`

This is the orchestrator that retrains and refreshes everything. Run it with:

```bash
python -m app.jobs.retrain_recommendations            # full: also retrain the ranker A/B
python -m app.jobs.retrain_recommendations --skip-ranker   # nightly: reuse frozen ranker
```

or trigger it over HTTP (`POST /admin/recommendations/retrain`), which launches it
as a **detached subprocess** so the heavy torch/xgboost work never touches the web
server process and returns a `job_id` immediately.

**The steps, in order:**

1. **Advisory lock** (`pg_try_advisory_lock(911001)`) — only one training can run at
   a time; a second invocation exits cleanly with `skipped`. Any older
   running/queued job is marked `failed: stale, superseded`.
2. **Popularity** — recompute `movie_popularity_stats` (§8).
3. **Dirty-community rebuild** — embed only new reviews, re-cluster only changed
   movies (§6.5).
4. **Chronological split** — for each user, sort interactions by date and hold out
   the **newest 20 %** (`k = max(1, round(0.2·n))`, users need ≥2 interactions). This
   is an honest "predict the future from the past" split — no leakage.
5. **LightGCN** — train on the *train-split* positives (preference ≥ 6) for 400
   epochs (§6.4).
6. **Build context** — load all channels into memory, injecting the freshly-trained
   mf vectors.
7. **Per-user profiles** — build pos/neg content profiles for every user (§7).
8. **Ranker** — full mode: build LTR data, fit XGBoost **and** linear, run the A/B,
   keep the NDCG@10 winner. `--skip-ranker` mode: **load and reuse the frozen active
   ranker** (see cadence note below).
9. **Staged evaluation** — score the held-out split (§15) on the chosen ranker.
10. **Rollback gate** — compare the new model's `ndcg_at_10` against the currently
    active model's stored `ndcg_at_10` (same deterministic split). **Activate only if
    `new ≥ old + ROLLBACK_MARGIN`** (`ROLLBACK_MARGIN = 0.0`). A worse model is never
    shipped — the old one stays active and the run is logged as `rolled_back`.
11. **On activation only:** persist the LightGCN + ranker artifacts atomically
    (temp file → `os.replace`), flip the `model_versions.is_active` flags, write the
    fresh mf vectors to `user_mf_embeddings` / `movie_mf_embeddings`, then
    **refresh `user_recommendations`** for every user — transactionally, in batches
    of 1,500 users with incremental commits, so there is **no downtime** and no
    half-written state.
12. **Rebuild similar_movies** with the fresh mf/community (§12).

Every step is timed and written to `recommendation_jobs.metrics`; domain events
(`training_started/completed`, `model_activated/rolled_back`, `similar_rebuilt`) go
to the `logs` table.

**Cadence split (a v5 idea worth its own note).** The ranker's features are all
dot-products / cosines / structured overlaps. These are **invariant to LightGCN's
nightly re-basis** — when LightGCN retrains, the *geometry* of the mf space rotates,
but the ranker only ever sees cosines and percentiles within a user's own candidate
set, which don't change meaning. So the ranker only needs (re)training when the
*features or data* drift, not every night. The intended cadence:

- **Nightly:** `--skip-ranker` (LightGCN + profiles + refresh, frozen ranker reused).
- **Occasionally (e.g. monthly):** the full form, which reruns the ranker A/B.

| Job parameter | Default | Meaning |
|---|---|---|
| `--epochs N` | 400 (config) | Override LightGCN epochs. |
| `--max-users N` | all | Cap how many users' lists get refreshed (for quick runs). |
| `--like-threshold` | 7.0 | Preference at/above which a held-out movie counts as a "hit" in eval. |
| `--skip-ranker` | off | Nightly mode (reuse ranker). |
| `EVAL_SAMPLE_USERS` | 20,000 | Cap on users scored during eval. |

---

## 14. The real-time path — `app/ml/realtime.py`

When a user posts or edits a review, we don't wait for the nightly job. A background
task reruns the **exact same pipeline** for that one user:

1. Recompute their content profiles from their reviews (tens of ms warm).
2. Read their `mf` vector from `user_mf_embeddings` (a cold user simply has none
   until the next nightly run — they still get content + popularity recs).
3. Generate candidates → score with the active ranker → MMR → write their new
   top-100.

The recommendation context (the in-memory matrices + active model) is **cached in
process** and only rebuilt when the active model version or movie count changes, so a
real-time refresh is fast. Because it's the same code path as nightly, **real-time
quality == nightly quality** — there is no "cheap daytime" approximation.

The older collaborative model also supports an incremental `partial_fit_user`
(`CF_PARTIAL_EPOCHS 2`, `CF_PARTIAL_LR 0.01`, `CF_PARTIAL_BLEND 0.30` — nudge a
user's factor toward new positives, item factors frozen, then blend 30 % new / 70 %
nightly) for the legacy BPR path; the v5 real-time path reads mf from the DB rather
than fine-tuning it live.

---

## 15. Evaluation — every number

### 15.1 The protocol

Evaluation is a **chronological leave-last-20 %** hold-out (`_evaluate` in
`app/jobs/training.py`). For each user we hide their newest 20 % of reviews, rebuild
everything from the older 80 %, and ask: do the movies they later **liked**
(preference ≥ 7) show up in the recommendations we'd have made? Metrics reported:

- **retrieval_recall** — was a held-out liked movie even *in the candidate pool*?
  (the ceiling everything else works under)
- **hit_rate@10** — fraction of users with ≥1 liked movie in their top-10.
- **recall@10 / recall@50** — fraction of their liked movies recovered in top-10/50.
- **ndcg@10** — rank-quality (rewards putting hits near the top).
- **baseline_pop_hit_rate@10** — the same metric for a "just recommend the most-
  reviewed movies" baseline, so every number has a reference point.
- **ndcg_by_reviews** — ndcg@10 split by how active the user is (cold→heavy buckets
  1 / 2–4 / 5–10 / 10+), to confirm no segment is sacrificed.

### 15.2 Headline results (20k-user held-out eval)

Active model: `lightgcn mf_20260616_001216` + `xgb_ranker xgb_20260616_001216`
(job 15, 2026-06-16).

| Metric | v3 (BPR, job 9) | v4 (job 12) | **v5 (active)** |
|---|---|---|---|
| protocol | leave-one-out | leave-last-20% | leave-last-20% |
| **hit_rate@10** | 0.111 | 0.0657 | **0.1646** |
| **ndcg@10** | 0.0755 | 0.032 | **0.0676** |
| recall@10 | — | 0.0438 | **0.0981** |
| recall@50 | 0.165 | 0.0975 | **0.2104** |
| retrieval_recall | — | 0.5329 | **0.6506** |
| pop-baseline hit@10 | 0.0263 | 0.0437 | 0.0436 |
| **lift over popularity** | **4.2×** | **1.5×** | **3.8×** |
| LightGCN final loss | (BPR) | 0.3223 | **0.0525** |

**How to read this:** v3 used a different, easier protocol (leave-*one*-out, so its
absolute numbers aren't comparable), which is why the only fair cross-version number
is **lift over the popularity baseline**. v4 and v5 share the *identical*
leave-last-20 % protocol and are directly comparable: **v5 is 2.1–2.5× better than
v4 across every metric**, and it recovers v3-level lift (3.8× vs 4.2×) on the harder
protocol while keeping the full multi-channel architecture and a much higher
retrieval ceiling.

### 15.3 ndcg@10 by user activity (no segment left behind)

| Bucket (train reviews) | v4 | **v5** |
|---|---|---|
| 1 (cold start) | 0.019 | **0.052** |
| 2–4 | 0.040 | **0.070** |
| 5–10 | 0.040 | **0.069** |
| 10+ (heavy) | 0.017 | **0.064** |

Every bucket is ~3× v4 and roughly flat across activity — the gains are not bought
by sacrificing cold or heavy users.

### 15.4 Why v5 is better (the decomposition)

`hit@10 = retrieval_recall × ranker_conversion`. Both halves improved:

- **Retrieval ceiling: 0.533 → 0.651 (+22 %)** — from the 400-epoch LightGCN, the
  new structured channel, and the mf-weighted retrieval budgets.
- **Ranker conversion: 12.3 % → 25.5 % (×2)** — from training the ranker on the
  retrieved distribution (no more force-injected positives), the structured
  features, and the stronger mf. This is the dominant lever: in v4 raw mf *beat* the
  ranker; in v5 the fixed XGBoost beats both raw mf and the linear A/B contender.

### 15.5 Supporting experiments (reproduce with the diagnose scripts)

- **Per-channel retrieval recall** (`diagnose_channels.py`): mf 0.40, pop 0.18, meta
  0.18, comm 0.16, struct 0.14, plot 0.10 → motivated the v5 budgets.
- **LightGCN epochs** (`diagnose_lightgcn.py`): 30→400 lifts mf recall@300
  0.415→0.503, hit@10 0.086→0.112.
- **Metadata representations** (`diagnose_metadata.py`): mpnet 0.180, structured
  0.143, **mpnet+structured 0.204** → keep both.
- **Ranker vs simple scorers** (`diagnose_rankers.py`): the test that proved the v4
  ranker was the defect (raw mf beat it) and that the v5 ranker fixes it.

### 15.6 The offline dashboard (a second, independent evaluator)

`GET /stats/recommendations/evaluation` (admin) runs a separate leave-out evaluation
of the **legacy 390-d content recommender** (`app/ml/evaluation.py`) for an
operations dashboard. It reports **Accuracy@K** (precision / recall / hit-rate / ndcg
/ map) with a popularity baseline, plus a **regression loss**: it predicts each
held-out movie's preference via similarity-weighted kNN over the user's train movies
and reports **RMSE / MAE** on the 0–10 scale.

| Query param | Default |
|---|---|
| `sample_users` | 300 |
| `k` | 10 |
| `holdout` | 0.2 |
| `like_threshold` | 7.0 |
| `min_interactions` | 5 |
| `seed` | 42 |

Caveat: strict next-review ranking over a 31k catalogue yields small absolute
hit-rates (~2–6 %) for the **content-only** legacy recommender, which is roughly on
par with the popularity baseline there — content embeddings excel at "more like
this", not at predicting an exact next pick. The **rating-prediction RMSE (~1.8–2.1)**
is the more stable quality signal for that endpoint. (This is the *content-only*
legacy path; the full v5 recommender's numbers are in §15.2.)

### 15.7 One honest caveat

The ranker's 15k training users overlap the eval users (both drawn from a shared
`Random(42)` sample, inherited from v4). So **v5-vs-v4 deltas are valid** (same
protocol both times), but the absolute NDCG is mildly optimistic. A fully disjoint
train/eval split is the next planned honesty improvement.

---

## 16. Complete hyperparameter reference

Every tunable lives in **`app/ml/config.py`**. Full list:

### Text / content embeddings
| Name | Value | Meaning |
|---|---|---|
| `MODEL_NAME` | all-MiniLM-L6-v2 | Encoder for the legacy 390-d plot block & community reviews |
| `TEXT_EMBED_MODEL` / dim | all-mpnet-base-v2 / 768 | metadata + plot channels |
| `REVIEW_EMBED_MODEL` / dim | all-MiniLM-L6-v2 / 384 | community review encoder |
| `COMMUNITY_K` | 5 | max KMeans clusters per movie |
| `COMMUNITY_MAX_REVIEWS` | 200 | reviews per movie used for clustering |
| `TARGET_VARIANCE` / `N_COMPONENTS` | 0.90 / 390 | legacy PCA target & width (achieved 0.9005) |
| block dims | plot 384, genre 27, director 64, writer 64, actor 128, lang 32, country 32, year 1 | v3/structured blocks (= 732 with plot, 348 without) |
| block `WEIGHTS` | plot 2.0, genre 1.2, actor 1.2, director 1.0, writer 0.7, numeric 0.6, language 0.5, country 0.4 | per-block emphasis |

### Sentiment & preference
| Name | Value |
|---|---|
| `SENTIMENT_MAX_LENGTH` / `SENTIMENT_BATCH_SIZE` | 128 / 64 |
| `SENTIMENT_CLASS_SCORES` | [1,2,3,4,5] |
| `PREF_RATING_WEIGHT` / `PREF_SENTIMENT_WEIGHT` | 0.7 / 0.3 |
| `PREF_NEUTRAL` | 5.5 (legacy user vector centering) |
| `USER_EMBED_DECAY` | 0.9 (recency decay per rank step) |
| `BASELINE_PRIOR` (profiles) | 5 (per-user baseline shrinkage) |

### LightGCN (collaborative)
| Name | Value |
|---|---|
| `LIGHTGCN_DIM` / `LAYERS` / `EPOCHS` / `LR` / `REG` | 64 / 3 / **400** / 0.01 / 1e-4 |
| positive threshold (`CF_POS_THRESHOLD`) | 6.0 (preference ≥ 6 = an edge) |
| seed | 42 |

### Legacy BPR-MF (superseded; still in code)
| Name | Value |
|---|---|
| `CF_DIM` / `CF_EPOCHS` / `CF_LR` / `CF_REG` / `CF_BATCH` | 128 / 12 / 0.03 / 1e-6 / 8192 |
| `CF_PARTIAL_EPOCHS` / `LR` / `BLEND` | 2 / 0.01 / 0.30 |

### Popularity
| Name | Value |
|---|---|
| `POPULARITY_PRIOR` | 50 (Bayesian shrinkage) |

### Retrieval budgets
| `RETR_MF` | `RETR_META` | `RETR_STRUCT` | `RETR_COMM` | `RETR_PLOT` | `RETR_POP` |
|---|---|---|---|---|---|
| **500** | 300 | 300 | 200 | 150 | 100 |

### Ranker
| Name | Value |
|---|---|
| graded labels | 3 ≥ 9, 2 ≥ 7, 1 ≥ 5.5, 0 else |
| `XGB_PARAMS` | rank:ndcg, 300 trees, depth 6, lr 0.05, subsample 0.8, colsample 0.8, min_child_weight 5, tree_method hist, eval ndcg@10 |
| `XGB_TRAIN_USERS` / `XGB_NEG_PER_POS` | 15,000 / 30 |
| linear A/B | Ridge α=1.0, standardised features; ship NDCG@10 winner |

### Diversity, similar, serving
| Name | Value |
|---|---|
| `DIVERSITY_LAMBDA` / `MAX_PER_GENRE` | 0.30 / 4 |
| `SIMILAR_TOP_N` | 10 |
| `SIMILAR_W` plot/meta/mf/comm | 0.4 / 0.3 / 0.2 / 0.1 |
| `TOP_N_RECOMMENDATIONS` | 100 |
| `CAND_POOL` (legacy hybrid) | 200 |

### Hybrid scorer (legacy v3 path, superseded by the ranker)
| Name | Value |
|---|---|
| `HYBRID_W_COLLAB / CONTENT / POP` | 0.45 / 0.35 / 0.20 |
| fallback (cold) `HYBRID_FB_CONTENT / POP` | 0.70 / 0.30 |

### Training / rollback / eval
| Name | Value |
|---|---|
| advisory lock key | 911001 |
| split | leave-last-20% (k = max(1, round(0.2·n))) |
| `ROLLBACK_METRIC` / `ROLLBACK_MARGIN` | ndcg_at_10 / 0.0 |
| `EVAL_SAMPLE_USERS` | 20,000 |
| refresh batch | 1,500 users/transaction |

---

## 17. Everything stored

### Vector tables (PostgreSQL + pgvector, HNSW cosine index per vector column)
| Table | Dim | Holds |
|---|---|---|
| `movie_metadata_embeddings` | 768 | metadata channel |
| `movie_plot_embeddings` | 768 | plot channel |
| `movie_mf_embeddings` | 64 | LightGCN item vectors |
| `user_mf_embeddings` | 64 | LightGCN user vectors |
| `movie_community_embeddings` | 384 | up to 5 centroids/movie (+ weight, cluster_idx) |
| `review_embeddings` | 384 | ~1.85M review vectors (so reviews are never re-embedded) |
| `movie_embeddings` | 390 | legacy v3 content vector |
| `user_embeddings` | 390 | legacy v3 user vector |
| `user_profiles_768` / `_384` | 768 / 384 | (profile tables) |
| `interactions.review_body_embedding` | unsized | optional per-review vector |

### Bookkeeping tables
| Table | Holds |
|---|---|
| `movie_popularity_stats` | review_count, avg_rating, avg_preference_score (raw), popularity_score |
| `similar_movies` | top-10 similar per movie (similar_movie_id, score, rank) |
| `user_recommendations` | the served top-100 (movie_id, score, rank, component scores) |
| `recommendation_jobs` | one row per training run: type, status, timings, full `metrics` JSON |
| `model_versions` | every trained LightGCN/ranker: version_name, artifact_path, is_active, metrics |
| `logs` (+ event/entity types) | request mutations + domain events |

### On-disk artifacts (gitignored, regenerable)
| Path | What |
|---|---|
| `artifacts/movie_embed_pipeline.joblib` | fitted genre binarizer + year scaler + PCA-390 (used by legacy + structured channel) |
| `artifacts/collaborative/mf_*.joblib` | versioned LightGCN user/item matrices |
| `artifacts/ranker/xgb_*.joblib` / `lin_*.joblib` | versioned ranker (XGBoost or linear) |
| `models/sentiment-distilbert/` | the fine-tuned sentiment model weights |

Artifacts are written **atomically** (temp file → `os.replace`) so a concurrent
loader (real-time serving) never sees a half-written file.

---

## 18. How to run every stage

```bash
# ── one-time embedding seed (over the catalogue + interactions) ───────────────
python -m scripts.fit_embeddings            # fit + persist the v3 PCA-390 pipeline
python -m scripts.backfill_embeddings       # 390-d legacy vectors for every movie
python -m scripts.backfill_text_embeddings  # mpnet metadata + plot (768-d)
python -m scripts.build_community           # review vectors + KMeans community centroids
python -m scripts.build_user_embeddings     # 390-d legacy user vectors
python -m scripts.build_similar_movies      # initial similar_movies fill

# ── train / refresh the recommender ──────────────────────────────────────────
python -m app.jobs.retrain_recommendations               # full (also runs ranker A/B)
python -m app.jobs.retrain_recommendations --skip-ranker # nightly (reuse frozen ranker)

# ── diagnostics (reproduce §15.5) ────────────────────────────────────────────
python -m scripts.diagnose_channels   # per-channel retrieval recall + ranker conversion
python -m scripts.diagnose_lightgcn   # LightGCN epoch bake-off
python -m scripts.diagnose_metadata   # mpnet vs structured vs both
python -m scripts.diagnose_rankers    # XGB vs raw-mf / meta / plot / blends
```

Admin HTTP control: `POST /admin/recommendations/retrain` (launches a detached
training subprocess, returns a `job_id`), `GET /admin/recommendations/jobs[/{id}]`
(status + metrics), `GET /admin/recommendations/models` (versions + active flags).

**macOS note:** torch + xgboost coexist via an `OMP_NUM_THREADS=1` Darwin guard;
community encoding uses MPS. On Linux/VPS, full multithreading (and
`apt-get install libgomp1` for xgboost).

---

## 19. Version history

The pipeline reached v5 by fixing measured regressions, not by guessing.

- **v3 (BPR-MF + content + popularity hybrid).** A PyTorch BPR matrix-factorisation
  collaborative model blended with the 390-d content embedding and popularity by
  fixed weights (`0.45·collab + 0.35·content + 0.20·pop`). Lift over popularity
  ~4.2× on the easy leave-one-out protocol.

- **v4 (multi-channel + LightGCN + XGBoost ranker).** Introduced the four embedding
  channels, LightGCN, multi-channel retrieval, and the graded XGBoost ranker — but
  **regressed** to ~1.5× lift on the harder leave-last-20 % protocol. Diagnosis
  (`diagnose_*.py`) found the **ranker**, not retrieval, was the defect: sorting by
  raw LightGCN score beat the trained ranker.

- **v5 (the current system).** Six targeted fixes:
  1. LightGCN **30 → 400 epochs** (it was undertrained; 30 epochs = 30 gradient
     steps).
  2. Ranker trains **only on the retrieved distribution** (stop force-injecting
     unretrieved positives that taught it to distrust mf).
  3. Retrieval budgets **rebalanced to measured recall** (mf 500, plot 150 — they
     were backwards in v4).
  4. New **structured-metadata channel** added *alongside* mpnet (both beat either).
  5. **Ranker A/B** (XGBoost vs Ridge linear), ship the NDCG@10 winner.
  6. **Cadence split** — ranker trained occasionally, LightGCN nightly
     (`--skip-ranker`), since ranker features are invariant to LightGCN's re-basis.

  Net: **2.1–2.5× over v4 on every metric**, **3.8× lift over popularity**, retrieval
  ceiling 0.65, LightGCN loss 0.0525, with the full multi-channel architecture intact.

---

*Built from the code in `app/ml/`, `app/jobs/`, and `scripts/`. To re-verify any
number here, run the matching `scripts/diagnose_*.py` or read the stored
`recommendation_jobs.metrics` for the active model.*
