"""Configuration for the movie embedding pipeline.

These constants mirror the original notebook (aaastreamer_3/movie_embeddings.ipynb)
exactly, so vectors produced here live in the same space as the training matrix.
"""
from pathlib import Path

# Sentence-transformer used for the plot embedding (384-d).
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Per-block output dimensionality.
PLOT_DIM = 384
DIRECTOR_DIM = 64
WRITER_DIM = 64
ACTOR_DIM = 128
LANG_DIM = 32
COUNTRY_DIM = 32
NUMERIC_DIM = 1
# genre is multi-hot (MultiLabelBinarizer) — its width = number of fitted classes.

# Block weights applied after L2-normalisation, before concatenation.
WEIGHTS = {
    "plot": 2.0,
    "genre": 1.2,
    "actor": 1.2,
    "director": 1.0,
    "writer": 0.7,
    "numeric": 0.6,
    "language": 0.5,
    "country": 0.4,
}

# Concatenation order MUST match the notebook (plot, genre, director, writer,
# actor, language, country, numeric).
BLOCK_ORDER = [
    "plot",
    "genre",
    "director",
    "writer",
    "actor",
    "language",
    "country",
    "numeric",
]

# Target explained variance and the resulting PCA width (must equal the
# movie_embeddings.embedding column dimension, vector(390)).
TARGET_VARIANCE = 0.90
N_COMPONENTS = 390

# Where the fitted pipeline artifact is stored.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "movie_embed_pipeline.joblib"

# Default training CSV (the original dataset).
DEFAULT_MOVIES_CSV = _PROJECT_ROOT.parent / "aaastreamer_3" / "data" / "movies.csv"

# --- v4 component embeddings ---------------------------------------------- #
# metadata + plot use a stronger sentence transformer (768-d, stateless so new
# movies are trivial); community uses MiniLM (384-d, fast over ~1.85M reviews).
TEXT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TEXT_EMBED_DIM = 768
REVIEW_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVIEW_EMBED_DIM = 384
COMMUNITY_K = 5                 # max KMeans clusters per movie
COMMUNITY_MAX_REVIEWS = 200     # cap reviews per movie used for clustering

# --- v5 structured-metadata channel --------------------------------------- #
# The v3 structured block embedding (genre multi-hot + hashed director/writer/
# actor/language/country + year), L2-normalised, WITHOUT the plot block. It is a
# complementary retrieval channel to the mpnet metadata template: structured
# alone is weaker than mpnet (0.143 vs 0.180 recall@300) but the two together
# beat either (0.204) because structured preserves exact genre/cast/year matches
# that the templated prose blurs. Built in-memory from the fitted v3 pipeline
# artifact (PIPELINE_PATH) at context-build time — no DB table, no migration.
STRUCT_ENABLED = True
# block dim = genre_classes(27) + director(64) + writer(64) + actor(128) +
# language(32) + country(32) + year(1) = 348 (genre width comes from the mlb).

# --- Sentiment model (fine-tuned DistilBERT, 5-class -> 1..10) ------------- #
SENTIMENT_MODEL_DIR = _PROJECT_ROOT / "models" / "sentiment-distilbert"
SENTIMENT_MAX_LENGTH = 128
SENTIMENT_BATCH_SIZE = 64
# Class indices 0..4 map to ordinal labels 1..5.
SENTIMENT_CLASS_SCORES = [1.0, 2.0, 3.0, 4.0, 5.0]

# --- Preference & user-embedding weighting -------------------------------- #
# preference_score = RATING_WEIGHT*rating + SENTIMENT_WEIGHT*sentiment (0..10).
PREF_RATING_WEIGHT = 0.7
PREF_SENTIMENT_WEIGHT = 0.3
# Neutral midpoint: centred weighting subtracts this so disliked movies push away.
PREF_NEUTRAL = 5.5
# Per-step recency decay applied by rank (newest review = rank 0 -> weight 1).
USER_EMBED_DECAY = 0.9

# Number of nearest neighbours stored per movie in similar_movies.
SIMILAR_TOP_N = 10
# Hybrid similar-movies blend (nightly): content (metadata) weighted higher than
# the collaborative MF item-factor similarity. Movies without an MF factor (new /
# too few reviews) fall back to content-only.
SIMILAR_W_CONTENT = 0.7
SIMILAR_W_COLLAB = 0.3
# v4 multi-channel similar-movies blend (content-weighted; mf/community fall back
# to content when missing, e.g. new/sparse movies).
SIMILAR_W_PLOT = 0.4
SIMILAR_W_META = 0.3
SIMILAR_W_MF = 0.2
SIMILAR_W_COMM = 0.1

# --- Collaborative filtering (PyTorch BPR matrix factorisation) ------------ #
CF_DIM = 128
CF_EPOCHS = 12
CF_LR = 0.03
CF_REG = 1e-6                 # L2 on user/item factors (user_alpha == item_alpha)
CF_BATCH = 8192
CF_POS_THRESHOLD = 6.0       # preference >= this counts as a positive
COLLAB_DIR = ARTIFACTS_DIR / "collaborative"

# --- LightGCN (v4 collaborative model; replaces BPR-MF) ------------------- #
# epochs: 30 full-batch steps was severely undertrained (mf is the single
# strongest channel). Bake-off (held-out split) showed 30->400 epochs lifts mf
# retrieval recall@300 0.415->0.503 and mf-as-scorer hit@10 0.086->0.112;
# multi-negative sampling (neg_k>1) did not help. See scripts/diagnose_lightgcn.py.
LIGHTGCN_DIM = 64
LIGHTGCN_LAYERS = 3
LIGHTGCN_EPOCHS = 400
LIGHTGCN_LR = 0.01
LIGHTGCN_REG = 1e-4

# Real-time fit_partial: kept deliberately small so one odd review only nudges a
# user's vector rather than re-learning it (the nightly model is the source of
# truth). Lower LR + a blend back toward the nightly factor.
CF_PARTIAL_EPOCHS = 2
CF_PARTIAL_LR = 0.01         # < CF_LR (0.03)
CF_PARTIAL_BLEND = 0.30      # new = 0.30*updated + 0.70*nightly_factor

# Bayesian shrinkage for average preference (popularity + ranker feature):
# (PRIOR*global_mean + sum) / (PRIOR + n). Avoids low-count averages dominating.
POPULARITY_PRIOR = 50

# --- Hybrid scoring weights ----------------------------------------------- #
HYBRID_W_COLLAB = 0.45
HYBRID_W_CONTENT = 0.35
HYBRID_W_POP = 0.20
# Fallback when collaborative score is unavailable (cold user/movie).
HYBRID_FB_CONTENT = 0.70
HYBRID_FB_POP = 0.30

# How many recommendations to persist per user.
TOP_N_RECOMMENDATIONS = 100
# Candidate pool size fed to reranking (XGBoost / diversity).
CAND_POOL = 200

# --- v5 multi-channel retrieval (union, deduped, ~1100 pool) -------------- #
# Budgets rebalanced to measured per-channel recall (scripts/diagnose_channels.py):
#   mf 0.40 (best) | pop 0.18 | meta 0.18 | struct 0.14 | comm 0.16 | plot 0.10 (worst).
# mf gets the largest budget; plot was over-budgeted at 400 for the weakest recall.
RETR_PLOT = 150
RETR_META = 300
RETR_STRUCT = 300            # v5 structured-metadata channel (complementary to meta)
RETR_MF = 500
RETR_COMM = 200
RETR_POP = 100

# --- Diversity (MMR) re-ranking ------------------------------------------- #
DIVERSITY_ENABLED = True
DIVERSITY_LAMBDA = 0.30      # 0 = pure relevance, 1 = pure diversity
DIVERSITY_MAX_PER_GENRE = 4  # cap per genre within the stored list

# --- XGBoost learning-to-rank reranker ------------------------------------ #
XGB_PARAMS = {
    "objective": "rank:ndcg",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
}
XGB_TRAIN_USERS = 15000      # sample of users used to build LTR training data
XGB_NEG_PER_POS = 30         # negatives sampled per positive (from candidate pool)
RANKER_DIR = ARTIFACTS_DIR / "ranker"

# --- Rollback ------------------------------------------------------------- #
# A newly trained model is only activated if it beats the current active model
# on this metric (same held-out split); otherwise the old model stays active.
ROLLBACK_METRIC = "ndcg_at_10"
ROLLBACK_MARGIN = 0.0        # new must exceed old by at least this much

# Evaluation: cap users scored during nightly eval (None = all held-out users).
EVAL_SAMPLE_USERS = 20000

# Reviews dataset (one CSV per part).
DEFAULT_REVIEWS_DIR = _PROJECT_ROOT.parent / "aaastreamer_3" / "data" / "reviews"
# Default password / email domain for users imported from the reviews dataset.
IMPORT_USER_PASSWORD = "1234"
IMPORT_EMAIL_DOMAIN = "test.com"
