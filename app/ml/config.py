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

# --- Collaborative filtering (PyTorch BPR matrix factorisation) ------------ #
CF_DIM = 128
CF_EPOCHS = 12
CF_LR = 0.03
CF_REG = 1e-6                 # L2 on user/item factors (user_alpha == item_alpha)
CF_BATCH = 8192
CF_POS_THRESHOLD = 6.0       # preference >= this counts as a positive
COLLAB_DIR = ARTIFACTS_DIR / "collaborative"

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
