from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import UserMovieStateType

ORM = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Generic
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    detail: str


class NamedCreate(BaseModel):
    name: str = Field(min_length=1)


class NamedUpdate(BaseModel):
    name: str = Field(min_length=1)


class NamedRead(BaseModel):
    model_config = ORM
    id: int
    name: str


# --------------------------------------------------------------------------- #
# Auth / users
# --------------------------------------------------------------------------- #
class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=150)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserUpdate(BaseModel):
    username: str | None = Field(default=None, min_length=1, max_length=150)
    email: EmailStr | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)
    is_active: bool | None = None
    is_admin: bool | None = None


class UserRead(BaseModel):
    model_config = ORM
    id: int
    username: str
    email: EmailStr
    is_active: bool
    is_admin: bool
    created_at: datetime
    updated_at: datetime


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class AccessToken(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


# --------------------------------------------------------------------------- #
# Movies
# --------------------------------------------------------------------------- #
class MovieBase(BaseModel):
    imdb_id: str
    movie_title: str
    year: int | None = None
    duration: int | None = None
    plot: str | None = None
    poster_url: str | None = None


class MovieCreate(MovieBase):
    pass


class MovieUpdate(BaseModel):
    imdb_id: str | None = None
    movie_title: str | None = None
    year: int | None = None
    duration: int | None = None
    plot: str | None = None
    poster_url: str | None = None


class MovieRead(MovieBase):
    model_config = ORM
    id: int
    created_at: datetime
    updated_at: datetime


class MovieDetail(MovieRead):
    genres: list[NamedRead] = []
    countries: list[NamedRead] = []
    languages: list[NamedRead] = []


# --------------------------------------------------------------------------- #
# People (TMDB-enriched)
# --------------------------------------------------------------------------- #
class PersonDetail(BaseModel):
    """A person plus lazily-fetched TMDB biography fields. `profile_url` is a
    ready-to-use image URL built from the stored TMDB path."""
    model_config = ORM
    id: int
    name: str
    tmdb_id: int | None = None
    biography: str | None = None
    profile_path: str | None = None
    profile_url: str | None = None
    birthday: str | None = None


# --------------------------------------------------------------------------- #
# Collections (TMDB-enriched)
# --------------------------------------------------------------------------- #
class CollectionRead(BaseModel):
    """A TMDB collection plus the member movies that exist in our catalogue.
    `movies` only contains films we actually have; absent members are omitted."""
    model_config = ORM
    id: int
    name: str
    overview: str | None = None
    poster_path: str | None = None
    poster_url: str | None = None
    backdrop_path: str | None = None
    backdrop_url: str | None = None
    movies: list[MovieRead] = []


# --------------------------------------------------------------------------- #
# Interactions
# --------------------------------------------------------------------------- #
class InteractionInput(BaseModel):
    """Client-writable interaction fields. `sentiment` and `preference_score`
    are computed server-side from the review text and rating."""
    rating: float | None = Field(default=None, ge=0, le=10)
    review_title: str | None = None
    review_body: str | None = None
    review_date: datetime | None = None


class InteractionCreate(InteractionInput):
    movie_id: int


class InteractionAdminCreate(InteractionCreate):
    user_id: int


class InteractionUpdate(InteractionInput):
    pass


class InteractionRead(BaseModel):
    model_config = ORM
    id: int
    user_id: int
    movie_id: int
    rating: float | None
    review_title: str | None
    review_body: str | None
    sentiment: float | None
    preference_score: float | None
    review_date: datetime | None
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------- #
# User movie state (watched / watchlist)
# --------------------------------------------------------------------------- #
class UserMovieStateCreate(BaseModel):
    movie_id: int
    state: UserMovieStateType


class UserMovieStateUpdate(BaseModel):
    state: UserMovieStateType


class UserMovieStateRead(BaseModel):
    model_config = ORM
    user_id: int
    movie_id: int
    state: UserMovieStateType
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class MovieEmbeddingUpsert(BaseModel):
    embedding: list[float]


class MovieEmbeddingRead(BaseModel):
    model_config = ORM
    movie_id: int
    embedding: list[float]
    created_at: datetime
    updated_at: datetime


class UserEmbeddingUpsert(BaseModel):
    embedding: list[float]


class UserEmbeddingRead(BaseModel):
    model_config = ORM
    user_id: int
    embedding: list[float]
    created_at: datetime
    updated_at: datetime


class VectorQuery(BaseModel):
    embedding: list[float]
    limit: int = Field(default=10, ge=1, le=100)
    metric: str = Field(default="cosine", pattern="^(cosine|l2|inner)$")


class MovieMetadataInput(BaseModel):
    """Raw movie metadata for ad-hoc embedding generation (no DB write)."""
    plot: str | None = None
    genre: list[str] = []
    director: list[str] = []
    writer: list[str] = []
    actors: list[str] = []
    language: list[str] = []
    country: list[str] = []
    year: int | None = None


class GeneratedEmbedding(BaseModel):
    dim: int
    embedding: list[float]


class ScoredMovie(BaseModel):
    movie: MovieRead
    distance: float


# --------------------------------------------------------------------------- #
# Recommendations / similar movies
# --------------------------------------------------------------------------- #
class SimilarMovieUpsert(BaseModel):
    similar_movie_id: int
    score: float
    rank: int


class SimilarMovieRead(BaseModel):
    model_config = ORM
    movie_id: int
    similar_movie_id: int
    score: float
    rank: int
    created_at: datetime
    updated_at: datetime


class UserRecommendationUpsert(BaseModel):
    movie_id: int
    score: float
    rank: int


class UserRecommendationRead(BaseModel):
    model_config = ORM
    user_id: int
    movie_id: int
    score: float
    rank: int
    created_at: datetime
    updated_at: datetime


class BecauseYouEnjoyedRead(BaseModel):
    """A "Because you enjoyed <movie>" carousel: the source movie the user last
    rated highly, plus movies similar to it (excluding ones they've already seen)."""
    source_movie: MovieRead
    preference_score: float | None = None
    movies: list[MovieRead] = []


# --------------------------------------------------------------------------- #
# Movie associations (genres/countries/languages/people)
# --------------------------------------------------------------------------- #
class MoviePersonCreate(BaseModel):
    person_id: int
    role_id: int


class MoviePersonRead(BaseModel):
    model_config = ORM
    movie_id: int
    person_id: int
    role_id: int


class MovieRefLink(BaseModel):
    """Generic body for linking a movie to a genre/country/language id."""
    ref_id: int


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class LogTypeCreate(BaseModel):
    name: str
    description: str | None = None


class LogTypeUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class LogTypeRead(BaseModel):
    model_config = ORM
    id: int
    name: str
    description: str | None = None


class LogCreate(BaseModel):
    user_id: int | None = None
    event_type_id: int
    entity_type_id: int | None = None
    entity_id: str | None = None
    details: dict = Field(default_factory=dict)


class LogRead(BaseModel):
    model_config = ORM
    id: int
    user_id: int | None
    event_type_id: int
    entity_type_id: int | None
    entity_id: str | None
    details: dict
    created_at: datetime


# --------------------------------------------------------------------------- #
# Stats / analytics
# --------------------------------------------------------------------------- #
class GlobalStats(BaseModel):
    total_movies: int
    total_users: int
    active_users: int
    total_interactions: int
    total_ratings: int
    overall_average_rating: float | None
    total_reviews: int


class MovieStats(BaseModel):
    movie_id: int
    average_rating: float | None
    rating_count: int
    review_count: int
    average_sentiment: float | None
    watched_count: int
    watchlist_count: int


class UserStats(BaseModel):
    user_id: int
    interaction_count: int
    rating_count: int
    average_rating_given: float | None
    review_count: int
    watched_count: int
    watchlist_count: int


class RatedMovie(BaseModel):
    model_config = ORM
    id: int
    movie_title: str
    average_rating: float | None = None
    rating_count: int = 0
    # Bayesian-weighted quality score used for ranking (top-rated). None for the
    # legacy count-based endpoints that don't compute it.
    score: float | None = None


class TrendingMovie(BaseModel):
    """A movie ranked by recent momentum (velocity). Windows are anchored to the
    latest review_date in the data, not wall-clock now."""
    id: int
    movie_title: str
    recent_count: int
    prior_count: int
    recent_avg_preference: float | None = None
    trend_score: float


class GenreCount(BaseModel):
    genre_id: int
    name: str
    movie_count: int


class CountItem(BaseModel):
    id: int
    name: str
    count: int


class RatingDistributionBucket(BaseModel):
    bucket: int
    count: int


# --------------------------------------------------------------------------- #
# Recommendation evaluation
# --------------------------------------------------------------------------- #
class RankingMetrics(BaseModel):
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    ndcg_at_k: float
    map_at_k: float


class RegressionMetrics(BaseModel):
    rmse: float
    mae: float
    predictions: int


class RetrainResponse(BaseModel):
    job_id: int
    status: str
    detail: str


class RecommendationJobRead(BaseModel):
    model_config = ORM
    id: int
    job_type: str
    status: str
    started_at: datetime | None
    finished_at: datetime | None
    error_message: str | None
    metrics: dict
    triggered_by_user_id: int | None
    created_at: datetime


class ModelVersionRead(BaseModel):
    model_config = ORM
    id: int
    model_type: str
    version_name: str
    artifact_path: str
    metrics: dict
    is_active: bool
    created_at: datetime


class RecommendationEvaluation(BaseModel):
    k: int
    holdout_fraction: float
    like_threshold: float
    min_interactions: int
    seed: int
    candidate_users: int
    users_evaluated_ranking: int
    model: RankingMetrics
    popularity_baseline: RankingMetrics
    rating_prediction: RegressionMetrics
    notes: str
