import enum
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    Enum,
    ForeignKey,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.database import Base

EMBED_DIM = settings.embedding_dim


class UserMovieStateType(str, enum.Enum):
    watched = "watched"
    watchlist = "watchlist"


# --------------------------------------------------------------------------- #
# Reference / lookup tables
# --------------------------------------------------------------------------- #
class Person(Base):
    __tablename__ = "people"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)


class Country(Base):
    __tablename__ = "countries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)


class Language(Base):
    __tablename__ = "languages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)


# --------------------------------------------------------------------------- #
# Core entities
# --------------------------------------------------------------------------- #
class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    imdb_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    movie_title: Mapped[str] = mapped_column(String, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer)
    duration: Mapped[int | None] = mapped_column(Integer)
    plot: Mapped[str | None] = mapped_column(Text)
    poster_url: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    genres: Mapped[list[Genre]] = relationship(
        secondary="movie_genres", lazy="selectin"
    )
    countries: Mapped[list[Country]] = relationship(
        secondary="movie_countries", lazy="selectin"
    )
    languages: Mapped[list[Language]] = relationship(
        secondary="movie_languages", lazy="selectin"
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    is_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


# --------------------------------------------------------------------------- #
# Embeddings & recommendations
# --------------------------------------------------------------------------- #
class MovieEmbedding(Base):
    __tablename__ = "movie_embeddings"

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


# --- v4 component embeddings (separate vector spaces) ---------------------- #
class MovieMetadataEmbedding(Base):
    __tablename__ = "movie_metadata_embeddings"
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class MoviePlotEmbedding(Base):
    __tablename__ = "movie_plot_embeddings"
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class MovieMfEmbedding(Base):
    __tablename__ = "movie_mf_embeddings"
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class UserMfEmbedding(Base):
    __tablename__ = "user_mf_embeddings"
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class MovieCommunityEmbedding(Base):
    __tablename__ = "movie_community_embeddings"
    __table_args__ = (PrimaryKeyConstraint("movie_id", "cluster_idx"),)
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    cluster_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    weight: Mapped[float] = mapped_column(Double, nullable=False, server_default="1.0")
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class UserProfile768(Base):
    __tablename__ = "user_profiles_768"
    __table_args__ = (PrimaryKeyConstraint("user_id", "kind"),)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class UserProfile384(Base):
    __tablename__ = "user_profiles_384"
    __table_args__ = (PrimaryKeyConstraint("user_id", "kind"),)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now())


class UserEmbedding(Base):
    __tablename__ = "user_embeddings"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBED_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SimilarMovie(Base):
    __tablename__ = "similar_movies"
    __table_args__ = (PrimaryKeyConstraint("movie_id", "rank"),)

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False
    )
    similar_movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    score: Mapped[float] = mapped_column(Double, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class UserRecommendation(Base):
    __tablename__ = "user_recommendations"
    __table_args__ = (PrimaryKeyConstraint("user_id", "rank"),)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    score: Mapped[float] = mapped_column(Double, nullable=False)  # final_score
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    # component scores (nullable; populated by the hybrid scorer)
    content_score: Mapped[float | None] = mapped_column(Double)
    collaborative_score: Mapped[float | None] = mapped_column(Double)
    popularity_score: Mapped[float | None] = mapped_column(Double)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


# --------------------------------------------------------------------------- #
# Interactions (ratings / reviews)
# --------------------------------------------------------------------------- #
class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_interactions_user_movie"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rating: Mapped[float | None] = mapped_column(Double)
    review_title: Mapped[str | None] = mapped_column(Text)
    review_body: Mapped[str | None] = mapped_column(Text)
    review_body_embedding: Mapped[list[float] | None] = mapped_column(Vector())
    sentiment: Mapped[float | None] = mapped_column(Double)
    preference_score: Mapped[float | None] = mapped_column(Double)
    review_date: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


# --------------------------------------------------------------------------- #
# Movie <-> reference junctions
# --------------------------------------------------------------------------- #
class MoviePerson(Base):
    __tablename__ = "movie_people"
    __table_args__ = (PrimaryKeyConstraint("movie_id", "person_id", "role_id"),)

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey("people.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role_id: Mapped[int] = mapped_column(
        ForeignKey("roles.id"), nullable=False, index=True
    )


class MovieCountry(Base):
    __tablename__ = "movie_countries"
    __table_args__ = (PrimaryKeyConstraint("movie_id", "country_id"),)

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False
    )
    country_id: Mapped[int] = mapped_column(
        ForeignKey("countries.id", ondelete="CASCADE"), nullable=False, index=True
    )


class MovieGenre(Base):
    __tablename__ = "movie_genres"
    __table_args__ = (PrimaryKeyConstraint("movie_id", "genre_id"),)

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False
    )
    genre_id: Mapped[int] = mapped_column(
        ForeignKey("genres.id", ondelete="CASCADE"), nullable=False, index=True
    )


class MovieLanguage(Base):
    __tablename__ = "movie_languages"
    __table_args__ = (PrimaryKeyConstraint("movie_id", "language_id"),)

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False
    )
    language_id: Mapped[int] = mapped_column(
        ForeignKey("languages.id", ondelete="CASCADE"), nullable=False, index=True
    )


# --------------------------------------------------------------------------- #
# Auth & user state
# --------------------------------------------------------------------------- #
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class UserMovieState(Base):
    __tablename__ = "user_movie_states"
    __table_args__ = (PrimaryKeyConstraint("user_id", "movie_id"),)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    state: Mapped[UserMovieStateType] = mapped_column(
        Enum(
            UserMovieStateType,
            name="user_movie_state_type",
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class LogEventType(Base):
    __tablename__ = "log_event_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)


class LogEntityType(Base):
    __tablename__ = "log_entity_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    event_type_id: Mapped[int] = mapped_column(
        ForeignKey("log_event_types.id"), nullable=False, index=True
    )
    entity_type_id: Mapped[int | None] = mapped_column(
        ForeignKey("log_entity_types.id"), index=True
    )
    entity_id: Mapped[str | None] = mapped_column(String, index=True)
    details: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


# --------------------------------------------------------------------------- #
# Recommendation infrastructure (jobs, popularity, model versions)
# --------------------------------------------------------------------------- #
class RecommendationJob(Base):
    __tablename__ = "recommendation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default="queued", index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_message: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    triggered_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class MoviePopularityStats(Base):
    __tablename__ = "movie_popularity_stats"

    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    review_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    avg_rating: Mapped[float | None] = mapped_column(Double)
    avg_preference_score: Mapped[float | None] = mapped_column(Double)
    popularity_score: Mapped[float] = mapped_column(
        Double, nullable=False, server_default="0"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version_name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    artifact_path: Mapped[str] = mapped_column(String, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
