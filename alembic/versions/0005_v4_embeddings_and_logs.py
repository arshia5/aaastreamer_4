"""recommender v4: component embedding tables, MF-in-DB, taste profiles, log seeds

Revision ID: 0005_v4_embeddings_and_logs
Revises: 0004_recommendation_infra
Create Date: 2026-06-14
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_v4_embeddings_and_logs"
down_revision: Union[str, None] = "0004_recommendation_infra"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# (event/entity) types seeded so domain logging works out of the box.
EVENT_TYPES = [
    ("http_request", "An API request was handled"),
    ("user_registered", "A new user signed up"),
    ("user_login", "A user logged in"),
    ("review_created", "An interaction/review was created"),
    ("review_updated", "An interaction/review was updated"),
    ("review_deleted", "An interaction/review was deleted"),
    ("movie_created", "A movie was added"),
    ("movie_updated", "A movie was edited"),
    ("embedding_generated", "A movie embedding was (re)generated"),
    ("recommendations_served", "Recommendations were served to a user"),
    ("user_embedding_updated", "A user embedding was recomputed"),
    ("training_started", "A recommendation training job started"),
    ("training_completed", "A recommendation training job finished"),
    ("model_activated", "A new model version was activated"),
    ("model_rolled_back", "A new model version was rejected (rollback)"),
    ("similar_rebuilt", "similar_movies was rebuilt"),
    ("error", "An unhandled error occurred"),
]
ENTITY_TYPES = [
    ("user", "users table"),
    ("movie", "movies table"),
    ("interaction", "interactions table"),
    ("model", "model_versions table"),
    ("job", "recommendation_jobs table"),
    ("recommendation", "user_recommendations"),
    ("http", "an http endpoint"),
]


def _emb_table(name: str, id_col: str, ref: str, dim: int) -> None:
    op.execute(
        f"CREATE TABLE {name} ("
        f"  {id_col} integer PRIMARY KEY REFERENCES {ref}(id) ON DELETE CASCADE,"
        f"  embedding vector({dim}) NOT NULL,"
        f"  updated_at timestamp NOT NULL DEFAULT now())"
    )
    op.execute(
        f"CREATE INDEX ix_{name}_hnsw ON {name} "
        f"USING hnsw (embedding vector_cosine_ops)"
    )


def upgrade() -> None:
    # --- component embedding tables -------------------------------------- #
    _emb_table("movie_metadata_embeddings", "movie_id", "movies", 768)
    _emb_table("movie_plot_embeddings", "movie_id", "movies", 768)
    _emb_table("movie_mf_embeddings", "movie_id", "movies", 64)
    _emb_table("user_mf_embeddings", "user_id", "users", 64)

    # community: up to K centroids per movie (review-cluster centres)
    op.execute(
        "CREATE TABLE movie_community_embeddings ("
        "  movie_id integer NOT NULL REFERENCES movies(id) ON DELETE CASCADE,"
        "  cluster_idx integer NOT NULL,"
        "  weight double precision NOT NULL DEFAULT 1.0,"
        "  embedding vector(384) NOT NULL,"
        "  updated_at timestamp NOT NULL DEFAULT now(),"
        "  PRIMARY KEY (movie_id, cluster_idx))"
    )
    op.execute(
        "CREATE INDEX ix_movie_community_embeddings_hnsw "
        "ON movie_community_embeddings USING hnsw (embedding vector_cosine_ops)"
    )

    # user taste profiles (positive & negative) per content channel.
    # Two tables because pgvector columns need a fixed dim.
    op.execute(
        "CREATE TABLE user_profiles_768 ("
        "  user_id integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,"
        "  kind varchar NOT NULL,"          # meta_pos|meta_neg|plot_pos|plot_neg
        "  embedding vector(768) NOT NULL,"
        "  updated_at timestamp NOT NULL DEFAULT now(),"
        "  PRIMARY KEY (user_id, kind))"
    )
    op.execute(
        "CREATE TABLE user_profiles_384 ("
        "  user_id integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,"
        "  kind varchar NOT NULL,"          # comm_pos|comm_neg
        "  embedding vector(384) NOT NULL,"
        "  updated_at timestamp NOT NULL DEFAULT now(),"
        "  PRIMARY KEY (user_id, kind))"
    )

    # --- seed log types -------------------------------------------------- #
    conn = op.get_bind()
    for name, desc in EVENT_TYPES:
        conn.execute(
            sa.text("INSERT INTO log_event_types(name, description) VALUES (:n, :d) "
                    "ON CONFLICT (name) DO NOTHING"),
            {"n": name, "d": desc},
        )
    for name, desc in ENTITY_TYPES:
        conn.execute(
            sa.text("INSERT INTO log_entity_types(name, description) VALUES (:n, :d) "
                    "ON CONFLICT (name) DO NOTHING"),
            {"n": name, "d": desc},
        )


def downgrade() -> None:
    for t in [
        "user_profiles_384", "user_profiles_768", "movie_community_embeddings",
        "user_mf_embeddings", "movie_mf_embeddings",
        "movie_plot_embeddings", "movie_metadata_embeddings",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {t}")
    conn = op.get_bind()
    names = [n for n, _ in EVENT_TYPES]
    conn.execute(sa.text("DELETE FROM log_event_types WHERE name = ANY(:n)"),
                 {"n": names})
    ent = [n for n, _ in ENTITY_TYPES]
    conn.execute(sa.text("DELETE FROM log_entity_types WHERE name = ANY(:n)"),
                 {"n": ent})
