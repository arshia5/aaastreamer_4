"""recommendation infrastructure: jobs, popularity, model versions, rec score cols

Revision ID: 0004_recommendation_infra
Revises: 0003_movie_embeddings_hnsw
Create Date: 2026-06-14
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_recommendation_infra"
down_revision: Union[str, None] = "0003_movie_embeddings_hnsw"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "recommendation_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("job_type", sa.String, nullable=False),
        sa.Column("status", sa.String, nullable=False, server_default="queued"),
        sa.Column("started_at", sa.DateTime),
        sa.Column("finished_at", sa.DateTime),
        sa.Column("error_message", sa.Text),
        sa.Column("metrics", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "triggered_by_user_id",
            sa.Integer,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_recommendation_jobs_job_type", "recommendation_jobs", ["job_type"])
    op.create_index("ix_recommendation_jobs_status", "recommendation_jobs", ["status"])
    op.create_index("ix_recommendation_jobs_created_at", "recommendation_jobs", ["created_at"])

    op.create_table(
        "movie_popularity_stats",
        sa.Column(
            "movie_id",
            sa.Integer,
            sa.ForeignKey("movies.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("review_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("avg_rating", sa.Double),
        sa.Column("avg_preference_score", sa.Double),
        sa.Column("popularity_score", sa.Double, nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_type", sa.String, nullable=False),
        sa.Column("version_name", sa.String, nullable=False, unique=True),
        sa.Column("artifact_path", sa.String, nullable=False),
        sa.Column("metrics", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_model_versions_model_type", "model_versions", ["model_type"])
    op.create_index("ix_model_versions_is_active", "model_versions", ["is_active"])

    op.add_column("user_recommendations", sa.Column("content_score", sa.Double))
    op.add_column("user_recommendations", sa.Column("collaborative_score", sa.Double))
    op.add_column("user_recommendations", sa.Column("popularity_score", sa.Double))

    # HNSW index already created in 0003; ensure it exists (idempotent).
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_movie_embeddings_hnsw "
        "ON movie_embeddings USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_column("user_recommendations", "popularity_score")
    op.drop_column("user_recommendations", "collaborative_score")
    op.drop_column("user_recommendations", "content_score")
    op.drop_table("model_versions")
    op.drop_table("movie_popularity_stats")
    op.drop_table("recommendation_jobs")
