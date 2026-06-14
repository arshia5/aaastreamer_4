from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select

from app.core.config import settings
from app.deps import DB, CurrentAdmin, CurrentUser
from app.ml.db_records import movie_record_from_db
from app.ml.embedder import MovieEmbedder, PipelineNotFitted
from app.ml.similar import update_similar_movies_for
from app.ml.user_embedding import compute_user_vector
from app.models import Interaction, Movie, MovieEmbedding, User, UserEmbedding
from app.schemas import (
    GeneratedEmbedding,
    Message,
    MovieEmbeddingRead,
    MovieEmbeddingUpsert,
    MovieMetadataInput,
    ScoredMovie,
    UserEmbeddingRead,
    UserEmbeddingUpsert,
    VectorQuery,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

DIM = settings.embedding_dim


def _check_dim(vec: list[float]) -> None:
    if len(vec) != DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Embedding must have exactly {DIM} dimensions, got {len(vec)}",
        )


def _distance_expr(column, query: VectorQuery):
    if query.metric == "l2":
        return column.l2_distance(query.embedding)
    if query.metric == "inner":
        return column.max_inner_product(query.embedding)
    return column.cosine_distance(query.embedding)


def _get_embedder() -> MovieEmbedder:
    try:
        return MovieEmbedder.instance()
    except PipelineNotFitted as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )


# --------------------------------------------------------------------------- #
# Embedding generation (reproduces the aaastreamer_3 pipeline, 390-d / 90% var)
# --------------------------------------------------------------------------- #
@router.post("/generate", response_model=GeneratedEmbedding)
async def generate_embedding(payload: MovieMetadataInput, _: CurrentAdmin):
    """Compute a 390-d embedding from raw metadata without storing it."""
    embedder = _get_embedder()
    vec = await run_in_threadpool(embedder.embed_one, payload.model_dump())
    return GeneratedEmbedding(dim=len(vec), embedding=vec)


@router.post("/movies/{movie_id}/generate", response_model=MovieEmbeddingRead)
async def generate_movie_embedding(movie_id: int, db: DB, _: CurrentAdmin):
    """Build the movie's embedding from its DB metadata and upsert it."""
    embedder = _get_embedder()
    record = await movie_record_from_db(db, movie_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    vec = await run_in_threadpool(embedder.embed_one, record)
    emb = await db.get(MovieEmbedding, movie_id)
    if emb is None:
        emb = MovieEmbedding(movie_id=movie_id, embedding=vec)
        db.add(emb)
    else:
        emb.embedding = vec
    await db.commit()
    await db.refresh(emb)
    # Refresh similar_movies: this movie's neighbours + any list it now enters.
    await update_similar_movies_for(db, movie_id)
    return emb


# --------------------------------------------------------------------------- #
# Movie embeddings
# --------------------------------------------------------------------------- #
@router.get("/movies/{movie_id}", response_model=MovieEmbeddingRead)
async def get_movie_embedding(movie_id: int, db: DB):
    emb = await db.get(MovieEmbedding, movie_id)
    if emb is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return emb


@router.put("/movies/{movie_id}", response_model=MovieEmbeddingRead)
async def upsert_movie_embedding(
    movie_id: int, payload: MovieEmbeddingUpsert, db: DB, _: CurrentAdmin
):
    _check_dim(payload.embedding)
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    emb = await db.get(MovieEmbedding, movie_id)
    if emb is None:
        emb = MovieEmbedding(movie_id=movie_id, embedding=payload.embedding)
        db.add(emb)
    else:
        emb.embedding = payload.embedding
    await db.commit()
    await db.refresh(emb)
    await update_similar_movies_for(db, movie_id)
    return emb


@router.delete("/movies/{movie_id}", response_model=Message)
async def delete_movie_embedding(movie_id: int, db: DB, _: CurrentAdmin):
    emb = await db.get(MovieEmbedding, movie_id)
    if emb is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    await db.delete(emb)
    await db.commit()
    return Message(detail="Embedding deleted")


@router.post("/movies/search", response_model=list[ScoredMovie])
async def search_movies_by_vector(payload: VectorQuery, db: DB):
    """Nearest movies to a query vector (cosine/l2/inner-product)."""
    _check_dim(payload.embedding)
    dist = _distance_expr(MovieEmbedding.embedding, payload)
    stmt = (
        select(Movie, dist.label("distance"))
        .join(MovieEmbedding, MovieEmbedding.movie_id == Movie.id)
        .order_by(dist)
        .limit(payload.limit)
    )
    rows = (await db.execute(stmt)).all()
    return [ScoredMovie(movie=movie, distance=distance) for movie, distance in rows]


@router.get("/movies/{movie_id}/neighbors", response_model=list[ScoredMovie])
async def movie_neighbors(movie_id: int, db: DB, limit: int = 10, metric: str = "cosine"):
    """Nearest movies to a given movie's embedding (excludes itself)."""
    source = await db.get(MovieEmbedding, movie_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source embedding not found")
    query = VectorQuery(embedding=list(source.embedding), limit=limit, metric=metric)
    dist = _distance_expr(MovieEmbedding.embedding, query)
    stmt = (
        select(Movie, dist.label("distance"))
        .join(MovieEmbedding, MovieEmbedding.movie_id == Movie.id)
        .where(Movie.id != movie_id)
        .order_by(dist)
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [ScoredMovie(movie=movie, distance=distance) for movie, distance in rows]


# --------------------------------------------------------------------------- #
# User embeddings
# --------------------------------------------------------------------------- #
async def _build_user_embedding(db, user_id: int) -> UserEmbedding:
    """Recompute a user's embedding from their scored interactions
    (preference × rank-based recency decay) and upsert it."""
    rows = (
        await db.execute(
            select(
                Interaction.movie_id,
                Interaction.preference_score,
                Interaction.review_date,
            ).where(
                Interaction.user_id == user_id,
                Interaction.preference_score.is_not(None),
            )
        )
    ).all()
    if not rows:
        raise HTTPException(
            status_code=400, detail="User has no scored interactions to embed"
        )
    movie_ids = [r[0] for r in rows]
    embs = (
        await db.execute(
            select(MovieEmbedding.movie_id, MovieEmbedding.embedding).where(
                MovieEmbedding.movie_id.in_(movie_ids)
            )
        )
    ).all()
    lookup = {mid: vec for mid, vec in embs}
    vec = compute_user_vector([(r[0], r[1], r[2]) for r in rows], lookup)
    if vec is None:
        raise HTTPException(
            status_code=400,
            detail="No movie embeddings available for this user's movies",
        )
    emb = await db.get(UserEmbedding, user_id)
    if emb is None:
        emb = UserEmbedding(user_id=user_id, embedding=vec.tolist())
        db.add(emb)
    else:
        emb.embedding = vec.tolist()
    await db.commit()
    await db.refresh(emb)
    return emb


@router.post("/users/me/build", response_model=UserEmbeddingRead)
async def build_my_embedding(db: DB, current_user: CurrentUser):
    """Recompute the current user's embedding from their reviews."""
    return await _build_user_embedding(db, current_user.id)


@router.post("/users/{user_id}/build", response_model=UserEmbeddingRead)
async def build_user_embedding(user_id: int, db: DB, _: CurrentAdmin):
    if await db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return await _build_user_embedding(db, user_id)


@router.get("/users/{user_id}", response_model=UserEmbeddingRead)
async def get_user_embedding(user_id: int, db: DB, _: CurrentAdmin):
    emb = await db.get(UserEmbedding, user_id)
    if emb is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    return emb


@router.put("/users/{user_id}", response_model=UserEmbeddingRead)
async def upsert_user_embedding(
    user_id: int, payload: UserEmbeddingUpsert, db: DB, _: CurrentAdmin
):
    _check_dim(payload.embedding)
    if await db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    emb = await db.get(UserEmbedding, user_id)
    if emb is None:
        emb = UserEmbedding(user_id=user_id, embedding=payload.embedding)
        db.add(emb)
    else:
        emb.embedding = payload.embedding
    await db.commit()
    await db.refresh(emb)
    return emb


@router.delete("/users/{user_id}", response_model=Message)
async def delete_user_embedding(user_id: int, db: DB, _: CurrentAdmin):
    emb = await db.get(UserEmbedding, user_id)
    if emb is None:
        raise HTTPException(status_code=404, detail="Embedding not found")
    await db.delete(emb)
    await db.commit()
    return Message(detail="Embedding deleted")


async def _recommend_from_user_vector(
    db, user_id: int, limit: int, metric: str, exclude_seen: bool
) -> list[ScoredMovie]:
    source = await db.get(UserEmbedding, user_id)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail="User embedding not found (build it first)",
        )
    query = VectorQuery(embedding=list(source.embedding), limit=limit, metric=metric)
    dist = _distance_expr(MovieEmbedding.embedding, query)
    stmt = select(Movie, dist.label("distance")).join(
        MovieEmbedding, MovieEmbedding.movie_id == Movie.id
    )
    if exclude_seen:
        stmt = stmt.where(
            Movie.id.notin_(
                select(Interaction.movie_id).where(Interaction.user_id == user_id)
            )
        )
    stmt = stmt.order_by(dist).limit(limit)
    rows = (await db.execute(stmt)).all()
    return [ScoredMovie(movie=movie, distance=distance) for movie, distance in rows]


@router.get("/users/me/recommendations", response_model=list[ScoredMovie])
async def recommend_for_me(
    db: DB,
    current_user: CurrentUser,
    limit: int = 10,
    metric: str = "cosine",
    exclude_seen: bool = True,
):
    """Personalised recommendations from the current user's embedding."""
    return await _recommend_from_user_vector(
        db, current_user.id, limit, metric, exclude_seen
    )


@router.get("/users/{user_id}/recommendations", response_model=list[ScoredMovie])
async def recommend_for_user_embedding(
    user_id: int,
    db: DB,
    _: CurrentAdmin,
    limit: int = 10,
    metric: str = "cosine",
    exclude_seen: bool = True,
):
    """Recommend movies nearest to a user's embedding vector.

    By default excludes movies the user has already interacted with.
    """
    return await _recommend_from_user_vector(db, user_id, limit, metric, exclude_seen)
