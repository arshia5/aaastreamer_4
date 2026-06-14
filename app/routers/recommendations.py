from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete, select

from app.deps import DB, CurrentAdmin, CurrentUser, PageParams
from app.models import Movie, SimilarMovie, User, UserRecommendation
from app.schemas import (
    Message,
    MovieRead,
    SimilarMovieRead,
    SimilarMovieUpsert,
    UserRecommendationRead,
    UserRecommendationUpsert,
)

router = APIRouter(tags=["recommendations"])


# --------------------------------------------------------------------------- #
# Similar movies (precomputed)
# --------------------------------------------------------------------------- #
@router.get("/movies/{movie_id}/similar", response_model=list[SimilarMovieRead])
async def list_similar(movie_id: int, db: DB, page: PageParams):
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    stmt = (
        select(SimilarMovie)
        .where(SimilarMovie.movie_id == movie_id)
        .order_by(SimilarMovie.rank)
        .limit(page.limit)
        .offset(page.offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.get("/movies/{movie_id}/similar/movies", response_model=list[MovieRead])
async def list_similar_movies(movie_id: int, db: DB, page: PageParams):
    """Resolve the similar-movie ids into full movie records, by rank."""
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    stmt = (
        select(Movie)
        .join(SimilarMovie, SimilarMovie.similar_movie_id == Movie.id)
        .where(SimilarMovie.movie_id == movie_id)
        .order_by(SimilarMovie.rank)
        .limit(page.limit)
        .offset(page.offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.put("/movies/{movie_id}/similar/{rank}", response_model=SimilarMovieRead)
async def upsert_similar(
    movie_id: int, rank: int, payload: SimilarMovieUpsert, db: DB, _: CurrentAdmin
):
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    if await db.get(Movie, payload.similar_movie_id) is None:
        raise HTTPException(status_code=404, detail="Similar movie not found")
    row = await db.get(SimilarMovie, (movie_id, rank))
    if row is None:
        row = SimilarMovie(
            movie_id=movie_id,
            rank=rank,
            similar_movie_id=payload.similar_movie_id,
            score=payload.score,
        )
        db.add(row)
    else:
        row.similar_movie_id = payload.similar_movie_id
        row.score = payload.score
    await db.commit()
    await db.refresh(row)
    return row


@router.delete("/movies/{movie_id}/similar/{rank}", response_model=Message)
async def delete_similar(movie_id: int, rank: int, db: DB, _: CurrentAdmin):
    result = await db.execute(
        delete(SimilarMovie).where(
            SimilarMovie.movie_id == movie_id, SimilarMovie.rank == rank
        )
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")
    return Message(detail="Deleted")


# --------------------------------------------------------------------------- #
# User recommendations (precomputed)
# --------------------------------------------------------------------------- #
@router.get("/me/recommendations", response_model=list[UserRecommendationRead])
async def my_recommendations(db: DB, current_user: CurrentUser, page: PageParams):
    stmt = (
        select(UserRecommendation)
        .where(UserRecommendation.user_id == current_user.id)
        .order_by(UserRecommendation.rank)
        .limit(page.limit)
        .offset(page.offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.get("/me/recommendations/movies", response_model=list[MovieRead])
async def my_recommendation_movies(db: DB, current_user: CurrentUser, page: PageParams):
    stmt = (
        select(Movie)
        .join(UserRecommendation, UserRecommendation.movie_id == Movie.id)
        .where(UserRecommendation.user_id == current_user.id)
        .order_by(UserRecommendation.rank)
        .limit(page.limit)
        .offset(page.offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.get(
    "/users/{user_id}/recommendations", response_model=list[UserRecommendationRead]
)
async def user_recommendations(
    user_id: int, db: DB, _: CurrentAdmin, page: PageParams
):
    stmt = (
        select(UserRecommendation)
        .where(UserRecommendation.user_id == user_id)
        .order_by(UserRecommendation.rank)
        .limit(page.limit)
        .offset(page.offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.put(
    "/users/{user_id}/recommendations/{rank}",
    response_model=UserRecommendationRead,
)
async def upsert_user_recommendation(
    user_id: int,
    rank: int,
    payload: UserRecommendationUpsert,
    db: DB,
    _: CurrentAdmin,
):
    if await db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    if await db.get(Movie, payload.movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    row = await db.get(UserRecommendation, (user_id, rank))
    if row is None:
        row = UserRecommendation(
            user_id=user_id,
            rank=rank,
            movie_id=payload.movie_id,
            score=payload.score,
        )
        db.add(row)
    else:
        row.movie_id = payload.movie_id
        row.score = payload.score
    await db.commit()
    await db.refresh(row)
    return row


@router.delete(
    "/users/{user_id}/recommendations/{rank}", response_model=Message
)
async def delete_user_recommendation(
    user_id: int, rank: int, db: DB, _: CurrentAdmin
):
    result = await db.execute(
        delete(UserRecommendation).where(
            UserRecommendation.user_id == user_id,
            UserRecommendation.rank == rank,
        )
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Entry not found")
    return Message(detail="Deleted")
