from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import delete, func, select

from app.deps import DB, CurrentUser, PageParams
from app.models import Movie, UserMovieState, UserMovieStateType
from app.schemas import (
    Message,
    MovieRead,
    UserMovieStateCreate,
    UserMovieStateRead,
)

router = APIRouter(prefix="/me/movie-states", tags=["user-movie-states"])


@router.get("", response_model=list[UserMovieStateRead])
async def list_my_states(
    db: DB,
    current_user: CurrentUser,
    page: PageParams,
    state: UserMovieStateType | None = Query(default=None),
):
    stmt = select(UserMovieState).where(UserMovieState.user_id == current_user.id)
    if state is not None:
        stmt = stmt.where(UserMovieState.state == state)
    stmt = stmt.order_by(UserMovieState.updated_at.desc()).limit(page.limit).offset(
        page.offset
    )
    return (await db.execute(stmt)).scalars().all()


@router.get("/movies", response_model=list[MovieRead])
async def list_my_state_movies(
    db: DB,
    current_user: CurrentUser,
    page: PageParams,
    state: UserMovieStateType = Query(...),
):
    """Return the movies the current user has in a given state."""
    stmt = (
        select(Movie)
        .join(UserMovieState, UserMovieState.movie_id == Movie.id)
        .where(
            UserMovieState.user_id == current_user.id,
            UserMovieState.state == state,
        )
        .order_by(UserMovieState.updated_at.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.put("/{movie_id}", response_model=UserMovieStateRead)
async def set_my_state(
    movie_id: int,
    payload: UserMovieStateCreate,
    db: DB,
    current_user: CurrentUser,
):
    """Set or update watched/watchlist state for a movie."""
    if movie_id != payload.movie_id:
        raise HTTPException(status_code=400, detail="movie_id mismatch")
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    existing = await db.get(UserMovieState, (current_user.id, movie_id))
    if existing is None:
        existing = UserMovieState(
            user_id=current_user.id, movie_id=movie_id, state=payload.state
        )
        db.add(existing)
    else:
        existing.state = payload.state
    await db.commit()
    await db.refresh(existing)
    return existing


@router.delete("/{movie_id}", response_model=Message)
async def delete_my_state(movie_id: int, db: DB, current_user: CurrentUser):
    result = await db.execute(
        delete(UserMovieState).where(
            UserMovieState.user_id == current_user.id,
            UserMovieState.movie_id == movie_id,
        )
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="State not found")
    return Message(detail="State removed")
