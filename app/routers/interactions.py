from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.deps import DB, CurrentAdmin, CurrentUser, PageParams
from app.ml.realtime import handle_new_review
from app.ml.scoring import preference_score
from app.models import Interaction, Movie, User
from app.schemas import (
    InteractionAdminCreate,
    InteractionCreate,
    InteractionRead,
    InteractionUpdate,
    Message,
    MovieReviewRead,
)

router = APIRouter(prefix="/interactions", tags=["interactions"])


async def _ensure_movie(db, movie_id: int):
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")


async def _apply_scores(interaction: Interaction) -> None:
    """(Re)compute sentiment from the review text and preference_score from
    rating + sentiment. Runs the DistilBERT model only when text is present."""
    body = interaction.review_body
    if body and body.strip():
        try:
            from app.ml.sentiment import SentimentModel

            model = await run_in_threadpool(SentimentModel.instance)
        except Exception as exc:  # model missing / load failure
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Sentiment model unavailable: {exc}",
            )
        interaction.sentiment = await run_in_threadpool(model.predict_one, body)
    else:
        interaction.sentiment = None
    interaction.preference_score = preference_score(
        interaction.rating, interaction.sentiment
    )


@router.get("", response_model=list[InteractionRead])
async def list_interactions(
    db: DB,
    page: PageParams,
    _: CurrentAdmin,
    user_id: int | None = Query(default=None),
    movie_id: int | None = Query(default=None),
    has_rating: bool | None = Query(default=None),
    has_review: bool | None = Query(default=None),
):
    stmt = select(Interaction)
    if user_id is not None:
        stmt = stmt.where(Interaction.user_id == user_id)
    if movie_id is not None:
        stmt = stmt.where(Interaction.movie_id == movie_id)
    if has_rating is not None:
        stmt = stmt.where(
            Interaction.rating.is_not(None)
            if has_rating
            else Interaction.rating.is_(None)
        )
    if has_review is not None:
        stmt = stmt.where(
            Interaction.review_body.is_not(None)
            if has_review
            else Interaction.review_body.is_(None)
        )
    stmt = stmt.order_by(Interaction.id.desc()).limit(page.limit).offset(page.offset)
    return (await db.execute(stmt)).scalars().all()


@router.get("/me", response_model=list[InteractionRead])
async def list_my_interactions(db: DB, current_user: CurrentUser, page: PageParams):
    stmt = (
        select(Interaction)
        .where(Interaction.user_id == current_user.id)
        .order_by(Interaction.id.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.get("/movie/{movie_id}", response_model=list[MovieReviewRead])
async def list_movie_reviews(
    movie_id: int,
    db: DB,
    current_user: CurrentUser,
    page: PageParams,
    with_text_only: bool = Query(
        default=True, description="Only return interactions that have review text"),
):
    """Other users' reviews for a movie (for the movie page). Available to any
    logged-in user; exposes the author's username but not their email or the
    internal ML scores. Newest first."""
    await _ensure_movie(db, movie_id)
    stmt = (
        select(Interaction, User.username)
        .join(User, User.id == Interaction.user_id)
        .where(Interaction.movie_id == movie_id)
    )
    if with_text_only:
        stmt = stmt.where(func.length(func.trim(Interaction.review_body)) > 0)
    stmt = (
        stmt.order_by(
            func.coalesce(Interaction.review_date, Interaction.created_at).desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    rows = (await db.execute(stmt)).all()
    return [
        MovieReviewRead(
            id=i.id,
            user_id=i.user_id,
            username=username,
            rating=i.rating,
            review_title=i.review_title,
            review_body=i.review_body,
            review_date=i.review_date,
            created_at=i.created_at,
        )
        for i, username in rows
    ]


@router.get("/me/{movie_id}", response_model=InteractionRead)
async def get_my_interaction(movie_id: int, db: DB, current_user: CurrentUser):
    """The current user's own interaction for a movie (for the movie page).

    Returns 404 if they haven't reviewed/rated it yet. Available to any logged-in
    user — unlike the admin-only movie_id filter on `GET /interactions`."""
    stmt = select(Interaction).where(
        Interaction.user_id == current_user.id, Interaction.movie_id == movie_id
    )
    interaction = (await db.execute(stmt)).scalar_one_or_none()
    if interaction is None:
        raise HTTPException(status_code=404, detail="No interaction for this movie")
    return interaction


@router.post(
    "/me", response_model=InteractionRead, status_code=status.HTTP_201_CREATED
)
async def create_my_interaction(
    payload: InteractionCreate,
    db: DB,
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
):
    await _ensure_movie(db, payload.movie_id)
    interaction = Interaction(user_id=current_user.id, **payload.model_dump())
    await _apply_scores(interaction)
    db.add(interaction)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="You already have an interaction for this movie",
        )
    await db.refresh(interaction)
    await handle_new_review(
        db, current_user.id, interaction.movie_id,
        interaction.preference_score, background_tasks,
    )
    return interaction


@router.put("/me/{movie_id}", response_model=InteractionRead)
async def upsert_my_interaction(
    movie_id: int,
    payload: InteractionUpdate,
    db: DB,
    current_user: CurrentUser,
    background_tasks: BackgroundTasks,
):
    """Create or update the current user's interaction for a movie."""
    await _ensure_movie(db, movie_id)
    stmt = select(Interaction).where(
        Interaction.user_id == current_user.id, Interaction.movie_id == movie_id
    )
    interaction = (await db.execute(stmt)).scalar_one_or_none()
    if interaction is None:
        interaction = Interaction(
            user_id=current_user.id, movie_id=movie_id, **payload.model_dump()
        )
        db.add(interaction)
    else:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(interaction, field, value)
    await _apply_scores(interaction)
    await db.commit()
    await db.refresh(interaction)
    await handle_new_review(
        db, current_user.id, movie_id, interaction.preference_score, background_tasks,
    )
    return interaction


@router.post(
    "", response_model=InteractionRead, status_code=status.HTTP_201_CREATED
)
async def create_interaction(payload: InteractionAdminCreate, db: DB, _: CurrentAdmin):
    if await db.get(User, payload.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    await _ensure_movie(db, payload.movie_id)
    interaction = Interaction(**payload.model_dump())
    await _apply_scores(interaction)
    db.add(interaction)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail="Interaction already exists for user/movie"
        )
    await db.refresh(interaction)
    return interaction


@router.get("/{interaction_id}", response_model=InteractionRead)
async def get_interaction(
    interaction_id: int, db: DB, current_user: CurrentUser
):
    interaction = await db.get(Interaction, interaction_id)
    if interaction is None:
        raise HTTPException(status_code=404, detail="Interaction not found")
    if interaction.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not your interaction")
    return interaction


@router.patch("/{interaction_id}", response_model=InteractionRead)
async def update_interaction(
    interaction_id: int,
    payload: InteractionUpdate,
    db: DB,
    current_user: CurrentUser,
):
    interaction = await db.get(Interaction, interaction_id)
    if interaction is None:
        raise HTTPException(status_code=404, detail="Interaction not found")
    if interaction.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not your interaction")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(interaction, field, value)
    await _apply_scores(interaction)
    await db.commit()
    await db.refresh(interaction)
    return interaction


@router.delete("/{interaction_id}", response_model=Message)
async def delete_interaction(
    interaction_id: int, db: DB, current_user: CurrentUser
):
    interaction = await db.get(Interaction, interaction_id)
    if interaction is None:
        raise HTTPException(status_code=404, detail="Interaction not found")
    if interaction.user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not your interaction")
    await db.delete(interaction)
    await db.commit()
    return Message(detail="Interaction deleted")
