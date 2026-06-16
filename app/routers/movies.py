from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.deps import DB, CurrentAdmin, PageParams
from app.integrations import tmdb
from app.ml.movie_refresh import schedule_movie_refresh
from app.models import (
    Collection,
    Country,
    Genre,
    Language,
    Movie,
    MovieCountry,
    MovieGenre,
    MovieLanguage,
    MoviePerson,
    Person,
    Role,
)
from app.schemas import (
    CollectionRead,
    Message,
    MovieCreate,
    MovieDetail,
    MoviePersonCreate,
    MoviePersonRead,
    MovieRead,
    MovieRefLink,
    MovieUpdate,
    NamedRead,
)

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("", response_model=list[MovieRead])
async def list_movies(
    db: DB,
    page: PageParams,
    search: str | None = Query(default=None, description="Title substring"),
    year: int | None = Query(default=None),
    genre_id: int | None = Query(default=None),
    country_id: int | None = Query(default=None),
    language_id: int | None = Query(default=None),
    sort: str = Query(default="title", pattern="^(title|year|created|-created|-year)$"),
):
    stmt = select(Movie)
    if search:
        stmt = stmt.where(Movie.movie_title.ilike(f"%{search}%"))
    if year is not None:
        stmt = stmt.where(Movie.year == year)
    if genre_id is not None:
        stmt = stmt.where(
            Movie.id.in_(select(MovieGenre.movie_id).where(MovieGenre.genre_id == genre_id))
        )
    if country_id is not None:
        stmt = stmt.where(
            Movie.id.in_(
                select(MovieCountry.movie_id).where(MovieCountry.country_id == country_id)
            )
        )
    if language_id is not None:
        stmt = stmt.where(
            Movie.id.in_(
                select(MovieLanguage.movie_id).where(
                    MovieLanguage.language_id == language_id
                )
            )
        )
    order = {
        "title": Movie.movie_title.asc(),
        "year": Movie.year.asc(),
        "-year": Movie.year.desc(),
        "created": Movie.created_at.asc(),
        "-created": Movie.created_at.desc(),
    }[sort]
    stmt = stmt.order_by(order).limit(page.limit).offset(page.offset)
    return (await db.execute(stmt)).scalars().all()


@router.get("/count")
async def count_movies(db: DB):
    total = await db.scalar(select(func.count()).select_from(Movie))
    return {"count": total}


@router.get("/by-imdb/{imdb_id}", response_model=MovieDetail)
async def get_movie_by_imdb(imdb_id: str, db: DB):
    stmt = (
        select(Movie)
        .where(Movie.imdb_id == imdb_id)
        .options(
            selectinload(Movie.genres),
            selectinload(Movie.countries),
            selectinload(Movie.languages),
        )
    )
    movie = (await db.execute(stmt)).scalar_one_or_none()
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


@router.get("/{movie_id}", response_model=MovieDetail)
async def get_movie(movie_id: int, db: DB):
    stmt = (
        select(Movie)
        .where(Movie.id == movie_id)
        .options(
            selectinload(Movie.genres),
            selectinload(Movie.countries),
            selectinload(Movie.languages),
        )
    )
    movie = (await db.execute(stmt)).scalar_one_or_none()
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


# --------------------------------------------------------------------------- #
# TMDB collection (lazily fetched + cached on first view)
# --------------------------------------------------------------------------- #
async def _upsert_collection(db: DB, data: dict) -> None:
    """Insert/update the collections row from TMDB collection detail."""
    coll = await db.get(Collection, data["id"])
    if coll is None:
        coll = Collection(id=data["id"], name=data.get("name") or "")
        db.add(coll)
    coll.name = data.get("name") or coll.name
    coll.overview = data.get("overview") or None
    coll.poster_path = data.get("poster_path")
    coll.backdrop_path = data.get("backdrop_path")


async def _map_collection_members(db: DB, collection_id: int, parts: list[dict]) -> None:
    """Link any collection members that already exist in our catalogue.

    Matches by TMDB id first, then by (title, year). Members not in our DB are
    simply skipped — we only surface films we actually have."""
    now = datetime.utcnow()
    for part in parts:
        part_tmdb_id = part.get("id")
        movie = None
        if part_tmdb_id is not None:
            movie = await db.scalar(select(Movie).where(Movie.tmdb_id == part_tmdb_id))
        if movie is None:
            title = (part.get("title") or "").strip()
            if not title:
                continue
            rel = part.get("release_date") or ""
            year = int(rel[:4]) if rel[:4].isdigit() else None
            stmt = select(Movie).where(func.lower(Movie.movie_title) == title.lower())
            if year is not None:
                stmt = stmt.where(Movie.year == year)
            movie = await db.scalar(stmt.limit(1))
        if movie is not None:
            # We now know this movie's collection — cache it so clicking it later
            # doesn't re-query TMDB.
            movie.tmdb_id = part_tmdb_id
            movie.tmdb_collection_id = collection_id
            movie.tmdb_checked_at = now


async def _resolve_collection(db: DB, movie: Movie) -> None:
    """On first view, resolve the movie's TMDB collection and cache it.

    `tmdb_checked_at` is set only on a successful TMDB response (collection found
    *or* confirmed absent), so a movie with no collection isn't re-queried while
    transient network errors still retry."""
    if movie.tmdb_checked_at is not None or not tmdb.enabled():
        return
    try:
        tmdb_id = movie.tmdb_id
        if tmdb_id is None:
            summary = await tmdb.find_movie_by_imdb(movie.imdb_id)
            tmdb_id = summary["id"] if summary else None
        detail = await tmdb.get_movie(tmdb_id) if tmdb_id else None
        belongs = (detail or {}).get("belongs_to_collection")
        coll_data = await tmdb.get_collection(belongs["id"]) if belongs else None
    except tmdb.TMDBError:
        return  # transient — retry on next view, don't poison the cache
    movie.tmdb_checked_at = datetime.utcnow()
    if tmdb_id is not None:
        movie.tmdb_id = tmdb_id
    if coll_data:
        await _upsert_collection(db, coll_data)
        await _map_collection_members(db, coll_data["id"], coll_data.get("parts") or [])
        movie.tmdb_collection_id = coll_data["id"]
    await db.commit()
    await db.refresh(movie)


@router.get("/{movie_id}/collection", response_model=CollectionRead | None)
async def movie_collection(movie_id: int, db: DB):
    """The movie's TMDB collection with the member films we have in our catalogue,
    or `null` if it doesn't belong to one. Fetched from TMDB on first view, cached
    thereafter (including the 'no collection' result)."""
    movie = await db.get(Movie, movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    await _resolve_collection(db, movie)
    if movie.tmdb_collection_id is None:
        return None
    coll = await db.get(Collection, movie.tmdb_collection_id)
    if coll is None:
        return None
    members = (
        await db.execute(
            select(Movie)
            .where(Movie.tmdb_collection_id == coll.id)
            .order_by(Movie.year.asc().nulls_last(), Movie.movie_title.asc())
        )
    ).scalars().all()
    return CollectionRead(
        id=coll.id,
        name=coll.name,
        overview=coll.overview,
        poster_path=coll.poster_path,
        poster_url=tmdb.image_url(coll.poster_path),
        backdrop_path=coll.backdrop_path,
        backdrop_url=tmdb.image_url(coll.backdrop_path, "w780"),
        movies=members,
    )


@router.post("", response_model=MovieRead, status_code=status.HTTP_201_CREATED)
async def create_movie(
    payload: MovieCreate, db: DB, _: CurrentAdmin, background_tasks: BackgroundTasks
):
    movie = Movie(**payload.model_dump())
    db.add(movie)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="imdb_id already exists")
    await db.refresh(movie)
    # Automatically build the content embedding + similar movies (background).
    schedule_movie_refresh(background_tasks, movie.id)
    return movie


@router.patch("/{movie_id}", response_model=MovieRead)
async def update_movie(
    movie_id: int, payload: MovieUpdate, db: DB, _: CurrentAdmin,
    background_tasks: BackgroundTasks,
):
    movie = await db.get(Movie, movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    data = payload.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(movie, field, value)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="imdb_id already exists")
    await db.refresh(movie)
    # Re-embed only if a content-affecting field changed (plot/year/title).
    if data.keys() & {"plot", "year", "movie_title"}:
        schedule_movie_refresh(background_tasks, movie.id)
    return movie


@router.delete("/{movie_id}", response_model=Message)
async def delete_movie(movie_id: int, db: DB, _: CurrentAdmin):
    movie = await db.get(Movie, movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    await db.delete(movie)
    await db.commit()
    return Message(detail="Movie deleted")


# --------------------------------------------------------------------------- #
# Associations
# --------------------------------------------------------------------------- #
async def _ensure_movie(db, movie_id: int) -> None:
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")


async def _link_ref(db, link_model, ref_model, movie_id, ref_id, fk_name, bg=None):
    await _ensure_movie(db, movie_id)
    if await db.get(ref_model, ref_id) is None:
        raise HTTPException(status_code=404, detail=f"{ref_model.__name__} not found")
    db.add(link_model(movie_id=movie_id, **{fk_name: ref_id}))
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Link already exists")
    schedule_movie_refresh(bg, movie_id)  # metadata changed -> re-embed
    return Message(detail="Linked")


async def _unlink_ref(db, link_model, movie_id, ref_id, fk_col, bg=None):
    result = await db.execute(
        delete(link_model).where(
            link_model.movie_id == movie_id, fk_col == ref_id
        )
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Link not found")
    schedule_movie_refresh(bg, movie_id)
    return Message(detail="Unlinked")


# genres
@router.get("/{movie_id}/genres", response_model=list[NamedRead])
async def movie_genres(movie_id: int, db: DB):
    await _ensure_movie(db, movie_id)
    stmt = select(Genre).join(MovieGenre, MovieGenre.genre_id == Genre.id).where(
        MovieGenre.movie_id == movie_id
    )
    return (await db.execute(stmt)).scalars().all()


@router.post("/{movie_id}/genres", response_model=Message, status_code=201)
async def add_movie_genre(movie_id: int, payload: MovieRefLink, db: DB,
                          _: CurrentAdmin, background_tasks: BackgroundTasks):
    return await _link_ref(db, MovieGenre, Genre, movie_id, payload.ref_id,
                           "genre_id", background_tasks)


@router.delete("/{movie_id}/genres/{genre_id}", response_model=Message)
async def remove_movie_genre(movie_id: int, genre_id: int, db: DB,
                             _: CurrentAdmin, background_tasks: BackgroundTasks):
    return await _unlink_ref(db, MovieGenre, movie_id, genre_id,
                             MovieGenre.genre_id, background_tasks)


# countries
@router.get("/{movie_id}/countries", response_model=list[NamedRead])
async def movie_countries(movie_id: int, db: DB):
    await _ensure_movie(db, movie_id)
    stmt = select(Country).join(
        MovieCountry, MovieCountry.country_id == Country.id
    ).where(MovieCountry.movie_id == movie_id)
    return (await db.execute(stmt)).scalars().all()


@router.post("/{movie_id}/countries", response_model=Message, status_code=201)
async def add_movie_country(movie_id: int, payload: MovieRefLink, db: DB,
                            _: CurrentAdmin, background_tasks: BackgroundTasks):
    return await _link_ref(db, MovieCountry, Country, movie_id, payload.ref_id,
                           "country_id", background_tasks)


@router.delete("/{movie_id}/countries/{country_id}", response_model=Message)
async def remove_movie_country(movie_id: int, country_id: int, db: DB,
                               _: CurrentAdmin, background_tasks: BackgroundTasks):
    return await _unlink_ref(db, MovieCountry, movie_id, country_id,
                             MovieCountry.country_id, background_tasks)


# languages
@router.get("/{movie_id}/languages", response_model=list[NamedRead])
async def movie_languages(movie_id: int, db: DB):
    await _ensure_movie(db, movie_id)
    stmt = select(Language).join(
        MovieLanguage, MovieLanguage.language_id == Language.id
    ).where(MovieLanguage.movie_id == movie_id)
    return (await db.execute(stmt)).scalars().all()


@router.post("/{movie_id}/languages", response_model=Message, status_code=201)
async def add_movie_language(movie_id: int, payload: MovieRefLink, db: DB,
                             _: CurrentAdmin, background_tasks: BackgroundTasks):
    return await _link_ref(db, MovieLanguage, Language, movie_id, payload.ref_id,
                           "language_id", background_tasks)


@router.delete("/{movie_id}/languages/{language_id}", response_model=Message)
async def remove_movie_language(movie_id: int, language_id: int, db: DB,
                                _: CurrentAdmin, background_tasks: BackgroundTasks):
    return await _unlink_ref(db, MovieLanguage, movie_id, language_id,
                             MovieLanguage.language_id, background_tasks)


# people (cast & crew, with role)
@router.get("/{movie_id}/people", response_model=list[MoviePersonRead])
async def movie_people(movie_id: int, db: DB):
    await _ensure_movie(db, movie_id)
    stmt = select(MoviePerson).where(MoviePerson.movie_id == movie_id)
    return (await db.execute(stmt)).scalars().all()


@router.post("/{movie_id}/people", response_model=Message, status_code=201)
async def add_movie_person(
    movie_id: int, payload: MoviePersonCreate, db: DB, _: CurrentAdmin,
    background_tasks: BackgroundTasks,
):
    await _ensure_movie(db, movie_id)
    if await db.get(Person, payload.person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    if await db.get(Role, payload.role_id) is None:
        raise HTTPException(status_code=404, detail="Role not found")
    db.add(
        MoviePerson(
            movie_id=movie_id, person_id=payload.person_id, role_id=payload.role_id
        )
    )
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Link already exists")
    schedule_movie_refresh(background_tasks, movie_id)
    return Message(detail="Linked")


@router.delete(
    "/{movie_id}/people/{person_id}/roles/{role_id}", response_model=Message
)
async def remove_movie_person(
    movie_id: int, person_id: int, role_id: int, db: DB, _: CurrentAdmin,
    background_tasks: BackgroundTasks,
):
    result = await db.execute(
        delete(MoviePerson).where(
            MoviePerson.movie_id == movie_id,
            MoviePerson.person_id == person_id,
            MoviePerson.role_id == role_id,
        )
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Link not found")
    schedule_movie_refresh(background_tasks, movie_id)
    return Message(detail="Unlinked")
