"""People router with lazy TMDB biography enrichment.

Replaces the generic reference CRUD for `people`: same list/count/create/update/
delete behaviour, but `GET /people/{id}` returns a richer profile and, on first
view, fetches the person's biography/photo/birthday from TMDB by name and caches
it. `tmdb_checked_at` is only set on a successful TMDB response (hit *or* genuine
miss) so transient network errors retry instead of poisoning the cache.
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.deps import DB, CurrentAdmin, PageParams
from app.integrations import tmdb
from app.models import Person
from app.schemas import Message, NamedCreate, NamedRead, NamedUpdate, PersonDetail

router = APIRouter(prefix="/people", tags=["people"])


async def _enrich_from_tmdb(db: DB, person: Person) -> None:
    """Populate biography fields from TMDB on first view; no-op if cached."""
    if person.tmdb_checked_at is not None or not tmdb.enabled():
        return
    try:
        summary = await tmdb.search_person(person.name)
        detail = await tmdb.get_person(summary["id"]) if summary else None
    except tmdb.TMDBError:
        return  # transient failure — leave uncached so the next view retries
    # Reached TMDB successfully (even if it had no such person): cache the result.
    person.tmdb_checked_at = datetime.utcnow()
    if detail:
        person.tmdb_id = detail.get("id")
        person.biography = (detail.get("biography") or "").strip() or None
        person.profile_path = detail.get("profile_path")
        person.birthday = detail.get("birthday")
    await db.commit()
    await db.refresh(person)


def _to_detail(person: Person) -> PersonDetail:
    return PersonDetail(
        id=person.id,
        name=person.name,
        tmdb_id=person.tmdb_id,
        biography=person.biography,
        profile_path=person.profile_path,
        profile_url=tmdb.image_url(person.profile_path),
        birthday=person.birthday,
    )


@router.get("", response_model=list[NamedRead])
async def list_people(
    db: DB, page: PageParams, search: str | None = Query(default=None)
):
    stmt = select(Person)
    if search:
        stmt = stmt.where(Person.name.ilike(f"%{search}%"))
    stmt = stmt.order_by(Person.name).limit(page.limit).offset(page.offset)
    return (await db.execute(stmt)).scalars().all()


@router.get("/count")
async def count_people(db: DB):
    total = await db.scalar(select(func.count()).select_from(Person))
    return {"count": total}


@router.get("/{person_id}", response_model=PersonDetail)
async def get_person(person_id: int, db: DB):
    """Person profile. Lazily fetches & caches the TMDB biography on first view."""
    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="people not found")
    await _enrich_from_tmdb(db, person)
    return _to_detail(person)


@router.post("", response_model=NamedRead, status_code=status.HTTP_201_CREATED)
async def create_person(payload: NamedCreate, db: DB, _: CurrentAdmin):
    person = Person(name=payload.name)
    db.add(person)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Name already exists")
    await db.refresh(person)
    return person


@router.put("/{person_id}", response_model=NamedRead)
async def update_person(
    person_id: int, payload: NamedUpdate, db: DB, _: CurrentAdmin
):
    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="people not found")
    person.name = payload.name
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Name already exists")
    await db.refresh(person)
    return person


@router.delete("/{person_id}", response_model=Message)
async def delete_person(person_id: int, db: DB, _: CurrentAdmin):
    person = await db.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="people not found")
    await db.delete(person)
    await db.commit()
    return Message(detail="people deleted")
