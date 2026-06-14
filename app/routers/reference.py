from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.deps import DB, CurrentAdmin, PageParams
from app.models import Country, Genre, Language, Person, Role
from app.schemas import Message, NamedCreate, NamedRead, NamedUpdate


def build_reference_router(model, name: str, tag: str) -> APIRouter:
    """CRUD router for a simple `(id, name unique)` lookup table."""
    router = APIRouter(prefix=f"/{name}", tags=[tag])

    @router.get("", response_model=list[NamedRead])
    async def list_items(
        db: DB,
        page: PageParams,
        search: str | None = Query(default=None),
    ):
        stmt = select(model)
        if search:
            stmt = stmt.where(model.name.ilike(f"%{search}%"))
        stmt = stmt.order_by(model.name).limit(page.limit).offset(page.offset)
        return (await db.execute(stmt)).scalars().all()

    @router.get("/count")
    async def count_items(db: DB):
        total = await db.scalar(select(func.count()).select_from(model))
        return {"count": total}

    @router.get("/{item_id}", response_model=NamedRead)
    async def get_item(item_id: int, db: DB):
        item = await db.get(model, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        return item

    @router.post(
        "", response_model=NamedRead, status_code=status.HTTP_201_CREATED
    )
    async def create_item(payload: NamedCreate, db: DB, _: CurrentAdmin):
        item = model(name=payload.name)
        db.add(item)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=409, detail="Name already exists")
        await db.refresh(item)
        return item

    @router.put("/{item_id}", response_model=NamedRead)
    async def update_item(
        item_id: int, payload: NamedUpdate, db: DB, _: CurrentAdmin
    ):
        item = await db.get(model, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        item.name = payload.name
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=409, detail="Name already exists")
        await db.refresh(item)
        return item

    @router.delete("/{item_id}", response_model=Message)
    async def delete_item(item_id: int, db: DB, _: CurrentAdmin):
        item = await db.get(model, item_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"{tag} not found")
        await db.delete(item)
        await db.commit()
        return Message(detail=f"{tag} deleted")

    return router


people_router = build_reference_router(Person, "people", "people")
roles_router = build_reference_router(Role, "roles", "roles")
countries_router = build_reference_router(Country, "countries", "countries")
genres_router = build_reference_router(Genre, "genres", "genres")
languages_router = build_reference_router(Language, "languages", "languages")
