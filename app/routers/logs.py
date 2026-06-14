from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.deps import DB, CurrentAdmin
from app.models import Log, LogEntityType, LogEventType
from app.schemas import (
    LogCreate,
    LogRead,
    LogTypeCreate,
    LogTypeRead,
    LogTypeUpdate,
    Message,
)

router = APIRouter(tags=["logs"])


def _build_type_routes(model, name: str) -> APIRouter:
    sub = APIRouter(prefix=f"/{name}", tags=["logs"])

    @sub.get("", response_model=list[LogTypeRead])
    async def list_types(db: DB):
        return (await db.execute(select(model).order_by(model.id))).scalars().all()

    @sub.get("/{type_id}", response_model=LogTypeRead)
    async def get_type(type_id: int, db: DB):
        item = await db.get(model, type_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Type not found")
        return item

    @sub.post("", response_model=LogTypeRead, status_code=status.HTTP_201_CREATED)
    async def create_type(payload: LogTypeCreate, db: DB, _: CurrentAdmin):
        item = model(name=payload.name, description=payload.description)
        db.add(item)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=409, detail="Name already exists")
        await db.refresh(item)
        return item

    @sub.patch("/{type_id}", response_model=LogTypeRead)
    async def update_type(
        type_id: int, payload: LogTypeUpdate, db: DB, _: CurrentAdmin
    ):
        item = await db.get(model, type_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Type not found")
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(item, field, value)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            raise HTTPException(status_code=409, detail="Name already exists")
        await db.refresh(item)
        return item

    @sub.delete("/{type_id}", response_model=Message)
    async def delete_type(type_id: int, db: DB, _: CurrentAdmin):
        item = await db.get(model, type_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Type not found")
        await db.delete(item)
        await db.commit()
        return Message(detail="Type deleted")

    return sub


event_types_router = _build_type_routes(LogEventType, "log-event-types")
entity_types_router = _build_type_routes(LogEntityType, "log-entity-types")


logs_router = APIRouter(prefix="/logs", tags=["logs"])


@logs_router.get("", response_model=list[LogRead])
async def list_logs(
    db: DB,
    _: CurrentAdmin,
    user_id: int | None = Query(default=None),
    event_type_id: int | None = Query(default=None),
    entity_type_id: int | None = Query(default=None),
    entity_id: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    until: datetime | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    stmt = select(Log)
    if user_id is not None:
        stmt = stmt.where(Log.user_id == user_id)
    if event_type_id is not None:
        stmt = stmt.where(Log.event_type_id == event_type_id)
    if entity_type_id is not None:
        stmt = stmt.where(Log.entity_type_id == entity_type_id)
    if entity_id is not None:
        stmt = stmt.where(Log.entity_id == entity_id)
    if since is not None:
        stmt = stmt.where(Log.created_at >= since)
    if until is not None:
        stmt = stmt.where(Log.created_at <= until)
    stmt = stmt.order_by(Log.created_at.desc()).limit(limit).offset(offset)
    return (await db.execute(stmt)).scalars().all()


@logs_router.get("/count")
async def count_logs(db: DB, _: CurrentAdmin):
    total = await db.scalar(select(func.count()).select_from(Log))
    return {"count": total}


@logs_router.get("/{log_id}", response_model=LogRead)
async def get_log(log_id: int, db: DB, _: CurrentAdmin):
    log = await db.get(Log, log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")
    return log


@logs_router.post("", response_model=LogRead, status_code=status.HTTP_201_CREATED)
async def create_log(payload: LogCreate, db: DB, _: CurrentAdmin):
    if await db.get(LogEventType, payload.event_type_id) is None:
        raise HTTPException(status_code=404, detail="event_type_id not found")
    if (
        payload.entity_type_id is not None
        and await db.get(LogEntityType, payload.entity_type_id) is None
    ):
        raise HTTPException(status_code=404, detail="entity_type_id not found")
    log = Log(**payload.model_dump())
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


@logs_router.delete("/{log_id}", response_model=Message)
async def delete_log(log_id: int, db: DB, _: CurrentAdmin):
    log = await db.get(Log, log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")
    await db.delete(log)
    await db.commit()
    return Message(detail="Log deleted")
