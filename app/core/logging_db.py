"""Database event logging — writes rows to the `logs` table.

Three entry points:
- `write_log(db, ...)`   : within an existing AsyncSession (caller commits)
- `log_event(...)`       : fire-and-forget, opens its own session + commits
- `log_pg(conn, ...)`    : from an asyncpg connection (training jobs / scripts)

Event/entity type ids are resolved by name and cached in-process.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal

log = logging.getLogger("recsys.logs")

_event_cache: dict[str, int] = {}
_entity_cache: dict[str, int] = {}


async def _type_id(db, table: str, cache: dict, name: str) -> int:
    if name in cache:
        return cache[name]
    rid = await db.scalar(
        text(f"SELECT id FROM {table} WHERE name = :n"), {"n": name})
    if rid is None:
        rid = await db.scalar(
            text(f"INSERT INTO {table}(name) VALUES (:n) "
                 f"ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"),
            {"n": name})
    cache[name] = rid
    return rid


async def write_log(
    db: AsyncSession, event: str, *, user_id: int | None = None,
    entity_type: str | None = None, entity_id=None, details: dict | None = None,
) -> None:
    event_id = await _type_id(db, "log_event_types", _event_cache, event)
    entity_type_id = (
        await _type_id(db, "log_entity_types", _entity_cache, entity_type)
        if entity_type else None)
    await db.execute(
        text("INSERT INTO logs(user_id, event_type_id, entity_type_id, entity_id, details) "
             "VALUES (:u, :e, :et, :eid, CAST(:d AS jsonb))"),
        {"u": user_id, "e": event_id, "et": entity_type_id,
         "eid": None if entity_id is None else str(entity_id),
         "d": json.dumps(details or {})},
    )


async def log_event(event: str, **kwargs) -> None:
    """Fire-and-forget: own session + commit. Never raises."""
    try:
        async with AsyncSessionLocal() as db:
            await write_log(db, event, **kwargs)
            await db.commit()
    except Exception:
        log.warning("Failed to write log event %s", event, exc_info=True)


async def log_pg(conn, event: str, *, user_id: int | None = None,
                 entity_type: str | None = None, entity_id=None,
                 details: dict | None = None) -> None:
    """Log from an asyncpg connection (training jobs). Best-effort."""
    try:
        await conn.execute(
            "INSERT INTO logs(user_id, event_type_id, entity_type_id, entity_id, details) "
            "SELECT $1, (SELECT id FROM log_event_types WHERE name=$2), "
            "(SELECT id FROM log_entity_types WHERE name=$3), $4, $5::jsonb",
            user_id, event, entity_type,
            None if entity_id is None else str(entity_id),
            json.dumps(details or {}),
        )
    except Exception:
        log.warning("Failed to write pg log event %s", event, exc_info=True)
