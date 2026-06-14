import asyncio
import time

import jwt
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_db import log_event
from app.core.security import decode_access_token
from app.routers import (
    admin_recs,
    auth,
    embeddings,
    interactions,
    logs,
    movies,
    recommendations,
    reference,
    stats,
    states,
    users,
)

app = FastAPI(
    title=settings.project_name,
    version="1.0.0",
    description="Movie recommendation backend API (FastAPI + PostgreSQL/pgvector).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints too noisy / irrelevant to log.
_LOG_SKIP_PATHS = {"/health", "/", "/openapi.json", "/docs", "/redoc", "/favicon.ico"}


def _user_id_from_request(request: Request) -> int | None:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    try:
        payload = decode_access_token(auth[7:])
        return int(payload["sub"]) if payload.get("type") == "access" else None
    except Exception:
        return None


@app.middleware("http")
async def request_logger(request: Request, call_next):
    """Log every mutating request (POST/PUT/PATCH/DELETE) to the logs table."""
    start = time.time()
    response = await call_next(request)
    path = request.url.path
    if request.method != "GET" and path not in _LOG_SKIP_PATHS:
        asyncio.create_task(log_event(
            "http_request",
            user_id=_user_id_from_request(request),
            entity_type="http",
            entity_id=path,
            details={
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "ms": round((time.time() - start) * 1000, 1),
            },
        ))
    return response


@app.get("/", tags=["health"])
async def root():
    return {"name": settings.project_name, "status": "ok", "docs": "/docs"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


# Auth & users
app.include_router(auth.router)
app.include_router(users.router)

# Reference / lookup tables
app.include_router(reference.people_router)
app.include_router(reference.roles_router)
app.include_router(reference.countries_router)
app.include_router(reference.genres_router)
app.include_router(reference.languages_router)

# Catalogue
app.include_router(movies.router)

# User activity
app.include_router(interactions.router)
app.include_router(states.router)

# ML: embeddings & recommendations
app.include_router(embeddings.router)
app.include_router(recommendations.router)

# Analytics
app.include_router(stats.router)

# Recommendation admin (jobs, models, retrain)
app.include_router(admin_recs.router)

# Logging
app.include_router(logs.event_types_router)
app.include_router(logs.entity_types_router)
app.include_router(logs.logs_router)
