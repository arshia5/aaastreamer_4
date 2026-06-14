from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
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
