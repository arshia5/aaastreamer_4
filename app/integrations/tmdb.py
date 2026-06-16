"""Thin async client for the TMDB v3 REST API.

Used to lazily enrich person biographies and movie collections on first view,
caching the result in our own DB so each entity hits TMDB at most once.

All calls are best-effort: network/HTTP errors raise `TMDBError`, callers decide
how to degrade. A missing API key disables the integration (`enabled` is False).
"""
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger("app.integrations.tmdb")

BASE_URL = "https://api.themoviedb.org/3"
# Public CDN base for `*_path` image fields. Callers pick a size segment.
IMAGE_BASE = "https://image.tmdb.org/t/p"
_TIMEOUT = httpx.Timeout(8.0)


class TMDBError(RuntimeError):
    """Raised when a TMDB request fails (network, timeout, non-2xx)."""


def enabled() -> bool:
    return bool(settings.tmdb_api_key)


def image_url(path: str | None, size: str = "w500") -> str | None:
    """Build a full image URL from a TMDB `*_path`, or None if absent."""
    if not path:
        return None
    return f"{IMAGE_BASE}/{size}{path}"


async def _get(path: str, params: dict | None = None) -> dict:
    if not enabled():
        raise TMDBError("TMDB_API_KEY is not configured")
    params = {**(params or {}), "api_key": settings.tmdb_api_key}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{BASE_URL}{path}", params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.warning("TMDB request failed: %s %s", path, exc)
        raise TMDBError(str(exc)) from exc


# --- People ---------------------------------------------------------------- #
async def search_person(name: str) -> dict | None:
    """Return the best-matching person summary for a name, or None.

    TMDB orders search results by relevance/popularity, so the first hit is the
    canonical person in the overwhelming majority of cases.
    """
    data = await _get("/search/person", {"query": name, "include_adult": "false"})
    results = data.get("results") or []
    return results[0] if results else None


async def get_person(tmdb_id: int) -> dict:
    """Full person detail (biography, birthday, profile_path, ...)."""
    return await _get(f"/person/{tmdb_id}")


# --- Movies & collections -------------------------------------------------- #
async def find_movie_by_imdb(imdb_id: str) -> dict | None:
    """Resolve a TMDB movie summary from an IMDb id via /find, or None."""
    data = await _get(f"/find/{imdb_id}", {"external_source": "imdb_id"})
    results = data.get("movie_results") or []
    return results[0] if results else None


async def get_movie(tmdb_id: int) -> dict:
    """Full movie detail; includes `belongs_to_collection` when applicable."""
    return await _get(f"/movie/{tmdb_id}")


async def get_collection(collection_id: int) -> dict:
    """Collection detail including its `parts` (member movies)."""
    return await _get(f"/collection/{collection_id}")
