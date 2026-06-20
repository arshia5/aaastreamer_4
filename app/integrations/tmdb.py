"""Thin async client for the TMDB v3 REST API.

Used to lazily enrich person biographies and movie collections on first view,
caching the result in our own DB so each entity hits TMDB at most once.

All calls are best-effort: network/HTTP errors raise `TMDBError`, callers decide
how to degrade. A missing API key disables the integration (`enabled` is False).
"""
import asyncio
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger("app.integrations.tmdb")

BASE_URL = "https://api.themoviedb.org/3"
# Public CDN base for `*_path` image fields. Callers pick a size segment.
IMAGE_BASE = "https://image.tmdb.org/t/p"
_TIMEOUT = httpx.Timeout(8.0)

# Transient upstream conditions worth retrying: rate-limit + 5xx (502/503/504 are
# common when TMDB's edge is flapping). 4xx like 401/404 are not retried.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5  # seconds; doubled each retry


class TMDBError(RuntimeError):
    """Raised when a TMDB request fails (network, timeout, non-2xx)."""


def enabled() -> bool:
    return bool(settings.tmdb_api_key)


def image_url(path: str | None, size: str = "w500") -> str | None:
    """Build a full image URL from a TMDB `*_path`, or None if absent."""
    if not path:
        return None
    return f"{IMAGE_BASE}/{size}{path}"


async def _get(path: str, params: dict | None = None, *, retry: bool = False) -> dict:
    """Fetch a TMDB endpoint as JSON.

    retry=False (the default) fails fast on the first error: TMDB caches its 502s
    at the CDN edge, so rapid in-process retries just re-hit the same cached error
    while adding latency — bad for the user-blocking lazy-enrichment callers, which
    degrade gracefully and re-query on the next view anyway. Set retry=True only
    for non-blocking contexts (e.g. a background backfill job) where waiting out a
    genuinely transient hiccup is worth it.
    """
    if not enabled():
        raise TMDBError("TMDB_API_KEY is not configured")
    params = {**(params or {}), "api_key": settings.tmdb_api_key}
    max_attempts = _MAX_ATTEMPTS if retry else 1
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.get(f"{BASE_URL}{path}", params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                # Non-2xx: retry only transient upstream statuses, else fail fast.
                if exc.response.status_code not in _RETRY_STATUS:
                    logger.warning("TMDB request failed: %s %s", path, exc)
                    raise TMDBError(str(exc)) from exc
            except httpx.HTTPError as exc:
                # Network/timeout/connection errors are transient -> retry.
                last_exc = exc
            if attempt < max_attempts:
                delay = _BACKOFF_BASE * 2 ** (attempt - 1)
                logger.info("TMDB %s transient failure (attempt %d/%d), retrying in %.1fs: %s",
                            path, attempt, max_attempts, delay, last_exc)
                await asyncio.sleep(delay)
    if max_attempts > 1:
        logger.warning("TMDB request failed after %d attempts: %s %s",
                       max_attempts, path, last_exc)
    else:
        logger.warning("TMDB request failed: %s %s", path, last_exc)
    raise TMDBError(str(last_exc)) from last_exc


# --- People ---------------------------------------------------------------- #
async def search_person(name: str, *, retry: bool = False) -> dict | None:
    """Return the best-matching person summary for a name, or None.

    TMDB orders search results by relevance/popularity, so the first hit is the
    canonical person in the overwhelming majority of cases.
    """
    data = await _get("/search/person", {"query": name, "include_adult": "false"},
                      retry=retry)
    results = data.get("results") or []
    return results[0] if results else None


async def get_person(tmdb_id: int, *, retry: bool = False) -> dict:
    """Full person detail (biography, birthday, profile_path, ...)."""
    return await _get(f"/person/{tmdb_id}", retry=retry)


# --- Movies & collections -------------------------------------------------- #
async def find_movie_by_imdb(imdb_id: str, *, retry: bool = False) -> dict | None:
    """Resolve a TMDB movie summary from an IMDb id via /find, or None."""
    data = await _get(f"/find/{imdb_id}", {"external_source": "imdb_id"}, retry=retry)
    results = data.get("movie_results") or []
    return results[0] if results else None


async def get_movie(tmdb_id: int, *, retry: bool = False) -> dict:
    """Full movie detail; includes `belongs_to_collection` when applicable."""
    return await _get(f"/movie/{tmdb_id}", retry=retry)


async def get_collection(collection_id: int, *, retry: bool = False) -> dict:
    """Collection detail including its `parts` (member movies)."""
    return await _get(f"/collection/{collection_id}", retry=retry)
