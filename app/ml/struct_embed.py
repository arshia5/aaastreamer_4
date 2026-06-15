"""v5 structured-metadata channel.

Rebuilds the v3 structured block embedding (genre multi-hot + hashed director/
writer/actor/language/country + year), L2-normalised, but WITHOUT the plot block,
straight from the normalised DB tables. Used as a complementary retrieval channel
to the mpnet metadata template (see config.STRUCT_ENABLED).

Stateful only through the fitted v3 pipeline artifact (genre MLB + year scaler);
the hashing blocks are stateless. Returns a unit-row matrix aligned to ctx.ids.
"""
from __future__ import annotations

import logging

import joblib
import numpy as np

from app.ml import config
from app.ml.features import genre_block, hashed_block, year_block, l2_normalize

log = logging.getLogger("recsys.struct")

_STRUCT_ORDER = ["genre", "director", "writer", "actor", "language", "country", "numeric"]


async def build_struct_matrix(conn, ids: np.ndarray):
    """Return (matrix (N, D) unit rows, mask) aligned to ``ids`` (movie order).

    Returns (None, None) if the fitted pipeline artifact is missing.
    """
    if not config.STRUCT_ENABLED or not config.PIPELINE_PATH.exists():
        return None, None
    art = joblib.load(config.PIPELINE_PATH)
    mlb, scaler, med = art["genre_mlb"], art["year_scaler"], art["median_year"]

    idx = {int(m): i for i, m in enumerate(ids)}
    n = len(ids)
    toks = {k: [[] for _ in range(n)] for k in
            ["genre", "director", "writer", "actor", "language", "country"]}
    year = [None] * n
    for mid, y in await conn.fetch("SELECT id, year FROM movies WHERE year IS NOT NULL"):
        j = idx.get(mid)
        if j is not None:
            year[j] = y

    async def _fill(key, sql, name_sql):
        names = {r["id"]: r["name"] for r in await conn.fetch(name_sql)}
        for mid, oid in await conn.fetch(sql):
            j = idx.get(mid)
            if j is not None:
                toks[key][j].append(names.get(oid, ""))

    await _fill("genre", "SELECT movie_id, genre_id FROM movie_genres",
                "SELECT id, name FROM genres")
    await _fill("language", "SELECT movie_id, language_id FROM movie_languages",
                "SELECT id, name FROM languages")
    await _fill("country", "SELECT movie_id, country_id FROM movie_countries",
                "SELECT id, name FROM countries")
    # people split by role
    pname = {r["id"]: r["name"] for r in await conn.fetch("SELECT id, name FROM people")}
    role = {r["id"]: r["name"] for r in await conn.fetch("SELECT id, name FROM roles")}
    keymap = {"director": "director", "writer": "writer", "actor": "actor"}
    for mid, pid, rid in await conn.fetch(
            "SELECT movie_id, person_id, role_id FROM movie_people"):
        j = idx.get(mid)
        key = keymap.get(role.get(rid, ""))
        if j is not None and key:
            toks[key][j].append(pname.get(pid, ""))

    blocks = {
        "genre": genre_block(toks["genre"], mlb),
        "director": hashed_block(toks["director"], config.DIRECTOR_DIM),
        "writer": hashed_block(toks["writer"], config.WRITER_DIM),
        "actor": hashed_block(toks["actor"], config.ACTOR_DIM),
        "language": hashed_block(toks["language"], config.LANG_DIM),
        "country": hashed_block(toks["country"], config.COUNTRY_DIM),
        "numeric": year_block(year, scaler, med),
    }
    parts = []
    for name in _STRUCT_ORDER:
        m = blocks[name]
        w = config.WEIGHTS[name]
        parts.append((m * w) if name == "numeric" else (l2_normalize(m) * w))
    raw = np.concatenate(parts, axis=1).astype(np.float32)
    mask = np.linalg.norm(raw, axis=1) > 0
    log.info("Built structured-metadata channel: %d movies, dim=%d", n, raw.shape[1])
    return l2_normalize(raw), mask
