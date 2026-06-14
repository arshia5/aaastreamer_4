"""Load movies.csv into the relational schema.

Populates: movies, genres, languages, countries, people, roles, and the
movie_genres / movie_languages / movie_countries / movie_people join tables.

Idempotent: re-running upserts via ON CONFLICT DO NOTHING.

Usage:
    python -m scripts.load_movies [path/to/movies.csv]
"""
import asyncio
import re
import sys
import time

import asyncpg
import numpy as np
import pandas as pd

from app.core.config import settings
from app.ml import config as mlconfig
from app.ml.features import clean_tokens

RUNTIME_RE = re.compile(r"(\d+)")
ROLE_FIELDS = {"director": "director", "writer": "writer", "actor": "actors"}


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


def parse_runtime(value) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m = RUNTIME_RE.search(str(value))
    return int(m.group(1)) if m else None


def to_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


async def upsert_names(conn, table: str, names: set[str]) -> dict[str, int]:
    names = sorted(n for n in names if n)
    if names:
        await conn.execute(
            f"INSERT INTO {table}(name) SELECT unnest($1::text[]) "
            f"ON CONFLICT (name) DO NOTHING",
            names,
        )
    rows = await conn.fetch(f"SELECT id, name FROM {table}")
    return {r["name"]: r["id"] for r in rows}


async def main(csv_path: str) -> None:
    t0 = time.time()
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} movies")

    conn = await asyncpg.connect(dsn())
    try:
        # --- reference entities ---------------------------------------- #
        genre_set, lang_set, country_set, person_set = set(), set(), set(), set()
        for _, row in df.iterrows():
            genre_set.update(clean_tokens(row.get("genre")))
            lang_set.update(clean_tokens(row.get("language")))
            country_set.update(clean_tokens(row.get("country")))
            for field in ROLE_FIELDS.values():
                person_set.update(clean_tokens(row.get(field)))

        genre_map = await upsert_names(conn, "genres", genre_set)
        lang_map = await upsert_names(conn, "languages", lang_set)
        country_map = await upsert_names(conn, "countries", country_set)
        person_map = await upsert_names(conn, "people", person_set)
        role_map = await upsert_names(conn, "roles", set(ROLE_FIELDS.keys()))
        print(
            f"  genres={len(genre_map)} languages={len(lang_map)} "
            f"countries={len(country_map)} people={len(person_map)}"
        )

        # --- movies ----------------------------------------------------- #
        imdb_ids = df["imdbid"].astype(str).tolist()
        titles = df["title"].astype(str).tolist()
        years = [to_int(v) for v in df["year"].tolist()]
        durations = [parse_runtime(v) for v in df["runtime"].tolist()]
        plots = [None if pd.isna(v) else str(v) for v in df["plot"].tolist()]
        posters = [None if pd.isna(v) else str(v) for v in df["poster_url"].tolist()]

        await conn.execute(
            "INSERT INTO movies(imdb_id, movie_title, year, duration, plot, poster_url) "
            "SELECT * FROM unnest("
            "$1::text[],$2::text[],$3::int[],$4::int[],$5::text[],$6::text[]) "
            "ON CONFLICT (imdb_id) DO NOTHING",
            imdb_ids, titles, years, durations, plots, posters,
        )
        rows = await conn.fetch("SELECT id, imdb_id FROM movies")
        movie_map = {r["imdb_id"]: r["id"] for r in rows}
        print(f"  movies in db: {len(movie_map)}")

        # --- join tables ------------------------------------------------ #
        mg, ml, mc = set(), set(), set()
        mp = set()
        for _, row in df.iterrows():
            mid = movie_map.get(str(row["imdbid"]))
            if mid is None:
                continue
            for g in clean_tokens(row.get("genre")):
                if g in genre_map:
                    mg.add((mid, genre_map[g]))
            for lang in clean_tokens(row.get("language")):
                if lang in lang_map:
                    ml.add((mid, lang_map[lang]))
            for c in clean_tokens(row.get("country")):
                if c in country_map:
                    mc.add((mid, country_map[c]))
            for role, field in ROLE_FIELDS.items():
                rid = role_map[role]
                for person in clean_tokens(row.get(field)):
                    pid = person_map.get(person)
                    if pid is not None:
                        mp.add((mid, pid, rid))

        await _insert_pairs(conn, "movie_genres", "genre_id", mg)
        await _insert_pairs(conn, "movie_languages", "language_id", ml)
        await _insert_pairs(conn, "movie_countries", "country_id", mc)
        await _insert_people(conn, mp)
        print(
            f"  links: genres={len(mg)} languages={len(ml)} "
            f"countries={len(mc)} people={len(mp)}"
        )
    finally:
        await conn.close()
    print(f"Done in {time.time() - t0:.1f}s")


async def _insert_pairs(conn, table: str, fk: str, pairs: set):
    if not pairs:
        return
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    await conn.execute(
        f"INSERT INTO {table}(movie_id, {fk}) "
        f"SELECT * FROM unnest($1::int[],$2::int[]) ON CONFLICT DO NOTHING",
        a, b,
    )


async def _insert_people(conn, triples: set):
    if not triples:
        return
    a = [t[0] for t in triples]
    b = [t[1] for t in triples]
    c = [t[2] for t in triples]
    await conn.execute(
        "INSERT INTO movie_people(movie_id, person_id, role_id) "
        "SELECT * FROM unnest($1::int[],$2::int[],$3::int[]) "
        "ON CONFLICT DO NOTHING",
        a, b, c,
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(mlconfig.DEFAULT_MOVIES_CSV)
    asyncio.run(main(path))
