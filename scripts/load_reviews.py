"""Load the reviews dataset into users + interactions.

- Creates a user per unique reviewer (email = <username>@test.com, password 1234,
  hashed once and reused for speed).
- Bulk-loads interactions: rating + sentiment (already in the data, clamped to
  0..10) and preference_score = 0.7*rating + 0.3*sentiment.
- Dedupes to one interaction per (user, movie), keeping the most recent review.

Usage:
    python -m scripts.load_reviews [path/to/reviews_dir]
"""
import asyncio
import glob
import os
import sys
import time

import asyncpg
import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.security import hash_password
from app.ml import config as mlconfig

CHUNK = 100_000
COLS = [
    "user_id", "movie_id", "rating", "sentiment", "preference_score",
    "review_title", "review_body", "review_date",
]


def dsn() -> str:
    return settings.database_url.replace("+asyncpg", "")


def _none(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v


async def main(reviews_dir: str) -> None:
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(reviews_dir, "*.csv")))
    print(f"{len(files)} review files in {reviews_dir}")

    conn = await asyncpg.connect(dsn())
    try:
        # --- 1. unique reviewers -> users ------------------------------- #
        print("Collecting usernames ...")
        usernames: set[str] = set()
        for f in files:
            for ch in pd.read_csv(f, usecols=["username"], chunksize=500_000):
                usernames.update(ch["username"].dropna().astype(str).tolist())
        usernames.discard("")
        names = sorted(usernames)
        print(f"  {len(names):,} unique reviewers")

        pw_hash = hash_password(mlconfig.IMPORT_USER_PASSWORD)  # hash '1234' once
        domain = mlconfig.IMPORT_EMAIL_DOMAIN
        await conn.execute(
            "INSERT INTO users(username, email, password_hash) "
            "SELECT u, u || $2, $3 FROM unnest($1::text[]) AS u "
            "ON CONFLICT (username) DO NOTHING",
            names, f"@{domain}", pw_hash,
        )
        user_map = {r["username"]: r["id"]
                    for r in await conn.fetch("SELECT id, username FROM users")}
        movie_map = {r["imdb_id"]: r["id"]
                     for r in await conn.fetch("SELECT id, imdb_id FROM movies")}
        print(f"  users in db: {len(user_map):,} | movies: {len(movie_map):,}")

        # --- 2. stage all review rows ----------------------------------- #
        await conn.execute("DROP TABLE IF EXISTS _stg_reviews")
        await conn.execute(
            "CREATE UNLOGGED TABLE _stg_reviews ("
            "user_id int, movie_id int, rating double precision, "
            "sentiment double precision, preference_score double precision, "
            "review_title text, review_body text, review_date timestamp)"
        )
        rw, sw = mlconfig.PREF_RATING_WEIGHT, mlconfig.PREF_SENTIMENT_WEIGHT
        staged = 0
        print("Staging review rows ...")
        for f in files:
            for ch in pd.read_csv(
                f,
                usecols=["username", "imdb_id", "rating", "sentiment",
                         "review_date", "review_summery", "review_detail"],
                chunksize=CHUNK,
            ):
                uid = ch["username"].astype(str).map(user_map)
                mid = ch["imdb_id"].astype(str).map(movie_map)
                rating = pd.to_numeric(ch["rating"], errors="coerce").clip(0, 10)
                sent = pd.to_numeric(ch["sentiment"], errors="coerce").clip(0, 10)
                pref = rw * rating + sw * sent
                pref = pref.fillna(rating).fillna(sent)
                date = pd.to_datetime(
                    ch["review_date"], format="%d %B %Y", errors="coerce"
                )
                records = []
                for u, m, r, s, p, ti, bo, dt in zip(
                    uid, mid, rating, sent, pref,
                    ch["review_summery"], ch["review_detail"], date,
                ):
                    if pd.isna(u) or pd.isna(m):
                        continue
                    records.append((
                        int(u), int(m), _none(r), _none(s), _none(p),
                        _none(ti), _none(bo),
                        None if pd.isna(dt) else dt.to_pydatetime(),
                    ))
                if records:
                    await conn.copy_records_to_table(
                        "_stg_reviews", records=records, columns=COLS
                    )
                    staged += len(records)
            print(f"  staged {staged:,} (through {os.path.basename(f)})")

        # --- 3. dedupe -> interactions ---------------------------------- #
        print("Inserting deduped interactions ...")
        inserted = await conn.execute(
            "INSERT INTO interactions"
            "(user_id, movie_id, rating, sentiment, preference_score,"
            " review_title, review_body, review_date) "
            "SELECT DISTINCT ON (user_id, movie_id) user_id, movie_id, rating, "
            "sentiment, preference_score, review_title, review_body, review_date "
            "FROM _stg_reviews "
            "ORDER BY user_id, movie_id, review_date DESC NULLS LAST "
            "ON CONFLICT (user_id, movie_id) DO NOTHING"
        )
        await conn.execute("DROP TABLE _stg_reviews")
        total = await conn.fetchval("SELECT count(*) FROM interactions")
        print(f"  staged={staged:,} -> interactions now {total:,} ({inserted})")
    finally:
        await conn.close()
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(mlconfig.DEFAULT_REVIEWS_DIR)
    asyncio.run(main(path))
