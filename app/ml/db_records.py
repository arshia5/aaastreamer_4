"""Reconstruct an embedding input record from a movie's relational metadata."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Country,
    Genre,
    Language,
    Movie,
    MovieCountry,
    MovieGenre,
    MovieLanguage,
    MoviePerson,
    Person,
    Role,
)

_ROLE_KEYS = {"director": "director", "writer": "writer", "actor": "actors"}


async def movie_record_from_db(db: AsyncSession, movie_id: int) -> dict | None:
    """Build the {plot, genre, director, writer, actors, language, country, year}
    record the embedding pipeline expects, from the DB."""
    movie = await db.get(Movie, movie_id)
    if movie is None:
        return None

    genres = (
        await db.execute(
            select(Genre.name)
            .join(MovieGenre, MovieGenre.genre_id == Genre.id)
            .where(MovieGenre.movie_id == movie_id)
        )
    ).scalars().all()
    languages = (
        await db.execute(
            select(Language.name)
            .join(MovieLanguage, MovieLanguage.language_id == Language.id)
            .where(MovieLanguage.movie_id == movie_id)
        )
    ).scalars().all()
    countries = (
        await db.execute(
            select(Country.name)
            .join(MovieCountry, MovieCountry.country_id == Country.id)
            .where(MovieCountry.movie_id == movie_id)
        )
    ).scalars().all()

    people_rows = (
        await db.execute(
            select(Role.name, Person.name)
            .join(MoviePerson, MoviePerson.role_id == Role.id)
            .join(Person, Person.id == MoviePerson.person_id)
            .where(MoviePerson.movie_id == movie_id)
        )
    ).all()

    by_role: dict[str, list[str]] = {"director": [], "writer": [], "actors": []}
    for role_name, person_name in people_rows:
        key = _ROLE_KEYS.get(role_name)
        if key:
            by_role[key].append(person_name)

    return {
        "title": movie.movie_title,
        "plot": movie.plot or "",
        "genre": list(genres),
        "director": by_role["director"],
        "writer": by_role["writer"],
        "actors": by_role["actors"],
        "language": list(languages),
        "country": list(countries),
        "year": movie.year,
    }
