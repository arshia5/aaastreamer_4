from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import Integer, cast, func, select, text

from app.deps import DB, CurrentAdmin, CurrentUser
from app.ml.evaluation import compute_metrics, fetch_eval_data
from app.models import (
    Country,
    Genre,
    Interaction,
    Language,
    Movie,
    MovieCountry,
    MovieGenre,
    MovieLanguage,
    User,
    UserMovieState,
    UserMovieStateType,
)
from app.ml import config
from app.schemas import (
    CountItem,
    GenreCount,
    GlobalStats,
    MovieStats,
    RatedMovie,
    RatingDistributionBucket,
    RecommendationEvaluation,
    TrendingMovie,
    UserStats,
)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/global", response_model=GlobalStats)
async def global_stats(db: DB):
    total_movies = await db.scalar(select(func.count()).select_from(Movie))
    total_users = await db.scalar(select(func.count()).select_from(User))
    active_users = await db.scalar(
        select(func.count()).select_from(User).where(User.is_active.is_(True))
    )
    total_interactions = await db.scalar(
        select(func.count()).select_from(Interaction)
    )
    total_ratings = await db.scalar(
        select(func.count()).where(Interaction.rating.is_not(None))
    )
    overall_avg = await db.scalar(select(func.avg(Interaction.rating)))
    total_reviews = await db.scalar(
        select(func.count()).where(Interaction.review_body.is_not(None))
    )
    return GlobalStats(
        total_movies=total_movies or 0,
        total_users=total_users or 0,
        active_users=active_users or 0,
        total_interactions=total_interactions or 0,
        total_ratings=total_ratings or 0,
        overall_average_rating=float(overall_avg) if overall_avg is not None else None,
        total_reviews=total_reviews or 0,
    )


@router.get("/users/{user_id}", response_model=UserStats)
async def user_stats(user_id: int, db: DB, current_user: CurrentUser):
    if user_id != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not allowed")
    if await db.get(User, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")

    row = (
        await db.execute(
            select(
                func.count(Interaction.id),
                func.count(Interaction.rating),
                func.avg(Interaction.rating),
                func.count(Interaction.review_body),
            ).where(Interaction.user_id == user_id)
        )
    ).one()
    interaction_count, rating_count, avg_rating, review_count = row

    watched = await db.scalar(
        select(func.count())
        .select_from(UserMovieState)
        .where(
            UserMovieState.user_id == user_id,
            UserMovieState.state == UserMovieStateType.watched,
        )
    )
    watchlist = await db.scalar(
        select(func.count())
        .select_from(UserMovieState)
        .where(
            UserMovieState.user_id == user_id,
            UserMovieState.state == UserMovieStateType.watchlist,
        )
    )
    return UserStats(
        user_id=user_id,
        interaction_count=interaction_count or 0,
        rating_count=rating_count or 0,
        average_rating_given=float(avg_rating) if avg_rating is not None else None,
        review_count=review_count or 0,
        watched_count=watched or 0,
        watchlist_count=watchlist or 0,
    )


@router.get("/me", response_model=UserStats)
async def my_stats(db: DB, current_user: CurrentUser):
    return await user_stats(current_user.id, db, current_user)


@router.get("/movies/top-rated", response_model=list[RatedMovie])
async def top_rated_movies(
    db: DB,
    limit: int = Query(default=10, ge=1, le=100),
    min_ratings: int = Query(default=1, ge=1, description="Minimum number of ratings"),
):
    """Highest-quality movies by a Bayesian-weighted score.

    Ranks by  (PRIOR*global_mean + sum(preference_score)) / (PRIOR + n)  rather
    than the naive average, so a handful of 10/10 reviews can't outrank a film
    with thousands of consistently-strong ones. `score` is that weighted value;
    `average_rating` is the honest raw mean shown alongside it.
    """
    sql = text(
        f"""
        WITH gm AS (SELECT avg(preference_score) AS m FROM interactions)
        SELECT m.id, m.movie_title,
               avg(i.rating) AS average_rating,
               count(i.preference_score) AS rating_count,
               ({config.POPULARITY_PRIOR} * (SELECT m FROM gm)
                + coalesce(sum(i.preference_score), 0))
               / ({config.POPULARITY_PRIOR} + count(i.preference_score)) AS score
        FROM movies m
        JOIN interactions i ON i.movie_id = m.id
        WHERE i.preference_score IS NOT NULL
        GROUP BY m.id, m.movie_title
        HAVING count(i.preference_score) >= :min_ratings
        ORDER BY score DESC
        LIMIT :limit
        """
    )
    rows = (await db.execute(sql, {"min_ratings": min_ratings, "limit": limit})).all()
    return [
        RatedMovie(
            id=r.id,
            movie_title=r.movie_title,
            average_rating=float(r.average_rating) if r.average_rating is not None else None,
            rating_count=r.rating_count,
            score=float(r.score) if r.score is not None else None,
        )
        for r in rows
    ]


@router.get("/movies/trending", response_model=list[TrendingMovie])
async def trending_movies(
    db: DB,
    limit: int = Query(default=10, ge=1, le=100),
    window_days: int = Query(default=30, ge=1, le=365,
                             description="Length of the recent window (days)"),
    min_recent: int = Query(default=3, ge=1,
                            description="Minimum reviews in the recent window to qualify"),
):
    """Movies gaining momentum (velocity), not just evergreen-popular ones.

    The window is anchored to the most recent review_date in the data (the dataset
    is historical, so wall-clock now() would match nothing). We compare the recent
    `window_days` against the equally-long window before it:

        trend_score = recent_count / (prior_count + k) * ln(1 + recent_count)

    The ratio rewards acceleration; the ln(volume) term keeps a 2-review blip from
    topping a genuine surge of hundreds.
    """
    k = 3  # smoothing: dampens divide-by-tiny-prior spikes
    sql = text(
        """
        WITH anchor AS (
            SELECT max(review_date) AS t FROM interactions WHERE review_date IS NOT NULL
        ),
        windows AS (
            SELECT i.movie_id,
                   count(*) FILTER (
                       WHERE i.review_date > a.t - make_interval(days => :n)) AS recent_count,
                   count(*) FILTER (
                       WHERE i.review_date <= a.t - make_interval(days => :n)) AS prior_count,
                   avg(i.preference_score) FILTER (
                       WHERE i.review_date > a.t - make_interval(days => :n)) AS recent_avg_pref
            FROM interactions i CROSS JOIN anchor a
            WHERE i.review_date IS NOT NULL
              AND i.review_date > a.t - make_interval(days => 2 * :n)
            GROUP BY i.movie_id
        )
        SELECT m.id, m.movie_title, w.recent_count, w.prior_count, w.recent_avg_pref,
               (w.recent_count::float / (w.prior_count + :k))
               * ln(1 + w.recent_count) AS trend_score
        FROM windows w
        JOIN movies m ON m.id = w.movie_id
        WHERE w.recent_count >= :min_recent
        ORDER BY trend_score DESC
        LIMIT :limit
        """
    )
    rows = (await db.execute(
        sql, {"n": window_days, "k": k, "min_recent": min_recent, "limit": limit})).all()
    return [
        TrendingMovie(
            id=r.id,
            movie_title=r.movie_title,
            recent_count=r.recent_count,
            prior_count=r.prior_count,
            recent_avg_preference=(
                float(r.recent_avg_pref) if r.recent_avg_pref is not None else None),
            trend_score=float(r.trend_score),
        )
        for r in rows
    ]


@router.get("/movies/most-rated", response_model=list[RatedMovie])
async def most_rated_movies(db: DB, limit: int = Query(default=10, ge=1, le=100)):
    avg_rating = func.avg(Interaction.rating).label("average_rating")
    rating_count = func.count(Interaction.rating).label("rating_count")
    stmt = (
        select(Movie.id, Movie.movie_title, avg_rating, rating_count)
        .join(Interaction, Interaction.movie_id == Movie.id)
        .group_by(Movie.id, Movie.movie_title)
        .order_by(rating_count.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [
        RatedMovie(
            id=mid,
            movie_title=title,
            average_rating=float(avg) if avg is not None else None,
            rating_count=cnt,
        )
        for mid, title, avg, cnt in rows
    ]


@router.get("/movies/most-watched", response_model=list[CountItem])
async def most_watched_movies(db: DB, limit: int = Query(default=10, ge=1, le=100)):
    cnt = func.count().label("count")
    stmt = (
        select(Movie.id, Movie.movie_title, cnt)
        .join(UserMovieState, UserMovieState.movie_id == Movie.id)
        .where(UserMovieState.state == UserMovieStateType.watched)
        .group_by(Movie.id, Movie.movie_title)
        .order_by(cnt.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [CountItem(id=mid, name=title, count=c) for mid, title, c in rows]


@router.get("/genres/distribution", response_model=list[GenreCount])
async def genre_distribution(db: DB):
    cnt = func.count(MovieGenre.movie_id).label("movie_count")
    stmt = (
        select(Genre.id, Genre.name, cnt)
        .join(MovieGenre, MovieGenre.genre_id == Genre.id)
        .group_by(Genre.id, Genre.name)
        .order_by(cnt.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [GenreCount(genre_id=gid, name=name, movie_count=c) for gid, name, c in rows]


@router.get("/countries/distribution", response_model=list[CountItem])
async def country_distribution(db: DB):
    cnt = func.count(MovieCountry.movie_id).label("count")
    stmt = (
        select(Country.id, Country.name, cnt)
        .join(MovieCountry, MovieCountry.country_id == Country.id)
        .group_by(Country.id, Country.name)
        .order_by(cnt.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [CountItem(id=cid, name=name, count=c) for cid, name, c in rows]


@router.get("/languages/distribution", response_model=list[CountItem])
async def language_distribution(db: DB):
    cnt = func.count(MovieLanguage.movie_id).label("count")
    stmt = (
        select(Language.id, Language.name, cnt)
        .join(MovieLanguage, MovieLanguage.language_id == Language.id)
        .group_by(Language.id, Language.name)
        .order_by(cnt.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [CountItem(id=lid, name=name, count=c) for lid, name, c in rows]


# Parametrised movie routes are declared last so literal paths above
# (e.g. /movies/top-rated) are matched first.
@router.get("/recommendations/evaluation", response_model=RecommendationEvaluation)
async def evaluate_recommendations(
    db: DB,
    _: CurrentAdmin,
    sample_users: int = Query(300, ge=10, le=3000),
    k: int = Query(10, ge=1, le=100),
    holdout: float = Query(0.2, gt=0.0, lt=1.0),
    like_threshold: float = Query(7.0, ge=0.0, le=10.0),
    min_interactions: int = Query(5, ge=2, le=200),
    seed: int = Query(42),
    refresh: bool = Query(False, description="Reload the movie-embedding cache"),
):
    """Offline leave-out evaluation of the embedding recommender.

    For a random sample of users, hides their most recent reviews, rebuilds the
    user vector from the earlier ones, and scores how well recommendations
    recover the held-out *liked* movies. Higher accuracy / lower loss is better.
    """
    data = await fetch_eval_data(db, sample_users, min_interactions, seed, refresh)
    metrics = await run_in_threadpool(
        compute_metrics, data, k, holdout, like_threshold
    )
    return RecommendationEvaluation(
        k=k,
        holdout_fraction=holdout,
        like_threshold=like_threshold,
        min_interactions=min_interactions,
        seed=seed,
        candidate_users=metrics["candidates"],
        users_evaluated_ranking=metrics["users_evaluated_ranking"],
        model=metrics["model"],
        popularity_baseline=metrics["popularity_baseline"],
        rating_prediction=metrics["rating_prediction"],
        notes=(
            f"Leave-out eval over {metrics['users_evaluated_ranking']} users with "
            f">=1 liked (preference>={like_threshold}) held-out movie. "
            "Ranking metrics @K compare the model vs a popularity baseline; "
            "rating_prediction RMSE/MAE are on the 0-10 preference scale."
        ),
    )


@router.get("/movies/{movie_id}", response_model=MovieStats)
async def movie_stats(movie_id: int, db: DB):
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")

    rating_row = (
        await db.execute(
            select(
                func.avg(Interaction.rating),
                func.count(Interaction.rating),
                func.avg(Interaction.sentiment),
                func.count(Interaction.review_body),
            ).where(Interaction.movie_id == movie_id)
        )
    ).one()
    avg_rating, rating_count, avg_sentiment, review_count = rating_row

    watched = await db.scalar(
        select(func.count())
        .select_from(UserMovieState)
        .where(
            UserMovieState.movie_id == movie_id,
            UserMovieState.state == UserMovieStateType.watched,
        )
    )
    watchlist = await db.scalar(
        select(func.count())
        .select_from(UserMovieState)
        .where(
            UserMovieState.movie_id == movie_id,
            UserMovieState.state == UserMovieStateType.watchlist,
        )
    )
    return MovieStats(
        movie_id=movie_id,
        average_rating=float(avg_rating) if avg_rating is not None else None,
        rating_count=rating_count or 0,
        review_count=review_count or 0,
        average_sentiment=float(avg_sentiment) if avg_sentiment is not None else None,
        watched_count=watched or 0,
        watchlist_count=watchlist or 0,
    )


@router.get("/movies/{movie_id}/rating-distribution",
            response_model=list[RatingDistributionBucket])
async def movie_rating_distribution(movie_id: int, db: DB):
    """Histogram of ratings rounded to integer buckets (0..10)."""
    if await db.get(Movie, movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    bucket = cast(func.round(Interaction.rating), Integer).label("bucket")
    stmt = (
        select(bucket, func.count().label("count"))
        .where(Interaction.movie_id == movie_id, Interaction.rating.is_not(None))
        .group_by(bucket)
        .order_by(bucket)
    )
    rows = (await db.execute(stmt)).all()
    return [RatingDistributionBucket(bucket=b, count=c) for b, c in rows]
