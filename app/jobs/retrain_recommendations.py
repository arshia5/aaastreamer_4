"""CLI entrypoint for the nightly/manual full recommendation training.

    # initial / on-demand: retrain LightGCN AND the ranker (XGBoost-vs-linear A/B)
    python -m app.jobs.retrain_recommendations [--epochs N] [--max-users N]

    # nightly cron: retrain LightGCN + profiles + refresh, REUSE the frozen ranker
    python -m app.jobs.retrain_recommendations --skip-ranker

The ranker's features are invariant to LightGCN's nightly re-basis, so it only
needs (re)training when features/data drift — run the default form occasionally
(e.g. monthly) and --skip-ranker every night. Runs outside the web lifecycle.
"""
import argparse
import asyncio
import json
import logging

from app.jobs.training import run_full_recommendation_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-users", type=int, default=None,
                    help="Cap number of users whose recommendations are refreshed")
    ap.add_argument("--like-threshold", type=float, default=7.0)
    ap.add_argument("--job-id", type=int, default=None,
                    help="Update an existing recommendation_jobs row instead of creating one")
    ap.add_argument("--triggered-by", type=int, default=None)
    ap.add_argument("--skip-ranker", action="store_true",
                    help="Nightly mode: reuse the frozen active ranker instead of "
                         "retraining it (LightGCN + profiles + refresh still run)")
    args = ap.parse_args()

    kwargs = dict(
        like_threshold=args.like_threshold,
        max_refresh_users=args.max_users,
        job_id=args.job_id,
        triggered_by_user_id=args.triggered_by,
        retrain_ranker=not args.skip_ranker,
    )
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    result = asyncio.run(run_full_recommendation_training(**kwargs))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
