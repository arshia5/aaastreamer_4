"""Admin endpoints for the recommendation training jobs and model versions."""
import logging
import subprocess
import sys

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.deps import DB, CurrentAdmin
from app.models import ModelVersion, RecommendationJob
from app.schemas import (
    ModelVersionRead,
    RecommendationJobRead,
    RetrainResponse,
)

log = logging.getLogger("recsys.admin")
router = APIRouter(prefix="/admin/recommendations", tags=["admin-recommendations"])


@router.post("/retrain", response_model=RetrainResponse, status_code=202)
async def trigger_retrain(
    db: DB,
    admin: CurrentAdmin,
    epochs: int | None = Query(default=None, ge=1, le=100),
    max_users: int | None = Query(
        default=None, ge=1, description="Cap users whose recs are refreshed"
    ),
):
    """Queue a full retrain (popularity + user embeddings + collaborative MF +
    XGBoost ranker + eval/rollback + recommendation refresh).

    Runs as a detached subprocess so the heavy torch/xgboost work never touches
    the web server process. Returns immediately with the job id to poll."""
    job = RecommendationJob(
        job_type="full_training", status="queued", triggered_by_user_id=admin.id
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    cmd = [sys.executable, "-m", "app.jobs.retrain_recommendations",
           "--job-id", str(job.id), "--triggered-by", str(admin.id)]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if max_users is not None:
        cmd += ["--max-users", str(max_users)]
    try:
        subprocess.Popen(cmd, start_new_session=True)
    except Exception as exc:
        job.status = "failed"
        job.error_message = f"Failed to launch training subprocess: {exc}"
        await db.commit()
        raise HTTPException(status_code=500, detail=str(exc))

    return RetrainResponse(
        job_id=job.id, status="queued",
        detail="Full retrain launched; poll /admin/recommendations/jobs/{id}",
    )


@router.get("/jobs", response_model=list[RecommendationJobRead])
async def list_jobs(
    db: DB, _: CurrentAdmin, limit: int = Query(default=20, ge=1, le=200)
):
    rows = await db.execute(
        select(RecommendationJob).order_by(RecommendationJob.id.desc()).limit(limit)
    )
    return rows.scalars().all()


@router.get("/jobs/{job_id}", response_model=RecommendationJobRead)
async def get_job(job_id: int, db: DB, _: CurrentAdmin):
    job = await db.get(RecommendationJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/models", response_model=list[ModelVersionRead])
async def list_models(
    db: DB, _: CurrentAdmin, model_type: str | None = Query(default=None)
):
    stmt = select(ModelVersion).order_by(ModelVersion.id.desc())
    if model_type:
        stmt = stmt.where(ModelVersion.model_type == model_type)
    return (await db.execute(stmt)).scalars().all()
