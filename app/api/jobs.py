"""Whitelisted job control API for the acceptance pages."""

from fastapi import APIRouter, HTTPException

from app.core.jobs import JOBS, job_status, start_job, stop_job

api_router = APIRouter(prefix="/api/jobs")


@api_router.post("/{name}/run", status_code=202)
def run_job(name: str) -> dict:
    if name not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job")
    state = start_job(name)
    if state is None:
        raise HTTPException(
            status_code=409, detail=f"job {name!r} is already running"
        )
    return state


@api_router.get("/{name}/status")
def get_status(name: str) -> dict:
    state = job_status(name)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return state


@api_router.post("/{name}/stop")
def stop(name: str) -> dict:
    if name not in JOBS:
        raise HTTPException(status_code=404, detail="unknown job")
    return {"stopped": stop_job(name)}
