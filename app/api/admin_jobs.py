"""白名单作业运行 API(ch09 Task 10)。

``POST /api/admin/jobs/{name}/run``: 202 已启动、409 同一作业运行中、
404 未知作业。``GET .../status`` 返回页面轮询所需的运行状态。作业名之外
没有任何参数入口,命令片段永远进不了运行器。
"""

from fastapi import APIRouter, HTTPException

from app.services.admin_jobs import ALLOWED_JOBS, job_status, start_job

api_router = APIRouter(prefix="/api/admin/jobs")


@api_router.post("/{name}/run", status_code=202)
def run_job(name: str) -> dict:
    """Start one allow-listed job in the background."""
    if name not in ALLOWED_JOBS:
        raise HTTPException(status_code=404, detail="unknown job")
    state = start_job(name)
    if state is None:
        raise HTTPException(status_code=409, detail=f"job {name!r} is already running")
    return state


@api_router.get("/{name}/status")
def get_status(name: str) -> dict:
    """Return the latest state (running/timestamps/exit_code/output_tail)."""
    state = job_status(name)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return state
