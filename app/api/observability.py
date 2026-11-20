"""观测总览 API 与 ``/observability`` 页面路由(ch09 Task 10)。

三块数据(意图成本账 / 评估趋势 / 置信度校准)各自由独立 try/except 组装,
任一产物缺失或读取失败只把本块标为 ``present=false`` 并附 hint/error,
其余块照常返回。API 不重算任何指标:占比、Youden J、推荐阈值都来自脚本
写好的产物。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import session as db_session
from app.models import EvalRun

logger = logging.getLogger(__name__)

PAGE = Path(__file__).resolve().parents[1] / "static" / "observability.html"

COST_REPORT = "cost_by_intent.json"
CALIBRATION_REPORT = "confidence_calibration.json"
EVAL_RUN_LIMIT = 10

page_router = APIRouter()
api_router = APIRouter(prefix="/api/observability")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _report_path(settings: Settings, filename: str) -> Path:
    return Path(settings.ch09_report_dir) / filename


def _cost_block(settings: Settings) -> dict:
    """Read the cost report written by ``app.jobs.cost_by_intent``."""
    path = _report_path(settings, COST_REPORT)
    if not path.is_file():
        return {
            "present": False,
            "hint": "还没有成本报告:先聊几句,再点“重跑 意图成本账”。",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("cost report must be a JSON object")
    except Exception as exc:  # noqa: BLE001 - one block must not break others
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"}
    return {**data, "present": True}


def _eval_block() -> dict:
    """Read the latest ``EvalRun`` rows in chronological order for trends."""
    try:
        session: Session = db_session.get_session_factory()()
    except Exception as exc:  # noqa: BLE001 - DB failure is isolated to this block
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        rows = (
            session.query(EvalRun)
            .order_by(EvalRun.id.desc())
            .limit(EVAL_RUN_LIMIT)
            .all()
        )
        runs = [
            {
                "id": row.id,
                "triggered_by": row.triggered_by,
                "dataset_size": row.dataset_size,
                "metrics": row.metrics or {},
                "created_at": _iso(row.created_at),
            }
            for row in reversed(rows)
        ]
    except Exception as exc:  # noqa: BLE001 - querying may fail independently
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        session.close()

    if not runs:
        return {
            "present": False,
            "runs": [],
            "hint": "还没有评估记录:先跑一轮评估(make eval-flywheel)。",
        }
    return {"present": True, "runs": runs}


def _calibration_block(settings: Settings) -> dict:
    """Read the calibration report and compare it with the live threshold."""
    path = _report_path(settings, CALIBRATION_REPORT)
    if not path.is_file():
        return {
            "present": False,
            "hint": "还没有校准报告:先扫一遍(make calibrate-confidence)。",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("calibration report must be a JSON object")
        recommended = data.get("recommended") or {}
        recommended_threshold = float(recommended["threshold"])
    except Exception as exc:  # noqa: BLE001 - isolate a broken report
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"}

    current_threshold = float(settings.evidence_confidence_threshold)
    return {
        "present": True,
        "generated_at": data.get("generated_at"),
        "recommended": recommended,
        "current_threshold": current_threshold,
        "needs_update": recommended_threshold != current_threshold,
    }


def _safe(block, *args) -> dict:
    """Last-resort guard so an unexpected block bug cannot sink the page."""
    try:
        return block(*args)
    except Exception as exc:  # noqa: BLE001
        logger.exception("observability block failed")
        return {"present": False, "error": f"{type(exc).__name__}: {exc}"}


@page_router.get("/observability", response_class=FileResponse)
def observability_page() -> FileResponse:
    """Serve the observability page when the (Task 11) HTML file exists."""
    if not PAGE.is_file():
        raise HTTPException(status_code=404, detail="observability page is missing")
    return FileResponse(
        PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@api_router.get("/overview")
def overview(settings: Settings = Depends(get_settings)) -> dict:
    """Return the three independent observability blocks."""
    return {
        "cost": _safe(_cost_block, settings),
        "eval": _safe(_eval_block),
        "calibration": _safe(_calibration_block, settings),
    }
