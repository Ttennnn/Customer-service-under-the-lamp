"""Read-only acceptance dashboard API for ch10 (nine evidence gates)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core import paths

REPORT_DIR = paths.REPORT_DIR
DATA_DIR = paths.DATA_DIR
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
SERVICE_URL = "http://127.0.0.1:8110"

page_router = APIRouter()
api_router = APIRouter(prefix="/api/acceptance")

PAGES = {
    "/acceptance": "acceptance.html",
    "/acceptance/eval": "acceptance-eval.html",
    "/acceptance/data": "acceptance-data.html",
    "/acceptance/errors": "acceptance-errors.html",
}


class ClassifyRequest(BaseModel):
    texts: list[str]


def _read_json(name: str) -> dict | None:
    try:
        return json.loads(
            (Path(REPORT_DIR) / name).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None


def _service_health() -> bool:
    try:
        response = httpx.get(f"{SERVICE_URL}/healthz", timeout=2)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def _service_health_payload() -> dict | None:
    try:
        response = httpx.get(f"{SERVICE_URL}/healthz", timeout=2)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def _gate(
    gate_id: str,
    label: str,
    present: bool,
    passed: bool,
    job: str,
    detail: dict | str | None = None,
) -> dict:
    if not present:
        state = "missing"
    else:
        state = "pass" if passed else "fail"
    return {
        "id": gate_id,
        "label": label,
        "present": present,
        "state": state,
        "pass": state == "pass",
        "job": job,
        "log": f"log/acceptance/{job}.log",
        "detail": detail,
        "hint": None if present else f"make {job}",
    }


def _gates() -> list[dict]:
    try:
        manifest = json.loads(
            (Path(DATA_DIR) / "dataset" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        manifest = None
    golden = _read_json("golden_report.json")
    train = _read_json("train_report.json")
    model_config = Path(DATA_DIR) / "model" / "config.json"
    export = _read_json("export_report.json")
    eval_report = _read_json("eval_report.json")
    threshold = _read_json("threshold_scan.json")
    errors_md = Path(REPORT_DIR) / "error_samples.md"
    classify = _read_json("classify_run.json")

    corpus_ok = bool(
        manifest
        and all(value == 0 for value in (manifest.get("leak") or {}).values())
        and not manifest.get("coverage_gaps")
    )
    golden_ok = bool(
        golden and float(golden.get("pass_rate", 0)) >= 0.8
    )
    train_present = train is not None and model_config.is_file()
    export_ok = bool(export and export.get("ok") is True)
    per_class = (eval_report or {}).get("per_class") or []
    eval_present = len(per_class) == 17
    red = [
        row["label"]
        for row in per_class
        if row.get("pass") is False
    ]
    threshold_ok = bool(threshold and threshold.get("matches") is True)
    matrices = (eval_report or {}).get("confusion_matrices") or {}
    confusion_ok = len(matrices) == 17
    error_counts = (eval_report or {}).get("error_counts") or {}
    errors_ok = error_counts.get("reconciled") is True and errors_md.is_file()
    classify_ok = bool(
        classify
        and classify.get("ok") is True
        and (
            int(classify.get("classified", 0)) > 0
            or classify.get("skipped") is True
        )
    )

    return [
        _gate(
            "corpus",
            "语料与数据集",
            manifest is not None,
            corpus_ok,
            "ch10-dataset",
            manifest
            and {
                "counts": manifest.get("counts"),
                "leak": manifest.get("leak"),
                "coverage_gaps": manifest.get("coverage_gaps"),
            },
        ),
        _gate(
            "golden",
            "黄金样例闸",
            golden is not None,
            golden_ok,
            "ch10-golden",
            golden
            and {
                "pass_rate": golden.get("pass_rate"),
                "mismatches": len(golden.get("mismatches") or []),
            },
        ),
        _gate(
            "train",
            "训练产物",
            train_present,
            train_present,
            "ch10-train",
            train
            and {
                "train_samples": train.get("train_samples"),
                "val_samples": train.get("val_samples"),
                "early_stopped": train.get("early_stopped"),
            },
        ),
        _gate(
            "export",
            "ONNX 导出与服务",
            export is not None,
            export_ok,
            "ch10-export",
            export
            and {
                "ok": export.get("ok"),
                "label_match_rate": export.get("label_match_rate"),
                "threshold": export.get("threshold"),
            },
        ),
        _gate(
            "eval",
            "测试集评测",
            eval_present,
            eval_present,
            "ch10-eval",
            {
                "micro_f1": (eval_report or {})
                .get("overall", {})
                .get("micro_f1"),
                "red_lines": red,
                "note": "红线是诊断展示，不是交付硬闸门",
            }
            if eval_report
            else None,
        ),
        _gate(
            "threshold",
            "阈值扫描",
            threshold is not None,
            threshold_ok,
            "ch10-threshold-scan",
            threshold
            and {
                "expected": threshold.get("expected"),
                "best": threshold.get("best_threshold"),
                "matches": threshold.get("matches"),
            },
        ),
        _gate(
            "confusion",
            "混淆矩阵",
            eval_report is not None,
            confusion_ok,
            "ch10-eval",
            {"matrices": len(matrices)},
        ),
        _gate(
            "errors",
            "错例复核",
            eval_report is not None and errors_md.is_file(),
            errors_ok,
            "ch10-eval",
            error_counts,
        ),
        _gate(
            "classify",
            "旁路批量归类",
            classify is not None,
            classify_ok,
            "classify-pool",
            classify
            and {
                "pending": classify.get("pending"),
                "classified": classify.get("classified"),
                "skipped": classify.get("skipped"),
            },
        ),
    ]


def _page_for(request_path: str):
    filename = PAGES.get(request_path)
    if filename is None:
        raise HTTPException(status_code=404, detail="acceptance page is missing")
    path = Path(STATIC_DIR) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="acceptance page is missing")
    return FileResponse(
        path,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@page_router.get("/acceptance")
def acceptance_home():
    return _page_for("/acceptance")


@page_router.get("/acceptance/eval")
def acceptance_eval_page():
    return _page_for("/acceptance/eval")


@page_router.get("/acceptance/data")
def acceptance_data_page():
    return _page_for("/acceptance/data")


@page_router.get("/acceptance/errors")
def acceptance_errors_page():
    return _page_for("/acceptance/errors")


@api_router.get("/overview")
def overview() -> dict:
    gates = _gates()
    return {
        "gates": gates,
        "passed": sum(1 for gate in gates if gate["state"] == "pass"),
        "total": len(gates),
        "classifier_online": _service_health(),
    }


@api_router.get("/eval")
def eval_report() -> dict:
    data = _read_json("eval_report.json")
    if data is None:
        return {"present": False, "hint": "make ch10-eval"}
    return {
        **data,
        "present": True,
        "threshold_scan": _read_json("threshold_scan.json"),
    }


@api_router.get("/data")
def data_report() -> dict:
    manifest_path = Path(DATA_DIR) / "dataset" / "manifest.json"
    if not manifest_path.is_file():
        return {"present": False, "hint": "make ch10-dataset"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"present": False, "hint": "make ch10-dataset"}

    files = []
    for directory in (
        Path(DATA_DIR) / "model",
        Path(DATA_DIR) / "onnx",
        Path(REPORT_DIR),
    ):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file():
                stat = path.stat()
                files.append(
                    {
                        "path": str(path.relative_to(paths.ROOT)),
                        "size": stat.st_size,
                        "modified_at": stat.st_mtime,
                    }
                )
    return {
        "present": True,
        "manifest": manifest,
        "files": files,
    }


@api_router.get("/errors")
def error_report() -> dict:
    data = _read_json("eval_report.json")
    if data is None:
        return {"present": False, "hint": "make ch10-eval"}
    return {
        "present": True,
        "error_counts": data.get("error_counts") or {},
        "errors": data.get("errors") or [],
    }


@api_router.get("/service")
def service_report() -> dict:
    health = _service_health_payload()
    return {"online": health is not None, "health": health}


@api_router.post("/classify")
def classify_proxy(request: ClassifyRequest) -> dict:
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    if len(request.texts) > 64:
        raise HTTPException(
            status_code=400, detail="texts must contain at most 64 items"
        )
    if any(not text.strip() for text in request.texts):
        raise HTTPException(
            status_code=400, detail="texts must not contain blank items"
        )
    try:
        response = httpx.post(
            f"{SERVICE_URL}/classify",
            json={"texts": request.texts},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"classifier service unavailable: {exc}"
        ) from exc
    return response.json()
