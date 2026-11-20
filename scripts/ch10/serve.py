"""Lightweight FastAPI classifier service on :8110 (ONNX Runtime only)."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core import paths
from app.core.taxonomy import TOPIC_CLASSES

PID_PATH = paths.ROOT / "log" / "classifier.pid"
LOG_PATH = paths.ROOT / "log" / "classifier.log"
MAX_BATCH = 64


class ClassifyRequest(BaseModel):
    texts: list[str]


def create_app(classifier=None) -> FastAPI:
    app = FastAPI(title="ch10 topic classifier")

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "labels": len(TOPIC_CLASSES),
            "threshold": getattr(classifier, "threshold", None),
            "model": getattr(classifier, "model_name", "onnx"),
        }

    @app.post("/classify")
    def classify(request: ClassifyRequest) -> dict:
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts must not be empty")
        if len(request.texts) > MAX_BATCH:
            raise HTTPException(
                status_code=400,
                detail=f"texts must contain at most {MAX_BATCH} items",
            )
        if any(not text.strip() for text in request.texts):
            raise HTTPException(
                status_code=400, detail="texts must not contain blank items"
            )
        if classifier is None:
            raise HTTPException(
                status_code=503, detail="classifier is not loaded"
            )
        return {"items": classifier.classify(request.texts)}

    return app


def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=1.0
        ) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - any failure means not healthy
        return False


def _wait_health(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health_ok(port):
            return True
        time.sleep(0.5)
    return False


def _stop() -> int:
    if not PID_PATH.exists():
        print("classifier: no pid file; not running")
        return 0
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        PID_PATH.unlink(missing_ok=True)
        print("classifier: invalid pid file removed")
        return 1
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    PID_PATH.unlink(missing_ok=True)
    print("classifier: stopped")
    return 0


def _daemon(
    host: str,
    port: int,
    model: str,
    tokenizer: str,
    threshold: str,
) -> int:
    if _health_ok(port):
        print(f"classifier: already healthy on :{port}")
        return 0
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "scripts.ch10.serve",
        "--serve",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--threshold",
        threshold,
    ]
    kwargs: dict = {"cwd": str(paths.ROOT)}
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    with LOG_PATH.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            **kwargs,
        )
    PID_PATH.write_text(str(process.pid), encoding="utf-8")
    if _wait_health(port):
        print(f"classifier: healthy on :{port} (pid {process.pid})")
        return 0
    print(f"classifier: failed to start; see {LOG_PATH}")
    _stop()
    return 1


def _serve(
    host: str,
    port: int,
    model: str,
    tokenizer: str,
    threshold: str,
) -> int:
    import uvicorn

    from scripts.ch10.inference_lib import Classifier

    classifier = Classifier(model, tokenizer, threshold)
    app = create_app(classifier)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch10 classifier service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--model", default="data/ch10/onnx/model.onnx")
    parser.add_argument("--tokenizer", default="data/ch10/onnx")
    parser.add_argument(
        "--threshold", default="data/ch10/model/threshold.json"
    )
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args(argv)

    if args.stop:
        return _stop()
    if args.daemon:
        return _daemon(
            args.host, args.port, args.model, args.tokenizer, args.threshold
        )
    if args.serve:
        return _serve(
            args.host, args.port, args.model, args.tokenizer, args.threshold
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
