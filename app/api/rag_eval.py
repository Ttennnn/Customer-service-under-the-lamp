"""Read-only RAG evaluation page plus its run/status/report API."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.services import rag_eval

PAGE = Path(__file__).resolve().parents[1] / "static" / "rag_eval.html"

page_router = APIRouter()
api_router = APIRouter(prefix="/rag-eval/api")


@page_router.get("/rag-eval")
def rag_eval_page():
    if not PAGE.is_file():
        raise HTTPException(status_code=500, detail="rag eval page is missing")
    return FileResponse(
        PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@api_router.get("/status")
def status():
    return rag_eval.status()


@api_router.get("/report")
def report():
    payload = rag_eval.report()
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={"detail": "no evaluation report yet"},
        )
    return payload


@api_router.get("/report.md")
def report_markdown():
    path = rag_eval.report_markdown_path()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no evaluation report yet")
    return FileResponse(
        path, media_type="text/markdown; charset=utf-8", filename=path.name
    )


@api_router.post("/run", status_code=202)
def run():
    started = rag_eval.start_eval()
    if started is None:
        raise HTTPException(
            status_code=409, detail="an evaluation is already running"
        )
    return {"status": "running", "started_at": started["started_at"]}
