"""Tests for the /rag-eval page and its run/status/report API."""

import json

from fastapi.testclient import TestClient

from app.main import app
from app.services import rag_eval


client = TestClient(app)


def test_rag_eval_page_is_served():
    response = client.get("/rag-eval")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "RAG 评估" in body
    assert "/rag-eval/api/report" in body
    assert "/rag-eval/api/run" in body
    assert "重跑评估" in body


def test_status_is_idle_by_default():
    response = client.get("/rag-eval/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["running"] is False
    assert "report_path" in payload


def test_report_returns_404_without_json(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_eval, "JSON_PATH", tmp_path / "missing.json")
    response = client.get("/rag-eval/api/report")
    assert response.status_code == 404


def test_report_reads_json_sidecar(monkeypatch, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"cases": 35, "overall": {"dense": {}}}), encoding="utf-8"
    )
    monkeypatch.setattr(rag_eval, "JSON_PATH", path)
    response = client.get("/rag-eval/api/report")
    assert response.status_code == 200
    assert response.json()["cases"] == 35


def test_report_markdown_returns_file(monkeypatch, tmp_path):
    path = tmp_path / "report.md"
    path.write_text("# report", encoding="utf-8")
    monkeypatch.setattr(rag_eval, "report_markdown_path", lambda: path)
    response = client.get("/rag-eval/api/report.md")
    assert response.status_code == 200
    assert response.text == "# report"


def test_run_starts_evaluation(monkeypatch):
    monkeypatch.setattr(
        rag_eval,
        "start_eval",
        lambda: {"running": True, "started_at": "2026-11-17T00:00:00"},
    )
    response = client.post("/rag-eval/api/run")
    assert response.status_code == 202
    assert response.json()["status"] == "running"


def test_run_returns_409_when_already_running(monkeypatch):
    monkeypatch.setattr(rag_eval, "start_eval", lambda: None)
    response = client.post("/rag-eval/api/run")
    assert response.status_code == 409
