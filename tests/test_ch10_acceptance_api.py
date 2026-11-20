from fastapi.testclient import TestClient

from app.api import acceptance
from app.main import app


def test_missing_artifacts_are_present_false_not_500(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "REPORT_DIR", tmp_path)
    body = TestClient(app).get("/api/acceptance/eval").json()
    assert body["present"] is False
    assert "ch10-eval" in body["hint"]


def test_overview_counts_nine_gates_and_service_offline(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(acceptance, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(acceptance, "_service_health", lambda: False)
    body = TestClient(app).get("/api/acceptance/overview").json()
    assert len(body["gates"]) == 9
    assert body["total"] == 9
    assert body["classifier_online"] is False
    assert any(
        gate["present"] is False and gate["state"] == "missing"
        for gate in body["gates"]
    )


def test_invalid_topic_label_is_rejected_by_classify_proxy(monkeypatch):
    monkeypatch.setattr(acceptance, "_service_health", lambda: True)
    response = TestClient(app).post(
        "/api/acceptance/classify", json={"texts": []}
    )
    assert response.status_code == 400
