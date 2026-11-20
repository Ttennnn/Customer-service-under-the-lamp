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


def test_wrong_shape_report_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance, "REPORT_DIR", tmp_path)
    (tmp_path / "eval_report.json").write_text("[]", encoding="utf-8")
    body = TestClient(app).get("/api/acceptance/eval").json()
    assert body["present"] is False


def test_corpus_gate_requires_augment_leak(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(acceptance, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(acceptance, "DATA_DIR", tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"train": 1, "val": 1, "test": 1},
                "leak": {"train_val": 0, "train_test": 0, "val_test": 0},
                "coverage_gaps": [],
            }
        ),
        encoding="utf-8",
    )
    body = TestClient(app).get("/api/acceptance/overview").json()
    corpus = next(g for g in body["gates"] if g["id"] == "corpus")
    assert corpus["present"] is True
    assert corpus["state"] == "fail"


def test_acceptance_assets_are_served():
    client = TestClient(app)
    assert client.get("/static/acceptance.css").status_code == 200
    assert client.get("/static/acceptance.js").status_code == 200
