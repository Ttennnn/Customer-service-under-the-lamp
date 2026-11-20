from fastapi.testclient import TestClient

from scripts.ch10.serve import create_app


class FakeClassifier:
    threshold = 0.55

    def classify(self, texts):
        assert texts == ["买大了想退"]
        return [
            {
                "labels": ["退换货", "尺码"],
                "scores": {"退换货": 0.8, "尺码": 0.9},
            }
        ]


def test_healthz_and_classify():
    client = TestClient(create_app(FakeClassifier()))
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["threshold"] == 0.55

    body = client.post("/classify", json={"texts": ["买大了想退"]}).json()
    assert body["items"][0]["labels"] == ["退换货", "尺码"]


def test_classify_rejects_empty_and_too_large():
    client = TestClient(create_app(FakeClassifier()))
    assert client.post("/classify", json={"texts": []}).status_code == 400
    assert (
        client.post("/classify", json={"texts": ["x"] * 65}).status_code
        == 400
    )
