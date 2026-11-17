import json
from pathlib import Path

from app.agent.routing import INTENT_ROUTES

CASES = Path(__file__).parent / "eval" / "ch05_intent_cases.json"


def test_intent_cases_cover_all_classes():
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(rows) == 42
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    by_intent = {}
    for row in rows:
        assert row["text"].strip()
        assert row["intent"] in INTENT_ROUTES
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
    assert by_intent == {intent: 6 for intent in INTENT_ROUTES}


def test_compute_metrics_counts_confusion():
    from app.jobs.eval_intent import compute_metrics

    out = compute_metrics(["物流", "物流", "投诉"], ["物流", "订单", "投诉"])
    assert out["total"] == 3
    assert out["correct"] == 2
    assert out["per_class"]["物流"] == {"total": 2, "correct": 1}
    assert out["confusion"]["物流"]["订单"] == 1
