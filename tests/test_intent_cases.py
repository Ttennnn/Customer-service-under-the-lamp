import json
import re
from pathlib import Path

from app.agent.intent import INTENT_SYSTEM_PROMPT
from app.agent.routing import INTENT_ROUTES

CASES = Path(__file__).parent / "eval" / "ch06_intent_cases.json"

EXPECTED_CLASSES = {
    "物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "人工", "闲聊", "其他"
}

#: 第九类「人工」本轮只从标注样例新增两条,其余类别沿用每类至少 6 条。
MIN_CASES_PER_CLASS = {label: 6 for label in EXPECTED_CLASSES}
MIN_CASES_PER_CLASS["人工"] = 2

PROMPT_EXAMPLES = re.findall(r"「([^」]+)」", INTENT_SYSTEM_PROMPT)


def test_intent_cases_cover_all_classes():
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(rows) >= 56
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    by_intent = {}
    for row in rows:
        assert row["text"].strip()
        assert row["intent"] in INTENT_ROUTES
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
    assert set(by_intent) == EXPECTED_CLASSES
    for label, count in by_intent.items():
        assert count >= MIN_CASES_PER_CLASS[label]


def test_intent_cases_do_not_reuse_prompt_examples():
    """Held-out validation: no case may contain a prompt few-shot example."""
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    assert PROMPT_EXAMPLES, "prompt examples not found; update the extraction"
    for row in rows:
        for example in PROMPT_EXAMPLES:
            assert example not in row["text"], (
                f"{row['id']} reuses prompt example {example!r}: {row['text']!r}"
            )


def test_intent_cases_cover_boundaries_and_history_schema():
    rows = json.loads(CASES.read_text(encoding="utf-8"))
    by_text = {row["text"]: row["intent"] for row in rows}

    assert by_text["上个月买的电饭煲还能不能退"] == "退款退货"
    assert by_text["拆开就有故障，保修要走什么流程"] == "售后"
    assert by_text["我要投诉你们，钱也一起退了吧"] == "投诉"
    assert by_text["嗨，有人吗"] == "闲聊"
    assert by_text["能给我编一首诗吗"] == "其他"

    drift = [row for row in rows if row.get("history")]
    assert len(drift) >= 6
    for row in drift:
        assert isinstance(row["history"], list) and row["history"]
        for entry in row["history"]:
            assert entry["role"] in {"user", "assistant", "system"}
            assert isinstance(entry["content"], str) and entry["content"].strip()


def test_compute_metrics_counts_confusion():
    from app.jobs.eval_intent import compute_metrics

    out = compute_metrics(["物流", "物流", "投诉"], ["物流", "订单", "投诉"])
    assert out["total"] == 3
    assert out["correct"] == 2
    assert out["per_class"]["物流"] == {"total": 2, "correct": 1}
    assert out["confusion"]["物流"]["订单"] == 1


def test_summarize_excludes_errors_and_scores_real_other():
    from app.jobs.eval_intent import summarize

    results = [
        {
            "id": "1",
            "text": "x",
            "expected": "闲聊",
            "predicted": "其他",
            "confidence": 0.9,
            "error": "",
            "correct": False,
        },
        {
            "id": "2",
            "text": "y",
            "expected": "其他",
            "predicted": "其他",
            "confidence": 0.9,
            "error": "",
            "correct": True,
        },
        {
            "id": "3",
            "text": "z",
            "expected": "物流",
            "predicted": None,
            "confidence": None,
            "error": "ValueError: boom",
            "correct": False,
        },
    ]
    metrics, wrong, errors = summarize(results)
    assert metrics["total"] == 2
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == 0.5
    assert [case["id"] for case in wrong] == ["1"]
    assert [case["id"] for case in errors] == ["3"]
