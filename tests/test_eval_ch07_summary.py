"""Pure-function tests for the ch07 summary prompt evaluation."""

import json
from pathlib import Path

from app.jobs.eval_ch07_summary import evaluate_case, summarize

CASES = Path(__file__).parent / "eval" / "ch07_summary_cases.json"


def test_evaluate_case_contains_and_banned():
    case = {"id": "c1", "contains": ["1001", "退款"], "not_contains": ["你好"]}
    assert evaluate_case(case, "订单 1001 要求退款")["passed"] is True
    assert evaluate_case(case, "你好，订单 1001 要求退款")["passed"] is False


def test_summarize_counts_only_valid():
    results = [
        {"valid": True, "passed": True},
        {"valid": True, "passed": False},
        {"valid": False, "passed": False},
    ]
    out = summarize(results)
    assert out["total"] == 2 and out["passed"] == 1 and out["accuracy"] == 0.5


def test_evaluate_case_max_chars_and_regex():
    case = {"id": "c2", "max_chars": 6, "not_contains_regex": r"\d{4,}"}
    assert evaluate_case(case, "没有编号")["passed"] is True

    too_long = evaluate_case(case, "没有编号，这是一段超长摘要")
    assert too_long["passed"] is False
    assert "too long" in too_long["reason"]

    fabricated = evaluate_case(case, "订单 1234 已处理")
    assert fabricated["passed"] is False
    assert "1234" in fabricated["reason"]


def test_evaluate_case_handles_missing_summary():
    result = evaluate_case({"id": "c3", "contains": ["退款"]}, None)
    assert result["passed"] is False
    assert "missing" in result["reason"]


REQUIRED_IDS = {
    "order-refund",
    "product-warranty",
    "chitchat-only",
    "order-corrected",
    "no-fabrication",
    "refund-rejected",
}


def test_cases_schema_and_required_scenarios():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(cases) >= 6
    by_id = {case["id"]: case for case in cases}
    assert REQUIRED_IDS <= set(by_id)
    assert len(by_id) == len(cases)

    order_refund = by_id["order-refund"]
    assert {"1001", "13800000000", "退款"} <= set(order_refund["contains"])
    assert {"你好", "谢谢"} <= set(order_refund["not_contains"])

    warranty = by_id["product-warranty"]
    assert {"LD-200", "保修", "灯不亮"} <= set(warranty["contains"])
    assert "退款" in warranty["not_contains"]

    chitchat = by_id["chitchat-only"]
    assert chitchat["contains"] == []
    assert {"订单", "退款", "手机"} <= set(chitchat["not_contains"])
    assert chitchat["max_chars"] == 60

    corrected = by_id["order-corrected"]
    assert corrected["contains"] == ["1002"]
    assert corrected["not_contains"] == ["1001"]

    assert by_id["no-fabrication"]["not_contains_regex"] == r"\d{4,}"

    rejected = by_id["refund-rejected"]
    assert {"拒绝", "超过七天"} <= set(rejected["contains"])

    for case in cases:
        assert isinstance(case["history"], list)
        assert "prior_summary" in case
        for row in case["history"]:
            assert row["role"] in {"user", "assistant", "system"}
            assert isinstance(row["content"], str) and row["content"].strip()
