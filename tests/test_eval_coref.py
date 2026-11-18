import json
from pathlib import Path

from app.jobs.eval_coref import evaluate_case, summarize

CASES = Path(__file__).parent / "eval" / "ch06_coref_cases.json"


def test_evaluate_case_contains_and_passthrough():
    assert evaluate_case({"id": "C1", "contains": ["1001"], "not_contains": ["它"]}, "订单 1001 能退吗")["passed"]
    assert not evaluate_case({"id": "C2", "contains": ["1001"]}, "能退吗")["passed"]
    assert evaluate_case({"id": "C3", "passthrough": True, "message": "运费怎么算"}, "运费怎么算")["passed"]
    assert not evaluate_case({"id": "C4", "passthrough": True, "message": "运费怎么算"}, "运费如何计算")["passed"]


def test_summarize_counts():
    out = summarize([{"passed": True}, {"passed": False}])
    assert out["total"] == 2 and out["passed"] == 1 and out["accuracy"] == 0.5


def test_coref_cases_schema():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(cases) >= 10
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert isinstance(case["message"], str) and case["message"].strip()
        assert isinstance(case["history"], list)
        for row in case["history"]:
            assert row["role"] in {"user", "assistant", "system"}
            assert isinstance(row["content"], str) and row["content"].strip()
        if case.get("passthrough"):
            assert not case.get("contains")
            assert not case.get("not_contains")
        else:
            assert case.get("contains") or case.get("not_contains")


def test_coref_cases_cover_required_scenarios():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    by_message = {case["message"]: case for case in cases}

    pronoun = by_message["它能退吗"]
    assert "1001" in pronoun["contains"]
    assert "它" in pronoun["not_contains"]

    product = by_message["这个还有别的颜色吗"]
    assert "XM5" in product["contains"]

    assert by_message["订单 1001 什么时候发货"]["passthrough"] is True

    colloquial = by_message["邮费怎么算"]
    assert colloquial["history"] == []
    assert "运费" in colloquial["contains"]

    selector = [case for case in cases if "这个能退吗" == case["message"]]
    assert selector, "missing order-selector pronoun case"
    assert any(
        row["role"] == "system" and "[订单选择器]" in row["content"]
        for row in selector[0]["history"]
    )
    assert "1001" in selector[0]["contains"]
