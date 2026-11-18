import json
from pathlib import Path

from app.jobs.eval_expand import evaluate_case, summarize

CASES = Path(__file__).parent / "eval" / "ch06_expand_cases.json"


def test_evaluate_case_checks_three_and_entities():
    case = {"id": "E1", "must_include": ["1001"]}
    assert evaluate_case(case, ["订单 1001 退货政策", "退货条件", "退款流程"])["passed"]
    assert not evaluate_case(case, ["只有一条"])["passed"]
    assert not evaluate_case(case, ["退货条件", "退款流程", "其他"])["passed"]


def test_evaluate_case_rejects_duplicates_and_blank_queries():
    case = {"id": "E2", "must_include": ["1002"]}
    assert not evaluate_case(case, ["订单 1002 退货", "订单 1002 退货", "退款流程"])[
        "passed"
    ]
    assert not evaluate_case(case, ["订单 1002 退货", "   ", "退款流程"])["passed"]
    assert evaluate_case(case, ["订单 1002 退货", "退款条件", "退款流程"])["passed"]


def test_summarize_counts():
    out = summarize([{"passed": True}, {"passed": False}])
    assert out["total"] == 2 and out["passed"] == 1


def test_summarize_excludes_invalid_fallback_rows():
    out = summarize(
        [
            {"passed": True},
            {"passed": True},
            {"passed": False, "valid": False},
        ]
    )
    assert out["total"] == 2
    assert out["passed"] == 2
    assert out["accuracy"] == 1.0


def test_expand_cases_schema():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    assert len(cases) >= 6
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        assert isinstance(case["query"], str) and case["query"].strip()
        assert isinstance(case["order_data"], dict)
        assert isinstance(case["must_include"], list) and case["must_include"]
        for entity in case["must_include"]:
            assert isinstance(entity, str) and entity.strip()


def test_expand_cases_cover_required_scenarios():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    queries = " ".join(case["query"] for case in cases)
    assert "签收" in queries and "退货" in queries
    assert "质量" in queries and "退货" in queries
    assert "运输中" in queries and "取消" in queries
    assert "保修" in queries and "维修" in queries
