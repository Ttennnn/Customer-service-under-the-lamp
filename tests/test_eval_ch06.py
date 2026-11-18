"""ch06 多轮验收脚本的纯函数测试(不构建 LLM、不访问网络)。"""

from app.jobs.eval_ch06 import TURNS, evaluate_turn, summarize


def test_summarize_dialogue_pass_fail():
    results = [
        {
            "expected": "物流",
            "intent": "物流",
            "resolved_query": "订单 1001 物流",
            "must_include": ["1001"],
        },
        {
            "expected": "退款退货",
            "intent": "物流",
            "resolved_query": "它能退吗",
            "must_include": ["1001"],
        },
    ]
    out = summarize(results)
    assert out["passed"] == 1 and out["failed"] == 1


def test_summarize_checks_required_entities_and_banned_pronouns():
    results = [
        {
            "expected": "退款退货",
            "intent": "退款退货",
            "resolved_query": "订单 1001 能退吗",
            "must_include": ["1001"],
            "must_not_include": ["它"],
        },
        {
            "expected": "退款退货",
            "intent": "退款退货",
            "resolved_query": "订单 1001 能退吗",
            "must_include": ["1002"],
        },
        {
            "expected": "退款退货",
            "intent": "退款退货",
            "resolved_query": "它能退吗",
            "must_include": [],
            "must_not_include": ["它"],
        },
    ]
    out = summarize(results)
    assert out["passed"] == 1 and out["failed"] == 2


def test_summarize_counts_errors_and_fallbacks_as_failed():
    results = [
        {
            "expected": "物流",
            "intent": None,
            "resolved_query": "订单 1001 的物流到哪了",
            "must_include": ["1001"],
            "error": "APIConnectionError: boom",
        },
        {
            "expected": "物流",
            "intent": "物流",
            "resolved_query": "订单 1001 的物流到哪了",
            "must_include": ["1001"],
            "valid": False,
            "fallback_reason": "timeout",
        },
    ]
    out = summarize(results)
    assert out["passed"] == 0 and out["failed"] == 2


def test_dialogue_turns_match_acceptance_spec():
    assert [turn["message"] for turn in TURNS] == [
        "订单 1001 的物流到哪了",
        "它能退吗",
        "那它现在到哪了",
    ]
    assert [turn["expected"] for turn in TURNS] == ["物流", "退款退货", "物流"]
    assert "1001" in TURNS[0]["must_include"]
    assert "1001" in TURNS[1]["must_include"]
    assert "它" in TURNS[1]["must_not_include"]
    assert "1001" in TURNS[2]["must_include"]


def test_evaluate_turn_reports_each_failure_reason():
    verdict = evaluate_turn(
        {
            "expected": "退款退货",
            "intent": "物流",
            "resolved_query": "它能退吗",
            "must_include": ["1001"],
            "must_not_include": ["它"],
        }
    )
    assert verdict["passed"] is False
    assert "intent:" in verdict["reason"]
    assert "missing: '1001'" in verdict["reason"]
    assert "banned: '它'" in verdict["reason"]
