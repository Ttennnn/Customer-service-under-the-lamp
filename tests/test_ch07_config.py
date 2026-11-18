import pytest
from app.config import Settings
from app.services.context_budget import compute_budget, validate_budget


def _base(**kw):
    return Settings(_env_file=None, openai_api_key="x", **kw)


def test_effective_aliases():
    s = _base(max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5)
    assert s.effective_max_agent_steps == 3
    assert s.effective_tool_result_max_tokens == 1200
    assert s.effective_rerank_top_k == 5
    s2 = _base()
    assert s2.effective_max_agent_steps == s2.agent_max_steps
    assert s2.effective_tool_result_max_tokens == s2.agent_tool_result_max_chars
    assert s2.effective_rerank_top_k == s2.rerank_top_n


def test_demo_budget_exact_numbers():
    s = _base(model_context_window=18000, max_output_tokens=2000, max_user_input_tokens=2000,
              max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5)
    b = compute_budget(s)
    assert (b.history, b.layer1, b.layer2) == (5650, 3955, 1695)
    assert b.layer1 + b.layer2 == b.history


def test_default_budget_keeps_twenty_turns_raw():
    b = compute_budget(_base(model_context_window=65536))
    assert b.history == 9000
    assert (b.layer1, b.layer2) == (6300, 2700)
    assert b.layer1 >= 20 * 300  # 20 轮稳态原文装得下


def test_window_only_goes_to_zero_and_warns(caplog):
    s = _base(model_context_window=18000)
    b = validate_budget(s)
    assert b.history == 0
    assert any("上下文预算不足" in rec.message for rec in caplog.records)
