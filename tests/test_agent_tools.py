from app.agent.tools import (
    AGENT_TOOLS,
    suggested_actions_from_handoff,
)


def test_agent_tools_exclude_faq_and_create_ticket():
    assert [t.name for t in AGENT_TOOLS] == [
        "query_order", "query_product", "query_logistics", "suggest_handoff"
    ]


def test_suggested_actions_are_independent_and_deduped():
    out = suggested_actions_from_handoff(["transfer", "ticket", "transfer"], "投诉")
    assert out == [
        {"action": "transfer", "label": "转人工"},
        {"action": "ticket", "label": "建工单", "ticket_type": "投诉"},
    ]


def test_invalid_handoff_option_is_dropped():
    assert suggested_actions_from_handoff(["bogus", "ticket"], "咨询") == [
        {"action": "ticket", "label": "建工单", "ticket_type": "咨询"}
    ]


def test_agent_prompt_has_handoff_and_no_ticket_rules():
    from app.prompts.customer_service import render_agent_system_prompt
    prompt = render_agent_system_prompt(
        evidence=[{"n": 1, "section_path": "退货政策", "question": "能退吗", "answer": "7 天", "score": 0.9}]
    )
    assert "suggest_handoff" in prompt
    assert "不得调用 create_ticket" in prompt
    assert "工具返回" in prompt and "[n]" in prompt
    assert "串行" in prompt
    assert "[1] 章节:退货政策" in prompt
