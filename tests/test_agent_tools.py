import json

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


def test_tools_for_route_splits_refund_and_business():
    from app.agent.tools import tools_for_route
    assert [t.name for t in tools_for_route("refund_flow")] == ["submit_refund", "suggest_handoff"]
    assert [t.name for t in tools_for_route("business")] == [
        "query_order", "query_product", "query_logistics", "suggest_handoff"
    ]


def test_tools_for_route_defaults_to_general_tools():
    from app.agent.tools import tools_for_route
    for route in ("knowledge", "", "escalate"):
        assert [t.name for t in tools_for_route(route)] == [
            "query_order", "query_product", "query_logistics", "suggest_handoff"
        ]


def test_submit_refund_only_acknowledges():
    from app.agent.tools import submit_refund
    payload = json.loads(submit_refund.invoke({"order_id": "1001"}))
    assert payload["acknowledged"] is True
    assert payload["order_id"] == "1001"


def test_suggested_actions_from_refund():
    from app.agent.tools import suggested_actions_from_refund
    assert suggested_actions_from_refund("1001") == [
        {"action": "refund_form", "label": "提交退款工单", "order_id": "1001"}
    ]


def test_agent_prompt_has_handoff_and_no_ticket_rules():
    from app.prompts.customer_service import (
        render_agent_system_prompt,
        render_turn_material,
    )

    prompt = render_agent_system_prompt()
    assert "suggest_handoff" in prompt
    assert "不得调用 create_ticket" in prompt
    assert "工具返回" in prompt and "[n]" in prompt
    assert "串行" in prompt
    material = render_turn_material(
        "",
        [{"n": 1, "section_path": "退货政策", "question": "能退吗", "answer": "7 天", "score": 0.9}],
        None,
        "",
    )
    assert "[1] 章节:退货政策" in material
