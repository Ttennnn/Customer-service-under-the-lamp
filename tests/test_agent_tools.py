import json

from langchain_core.tools import tool

from app.agent.tools import (
    legacy_tools,
    suggested_actions_from_handoff,
    tools_for_route,
)
from app.tools.registry import builtin_specs
from app.tools.spec import spec_from_tool


def test_business_route_binds_exposed_registry_tools_in_registry_order():
    names = [t.name for t in tools_for_route("business", builtin_specs())]

    assert names == [
        "query_order",
        "query_product",
        "query_promotions",
        "suggest_handoff",
        "create_ticket",
    ]
    assert "query_faq" not in names


def test_refund_route_binds_controls_and_aftersales_only():
    specs = builtin_specs()

    @tool
    def query_warranty(order_id: str) -> str:
        """查询保修。"""
        return "{}"

    specs["query_warranty"] = spec_from_tool(
        query_warranty, source="mcp", mcp_server="aftersales"
    )

    names = [t.name for t in tools_for_route("refund_flow", specs)]

    assert names == [
        "submit_refund",
        "suggest_handoff",
        "create_ticket",
        "query_warranty",
    ]
    assert "query_order" not in names
    assert "query_faq" not in names


def test_other_routes_exclude_aftersales_mcp_tools():
    specs = builtin_specs()

    @tool
    def query_warranty(order_id: str) -> str:
        """查询保修。"""
        return "{}"

    specs["query_warranty"] = spec_from_tool(
        query_warranty, source="mcp", mcp_server="aftersales"
    )

    for route in ("knowledge", "business", "", "escalate"):
        names = [t.name for t in tools_for_route(route, specs)]
        assert "query_warranty" not in names
        assert "submit_refund" not in names
        assert "query_faq" not in names


def test_route_scoped_spec_only_binds_to_its_route():
    specs = builtin_specs()
    assert "submit_refund" in [
        t.name for t in tools_for_route("refund_flow", specs)
    ]
    assert "submit_refund" not in [
        t.name for t in tools_for_route("knowledge", specs)
    ]


def test_legacy_tools_are_the_three_read_tools():
    names = [t.name for t in legacy_tools(builtin_specs())]
    assert names == ["query_order", "query_product", "query_faq"]


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


def test_agent_prompt_has_handoff_and_ticket_confirmation_rules():
    from app.prompts.customer_service import (
        render_after_sales_agent_system_prompt,
        render_agent_system_prompt,
        render_turn_material,
    )

    prompt = render_agent_system_prompt()
    assert "suggest_handoff" in prompt
    assert "不得调用 create_ticket" not in prompt
    assert "明确要求建工单" in prompt
    assert "确认预览" in prompt
    assert "不得编造" in prompt
    assert "工单号" in prompt
    assert "工具返回" in prompt and "[n]" in prompt
    assert "串行" in prompt

    after_sales = render_after_sales_agent_system_prompt()
    assert "明确要求建工单" in after_sales
    assert "确认预览" in after_sales
    assert "不得编造" in after_sales
    material = render_turn_material(
        "",
        [{"n": 1, "section_path": "退货政策", "question": "能退吗", "answer": "7 天", "score": 0.9}],
        None,
        "",
    )
    assert "[1] 章节:退货政策" in material


def test_agent_prompt_mentions_the_promotions_tool():
    from app.prompts.customer_service import render_agent_system_prompt

    prompt = render_agent_system_prompt()
    assert "query_promotions" in prompt
    assert "优惠" in prompt
