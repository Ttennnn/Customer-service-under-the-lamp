import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from app.config import Settings
from app.services.context import (
    assemble_agent_prompt, choose_degradation, log_context, message_tokens,
    render_l2, split_layers,
)


def _settings():
    return Settings(_env_file=None, openai_api_key="x", assistant_l2_chars=40)


def _hist():
    return [
        HumanMessage(content="问1", id="db-10"),
        AIMessage(content="答1" * 40),
        HumanMessage(content="问2", id="db-20"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        ToolMessage(content="X" * 500, tool_call_id="c1"),
        AIMessage(content="答2"),
        HumanMessage(content="问3", id="db-30"),
        AIMessage(content="答3"),
    ]


def test_split_layers_by_user_anchors():
    # layer1_from=10（问1 的 db id）→ L1 从下一条用户消息 问2 开始
    l2, l1 = split_layers(_hist(), summary_upto_msg_id=None, layer1_from_msg_id=10)
    assert [m.content for m in l2] == ["问1", "答1" * 40]
    assert l1[0].content == "问2" and l1[-1].content == "答3"


def test_split_layers_null_anchors():
    l2, l1 = split_layers(_hist(), None, None)
    assert l2 == [] and len(l1) == 8


def test_split_layers_null_layer1_anchor_does_not_compare_ids_to_message_count():
    # db ids are global autoincrements: db-20 must not look "past" a 4-message list.
    hist = [
        HumanMessage(content="问1", id="db-2"),
        AIMessage(content="答1"),
        HumanMessage(content="问2", id="db-20"),
        AIMessage(content="答2"),
    ]
    l2, l1 = split_layers(hist, summary_upto_msg_id=None, layer1_from_msg_id=None)
    assert l2 == [] and l1 == hist

    l2, l1 = split_layers(hist, summary_upto_msg_id=None, layer1_from_msg_id=2)
    assert l2 == hist[:2]
    assert l1[0].content == "问2" and l1[-1].content == "答2"


def test_render_l2_keeps_user_full_prefixes_assistant_and_marks_tool():
    # summary_upto=10 → L2 从 问2 起；layer1_from=20 → L1 从 问3 起
    l2, _ = split_layers(_hist(), 10, 20)
    rendered = render_l2(l2, _settings())
    assert rendered[0].content == "问2"
    assert rendered[1].content.startswith("[调用工具: query_order]")
    assert rendered[2].content == "[工具结果: query_order 已省略 500 字]"
    assert len(rendered[3].content) <= 41  # 40 + ellipsis


def test_message_tokens_includes_role_overhead():
    assert message_tokens([HumanMessage(content="abcd")]) == 8


def test_choose_degradation_keeps_one_turn_and_drops_oldest_first():
    one_turn = [HumanMessage(content="问3", id="db-30"), AIMessage(content="答3")]
    assert choose_degradation(one_turn, 20, l1_budget=1, low_water_ratio=0.5) is None
    two_turns = [
        HumanMessage(content="问2", id="db-20"), AIMessage(content="答2"),
        HumanMessage(content="问3", id="db-30"), AIMessage(content="答3"),
    ]
    assert choose_degradation(two_turns, 20, l1_budget=1, low_water_ratio=0.5) == 30


def test_assemble_agent_prompt_inserts_material_after_user_before_tools():
    system = SystemMessage(content="SYS")
    current = [
        HumanMessage(content="当前问题"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="工具", tool_call_id="c1"),
    ]
    prompt = assemble_agent_prompt(system, [], current, "【摘要】订单 1001", 10000)
    assert prompt[0] is system
    assert prompt[1].content == "当前问题"
    assert prompt[2].content.startswith("【摘要】")
    assert prompt[3].type == "ai"


def test_assemble_agent_prompt_trims_history_but_keeps_current_and_material():
    system = SystemMessage(content="SYS")
    history = [HumanMessage(content="旧" * 100, id=f"db-{i}") for i in range(5)]
    current = [HumanMessage(content="当前问题")]
    prompt = assemble_agent_prompt(system, history, current, "材料", 40)
    assert any(m.content == "当前问题" for m in prompt)
    assert any(m.content == "材料" for m in prompt)
    assert len(prompt) < 7


def test_log_context_contains_summary_preview_and_count(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test.ch07")
    log_context(logger, "history_ctx", "第1段：订单 1001", [HumanMessage(content="问1")], 1)
    text = caplog.text
    assert "history_ctx" in text and "第1段：订单 1001" in text and "问1" in text and '"window_count": 1' in text
