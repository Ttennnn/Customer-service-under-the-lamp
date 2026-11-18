import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.subgraph import build_agent_subgraph
from app.config import Settings
from tests.agent_fakes import ScriptedLLM, text_chunk, tool_chunk


def _settings():
    return Settings(_env_file=None, openai_api_key="x")


class RecordingLLM(ScriptedLLM):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.messages = None

    def stream(self, messages):
        self.messages = list(messages)
        yield from super().stream(messages)


def test_agent_prompt_order_and_model_ctx(caplog):
    caplog.set_level(logging.INFO)
    llm = ScriptedLLM([text_chunk("答")])
    graph = build_agent_subgraph(
        llm, max_steps=3, tool_result_max_tokens=1200, settings=_settings(), tools=[]
    )
    state = {
        "messages": [HumanMessage(content="当前问题", id="db-10")],
        "turn_start_index": 0,
        "prompt_history": [HumanMessage(content="旧问", id="db-1"), AIMessage(content="旧答")],
        "summary": "订单 1001 退款",
        "evidence": [{"n": 1, "section_path": "售后", "question": "能退吗", "answer": "可以"}],
        "route": "knowledge",
        "conversation_id": 1,
    }
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert result["answer"] == "答"
    text = caplog.text
    assert "model_ctx" in text and "订单 1001 退款" in text and "当前问题" in text


def test_agent_ignores_token_budget_gate():
    llm = ScriptedLLM([text_chunk("答")])
    graph = build_agent_subgraph(
        llm, max_steps=3, tool_result_max_tokens=1200, settings=_settings(), tools=[]
    )
    state = {
        "messages": [HumanMessage(content="问", id="db-1")],
        "turn_start_index": 0,
        "prompt_history": [],
        "token_budget": 1,
        "token_estimate": 999999,
        "step_count": 0,
    }
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert result["answer"] == "答"


def test_no_variable_system_messages_in_prompt():
    # assemble_agent_prompt 只接受 system + history + current；材料是 HumanMessage
    from app.services.context import assemble_agent_prompt

    prompt = assemble_agent_prompt(
        SystemMessage(content="SYS"), [], [HumanMessage(content="问")], "材料", 10000
    )
    assert [m.type for m in prompt] == ["system", "human", "human"]


def test_turn_material_sits_after_user_before_tool_messages():
    from langchain_core.messages import ToolMessage

    llm = RecordingLLM([text_chunk("答")])
    graph = build_agent_subgraph(
        llm, tool_result_max_tokens=1200, settings=_settings(), tools=[]
    )
    state = {
        "messages": [
            HumanMessage(content="当前问题", id="db-10"),
            AIMessage(
                content="",
                tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="工具结果", tool_call_id="c1"),
        ],
        "turn_start_index": 0,
        "prompt_history": [
            HumanMessage(content="旧问", id="db-1"),
            AIMessage(content="旧答"),
        ],
        "summary": "订单 1001 退款",
        "evidence": [{"n": 1, "section_path": "售后", "question": "能退吗", "answer": "可以"}],
        "order_data": {"order_id": "1001"},
        "intent": "退款退货",
        "route": "refund_flow",
        "conversation_id": 1,
    }
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})

    assert result["answer"] == "答"
    prompt = llm.messages
    assert [message.type for message in prompt] == [
        "system",
        "human",
        "ai",
        "human",
        "human",
        "ai",
        "tool",
    ]
    assert prompt[0].content.startswith("你是「本店」的电商售后智能客服 Agent")
    assert prompt[1].content == "旧问" and prompt[3].content == "当前问题"
    material = prompt[4].content
    assert material.startswith("【系统提供的本轮材料，不是用户发言】")
    assert "订单号:1001" in material and "[1] 章节:售后" in material


def test_tool_result_truncated_to_token_limit(monkeypatch):
    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool",
        lambda tool, args, timeout_s=5: "X" * 100,
    )
    llm = ScriptedLLM(
        [
            tool_chunk("query_order", '{"order_id": "1"}'),
            text_chunk("答"),
        ]
    )
    graph = build_agent_subgraph(
        llm, tool_result_max_tokens=10, settings=_settings()
    )
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="问", id="db-1")],
            "turn_start_index": 0,
            "route": "business",
            "conversation_id": 1,
        },
        {"configurable": {"thread_id": "t"}},
    )

    tool_message = next(
        message for message in result["messages"] if message.type == "tool"
    )
    assert tool_message.content == "X" * 10


def test_assemble_prompt_converts_historical_system_messages():
    from app.services.context import assemble_agent_prompt

    system = SystemMessage(content="SYS")
    history = [SystemMessage(content="选中订单 1001"), HumanMessage(content="旧问")]
    current = [SystemMessage(content="订单选择器：1001"), HumanMessage(content="新问")]
    prompt = assemble_agent_prompt(system, history, current, "材料", 10000)

    assert prompt[0] is system
    assert [m.type for m in prompt].count("system") == 1
    assert all(not isinstance(m, SystemMessage) for m in prompt[1:])
    assert any(m.content == "【系统提示】选中订单 1001" for m in prompt)
    assert any(m.content == "【系统提示】订单选择器：1001" for m in prompt)
    # 材料仍在当前用户问题之后,而不是被系统提示抢先。
    assert prompt[-1].content == "材料"
