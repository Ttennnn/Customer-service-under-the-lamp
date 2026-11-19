from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agent.raw_loop import AgentLoopResult, run_react_loop
from app.tools.registry import ALL_TOOLS


class ScriptedLLM:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages):
        yield self.chunks.pop(0)


def text_chunk(text):
    return AIMessageChunk(content=text)


def tool_chunk(name, args, call_id="c1"):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "id": call_id, "index": 0}],
    )


def test_raw_loop_converges_without_tool():
    events = []
    result = run_react_loop(
        ScriptedLLM([text_chunk("你好")]),
        ALL_TOOLS,
        [HumanMessage(content="在吗")],
        emit=events.append,
    )
    assert result.answer == "你好"
    assert result.steps == 1
    assert result.exhausted is False
    assert events == [{"type": "token", "text": "你好"}]


def test_raw_loop_executes_one_tool_then_converges(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.agent.raw_loop.execute_tool",
        lambda tool, args, timeout_s=5, retries=1: calls.append((tool.name, args))
        or '{"order_id":"1001","status":"已发货"}',
    )
    result = run_react_loop(
        ScriptedLLM([
            tool_chunk("query_order", '{"order_id":"1001"}'),
            text_chunk("订单已发货"),
        ]),
        ALL_TOOLS,
        [HumanMessage(content="订单 1001 状态")],
    )
    assert calls == [("query_order", {"order_id": "1001"})]
    assert result.answer == "订单已发货"
    assert result.steps == 2


def test_raw_loop_supports_two_tool_steps(monkeypatch):
    monkeypatch.setattr(
        "app.agent.raw_loop.execute_tool",
        lambda tool, args, timeout_s=5, retries=1: f"{tool.name}-ok",
    )
    result = run_react_loop(
        ScriptedLLM([
            tool_chunk("query_order", '{"order_id":"1001"}', "c1"),
            tool_chunk("query_product", '{"product_name":"无线蓝牙耳机"}', "c2"),
            text_chunk("已查到订单和物流"),
        ]),
        ALL_TOOLS,
        [HumanMessage(content="先查订单再查物流")],
    )
    assert result.steps == 3
    assert result.answer == "已查到订单和物流"


def test_raw_loop_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr(
        "app.agent.raw_loop.execute_tool",
        lambda tool, args, timeout_s=5, retries=1: "ok",
    )
    result = run_react_loop(
        ScriptedLLM([tool_chunk("query_order", '{"order_id":"1001"}')] * 3),
        ALL_TOOLS,
        [HumanMessage(content="循环")],
        max_steps=2,
    )
    assert result.steps == 2
    assert result.exhausted is True
    assert result.answer == ""


def test_raw_loop_stops_at_token_budget():
    result = run_react_loop(
        ScriptedLLM([text_chunk("很长" * 100)]),
        ALL_TOOLS,
        [HumanMessage(content="预算")],
        token_budget=1,
    )
    assert result.exhausted is True
    assert result.answer == ""
