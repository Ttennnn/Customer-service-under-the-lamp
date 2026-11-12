from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.services.context import build_messages, estimate_tokens


def test_estimate_tokens_is_len():
    assert estimate_tokens("你好") == 2


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_build_messages_keeps_system_and_current_when_over_budget():
    system = SystemMessage(content="SYS" * 500)          # 大 system
    current = HumanMessage(content="CUR" * 500)          # 大 current,单条即超预算
    msgs = build_messages(system, [HumanMessage(content="old")], current, max_tokens=10)
    assert msgs[0] == system and msgs[-1] == current


def test_build_messages_trims_oldest_first():
    system = SystemMessage(content="S")
    history = [HumanMessage(content="a" * 10), AIMessage(content="b" * 10)]
    current = HumanMessage(content="c")
    msgs = build_messages(system, history, current, max_tokens=15)
    assert [m.content for m in msgs] == ["S", "b" * 10, "c"]


def test_build_messages_keeps_all_when_within_budget():
    system = SystemMessage(content="S")
    history = [HumanMessage(content="a" * 2), AIMessage(content="b" * 3)]
    current = HumanMessage(content="c")
    msgs = build_messages(system, history, current, max_tokens=100)
    assert [m.content for m in msgs] == ["S", "a" * 2, "b" * 3, "c"]


def test_build_messages_drops_all_history_when_budget_exhausted():
    system = SystemMessage(content="SS")
    history = [HumanMessage(content="old")]
    current = HumanMessage(content="ccc")
    msgs = build_messages(system, history, current, max_tokens=5)
    assert [m.content for m in msgs] == ["SS", "ccc"]


def test_build_messages_handles_empty_history():
    system = SystemMessage(content="S")
    current = HumanMessage(content="c")
    msgs = build_messages(system, [], current, max_tokens=10)
    assert [m.content for m in msgs] == ["S", "c"]


def test_build_messages_allows_exact_budget():
    system = SystemMessage(content="S")
    history = [HumanMessage(content="ab")]
    current = HumanMessage(content="c")
    msgs = build_messages(system, history, current, max_tokens=4)
    assert [m.content for m in msgs] == ["S", "ab", "c"]


def test_estimate_tokens_none_is_zero():
    assert estimate_tokens(None) == 0

def test_build_messages_keeps_tool_round_atomic():
    system = SystemMessage(content="S")
    hist = [
        HumanMessage(content="old"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="X" * 20, tool_call_id="c1"),
        AIMessage(content="answer"),
    ]
    msgs = build_messages(system, hist, HumanMessage(content="now"), max_tokens=30)
    # 只裁掉最旧的 HumanMessage("old"),tool_calls+tool 对作为一块保留
    names = [type(m).__name__ for m in msgs]
    assert names == ["SystemMessage", "AIMessage", "ToolMessage", "AIMessage", "HumanMessage"]

def test_build_messages_drops_tool_round_atomically():
    system = SystemMessage(content="S")
    hist = [
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="X" * 100, tool_call_id="c1"),
        AIMessage(content="answer"),
    ]
    msgs = build_messages(system, hist, HumanMessage(content="now"), max_tokens=30)
    # tool 结果太大 → tool_calls 与其 tool 结果整块丢弃,不能只丢一半留下孤儿 ToolMessage
    names = [type(m).__name__ for m in msgs]
    assert names == ["SystemMessage", "AIMessage", "HumanMessage"]
