from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.services.context import build_messages, estimate_tokens, trim_messages


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


def test_trim_messages_keeps_system_and_atomic_tool_round():
    from app.services.context import trim_messages

    system = SystemMessage(content="S")
    history = [
        HumanMessage(content="old"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}]),
        ToolMessage(content="X" * 20, tool_call_id="c1"),
        AIMessage(content="answer"),
    ]
    msgs = trim_messages(system, history, max_tokens=30)
    assert [type(m).__name__ for m in msgs] == [
        "SystemMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]


def test_trim_messages_keeps_newest_block_when_it_exceeds_budget():
    system = SystemMessage(content="S")
    history = [HumanMessage(content="old"), HumanMessage(content="X" * 10)]
    msgs = trim_messages(system, history, max_tokens=5)
    # 最新块单块超预算也必须保留;更旧的块被裁掉。
    assert [m.content for m in msgs] == ["S", "X" * 10]


def test_trim_messages_keeps_newest_block_when_it_exactly_fits_strict_budget():
    system = SystemMessage(content="S")  # budget = 5
    history = [HumanMessage(content="old"), HumanMessage(content="abcde")]
    msgs = trim_messages(system, history, max_tokens=6)
    # 严格预算下恰好用满 5 的最新块不能被丢弃。
    assert [m.content for m in msgs] == ["S", "abcde"]


def test_trim_messages_keeps_newest_tool_round_when_it_exceeds_budget():
    system = SystemMessage(content="S")
    history = [
        HumanMessage(content="older"),
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {}, "id": "c2"}]),
        ToolMessage(content="Y" * 10, tool_call_id="c2"),
    ]
    msgs = trim_messages(system, history, max_tokens=5)
    # 最新工具轮整块超预算时仍完整保留,tool_calls 与 ToolMessage 不拆散。
    assert [type(m).__name__ for m in msgs] == [
        "SystemMessage", "AIMessage", "ToolMessage"
    ]
    assert msgs[-1].content == "Y" * 10


def test_trim_messages_keeps_newest_block_when_system_alone_exceeds_budget():
    system = SystemMessage(content="S" * 10)
    history = [HumanMessage(content="older"), HumanMessage(content="new")]
    msgs = trim_messages(system, history, max_tokens=5)
    # system 已超预算时仍保留最新历史块,只裁更旧的块。
    assert [m.content for m in msgs] == ["S" * 10, "new"]
