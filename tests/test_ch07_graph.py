"""Ch07 parent-graph wiring: prompt_history, summary and effective limits."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph
from app.agent.intent import IntentResult
from app.config import Settings
from tests.agent_fakes import ScriptedStructuredLLM, text_chunk


def _settings(**kw):
    return Settings(_env_file=None, openai_api_key="x", **kw)


CONFIG = {"configurable": {"thread_id": "t"}}

MATERIAL_HEADER = "【系统提供的本轮材料，不是用户发言】"


def test_coref_uses_prompt_history_not_full_state(monkeypatch):
    seen = {}

    def fake_resolve(llm, message, history, **kwargs):
        seen["history"] = list(history)
        return message

    monkeypatch.setattr("app.agent.graph.resolve_query", fake_resolve)
    monkeypatch.setattr(
        "app.agent.graph.build_agent_subgraph",
        lambda *a, **k: lambda state: state,
    )
    graph = build_graph(object(), ScriptedStructuredLLM("", None), settings=_settings())
    state = {
        "user_message": "它呢",
        "messages": [
            HumanMessage(content="旧问", id="db-1"),
            HumanMessage(content="中间问", id="db-2"),
            HumanMessage(content="新问", id="db-3"),
        ],
        "prompt_history": [HumanMessage(content="旧问", id="db-1")],
    }
    graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert [m.content for m in seen["history"]] == ["旧问"]


def test_coref_and_intent_receive_summary_line():
    from app.agent.coref import resolve_query
    from app.agent.intent import IntentResult, classify_intent
    from tests.agent_fakes import ScriptedStructuredLLM

    coref_llm = ScriptedStructuredLLM("完整问题", None)
    resolve_query(
        coref_llm, "它呢", [HumanMessage(content="旧问")], summary="订单 1001 退款"
    )
    assert "早前对话摘要" in coref_llm.messages[-1].content
    assert "订单 1001 退款" in coref_llm.messages[-1].content

    intent_llm = ScriptedStructuredLLM(
        "", IntentResult(intent="退款退货", confidence=0.9)
    )
    classify_intent(
        intent_llm, "它呢", [HumanMessage(content="旧问")], summary="订单 1001 退款"
    )
    assert "早前对话摘要" in intent_llm.messages[-1].content


def test_fetch_order_picker_note_is_human_message(monkeypatch):
    import app.agent.graph as graph_mod

    monkeypatch.setattr(graph_mod, "list_user_orders", lambda sid: [{"order_id": "1001"}])
    monkeypatch.setattr(graph_mod, "interrupt", lambda payload: "1001")
    monkeypatch.setattr(graph_mod, "execute_tool", lambda *a, **k: '{"order_id": "1001"}')
    out = graph_mod.fetch_order_node({"session_id": "s", "resolved_query": "退款"})
    assert isinstance(out["messages"][0], HumanMessage)
    assert "订单 1001" in out["messages"][0].content


class RecordingLLM:
    """Scripted LLM that records the assembled Agent prompt."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.messages = []

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages):
        self.messages = list(messages)
        yield from self.chunks


def _multi_turn_state():
    history = []
    for index in range(1, 7):
        history.append(
            HumanMessage(content=f"历史问题{index}" + "旧" * 180, id=f"db-{index * 2 - 1}")
        )
        history.append(
            AIMessage(content=f"历史回答{index}" + "答" * 180, id=f"db-{index * 2}")
        )
    current = HumanMessage(content="当前问题", id="db-13")
    return {
        "messages": [*history, current],
        "user_message": "当前问题",
        "prompt_history": history,
        "turn_start_index": len(history),
        "summary": "早前摘要",
        "evidence": [
            {
                "n": 1,
                "section_path": "售后",
                "question": "能退吗",
                "answer": "可以",
            }
        ],
        "conversation_id": 1,
    }


def test_multi_turn_material_follows_current_user_and_history_is_trimmed():
    llm = RecordingLLM([text_chunk("答")])
    intent_llm = ScriptedStructuredLLM(
        "当前问题", IntentResult(intent="物流", confidence=0.9)
    )
    graph = build_graph(
        llm,
        intent_llm,
        checkpointer=InMemorySaver(),
        settings=_settings(model_context_window=2000),
    )

    result = graph.invoke(_multi_turn_state(), CONFIG)

    assert result["answer"] == "答"
    prompt = llm.messages
    # Exactly one constant system prompt at the front.
    assert prompt[0].type == "system"
    assert sum(message.type == "system" for message in prompt) == 1

    contents = [message.content for message in prompt]
    # Oversized history is trimmed: the oldest turn is dropped, while the
    # current user message and the turn material survive.
    assert not any("历史问题1" in content for content in contents)
    assert "当前问题" in contents
    material = next(
        index
        for index, content in enumerate(contents)
        if content.startswith(MATERIAL_HEADER)
    )
    user_indexes = [
        index
        for index, message in enumerate(prompt)
        if message.type == "human" and not message.content.startswith(MATERIAL_HEADER)
    ]
    # Several user turns survive, and the material sits after the current/last
    # user message, not after the first one.
    assert len(user_indexes) >= 2
    assert user_indexes[-1] < material
    assert material == user_indexes[-1] + 1
    assert "早前摘要" in contents[material]
