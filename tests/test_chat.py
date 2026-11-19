from langchain_core.messages import AIMessage

from app.agent.tools import legacy_tools
from app.config import Settings
from app.models import Message, Ticket, ToolAuditLog
from app.services.chat import stream_chat
from app.tools.registry import get_all_specs
from app.tools.spec import ToolRun


def _legacy_binding():
    settings = Settings(_env_file=None, openai_api_key="test")
    specs = get_all_specs(settings)
    return specs, legacy_tools(specs)


SPECS, LEGACY_TOOLS = _legacy_binding()


class FakeChunk:
    def __init__(self, content): self.content = content

class FakeBoundLLM:
    def __init__(self, tool_call=False):
        self.tool_call = tool_call
    def invoke(self, messages):
        if self.tool_call:
            return AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])
        return AIMessage(content="")
    def stream(self, messages):
        yield FakeChunk("已"); yield FakeChunk("查到")

class FakeLLM:
    def __init__(self, tool_call=False):
        self.tool_call = tool_call
    def bind_tools(self, tools):
        return FakeBoundLLM(tool_call=self.tool_call)
    def stream(self, messages):
        yield FakeChunk("直接"); yield FakeChunk("回答")

def test_stream_chat_tool_round_emits_frame_and_persists(session):
    events = list(stream_chat(FakeLLM(tool_call=True), session, LEGACY_TOOLS, "SYS", "u1", "订单1001物流", 1000, specs=SPECS))
    assert any(e.startswith('{"type":"tool"') for e in events)
    assert events[-1] == "查到"  # 最终答案 token
    rows = session.query(Message).all()
    roles = [r.role for r in rows]
    assert roles == ["user", "assistant", "tool", "assistant"]

def test_stream_chat_no_tool_still_streams(session):
    events = list(stream_chat(FakeLLM(tool_call=False), session, LEGACY_TOOLS, "SYS", "u1", "你好", 1000, specs=SPECS))
    assert events == ["直接", "回答"]

class FakeTicketBoundLLM:
    def invoke(self, messages):
        return AIMessage(content="", tool_calls=[{"name": "create_ticket", "args": {"description": "屏幕碎了", "ticket_type": "售后"}, "id": "c1"}])
    def stream(self, messages):
        yield FakeChunk("已"); yield FakeChunk("转人工")

class FakeTicketLLM:
    def __init__(self):
        self.bound_names = None
    def bind_tools(self, tools):
        self.bound_names = [t.name for t in tools]
        return FakeTicketBoundLLM()
    def stream(self, messages):
        yield FakeChunk("直接"); yield FakeChunk("回答")

class FakeFaqBoundLLM:
    def invoke(self, messages):
        return AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": "邮费"}, "id": "c1"}])
    def stream(self, messages):
        yield FakeChunk("已"); yield FakeChunk("查到")

class FakeFaqLLM:
    def bind_tools(self, tools, **kwargs):
        return FakeFaqBoundLLM()
    def stream(self, messages):
        yield FakeChunk("直接"); yield FakeChunk("回答")

def test_query_faq_engine_call_pins_legacy_retry_budget(session, monkeypatch):
    calls = []

    def fake_execute_tool_call(tool_call, cid, **kwargs):
        calls.append(
            (
                tool_call["name"],
                tool_call["args"],
                kwargs["retries"],
                kwargs["timeout"],
            )
        )
        return ToolRun(status="ok", content="问:邮费怎么算?\n答:满 99 元包邮")

    monkeypatch.setattr("app.services.chat.execute_tool_call", fake_execute_tool_call)
    events = list(
        stream_chat(
            FakeFaqLLM(), session, LEGACY_TOOLS, "SYS", "u1", "邮费", 1000, specs=SPECS
        )
    )
    # One attempt at the historical 25s budget — not spec 30s x 3 retries.
    assert calls == [("query_faq", {"keyword": "邮费"}, 0, 25.0)]
    assert events[-1] == "查到"


def test_legacy_tool_call_goes_through_engine(session, monkeypatch):
    calls = []

    def fake_execute_tool_call(tool_call, cid, **kwargs):
        calls.append((tool_call["name"], tool_call["args"], set(kwargs["specs"])))
        return ToolRun(status="ok", content="订单已签收")

    monkeypatch.setattr("app.services.chat.execute_tool_call", fake_execute_tool_call)
    events = list(
        stream_chat(
            FakeLLM(tool_call=True),
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "订单1001物流",
            1000,
            specs=SPECS,
        )
    )
    assert calls == [
        (
            "query_order",
            {"order_id": "1001"},
            {"query_order", "query_product", "query_faq"},
        )
    ]
    assert events[-1] == "查到"


def test_legacy_chat_does_not_bind_or_execute_create_ticket(session):
    llm = FakeTicketLLM()
    events = list(
        stream_chat(llm, session, LEGACY_TOOLS, "SYS", "u1", "投诉", 1000, specs=SPECS)
    )
    # Legacy binds exactly the three read tools; a hallucinated create_ticket
    # reaches the engine as an unknown tool (audited) and cannot write.
    assert llm.bound_names == ["query_order", "query_product", "query_faq"]
    assert session.query(Ticket).count() == 0
    audit = session.query(ToolAuditLog).filter_by(tool_name="create_ticket").one()
    assert audit.status == "失败"
    assert audit.tool_source == "builtin"
    tool_rows = [row for row in session.query(Message).all() if row.role == "tool"]
    assert len(tool_rows) == 1
    assert "未知工具" in tool_rows[0].content


def test_legacy_specs_none_cannot_execute_create_ticket(session):
    llm = FakeTicketLLM()
    events = list(
        stream_chat(llm, session, LEGACY_TOOLS, "SYS", "u1", "投诉", 1000)
    )

    # ``specs=None`` must derive the allowlist from the bound tools, not fall
    # back to the full registry.
    assert llm.bound_names == ["query_order", "query_product", "query_faq"]
    assert session.query(Ticket).count() == 0
    audit = session.query(ToolAuditLog).filter_by(tool_name="create_ticket").one()
    assert audit.status == "失败"
    assert "未知工具" in (audit.error_message or "")
    assert events[-1] == "转人工"


def test_legacy_specs_none_passes_only_bound_specs_to_engine(session, monkeypatch):
    calls = []

    def fake_execute_tool_call(tool_call, cid, **kwargs):
        calls.append(set(kwargs["specs"]))
        return ToolRun(status="ok", content="订单已签收")

    monkeypatch.setattr("app.services.chat.execute_tool_call", fake_execute_tool_call)
    events = list(
        stream_chat(
            FakeLLM(tool_call=True),
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "订单1001物流",
            1000,
        )
    )

    assert calls == [{"query_order", "query_product", "query_faq"}]
    assert events[-1] == "查到"
