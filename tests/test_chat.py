import json
import threading
from langchain_core.messages import AIMessage
from app.models import Message, Ticket
from app.services.history import get_or_create_conversation
from app.tools.registry import ALL_TOOLS
from app.services.chat import stream_chat

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
    events = list(stream_chat(FakeLLM(tool_call=True), session, ALL_TOOLS, "SYS", "u1", "订单1001物流", 1000))
    assert any(e.startswith('{"type":"tool"') for e in events)
    assert events[-1] == "查到"  # 最终答案 token
    rows = session.query(Message).all()
    roles = [r.role for r in rows]
    assert roles == ["user", "assistant", "tool", "assistant"]

def test_stream_chat_no_tool_still_streams(session):
    events = list(stream_chat(FakeLLM(tool_call=False), session, ALL_TOOLS, "SYS", "u1", "你好", 1000))
    assert events == ["直接", "回答"]

class FakeTicketBoundLLM:
    def invoke(self, messages):
        return AIMessage(content="", tool_calls=[{"name": "create_ticket", "args": {"description": "屏幕碎了", "ticket_type": "售后"}, "id": "c1"}])
    def stream(self, messages):
        yield FakeChunk("已"); yield FakeChunk("转人工")

class FakeTicketLLM:
    def bind_tools(self, tools):
        return FakeTicketBoundLLM()
    def stream(self, messages):
        yield FakeChunk("直接"); yield FakeChunk("回答")

def test_create_ticket_receives_cid_across_threads(session):
    cid = get_or_create_conversation(session, "u1").id
    gen = stream_chat(FakeTicketLLM(), session, ALL_TOOLS, "SYS", "u1", "投诉", 1000)
    first = next(gen)
    assert first.startswith('{"type":"tool"')
    errors = []
    def drain():
        try:
            for _ in gen:
                pass
        except Exception as exc:
            errors.append(exc)
    t = threading.Thread(target=drain)
    t.start()
    t.join()
    assert not errors
    assert session.query(Ticket).one().conversation_id == cid
