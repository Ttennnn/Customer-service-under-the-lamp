"""Chat orchestration gate: sources frame, refusal, mixed intent, citations.

These tests exercise ``stream_chat`` end-to-end with fake LLMs and a
monkeypatched ``retrieve_hybrid`` so no live LLM/Milvus/MySQL is touched.
The real engine path is used for ``query_faq`` cases so the citation
collector is populated through the same contextvar path as production.
"""

import json
import logging

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from app.agent.tools import legacy_tools
from app.config import Settings
from app.models import LowConfidenceQuestion, Message
from app.rag.retrieval import (
    RetrievalResult,
    RetrievalStrategy,
    RetrievedChunk,
)
from app.rag.self_check import EvidenceAssessment
from app.services.chat import stream_chat
from app.tools.context import current_citation_collector
from app.tools.faq import ERROR_TEXT
from app.tools.registry import get_all_specs
from app.tools.spec import ToolRun


def _legacy_binding():
    settings = Settings(_env_file=None, openai_api_key="test")
    specs = get_all_specs(settings)
    return specs, legacy_tools(specs)


SPECS, LEGACY_TOOLS = _legacy_binding()


class FakeChunk:
    def __init__(self, content):
        self.content = content


class FakeBoundLLM:
    def __init__(self, tool_calls, final_tokens=("已", "查到")):
        self.tool_calls = tool_calls
        self.final_tokens = list(final_tokens)
        self.stream_calls = 0
        self.stream_messages = None

    def invoke(self, messages):
        return AIMessage(content="", tool_calls=self.tool_calls)

    def stream(self, messages):
        self.stream_calls += 1
        self.stream_messages = messages
        for token in self.final_tokens:
            yield FakeChunk(token)


class FakeLLM:
    def __init__(self, tool_calls, final_tokens=("已", "查到")):
        self.bound = FakeBoundLLM(tool_calls, final_tokens)
        self.bind_calls = []

    def bind_tools(self, tools, **kwargs):
        self.bind_calls.append(kwargs)
        return self.bound

    def stream(self, messages):
        for token in self.bound.final_tokens:
            yield FakeChunk(token)


class FakeAssessmentLLM:
    """Stands in for the temperature-0 self-check LLM."""

    def __init__(self, assessment=None, error=None):
        self.assessment = assessment
        self.error = error
        self.calls = 0

    def with_structured_output(self, schema, method=None):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.assessment


def faq_call(call_id="c1", keyword="邮费怎么算"):
    return {"name": "query_faq", "args": {"keyword": keyword}, "id": call_id}


def order_call(call_id="c1"):
    return {"name": "query_order", "args": {"order_id": "1001"}, "id": call_id}


def make_chunk(chunk_id=1):
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=0.9,
        questions="邮费怎么算",
        answer="满 99 元包邮",
        section_path="配送政策 > 邮费",
        category="商品",
        content_type="faq",
    )


def make_result(chunks):
    from app.rag.query_understanding import QueryAnalysis

    analysis = QueryAnalysis("q", "q", [], "fallback")
    return RetrievalResult(chunks, RetrievalStrategy.HYBRID_RERANK, analysis)


def json_frames(frames):
    return [json.loads(frame) for frame in frames if frame.startswith("{")]


def source_frames(frames):
    return [frame for frame in json_frames(frames) if frame.get("type") == "sources"]


@pytest.fixture(autouse=True)
def _reset_collector():
    """Keep a collector set by one test from leaking into the next one."""
    current_citation_collector.set(None)
    yield
    current_citation_collector.set(None)


def test_query_faq_emits_sources_frame(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.faq.retrieve_hybrid",
        lambda *a, **k: make_result([make_chunk(1)]),
    )
    llm = FakeLLM([faq_call()], final_tokens=("回答见[1]",))
    assessment = FakeAssessmentLLM(EvidenceAssessment(useful=True))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "邮费怎么算",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    assert any(tc.get("type") == "tool" for tc in json_frames(frames))
    sources = source_frames(frames)
    assert len(sources) == 1
    payload = sources[0]["sources"]
    assert [s["n"] for s in payload] == [1]
    assert payload[0]["chunk_id"] == 1
    assert payload[0]["section_path"] == "配送政策 > 邮费"
    assert payload[0]["content_type"] == "faq"
    assert frames[-1] == "回答见[1]"
    assert assessment.calls == 1
    assert session.query(LowConfidenceQuestion).count() == 0

    rows = session.query(Message).order_by(Message.id).all()
    assert rows[-1].role == "assistant" and rows[-1].content == "回答见[1]"


def test_pure_knowledge_question_forces_query_faq_tool_choice(
    session, monkeypatch
):
    monkeypatch.setattr(
        "app.tools.builtin.faq.retrieve_hybrid",
        lambda *a, **k: make_result([make_chunk(1)]),
    )
    llm = FakeLLM([faq_call()], final_tokens=("回答见[1]",))
    assessment = FakeAssessmentLLM(EvidenceAssessment(useful=True))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "LX-A100 的色温是多少",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    forced = {"type": "function", "function": {"name": "query_faq"}}
    assert llm.bind_calls[0] == {"tool_choice": forced}
    # The forced choice must apply to the initial call only; the final answer
    # streams under automatic tool choice.
    assert llm.bind_calls[-1] == {}
    assert len(source_frames(frames)) == 1
    assert assessment.calls == 1
    assert frames[-1] == "回答见[1]"


def test_order_only_question_uses_automatic_tool_choice(session, monkeypatch):
    calls = []

    def fake_execute_tool_call(tool_call, cid, **kwargs):
        calls.append(tool_call["name"])
        return ToolRun(status="ok", content='{"order_id":"1001","status":"运输中"}')

    monkeypatch.setattr("app.services.chat.execute_tool_call", fake_execute_tool_call)
    llm = FakeLLM([order_call()], final_tokens=("订单", "运输中"))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "我的订单到哪了",
            1000,
            specs=SPECS,
        )
    )

    assert calls == ["query_order"]
    assert llm.bind_calls == [{}]
    assert not source_frames(frames)
    assert frames[-2:] == ["订单", "运输中"]


def test_mixed_question_uses_automatic_tool_choice(session, monkeypatch):
    calls = []

    def fake_execute_tool_call(tool_call, cid, **kwargs):
        calls.append(tool_call["name"])
        return ToolRun(status="ok", content='{"order_id":"1001","status":"运输中"}')

    monkeypatch.setattr("app.services.chat.execute_tool_call", fake_execute_tool_call)
    llm = FakeLLM([order_call()], final_tokens=("订单已查到",))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "LX-Z900 的色温是多少，另外我的订单1001到哪了",
            1000,
            specs=SPECS,
        )
    )

    assert calls == ["query_order"]
    assert llm.bind_calls == [{}]
    assert not source_frames(frames)
    assert frames[-1] == "订单已查到"


def test_query_faq_useful_false_records_pool_and_refuses(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.faq.retrieve_hybrid",
        lambda *a, **k: make_result([make_chunk(1)]),
    )
    llm = FakeLLM([faq_call()], final_tokens=("不应出现",))
    assessment = FakeAssessmentLLM(
        EvidenceAssessment(useful=False, reason="证据不足")
    )

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "支持以旧换新吗",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    assert len(source_frames(frames)) == 1
    assert any("知识库暂未收录足够的信息" in f for f in frames)
    assert llm.bound.stream_calls == 0

    row = session.query(LowConfidenceQuestion).one()
    assert row.conversation_id is not None
    assert row.raw_question == "支持以旧换新吗"
    assert row.reason == "证据不足"
    assert row.source == "self_check"

    rows = session.query(Message).order_by(Message.id).all()
    assert rows[-1].role == "assistant"
    assert "知识库暂未收录足够的信息" in rows[-1].content


def test_query_faq_infra_error_refuses_without_pool(session, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr("app.tools.builtin.faq.retrieve_hybrid", boom)
    llm = FakeLLM([faq_call()], final_tokens=("不应出现",))
    assessment = FakeAssessmentLLM(EvidenceAssessment(useful=True))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "邮费怎么算",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    sources = source_frames(frames)
    assert sources == [{"type": "sources", "sources": []}]
    assert any(ERROR_TEXT in f for f in frames)
    assert assessment.calls == 0
    assert llm.bound.stream_calls == 0
    assert session.query(LowConfidenceQuestion).count() == 0

    rows = session.query(Message).order_by(Message.id).all()
    assert rows[-1].role == "assistant" and rows[-1].content == ERROR_TEXT


def test_query_order_only_skips_sources_and_self_check(session, monkeypatch):
    calls = []

    def fake_execute_tool_call(tool_call, cid, **kwargs):
        calls.append(tool_call["name"])
        return ToolRun(status="ok", content='{"order_id":"1001","status":"运输中"}')

    monkeypatch.setattr("app.services.chat.execute_tool_call", fake_execute_tool_call)
    llm = FakeLLM([order_call()], final_tokens=("订单", "运输中"))
    assessment = FakeAssessmentLLM(error=RuntimeError("must not be called"))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "订单1001到哪了",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    assert calls == ["query_order"]
    assert not source_frames(frames)
    assert assessment.calls == 0
    assert frames[-2:] == ["订单", "运输中"]
    assert session.query(LowConfidenceQuestion).count() == 0


def test_mixed_intent_keeps_other_tool_answer(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.faq.retrieve_hybrid",
        lambda *a, **k: make_result([make_chunk(1)]),
    )
    llm = FakeLLM(
        [order_call("c1"), faq_call("c2")],
        final_tokens=("订单已查到;知识部分无法确认",),
    )
    assessment = FakeAssessmentLLM(
        EvidenceAssessment(useful=False, reason="只沾边")
    )

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "订单1001到哪了,支持以旧换新吗",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    assert len(source_frames(frames)) == 1
    assert llm.bound.stream_calls == 1
    assert frames[-1] == "订单已查到;知识部分无法确认"

    system_messages = [
        m for m in llm.bound.stream_messages if isinstance(m, SystemMessage)
    ]
    constraints = [m for m in system_messages if "无法确认" in m.content]
    assert len(constraints) == 1
    assert "本轮知识库证据不足" in constraints[0].content
    assert "不得猜测" in constraints[0].content

    assert session.query(LowConfidenceQuestion).count() == 1
    rows = session.query(Message).order_by(Message.id).all()
    assert rows[-1].role == "assistant"
    assert rows[-1].content == "订单已查到;知识部分无法确认"


def test_self_check_failure_refuses_without_pool(session, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.faq.retrieve_hybrid",
        lambda *a, **k: make_result([make_chunk(1)]),
    )
    llm = FakeLLM([faq_call()], final_tokens=("不应出现",))
    assessment = FakeAssessmentLLM(error=RuntimeError("self-check down"))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "邮费怎么算",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    assert len(source_frames(frames)) == 1
    assert any("知识库暂时无法确认这个问题" in f for f in frames)
    assert assessment.calls == 2  # default retries=1 -> two attempts
    assert llm.bound.stream_calls == 0
    assert session.query(LowConfidenceQuestion).count() == 0

    rows = session.query(Message).order_by(Message.id).all()
    assert rows[-1].role == "assistant"
    assert "知识库暂时无法确认这个问题" in rows[-1].content


def test_mixed_intent_self_check_failure_keeps_answer_without_pool(
    session, monkeypatch
):
    monkeypatch.setattr(
        "app.tools.builtin.faq.retrieve_hybrid",
        lambda *a, **k: make_result([make_chunk(1)]),
    )
    llm = FakeLLM(
        [order_call("c1"), faq_call("c2")],
        final_tokens=("订单已查到;知识部分无法确认",),
    )
    assessment = FakeAssessmentLLM(error=RuntimeError("self-check down"))

    frames = list(
        stream_chat(
            llm,
            session,
            LEGACY_TOOLS,
            "SYS",
            "u1",
            "订单1001到哪了,支持以旧换新吗",
            1000,
            self_check_llm=assessment,
            specs=SPECS,
        )
    )

    assert llm.bound.stream_calls == 1
    assert frames[-1] == "订单已查到;知识部分无法确认"
    constraints = [
        m
        for m in llm.bound.stream_messages
        if isinstance(m, SystemMessage) and "无法确认" in m.content
    ]
    assert len(constraints) == 1
    assert "本轮知识库证据不足" in constraints[0].content
    # No pool row on a self-check infrastructure failure, so the constraint
    # must not claim the question was recorded.
    assert "已记入低置信度问题池" not in constraints[0].content
    assert session.query(LowConfidenceQuestion).count() == 0


def test_unknown_citation_is_logged_not_blocking(session, monkeypatch, caplog):
    monkeypatch.setattr(
        "app.tools.builtin.faq.retrieve_hybrid",
        lambda *a, **k: make_result([make_chunk(1)]),
    )
    llm = FakeLLM([faq_call()], final_tokens=("结论见[1]和[9]",))
    assessment = FakeAssessmentLLM(EvidenceAssessment(useful=True))

    with caplog.at_level(logging.WARNING, logger="app.services.chat"):
        frames = list(
            stream_chat(
                llm,
                session,
                LEGACY_TOOLS,
                "SYS",
                "u1",
                "邮费怎么算",
                1000,
                self_check_llm=assessment,
                specs=SPECS,
            )
        )

    assert frames[-1] == "结论见[1]和[9]"
    assert "9" in caplog.text
    assert any("unknown citation" in record.message for record in caplog.records)
    rows = session.query(Message).order_by(Message.id).all()
    assert rows[-1].role == "assistant"
    assert rows[-1].content == "结论见[1]和[9]"
