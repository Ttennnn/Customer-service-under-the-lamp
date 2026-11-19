"""Graph-level create_ticket interrupt confirmation flow.

The security-critical invariant: no Ticket row may exist unless the user
explicitly confirmed on the front end, and no tool (read or write) may execute
before the confirmation interrupt is resolved.
"""

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.subgraph import build_agent_subgraph
from app.config import Settings
from app.models import Ticket, ToolAuditLog
from app.tools.spec import ToolRun
from app.tools.ticket import TICKET_NO_PATTERN
from tests.agent_fakes import ScriptedLLM, text_chunk, tool_chunk


def _settings():
    return Settings(_env_file=None, openai_api_key="test")


def _initial_state(**overrides):
    state = {
        "messages": [HumanMessage(content="我要建工单")],
        "evidence": [],
        "route": "business",
        "step_count": 0,
        "token_estimate": 0,
        "token_budget": 12000,
        "max_steps": 6,
        "suggested_actions": [],
        "conversation_id": 1,
        "exhausted": False,
    }
    state.update(overrides)
    return state


def _config():
    return {"configurable": {"thread_id": "ticket-1"}}


def _confirm_chunk():
    return tool_chunk(
        "create_ticket",
        '{"description":"猫砂盆漏电，需要售后跟进","ticket_type":"售后"}',
    )


def _dual_tool_chunk():
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_order",
                "args": '{"order_id":"1001"}',
                "id": "c1",
                "index": 0,
            },
            {
                "name": "create_ticket",
                "args": '{"description":"需要建单跟进","ticket_type":"售后"}',
                "id": "c2",
                "index": 1,
            },
        ],
    )


def _two_ticket_chunk():
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "create_ticket",
                "args": '{"description":"第一件事","ticket_type":"售后"}',
                "id": "c1",
                "index": 0,
            },
            {
                "name": "create_ticket",
                "args": '{"description":"第二件事","ticket_type":"投诉"}',
                "id": "c2",
                "index": 1,
            },
        ],
    )


def _tool_messages(result):
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


def test_confirm_creates_ticket_and_emits_number(session):
    llm = ScriptedLLM([_confirm_chunk(), text_chunk("已为您建单")])
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )

    first = graph.invoke(_initial_state(), _config())

    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["kind"] == "confirm_ticket"
    assert first["__interrupt__"][0].value["preview"] == {
        "ticket_type": "售后",
        "description": "猫砂盆漏电，需要售后跟进",
    }
    # Nothing may be written before the user confirms.
    assert session.query(Ticket).count() == 0
    assert session.query(ToolAuditLog).count() == 0

    final = graph.invoke(Command(resume={"confirmed": True}), _config())

    assert TICKET_NO_PATTERN.match(final["created_ticket_no"])
    assert session.query(Ticket).count() == 1
    ticket = session.query(Ticket).one()
    assert ticket.ticket_no == final["created_ticket_no"]
    assert [row.status for row in session.query(ToolAuditLog).all()] == ["成功"]
    assert any(
        final["created_ticket_no"] in message.content
        for message in _tool_messages(final)
    )


def test_cancel_creates_no_ticket_and_audits_permission_denied(session):
    llm = ScriptedLLM([_confirm_chunk(), text_chunk("已取消建单")])
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )

    first = graph.invoke(_initial_state(), _config())
    assert "__interrupt__" in first

    final = graph.invoke(Command(resume={"confirmed": False}), _config())

    assert not final.get("created_ticket_no")
    assert session.query(Ticket).count() == 0
    row = session.query(ToolAuditLog).one()
    assert row.status == "权限拒绝"
    assert "取消" in (row.error_message or "")
    assert any("取消" in message.content for message in _tool_messages(final))


def test_invalid_resume_fails_closed(session):
    llm = ScriptedLLM([_confirm_chunk(), text_chunk("已取消建单")])
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )

    assert "__interrupt__" in graph.invoke(_initial_state(), _config())

    final = graph.invoke(Command(resume="yes"), _config())

    assert not final.get("created_ticket_no")
    assert session.query(Ticket).count() == 0
    row = session.query(ToolAuditLog).one()
    assert row.status == "权限拒绝"
    assert "确认信息无效" in (row.error_message or "")
    assert any(
        "确认信息无效" in message.content for message in _tool_messages(final)
    )


def test_read_tool_runs_once_across_interrupt(session, monkeypatch):
    counter = {"n": 0}

    def counting_execute(tool_call, conversation_id, **kwargs):
        if tool_call["name"] == "query_order":
            counter["n"] += 1
        return ToolRun(status="ok", content="订单已签收")

    monkeypatch.setattr(
        "app.agent.subgraph.execute_tool_call", counting_execute
    )
    llm = ScriptedLLM([_dual_tool_chunk(), text_chunk("已处理")])
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )

    first = graph.invoke(_initial_state(), _config())
    assert "__interrupt__" in first
    assert counter["n"] == 0  # nothing executes before confirmation

    final = graph.invoke(Command(resume={"confirmed": True}), _config())

    assert counter["n"] == 1
    assert session.query(Ticket).count() == 1
    assert any(
        message.content == "订单已签收" for message in _tool_messages(final)
    )


def test_validation_failure_does_not_interrupt(session):
    llm = ScriptedLLM(
        [
            tool_chunk("create_ticket", '{"ticket_type":"售后"}'),
            text_chunk("请补充问题描述"),
        ]
    )
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )

    result = graph.invoke(_initial_state(), _config())

    assert "__interrupt__" not in result
    assert session.query(Ticket).count() == 0
    row = session.query(ToolAuditLog).one()
    assert row.status == "校验拦下"
    assert "description" in (row.error_message or "")
    assert any(
        "description" in message.content for message in _tool_messages(result)
    )


def test_second_create_ticket_is_denied_without_library_call(session):
    llm = ScriptedLLM([_two_ticket_chunk(), text_chunk("已处理")])
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )

    first = graph.invoke(_initial_state(), _config())
    assert "__interrupt__" in first
    assert len(first["__interrupt__"]) == 1
    assert (
        first["__interrupt__"][0].value["preview"]["description"] == "第一件事"
    )
    assert session.query(Ticket).count() == 0

    final = graph.invoke(Command(resume={"confirmed": True}), _config())

    assert session.query(Ticket).count() == 1
    statuses = [row.status for row in session.query(ToolAuditLog).all()]
    assert statuses == ["成功", "校验拦下"]
    rows = session.query(ToolAuditLog).all()
    assert "一轮只能处理一个建工单请求" in (rows[1].error_message or "")
    assert any(
        "一轮只能处理一个建工单请求" in message.content
        for message in _tool_messages(final)
    )


def test_created_ticket_state_is_serializable(session):
    llm = ScriptedLLM([_confirm_chunk(), text_chunk("已为您建单")])
    graph = build_agent_subgraph(
        llm, settings=_settings(), checkpointer=InMemorySaver()
    )
    graph.invoke(_initial_state(), _config())
    final = graph.invoke(Command(resume={"confirmed": True}), _config())

    assert isinstance(final["created_ticket_no"], str)
