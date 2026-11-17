"""第 5 章父工作流图(设计文档 §6.1-§6.6)。

父图负责:

- 指代消解(本章透传)与七类意图分类;
- 硬编码意图 → 四出口路由;
- 知识类强制预检索 + 置信度闸;闸失败直接写兜底话术,不进 Agent;
- 投诉 / 闲聊固定话术,不调模型;
- 业务类与闸通过的知识类进入 Agent ReAct 子图;
- 统一结构化日志。

图不在 import 时构建:``build_graph`` 由 API 首次请求时懒加载调用。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agent.confidence import assess_confidence
from app.agent.intent import classify_intent
from app.agent.routing import route_for_intent
from app.agent.state import CustomerServiceState
from app.agent.subgraph import build_agent_subgraph
from app.config import Settings, get_settings
from app.rag.retrieval import retrieve_hybrid
from app.tools.context import CitationCollector
from app.tools.faq import order_for_prompt

logger = logging.getLogger(__name__)

KNOWLEDGE_GAP_TEXT = (
    "抱歉,我暂时没有找到足够可靠的知识库证据来确认这个问题。"
    "建议您联系人工客服进一步处理。"
)
INFRA_TEXT = "知识库暂时不可用,请稍后重试或联系人工客服。"
RERANK_FAILED_TEXT = "知识库暂时无法完成可靠性校验,请稍后重试或联系人工客服。"
COMPLAINT_TEXT = (
    "非常抱歉给您带来不好的体验，您的感受我们很重视。"
    "您可以选择转接人工客服进一步处理，或创建一个工单，我们会尽快跟进。"
)
CHITCHAT_TEXT = "您好，我是客服小猫，很高兴为您服务。请问有什么可以帮您的？"
COMPLAINT_ACTIONS: list[dict[str, str]] = [
    {"action": "transfer", "label": "转人工"},
    {"action": "ticket", "label": "建工单", "ticket_type": "投诉"},
]


@dataclass
class CheckpointerHandle:
    """An open SQLite checkpointer plus its connection closer."""

    saver: SqliteSaver
    close: Callable[[], None]


def open_sqlite_checkpointer(path: str) -> CheckpointerHandle:
    """Open SQLite, configure WAL/busy_timeout, run setup(), return saver + closer."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        saver = SqliteSaver(conn)
        saver.setup()
    except Exception:
        conn.close()
        raise
    return CheckpointerHandle(saver=saver, close=conn.close)


@contextmanager
def sqlite_checkpointer(path: str) -> Iterator[SqliteSaver]:
    """Yield a configured ``SqliteSaver``; close the connection on exit."""
    handle = open_sqlite_checkpointer(path)
    try:
        yield handle.saver
    finally:
        handle.close()


def _emit(text: str) -> None:
    """Send a fixed answer as one token frame."""
    get_stream_writer()({"type": "token", "text": text})


def build_graph(
    llm,
    intent_llm,
    *,
    checkpointer=None,
    settings: Settings | None = None,
):
    """Build and compile the parent workflow graph."""
    settings = settings or get_settings()

    agent_subgraph = build_agent_subgraph(
        llm,
        max_steps=settings.agent_max_steps,
        token_budget=settings.agent_token_budget,
        tool_result_max_chars=settings.agent_tool_result_max_chars,
        max_context_tokens=settings.max_context_tokens,
    )

    def resolve_reference(state: CustomerServiceState) -> dict[str, Any]:
        logger.info("ch05 resolve_reference passthrough")
        return {"resolved_message": state.get("user_message", "")}

    def classify_intent_node(state: CustomerServiceState) -> dict[str, Any]:
        message = state.get("resolved_message") or state.get("user_message", "")
        result = classify_intent(intent_llm, message)
        return {"intent": result.intent}

    def route_intent_node(state: CustomerServiceState) -> dict[str, Any]:
        return {"route": route_for_intent(state.get("intent", ""))}

    def retrieve_knowledge_node(state: CustomerServiceState) -> dict[str, Any]:
        intent = state.get("intent")
        try:
            result = retrieve_hybrid(state["resolved_message"])
        except Exception:  # noqa: BLE001 - 基础设施故障走保守兜底
            logger.exception(
                "ch05 retrieval_forced intent=%s failed", intent
            )
            return {
                "evidence": [],
                "sources": [],
                "confidence": None,
                "gate_passed": False,
                "gate_reason": "infra_error",
                "answer": INFRA_TEXT,
            }

        top_score = max((chunk.score for chunk in result.chunks), default=None)
        logger.info(
            "ch05 retrieval_forced intent=%s hits=%d top_score=%s rerank_failed=%s",
            intent,
            len(result.chunks),
            top_score,
            result.rerank_failed,
        )
        if result.rerank_failed:
            # RRF fallback scores are not comparable with the rerank threshold,
            # so they must short-circuit before confidence assessment.
            return {
                "evidence": [],
                "sources": [],
                "confidence": None,
                "gate_passed": False,
                "gate_reason": "rerank_failed",
                "rerank_failed": True,
                "answer": RERANK_FAILED_TEXT,
            }

        ordered = order_for_prompt(result.chunks)
        collector = CitationCollector()
        collector.add(ordered)
        sources = collector.payload()
        evidence = [
            {
                "n": source["n"],
                "section_path": source["section_path"],
                "question": source["question"],
                "answer": source["answer"],
                "score": source["score"],
            }
            for source in sources
        ]
        verdict = assess_confidence(
            result.chunks,
            settings.confidence_min_score,
            rerank_failed=result.rerank_failed,
        )
        return {
            "evidence": evidence,
            "sources": sources,
            "confidence": verdict.top_score,
            "gate_passed": verdict.passed,
            "gate_reason": verdict.reason,
            "rerank_failed": False,
        }

    def confidence_gate_node(state: CustomerServiceState) -> dict[str, Any]:
        reason = state.get("gate_reason")
        if reason == "infra_error":
            _emit(INFRA_TEXT)
            return {"answer": INFRA_TEXT, "messages": [AIMessage(content=INFRA_TEXT)]}
        if reason == "rerank_failed":
            _emit(RERANK_FAILED_TEXT)
            return {
                "answer": RERANK_FAILED_TEXT,
                "messages": [AIMessage(content=RERANK_FAILED_TEXT)],
            }
        if state.get("gate_passed"):
            get_stream_writer()(
                {"type": "sources", "sources": list(state.get("sources") or [])}
            )
            return {"gate_passed": True}
        _emit(KNOWLEDGE_GAP_TEXT)
        return {
            "answer": KNOWLEDGE_GAP_TEXT,
            "messages": [AIMessage(content=KNOWLEDGE_GAP_TEXT)],
        }

    def complaint_reply_node(state: CustomerServiceState) -> dict[str, Any]:
        _emit(COMPLAINT_TEXT)
        return {
            "answer": COMPLAINT_TEXT,
            "messages": [AIMessage(content=COMPLAINT_TEXT)],
            "suggested_actions": [dict(action) for action in COMPLAINT_ACTIONS],
        }

    def chitchat_reply_node(state: CustomerServiceState) -> dict[str, Any]:
        _emit(CHITCHAT_TEXT)
        return {
            "answer": CHITCHAT_TEXT,
            "messages": [AIMessage(content=CHITCHAT_TEXT)],
            "suggested_actions": [],
        }

    def record_log_node(
        state: CustomerServiceState, config: RunnableConfig
    ) -> dict[str, Any]:
        configurable = config.get("configurable", {})
        payload = {
            "thread_id": configurable.get("thread_id"),
            "conversation_id": state.get("conversation_id"),
            "intent": state.get("intent"),
            "route": state.get("route"),
            "gate_passed": state.get("gate_passed", False),
            "gate_reason": state.get("gate_reason"),
            "confidence": state.get("confidence"),
            "retrieval_hits": len(state.get("evidence") or []),
            "rerank_failed": state.get("rerank_failed", False),
            "tool_steps": state.get("step_count", 0),
            "token_estimate": state.get("token_estimate", 0),
            "suggested_actions": state.get("suggested_actions") or [],
            "answer_chars": len(state.get("answer") or ""),
        }
        logger.info("ch05 turn_log %s", json.dumps(payload, ensure_ascii=False))
        return {}

    def route_edge(state: CustomerServiceState) -> str:
        return state["route"]

    def gate_edge(state: CustomerServiceState) -> str:
        return "agent" if state.get("gate_passed") else "record_log"

    graph = StateGraph(CustomerServiceState)
    graph.add_node("resolve_reference", resolve_reference)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("route_intent", route_intent_node)
    graph.add_node("retrieve_knowledge", retrieve_knowledge_node)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("complaint_reply", complaint_reply_node)
    graph.add_node("chitchat_reply", chitchat_reply_node)
    graph.add_node("agent", agent_subgraph)
    graph.add_node("record_log", record_log_node)

    graph.add_edge(START, "resolve_reference")
    graph.add_edge("resolve_reference", "classify_intent")
    graph.add_edge("classify_intent", "route_intent")
    graph.add_conditional_edges(
        "route_intent",
        route_edge,
        {
            "knowledge": "retrieve_knowledge",
            "business": "agent",
            "complaint": "complaint_reply",
            "chitchat": "chitchat_reply",
        },
    )
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        gate_edge,
        {"agent": "agent", "record_log": "record_log"},
    )
    graph.add_edge("complaint_reply", "record_log")
    graph.add_edge("chitchat_reply", "record_log")
    graph.add_edge("agent", "record_log")
    graph.add_edge("record_log", END)
    return graph.compile(checkpointer=checkpointer)
