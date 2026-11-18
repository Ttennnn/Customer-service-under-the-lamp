"""第 6 章父工作流图(设计文档 §6.1-§6.11)。

父图负责:

- 指代消解 / Query 改写(``resolve_query``)与八类意图分类;
- 硬编码意图 → 五出口路由 ``escalate / fallback_script / knowledge /
  refund_flow / business``;
- 知识类与退款类强制预检索 + 置信度闸;闸失败直接写兜底话术,不进 Agent;
- 投诉固定安抚话术、闲聊/其他固定文案,不调模型;
- 业务类与闸通过的知识/退款类进入 ``main_agent``(ch05 ReAct 子图);
- 统一结构化日志。

退款子流程的 ``fetch_order`` 已实现订单槽位与 interrupt/resume,
``retrieve_policy`` 已实现多 query 扩写、检索与合并;闸通过后暂时仍落
``record_log``,由 Task 14 接到 ``main_agent``。只有 ``refund_flow``
路由会走到它们。

图不在 import 时构建:``build_graph`` 由 API 首次请求时懒加载调用。
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agent.confidence import assess_confidence
from app.agent.coref import resolve_query
from app.agent.intent import classify_intent
from app.agent.routing import route_for_intent
from app.agent.state import CustomerServiceState
from app.agent.subgraph import build_agent_subgraph
from app.config import Settings, get_settings
from app.rag.query_understanding import (
    QueryAnalysis,
    expand_queries,
    expand_synonyms,
)
from app.rag.retrieval import RetrievedChunk, retrieve_hybrid
from app.tools.business import list_user_orders, query_order
from app.tools.context import CitationCollector
from app.tools.executor import execute_tool
from app.tools.faq import order_for_prompt
from app.tools.registry import tool_timeout

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
CHITCHAT_TEXT = (
    "您好，我是客服小猫，主要帮您处理订单、物流和售后问题。"
    "请问有什么可以帮您的？"
)
OTHER_TEXT = (
    "抱歉，我还没完全理解您的需求。"
    "请具体说明您要咨询的订单、物流、商品或售后问题，我再帮您处理。"
)
ORDER_FAILED_TEXT = "抱歉，订单信息查询失败，请稍后重试或联系人工客服。"
ORDER_INVALID_TEXT = "您选择的订单无效，请重新选择订单或联系人工客服。"
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


_ORDER_PREFIX_RE = re.compile(r"订单(?:号)?[：:\s]*([0-9]{4,})")
_ORDER_SUFFIX_RE = re.compile(r"([0-9]{4,})\s*号订单")
_BARE_NUMBER_RE = re.compile(r"(?<![0-9])([0-9]{4,})(?![0-9])")
_DATE_UNITS = ("年", "月", "日")
_AMOUNT_UNITS = ("元", "块", "¥", "￥", "$")
_AMOUNT_PREFIXES = ("金额", "价格", "售价", "¥", "￥", "$")


def extract_order_id(text: str) -> str | None:
    """Extract an order ID from *text*, ignoring amounts and dates.

    Order-adjacent numbers (``订单 1001`` / ``1001号订单``) win; a bare
    number is accepted only as a last resort and never when it looks like a
    year, a date component, or a monetary amount.
    """
    for pattern in (_ORDER_PREFIX_RE, _ORDER_SUFFIX_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)
    for match in _BARE_NUMBER_RE.finditer(text):
        candidate = match.group(1)
        after = text[match.end() :].lstrip()
        if after[:1] in _DATE_UNITS or after[:1] in _AMOUNT_UNITS:
            continue
        before = text[: match.start()].rstrip()
        if before.endswith(_AMOUNT_PREFIXES):
            continue
        if 1900 <= int(candidate) <= 2099:
            continue
        return candidate
    return None


def fetch_order_node(state: CustomerServiceState) -> dict[str, Any]:
    """Resolve the order slot, then fetch the order exactly once.

    Only read-only work (regex, ``list_user_orders``, logging) happens before
    ``interrupt()``: LangGraph restarts the node from its beginning on resume,
    so any side effect there would run twice.
    """
    orders = list_user_orders(state.get("session_id", ""))
    order_ids = {order["order_id"] for order in orders}
    query = state.get("resolved_query") or ""

    order_id = extract_order_id(query)
    source = "message"
    messages: list[Any] = []
    if order_id not in order_ids:
        logger.info(
            "ch06 fetch_order slot missing resolved_query=%r", query
        )
        selected = interrupt(
            {"kind": "select_order", "slot": "order_id", "orders": orders}
        )
        source = "resume"
        if not isinstance(selected, str) or selected not in order_ids:
            logger.warning("ch06 fetch_order invalid resume %r", selected)
            _emit(ORDER_INVALID_TEXT)
            return {
                "answer": ORDER_INVALID_TEXT,
                "messages": [AIMessage(content=ORDER_INVALID_TEXT)],
                "gate_passed": False,
                "gate_reason": "order_slot_failed",
            }
        order_id = selected
        messages.append(
            SystemMessage(content=f"[订单选择器] 用户选择了订单 {order_id}")
        )

    logger.info("ch06 fetch_order order_id=%s source=%s", order_id, source)
    try:
        raw = execute_tool(
            query_order,
            {"order_id": order_id},
            timeout_s=tool_timeout("query_order"),
            retries=0,
        )
        order_data = json.loads(raw)
        if not isinstance(order_data, dict):
            raise ValueError("order payload is not a JSON object")
    except Exception:  # noqa: BLE001 - 工具/解析失败走兜底,不进 Agent
        logger.exception("ch06 fetch_order failed order_id=%s", order_id)
        _emit(ORDER_FAILED_TEXT)
        failure: dict[str, Any] = {
            "answer": ORDER_FAILED_TEXT,
            "messages": [*messages, AIMessage(content=ORDER_FAILED_TEXT)],
            "gate_passed": False,
            "gate_reason": "order_fetch_failed",
        }
        # A valid picker selection already filled the slot; keep it (and its
        # SystemMessage above) even though the lookup failed.
        if source == "resume":
            failure["order_id"] = order_id
        return failure

    return {"order_id": order_id, "order_data": order_data, "messages": messages}


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
        message = state.get("user_message", "")
        history = list(state.get("messages") or [])[:-1]
        resolved = resolve_query(intent_llm, message, history)
        return {"resolved_query": resolved}

    def classify_intent_node(state: CustomerServiceState) -> dict[str, Any]:
        query = state.get("resolved_query") or state.get("user_message", "")
        history = list(state.get("messages") or [])[:-1]
        result = classify_intent(intent_llm, query, history)
        return {"intent": result.intent, "intent_confidence": result.confidence}

    def route_by_intent(state: CustomerServiceState) -> dict[str, Any]:
        return {"route": route_for_intent(state.get("intent", ""))}

    def retrieve_knowledge_node(state: CustomerServiceState) -> dict[str, Any]:
        intent = state.get("intent")
        query = state.get("resolved_query") or state.get("user_message", "")
        # Pre-built analysis reuses the coref rewrite instead of asking the
        # retrieval layer to rewrite the query a second time.
        analysis = QueryAnalysis(
            original=state.get("user_message", ""),
            rewritten=query,
            keywords=expand_synonyms(query),
            rewritten_by="llm",
        )
        try:
            result = retrieve_hybrid(query, analysis=analysis)
        except Exception:  # noqa: BLE001 - 基础设施故障走保守兜底
            logger.exception(
                "ch06 retrieval_forced intent=%s failed", intent
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
            "ch06 retrieval_forced intent=%s hits=%d top_score=%s rerank_failed=%s",
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

    def script_reply_node(state: CustomerServiceState) -> dict[str, Any]:
        text = OTHER_TEXT if state.get("intent") == "其他" else CHITCHAT_TEXT
        _emit(text)
        return {
            "answer": text,
            "messages": [AIMessage(content=text)],
            "suggested_actions": [],
        }

    def retrieve_policy_node(state: CustomerServiceState) -> dict[str, Any]:
        """Expand the refund query, merge hits from all queries, then gate.

        Each expanded query is retrieved with a prebuilt analysis so
        ``retrieve_hybrid`` does not rewrite it a second time.  Both failure
        modes short-circuit the whole round: an exception means the
        knowledge base is unusable, and any rerank fallback means the scores
        are RRF ranks that must not be compared with the rerank threshold.
        """
        query = state.get("resolved_query") or state.get("user_message", "")
        order_data = state.get("order_data") or {}
        order_id = state.get("order_id")
        queries = expand_queries(intent_llm, query, order_data)

        merged: dict[int, RetrievedChunk] = {}
        for expanded in queries:
            analysis = QueryAnalysis(
                original=expanded,
                rewritten=expanded,
                keywords=expand_synonyms(expanded),
                rewritten_by="llm",
            )
            try:
                result = retrieve_hybrid(expanded, analysis=analysis)
            except Exception:  # noqa: BLE001 - 任一 query 基础设施故障整趟兜底
                logger.exception(
                    "ch06 retrieve_policy order_id=%s query=%r failed",
                    order_id,
                    expanded,
                )
                return {
                    "expanded_queries": queries,
                    "evidence": [],
                    "sources": [],
                    "confidence": None,
                    "gate_passed": False,
                    "gate_reason": "infra_error",
                }
            if result.rerank_failed:
                logger.warning(
                    "ch06 retrieve_policy order_id=%s query=%r rerank_failed",
                    order_id,
                    expanded,
                )
                return {
                    "expanded_queries": queries,
                    "evidence": [],
                    "sources": [],
                    "confidence": None,
                    "gate_passed": False,
                    "gate_reason": "rerank_failed",
                    "rerank_failed": True,
                }
            for chunk in result.chunks:
                current = merged.get(chunk.chunk_id)
                if current is None or chunk.score > current.score:
                    merged[chunk.chunk_id] = chunk

        merged_chunks = sorted(
            merged.values(), key=lambda chunk: chunk.score, reverse=True
        )[: settings.rerank_top_n]
        ordered = order_for_prompt(merged_chunks)
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
            merged_chunks,
            settings.confidence_min_score,
            rerank_failed=False,
        )
        logger.info(
            "ch06 retrieve_policy order_id=%s queries=%s hits=%d top_score=%s",
            order_id,
            queries,
            len(merged_chunks),
            verdict.top_score,
        )
        return {
            "expanded_queries": queries,
            "evidence": evidence,
            "sources": sources,
            "confidence": verdict.top_score,
            "gate_passed": verdict.passed,
            "gate_reason": verdict.reason,
            "rerank_failed": False,
        }

    def record_log_node(
        state: CustomerServiceState, config: RunnableConfig
    ) -> dict[str, Any]:
        configurable = config.get("configurable", {})
        suggested_actions = state.get("suggested_actions") or []
        refund_form = any(
            isinstance(action, dict) and action.get("action") == "refund_form"
            for action in suggested_actions
        )
        payload = {
            "thread_id": configurable.get("thread_id"),
            "conversation_id": state.get("conversation_id"),
            "resolved_query": state.get("resolved_query"),
            "intent": state.get("intent"),
            "intent_confidence": state.get("intent_confidence"),
            "route": state.get("route"),
            "order_id": state.get("order_id"),
            "expanded_queries": list(state.get("expanded_queries") or []),
            "gate_passed": state.get("gate_passed", False),
            "gate_reason": state.get("gate_reason"),
            "confidence": state.get("confidence"),
            "retrieval_hits": len(state.get("evidence") or []),
            "rerank_failed": state.get("rerank_failed", False),
            "tool_steps": state.get("step_count", 0),
            "token_estimate": state.get("token_estimate", 0),
            "suggested_actions": suggested_actions,
            "refund_form": refund_form,
            "answer_chars": len(state.get("answer") or ""),
        }
        logger.info("ch06 turn_log %s", json.dumps(payload, ensure_ascii=False))
        return {}

    def route_edge(state: CustomerServiceState) -> str:
        return state["route"]

    def order_edge(state: CustomerServiceState) -> str:
        # ``fetch_order`` only writes ``order_data`` on success; every failure
        # path sets ``gate_reason`` and skips policy retrieval.
        if state.get("order_data") is None:
            return "record_log"
        return "retrieve_policy"

    def gate_edge(state: CustomerServiceState) -> str:
        if not state.get("gate_passed"):
            return "record_log"
        # Task 14 connects the refund/after-sales agent; until then a passing
        # refund gate stays on record_log.
        if state.get("route") == "refund_flow":
            return "record_log"
        return "main_agent"

    graph = StateGraph(CustomerServiceState)
    graph.add_node("resolve_reference", resolve_reference)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("route_by_intent", route_by_intent)
    graph.add_node("retrieve_knowledge", retrieve_knowledge_node)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("complaint_reply", complaint_reply_node)
    graph.add_node("script_reply", script_reply_node)
    graph.add_node("fetch_order", fetch_order_node)
    graph.add_node("retrieve_policy", retrieve_policy_node)
    graph.add_node("main_agent", agent_subgraph)
    graph.add_node("record_log", record_log_node)

    graph.add_edge(START, "resolve_reference")
    graph.add_edge("resolve_reference", "classify_intent")
    graph.add_edge("classify_intent", "route_by_intent")
    graph.add_conditional_edges(
        "route_by_intent",
        route_edge,
        {
            "escalate": "complaint_reply",
            "fallback_script": "script_reply",
            "knowledge": "retrieve_knowledge",
            "refund_flow": "fetch_order",
            "business": "main_agent",
        },
    )
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    # Only a successful order lookup reaches policy retrieval; every failure
    # path logs straight away.
    graph.add_conditional_edges(
        "fetch_order",
        order_edge,
        {"retrieve_policy": "retrieve_policy", "record_log": "record_log"},
    )
    graph.add_edge("retrieve_policy", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        gate_edge,
        {"main_agent": "main_agent", "record_log": "record_log"},
    )
    graph.add_edge("complaint_reply", "record_log")
    graph.add_edge("script_reply", "record_log")
    graph.add_edge("main_agent", "record_log")
    graph.add_edge("record_log", END)
    return graph.compile(checkpointer=checkpointer)
