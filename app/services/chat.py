"""Tool-calling chat orchestration backed by the DB conversation history.

``stream_chat`` wires the pieces from earlier tasks together:

1. Resolve (or create) the active conversation for ``user_id``.
2. Rebuild history, assemble the prompt within the token budget, and
   persist the user turn before hitting the model.
3. Bind tools and ask the model once. If it emits tool calls, execute them
   first (persisting nothing), then persist the assistant tool_calls row and
   its result rows in a single commit, feed the results back, and stream the
   final answer. Single round only — no looping.
4. When ``query_faq`` ran, emit a ``sources`` frame, run the evidence
   self-check, and refuse (or constrain a mixed-intent answer) when the
   evidence is insufficient. Infrastructure failures never pollute the
   low-confidence pool; they only produce a conservative refusal.
5. After the final answer, warn about unknown ``[n]`` citations without
   blocking or regenerating.

``current_conversation_id`` is a thread-local ``ContextVar``. Starlette's
``EventSourceResponse`` iterates this generator via ``iterate_in_threadpool``,
which may call ``next()`` on a different thread each time, so the contextvar
must be set on the SAME thread that calls ``execute_tool`` (immediately before
it, with no ``yield`` in between) rather than once at the top of the
generator. It is intentionally not reset: each tool call re-sets it, and a
cross-thread ``reset`` would raise ``ValueError``. The citation collector
follows the same rule: it is created once per turn, but re-set immediately
before each ``execute_tool`` call so the copied context always points at the
same mutable collector.
"""

import json
import logging
import re
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from app.rag.router import is_pure_knowledge_question
from app.rag.self_check import assess_evidence
from app.services.context import build_messages
from app.services.history import append_message, get_or_create_conversation, load_history
from app.services.low_confidence import record_low_confidence
from app.tools.context import (
    CitationCollector,
    current_citation_collector,
    current_conversation_id,
)
from app.tools.executor import execute_tool
from app.tools.faq import ERROR_TEXT
from app.tools.registry import TOOLS_BY_NAME, TOOL_LABELS, tool_timeout

logger = logging.getLogger(__name__)

_FALLBACK_TEXT = "抱歉,我暂时无法完成这个操作,请稍后重试或联系人工客服。"
_INFRA_REFUSAL_TEXT = "知识库暂时不可用,请稍后重试或联系人工客服。"
_LOW_CONFIDENCE_REFUSAL_TEXT = (
    "抱歉,知识库暂未收录足够的信息来确认这个问题。建议您联系人工客服进一步处理。"
)
_SELF_CHECK_FAILURE_REFUSAL_TEXT = (
    "抱歉,知识库暂时无法确认这个问题,建议您联系人工客服进一步处理。"
)
_EVIDENCE_GAP_CONSTRAINT = (
    "本轮知识库证据不足:涉及知识库的问题必须回答『无法确认』,不得猜测;"
    "可以回答其他工具已返回的事实;"
)
_MIXED_INTENT_CONSTRAINT = (
    _EVIDENCE_GAP_CONSTRAINT + "原问题已记入低置信度问题池。"
)
_TOOL_FAILURE_TEXT = "工具执行失败"
_CITATION_RE = re.compile(r"\[(\d+)\]")
_FORCED_FAQ_TOOL_CHOICE = {"type": "function", "function": {"name": "query_faq"}}


def _persist_assistant(session: Session, conversation_id: int, text: str) -> None:
    """Persist an assistant turn and commit before the generator returns."""
    append_message(session, conversation_id, "assistant", text)
    session.commit()


def _faq_failed(results: list[tuple[dict, str]]) -> bool:
    """True if a ``query_faq`` call hit an infrastructure error."""
    return any(
        call["name"] == "query_faq" and result in (ERROR_TEXT, _TOOL_FAILURE_TEXT)
        for call, result in results
    )


def _has_other_tool_results(results: list[tuple[dict, str]]) -> bool:
    """True if this turn produced any non-``query_faq`` tool result."""
    return any(call["name"] != "query_faq" for call, _ in results)


def stream_chat(
    llm,
    session: Session,
    tools: list,
    system_prompt: str,
    user_id: str,
    message: str,
    max_tokens: int,
    self_check_llm=None,
) -> Iterator[str]:
    """Stream an assistant reply token by token, executing tools when asked.

    Yields bare text tokens for the final answer, plus JSON status frames of
    the shape ``{"type": "tool", "tool": <name>, "label": <中文>}`` whenever a
    tool is about to be executed. Once ``query_faq`` has run, a
    ``{"type": "sources", "sources": [...]}`` frame is yielded so the client
    can map ``[n]`` citations back to chunks.

    ``llm.stream`` / ``llm.bind_tools(...).stream`` are expected to yield
    objects whose ``.content`` attribute holds the text delta.
    ``self_check_llm`` is a temperature-0 model used for the evidence gate;
    ``None`` falls back to ``llm`` for backwards compatibility.
    """
    conversation = get_or_create_conversation(session, user_id)
    cid = conversation.id

    history = load_history(session, cid)
    messages = build_messages(
        SystemMessage(content=system_prompt),
        history,
        HumanMessage(content=message),
        max_tokens,
    )
    append_message(session, cid, "user", message)
    session.commit()

    # Knowledge-only turns deterministically force ``query_faq`` on the first
    # call, so the model cannot answer from history without consulting the
    # knowledge base. Mixed/order turns keep automatic tool choice.
    force_faq = is_pure_knowledge_question(message)
    bound = None
    if force_faq:
        try:
            bound = llm.bind_tools(tools, tool_choice=_FORCED_FAQ_TOOL_CHOICE)
        except Exception:  # noqa: BLE001 - provider may reject tool_choice
            logger.exception(
                "provider rejected forced query_faq tool choice at bind time; "
                "falling back to automatic tool choice"
            )
            force_faq = False
    if bound is None:
        bound = llm.bind_tools(tools)

    try:
        ai_msg = bound.invoke(messages)
    except Exception:
        if not force_faq:
            raise
        logger.exception(
            "provider rejected forced query_faq tool choice at call time; "
            "retrying with automatic tool choice"
        )
        force_faq = False
        bound = llm.bind_tools(tools)
        ai_msg = bound.invoke(messages)

    if not getattr(ai_msg, "tool_calls", None):
        if force_faq:
            logger.warning(
                "forced query_faq tool choice produced no tool call; "
                "answering without knowledge sources"
            )
        full = ""
        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", None) or ""
            if text:
                full += text
                yield text
        append_message(session, cid, "assistant", full)
        session.commit()
        return

    if force_faq and not any(
        tc["name"] == "query_faq" for tc in ai_msg.tool_calls
    ):
        logger.warning(
            "forced query_faq tool choice was not honored; "
            "no knowledge sources for this turn"
        )

    # One collector per turn; tools register their evidence on this mutable
    # object. It is set before the loop and re-set right before each call.
    collector = CitationCollector()
    current_citation_collector.set(collector)

    # Execute every tool first, persisting nothing yet. This keeps the caller
    # session free of uncommitted writes while a tool (e.g. create_ticket)
    # opens its own session, and lets the whole round commit atomically below.
    results: list[tuple[dict, str]] = []
    for tc in ai_msg.tool_calls:
        yield json.dumps(
            {"type": "tool", "tool": tc["name"], "label": TOOL_LABELS.get(tc["name"], tc["name"])},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        tool = TOOLS_BY_NAME.get(tc["name"])
        if tool is None:
            results.append((tc, f"工具 {tc['name']} 不存在,请告知用户改用人工客服。"))
            continue
        current_conversation_id.set(cid)
        current_citation_collector.set(collector)
        results.append(
            (
                tc,
                execute_tool(
                    tool,
                    tc["args"],
                    timeout_s=tool_timeout(tc["name"]),
                    retries=0 if tc["name"] == "query_faq" else 1,
                ),
            )
        )

    # Persist the assistant tool_calls row and its result rows atomically, so
    # an interruption can never leave a dangling tool_calls with no results.
    append_message(session, cid, "assistant", tool_calls=ai_msg.tool_calls)
    tool_messages: list[ToolMessage] = []
    for tc, result in results:
        append_message(session, cid, "tool", result, tool_call_id=tc["id"])
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    session.commit()

    final_messages = [*messages, ai_msg, *tool_messages]
    sources: list[dict] = []

    if collector.faq_called:
        # Snapshot after the tool loop: late writes from a worker thread must
        # not mutate the frame after it has been emitted.
        sources = collector.payload()
        yield json.dumps(
            {"type": "sources", "sources": sources},
            ensure_ascii=False,
            separators=(",", ":"),
        )

        if not sources and _faq_failed(results):
            # Knowledge base outage, not a knowledge gap: refuse without
            # writing to the low-confidence pool.
            logger.warning(
                "query_faq infrastructure failure; refusing without recording "
                "a low-confidence question"
            )
            yield _INFRA_REFUSAL_TEXT
            _persist_assistant(session, cid, _INFRA_REFUSAL_TEXT)
            return

        try:
            assessment = assess_evidence(self_check_llm or llm, message, sources)
        except Exception:  # noqa: BLE001 - infra failure must not end the turn
            logger.exception(
                "evidence self-check failed; refusing conservatively without "
                "recording a low-confidence question"
            )
            if _has_other_tool_results(results):
                # No pool row is written here, so use the constraint without
                # the "recorded in the low-confidence pool" clause.
                final_messages.append(
                    SystemMessage(content=_EVIDENCE_GAP_CONSTRAINT)
                )
            else:
                yield _SELF_CHECK_FAILURE_REFUSAL_TEXT
                _persist_assistant(session, cid, _SELF_CHECK_FAILURE_REFUSAL_TEXT)
                return
        else:
            if not assessment.useful:
                try:
                    record_low_confidence(
                        session,
                        cid,
                        message,
                        assessment.reason,
                        source="self_check",
                    )
                except Exception:  # noqa: BLE001 - a pool write must not break chat
                    logger.exception("failed to record low-confidence question")
                    session.rollback()

                if _has_other_tool_results(results):
                    final_messages.append(
                        SystemMessage(content=_MIXED_INTENT_CONSTRAINT)
                    )
                else:
                    yield _LOW_CONFIDENCE_REFUSAL_TEXT
                    _persist_assistant(session, cid, _LOW_CONFIDENCE_REFUSAL_TEXT)
                    return

    # The forced choice applies to the initial call only: stream the final
    # answer under automatic tool choice so it cannot force another tool call.
    final_bound = llm.bind_tools(tools) if force_faq else bound

    final_full = ""
    for chunk in final_bound.stream(final_messages):
        text = getattr(chunk, "content", None) or ""
        if text:
            final_full += text
            yield text

    # Single-round only: if the model produced no text (with or without asking
    # for another tool round), emit a fallback so the turn always has a reply.
    if not final_full:
        yield _FALLBACK_TEXT
        final_full = _FALLBACK_TEXT

    if collector.faq_called:
        valid_numbers = {source["n"] for source in sources}
        cited_numbers = {int(number) for number in _CITATION_RE.findall(final_full)}
        unknown = sorted(cited_numbers - valid_numbers)
        if unknown:
            # Log only: the stream cannot be rolled back, and the frontend
            # renders unknown citations as plain text.
            logger.warning("unknown citation numbers in final answer: %s", unknown)

    append_message(session, cid, "assistant", final_full)
    session.commit()
