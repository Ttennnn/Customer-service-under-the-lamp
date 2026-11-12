"""Tool-calling chat orchestration backed by the DB conversation history.

``stream_chat`` wires the pieces from earlier tasks together:

1. Resolve (or create) the active conversation for ``user_id``.
2. Rebuild history, assemble the prompt within the token budget, and
   persist the user turn before hitting the model.
3. Bind tools and ask the model once. If it emits tool calls, execute them
   first (persisting nothing), then persist the assistant tool_calls row and
   its result rows in a single commit, feed the results back, and stream the
   final answer. Single round only — no looping.

``current_conversation_id`` is a thread-local ``ContextVar``. Starlette's
``EventSourceResponse`` iterates this generator via ``iterate_in_threadpool``,
which may call ``next()`` on a different thread each time, so the contextvar
must be set on the SAME thread that calls ``execute_tool`` (immediately before
it, with no ``yield`` in between) rather than once at the top of the
generator. It is intentionally not reset: each tool call re-sets it, and a
cross-thread ``reset`` would raise ``ValueError``.
"""

import json
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from app.services.context import build_messages
from app.services.history import append_message, get_or_create_conversation, load_history
from app.tools.context import current_conversation_id
from app.tools.executor import execute_tool
from app.tools.registry import TOOLS_BY_NAME, TOOL_LABELS

_FALLBACK_TEXT = "抱歉,我暂时无法完成这个操作,请稍后重试或联系人工客服。"


def stream_chat(
    llm,
    session: Session,
    tools: list,
    system_prompt: str,
    user_id: str,
    message: str,
    max_tokens: int,
) -> Iterator[str]:
    """Stream an assistant reply token by token, executing tools when asked.

    Yields bare text tokens for the final answer, plus JSON status frames of
    the shape ``{"type": "tool", "tool": <name>, "label": <中文>}`` whenever a
    tool is about to be executed.

    ``llm.stream`` / ``llm.bind_tools(...).stream`` are expected to yield
    objects whose ``.content`` attribute holds the text delta.
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

    bound = llm.bind_tools(tools)
    ai_msg = bound.invoke(messages)

    if not getattr(ai_msg, "tool_calls", None):
        full = ""
        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", None) or ""
            if text:
                full += text
                yield text
        append_message(session, cid, "assistant", full)
        session.commit()
        return

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
        results.append((tc, execute_tool(tool, tc["args"])))

    # Persist the assistant tool_calls row and its result rows atomically, so
    # an interruption can never leave a dangling tool_calls with no results.
    append_message(session, cid, "assistant", tool_calls=ai_msg.tool_calls)
    tool_messages: list[ToolMessage] = []
    for tc, result in results:
        append_message(session, cid, "tool", result, tool_call_id=tc["id"])
        tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    session.commit()

    final_messages = [*messages, ai_msg, *tool_messages]
    final_full = ""
    for chunk in bound.stream(final_messages):
        text = getattr(chunk, "content", None) or ""
        if text:
            final_full += text
            yield text

    # Single-round only: if the model produced no text (with or without asking
    # for another tool round), emit a fallback so the turn always has a reply.
    if not final_full:
        yield _FALLBACK_TEXT
        final_full = _FALLBACK_TEXT

    append_message(session, cid, "assistant", final_full)
    session.commit()
