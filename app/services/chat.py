"""Tool-calling chat orchestration backed by the DB conversation history.

``stream_chat`` wires the pieces from earlier tasks together:

1. Resolve (or create) the active conversation for ``user_id``.
2. Set the ``current_conversation_id`` contextvar so tools like
   ``create_ticket`` can read it (the executor propagates it into its
   worker thread via ``copy_context``).
3. Rebuild history, assemble the prompt within the token budget, and
   persist the user turn before hitting the model.
4. Bind tools and ask the model once. If it emits tool calls, execute them,
   persist the tool round, feed the results back, and stream the final
   answer. Single round only — no looping.
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

    token = current_conversation_id.set(cid)
    try:
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

        append_message(session, cid, "assistant", tool_calls=ai_msg.tool_calls)
        tool_messages: list[ToolMessage] = []
        for tc in ai_msg.tool_calls:
            yield json.dumps(
                {"type": "tool", "tool": tc["name"], "label": TOOL_LABELS[tc["name"]]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            result = execute_tool(TOOLS_BY_NAME[tc["name"]], tc["args"])
            append_message(session, cid, "tool", result, tool_call_id=tc["id"])
            tool_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        final_messages = [*messages, ai_msg, *tool_messages]
        final_full = ""
        final_tool_calls: list = []
        for chunk in bound.stream(final_messages):
            text = getattr(chunk, "content", None) or ""
            if text:
                final_full += text
                yield text
            tcs = getattr(chunk, "tool_calls", None)
            if tcs:
                final_tool_calls.extend(tcs)

        # Single-round only: if the model wanted another tool round and said
        # nothing, emit a fallback rather than looping.
        if not final_full and final_tool_calls:
            yield _FALLBACK_TEXT
            final_full = _FALLBACK_TEXT

        append_message(session, cid, "assistant", final_full)
        session.commit()
    finally:
        current_conversation_id.reset(token)
