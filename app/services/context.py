import json
import logging
from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages as lc_trim_messages,
)

from app.config import Settings

_logger = logging.getLogger(__name__)

#: Per-message overhead used by :func:`message_tokens` (role marker + separators).
_ROLE_TOKENS = 4
#: Historical ``SystemMessage`` rows are truncated to this many characters in L2.
_L2_SYSTEM_PREVIEW_CHARS = 40


def estimate_tokens(text: str | None) -> int:
    """Conservative token estimate: one character counts as one token.

    ``None`` content (e.g. an assistant tool-call row with a NULL ``content``)
    counts as zero tokens.
    """
    if text is None:
        return 0
    return len(text)


def _partition_blocks(history: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """Split history into atomic blocks in forward order.

    An ``AIMessage`` carrying tool calls absorbs the ``ToolMessage``(s) that
    immediately follow it, so a tool call is never separated from its result.
    """
    blocks: list[list[BaseMessage]] = []
    for message in history:
        if (
            isinstance(message, ToolMessage)
            and blocks
            and isinstance(blocks[-1][0], AIMessage)
            and blocks[-1][0].tool_calls
        ):
            blocks[-1].append(message)
        else:
            blocks.append([message])
    return blocks


def _keep_newest_blocks(
    blocks: Sequence[Sequence[BaseMessage]],
    budget: int,
    *,
    inclusive: bool,
    always_keep_newest: bool = False,
) -> list[BaseMessage]:
    """Keep the newest whole blocks that fit in *budget*.

    ``inclusive=True`` allows a block whose cost makes the total exactly equal
    to *budget*; ``inclusive=False`` requires the total to stay strictly below
    it. Either way a block is kept or dropped atomically.

    ``always_keep_newest=True`` retains the newest block unconditionally, even
    when it alone exceeds *budget*. The returned total may then exceed the
    budget; only older blocks are trimmed.
    """
    kept_blocks: list[list[BaseMessage]] = []
    used = 0
    for index, block in enumerate(reversed(blocks)):
        cost = sum(estimate_tokens(m.content) for m in block)
        if index == 0 and always_keep_newest:
            kept_blocks.append(list(block))
            used += cost
            continue
        fits = used + cost <= budget if inclusive else used + cost < budget
        if not fits:
            break
        kept_blocks.append(list(block))
        used += cost
    kept_blocks.reverse()
    return [message for block in kept_blocks for message in block]


def trim_messages(
    system: SystemMessage,
    history: Sequence[BaseMessage],
    max_tokens: int,
) -> list[BaseMessage]:
    """Trim *history* so the assembled prompt stays under *max_tokens*.

    The result is ``[system] + newest_kept_blocks``. ``system`` is always
    first and is never dropped, even when it alone exceeds the budget. History
    blocks are atomic: a tool call and its tool result stay together.

    The newest history block (the current user question or the latest tool
    round) is always retained, even when it alone exceeds the budget, so the
    Agent never loses the question or the freshest tool results. Older blocks
    are kept strictly below ``max_tokens`` when possible; the prompt may
    therefore exceed ``max_tokens`` when the newest block is oversized.
    """
    budget = max_tokens - estimate_tokens(system.content)
    blocks = _partition_blocks(history)
    kept = _keep_newest_blocks(
        blocks, budget, inclusive=False, always_keep_newest=True
    )
    return [system, *kept]


def build_agent_messages(
    system: SystemMessage,
    messages: Sequence[BaseMessage],
    max_tokens: int,
) -> list[BaseMessage]:
    """Assemble the Agent prompt: *system* first plus trimmed *messages*."""
    return trim_messages(system, messages, max_tokens)


def build_messages(
    system: SystemMessage,
    history: list[BaseMessage],
    current: HumanMessage,
    max_tokens: int,
) -> list[BaseMessage]:
    """Assemble the prompt within a token budget.

    The result is ``[system] + trimmed_history + [current]``. ``system`` and
    ``current`` are never dropped, even when they alone exceed the budget.

    History is trimmed atomically: each ordinary message forms its own block,
    while an ``AIMessage`` carrying tool calls absorbs the ``ToolMessage``(s)
    that immediately follow it. A whole block is either kept or dropped, so a
    tool call is never left without its tool result.
    """
    budget = max_tokens - estimate_tokens(system.content) - estimate_tokens(current.content)
    blocks = _partition_blocks(history)
    kept = _keep_newest_blocks(blocks, budget, inclusive=True)
    return [system, *kept, current]


_ROLE_LABELS = {
    HumanMessage: "用户",
    AIMessage: "客服",
    SystemMessage: "系统",
}


def render_recent_dialogue(
    messages: Sequence[BaseMessage],
    max_chars: int = 1200,
) -> str:
    """Render recent user/assistant/system text in chronological order.

    Only text content is kept: ``ToolMessage`` is skipped, as are assistant
    rows with no text (e.g. pure tool calls) and empty messages. Messages are
    walked newest-first within a *max_chars* budget, so older lines are dropped
    first. The newest usable line is always kept, even when it alone exceeds
    the budget, because it is the turn the current question refers back to.

    Each line is ``用户：...`` / ``客服：...`` / ``系统：...``; lines are joined
    with newlines. Returns ``""`` when nothing usable is found.
    """
    lines: list[str] = []
    used = 0
    for message in reversed(messages):
        role = _ROLE_LABELS.get(type(message))
        if role is None:
            continue
        content = message.content
        if not isinstance(content, str) or not content.strip():
            continue
        line = f"{role}：{content}"
        cost = len(line) + 1  # + newline separator
        if lines and used + cost > max_chars:
            break
        lines.append(line)
        used += cost
    lines.reverse()
    return "\n".join(lines)


def message_tokens(messages: Sequence[BaseMessage]) -> int:
    """Estimate the prompt cost of *messages*: content tokens + 4 per message."""
    return sum(estimate_tokens(message.content) + _ROLE_TOKENS for message in messages)


def _db_id(message: BaseMessage) -> int | None:
    """Return the numeric part of a ``db-{id}`` message id, or ``None``."""
    message_id = getattr(message, "id", None)
    if not isinstance(message_id, str) or not message_id.startswith("db-"):
        return None
    try:
        return int(message_id[len("db-"):])
    except ValueError:
        return None


def _first_user_after(messages: Sequence[BaseMessage], threshold: int) -> int | None:
    """Index of the first user message whose ``db-`` id is greater than *threshold*."""
    for index, message in enumerate(messages):
        if not isinstance(message, HumanMessage):
            continue
        db_id = _db_id(message)
        if db_id is not None and db_id > threshold:
            return index
    return None


def split_layers(
    messages: Sequence[BaseMessage],
    summary_upto_msg_id: int | None,
    layer1_from_msg_id: int | None,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split *messages* into ``(l2, l1)`` anchored on user messages.

    ``summary_upto_msg_id`` is the last message folded into L3, so L2 starts at
    the first user message with a larger ``db-`` id. ``layer1_from_msg_id`` is
    the last L2 message, so L1 starts at the first user message with a larger
    ``db-`` id. Both boundaries are exclusive: the anchor message itself stays
    in the earlier layer.

    A missing ``layer1_from_msg_id`` means no L2 window is closed yet, so the
    whole history is L1. A missing ``summary_upto_msg_id`` means L2 starts at
    the beginning. Message ids are global autoincrements, never compared against
    the number of messages. If no user anchor can be found at all, the whole
    history is L1.
    """
    items = list(messages)
    if layer1_from_msg_id is None:
        return [], items
    l2_threshold = summary_upto_msg_id if summary_upto_msg_id is not None else 0
    l2_start = _first_user_after(items, l2_threshold)
    l1_start = _first_user_after(items, layer1_from_msg_id)
    if l2_start is None or l1_start is None:
        return [], items
    return items[l2_start:l1_start], items[l1_start:]


def _string_content(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def render_l2(messages: Sequence[BaseMessage], settings: Settings) -> list[BaseMessage]:
    """Render the derived L2 view, compressing every non-user message.

    User text is kept verbatim. Assistant text keeps only the first
    ``assistant_l2_chars`` characters (plus an ellipsis) and gains a
    ``[调用工具: ...]`` prefix when it carries tool calls. Tool results collapse
    to a one-line marker when longer than ``tool_result_marker_chars``; the tool
    name is recovered from the matching tool call id. Historical system rows keep
    their first 40 characters. Message identity (ids, tool calls, tool call ids)
    is preserved so the rendered history stays a valid chat payload.
    """
    tool_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            call_id = call.get("id")
            if call_id:
                tool_names[call_id] = call.get("name") or ""

    rendered: list[BaseMessage] = []
    for message in messages:
        content = _string_content(message)
        if isinstance(message, HumanMessage):
            rendered.append(message)
        elif isinstance(message, AIMessage):
            prefix = ""
            if message.tool_calls:
                names = ", ".join(call.get("name") or "未知工具" for call in message.tool_calls)
                prefix = f"[调用工具: {names}]"
            text = (
                content[: settings.assistant_l2_chars] + "…" if content else ""
            )
            rendered.append(message.model_copy(update={"content": prefix + text}))
        elif isinstance(message, ToolMessage):
            if len(content) > settings.tool_result_marker_chars:
                name = tool_names.get(message.tool_call_id) or message.name or "未知工具"
                text = f"[工具结果: {name} 已省略 {len(content)} 字]"
            else:
                text = content[: settings.tool_result_marker_chars]
            rendered.append(message.model_copy(update={"content": text}))
        elif isinstance(message, SystemMessage):
            rendered.append(
                message.model_copy(update={"content": content[:_L2_SYSTEM_PREVIEW_CHARS]})
            )
        else:
            rendered.append(message)
    return rendered


def choose_degradation(
    messages: Sequence[BaseMessage],
    layer1_from_msg_id: int | None,
    l1_budget: int,
    low_water_ratio: float,
) -> int | None:
    """Pick the user turn where the degraded L1 should start.

    Walks the user turns of *messages* oldest first and drops whole turns until
    the remaining tail fits in ``l1_budget * low_water_ratio``. The newest turn
    is never dropped, so a single oversized turn is kept as-is. Returns the
    ``db-`` id of the first kept turn, or ``None`` when nothing needs dropping
    (or only one anchored turn exists).
    """
    items = list(messages)
    turns: list[tuple[int, int]] = []
    for index, message in enumerate(items):
        if not isinstance(message, HumanMessage):
            continue
        db_id = _db_id(message)
        if db_id is None:
            continue
        if layer1_from_msg_id is not None and db_id < layer1_from_msg_id:
            continue
        turns.append((index, db_id))
    if len(turns) < 2:
        return None

    target_tokens = l1_budget * low_water_ratio
    for position, (index, db_id) in enumerate(turns):
        if message_tokens(items[index:]) <= target_tokens:
            return None if position == 0 else db_id
    return turns[-1][1]


def lc_trim_l1(messages: Sequence[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """LangChain backstop trim for L1: keep the newest messages on a user boundary.

    Falls back to "the last HumanMessage onward" when LangChain trims everything
    (a single latest message exceeds the budget), and to an empty list when the
    history has no user message at all.
    """
    items = list(messages)
    if not items:
        return []
    trimmed = lc_trim_messages(
        items,
        max_tokens=max_tokens,
        token_counter=message_tokens,
        strategy="last",
        start_on="human",
        include_system=False,
    )
    if trimmed:
        return trimmed
    for index in range(len(items) - 1, -1, -1):
        if isinstance(items[index], HumanMessage):
            return items[index:]
    return []


def assemble_agent_prompt(
    system: SystemMessage,
    history: Sequence[BaseMessage],
    current: Sequence[BaseMessage],
    material: str | None,
    max_prompt_tokens: int,
) -> list[BaseMessage]:
    """Build the model prompt: system + history + current turn (+ material).

    The per-turn material (summary, evidence, order data, intent) is inserted as
    a ``HumanMessage`` right after the first user message of the current turn,
    so it never becomes a system prompt or precedes the user question. When the
    assembled prompt exceeds *max_prompt_tokens*, only *history* is trimmed with
    the LangChain backstop; the system message, the current turn and the material
    are always kept. A warning is logged if the prompt is still over budget.
    """
    current_messages = list(current)
    if material:
        insert_at = next(
            (
                index + 1
                for index, message in enumerate(current_messages)
                if isinstance(message, HumanMessage)
            ),
            len(current_messages),
        )
        current_messages.insert(insert_at, HumanMessage(content=material))

    prompt = [system, *history, *current_messages]
    if message_tokens(prompt) <= max_prompt_tokens:
        return prompt

    fixed_tokens = message_tokens([system, *current_messages])
    history_budget = max(0, max_prompt_tokens - fixed_tokens)
    prompt = [system, *lc_trim_l1(history, history_budget), *current_messages]
    if message_tokens(prompt) > max_prompt_tokens:
        _logger.warning(
            "prompt over budget: tokens=%d > max_prompt_tokens=%d",
            message_tokens(prompt),
            max_prompt_tokens,
        )
    return prompt


def log_context(
    logger: logging.Logger,
    tag: str,
    summary: str,
    messages: Sequence[BaseMessage],
    window_count: int,
    preview_chars: int = 40,
) -> None:
    """Log a one-line JSON snapshot of the prompt window under *tag*."""
    payload = {
        "summary": summary,
        "window_count": window_count,
        "tokens": message_tokens(messages),
        "messages": [
            {"role": type(message).__name__, "preview": str(message.content)[:preview_chars]}
            for message in messages
        ],
    }
    logger.info("%s %s", tag, json.dumps(payload, ensure_ascii=False))

