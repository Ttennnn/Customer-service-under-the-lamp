"""Per-turn runtime context preparation (chapter 7).

Rebuilds the layered history window for one turn: merges MySQL rows that
predate the checkpoint state, splits L2/L1 on user anchors, degrades L1 by
moving its persisted anchor when it exceeds budget, renders L2, and asks the
background summary job to compress L2 when it exceeds its budget.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.config import Settings
from app.models import Conversation, Message
from app.services.context import (
    choose_degradation,
    estimate_tokens,
    log_context,
    message_tokens,
    render_l2,
    split_layers,
)
from app.services.context_budget import ContextBudget, compute_budget
from app.services.history import (
    list_dialog_messages,
    previous_message_id,
    update_layer1_from,
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextPlan:
    """The layered history window prepared for one turn."""

    summary: str
    summary_upto_msg_id: int | None
    layer1_from_msg_id: int | None
    history_l2: list[BaseMessage]
    history_l1: list[BaseMessage]
    prompt_history: list[BaseMessage]
    turn_start_index: int
    l2_tokens: int
    l1_tokens: int
    budget: ContextBudget
    degraded: bool


def schedule_summary(conversation_id: int, settings: Settings) -> bool:
    """Start the background summary job for *conversation_id*.

    The import is deliberately late so this module stays importable before the
    summary job exists; tests monkeypatch this module-level wrapper.
    """
    from app.services.summary_job import schedule_summary as _schedule

    return _schedule(conversation_id, settings)


def _db_id(message: BaseMessage) -> int | None:
    """Return the numeric part of a ``db-{id}`` message id, or ``None``."""
    message_id = getattr(message, "id", None)
    if not isinstance(message_id, str) or not message_id.startswith("db-"):
        return None
    try:
        return int(message_id[len("db-"):])
    except ValueError:
        return None


def _first_db_id(messages: Sequence[BaseMessage]) -> int | None:
    for message in messages:
        db_id = _db_id(message)
        if db_id is not None:
            return db_id
    return None


def _last_db_id(messages: Sequence[BaseMessage]) -> int | None:
    for message in reversed(messages):
        db_id = _db_id(message)
        if db_id is not None:
            return db_id
    return None


def _row_to_message(row: Message) -> BaseMessage:
    """Convert a persisted user/assistant row to a ``db-{id}`` chat message."""
    message_id = f"db-{row.id}"
    if row.role == "user":
        return HumanMessage(content=row.content or "", id=message_id)
    return AIMessage(content=row.content or "", id=message_id)


def _in_window(
    row: Message, summary_upto_msg_id: int | None, current_msg_id: int | None
) -> bool:
    """True when *row* belongs after the summary and before the current user."""
    if summary_upto_msg_id is not None and row.id <= summary_upto_msg_id:
        return False
    if current_msg_id is not None and row.id >= current_msg_id:
        return False
    return True


def _merge_db_history(
    session,
    conversation: Conversation,
    history: Sequence[BaseMessage],
    summary_upto_msg_id: int | None,
    current_msg_id: int | None,
) -> list[BaseMessage]:
    """Prepend MySQL rows that are missing from the checkpoint state.

    The state only carries messages produced since the ``conv:{id}`` thread
    was created. Whenever its oldest ``db-`` user id is later than the
    conversation's first message id, the user/assistant rows before it are
    loaded and prepended, stopping after the summary anchor and excluding the
    current user row.
    """
    messages = list(history)
    rows = list_dialog_messages(session, conversation.id)
    if not rows:
        return messages

    if not messages:
        return [
            _row_to_message(row)
            for row in rows
            if _in_window(row, summary_upto_msg_id, current_msg_id)
        ]

    oldest_id = next(
        (
            _db_id(message)
            for message in messages
            if isinstance(message, HumanMessage) and _db_id(message) is not None
        ),
        None,
    )
    if oldest_id is None:
        return messages  # no db anchors: keep the state as-is (fallback)

    if oldest_id <= rows[0].id:
        return messages  # the state already reaches the conversation start

    prefix = [
        _row_to_message(row)
        for row in rows
        if row.id < oldest_id and _in_window(row, summary_upto_msg_id, current_msg_id)
    ]
    return prefix + messages


def _summary_upto_id(
    session,
    conversation_id: int,
    l2: Sequence[BaseMessage],
    l1: Sequence[BaseMessage],
    layer1_from_msg_id: int | None,
) -> int | None:
    """Return the upper bound of the L2 batch the summary job should cover."""
    last_l2_id = _last_db_id(l2)
    if layer1_from_msg_id is not None and last_l2_id == layer1_from_msg_id:
        return layer1_from_msg_id
    first_l1_user_id = next(
        (
            _db_id(message)
            for message in l1
            if isinstance(message, HumanMessage) and _db_id(message) is not None
        ),
        None,
    )
    if first_l1_user_id is not None:
        boundary = previous_message_id(session, conversation_id, first_l1_user_id)
        if boundary is not None:
            return boundary
    return last_l2_id if last_l2_id is not None else layer1_from_msg_id


def prepare_context(
    session,
    conversation: Conversation,
    prior_messages: Sequence[BaseMessage],
    turn_start_index: int,
    settings: Settings,
    *,
    current_msg_id: int | None = None,
) -> ContextPlan:
    """Prepare the layered history window for the current turn.

    ``prior_messages`` is the checkpoint state before the current user row;
    ``turn_start_index`` cuts everything from the current turn onward (for a
    resumed turn it is the index of the last stored ``HumanMessage``).
    ``current_msg_id`` keeps the just-persisted user row out of the MySQL
    merge when the stored state does not include it yet.
    """
    summary = conversation.summary or ""
    summary_upto = conversation.summary_upto_msg_id
    layer1_from = conversation.layer1_from_msg_id

    history = list(prior_messages)[:turn_start_index]
    merged = _merge_db_history(
        session, conversation, history, summary_upto, current_msg_id
    )

    l2, l1 = split_layers(merged, summary_upto, layer1_from)
    l1_tokens = message_tokens(l1)
    budget = compute_budget(settings, estimate_tokens(summary))
    degraded = False

    if l1_tokens > budget.layer1 and l1:
        target_user = choose_degradation(
            l1,
            layer1_from,
            budget.layer1,
            settings.layer1_degrade_low_water_ratio,
        )
        if target_user is None:
            _logger.warning(
                "层1 仅剩一轮无法降级 l1_tokens=%d layer1_budget=%d",
                l1_tokens,
                budget.layer1,
            )
        else:
            boundary = previous_message_id(session, conversation.id, target_user)
            update_layer1_from(session, conversation.id, boundary)
            old_anchor = layer1_from
            layer1_from = boundary
            l2, l1 = split_layers(merged, summary_upto, layer1_from)
            new_l1_tokens = message_tokens(l1)
            degraded = True
            _logger.info(
                "层1 降级 %s→%s l1_tokens=%d→%d",
                old_anchor,
                boundary,
                l1_tokens,
                new_l1_tokens,
            )
            l1_tokens = new_l1_tokens
            if l1_tokens > budget.layer1:
                _logger.warning(
                    "层1 降级后仍超预算 l1_tokens=%d layer1_budget=%d",
                    l1_tokens,
                    budget.layer1,
                )

    l2_rendered = render_l2(l2, settings)
    l2_tokens = message_tokens(l2_rendered)

    if l2_tokens > budget.layer2 and l2:
        from_id = _first_db_id(l2)
        upto_id = _summary_upto_id(
            session, conversation.id, l2, l1, layer1_from
        )
        _logger.info(
            "summary trigger conversation=%d from=%s upto=%s "
            "l2_tokens=%d layer2_budget=%d",
            conversation.id,
            from_id,
            upto_id,
            l2_tokens,
            budget.layer2,
        )
        schedule_summary(conversation.id, settings)

    prompt_history = list(l2_rendered) + list(l1)
    log_context(
        _logger,
        "history_ctx",
        summary,
        prompt_history,
        len(prompt_history),
        settings.context_log_preview_chars,
    )

    return ContextPlan(
        summary=summary,
        summary_upto_msg_id=summary_upto,
        layer1_from_msg_id=layer1_from,
        history_l2=l2_rendered,
        history_l1=l1,
        prompt_history=prompt_history,
        turn_start_index=turn_start_index,
        l2_tokens=l2_tokens,
        l1_tokens=l1_tokens,
        budget=budget,
        degraded=degraded,
    )
