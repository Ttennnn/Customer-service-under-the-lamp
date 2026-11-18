from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


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
