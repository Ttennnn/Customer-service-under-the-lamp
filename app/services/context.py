from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def estimate_tokens(text: str | None) -> int:
    """Conservative token estimate: one character counts as one token.

    ``None`` content (e.g. an assistant tool-call row with a NULL ``content``)
    counts as zero tokens.
    """
    if text is None:
        return 0
    return len(text)


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

    # Partition history into atomic blocks in forward order.
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

    # Keep the newest blocks that fit within the remaining budget.
    kept_blocks: list[list[BaseMessage]] = []
    used = 0
    for block in reversed(blocks):
        cost = sum(estimate_tokens(m.content) for m in block)
        if used + cost > budget:
            break
        kept_blocks.append(block)
        used += cost
    kept_blocks.reverse()

    kept_messages = [message for block in kept_blocks for message in block]
    return [system, *kept_messages, current]
