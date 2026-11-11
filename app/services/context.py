from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: one character counts as one token."""
    return len(text)


def build_messages(
    system: SystemMessage,
    history: list[BaseMessage],
    current: HumanMessage,
    max_tokens: int,
) -> list[BaseMessage]:
    """Assemble the prompt within a token budget.

    The result is ``[system] + trimmed_history + [current]``. When the
    estimated total exceeds ``max_tokens``, history messages are dropped
    oldest-first. ``system`` and ``current`` are never dropped, even when
    they alone exceed the budget.
    """
    budget = max_tokens - estimate_tokens(system.content) - estimate_tokens(current.content)

    kept: list[BaseMessage] = []
    used = 0
    for message in reversed(history):
        cost = estimate_tokens(message.content)
        if used + cost > budget:
            break
        kept.append(message)
        used += cost
    kept.reverse()

    return [system, *kept, current]
