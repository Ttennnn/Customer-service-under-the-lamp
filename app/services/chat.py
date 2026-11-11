from collections.abc import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.context import build_messages
from app.services.session import SessionStore


def stream_chat(
    llm,
    store: SessionStore,
    system_prompt: str,
    session_id: str,
    message: str,
    max_tokens: int,
) -> Iterator[str]:
    """Stream an assistant reply token by token and persist the exchange.

    ``llm.stream(messages)`` is expected to yield objects whose ``.content``
    attribute holds the text delta (e.g. LangChain ``AIMessageChunk``).
    """
    history = store.get(session_id)
    messages = build_messages(
        SystemMessage(content=system_prompt),
        history,
        HumanMessage(content=message),
        max_tokens,
    )
    full = ""
    store.append(session_id, HumanMessage(content=message))
    for chunk in llm.stream(messages):
        text = chunk.content
        if text:
            full += text
            yield text
    store.append(session_id, AIMessage(content=full))
