"""Ch09 Langfuse v3 observability core.

Tracing is optional: when any of ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_BASE_URL`` is missing the graph is left
untraced and every helper degrades to a no-op.  Failures while importing or
initialising the SDK never reach the request path.

v3 API facts verified against the installed ``langfuse==3.15.0``:

- ``langfuse.Langfuse(public_key=..., secret_key=..., host=...)``.  ``host``
  is accepted (``base_url`` wins when both are set) and the client registers
  itself as the process singleton through ``LangfuseResourceManager``.
- ``langfuse.langchain.CallbackHandler()`` calls ``get_client()``, i.e. it
  binds to that single registered client.
- ``client.update_current_trace(*, metadata=..., tags=[...])`` is an instance
  method on the client.
- LangChain reads ``langfuse_session_id`` / ``langfuse_user_id`` /
  ``langfuse_tags`` from the run config ``metadata``.

The handler is shared by every request (attached once to the compiled graph
with ``with_config``).  ``compile().with_config(...)`` returns a
``RunnableBinding`` without ``get_state``/``get_state_history``, so every
state read goes through ``_state_graph``; ``stream`` keeps using the binding
so the callback fires.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Process-level singletons: one SDK client and one callback handler serve
# every request.  The lock is re-entrant because creating the handler first
# ensures the client exists (CallbackHandler attaches to the default client).
_lock = threading.RLock()
_client: Any = None
_handler: Any = None


def tracing_enabled(settings: Any) -> bool:
    """Whether Langfuse credentials are complete enough to trace."""
    return bool(
        getattr(settings, "langfuse_public_key", "")
        and getattr(settings, "langfuse_secret_key", "")
        and getattr(settings, "langfuse_base_url", "")
    )


def _new_client(settings: Any):
    """Build the Langfuse v3 client (import is lazy so tests stay offline)."""
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
        environment=getattr(settings, "langfuse_environment", None) or None,
    )


def _new_handler():
    """Build the Langfuse LangChain callback handler."""
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def get_langfuse_client(settings: Any):
    """Return the process-level Langfuse client, creating it on first use."""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            _client = _new_client(settings)
        return _client


def get_callback_handler(settings: Any):
    """Return the shared LangChain callback handler.

    Creating the handler requires the default client, so the client is
    initialised first when needed.
    """
    global _handler
    if _handler is not None:
        return _handler
    with _lock:
        if _handler is None:
            get_langfuse_client(settings)
            _handler = _new_handler()
        return _handler


def attach_tracing(graph, settings: Any):
    """Bind the shared callback handler to a compiled graph.

    Returns the original graph when tracing is disabled or when the SDK /
    handler cannot be initialised: observability must never block chat.
    """
    if not tracing_enabled(settings):
        return graph
    try:
        handler = get_callback_handler(settings)
        return graph.with_config({"callbacks": [handler]})
    except Exception:  # noqa: BLE001 - tracing is optional
        logger.warning(
            "failed to attach langfuse tracing; continuing untraced",
            exc_info=True,
        )
        return graph


def _state_graph(graph):
    """Return the underlying stateful graph of a ``with_config`` binding.

    ``RunnableBinding`` forwards ``stream``/``invoke`` but not
    ``get_state``/``get_state_history``; the bound compiled graph has them.
    """
    return getattr(graph, "bound", graph)


def update_intent_trace(settings: Any, intent: str, confidence: float) -> None:
    """Annotate the current trace with the classified intent.

    Never raises into the graph; Langfuse being unreachable must not affect
    the user-facing turn.
    """
    if not tracing_enabled(settings):
        return
    try:
        get_langfuse_client(settings).update_current_trace(
            metadata={"intent": intent, "intent_confidence": confidence},
            tags=[f"intent:{intent}"],
        )
    except Exception:  # noqa: BLE001 - observability is best effort
        logger.warning("langfuse update_current_trace failed", exc_info=True)


def flush_tracing() -> None:
    """Flush pending spans on shutdown, ignoring any SDK failure."""
    client = _client
    if client is None:
        return
    try:
        client.flush()
    except Exception:  # noqa: BLE001 - shutdown must not crash the process
        logger.warning("langfuse flush failed", exc_info=True)
