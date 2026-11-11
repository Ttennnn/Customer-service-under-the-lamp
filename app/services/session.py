import threading

from langchain_core.messages import BaseMessage


class SessionStore:
    """In-memory conversation history keyed by session id.

    Suitable for the single-user v1; all state is lost on restart.
    """

    def __init__(self, max_sessions: int = 1000):
        self._max_sessions = max_sessions
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[BaseMessage]:
        """Return a copy of the session history (empty list if unknown)."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, message: BaseMessage) -> None:
        """Append a message, evicting the oldest session when at capacity."""
        with self._lock:
            if session_id not in self._sessions:
                if self._sessions and len(self._sessions) >= self._max_sessions:
                    oldest = next(iter(self._sessions))
                    del self._sessions[oldest]
                self._sessions[session_id] = []
            self._sessions[session_id].append(message)
