"""Shared pytest fixtures.

Provides a file-backed SQLite ``session`` fixture and monkeypatches
``app.db.session.get_session_factory`` so later tools that open their own
session (possibly in a worker thread) hit the same database file.

We deliberately use a FILE-backed SQLite DB (not in-memory ``StaticPool``):
a single shared in-memory connection would collide across those separate
sessions/threads, whereas a file DB gives each session its own connection to
the same file.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base

# The local .env carries ch07 demo overrides. Importing pymilvus during test
# collection calls load_dotenv(), which copies them into os.environ and
# defeats Settings(_env_file=None). Clear the demo knobs so tests stay
# hermetic; tests that exercise env overrides set them back via monkeypatch.
_CH07_DEMO_ENV = (
    "MAX_AGENT_STEPS",
    "TOOL_RESULT_MAX_TOKENS",
    "RERANK_TOP_K",
    "MODEL_CONTEXT_WINDOW",
    "MAX_OUTPUT_TOKENS",
    "MAX_USER_INPUT_TOKENS",
    "SAFETY_MARGIN_RATIO",
    "SYSTEM_PROMPT_RESERVE_TOKENS",
    "EVIDENCE_TOKENS_PER_CHUNK",
    "SUMMARY_RESERVE_TOKENS",
    "HISTORY_TARGET_TURNS",
    "HISTORY_TOKENS_PER_TURN",
    "LAYER1_RATIO",
    "LAYER1_DEGRADE_LOW_WATER_RATIO",
    "ASSISTANT_L2_CHARS",
    "TOOL_RESULT_MARKER_CHARS",
)


@pytest.fixture(autouse=True)
def _clear_local_demo_env(monkeypatch):
    for name in _CH07_DEMO_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def session(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test.db",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)

    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr("app.db.session.get_session_factory", lambda: TestSession)

    db = TestSession()
    try:
        yield db
    finally:
        db.close()
