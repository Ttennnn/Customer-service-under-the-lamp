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
