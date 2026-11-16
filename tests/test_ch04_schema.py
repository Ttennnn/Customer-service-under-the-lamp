from app.db.init_ch04 import TABLES, _load_statements
from app.models import FaithCase, LowConfidenceQuestion


def test_ddl_contains_both_tables_and_utf8_statement():
    sql = "\n".join(_load_statements())
    assert "CREATE TABLE low_confidence_questions" in sql
    assert "CREATE TABLE faith_cases" in sql
    assert "SET NAMES utf8mb4" in sql
    assert TABLES == ("low_confidence_questions", "faith_cases")


def test_load_statements_drops_comments_and_empty_statements():
    statements = _load_statements()
    assert len(statements) == len(TABLES) + 1  # SET NAMES + one CREATE per table
    assert all(not statement.startswith("--") for statement in statements)


def test_low_confidence_question_roundtrip(session):
    row = LowConfidenceQuestion(
        conversation_id=None, raw_question="支持以旧换新吗", source="self_check",
        reason="证据不足",
    )
    session.add(row); session.commit(); session.refresh(row)
    assert row.id > 0 and row.source == "self_check" and row.created_at is not None


def test_faith_case_json_and_enum(session):
    row = FaithCase(
        eval_id="A01", bucket="A_policy", query="退款多久", strategy="hybrid_rerank",
        answer="答案", reason="编造", citations=[{"n": 1, "chunk_id": 3}],
        judge_model="deepseek-chat",
    )
    session.add(row); session.commit(); session.refresh(row)
    assert row.status == "未解决" and row.seen_count == 1
    assert row.citations == [{"n": 1, "chunk_id": 3}]


class _FakeConn:
    def __init__(self):
        self.executed = []

    def execute(self, statement):
        self.executed.append(str(statement))


class _FakeEngine:
    def __init__(self, conn):
        self._conn = conn

    def begin(self):
        conn = self._conn

        class _Context:
            def __enter__(self):
                return conn

            def __exit__(self, *exc_info):
                return False

        return _Context()


def test_apply_ch04_creates_both_tables_when_absent(monkeypatch):
    import app.db.init_ch04 as init_ch04

    monkeypatch.setattr(init_ch04, "_existing_tables", lambda conn: set())
    conn = _FakeConn()
    init_ch04.apply_ch04(_FakeEngine(conn))

    assert len(conn.executed) == len(init_ch04._load_statements())
    joined = "\n".join(conn.executed)
    assert "CREATE TABLE low_confidence_questions" in joined
    assert "CREATE TABLE faith_cases" in joined


def test_apply_ch04_noops_when_both_tables_exist(monkeypatch):
    import app.db.init_ch04 as init_ch04

    monkeypatch.setattr(init_ch04, "_existing_tables", lambda conn: set(TABLES))
    conn = _FakeConn()
    init_ch04.apply_ch04(_FakeEngine(conn))

    assert conn.executed == []


def test_apply_ch04_raises_on_partial_schema(monkeypatch):
    import pytest

    import app.db.init_ch04 as init_ch04

    monkeypatch.setattr(
        init_ch04, "_existing_tables", lambda conn: {"low_confidence_questions"}
    )
    conn = _FakeConn()
    with pytest.raises(RuntimeError, match="partially initialized"):
        init_ch04.apply_ch04(_FakeEngine(conn))
    assert conn.executed == []
