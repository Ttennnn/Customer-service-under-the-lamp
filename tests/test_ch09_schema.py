from pathlib import Path

from app.db.sql_utils import split_statements
from app.models import EvalRun, LowConfidenceQuestion, ReviewQueue


def test_ch09_ddl_has_required_tables_columns_and_fk():
    sql = Path("sql/ch09-ddl.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE review_queue" in sql
    assert "CREATE TABLE eval_runs" in sql
    assert "retrieved_chunks" in sql
    assert "matched_review_id" in sql
    assert "fk_lcq_review" in sql


def test_ch09_ddl_keeps_user_types_and_comments():
    sql = Path("sql/ch09-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4;" in sql
    assert "ENUM('待审','通过','驳回')" in sql
    assert "ENUM('定时','手动')" in sql
    assert "BIGINT UNSIGNED" in sql
    assert "COMMENT='飞轮待审队列'" in sql
    assert "ON DELETE SET NULL" in sql


def test_split_statements_handles_comment_semicolons():
    sql = "CREATE TABLE t (c INT COMMENT 'a;b'); ALTER TABLE t ADD COLUMN d INT;"
    assert len(split_statements(sql)) == 2


def test_orm_tables_can_be_created(session):
    assert session.query(ReviewQueue).count() == 0
    assert session.query(EvalRun).count() == 0
    assert LowConfidenceQuestion.__table__.c.retrieved_chunks is not None
    assert LowConfidenceQuestion.__table__.c.matched_review_id is not None


def test_review_queue_defaults_and_roundtrip(session):
    row = ReviewQueue(normalized_question="支持以旧换新吗")
    session.add(row)
    session.commit()
    session.refresh(row)
    assert row.id > 0
    assert row.occurrence_count == 1
    assert row.review_status == "待审"
    assert row.ai_suggested_answer is None
    assert row.created_at is not None and row.updated_at is not None


def test_eval_run_json_and_enum(session):
    row = EvalRun(
        triggered_by="手动",
        dataset_size=300,
        metrics={"recall_at_k": 0.82, "mrr": 0.71},
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    assert row.id > 0
    assert row.triggered_by == "手动"
    assert row.metrics == {"recall_at_k": 0.82, "mrr": 0.71}
    assert row.created_at is not None


class _FakeConn:
    def __init__(self):
        self.executed = []
        self.driver_executed = []

    def execute(self, statement):
        self.executed.append(str(statement))

    def exec_driver_sql(self, statement):
        self.driver_executed.append(statement)
        self.executed.append(statement)


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


def test_apply_ch09_creates_everything_when_absent(monkeypatch):
    import app.db.init_ch09 as init_ch09

    monkeypatch.setattr(init_ch09, "_existing_tables", lambda conn: set())
    monkeypatch.setattr(init_ch09, "_existing_columns", lambda conn: set())
    conn = _FakeConn()
    init_ch09.apply_ch09(_FakeEngine(conn))

    statements = init_ch09._load_statements()
    assert len(conn.executed) == len(statements)
    assert len(conn.driver_executed) == len(statements)
    # The JSON example in eval_runs.metrics must reach the driver as raw SQL,
    # not through text() (which would treat ":0.82" as a bind parameter).
    assert any("0.82" in statement for statement in conn.driver_executed)
    joined = "\n".join(conn.executed)
    assert "CREATE TABLE review_queue" in joined
    assert "CREATE TABLE eval_runs" in joined
    assert "ALTER TABLE low_confidence_questions" in joined


def test_apply_ch09_noops_when_schema_is_complete(monkeypatch):
    import app.db.init_ch09 as init_ch09

    monkeypatch.setattr(
        init_ch09, "_existing_tables", lambda conn: set(init_ch09.TABLES)
    )
    monkeypatch.setattr(
        init_ch09,
        "_existing_columns",
        lambda conn: set(init_ch09.NEW_COLUMNS["low_confidence_questions"]),
    )
    conn = _FakeConn()
    init_ch09.apply_ch09(_FakeEngine(conn))

    assert conn.executed == []


def test_apply_ch09_raises_on_partial_schema(monkeypatch):
    import pytest

    import app.db.init_ch09 as init_ch09

    monkeypatch.setattr(
        init_ch09, "_existing_tables", lambda conn: {"review_queue"}
    )
    monkeypatch.setattr(init_ch09, "_existing_columns", lambda conn: set())
    conn = _FakeConn()
    with pytest.raises(RuntimeError, match="partially initialized"):
        init_ch09.apply_ch09(_FakeEngine(conn))
    assert conn.executed == []
