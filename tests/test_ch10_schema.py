from pathlib import Path

import pytest

from app.models import LowConfidenceQuestion, TopicClassification


def test_ch10_ddl_is_verbatim_user_ddl():
    sql = Path("sql/ch10-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4;" in sql
    assert "CREATE TABLE topic_classifications" in sql
    assert "BIGINT UNSIGNED" in sql
    assert "labels        JSON            NOT NULL" in sql
    assert "UNIQUE KEY uk_question_id (question_id)" in sql
    assert "CONSTRAINT fk_topic_question FOREIGN KEY (question_id)" in sql
    assert "COMMENT='主题分类结果'" in sql


def test_orm_roundtrip_and_unique_question(session):
    q = LowConfidenceQuestion(raw_question="买大了想退", source="self_check")
    session.add(q)
    session.commit()
    session.refresh(q)

    row = TopicClassification(question_id=q.id, labels=["尺码", "退换货"])
    session.add(row)
    session.commit()
    session.refresh(row)
    assert row.id > 0
    assert row.labels == ["尺码", "退换货"]
    assert row.classified_at is not None

    session.add(TopicClassification(question_id=q.id, labels=["物流"]))
    with pytest.raises(Exception):
        session.commit()
    session.rollback()


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


def test_apply_ch10_creates_then_noops(monkeypatch):
    import app.db.init_ch10 as init_ch10

    monkeypatch.setattr(init_ch10, "_existing_table", lambda conn: False)
    conn = _FakeConn()
    init_ch10.apply_ch10(_FakeEngine(conn))
    assert len(conn.driver_executed) == len(init_ch10._load_statements())

    monkeypatch.setattr(init_ch10, "_existing_table", lambda conn: True)
    conn2 = _FakeConn()
    init_ch10.apply_ch10(_FakeEngine(conn2))
    assert conn2.executed == []
