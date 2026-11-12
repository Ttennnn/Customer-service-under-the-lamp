from datetime import datetime
from types import SimpleNamespace

import pytest

from app.models import Conversation, KnowledgeChunk, Message, QaExtractionStaging
from app.jobs import mine_qa as mine_qa_module
from app.jobs.mine_qa import (
    MAX_TRANSCRIPT_MESSAGES,
    _truncate_messages,
    load_transcripts,
    mine_qa,
)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema, method=None):
        llm = self

        class S:
            def invoke(self, messages):
                llm.calls += 1
                return schema(items=[
                    {"question": "邮费是多少", "answer": "满99包邮", "category": "物流"},
                    {"question": "邮费是多少?", "answer": "满99包邮", "category": "物流"},
                ])

        return S()


class FakeSemanticLLM(FakeLLM):
    """Two non-identical questions that the fake embedder maps together."""

    def with_structured_output(self, schema, method=None):
        llm = self

        class S:
            def invoke(self, messages):
                llm.calls += 1
                return schema(items=[
                    {"question": "邮费是多少", "answer": "满99包邮", "category": "物流"},
                    {"question": "邮费怎么算", "answer": "满99包邮", "category": "物流"},
                ])

        return S()


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


class FakeStore:
    def __init__(self):
        self.items = []

    def upsert(self, items):
        self.items.extend(items)


def _conv(session, uid, *, updated_at=None, turns=None):
    c = Conversation(user_id=uid, status="已结束")
    if updated_at is not None:
        c.updated_at = updated_at
    session.add(c)
    session.flush()
    if turns is None:
        turns = [
            ("user", "邮费是多少"),
            ("assistant", "满99包邮"),
        ]
    session.add_all([
        Message(conversation_id=c.id, role=role, content=content)
        for role, content in turns
    ])
    session.commit()
    return c


def test_mine_dedups_and_inserts(session):
    _conv(session, "u1")
    r = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    assert r.extracted == 2 and r.kept == 1 and r.discarded == 1 and r.inserted == 1
    assert session.query(QaExtractionStaging).count() == 2
    assert session.query(QaExtractionStaging).filter_by(status="discarded").count() == 1
    assert session.query(KnowledgeChunk).filter_by(content_type="mined_qa").count() == 1


def test_rerun_skips_already_mined(session):
    _conv(session, "u1")
    mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    r = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    assert r.extracted == 0 and r.inserted == 0


def test_dry_run_writes_nothing(session):
    _conv(session, "u1")
    r = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore(), dry_run=True)
    assert r.kept == 1
    assert session.query(QaExtractionStaging).count() == 0
    assert session.query(KnowledgeChunk).count() == 0


def test_stale_extracted_rows_are_cleared_before_re_extract(session):
    _conv(session, "u1")
    session.add(QaExtractionStaging(batch_no="old", source_ref="1", question="旧", answer="旧"))
    session.commit()
    mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore())
    assert session.query(QaExtractionStaging).filter_by(batch_no="old").count() == 0


# --- supplementary contract tests ---

def test_non_default_dedup_threshold_is_forwarded(session, monkeypatch):
    _conv(session, "u")
    seen = []
    real = mine_qa_module.dedup_questions_with_stats

    def spy(candidates, existing, embedder, threshold=0.92):
        seen.append(threshold)
        return real(candidates, existing, embedder, threshold=threshold)

    monkeypatch.setattr(mine_qa_module, "dedup_questions_with_stats", spy)
    mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore(), threshold=0.42)
    assert seen == [0.42]


def test_truncate_messages_keeps_head_and_tail():
    messages = [
        ("user" if index % 2 == 0 else "assistant", f"m{index}")
        for index in range(60)
    ]
    truncated = _truncate_messages(messages, max_messages=10)
    assert truncated == messages[:5] + messages[-5:]
    assert _truncate_messages(messages[:10], max_messages=10) == messages[:10]


def test_load_transcripts_truncates_overlong_conversation(session):
    turns = [
        ("user" if index % 2 == 0 else "assistant", f"m{index}")
        for index in range(60)
    ]
    c = _conv(session, "long", turns=turns)

    (transcript,) = load_transcripts(session)
    assert transcript.source_ref == str(c.id)
    assert len(transcript.messages) == MAX_TRANSCRIPT_MESSAGES
    assert transcript.messages[0] == ("user", "m0")
    assert transcript.messages[-1] == ("assistant", "m59")
    assert ("user", "m30") not in transcript.messages


def test_report_exposes_per_batch_and_exact_discard_counts(session):
    _conv(session, "u1")
    _conv(session, "u2")

    report = mine_qa(
        session, FakeLLM(), FakeEmbedder(), FakeStore(), batch_size=1, do_vectorize=False
    )
    assert [(d.conversations, d.extracted) for d in report.batch_details] == [
        (1, 2),
        (1, 2),
    ]
    assert all(d.batch_no for d in report.batch_details)
    assert report.extracted == 4
    assert report.kept == 1
    # Candidates 2-4 collide exactly with the first normalised question.
    assert report.exact_discarded == 3
    assert report.approximate_discarded == 0


def test_report_counts_approximate_discards(session):
    _conv(session, "u")

    report = mine_qa(
        session, FakeSemanticLLM(), FakeEmbedder(), FakeStore(), do_vectorize=False
    )
    assert report.extracted == 2 and report.kept == 1
    assert report.exact_discarded == 0
    assert report.approximate_discarded == 1


def test_load_transcripts_requires_user_and_assistant_content(session):
    _conv(session, "only-user", turns=[("user", "问")])
    _conv(session, "only-assistant", turns=[("assistant", "答")])
    valid = _conv(session, "valid")

    assert [t.source_ref for t in load_transcripts(session)] == [str(valid.id)]


def test_load_transcripts_serializes_only_user_and_assistant_turns(session):
    c = _conv(session, "u", turns=[
        ("user", "问题"),
        ("tool", "工具结果"),
        ("assistant", "回答"),
    ])

    transcripts = load_transcripts(session)
    assert [t.source_ref for t in transcripts] == [str(c.id)]
    assert transcripts[0].messages == [("user", "问题"), ("assistant", "回答")]


def test_load_transcripts_filters_since_orders_and_limits(session):
    first = _conv(session, "u1", updated_at=datetime(2026, 1, 1))
    second = _conv(session, "u2", updated_at=datetime(2026, 1, 2))
    third = _conv(session, "u3", updated_at=datetime(2026, 1, 3))

    assert [t.source_ref for t in load_transcripts(session)] == [
        str(first.id), str(second.id), str(third.id)
    ]
    assert [t.source_ref for t in load_transcripts(
        session, since=datetime(2026, 1, 2)
    )] == [str(second.id), str(third.id)]
    assert [t.source_ref for t in load_transcripts(session, limit=1)] == [str(first.id)]


def test_load_transcripts_skips_mined_and_force_removes_staging(session):
    c = _conv(session, "u")
    session.add(QaExtractionStaging(
        batch_no="b", source_ref=str(c.id), question="q", answer="a", status="kept"
    ))
    session.commit()

    assert load_transcripts(session) == []
    assert [t.source_ref for t in load_transcripts(session, force=True)] == [str(c.id)]
    assert session.query(QaExtractionStaging).count() == 0


def test_load_transcripts_clears_stale_extracted_before_skip_check(session):
    c = _conv(session, "u")
    session.add(QaExtractionStaging(
        batch_no="old", source_ref=str(c.id), question="旧", answer="旧", status="extracted"
    ))
    session.commit()

    # A stale extracted row must not make the conversation look already mined.
    assert [t.source_ref for t in load_transcripts(session)] == [str(c.id)]
    assert session.query(QaExtractionStaging).count() == 0


def test_batch_size_controls_batches_and_attributes_sources(session):
    first = _conv(session, "u1")
    second = _conv(session, "u2")
    llm = FakeLLM()

    report = mine_qa(
        session, llm, FakeEmbedder(), FakeStore(), batch_size=1, do_vectorize=False
    )
    assert report.batches == 2 and report.extracted == 4
    assert llm.calls == 2
    refs = {ref for (ref,) in session.query(QaExtractionStaging.source_ref).distinct()}
    assert refs == {str(first.id), str(second.id)}

    rerun = mine_qa(
        session, FakeLLM(), FakeEmbedder(), FakeStore(), batch_size=1, do_vectorize=False
    )
    assert rerun.extracted == 0


def test_one_batch_marks_every_conversation_mined(session):
    first = _conv(session, "u1")
    second = _conv(session, "u2")

    report = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore(), do_vectorize=False)
    assert report.batches == 1 and report.extracted == 2
    refs = {ref for (ref,) in session.query(QaExtractionStaging.source_ref).distinct()}
    assert refs == {str(first.id), str(second.id)}

    rerun = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore(), do_vectorize=False)
    assert rerun.extracted == 0


def test_dry_run_does_not_delete_stale_rows_or_write(session):
    c = _conv(session, "u")
    session.add(QaExtractionStaging(
        batch_no="old", source_ref=str(c.id), question="旧", answer="旧"
    ))
    session.commit()

    report = mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore(), dry_run=True, force=True)
    assert report.extracted == 2 and report.kept == 1 and report.inserted == 0
    assert session.query(QaExtractionStaging).filter_by(batch_no="old").count() == 1
    assert session.query(KnowledgeChunk).count() == 0


def test_no_vectorize_leaves_inserted_rows_pending(session):
    _conv(session, "u")
    store = FakeStore()

    report = mine_qa(session, FakeLLM(), FakeEmbedder(), store, do_vectorize=False)
    assert report.vectorized == 0 and store.items == []
    chunk = session.query(KnowledgeChunk).filter_by(content_type="mined_qa").one()
    assert chunk.vectorize_status == "pending"


def test_force_re_extracts_without_duplicating_knowledge(session):
    _conv(session, "u")
    mine_qa(session, FakeLLM(), FakeEmbedder(), FakeStore(), do_vectorize=False)

    report = mine_qa(
        session, FakeLLM(), FakeEmbedder(), FakeStore(), force=True, do_vectorize=False
    )
    assert report.extracted == 2
    assert report.kept == 0 and report.discarded == 2 and report.inserted == 0
    assert session.query(KnowledgeChunk).filter_by(content_type="mined_qa").count() == 1
    assert session.query(QaExtractionStaging).count() == 2


def _wire_cli(monkeypatch, session, *, store=None):
    from app.jobs import mine_qa as job

    if store is None:
        store = FakeStore()
    llm = FakeLLM()
    monkeypatch.setattr(
        job,
        "get_settings",
        lambda: SimpleNamespace(qa_mining_batch_size=7, qa_dedup_threshold=0.92),
    )
    monkeypatch.setattr(job, "build_llm", lambda settings: llm)
    monkeypatch.setattr(job, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(job, "get_vector_store", lambda: store)
    monkeypatch.setattr(job, "get_session_factory", lambda: (lambda: session))
    return job, llm


def test_cli_mines_with_limit_and_no_vectorize(session, monkeypatch, capsys):
    _conv(session, "u")
    job, _ = _wire_cli(monkeypatch, session)

    assert job.main(["--limit", "1", "--batch-size", "1", "--no-vectorize"]) == 0
    out = capsys.readouterr().out
    assert "batches=1" in out and "extracted=2" in out and "vectorized=0" in out
    assert "conversations=1 extracted=2" in out
    assert "exact_discarded=1" in out and "approximate_discarded=0" in out
    session.expire_all()
    assert session.query(QaExtractionStaging).count() == 2
    assert (
        session.query(KnowledgeChunk)
        .filter_by(content_type="mined_qa", vectorize_status="pending")
        .count()
        == 1
    )


def test_cli_forwards_configured_dedup_threshold(session, monkeypatch):
    _conv(session, "u")
    job, _ = _wire_cli(monkeypatch, session)
    monkeypatch.setattr(
        job,
        "get_settings",
        lambda: SimpleNamespace(qa_mining_batch_size=7, qa_dedup_threshold=0.55),
    )
    seen = []
    real = job.dedup_questions_with_stats

    def spy(candidates, existing, embedder, threshold=0.92):
        seen.append(threshold)
        return real(candidates, existing, embedder, threshold=threshold)

    monkeypatch.setattr(job, "dedup_questions_with_stats", spy)
    assert job.main(["--dry-run", "--force"]) == 0
    assert seen == [0.55]


def test_cli_dry_run_never_touches_store_or_database(session, monkeypatch, capsys):
    _conv(session, "u")
    job, _ = _wire_cli(monkeypatch, session)
    monkeypatch.setattr(
        job, "get_vector_store", lambda: pytest.fail("dry-run must not build a store")
    )

    assert job.main(["--dry-run", "--force", "--since", "2026-01-01"]) == 0
    assert "kept=1" in capsys.readouterr().out
    assert session.query(QaExtractionStaging).count() == 0
    assert session.query(KnowledgeChunk).count() == 0
