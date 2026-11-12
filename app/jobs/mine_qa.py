"""Mine reusable Q&A pairs from conversations, with staging and global dedup.

The job is deliberately *rerunnable-convergent* rather than resumable: a run
selects conversations that have not been mined yet, deletes half-finished
staging rows left by an interrupted run, re-extracts, deduplicates across all
batches, and finally inserts the surviving pairs into ``knowledge_chunks``.
Insertion is idempotent (content fingerprints), so rerunning after an
interruption never duplicates knowledge.

``qa_extraction_staging`` has no ``category`` column, so the LLM category only
lives in memory for the current run; rows that stay ``kept``/``discarded`` are
traceability records, and a rerun re-extracts instead of resuming.  Because
``extract_qa_batch`` returns one flat list for a whole batch of transcripts
(without per-item provenance), staging rows are attributed round-robin across
that batch's conversations; ``source_ref`` is only a traceability aid and never
reaches the final knowledge chunk.
"""

from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass
from datetime import datetime

from app.config import build_llm, get_settings
from app.db.session import get_session_factory
from app.models import Conversation, KnowledgeChunk, Message, QaExtractionStaging
from app.rag.embedding import get_embedder
from app.rag.ingest import MINED_CONTENT_TYPE, MinedQA, insert_mined_qa
from app.rag.milvus_store import get_vector_store
from app.rag.vectorize import vectorize_pending
from app.services.qa_mining import (
    ConversationTranscript,
    ExtractedQA,
    dedup_questions,
    extract_qa_batch,
)

_BATCH_NO_FORMAT = "qa-%Y%m%d%H%M%S-"


@dataclass(frozen=True)
class MiningReport:
    batches: int
    extracted: int
    kept: int
    discarded: int
    inserted: int
    vectorized: int


def _select_candidates(session, since=None) -> list[ConversationTranscript]:
    """Return mineable conversations, oldest ``updated_at`` first.

    Only conversations with at least one user and one assistant message (both
    with content) qualify.  Ties on ``updated_at`` fall back to conversation
    id so batching and ``--limit`` stay deterministic.
    """
    user_conversations = (
        session.query(Message.conversation_id)
        .filter(Message.role == "user", Message.content.isnot(None))
        .subquery()
    )
    assistant_conversations = (
        session.query(Message.conversation_id)
        .filter(Message.role == "assistant", Message.content.isnot(None))
        .subquery()
    )
    conversation_query = session.query(Conversation).filter(
        Conversation.id.in_(session.query(user_conversations.c.conversation_id)),
        Conversation.id.in_(session.query(assistant_conversations.c.conversation_id)),
    )
    if since is not None:
        conversation_query = conversation_query.filter(Conversation.updated_at >= since)
    conversations = conversation_query.order_by(
        Conversation.updated_at, Conversation.id
    ).all()
    if not conversations:
        return []

    conversation_ids = [conversation.id for conversation in conversations]
    messages = (
        session.query(Message)
        .filter(
            Message.conversation_id.in_(conversation_ids),
            Message.role.in_(("user", "assistant")),
            Message.content.isnot(None),
        )
        .order_by(Message.id)
        .all()
    )
    turns_by_conversation: dict[int, list[tuple[str, str]]] = {}
    for message in messages:
        content = (message.content or "").strip()
        if not content:
            continue
        turns_by_conversation.setdefault(message.conversation_id, []).append(
            (message.role, message.content)
        )

    transcripts: list[ConversationTranscript] = []
    for conversation in conversations:
        turns = turns_by_conversation.get(conversation.id, [])
        roles = {role for role, _ in turns}
        if "user" not in roles or "assistant" not in roles:
            continue
        transcripts.append(
            ConversationTranscript(source_ref=str(conversation.id), messages=turns)
        )
    return transcripts


def _cleanup_staging(session, source_refs: list[str], *, force: bool) -> None:
    """Delete stale staging rows for the selected conversations.

    Without ``force`` only ``extracted`` rows (half-finished work from an
    interrupted run) are removed.  With ``force`` every prior row for those
    conversations is removed so they are re-extracted from scratch.
    """
    if not source_refs:
        return
    query = session.query(QaExtractionStaging).filter(
        QaExtractionStaging.source_ref.in_(source_refs)
    )
    if not force:
        query = query.filter(QaExtractionStaging.status == "extracted")
    query.delete(synchronize_session=False)
    session.commit()


def _mined_source_refs(session, source_refs: list[str]) -> set[str]:
    """Return the source refs that already have ``kept``/``discarded`` rows."""
    if not source_refs:
        return set()
    rows = (
        session.query(QaExtractionStaging.source_ref)
        .filter(
            QaExtractionStaging.source_ref.in_(source_refs),
            QaExtractionStaging.status.in_(("kept", "discarded")),
        )
        .distinct()
        .all()
    )
    return {source_ref for (source_ref,) in rows}


def _select_for_run(
    session, *, limit, since, force: bool, cleanup: bool
) -> list[ConversationTranscript]:
    """Shared selection pipeline for a real run and a dry run.

    Cleanup must run *before* the skip check: otherwise a stale ``extracted``
    row would make its conversation look already mined.  Only ``kept`` and
    ``discarded`` rows cause a skip, and ``force`` disables that skip.  ``limit``
    is applied last, after ordering, so it caps the work actually sent to the
    LLM.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must not be negative")

    transcripts = _select_candidates(session, since)
    source_refs = [transcript.source_ref for transcript in transcripts]
    if cleanup:
        _cleanup_staging(session, source_refs, force=force)
    if not force:
        mined = _mined_source_refs(session, source_refs)
        transcripts = [
            transcript for transcript in transcripts if transcript.source_ref not in mined
        ]
    if limit is not None:
        transcripts = transcripts[:limit]
    return transcripts


def load_transcripts(
    session, *, limit=None, since=None, force=False
) -> list[ConversationTranscript]:
    """Select conversations to mine, cleaning up stale ``extracted`` rows first."""
    return _select_for_run(
        session, limit=limit, since=since, force=force, cleanup=True
    )


def _new_batch_no(used: set[str]) -> str:
    """Return a fresh ``qa-YYYYmmddHHMMSS-<4 hex>`` batch identifier."""
    while True:
        batch_no = datetime.now().strftime(_BATCH_NO_FORMAT) + secrets.token_hex(2)
        if batch_no not in used:
            used.add(batch_no)
            return batch_no


def _existing_questions(session) -> list[str]:
    """Flatten ``knowledge_chunks.questions`` into individual lines."""
    questions: list[str] = []
    for (value,) in session.query(KnowledgeChunk.questions).all():
        for line in (value or "").splitlines():
            line = line.strip()
            if line:
                questions.append(line)
    return questions


def _mined_chunk_count(session) -> int:
    return (
        session.query(KnowledgeChunk)
        .filter(KnowledgeChunk.content_type == MINED_CONTENT_TYPE)
        .count()
    )


def mine_qa(
    session,
    llm,
    embedder,
    store,
    *,
    limit=None,
    batch_size=10,
    since=None,
    force=False,
    dry_run=False,
    do_vectorize=True,
) -> MiningReport:
    """Extract, stage, deduplicate, and insert Q&A pairs from conversations.

    ``dry_run`` performs extraction and the same in-memory dedup but skips the
    staging cleanup and every write, including vectorization.  ``store`` may be
    ``None`` when ``do_vectorize`` is false or ``dry_run`` is true.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    transcripts = _select_for_run(
        session, limit=limit, since=since, force=force, cleanup=not dry_run
    )

    used_batch_nos: set[str] = set()
    batch_nos: list[str] = []
    extracted_items: list[ExtractedQA] = []
    batches = 0

    for start in range(0, len(transcripts), batch_size):
        batch = transcripts[start : start + batch_size]
        batches += 1
        items = extract_qa_batch(llm, batch)
        extracted_items.extend(items)
        if dry_run:
            continue

        # One fresh batch id per LLM batch; rows are committed batch by batch
        # so an interrupted run leaves obvious half-finished staging rows.
        batch_no = _new_batch_no(used_batch_nos)
        batch_nos.append(batch_no)
        for index, item in enumerate(items):
            session.add(
                QaExtractionStaging(
                    batch_no=batch_no,
                    source_ref=batch[index % len(batch)].source_ref,
                    question=item.question,
                    answer=item.answer,
                    status="extracted",
                )
            )
        session.commit()

    keep_flags = dedup_questions(
        [item.question for item in extracted_items],
        _existing_questions(session),
        embedder,
    )
    kept = sum(keep_flags)
    discarded = len(keep_flags) - kept

    if dry_run:
        return MiningReport(
            batches=batches,
            extracted=len(extracted_items),
            kept=kept,
            discarded=discarded,
            inserted=0,
            vectorized=0,
        )

    rows: list[QaExtractionStaging] = []
    if batch_nos:
        rows = (
            session.query(QaExtractionStaging)
            .filter(
                QaExtractionStaging.batch_no.in_(batch_nos),
                QaExtractionStaging.status == "extracted",
            )
            .order_by(QaExtractionStaging.id)
            .all()
        )
    if len(rows) != len(extracted_items):
        raise RuntimeError(
            "staging rows do not match extracted items: "
            f"{len(rows)} rows vs {len(extracted_items)} items"
        )

    kept_items: list[MinedQA] = []
    for staging, item, keep in zip(rows, extracted_items, keep_flags):
        if keep:
            staging.status = "kept"
            kept_items.append(
                MinedQA(question=item.question, answer=item.answer, category=item.category)
            )
        else:
            staging.status = "discarded"

    # Do not commit the status changes separately: ``insert_mined_qa`` commits,
    # so the kept/discarded verdicts and the knowledge rows commit atomically.
    # Otherwise a crash between the two commits would mark conversations mined
    # while their knowledge was never inserted, and the rerun would skip them.
    before = _mined_chunk_count(session)
    insert_mined_qa(session, kept_items)
    inserted = _mined_chunk_count(session) - before

    vectorized = 0
    if do_vectorize and store is not None:
        vectorized = vectorize_pending(session, embedder, store).processed

    return MiningReport(
        batches=batches,
        extracted=len(extracted_items),
        kept=kept,
        discarded=discarded,
        inserted=inserted,
        vectorized=vectorized,
    )


def _parse_since(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "invalid ISO datetime "
            f"{value!r}; expected e.g. 2026-11-12 or 2026-11-12T15:30:00"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine reusable Q&A pairs from conversations into the knowledge base."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="mine at most this many conversations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="conversations per LLM extraction batch (default: QA_MINING_BATCH_SIZE)",
    )
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        help="only mine conversations updated at or after this ISO datetime",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete prior staging rows for selected conversations and re-mine",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="extract and deduplicate in memory without writing anything",
    )
    parser.add_argument(
        "--no-vectorize",
        action="store_true",
        help="leave newly inserted rows pending for a later vectorize run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: mine conversations and print a one-line summary."""
    args = _build_parser().parse_args(argv)
    settings = get_settings()
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else settings.qa_mining_batch_size
    )
    do_vectorize = not args.no_vectorize

    session = get_session_factory()()
    try:
        report = mine_qa(
            session,
            build_llm(settings),
            get_embedder(),
            get_vector_store() if (do_vectorize and not args.dry_run) else None,
            limit=args.limit,
            batch_size=batch_size,
            since=args.since,
            force=args.force,
            dry_run=args.dry_run,
            do_vectorize=do_vectorize,
        )
    finally:
        session.close()

    print(
        "mine_qa: "
        f"batches={report.batches} extracted={report.extracted} "
        f"kept={report.kept} discarded={report.discarded} "
        f"inserted={report.inserted} vectorized={report.vectorized}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
