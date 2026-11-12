"""Mine reusable Q&A pairs from customer-service conversations.

The extraction half turns raw transcripts into a structured batch through the
project's DeepSeek-compatible JSON structured-output path (see
``app/services/extract.py``). The helpers below normalise categories/questions
and deduplicate candidates both exactly (normalised text) and semantically
(cosine similarity), so later pipeline stages can stage only fresh questions.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

QA_CATEGORIES: tuple[str, ...] = ("售后", "物流", "支付", "商品", "其他")
DEFAULT_CATEGORY = "其他"

QA_EXTRACTION_SYSTEM_PROMPT = f"""\
你是电商客服知识库的问答挖掘助手。请从给定的对话记录(JSON 数组)中抽取可复用的问答对,并仅以 JSON 格式输出。

抽取要求:
1. 只抽取可复用、与具体订单无关的问答。涉及具体订单号、物流单号、收件人地址/电话、个人账户等一次性信息的对话不要抽取。
2. question 必须保留用户真实的问法,不要改写成书面语或抽象标题;answer 压缩客服回答,保留关键事实、金额、时限与条件,去掉寒暄和重复。
3. category 必须是以下枚举之一:{ "、".join(QA_CATEGORIES) }。无法归类的填「其他」。
4. 只输出一个合法的 JSON 对象,格式为:
{{"items": [{{"question": "邮费是多少", "answer": "满 99 元包邮,未满收取 8 元运费", "category": "物流"}}]}}
5. 如果没有任何可复用的问答,输出 {{"items": []}}。

不要输出任何解释、前后缀或 Markdown 代码块。"""


class ExtractedQA(BaseModel):
    """A single reusable question/answer pair mined from conversations."""

    question: str
    answer: str
    category: str


class QaExtractionBatch(BaseModel):
    """Structured-output wrapper: the model returns ``{"items": [...]}``."""

    items: list[ExtractedQA]


@dataclass(frozen=True)
class ConversationTranscript:
    """One conversation to mine. ``messages`` holds only user/assistant turns."""

    source_ref: str
    messages: list[tuple[str, str]]


def normalize_category(value: str) -> str:
    """Return ``value`` when it is a known category, otherwise 「其他」."""
    cleaned = (value or "").strip()
    return cleaned if cleaned in QA_CATEGORIES else DEFAULT_CATEGORY


def normalize_question(text: str) -> str:
    """Normalise a question for exact-match dedup.

    NFKC folds full-width characters, lower-casing ignores case, and removing
    whitespace/punctuation makes 「邮费，是多少？」 and 「邮费是多少」 collide
    while Chinese characters are preserved (Python's ``\\w`` is Unicode-aware).
    """
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[\s\W_]+", "", normalized)


def _cosine(a, b) -> float:
    """Cosine similarity, returning 0.0 for zero-norm (or empty) vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_qa_batch(llm, transcripts: list[ConversationTranscript]) -> list[ExtractedQA]:
    """Extract reusable Q&A pairs from ``transcripts`` in one LLM call.

    DeepSeek supports JSON output through the chat-completions
    ``response_format={"type": "json_object"}`` endpoint, which LangChain
    exposes as ``method="json_mode"``. Blank questions/answers are dropped and
    unknown categories fall back to 「其他」.
    """
    payload = [
        {
            "source_ref": transcript.source_ref,
            "messages": [
                {"role": role, "content": content}
                for role, content in transcript.messages
            ],
        }
        for transcript in transcripts
    ]
    messages = [
        SystemMessage(content=QA_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    structured = llm.with_structured_output(QaExtractionBatch, method="json_mode")
    batch = structured.invoke(messages)

    extracted: list[ExtractedQA] = []
    for item in batch.items:
        question = (item.question or "").strip()
        answer = (item.answer or "").strip()
        if not question or not answer:
            continue
        extracted.append(
            ExtractedQA(
                question=question,
                answer=answer,
                category=normalize_category(item.category),
            )
        )
    return extracted


def dedup_questions(
    candidates: list[str],
    existing: list[str],
    embedder,
    threshold: float = 0.92,
) -> list[bool]:
    """Return one keep/drop flag per candidate, in input order.

    A candidate is dropped when its normalised text matches an existing
    question or an earlier kept candidate exactly; surviving candidates are
    then embedded and dropped when their cosine similarity reaches
    ``threshold`` against any existing question or already-kept candidate.
    """
    existing_texts = [q for q in existing if (q or "").strip()]

    kept_norms: set[str] = set()
    for question in existing_texts:
        norm = normalize_question(question)
        if norm:
            kept_norms.add(norm)

    result = [False] * len(candidates)
    to_embed: list[int] = []
    for index, candidate in enumerate(candidates):
        norm = normalize_question(candidate)
        # Empty (punctuation-only / blank) questions are not storable knowledge.
        if not norm or norm in kept_norms:
            continue
        kept_norms.add(norm)
        to_embed.append(index)

    if not to_embed:
        return result

    candidate_vectors = embedder.embed([candidates[i] for i in to_embed])
    existing_vectors = embedder.embed(existing_texts) if existing_texts else []
    if len(candidate_vectors) != len(to_embed):
        raise ValueError(
            f"embedder returned {len(candidate_vectors)} vectors for {len(to_embed)} candidates"
        )
    if len(existing_vectors) != len(existing_texts):
        raise ValueError(
            f"embedder returned {len(existing_vectors)} vectors for {len(existing_texts)} existing questions"
        )

    kept_vectors: list[list[float]] = [list(vector) for vector in existing_vectors]
    for index, vector in zip(to_embed, candidate_vectors):
        if any(_cosine(vector, kept) >= threshold for kept in kept_vectors):
            continue
        result[index] = True
        kept_vectors.append(list(vector))

    return result
