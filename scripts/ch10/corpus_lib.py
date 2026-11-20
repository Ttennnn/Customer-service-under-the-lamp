"""Shared corpus primitives for ch10: samples, PII redaction, dedupe, JSONL."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from app.core.taxonomy import normalize_labels

_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_ORDER_RE = re.compile(r"\b[A-Za-z]{1,4}\d{8,}\b")
_WECHAT_RE = re.compile(
    r"(?:微信|WeChat|wechat|wx)\s*[:：]?\s*[A-Za-z][A-Za-z0-9_-]{5,19}"
)
_QQ_RE = re.compile(
    r"(?:QQ|qq)\s*[:：]?\s*[A-Za-z0-9][A-Za-z0-9_-]{5,19}"
)

_PUNCT_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "？": "?",
        "！": "!",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
    }
)


@dataclass
class Sample:
    id: str
    text: str
    labels: tuple[str, ...] = ()
    origin: str = "real"
    source_refs: tuple[dict, ...] = ()
    meta: dict = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Collapse whitespace and normalize harmless full-width punctuation."""
    return " ".join(text.translate(_PUNCT_TRANSLATION).split())


def stable_id(origin: str, text: str) -> str:
    return hashlib.sha1(f"{origin}|{text}".encode("utf-8")).hexdigest()[:12]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace phone/order/WeChat/QQ identifiers and report which fired."""
    flags: list[str] = []
    for pattern, placeholder, flag in (
        (_PHONE_RE, "[手机号]", "phone"),
        (_ORDER_RE, "[订单号]", "order"),
        (_WECHAT_RE, "[微信号]", "wechat"),
        (_QQ_RE, "[QQ号]", "qq"),
    ):
        text, count = pattern.subn(placeholder, text)
        if count:
            flags.append(flag)
    return text, flags


def dedupe_exact(samples: Iterable[Sample]) -> list[Sample]:
    """Keep the first sample per normalized text and merge source refs."""
    kept: list[Sample] = []
    index: dict[str, int] = {}
    for sample in samples:
        key = normalize_text(sample.text)
        if key not in index:
            index[key] = len(kept)
            kept.append(sample)
            continue
        existing = kept[index[key]]
        merged_refs = existing.source_refs + tuple(
            ref for ref in sample.source_refs if ref not in existing.source_refs
        )
        kept[index[key]] = replace(existing, source_refs=merged_refs)
    return kept


def _grams(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= 3:
        return {normalized}
    return {
        normalized[i : i + 3] for i in range(len(normalized) - 2)
    }


def jaccard_3gram(a: str, b: str) -> float:
    grams_a = _grams(a)
    grams_b = _grams(b)
    union = grams_a | grams_b
    if not union:
        return 1.0
    return len(grams_a & grams_b) / len(union)


def dedupe_near(
    samples: Iterable[Sample], threshold: float = 0.85
) -> tuple[list[Sample], int]:
    """Greedily merge near-identical texts, keeping the first occurrence."""
    kept: list[Sample] = []
    merged = 0
    for sample in samples:
        match_index = None
        for index, existing in enumerate(kept):
            if jaccard_3gram(existing.text, sample.text) >= threshold:
                match_index = index
                break
        if match_index is None:
            kept.append(sample)
            continue
        existing = kept[match_index]
        merged_refs = existing.source_refs + tuple(
            ref for ref in sample.source_refs if ref not in existing.source_refs
        )
        kept[match_index] = replace(existing, source_refs=merged_refs)
        merged += 1
    return kept, merged


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_counts(samples: Iterable[Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        for label in sample.labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def apply_corrections(
    samples: Iterable[Sample],
    corrections: dict[str, list[str]],
) -> list[Sample]:
    """Apply human audit corrections keyed by stable sample id."""
    samples = list(samples)
    known_ids = {sample.id for sample in samples}
    unknown = set(corrections) - known_ids
    if unknown:
        raise KeyError(f"corrections reference unknown sample ids: {sorted(unknown)}")

    corrected: list[Sample] = []
    for sample in samples:
        if sample.id not in corrections:
            corrected.append(sample)
            continue
        labels = tuple(normalize_labels(corrections[sample.id]))
        corrected.append(
            replace(
                sample,
                labels=labels,
                meta={**sample.meta, "audited": True},
            )
        )
    return corrected
