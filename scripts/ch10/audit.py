"""Export a human-audit Markdown sample from the labeled corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.taxonomy import TOPIC_CLASSES
from scripts.ch10.corpus_lib import Sample, dedupe_exact, read_jsonl


def review_rows(
    real: list[Sample],
    synthetic: list[Sample],
    disagreements: list[dict] | None = None,
) -> list[dict]:
    """All real samples plus up to 3 synthetic samples per class.

    Rows are unique by text across the whole table; a class whose samples were
    already shown under another label may show fewer than three.
    """
    rows: list[dict] = []
    seen_texts: set[str] = set()

    for sample in dedupe_exact(real):
        if sample.text in seen_texts:
            continue
        seen_texts.add(sample.text)
        rows.append(
            {
                "id": sample.id,
                "text": sample.text,
                "labels": list(sample.labels),
                "origin": "real",
                "note": sample.meta.get("source") or "真实问题",
            }
        )

    for label in TOPIC_CLASSES:
        taken = 0
        for sample in synthetic:
            if taken >= 3:
                break
            if label not in sample.labels or sample.text in seen_texts:
                continue
            seen_texts.add(sample.text)
            rows.append(
                {
                    "id": sample.id,
                    "text": sample.text,
                    "labels": list(sample.labels),
                    "origin": "synthetic",
                    "note": f"模拟题抽审（{label}）",
                }
            )
            taken += 1

    for item in disagreements or []:
        if item["text"] in seen_texts:
            continue
        seen_texts.add(item["text"])
        rows.append(
            {
                "id": item["id"],
                "text": item["text"],
                "labels": list(item.get("generated", [])),
                "origin": "synthetic",
                "note": (
                    "生成/盲标不一致: "
                    f"生成={item.get('generated')} 盲标={item.get('predicted')}"
                ),
            }
        )
    return rows


def export_review(rows: list[dict], path: Path) -> str:
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# ch10 语料人工抽审",
        "",
        "| id | 句子 | 标签 | 来源 | 备注 |",
        "|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| - | （无样本） | - | - | - |")
    for row in rows:
        lines.append(
            "| {id} | {text} | {labels} | {origin} | {note} |".format(
                id=cell(row["id"]),
                text=cell(row["text"]),
                labels=cell("、".join(row["labels"])),
                origin=cell(row["origin"]),
                note=cell(row.get("note", "")),
            )
        )
    text = "\n".join(lines) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _load_samples(path: Path) -> list[Sample]:
    return [
        Sample(
            id=row["id"],
            text=row["text"],
            labels=tuple(row.get("labels", ())),
            origin=row.get("origin", "real"),
            source_refs=tuple(row.get("source_refs", ())),
            meta=dict(row.get("meta", {})),
        )
        for row in read_jsonl(path)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export ch10 human-audit sample")
    parser.add_argument("--labeled", default="data/ch10/corpus_labeled.jsonl")
    parser.add_argument("--cross-check", default="data/ch10/reports/cross_check.json")
    parser.add_argument("--out", default="data/ch10/sample_review.md")
    args = parser.parse_args(argv)

    labeled_path = Path(args.labeled)
    if not labeled_path.exists():
        print(f"audit: missing {labeled_path}; run the corpus pipeline first")
        return 1
    samples = _load_samples(labeled_path)
    real = [sample for sample in samples if sample.origin == "real"]
    synthetic = [sample for sample in samples if sample.origin != "real"]

    disagreements: list[dict] = []
    cross_path = Path(args.cross_check)
    if cross_path.exists():
        report = json.loads(cross_path.read_text(encoding="utf-8"))
        disagreements = report.get("disagreements", [])

    rows = review_rows(real, synthetic, disagreements)
    export_review(rows, Path(args.out))
    print(
        f"audit: wrote {len(rows)} rows "
        f"(real={len(real)}, synthetic_sample={len(synthetic)}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
