"""Export the trained classifier to ONNX and make ONNX the threshold authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.ch10.corpus_lib import read_jsonl
from scripts.ch10.train_lib import scan_threshold


def compare_predictions(
    torch_probs: list[list[float]],
    onnx_probs: list[list[float]],
    threshold: float,
    atol: float = 1e-4,
) -> dict:
    if len(torch_probs) != len(onnx_probs):
        return {
            "max_logit_diff": float("inf"),
            "label_match_rate": 0.0,
            "ok": False,
        }
    max_diff = 0.0
    label_matches = 0
    for torch_row, onnx_row in zip(torch_probs, onnx_probs):
        if len(torch_row) != len(onnx_row):
            return {
                "max_logit_diff": float("inf"),
                "label_match_rate": 0.0,
                "ok": False,
            }
        max_diff = max(
            max_diff,
            max(abs(float(a) - float(b)) for a, b in zip(torch_row, onnx_row)),
        )
        torch_labels = tuple(value >= threshold for value in torch_row)
        onnx_labels = tuple(value >= threshold for value in onnx_row)
        if torch_labels == onnx_labels:
            label_matches += 1
    label_match_rate = label_matches / len(torch_probs) if torch_probs else 0.0
    return {
        "max_logit_diff": max_diff,
        "label_match_rate": label_match_rate,
        "ok": max_diff <= atol and label_match_rate == 1.0,
    }


def _sigmoid(values):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-values))


def _batch(items: list[dict], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def export_model(
    model_dir: Path,
    out_dir: Path,
    val_path: Path,
    reports_dir: Path,
    opset: int = 17,
) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    reports_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # SDPA and the exported math-attention graph can differ numerically on
    # CPU (we saw ~0.16 probability drift). Export and parity-check against
    # eager attention so the ONNX graph is the single inference truth.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, attn_implementation="eager"
    )
    model.eval()

    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

    wrapper = _LogitsWrapper(model)
    rows = read_jsonl(Path(val_path))
    if not rows:
        raise ValueError("validation split is empty")
    dummy = tokenizer(
        rows[0]["text"],
        truncation=True,
        max_length=32,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    token_type_ids = dummy.get(
        "token_type_ids", torch.zeros_like(input_ids)
    )
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, token_type_ids),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "labels"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    # The legacy exporter can leave the module in training mode; without this
    # the torch reference below runs with dropout and the parity check lies.
    model.eval()
    wrapper.eval()
    tokenizer.save_pretrained(out_dir)

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_names = [item.name for item in session.get_inputs()]

    def _onnx_probs(texts: list[str]):
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="np",
        )
        feed = {
            name: encoded[name].astype(np.int64)
            for name in input_names
            if name in encoded
        }
        logits = session.run(None, feed)[0]
        return _sigmoid(np.asarray(logits, dtype=np.float64))

    def _torch_probs(texts: list[str]):
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = wrapper(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded.get(
                    "token_type_ids", torch.zeros_like(encoded["input_ids"])
                ),
            )
        return _sigmoid(logits.cpu().numpy())

    sample_rows = rows[:32]
    texts = [row["text"] for row in sample_rows]
    torch_probs = _torch_probs(texts).tolist()
    onnx_probs = _onnx_probs(texts).tolist()
    parity = compare_predictions(torch_probs, onnx_probs, 0.5)

    all_probs: list[list[float]] = []
    for chunk in _batch(rows, 32):
        all_probs.extend(
            _onnx_probs([row["text"] for row in chunk]).tolist()
        )
    gold_sets = [
        {label for label in row.get("labels", [])} for row in rows
    ]
    scan = scan_threshold(all_probs, gold_sets)

    report = {
        "opset": opset,
        "samples": len(rows),
        "max_logit_diff": parity["max_logit_diff"],
        "label_match_rate": parity["label_match_rate"],
        "ok": parity["ok"],
        "onnx_path": str(onnx_path),
        "tokenizer_dir": str(out_dir),
        "val_samples": len(rows),
        "threshold": scan["best_threshold"],
        "val_micro_f1": scan["best_micro_f1"],
        "scan": scan["scan"],
    }
    (reports_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "val_scores.json").write_text(
        json.dumps(
            {
                "thresholds": [row["threshold"] for row in scan["scan"]],
                "rows": [
                    {
                        "id": rows[index].get("id", str(index)),
                        "probs": all_probs[index],
                        "gold": sorted(gold_sets[index]),
                    }
                    for index in range(len(rows))
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if parity["ok"]:
        threshold_payload = {
            "source": "onnx",
            "threshold": scan["best_threshold"],
            "val_micro_f1": scan["best_micro_f1"],
            "scan": scan["scan"],
        }
        (model_dir / "threshold.json").write_text(
            json.dumps(threshold_payload, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export ch10 model to ONNX")
    parser.add_argument("--model", default="data/ch10/model")
    parser.add_argument("--out", default="data/ch10/onnx")
    parser.add_argument("--val", default="data/ch10/dataset/val.jsonl")
    parser.add_argument("--reports", default="data/ch10/reports")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    try:
        report = export_model(
            Path(args.model),
            Path(args.out),
            Path(args.val),
            Path(args.reports),
            opset=args.opset,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits
        print(f"export: failed: {exc}")
        return 1

    print(
        f"export: ok={report['ok']} samples={report['samples']} "
        f"max_diff={report['max_logit_diff']:.6g} "
        f"label_match={report['label_match_rate']:.3f} "
        f"threshold={report['threshold']:.2f}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
