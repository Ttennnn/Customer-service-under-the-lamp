"""Full-parameter fine-tuning of RoBERTa-wwm-ext for 17-class multi-label topics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from app.core.taxonomy import LABEL2ID, TOPIC_CLASSES, labels_from_scores
from scripts.ch10.corpus_lib import read_jsonl
from scripts.ch10.train_lib import compute_metrics, multi_hot, scan_threshold

DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"


class JsonlDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = {key: torch.tensor(value) for key, value in encoded.items()}
        item["labels"] = torch.tensor(
            multi_hot(row.get("labels", []), LABEL2ID),
            dtype=torch.float,
        )
        return item


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def _metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    probs = _sigmoid(logits)
    pred_sets = []
    gold_sets = []
    for row, gold in zip(probs, labels):
        scores = {
            label: float(row[index])
            for index, label in enumerate(TOPIC_CLASSES)
        }
        pred_sets.append(set(labels_from_scores(scores, 0.5)))
        gold_sets.append(
            {
                label
                for index, label in enumerate(TOPIC_CLASSES)
                if float(gold[index]) >= 0.5
            }
        )
    metrics = compute_metrics(gold_sets, pred_sets)
    return {
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
    }


def train_model(
    train_path: Path,
    val_path: Path,
    out_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    max_length: int = 128,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_jsonl(Path(train_path))
    val_rows = read_jsonl(Path(val_path))
    if not train_rows or not val_rows:
        raise ValueError("train and val splits must be non-empty")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(TOPIC_CLASSES),
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config
    )

    train_ds = JsonlDataset(train_rows, tokenizer, max_length)
    val_ds = JsonlDataset(val_rows, tokenizer, max_length)
    use_fp16 = torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=seed,
        fp16=use_fp16,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.0,
            )
        ],
    )

    started = time.time()
    trainer.train()
    duration = time.time() - started
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(out_dir)

    prediction = trainer.predict(val_ds)
    probs = _sigmoid(prediction.predictions)
    gold_sets = []
    for row in val_rows:
        gold_sets.append(
            {label for label in row.get("labels", []) if label in LABEL2ID}
        )
    scan = scan_threshold(probs.tolist(), gold_sets)
    threshold_payload = {
        "source": "torch",
        "threshold": scan["best_threshold"],
        "val_micro_f1": scan["best_micro_f1"],
        "scan": scan["scan"],
    }
    (out_dir / "threshold.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    eval_entries = [
        entry
        for entry in trainer.state.log_history
        if "eval_loss" in entry or "eval_micro_f1" in entry
    ]
    report = {
        "model_name": model_name,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "fp16": use_fp16,
        "duration_s": duration,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "early_stopped": len(eval_entries) < epochs,
        "epoch_metrics": eval_entries,
        "torch_threshold_scan": scan,
    }
    (out_dir.parent / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "reports" / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the ch10 classifier")
    parser.add_argument(
        "--train", default="data/ch10/dataset/train_augmented.jsonl"
    )
    parser.add_argument("--val", default="data/ch10/dataset/val.jsonl")
    parser.add_argument("--out", default="data/ch10/model")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        report = train_model(
            Path(args.train),
            Path(args.val),
            Path(args.out),
            model_name=args.model_name,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports and exits
        print(f"train: failed: {exc}")
        return 1
    print(
        "train: done "
        f"train={report['train_samples']} val={report['val_samples']} "
        f"best={report['torch_threshold_scan']['best_micro_f1']:.4f} "
        f"threshold={report['torch_threshold_scan']['best_threshold']:.2f} "
        f"duration={report['duration_s']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
