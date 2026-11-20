"""Shared ch10 artifact paths, anchored to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "ch10"
REPORT_DIR = DATA_DIR / "reports"
MODEL_DIR = DATA_DIR / "model"
ONNX_DIR = DATA_DIR / "onnx"
GENERATED_DIR = DATA_DIR / "generated"
GOLDEN_PATH = DATA_DIR / "golden_samples.jsonl"
CORRECTIONS_PATH = DATA_DIR / "audit_corrections.json"
LOG_DIR = ROOT / "log" / "acceptance"
