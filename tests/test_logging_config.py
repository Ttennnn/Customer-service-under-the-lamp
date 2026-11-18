import logging
from pathlib import Path

from app.logging_config import configure_file_logging


def test_configure_creates_file_and_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "app.log"
    monkeypatch.setenv("LOG_TO_FILE", "1")
    configure_file_logging(str(path), force=True)
    logging.getLogger("test.ch07").info("model_ctx hello")
    configure_file_logging(str(path), force=True)
    logging.getLogger("test.ch07").info("history_ctx world")
    text = path.read_text(encoding="utf-8")
    assert text.count("model_ctx hello") == 1
    assert "history_ctx world" in text


def test_log_to_file_zero_disables(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_TO_FILE", "0")
    path = tmp_path / "app.log"
    configure_file_logging(str(path), force=True)
    logging.getLogger("test.ch07").info("should not appear")
    assert not Path(path).exists()
