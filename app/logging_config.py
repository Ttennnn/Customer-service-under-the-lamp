"""File logging for the local demo.

The ch07 context-management traces are ordinary ``logging`` INFO lines. This
module attaches a UTF-8 ``FileHandler`` for ``log/app.log`` so those traces
survive on disk, while leaving pytest runs untouched unless a test opts in
with ``LOG_TO_FILE=1``.
"""

import logging
import os
import sys
from pathlib import Path

from app.config import get_settings

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

_TRUTHY = {"1", "true", "t", "yes", "y", "on"}


def _log_to_file_enabled() -> bool:
    """Honor an explicit ``LOG_TO_FILE`` override, else ``Settings``."""
    raw = os.environ.get("LOG_TO_FILE")
    if raw is not None:
        return raw.strip().lower() in _TRUTHY
    return get_settings().log_to_file


def _same_path(handler: logging.FileHandler, path: Path) -> bool:
    """Compare handler paths case-insensitively on Windows."""
    return os.path.normcase(handler.baseFilename) == os.path.normcase(str(path))


def _has_console_handler(root: logging.Logger) -> bool:
    """A plain ``StreamHandler`` exists (``FileHandler`` subclasses it)."""
    return any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    )


def configure_file_logging(path: str = "log/app.log", *, force: bool = False) -> None:
    """Attach a UTF-8 file handler for *path* to the root logger.

    Idempotent: a second call with the same path adds nothing. ``force=True``
    closes and replaces the existing same-path handler (closing first matters
    on Windows, where an open handler keeps the file locked). Tests only get a
    repository log when they explicitly ask for it with ``LOG_TO_FILE=1``.
    """
    if "pytest" in sys.modules and os.environ.get("LOG_TO_FILE") != "1":
        return
    if not _log_to_file_enabled():
        return

    log_path = Path(path)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    formatter = logging.Formatter(LOG_FORMAT)

    existing = [
        handler
        for handler in root.handlers
        if isinstance(handler, logging.FileHandler) and _same_path(handler, log_path)
    ]
    if existing and not force:
        # Already wired by an earlier call; still normalise the root level.
        root.setLevel(logging.INFO)
        return

    for handler in existing:
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if not _has_console_handler(root):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    root.setLevel(logging.INFO)
