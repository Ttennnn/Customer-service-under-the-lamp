"""Local development entry point.

PyCharm can run this file directly (green triangle) to start the FastAPI app
with auto-reload. It loads ``app.main:app`` through uvicorn, so the working
directory must be the repository root (where ``.env`` and ``knowledge/`` live).
"""

import logging

import uvicorn

from app.logging_config import configure_file_logging

if __name__ == "__main__":
    # Surface the ch05 retrieval_forced / turn_log INFO lines documented in
    # dev-notes when running the local demo with ``python run.py``. The file
    # handler goes on first so ``basicConfig`` never replaces it.
    configure_file_logging()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
