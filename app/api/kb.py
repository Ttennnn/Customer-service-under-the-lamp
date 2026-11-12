"""JSON endpoints backing the ``/kb`` knowledge-base page.

Only the data API lives here; the HTML page itself is served elsewhere.
``/files`` lists the *effective* corpus (root seed + uploads, with uploads
shadowing same-named seed files) so the page can show what a build will read.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.db import session as db_session
from app.rag.docs import effective_doc_paths
from app.services import kb_build

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/kb/api")


def _file_entry(path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "source": "upload" if path.parent.name == "uploads" else "seed",
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


@api_router.get("/files")
def list_files() -> dict:
    """List effective documents (root seed + uploads, uploads shadow)."""
    settings = get_settings()
    docs_dir = Path(settings.knowledge_docs_dir)
    return {"files": [_file_entry(path) for path in effective_doc_paths(docs_dir)]}


@api_router.get("/status")
def get_status() -> dict:
    """Document-domain counts plus the in-memory build state."""
    session = None
    try:
        session = db_session.get_session_factory()()
        return kb_build.status(session)
    except Exception:
        logger.exception("knowledge base status query failed")
        raise HTTPException(status_code=500, detail="database error") from None
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("failed to close the knowledge status session")


@api_router.post("/build", status_code=202)
def post_build() -> dict:
    """Start one background build; 409 when a build is already running."""
    if not kb_build.start_build():
        raise HTTPException(status_code=409, detail="a build is already running")
    return {"status": "running"}
