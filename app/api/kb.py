"""JSON endpoints and page route backing the ``/kb`` knowledge-base UI.

The page is a self-contained static HTML file served from ``app/static``; the
``/kb/api/*`` endpoints below supply its data.  ``/files`` lists the *effective*
corpus (root seed + uploads, with uploads shadowing same-named seed files) so
the page can show what a build will read.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db import session as db_session
from app.rag.docs import effective_doc_paths, load_document
from app.services import kb_build

logger = logging.getLogger(__name__)

page_router = APIRouter()
api_router = APIRouter(prefix="/kb/api")

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_FILES = 20
ALLOWED_CONTENT_TYPES = {"policy", "faq", "manual"}

KB_PAGE = Path(__file__).resolve().parents[1] / "static" / "kb.html"


@page_router.get("/kb", response_class=FileResponse)
def kb_page() -> FileResponse:
    """Serve the self-contained knowledge-entry page."""
    return FileResponse(KB_PAGE, media_type="text/html")


def has_frontmatter(text: str) -> bool:
    """True when the text starts with a frontmatter delimiter line.

    A file that merely *looks* like it starts frontmatter but is malformed is
    treated as "the file wins" and then rejected by :func:`load_document`
    instead of silently receiving a second frontmatter block.
    """
    lines = text.splitlines()
    return bool(lines) and lines[0].strip() == "---"


def inject_frontmatter(text: str, content_type: str, category: str | None) -> str:
    """Prepend page-supplied metadata to a frontmatter-less document."""
    header = ["---", f"content_type: {content_type}"]
    if category:
        header.append(f"category: {category}")
    header.append("---")
    return "\n".join(header) + "\n\n" + text


def uploads_dir(settings) -> Path:
    """Directory that receives uploaded documents for the effective corpus."""
    return Path(settings.knowledge_docs_dir) / "uploads"


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


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best-effort cleanup only
        logger.exception("failed to remove temporary upload %s", path)


@api_router.post("/upload")
def post_upload(
    files: list[UploadFile] = File(...),
    content_type: str = Form(...),
    category: str | None = Form(None),
) -> dict:
    """Validate markdown uploads, inject metadata when absent, and store them.

    Files without frontmatter receive the form's ``content_type``/``category``;
    files that already carry frontmatter keep their own metadata.  Every file
    is parsed with :func:`load_document` from a non-``.md`` temp file before an
    atomic ``os.replace`` into ``knowledge/uploads/``, so a failed upload never
    leaves a half-written document or a temp file behind.
    """
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="invalid content_type")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400, detail=f"at most {MAX_FILES} files per request"
        )
    if kb_build.is_running():
        raise HTTPException(status_code=409, detail="a build is already running")
    if category is not None and ("\n" in category or "\r" in category):
        raise HTTPException(status_code=400, detail="invalid category")

    target_dir = uploads_dir(get_settings())
    saved: list[dict] = []
    rejected: list[dict] = []

    for upload in files:
        raw_name = upload.filename or ""
        name = Path(raw_name).name

        if not raw_name or "/" in raw_name or "\\" in raw_name:
            rejected.append({"name": raw_name, "reason": "invalid filename"})
            continue
        if not name or name == ".md" or not name.lower().endswith(".md"):
            rejected.append(
                {"name": name, "reason": "only .md files are accepted"}
            )
            continue

        data = upload.file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            rejected.append(
                {"name": name, "reason": "file exceeds the 2 MiB size limit"}
            )
            continue
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            rejected.append({"name": name, "reason": "file is not valid UTF-8"})
            continue

        if not has_frontmatter(text):
            text = inject_frontmatter(text, content_type, category)

        target_dir.mkdir(parents=True, exist_ok=True)
        temp_path = target_dir / f".tmp-{uuid.uuid4().hex}"
        final_path = target_dir / name
        try:
            temp_path.write_text(text, encoding="utf-8", newline="\n")
            document = load_document(temp_path)
            os.replace(temp_path, final_path)
        except ValueError as exc:
            # Drop the temp path from loader messages before showing the user.
            reason = str(exc).replace(f"{temp_path}: ", "")
            rejected.append({"name": name, "reason": reason})
        except OSError:
            logger.exception("failed to store upload %s", name)
            rejected.append({"name": name, "reason": "failed to save file"})
        else:
            saved.append(
                {
                    "name": name,
                    "size": final_path.stat().st_size,
                    "content_type": document.content_type,
                }
            )
        finally:
            _cleanup(temp_path)

    return {"saved": saved, "rejected": rejected}
