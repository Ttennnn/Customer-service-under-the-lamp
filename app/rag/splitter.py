"""Turn a loaded markdown :class:`Document` into section-level chunks.

Task 4 handles structure only: headings are tracked with a stack while
triple-backtick code fences are skipped, and every section that has a
non-empty body becomes one :class:`Chunk`. Long-text recursive splitting,
overlap, and table row splitting are added in a later task, which is why
``max_chars`` and ``overlap_chars`` are accepted but unused for now.
"""

from dataclasses import dataclass
import re

from app.rag.docs import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_KEY_CLAUSE_RE = re.compile(r"不予|不得|必须|免责|期限内|时效")
_KEY_CLAUSE_MARKER = "【关键条款】"
_PATH_SEP = " > "


@dataclass(frozen=True)
class Chunk:
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str
    is_key_clause: bool


def _extract_sections(body: str) -> list[tuple[tuple[str, ...], str]]:
    """Return ``(heading_path, body)`` for sections with a non-empty body."""
    sections: list[tuple[tuple[str, ...], str]] = []
    stack: list[tuple[int, str]] = []
    path: tuple[str, ...] | None = None
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal lines
        answer = "\n".join(lines).strip()
        if path is not None and answer:
            sections.append((path, answer))
        lines = []

    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence

        if not in_fence:
            match = _HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                text = match.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                path = tuple(title for _, title in stack) + (text,)
                stack.append((level, text))
                continue

        if path is not None:
            lines.append(line)

    flush()
    return sections


def _is_key_clause(content_type: str, answer: str) -> bool:
    if _KEY_CLAUSE_MARKER in answer:
        return True
    return content_type in {"policy", "manual"} and _KEY_CLAUSE_RE.search(answer) is not None


def _build_chunk(doc: Document, path: tuple[str, ...], answer: str) -> Chunk:
    leaf = path[-1]
    if doc.content_type == "faq":
        # FAQ wording lives in the heading; category comes from frontmatter.
        category = doc.category or doc.title
    else:
        # Policy/manual: category is the parent heading path (or doc title
        # when the section itself is top-level).
        category = _PATH_SEP.join(path[:-1]) if len(path) > 1 else doc.title

    return Chunk(
        category=category,
        questions=leaf,
        answer=answer,
        section_path=_PATH_SEP.join(path),
        content_type=doc.content_type,
        is_key_clause=_is_key_clause(doc.content_type, answer),
    )


def split_document(doc: Document, max_chars: int = 500, overlap_chars: int = 80) -> list[Chunk]:
    """Split *doc* into one chunk per non-empty markdown section.

    ``max_chars`` / ``overlap_chars`` are reserved for the recursive
    long-text splitter and are intentionally unused in this task.
    """
    return [_build_chunk(doc, path, answer) for path, answer in _extract_sections(doc.body)]
