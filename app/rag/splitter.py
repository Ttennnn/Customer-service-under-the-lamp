"""Turn a loaded markdown :class:`Document` into retrieval-sized chunks.

Headings are tracked with a stack while triple-backtick code fences are
skipped, so every non-empty section becomes at least one :class:`Chunk`.
Oversized section bodies are then split recursively: blank-line paragraphs
first, then list items and Markdown tables, then sentence boundaries, and
finally a hard character cut when a single sentence is longer than
``max_chars``. Markdown tables keep their header and separator in every
row group, and consecutive non-table chunks overlap on sentence boundaries.
"""

from dataclasses import dataclass
import re

from app.rag.docs import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_KEY_CLAUSE_RE = re.compile(r"不予|不得|必须|免责|期限内|时效")
_KEY_CLAUSE_MARKER = "【关键条款】"
_PATH_SEP = " > "
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？；!?;…]+")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+")


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


def _split_paragraphs(text: str) -> list[str]:
    """Split *text* on blank lines, dropping empty paragraphs."""
    return [part.strip() for part in _BLANK_LINE_RE.split(text) if part.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split *text* after each run of sentence-ending punctuation.

    Leading whitespace stays attached to the sentence that follows it so a
    later join with ``""`` keeps the original line breaks.
    """
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        parts.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        parts.append(text[start:])
    return [part for part in parts if part.strip()]


def _split_list_items(text: str) -> list[str]:
    """Split *text* at bullet/numbered list markers, if it has any."""
    items: list[str] = []
    current: list[str] = []
    markers = 0
    for line in text.splitlines():
        if _LIST_ITEM_RE.match(line):
            markers += 1
            if current:
                items.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        items.append("\n".join(current).strip())
    items = [item for item in items if item]
    if markers == 0 or len(items) <= 1:
        return [text]
    return items


def _is_table_separator(line: str) -> bool:
    """Return True for a GFM separator row such as ``|---|---|``."""
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped:
        return False
    return _TABLE_SEPARATOR_RE.match(stripped) is not None


def _split_table_segments(text: str) -> list[tuple[str, object]]:
    """Return ordered ``("text", str)`` / ``("table", (head, sep, rows))`` parts."""
    lines = text.splitlines()
    segments: list[tuple[str, object]] = []
    buffer: list[str] = []
    index = 0

    def flush_text() -> None:
        nonlocal buffer
        piece = "\n".join(buffer).strip()
        if piece:
            segments.append(("text", piece))
        buffer = []

    while index < len(lines):
        line = lines[index]
        if "|" in line and index + 1 < len(lines) and _is_table_separator(lines[index + 1]):
            end = index + 2
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            flush_text()
            segments.append(
                (
                    "table",
                    (
                        line.strip(),
                        lines[index + 1].strip(),
                        [row.strip() for row in lines[index + 2:end]],
                    ),
                )
            )
            index = end
            continue
        buffer.append(line)
        index += 1
    flush_text()
    return segments


def _group_table_rows(
    header: str, separator: str, rows: list[str], max_chars: int
) -> list[str]:
    """Group table rows, repeating ``header`` + ``separator`` in every group."""
    groups: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join([header, separator, *current, row])
        if current and len(candidate) > max_chars:
            groups.append("\n".join([header, separator, *current]))
            current = [row]
        else:
            current.append(row)
    if current:
        groups.append("\n".join([header, separator, *current]))
    if not groups:
        groups.append(f"{header}\n{separator}")
    return groups


def _pack_units(units: list[str], max_chars: int, joiner: str) -> list[str]:
    """Greedily join *units* into chunks no longer than *max_chars*."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not unit.strip():
            continue
        if not current:
            current = unit
        elif len(current) + len(joiner) + len(unit) <= max_chars:
            current += joiner + unit
        else:
            chunks.append(current.strip())
            current = unit
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_paragraph(text: str, max_chars: int) -> list[str]:
    """Recursively split one paragraph by the task's priority order."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = _split_paragraphs(text)
    if len(paragraphs) > 1:
        units: list[str] = []
        for paragraph in paragraphs:
            units.extend(_split_paragraph(paragraph, max_chars))
        return _pack_units(units, max_chars, "\n\n")

    items = _split_list_items(text)
    if len(items) > 1:
        units = []
        for item in items:
            units.extend(_split_paragraph(item, max_chars))
        return _pack_units(units, max_chars, "\n")

    sentences = _split_sentences(text)
    if len(sentences) > 1:
        units = []
        for sentence in sentences:
            if len(sentence.strip()) > max_chars:
                units.extend(_split_paragraph(sentence, max_chars))
            else:
                units.append(sentence)
        return _pack_units(units, max_chars, "")

    # A single sentence (or unpunctuated line) exceeds the limit: hard cut.
    return [
        text[start:start + max_chars].strip()
        for start in range(0, len(text), max_chars)
        if text[start:start + max_chars].strip()
    ]


def _split_block(text: str, max_chars: int) -> list[tuple[str, bool]]:
    """Split one blank-line block into ``(text, is_table)`` pieces."""
    segments = _split_table_segments(text)
    if len(segments) == 1 and segments[0][0] == "text":
        return [(chunk, False) for chunk in _split_paragraph(text, max_chars)]

    pieces: list[tuple[str, bool]] = []
    for kind, payload in segments:
        if kind == "table":
            header, separator, rows = payload  # type: ignore[misc]
            table_text = "\n".join([header, separator, *rows])
            if len(table_text) <= max_chars:
                pieces.append((table_text, True))
            else:
                pieces.extend(
                    (group, True)
                    for group in _group_table_rows(header, separator, rows, max_chars)
                )
        else:
            pieces.extend(
                (chunk, False) for chunk in _split_paragraph(payload, max_chars)  # type: ignore[arg-type]
            )
    return pieces


def _overlap_tail(previous: str, overlap_chars: int) -> str:
    """Return the previous chunk's tail from after its first sentence boundary."""
    tail = previous[-overlap_chars:]
    match = _SENTENCE_BOUNDARY_RE.search(tail)
    if match is None:
        return ""
    return tail[match.end():].strip()


def _split_section(answer: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split a section body into chunk texts, applying overlap where allowed."""
    pieces: list[tuple[str, bool]] = []
    for paragraph in _split_paragraphs(answer):
        pieces.extend(_split_block(paragraph, max_chars))

    packed: list[tuple[str, bool]] = []
    current = ""
    for text, is_table in pieces:
        text = text.strip()
        if not text:
            continue
        if is_table:
            if current:
                packed.append((current, False))
                current = ""
            packed.append((text, True))
        elif not current:
            current = text
        elif len(current) + 2 + len(text) <= max_chars:
            current = f"{current}\n\n{text}"
        else:
            packed.append((current, False))
            current = text
    if current:
        packed.append((current, False))

    if overlap_chars <= 0:
        return [text for text, _ in packed]

    result: list[str] = []
    previous: str | None = None
    previous_is_table = False
    for text, is_table in packed:
        if previous is not None and not is_table and not previous_is_table:
            overlap = _overlap_tail(previous, overlap_chars)
            if overlap:
                text = overlap + text
        result.append(text)
        previous = text
        previous_is_table = is_table
    return result


def _build_chunk(doc: Document, path: tuple[str, ...], answer: str) -> Chunk:
    leaf = path[-1]
    # Explicit frontmatter category wins for every content type.  Without it,
    # FAQ falls back to the document title and policy/manual use the parent
    # heading path (or the document title for a top-level section).
    category = doc.category or (
        doc.title
        if doc.content_type == "faq"
        else (_PATH_SEP.join(path[:-1]) if len(path) > 1 else doc.title)
    )

    return Chunk(
        category=category,
        questions=leaf,
        answer=answer,
        section_path=_PATH_SEP.join(path),
        content_type=doc.content_type,
        is_key_clause=_is_key_clause(doc.content_type, answer),
    )


def split_document(doc: Document, max_chars: int = 500, overlap_chars: int = 80) -> list[Chunk]:
    """Split *doc* into retrieval-sized chunks in document order.

    Sections are split recursively when longer than *max_chars*; Markdown
    tables keep their header/separator in every row group and consecutive
    non-table chunks overlap by up to *overlap_chars*, starting at a
    sentence boundary. ``is_key_clause`` is recomputed for every chunk.

    Within one document, an identical ``(category, questions, answer)`` triple
    is emitted only once (first occurrence wins), so duplicate sections cannot
    persist as two retrieval chunks with different ``section_path`` values.
    """
    max_chars = max(1, max_chars)
    overlap_chars = max(0, overlap_chars)
    chunks: list[Chunk] = []
    seen: set[tuple[str, str, str]] = set()
    for path, answer in _extract_sections(doc.body):
        for piece in _split_section(answer, max_chars, overlap_chars):
            chunk = _build_chunk(doc, path, piece)
            key = (chunk.category, chunk.questions, chunk.answer)
            if key in seen:
                continue
            seen.add(key)
            chunks.append(chunk)
    return chunks
