"""Load the markdown knowledge corpus.

Sample documents use a deliberately simple frontmatter format so the project
does not need a YAML dependency::

    ---
    content_type: policy
    category: 售后
    ---
    # 标题

The body (including the H1) is kept verbatim for the splitter.
"""

from dataclasses import dataclass
from pathlib import Path
import re

CONTENT_TYPES = frozenset({"policy", "faq", "manual"})

_H1_RE = re.compile(r"^#\s+(.+?)\s*$")


@dataclass(frozen=True)
class Document:
    path: Path
    title: str
    content_type: str  # policy | faq | manual
    category: str | None
    body: str  # full markdown body including H1


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: missing frontmatter (expected leading '---')")

    meta: dict[str, str] = {}
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line == "---":
            return meta, "".join(lines[i + 1:])
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"{path}: invalid frontmatter line: {line!r}")
        meta[key.strip()] = value.strip()

    raise ValueError(f"{path}: unterminated frontmatter (missing closing '---')")


def _find_title(body: str, path: Path) -> str:
    for line in body.splitlines():
        match = _H1_RE.match(line)
        if match:
            return match.group(1)
    raise ValueError(f"{path}: missing H1 title")


def load_document(path: Path) -> Document:
    """Parse one markdown file into a :class:`Document`."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    meta, body = _split_frontmatter(text, path)

    content_type = meta.get("content_type", "")
    if content_type not in CONTENT_TYPES:
        raise ValueError(
            f"{path}: invalid or missing content_type {content_type!r}; "
            f"expected one of {sorted(CONTENT_TYPES)}"
        )

    return Document(
        path=path,
        title=_find_title(body, path),
        content_type=content_type,
        category=meta.get("category") or None,
        body=body,
    )


def _filename_sort_key(path: Path) -> tuple[bytes, str]:
    """Sort filenames deterministically, pinyin-friendly for Chinese names.

    GBK (like Windows Explorer) orders common Chinese characters by pinyin,
    so sorting on the GBK bytes keeps ``退货`` before ``运费``.  Names that
    cannot be encoded as GBK sort after encodable ones by Unicode order.
    """
    name = path.name
    try:
        return (name.encode("gbk"), name)
    except UnicodeEncodeError:
        return (b"\xff", name)


def effective_doc_paths(docs_dir: Path) -> list[Path]:
    """Return the effective corpus: root ``*.md`` plus ``uploads/*.md``.

    Files in ``uploads`` shadow same-named files in the root seed directory.
    Only these two layers are read; other subdirectories are ignored.  Paths
    are returned sorted by filename.
    """
    docs_dir = Path(docs_dir)
    by_name: dict[str, Path] = {p.name: p for p in docs_dir.glob("*.md")}
    uploads_dir = docs_dir / "uploads"
    if uploads_dir.is_dir():
        by_name.update({p.name: p for p in uploads_dir.glob("*.md")})
    return sorted(by_name.values(), key=_filename_sort_key)


def load_documents(docs_dir: Path) -> list[Document]:
    """Load the effective corpus (see :func:`effective_doc_paths`)."""
    return [load_document(p) for p in effective_doc_paths(docs_dir)]
