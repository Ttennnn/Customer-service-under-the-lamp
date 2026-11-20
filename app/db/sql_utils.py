"""Shared helpers for applying the project's raw SQL DDL files."""

from pathlib import Path


def split_statements(sql: str) -> list[str]:
    """Split on ``;`` boundaries that are not inside single-quoted strings.

    Some DDL comments contain literal ``;`` (e.g. ``如 A43;一题一行``), so a
    naive ``str.split(";")`` would cut CREATE statements in half.
    """
    statements: list[str] = []
    start = 0
    in_quote = False
    i = 0
    while i < len(sql):
        char = sql[i]
        if in_quote:
            if char == "\\":
                i += 2
                continue
            if char == "'":
                if i + 1 < len(sql) and sql[i + 1] == "'":
                    i += 2
                    continue
                in_quote = False
        elif char == "'":
            in_quote = True
        elif char == ";":
            statement = sql[start:i].strip()
            if statement:
                statements.append(statement)
            start = i + 1
        i += 1

    tail = sql[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def load_statements(path: Path) -> list[str]:
    """Read ``path``, drop ``--`` comment lines, and split into statements."""
    raw = path.read_text(encoding="utf-8")
    sql = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("--")
    )
    return split_statements(sql)
