"""Apply the ch04 RAG-advanced DDL to MySQL, idempotently.

The DDL in ``sql/ch04_rag_advanced.sql`` is authoritative for production. This
script is safe to rerun:

- both tables present  -> no-op (prints "already initialized")
- neither table present -> executes every statement in the SQL file
- only one table present -> raises a clear error instead of half-applying
"""

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.db.session import get_engine

SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "ch04_rag_advanced.sql"
TABLES = ("low_confidence_questions", "faith_cases")


def _split_statements(sql: str) -> list[str]:
    """Split on ``;`` boundaries that are not inside single-quoted strings.

    The ch04 DDL comments contain literal ``;`` (e.g. ``如 A43;一题一行``), so a
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


def _load_statements(path: Path = SQL_PATH) -> list[str]:
    """Read ``path``, drop ``--`` comment lines, and split into statements."""
    raw = path.read_text(encoding="utf-8")
    sql = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("--")
    )
    return _split_statements(sql)


def _existing_tables(conn: Connection) -> set[str]:
    """Return the subset of ``TABLES`` that already exist in the current DB."""
    placeholders = ", ".join(f":t{i}" for i in range(len(TABLES)))
    params = {f"t{i}": name for i, name in enumerate(TABLES)}
    rows = conn.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = DATABASE() "
            f"AND table_name IN ({placeholders})"
        ),
        params,
    )
    return {row[0] for row in rows}


def apply_ch04(engine: Engine) -> None:
    """Create the ch04 tables if needed, or no-op / fail on partial schema."""
    statements = _load_statements()
    expected = set(TABLES)

    with engine.begin() as conn:
        present = _existing_tables(conn)
        if present == expected:
            print("ch04 RAG advanced tables already initialized; nothing to do.")
            return
        if present:
            missing = sorted(expected - present)
            raise RuntimeError(
                "ch04 schema is partially initialized: "
                f"present={sorted(present)}, missing={missing}. "
                "Refusing to guess; inspect the database and finish or drop "
                "the partial schema manually before rerunning."
            )

        for statement in statements:
            conn.execute(text(statement))

    print(f"ch04 RAG advanced tables created: {', '.join(TABLES)}")


def main() -> None:
    """CLI entry point: apply the ch04 DDL to the configured MySQL database."""
    apply_ch04(get_engine())


if __name__ == "__main__":
    main()
