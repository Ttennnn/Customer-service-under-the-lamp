"""Apply the ch09 observability/flywheel DDL to MySQL, idempotently.

The DDL in ``sql/ch09-ddl.sql`` is authoritative for production. This script is
safe to rerun:

- both tables and both new ``low_confidence_questions`` columns present -> no-op
- none of them present -> executes every statement in the SQL file
- only part of them present -> raises a clear error instead of half-applying
"""

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.db.session import get_engine
from app.db.sql_utils import load_statements

SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "ch09-ddl.sql"
TABLES = ("review_queue", "eval_runs")
NEW_COLUMNS = {
    "low_confidence_questions": ("retrieved_chunks", "matched_review_id"),
}


def _load_statements(path: Path = SQL_PATH) -> list[str]:
    """Read ``path``, drop ``--`` comment lines, and split into statements."""
    return load_statements(path)


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


def _existing_columns(conn: Connection) -> set[str]:
    """Return the new columns that already exist on ``low_confidence_questions``."""
    table = "low_confidence_questions"
    columns = NEW_COLUMNS[table]
    placeholders = ", ".join(f":c{i}" for i in range(len(columns)))
    params = {f"c{i}": name for i, name in enumerate(columns)}
    params["table"] = table
    rows = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() "
            "AND table_name = :table "
            f"AND column_name IN ({placeholders})"
        ),
        params,
    )
    return {row[0] for row in rows}


def apply_ch09(engine: Engine) -> None:
    """Create the ch09 tables/columns if needed, or no-op / fail on partial."""
    statements = _load_statements()
    expected_tables = set(TABLES)
    expected_columns = set(NEW_COLUMNS["low_confidence_questions"])

    with engine.begin() as conn:
        present_tables = _existing_tables(conn)
        present_columns = _existing_columns(conn)

        tables_complete = present_tables == expected_tables
        columns_complete = present_columns == expected_columns
        if tables_complete and columns_complete:
            print("ch09 observability/flywheel schema already initialized; nothing to do.")
            return
        if not present_tables and not present_columns:
            for statement in statements:
                # Use the driver SQL path: the user DDL contains JSON examples
                # such as {"recall_at_k":0.82}, which SQLAlchemy's text()
                # parser would misread as a bind parameter named "0".
                conn.exec_driver_sql(statement)
        else:
            missing_tables = sorted(expected_tables - present_tables)
            missing_columns = sorted(expected_columns - present_columns)
            raise RuntimeError(
                "ch09 schema is partially initialized: "
                f"present_tables={sorted(present_tables)}, "
                f"missing_tables={missing_tables}, "
                f"present_columns={sorted(present_columns)}, "
                f"missing_columns={missing_columns}. "
                "Refusing to guess; inspect the database and finish or drop "
                "the partial schema manually before rerunning."
            )

    print(f"ch09 observability/flywheel schema created: {', '.join(TABLES)}")


def main() -> None:
    """CLI entry point: apply the ch09 DDL to the configured MySQL database."""
    apply_ch09(get_engine())


if __name__ == "__main__":
    main()
