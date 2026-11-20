"""Apply the ch10 topic-classification DDL to MySQL, idempotently.

``sql/ch10-ddl.sql`` is the authoritative schema. This script only creates the
single table and is safe to rerun: if the table already exists it prints a
no-op message instead of guessing at a partial migration.
"""

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from app.db.session import get_engine
from app.db.sql_utils import load_statements

SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "ch10-ddl.sql"
TABLE = "topic_classifications"


def _load_statements(path: Path = SQL_PATH) -> list[str]:
    return load_statements(path)


def _existing_table(conn: Connection) -> bool:
    row = conn.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = :table"
        ),
        {"table": TABLE},
    ).first()
    return row is not None


def apply_ch10(engine: Engine) -> None:
    statements = _load_statements()
    with engine.begin() as conn:
        if _existing_table(conn):
            print(
                "ch10 topic classification schema already initialized; nothing to do."
            )
            return
        # The DDL is raw MySQL; keep it on the driver path so any ':' inside
        # comments/values is never parsed as a SQLAlchemy bind parameter.
        for statement in statements:
            conn.exec_driver_sql(statement)

    print(f"ch10 topic classification schema created: {TABLE}")


def main() -> None:
    apply_ch10(get_engine())


if __name__ == "__main__":
    main()
