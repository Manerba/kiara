"""Zentrale DB-Hilfsfunktionen fuer Tests.

restore_test_schema() ist die einzige Stelle, die das vollstaendige
Test-DB-Schema definiert. Alle destruktiven Tests (DROP TABLE) MUESSEN
diese Funktion im Teardown aufrufen.

Nach dem Schema-Reset-Sprint: Source-of-Truth ist kiara/web/schema.sql,
geladen via kiara.web.schema.create_tables(conn). Alte Migrations-Kette
(MigrationManager, db_migrations) ist geloescht — entfaellt hier.
"""

# DB-DSN muss SOFORT gesetzt werden (vor allen kiara-Imports)
import kiara.shared.db as _db

_TEST_DSN = "postgresql://kiara:kiara@localhost:5432/kiara_test"
_db._dsn = _TEST_DSN


def _assert_test_database(conn) -> None:
    """Verhindert destruktive Test-Helper auf einer Nicht-Test-Datenbank."""
    row = conn.execute("SELECT current_database() AS db_name").fetchone()
    db_name = row["db_name"] if row else ""
    if "kiara_test" not in db_name:
        raise RuntimeError(f"Destruktiver Test-DB-Reset auf Nicht-Test-DB verweigert: {db_name!r}")


def _public_table_names(conn) -> list[str]:
    rows = conn.execute(
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    ).fetchall()
    return [row["tablename"] for row in rows]


def restore_test_schema():
    """Stellt das vollstaendige Test-DB-Schema wieder her.

    Droppt ALLE Tabellen und laedt das komplette Schema aus schema.sql.
    """
    from psycopg2 import sql as psycopg2_sql

    from kiara.shared.db import get_connection
    from kiara.web.schema import create_tables

    conn = get_connection()
    _assert_test_database(conn)
    # Phase 1: ALLE Tabellen droppen
    for table_name in _public_table_names(conn):
        conn.execute(
            psycopg2_sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                psycopg2_sql.Identifier(table_name)
            )
        )
    conn.commit()

    # Phase 2: Vollstaendiges Schema aus schema.sql
    create_tables(conn)
    conn.commit()
    conn.close()


def reset_test_data(conn=None) -> None:
    """Leert alle Test-Tabellen und setzt Sequenzen deterministisch zurueck.

    Das Schema bleibt erhalten. Falls vorher ein destruktiver Test Tabellen
    entfernt hat, stellt create_tables() den Schema-Zielzustand wieder her.
    """
    from psycopg2 import sql as psycopg2_sql

    from kiara.shared.db import get_connection
    from kiara.web.schema import create_tables

    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        _assert_test_database(conn)
        create_tables(conn)
        table_names = _public_table_names(conn)
        if table_names:
            conn.execute(
                psycopg2_sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                    psycopg2_sql.SQL(", ").join(
                        psycopg2_sql.Identifier(name) for name in table_names
                    )
                )
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()
