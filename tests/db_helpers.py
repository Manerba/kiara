"""Zentrale DB-Hilfsfunktionen fuer Tests.

restore_test_schema() ist die einzige Stelle, die das vollstaendige
Test-DB-Schema definiert. Alle destruktiven Tests (DROP TABLE) MUESSEN
diese Funktion im Teardown aufrufen.

Nach dem Schema-Reset-Sprint: Source-of-Truth ist kiara/web/schema.sql,
geladen via kiara.web.schema.create_tables(conn). Alte Migrations-Kette
(MigrationManager, db_migrations) ist geloescht — entfaellt hier.
"""

# DB-DSN muss SOFORT gesetzt werden (vor allen kiara-Imports)
import os

import kiara.shared.db as _db

# KIARA_TEST_DSN erlaubt getrennte Test-DBs fuer parallele Laeufe
# (z. B. Selbsttest in der Admin-UI und Agenten-Laeufe). Der Name muss
# "kiara_test" enthalten, sonst verweigern die destruktiven Helfer.
_TEST_DSN = os.environ.get("KIARA_TEST_DSN", "postgresql://kiara:kiara@localhost:5432/kiara_test")
TEST_DB_NAME = _TEST_DSN.rsplit("/", 1)[-1].split("?", 1)[0]
if "kiara_test" not in TEST_DB_NAME:
    raise RuntimeError(f"KIARA_TEST_DSN zeigt nicht auf eine Test-DB: {TEST_DB_NAME!r}")
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


def _missing_schema_tables(conn) -> set[str]:
    """Tabellen aus schema.sql, die in der Test-DB fehlen.

    create_tables() erkennt Vollstaendigkeit nur ueber die Tabellenanzahl
    (Basiswert 49). schema.sql und Migrationen legen inzwischen mehr Tabellen
    an; loescht ein Test eine einzelne Tabelle, bleibt die Anzahl darueber und
    die Tabelle wuerde nie wiederhergestellt.
    """
    import re

    from kiara.web.schema import _SCHEMA_SQL_PATH

    expected = set(
        re.findall(r"^CREATE TABLE public\.(\w+) \(", _SCHEMA_SQL_PATH.read_text(), re.M)
    )
    return expected - set(_public_table_names(conn))


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
        if _missing_schema_tables(conn):
            # Unvollstaendiges Schema: komplett neu laden (wie restore_test_schema).
            for table_name in _public_table_names(conn):
                conn.execute(
                    psycopg2_sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        psycopg2_sql.Identifier(table_name)
                    )
                )
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


_CANONICAL_SCHEMA_FINGERPRINT: str | None = None


def _schema_fingerprint(conn) -> str:
    """Struktur des public-Schemas: Spalten, Constraints, eigene Trigger."""
    import hashlib

    parts = []
    for query in (
        "SELECT table_name, column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns WHERE table_schema = 'public' "
        "ORDER BY table_name, column_name",
        "SELECT conrelid::regclass::text AS rel, conname, pg_get_constraintdef(oid) AS def "
        "FROM pg_constraint WHERE connamespace = 'public'::regnamespace "
        "ORDER BY rel, conname",
        "SELECT tgrelid::regclass::text AS rel, tgname FROM pg_trigger "
        "WHERE NOT tgisinternal ORDER BY rel, tgname",
    ):
        parts.extend(repr(tuple(row.values())) for row in conn.execute(query).fetchall())
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def ensure_canonical_schema() -> None:
    """Stellt das echte Schema wieder her, falls ein Test es veraendert hat.

    Viele aeltere Tests ersetzen Tabellen per DROP/CREATE durch eigene Kopien
    (ohne neuere Spalten, Constraints oder Trigger) und raeumen nicht auf.
    Der erste Aufruf laedt das Schema frisch und merkt sich seinen
    Fingerabdruck; spaetere Aufrufe laden nur bei Abweichung neu.
    """
    global _CANONICAL_SCHEMA_FINGERPRINT
    from kiara.shared.db import get_connection

    if _CANONICAL_SCHEMA_FINGERPRINT is not None:
        conn = get_connection()
        try:
            _assert_test_database(conn)
            if _schema_fingerprint(conn) == _CANONICAL_SCHEMA_FINGERPRINT:
                return
        finally:
            conn.rollback()
            conn.close()
    restore_test_schema()
    conn = get_connection()
    try:
        _CANONICAL_SCHEMA_FINGERPRINT = _schema_fingerprint(conn)
    finally:
        conn.rollback()
        conn.close()
