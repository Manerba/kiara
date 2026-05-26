"""KIara - Schema-Loader.

Die Quelle der Wahrheit fuer das DB-Schema ist ``kiara/web/schema.sql``
(generiert aus ``pg_dump --schema-only`` der Produktions-DB, bereinigt
um TEXT→TIMESTAMPTZ Korrekturen).

Dieses Python-Modul existiert nur noch als duenner Adapter, damit Tests
und Call-Sites weiterhin ``create_tables(conn)`` aufrufen koennen.
"""

from pathlib import Path

_SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


def create_tables(conn) -> None:
    """Fuehrt alle CREATE TABLE / CREATE INDEX / ALTER TABLE aus ``schema.sql`` aus.

    Semi-idempotent: Wenn alle erwarteten Tabellen vorhanden sind, wird das Schema
    nicht erneut geladen. Sobald aber EINE Tabelle fehlt (z.B. weil ein Test sie
    per DROP TABLE entfernt hat), wird das gesamte Schema gedroppt und neu geladen.
    Das ist noetig, weil schema.sql nicht mit ``IF NOT EXISTS`` arbeitet (pg_dump-
    Ausgabe) und ``CREATE COLLATION nocase`` bei zweitem Lauf scheitert.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM information_schema.tables "
        "WHERE table_schema='public' AND table_type='BASE TABLE'"
    ).fetchone()
    existing = row["cnt"] if row else 0
    if existing >= _EXPECTED_TABLE_COUNT:
        _migrate_audit_schema(conn)
        _migrate_group_schema(conn)
        _migrate_graph_schema(conn)
        _migrate_core_schema(conn)
        return
    # Unvollstaendig: ALLES droppen (inkl. Collation) und neu laden
    _drop_public_schema(conn)
    sql = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    conn.execute(sql)
    _migrate_audit_schema(conn)
    _migrate_group_schema(conn)
    _migrate_graph_schema(conn)
    _migrate_core_schema(conn)


def _migrate_core_schema(conn) -> None:
    from kiara.web.core_schema_migration import migrate_core_schema

    migrate_core_schema(conn)


def _migrate_group_schema(conn) -> None:
    from kiara.web.group_schema_migration import migrate_group_schema

    migrate_group_schema(conn)


def _migrate_audit_schema(conn) -> None:
    conn.execute("ALTER TABLE public.audit_anonymization ALTER COLUMN user_id DROP NOT NULL")
    conn.execute(
        "ALTER TABLE public.audit_anonymization ADD COLUMN IF NOT EXISTS correlation_id text"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_anon_corr_id "
        "ON public.audit_anonymization USING btree (correlation_id) "
        "WHERE (correlation_id IS NOT NULL)"
    )


def _migrate_graph_schema(conn) -> None:
    from kiara.web.graph_schema_migration import migrate_graph_schema

    migrate_graph_schema(conn)


# Anzahl ``CREATE TABLE public.``-Statements in schema.sql (Stand: 48).
# Schutzschwelle — wenn weniger Tabellen existieren, ist das Schema defekt.
_EXPECTED_TABLE_COUNT = 48


def _drop_public_schema(conn) -> None:
    """Droppt alle Tabellen + die ``nocase``-Collation im ``public``-Schema.

    Wird vor einem Reload von schema.sql aufgerufen, damit die nicht-idempotenten
    ``CREATE TABLE`` / ``CREATE COLLATION``-Statements sauber laufen.
    """
    from psycopg2 import sql as psycopg2_sql

    rows = conn.execute(
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public'"
    ).fetchall()
    for row in rows:
        conn.execute(
            psycopg2_sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                psycopg2_sql.Identifier(row["tablename"])
            )
        )
    conn.execute("DROP COLLATION IF EXISTS public.nocase CASCADE")
