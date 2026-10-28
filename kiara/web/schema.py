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
    if existing >= _EXPECTED_TABLE_COUNT - 1:
        _migrate_external_identity_schema(conn)
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
        _migrate_external_identity_schema(conn)
        _migrate_index_contract_schema(conn)
        return
    # Unvollstaendig: ALLES droppen (inkl. Collation) und neu laden
    _drop_public_schema(conn)
    sql = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    conn.execute(sql)
    # pg_dump setzt den Session-Suchpfad auf leer. Call-Sites verwenden nach
    # ``create_tables`` weiterhin bewusst unqualifizierte Tabellennamen; daher
    # den fuer KIara erwarteten Suchpfad in derselben Session wiederherstellen.
    conn.execute("SET search_path TO public")
    _migrate_audit_schema(conn)
    _migrate_group_schema(conn)
    _migrate_graph_schema(conn)
    _migrate_core_schema(conn)
    _migrate_external_identity_schema(conn)
    _migrate_index_contract_schema(conn)


def _migrate_core_schema(conn) -> None:
    from kiara.web.core_schema_migration import migrate_core_schema

    migrate_core_schema(conn)


def _migrate_external_identity_schema(conn) -> None:
    from kiara.web.external_identity_migration import migrate_external_identity_schema

    migrate_external_identity_schema(conn)


def _migrate_index_contract_schema(conn) -> None:
    from kiara.web.index_contract_migration import migrate_index_contract_schema

    migrate_index_contract_schema(conn)


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


# Anzahl der vor S4 vorhandenen Basistabellen. Die beiden S4-Tabellen werden
# additiv durch ``_migrate_index_contract_schema`` angelegt und duerfen deshalb
# den Vollstaendigkeits-Guard nicht anheben: Eine laufende S3-Installation hat
# beim ersten S4-Start definitionsgemaess noch genau diese 49 Tabellen.
_EXPECTED_TABLE_COUNT = 49


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
    # Die Trigger-Funktion ist kein Tabellenobjekt und ueberlebt deshalb den
    # Tabellen-Drop. Ohne den expliziten Drop scheitert ein anschliessender
    # Reload des pg_dump-Schemas mit DuplicateFunction.
    conn.execute("DROP FUNCTION IF EXISTS public.set_user_external_identities_updated_at() CASCADE")
    conn.execute("DROP COLLATION IF EXISTS public.nocase CASCADE")
