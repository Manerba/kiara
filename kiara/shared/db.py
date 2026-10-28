"""Gemeinsame DB-Verbindungsverwaltung fuer alle KIara-Module.

Stellt eine einheitliche Connection-Factory, Context-Manager und
SQL-Hilfsfunktionen bereit, die von Core-Modulen (graph, migrations,
anonymization, cloud) und Web-Modulen gleichermassen genutzt
werden koennen.

Seit der kiara-db-Extraktion werden Connection-Management und
Repositories von ``kiara_db`` bereitgestellt. Dieses Modul:
  - Re-exportiert die Connection-API aus ``kiara_db.connection``
  - Registriert einen DSN-Provider fuer Lazy-Init
  - Behaelt kiara-spezifische Utilities (Whitelists, CRUD-Helpers, Pagination)

Backend: PostgreSQL (psycopg2 mit RealDictCursor).

Beziehungen:
    - kiara.db                 -> Re-Export von get_connection, connect_*
    - kiara.web.db_utils       -> Re-Export von db_query, db_transaction, ...
    - kiara.web.database       -> nutzt get_db()
    - kiara.graph.store        -> nutzt connect_query / connect_transaction
    - kiara.anonymization.*    -> nutzt db_query / db_transaction
    - kiara.cloud.audit        -> nutzt db_query / db_transaction
    - kiara.generation.chain   -> nutzt db_query (lazy)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =========================================================================
# Re-Exports aus kiara_db.connection
# =========================================================================

from kiara_db.connection import (  # noqa: F401
    ConnectionWrapper,
    close_pool,
    configure_pool,
    connect_query,
    connect_transaction,
    db_query,
    db_transaction,
    get_connection,
    get_db,
    init_db,
    pool_stats,
    register_dsn_provider,
    set_statement_timeout_ms,
)

# =========================================================================
# Lokales DSN-Singleton + Lazy-Init-Provider
# =========================================================================

_dsn: str | None = None


def init_db_backend(config: dict) -> None:
    """Initialisiert das Datenbank-Backend aus der Config.

    Muss von jedem Entry-Point aufgerufen werden (app.py, admin_app.py,
    cli serve-worker, cli serve-admin).
    Ueberschreibt NICHT einen bereits gesetzten Test-DSN (kiara_test).
    """
    global _dsn
    if _dsn is not None and "kiara_test" in _dsn:
        return
    db_cfg = config.get("database", {})
    pg_cfg = db_cfg.get("postgresql", {})
    dsn = pg_cfg.get("dsn")
    if not dsn:
        raise ValueError("database.postgresql.dsn muss in kiara.yaml gesetzt sein")
    set_statement_timeout_ms(pg_cfg.get("statement_timeout_ms", 30_000))
    configure_pool(
        max_size=pg_cfg.get("pool_max_size", 16),
        wait_seconds=pg_cfg.get("pool_wait_seconds", 5),
    )
    init_db(dsn, force=True)
    try:
        from kiara.web.core_schema_migration import migrate_core_schema
        from kiara.web.external_identity_migration import migrate_external_identity_schema
        from kiara.web.graph_schema_migration import migrate_graph_schema
        from kiara.web.group_schema_migration import migrate_group_schema
        from kiara.web.index_contract_migration import migrate_index_contract_schema

        with db_transaction() as migration_conn:
            # Service-Starts laufen parallel. Schema-DDL und der gemeinsame
            # Advisory Lock duerfen nicht am normalen 30-s-Querytimeout
            # scheitern; SET LOCAL endet atomar mit dieser Transaktion.
            migration_conn.execute("SET LOCAL statement_timeout = '5min'")
            migration_conn.execute("SELECT pg_advisory_xact_lock(42)")
            migrate_group_schema(migration_conn)
            migrate_graph_schema(migration_conn)
            migrate_core_schema(migration_conn)
            migrate_external_identity_schema(migration_conn)
            migrate_index_contract_schema(migration_conn)
    except Exception:
        logger.exception("Schema-Migration nach DB-Init fehlgeschlagen")
        raise
    _dsn = dsn


def _lazy_dsn_provider() -> str | None:
    """Kompatibilitaet fuer vergessene init_db_backend()-Aufrufe und Tests mit _dsn."""
    global _dsn
    if _dsn is not None:
        return _dsn
    try:
        from kiara.config import load_config

        config = load_config()
        init_db_backend(config)
    except Exception:
        return None
    return _dsn


def _is_test_db() -> bool:
    """Prueft ob der aktuelle DSN auf die Test-DB zeigt."""
    return _dsn is not None and "kiara_test" in _dsn


def get_db_backend() -> str:
    """Gibt den Backend-Typ zurueck (immer 'postgresql')."""
    return "postgresql"


# DSN-Provider registrieren — Side-Effect beim Import dieses Moduls
register_dsn_provider(_lazy_dsn_provider)


# ---------------------------------------------------------------------------
# SQL-Injection-Schutz & Hilfsfunktionen
# ---------------------------------------------------------------------------

ALLOWED_TABLES: set[str] = {
    "users",
    "ldap_config",
    "mail_config",
    "datasource_config",
    "onlyoffice_config",
    "ai_config",
    "datasources",
    "groups",
    "user_groups",
    "datasource_groups",
    "audit_anonymization",
    "cloud_audit_log",
    "query_log",
    "workflow_exports",
    "health_log",
    "schema_version",
    "ki_backends",
    "prompt_templates",
}
"""Erlaubte Tabellennamen fuer dynamisches SQL (Whitelist)."""

ALLOWED_COLUMNS: set[str] = {
    "id",
    "timestamp",
    "session_id",
    "password",
    "bind_password",
    "jwt_secret",
    "server_url",
    "document_server_url",
    "wopi_external_base_url",
    "proof_validation_enabled",
    "embedding_url",
    "llm_url",
    "embedding_model",
    "llm_model",
    "prompt_template",
    "username",
    "webdav_url",
    "webdav_username",
    "webdav_password",
    "webdav_base_path",
    "smb_server",
    "smb_share",
    "smb_username",
    "smb_password",
    "smb_domain",
    "smb_mount_point",
    "type",
    "port",
    "bind_dn",
    "base_dn",
    "user_filter",
    "tls_mode",
    "selected_group",
    "smtp_server",
    "smtp_port",
    "encryption",
    "sender_address",
    "recipient_addresses",
    "notify_ollama_down",
    "notify_datasource_down",
    "notify_index_error",
    "notify_degradation",
    "status_report_enabled",
    "status_report_interval",
    "name",
    "path",
    "is_active",
    "sid",
    "cn",
    "dn",
    "member_count",
    "imported_at",
    "user_id",
    "group_id",
    "datasource_id",
    "cloud_enabled",
    "cloud_provider",
    "cloud_api_key",
    "cloud_model",
    "cloud_base_url",
    "cloud_mode",
    "anon_enabled",
    "anon_erp_enabled",
    "anon_erp_path",
    "anon_ldap_enabled",
    "anon_word_boundaries",
    "anon_min_term_length",
    "anon_excluded_terms",
    "anon_custom_patterns",
    "anon_audit_log",
    "anon_custom_entries",
    "restriction_level",
    "trusted",
    "question_anonymized",
    "answer_anonymized",
    "provider",
    "model",
    "tokens_in",
    "tokens_out",
    "entities_anonymized",
    "duration_ms",
    "question_hash",
    "question_excerpt",
    "answer_excerpt",
    "entity_count",
    "entity_types",
    "route",
    "content_class",
    "correlation_id",
    "routed_to",
    "model_used",
    "sources_count",
    "tools_used",
    "chunks_retrieved",
    "filename",
    "format",
    "title",
    "size_bytes",
    "template_used",
    "status",
    "components_json",
    "degradation_mode",
    "version",
    "description",
    "applied_at",
    "category",
    "url",
    "base_url",
    "api_key",
    "extra_config",
    "last_test_at",
    "last_test_ok",
    "last_test_error",
    "priority",
    "created_at",
    "updated_at",
    "label",
    "content",
    "chain_type",
    "is_cloud",
    "show_in_chat",
    "badge_color",
    "sort_order",
    "is_system",
}
"""Erlaubte Spaltennamen fuer dynamisches SQL (Whitelist)."""

_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUALIFIED_COLUMN_RE = re.compile(
    r"^(?:(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.)?"
    r"(?P<column>[A-Za-z_][A-Za-z0-9_]*)$"
)
_TABLE_REF_RE = re.compile(
    r"^(?P<table>[A-Za-z_][A-Za-z0-9_]*)(?:\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?$"
)
_JOIN_RE = re.compile(
    r"^(?P<kind>LEFT|INNER|RIGHT)\s+JOIN\s+"
    r"(?P<table>[A-Za-z_][A-Za-z0-9_]*)(?:\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?\s+"
    r"ON\s+(?P<left>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*=\s*"
    r"(?P<right>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)$",
    re.IGNORECASE,
)
_CONDITION_RE = re.compile(
    r"^(?P<col>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)"
    r"(?:(?:\s*(?P<op>=|!=|<>|<=|>=|<|>|LIKE|ILIKE)\s*%s)"
    r"|(?:\s+IS\s+(?:NOT\s+)?NULL)"
    r"|(?:\s+IN\s*\(%s(?:\s*,\s*%s)*\)))$",
    re.IGNORECASE,
)


def _validate_alias(alias: str | None, known_aliases: set[str], label: str) -> str | None:
    if not alias:
        return None
    if not _SQL_IDENTIFIER_RE.match(alias):
        raise ValueError(f"Ungueltiger SQL-Bezeichner in {label}: {alias!r}")
    known_aliases.add(alias)
    return alias


def _validate_column_ref(ref: str, known_aliases: set[str], label: str) -> str:
    match = _QUALIFIED_COLUMN_RE.match(ref.strip())
    if not match:
        raise ValueError(f"Ungueltiger SQL-Bezeichner in {label}: {ref!r}")
    alias = match.group("alias")
    column = match.group("column")
    if alias and alias not in known_aliases:
        raise ValueError(f"Ungueltiger SQL-Bezeichner in {label}: {ref!r}")
    validate_identifier(column, ALLOWED_COLUMNS, "column")
    return ref.strip()


def _validate_table_ref(table_ref: str, label: str, known_aliases: set[str]) -> str:
    match = _TABLE_REF_RE.match(table_ref.strip())
    if not match:
        raise ValueError(f"Ungueltiger SQL-Bezeichner in {label}: {table_ref!r}")
    table_name = validate_identifier(match.group("table"), ALLOWED_TABLES, "table")
    alias = _validate_alias(match.group("alias"), known_aliases, label)
    if alias:
        return f"{table_name} {alias}"
    known_aliases.add(table_name)
    return table_name


def _validate_join_clause(joins: str, known_aliases: set[str]) -> str:
    if not joins:
        return ""
    match = _JOIN_RE.match(joins.strip())
    if not match:
        raise ValueError(f"Ungueltiger SQL-Bezeichner in joins: {joins!r}")
    table_name = validate_identifier(match.group("table"), ALLOWED_TABLES, "table")
    alias = match.group("alias") or table_name
    _validate_alias(alias, known_aliases, "joins")
    left = _validate_column_ref(match.group("left"), known_aliases, "joins")
    right = _validate_column_ref(match.group("right"), known_aliases, "joins")
    kind = match.group("kind").upper()
    if match.group("alias"):
        return f"{kind} JOIN {table_name} {alias} ON {left} = {right}"
    return f"{kind} JOIN {table_name} ON {left} = {right}"


def _validate_columns_clause(columns: str, known_aliases: set[str]) -> str:
    if columns == "*":
        return columns
    parts: list[str] = []
    for raw_part in columns.split(","):
        part = raw_part.strip()
        if part.upper() == "COUNT(*)":
            parts.append("COUNT(*)")
            continue
        if part.endswith(".*"):
            alias = part[:-2]
            if alias not in known_aliases:
                raise ValueError(f"Ungueltiger SQL-Bezeichner in columns: {columns!r}")
            parts.append(part)
            continue
        parts.append(_validate_column_ref(part, known_aliases, "columns"))
    return ", ".join(parts)


def _validate_conditions_clause(conditions: list[str], known_aliases: set[str]) -> list[str]:
    validated = []
    for cond in conditions:
        cleaned = cond.strip()
        match = _CONDITION_RE.match(cleaned)
        if not match:
            raise ValueError(f"Ungueltiges SQL-Condition-Fragment: {cond!r}")
        _validate_column_ref(match.group("col"), known_aliases, "conditions")
        validated.append(cleaned)
    return validated


def _validate_order_by_clause(order_by: str, known_aliases: set[str]) -> str:
    parts = []
    for raw_part in order_by.split(","):
        tokens = raw_part.strip().split()
        if not tokens or len(tokens) > 2:
            raise ValueError(f"Ungueltiger SQL-Bezeichner in order_by: {order_by!r}")
        col = _validate_column_ref(tokens[0], known_aliases, "order_by")
        if len(tokens) == 2:
            direction = tokens[1].upper()
            if direction not in {"ASC", "DESC"}:
                raise ValueError(f"Ungueltiger SQL-Bezeichner in order_by: {order_by!r}")
            parts.append(f"{col} {direction}")
        else:
            parts.append(col)
    return ", ".join(parts)


def get_singleton_config(db, table: str):
    """Liest die einzige Konfigurationszeile (id=1) einer Singleton-Tabelle."""
    validate_identifier(table, ALLOWED_TABLES, "table")
    return db.execute(f"SELECT * FROM {table} WHERE id = 1").fetchone()


def get_datasource(db, ds_id: int):
    """Liest eine Datasource anhand ihrer ID."""
    return db.execute(
        "SELECT * FROM datasources WHERE id = %s",
        (ds_id,),
    ).fetchone()


def validate_identifier(value: str, allowed: set[str], kind: str = "identifier") -> str:
    """Prueft ob ein SQL-Bezeichner in der Allowlist enthalten ist."""
    if value not in allowed:
        raise ValueError(f"Invalid SQL {kind}: {value}")
    return value


def insert_row(db, table: str, fields: dict[str, Any], returning: str = "id") -> int:
    """Fuegt eine Zeile ein und gibt die neue Row-ID zurueck (RETURNING)."""
    validate_identifier(table, ALLOWED_TABLES, "table")
    for col in fields:
        validate_identifier(col, ALLOWED_COLUMNS, "column")
    if returning != "id":
        validate_identifier(returning, ALLOWED_COLUMNS, "column")
    cols = ", ".join(fields.keys())
    placeholders = ", ".join(["%s"] * len(fields))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) RETURNING {returning}"
    cursor = db.execute(sql, list(fields.values()))
    return cursor.fetchone()[returning]


def update_row(db, table: str, row_id: int, fields: dict[str, Any]) -> None:
    """Aktualisiert eine Zeile anhand ihrer ID."""
    validate_identifier(table, ALLOWED_TABLES, "table")
    for col in fields:
        validate_identifier(col, ALLOWED_COLUMNS, "column")
    set_clause = ", ".join(f"{k} = %s" for k in fields)
    db.execute(
        f"UPDATE {table} SET {set_clause} WHERE id = %s",
        [*list(fields.values()), row_id],
    )


def upsert_singleton_config(db, table: str, fields: dict[str, Any]) -> None:
    """Aktualisiert oder erstellt eine Singleton-Konfigurationszeile (id=1)."""
    validate_identifier(table, ALLOWED_TABLES, "table")
    for col in fields:
        validate_identifier(col, ALLOWED_COLUMNS, "column")

    existing = db.execute(f"SELECT id FROM {table} WHERE id = 1").fetchone()

    if existing:
        set_clause = ", ".join(f"{k} = %s" for k in fields)
        db.execute(f"UPDATE {table} SET {set_clause} WHERE id = 1", list(fields.values()))
    else:
        cols = ", ".join(["id", *list(fields.keys())])
        placeholders = ", ".join(["1"] + ["%s"] * len(fields))
        db.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", list(fields.values()))


# ---------------------------------------------------------------------------
# Paginierte Abfragen
# ---------------------------------------------------------------------------


def paginated_query(
    db,
    table: str,
    columns: str = "*",
    conditions: list[str] | None = None,
    params: list | None = None,
    order_by: str = "id DESC",
    page: int = 1,
    page_size: int = 50,
    joins: str = "",
) -> dict:
    """Fuehrt eine paginierte Query aus und gibt data + total zurueck."""
    if params is None:
        params = []
    if conditions is None:
        conditions = []

    known_aliases: set[str] = set()
    table = _validate_table_ref(table, "table", known_aliases)
    joins = _validate_join_clause(joins, known_aliases)
    columns = _validate_columns_clause(columns, known_aliases)
    conditions = _validate_conditions_clause(conditions, known_aliases)
    order_by = _validate_order_by_clause(order_by, known_aliases)

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    offset = (page - 1) * page_size

    from_clause = f"{table} {joins}" if joins else table

    total_row = db.execute(
        f"SELECT COUNT(*) AS count FROM {from_clause}{where}",
        params,
    ).fetchone()
    total = total_row["count"] if total_row else 0

    rows = db.execute(
        f"SELECT {columns} FROM {from_clause}{where} ORDER BY {order_by} LIMIT %s OFFSET %s",
        [*params, page_size, offset],
    ).fetchall()

    return {
        "data": [dict(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }
