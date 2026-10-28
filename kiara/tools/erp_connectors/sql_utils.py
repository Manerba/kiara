"""Shared SQL-Hilfsfunktionen fuer ERP-Abfragen.

Extrahiert die generische Column-Mapping-basierte SQL-Abfrage-Logik,
die sowohl vom Legacy-Backend (``erp_backend.ERPBackend``) als auch vom
neuen Konnektor (``generic_sql.GenericSQLConnector``) verwendet wird.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from uuid import UUID

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_CANONICAL_ID_FIELDS = {
    "customers": "customer_id",
    "articles": "article_id",
    "orders": "order_id",
    "production_orders": "production_order_id",
}
_SUBTABLES = {
    "orders": ("positions_table", "positions_fk", "positions_columns", "positions"),
    "production_orders": (
        "operations_table",
        "operations_fk",
        "operations_columns",
        "operations",
    ),
}


def _placeholder(driver: str) -> str:
    """Gibt den DB-API-Platzhalter fuer den jeweiligen Treiber zurueck."""
    if driver == "mssql":
        return "?"
    return "%s"


def _quote_alias(name: str, driver: str) -> str:
    """Quotet einen Spalten-Alias treiberspezifisch."""
    if driver == "mssql":
        return f"[{name}]"
    if driver == "mysql":
        return f"`{name}`"
    return f'"{name}"'


def _validate_identifier(name: str, context: str) -> str:
    """Stellt sicher, dass ein SQL-Identifier sicher ist."""
    if not _SAFE_IDENTIFIER.match(name):
        raise ValueError(f"Ungueltiger SQL-Identifier in {context}: {name!r}")
    return name


def _to_json_primitive(value):
    """Konvertiert DB-API-Typen rekursiv in JSON-taugliche Primitives.

    psycopg2.extras.Json nutzt ``json.dumps`` ohne ``default``-Hook. Reale
    ERP-Stammdaten liefern Decimal/date/datetime/UUID/bytes aus
    NUMERIC-, DATE- oder UUID-Spalten — diese landen sonst unveraendert in
    den Bulk-Dicts und sprengen den Graph-Sync beim ``Json(properties)``-
    Adapt. Aufrufer behalten die rohen Cursor-Werte fuer SQL-Parameter
    (z.B. parent_ids fuer IN-Filter); fuer den Property-Pfad wird die
    Sanitisierung VOR der Weitergabe ans GraphRAG-Sink erzwungen.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        try:
            return float(value)
        except (ValueError, OverflowError):
            return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_json_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_primitive(v) for v in value]
    return str(value)


def sql_query_entity(
    dsn: str,
    driver: str,
    mapping: dict,
    entity_id: str,
    *,
    statement_timeout_seconds: float | None = None,
) -> dict | None:
    """Generische SQL-Abfrage mit Spalten-Mapping (synchron).

    Baut einen SELECT mit Alias-Spalten und optionalem Sub-Table Join
    (Positionen oder Arbeitsgaenge).

    Args:
        dsn: Verbindungsstring fuer den SQLConnector.
        driver: Treiber-ID (``"auto"``, ``"mssql"``, ``"postgresql"``, ...).
        mapping: Spalten-Mapping-Dict mit ``table``, ``id_column``,
            ``columns`` und optionalen Sub-Table-Keys.
        entity_id: Wert der Primaer-ID, nach der gesucht wird.

    Returns:
        Entity-Dictionary mit gemappten Spaltennamen oder None.
    """
    from kiara.connectors.sql_connector import SQLConnector

    connector = SQLConnector(
        dsn,
        driver,
        statement_timeout=statement_timeout_seconds,
    )
    drv = connector._driver

    table = mapping.get("table")
    id_col = mapping.get("id_column")
    col_mapping = mapping.get("columns")
    if not all((table, id_col, col_mapping)):
        missing = [k for k in ("table", "id_column", "columns") if k not in mapping]
        raise ValueError(f"ERP-Mapping unvollstaendig, fehlende Keys: {missing}")

    # Identifiers validieren (SQL-Injection-Schutz)
    _validate_identifier(table, "table")
    _validate_identifier(id_col, "id_column")
    for app_col, db_col in col_mapping.items():
        _validate_identifier(db_col, "columns.db_col")
        _validate_identifier(app_col, "columns.app_col")

    # SELECT mit gemappten Spalten (Alias: DB-Spalte AS App-Spalte)
    ph = _placeholder(drv)
    select_cols = ", ".join(
        f"{db_col} AS {_quote_alias(app_col, drv)}" for app_col, db_col in col_mapping.items()
    )
    query = f"SELECT {select_cols} FROM {table} WHERE {id_col} = {ph}"

    # Beide Queries (Haupt-Entity + Sub-Tabelle) auf einer Verbindung
    # ausfuehren, damit sie denselben Datenbank-Snapshot sehen.
    with connector.connection() as conn:
        cur = conn.cursor()
        cur.execute(query, (entity_id,))
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchmany(1)

        if not rows:
            return None

        entity = dict(zip(columns, rows[0], strict=False))

        # Sub-Tabellen-Logik: Laedt 1:N-Beziehungen (z.B. Auftragspositionen,
        # Fertigungsarbeitsgaenge) als verschachtelte Liste in das Entity-Dict.
        # Das Mapping kann entweder "positions_table/fk/columns" oder
        # "operations_table/fk/columns" enthalten — aber nicht beides.
        # Der Join erfolgt ueber den Fremdschluessel (fk) = entity_id.
        sub_table_key = None
        sub_fk_key = None
        sub_cols_key = None
        sub_result_key = None

        if "positions_table" in mapping:
            sub_table_key = "positions_table"
            sub_fk_key = "positions_fk"
            sub_cols_key = "positions_columns"
            sub_result_key = "positions"
        elif "operations_table" in mapping:
            sub_table_key = "operations_table"
            sub_fk_key = "operations_fk"
            sub_cols_key = "operations_columns"
            sub_result_key = "operations"

        if sub_table_key and all(k in mapping for k in (sub_table_key, sub_fk_key, sub_cols_key)):
            sub_mapping = mapping[sub_cols_key]
            _validate_identifier(mapping[sub_table_key], sub_table_key)
            _validate_identifier(mapping[sub_fk_key], sub_fk_key)
            for app_col, db_col in sub_mapping.items():
                _validate_identifier(db_col, f"{sub_cols_key}.db_col")
                _validate_identifier(app_col, f"{sub_cols_key}.app_col")
            sub_select = ", ".join(
                f"{db_col} AS {_quote_alias(app_col, drv)}"
                for app_col, db_col in sub_mapping.items()
            )
            sub_query = (
                f"SELECT {sub_select} FROM {mapping[sub_table_key]} "
                f"WHERE {mapping[sub_fk_key]} = {ph}"
            )
            cur.execute(sub_query, (entity_id,))
            sub_columns = [desc[0] for desc in cur.description] if cur.description else []
            entity[sub_result_key] = [
                dict(zip(sub_columns, row, strict=False)) for row in cur.fetchall()
            ]

    return entity


def sql_query_all_entities(
    dsn: str,
    driver: str,
    entity_type: str,
    mapping: dict,
    *,
    chunk_size: int = 1000,
    statement_timeout_seconds: float | None = None,
) -> list[dict]:
    """Generische SQL-Bulk-Abfrage mit Spalten-Mapping (synchron)."""
    from kiara.connectors.sql_connector import SQLConnector

    canonical_id = _CANONICAL_ID_FIELDS.get(entity_type)
    if not canonical_id:
        raise ValueError(f"Unbekannter ERP-Entity-Typ fuer Bulk: {entity_type!r}")

    connector = SQLConnector(
        dsn,
        driver,
        statement_timeout=statement_timeout_seconds,
    )
    drv = connector._driver
    table, id_col, col_mapping = _validate_mapping(mapping)
    _validate_identifier(table, "table")
    _validate_identifier(id_col, "id_column")
    for app_col, db_col in col_mapping.items():
        _validate_identifier(db_col, "columns.db_col")
        _validate_identifier(app_col, "columns.app_col")

    select_cols = [
        f"{db_col} AS {_quote_alias(app_col, drv)}" for app_col, db_col in col_mapping.items()
    ]
    internal_parent_alias = "__kiara_parent_id"
    select_cols.append(f"{id_col} AS {_quote_alias(internal_parent_alias, drv)}")

    results: list[dict] = []
    offset = 0
    with connector.connection() as conn:
        cur = conn.cursor()
        while True:
            query, params = _bulk_page_query(
                table,
                id_col,
                select_cols,
                drv,
                chunk_size,
                offset,
            )
            cur.execute(query, params)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            rows = cur.fetchall()
            if not rows:
                break

            batch_entities = []
            parent_ids = []
            for row in rows:
                raw_dict = dict(zip(columns, row, strict=False))
                parent_value = raw_dict.pop(internal_parent_alias, None)
                if parent_value is None:
                    parent_value = raw_dict.get(canonical_id)
                if parent_value is None or str(parent_value).strip() == "":
                    raise ValueError(f"ERP-SQL-Bulk '{entity_type}' ohne Primaer-ID-Wert.")
                if canonical_id not in raw_dict:
                    raw_dict[canonical_id] = parent_value
                if raw_dict.get(canonical_id) is None or str(raw_dict[canonical_id]).strip() == "":
                    raise ValueError(
                        f"ERP-SQL-Bulk '{entity_type}' ohne kanonisches Feld {canonical_id}."
                    )
                batch_entities.append(_to_json_primitive(raw_dict))
                parent_ids.append(parent_value)

            _attach_subtable_rows(cur, drv, entity_type, mapping, batch_entities, parent_ids)
            results.extend(batch_entities)
            if len(rows) < chunk_size:
                break
            offset += chunk_size

    return results


def _validate_mapping(mapping: dict) -> tuple[str, str, dict]:
    table = mapping.get("table")
    id_col = mapping.get("id_column")
    col_mapping = mapping.get("columns")
    if not all((table, id_col, col_mapping)):
        missing = [k for k in ("table", "id_column", "columns") if k not in mapping]
        raise ValueError(f"ERP-Mapping unvollstaendig, fehlende Keys: {missing}")
    if not isinstance(col_mapping, dict) or not col_mapping:
        raise ValueError("ERP-Mapping columns muss ein nicht-leeres Objekt sein.")
    return table, id_col, col_mapping


def _bulk_page_query(
    table: str,
    id_col: str,
    select_cols: list[str],
    driver: str,
    chunk_size: int,
    offset: int,
) -> tuple[str, tuple]:
    select_sql = ", ".join(select_cols)
    ph = _placeholder(driver)
    if driver == "mssql":
        sql = (
            f"SELECT {select_sql} FROM {table} ORDER BY {id_col} "
            f"OFFSET {ph} ROWS FETCH NEXT {ph} ROWS ONLY"
        )
        return sql, (offset, chunk_size)
    sql = f"SELECT {select_sql} FROM {table} ORDER BY {id_col} LIMIT {ph} OFFSET {ph}"
    return sql, (chunk_size, offset)


def _attach_subtable_rows(
    cur,
    driver: str,
    entity_type: str,
    mapping: dict,
    batch_entities: list[dict],
    parent_ids: list,
) -> None:
    sub = _SUBTABLES.get(entity_type)
    if not sub or not parent_ids:
        return
    sub_table_key, sub_fk_key, sub_cols_key, result_key = sub
    if not all(k in mapping for k in (sub_table_key, sub_fk_key, sub_cols_key)):
        return

    sub_table = mapping[sub_table_key]
    sub_fk = mapping[sub_fk_key]
    sub_mapping = mapping[sub_cols_key]
    if not isinstance(sub_mapping, dict) or not sub_mapping:
        return

    _validate_identifier(sub_table, sub_table_key)
    _validate_identifier(sub_fk, sub_fk_key)
    for app_col, db_col in sub_mapping.items():
        _validate_identifier(db_col, f"{sub_cols_key}.db_col")
        _validate_identifier(app_col, f"{sub_cols_key}.app_col")

    internal_fk_alias = "__kiara_parent_fk"
    select_cols = [
        f"{sub_fk} AS {_quote_alias(internal_fk_alias, driver)}",
        *[
            f"{db_col} AS {_quote_alias(app_col, driver)}"
            for app_col, db_col in sub_mapping.items()
        ],
    ]
    ph = _placeholder(driver)
    placeholders = ", ".join([ph] * len(parent_ids))
    query = (
        f"SELECT {', '.join(select_cols)} FROM {sub_table} "
        f"WHERE {sub_fk} IN ({placeholders}) ORDER BY {sub_fk}"
    )
    cur.execute(query, tuple(parent_ids))
    columns = [desc[0] for desc in cur.description] if cur.description else []
    grouped: dict[object, list[dict]] = {pid: [] for pid in parent_ids}
    for row in cur.fetchall():
        item = dict(zip(columns, row, strict=False))
        parent = item.pop(internal_fk_alias, None)
        grouped.setdefault(parent, []).append(_to_json_primitive(item))

    for entity, parent_id in zip(batch_entities, parent_ids, strict=False):
        entity[result_key] = grouped.get(parent_id, [])
