"""Tests fuer ERP-SQL-Bulk-Helper."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from kiara.tools.erp_connectors.sql_utils import sql_query_all_entities


def _desc(columns):
    return [(c,) for c in columns]


def test_sql_query_all_entities_orders_with_positions():
    cursor = MagicMock()
    descriptions = [
        _desc(["order_id", "customer_id"]),
        _desc(["__kiara_parent_fk", "pos", "article_id"]),
        [],
    ]
    fetchall_results = [
        [("A-1", "K-1"), ("A-2", "K-2")],
        [("A-1", 1, "ART-1"), ("A-2", 1, "ART-2")],
        [],
    ]
    call_count = {"n": 0}

    def execute(_query, _params=None):
        idx = call_count["n"]
        cursor.description = descriptions[idx]
        call_count["n"] += 1

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = fetchall_results

    conn = MagicMock()
    conn.cursor.return_value = cursor
    connector = MagicMock()
    connector._driver = "postgresql"

    @contextmanager
    def connection():
        yield conn

    connector.connection = connection
    mapping = {
        "table": "orders",
        "id_column": "order_no",
        "columns": {"order_id": "order_no", "customer_id": "customer_no"},
        "positions_table": "positions",
        "positions_fk": "order_no",
        "positions_columns": {"pos": "pos_no", "article_id": "article_no"},
    }

    with patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector):
        rows = sql_query_all_entities("dsn", "postgresql", "orders", mapping)

    assert rows == [
        {
            "order_id": "A-1",
            "customer_id": "K-1",
            "positions": [{"pos": 1, "article_id": "ART-1"}],
        },
        {
            "order_id": "A-2",
            "customer_id": "K-2",
            "positions": [{"pos": 1, "article_id": "ART-2"}],
        },
    ]
    first_query = cursor.execute.call_args_list[0].args[0]
    assert "LIMIT" in first_query


def test_sql_query_all_entities_groups_subtables_by_internal_id_column():
    cursor = MagicMock()
    descriptions = [
        _desc(["order_id", "customer_id", "__kiara_parent_id"]),
        _desc(["__kiara_parent_fk", "pos", "article_id"]),
        [],
    ]
    fetchall_results = [
        [("A-1", "K-1", 101)],
        [(101, 1, "ART-1")],
        [],
    ]
    call_count = {"n": 0}

    def execute(_query, _params=None):
        idx = call_count["n"]
        cursor.description = descriptions[idx]
        call_count["n"] += 1

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = fetchall_results

    conn = MagicMock()
    conn.cursor.return_value = cursor
    connector = MagicMock()
    connector._driver = "postgresql"

    @contextmanager
    def connection():
        yield conn

    connector.connection = connection
    mapping = {
        "table": "orders",
        "id_column": "internal_id",
        "columns": {"order_id": "display_no", "customer_id": "customer_no"},
        "positions_table": "positions",
        "positions_fk": "order_internal_id",
        "positions_columns": {"pos": "pos_no", "article_id": "article_no"},
    }

    with patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector):
        rows = sql_query_all_entities("dsn", "postgresql", "orders", mapping)

    assert rows[0]["order_id"] == "A-1"
    assert rows[0]["positions"] == [{"pos": 1, "article_id": "ART-1"}]
    main_query = cursor.execute.call_args_list[0].args[0]
    sub_params = cursor.execute.call_args_list[1].args[1]
    assert 'internal_id AS "__kiara_parent_id"' in main_query
    assert sub_params == (101,)


# ---------------------------------------------------------------------------
# Issue #1044: V28-Testplan §3 — Multi-Driver + Fehlerpfade
# ---------------------------------------------------------------------------


def _build_connector(driver):
    """Mini-Helper: SQLConnector-Mock + Cursor + Connection-Context."""
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cursor
    connector = MagicMock()
    connector._driver = driver

    @contextmanager
    def connection():
        yield conn

    connector.connection = connection
    return connector, cursor


def _seed_cursor(cursor, descriptions, fetchall_results):
    """Konfiguriert cursor.execute so dass pro Aufruf die n-te Description +
    der n-te fetchall-Wert gesetzt werden."""
    call_count = {"n": 0}

    def execute(_query, _params=None):
        idx = call_count["n"]
        cursor.description = descriptions[idx]
        call_count["n"] += 1

    cursor.execute.side_effect = execute
    cursor.fetchall.side_effect = fetchall_results


def test_sql_bulk_mssql_uses_offset_fetch_syntax():
    """#1044 MSSQL: Pagination ueber OFFSET ? ROWS FETCH NEXT ? ROWS ONLY."""
    connector, cursor = _build_connector("mssql")
    _seed_cursor(
        cursor,
        descriptions=[_desc(["customer_id", "name"])],
        fetchall_results=[[]],
    )
    mapping = {
        "table": "customers",
        "id_column": "customer_no",
        "columns": {"customer_id": "customer_no", "name": "customer_name"},
    }
    with patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector):
        rows = sql_query_all_entities("dsn", "mssql", "customers", mapping)

    assert rows == []
    first_query = cursor.execute.call_args_list[0].args[0]
    assert "OFFSET" in first_query and "FETCH NEXT" in first_query
    assert "ROWS ONLY" in first_query
    # MSSQL-Reihenfolge: OFFSET zuerst, dann FETCH → Params (0, chunk_size)
    first_params = cursor.execute.call_args_list[0].args[1]
    assert first_params[0] == 0  # offset
    assert first_params[1] == 1000  # default chunk_size


def test_sql_bulk_mysql_uses_limit_offset():
    """#1044 MySQL: Pagination ueber LIMIT ? OFFSET ?."""
    connector, cursor = _build_connector("mysql")
    _seed_cursor(
        cursor,
        descriptions=[_desc(["customer_id", "name"])],
        fetchall_results=[[]],
    )
    mapping = {
        "table": "kunden",
        "id_column": "kdnr",
        "columns": {"customer_id": "kdnr", "name": "firma"},
    }
    with patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector):
        sql_query_all_entities("dsn", "mysql", "customers", mapping)

    first_query = cursor.execute.call_args_list[0].args[0]
    assert "LIMIT" in first_query and "OFFSET" in first_query
    assert "FETCH" not in first_query
    # MySQL/PG-Reihenfolge: LIMIT zuerst, dann OFFSET → Params (chunk_size, 0)
    first_params = cursor.execute.call_args_list[0].args[1]
    assert first_params[0] == 1000  # chunk_size
    assert first_params[1] == 0  # offset


def test_sql_bulk_unknown_entity_type_raises():
    """#1044 Unbekannter entity_type → ValueError VOR DB-Connect."""
    import pytest as _pt

    with _pt.raises(ValueError, match="Unbekannter ERP-Entity-Typ"):
        sql_query_all_entities(
            "dsn",
            "postgresql",
            "unknown-type",
            {
                "table": "x",
                "id_column": "x",
                "columns": {"x": "x"},
            },
        )


def test_sql_bulk_missing_mapping_raises():
    """#1044 Unvollstaendiges Mapping → ValueError."""
    import pytest as _pt

    with _pt.raises(ValueError, match="Mapping unvollstaendig"):
        sql_query_all_entities(
            "dsn",
            "postgresql",
            "customers",
            {
                "table": "x",  # id_column + columns fehlen
            },
        )


def test_sql_bulk_row_without_canonical_id_raises():
    """#1044 Row ohne Primaer-ID → ValueError beim Normalisieren."""
    import pytest as _pt

    connector, cursor = _build_connector("postgresql")
    _seed_cursor(
        cursor,
        descriptions=[_desc(["customer_id", "name"])],
        # DB liefert eine Row mit leerem canonical_id (customer_id)
        # UND leerem __kiara_parent_id — das loest ValueError aus
        fetchall_results=[[("", "NamedCompany")]],
    )
    mapping = {
        "table": "customers",
        "id_column": "customer_no",
        "columns": {"customer_id": "customer_no", "name": "firma"},
    }
    with (
        patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector),
        _pt.raises(ValueError, match="ohne Primaer-ID-Wert"),
    ):
        sql_query_all_entities("dsn", "postgresql", "customers", mapping)


def test_sql_bulk_empty_first_chunk_returns_empty_list():
    """#1044 Leere erste Page → keine Iteration, leere Liste."""
    connector, cursor = _build_connector("postgresql")
    _seed_cursor(
        cursor,
        descriptions=[_desc(["order_id", "customer_id"])],
        fetchall_results=[[]],
    )
    mapping = {
        "table": "orders",
        "id_column": "order_no",
        "columns": {"order_id": "order_no", "customer_id": "customer_no"},
    }
    with patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector):
        rows = sql_query_all_entities("dsn", "postgresql", "orders", mapping)

    assert rows == []
    # Kein zweiter Query (keine Sub-Tabelle, keine naechste Page)
    assert cursor.execute.call_count == 1


def test_sql_bulk_sanitizes_db_types_for_graph_json():
    """H-007: Decimal/date/datetime/UUID/bytes aus realen DB-Cursors duerfen
    NICHT als rohe Typen in batch_entities landen, sonst sprengt Json(props)
    den Graph-Sync. parent_ids fuer SQL-IN-Filter MUESSEN dagegen raw bleiben."""
    import json
    from datetime import date, datetime
    from decimal import Decimal
    from uuid import UUID

    connector, cursor = _build_connector("postgresql")
    order_uuid = UUID("12345678-1234-5678-1234-567812345678")
    descriptions = [
        _desc(["order_id", "customer_id", "order_date", "total_amount", "ref_uuid"]),
        _desc(["__kiara_parent_fk", "pos", "article_id", "qty"]),
        [],
    ]
    fetchall_results = [
        [
            (
                "A-1",
                "K-1",
                date(2026, 4, 30),
                Decimal("123.45"),
                order_uuid,
            ),
            (
                "A-2",
                "K-2",
                datetime(2026, 4, 30, 12, 0, 0),
                Decimal("99999999.99"),
                None,
            ),
        ],
        [
            ("A-1", 1, "ART-1", Decimal("2.5")),
            ("A-2", 1, "ART-2", Decimal("7.0")),
        ],
        [],
    ]
    _seed_cursor(cursor, descriptions, fetchall_results)

    mapping = {
        "table": "orders",
        "id_column": "order_no",
        "columns": {
            "order_id": "order_no",
            "customer_id": "customer_no",
            "order_date": "ts",
            "total_amount": "amount",
            "ref_uuid": "uuid_col",
        },
        "positions_table": "positions",
        "positions_fk": "order_no",
        "positions_columns": {
            "pos": "pos_no",
            "article_id": "article_no",
            "qty": "quantity",
        },
    }

    with patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector):
        rows = sql_query_all_entities("dsn", "postgresql", "orders", mapping)

    assert rows[0]["order_date"] == "2026-04-30"
    assert rows[0]["total_amount"] == 123.45
    assert rows[0]["ref_uuid"] == str(order_uuid)
    assert rows[1]["order_date"] == "2026-04-30T12:00:00"
    assert rows[1]["total_amount"] == 99999999.99
    assert rows[1]["ref_uuid"] is None
    assert rows[0]["positions"][0]["qty"] == 2.5
    assert rows[1]["positions"][0]["qty"] == 7.0

    json.dumps(rows)

    sub_params = cursor.execute.call_args_list[1].args[1]
    assert sub_params == ("A-1", "A-2")


def test_sql_bulk_production_orders_with_operations():
    """#1044 production_orders + operations Subtabelle funktioniert analog zu orders.positions."""
    connector, cursor = _build_connector("postgresql")
    descriptions = [
        _desc(["production_order_id", "order_reference"]),
        _desc(["__kiara_parent_fk", "step", "work_center"]),
        [],
    ]
    fetchall_results = [
        [("FA-1", "A-1")],
        [("FA-1", 1, "WC-A"), ("FA-1", 2, "WC-B")],
        [],
    ]
    _seed_cursor(cursor, descriptions, fetchall_results)
    mapping = {
        "table": "prod_orders",
        "id_column": "prod_no",
        "columns": {
            "production_order_id": "prod_no",
            "order_reference": "ref_order_no",
        },
        "operations_table": "prod_operations",
        "operations_fk": "prod_no",
        "operations_columns": {"step": "step_no", "work_center": "wc_code"},
    }
    with patch("kiara.connectors.sql_connector.SQLConnector", return_value=connector):
        rows = sql_query_all_entities("dsn", "postgresql", "production_orders", mapping)

    assert len(rows) == 1
    assert rows[0]["production_order_id"] == "FA-1"
    assert rows[0]["operations"] == [
        {"step": 1, "work_center": "WC-A"},
        {"step": 2, "work_center": "WC-B"},
    ]
