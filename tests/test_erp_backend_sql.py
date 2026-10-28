"""Tests fuer das ERP-Backend mit SQL-Anbindung (Phase 6)."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from kiara.tools.erp_backend import ERPBackend

# -- Vordefinierte SQL-Ergebnisse (simulieren SQLConnector-Cursor) --
# sql_utils.sql_query_entity() nutzt connector.connection() als Context-Manager
# und ruft cursor.execute() + cursor.fetchmany()/fetchall() direkt auf.
# Deshalb wird SQLConnector in sql_utils gepacht (nicht in sql_connector).

_ORDER_COLUMNS = [
    "order_id",
    "customer_id",
    "customer_name",
    "status",
    "order_date",
    "delivery_date",
    "total_amount",
]
_ORDER_ROW = (
    "A-1001",
    "K-1001",
    "Musterfirma GmbH",
    "In Bearbeitung",
    "2024-01-15",
    "2024-02-28",
    12500.00,
)

_POSITIONS_COLUMNS = ["pos", "article_id", "description", "quantity", "unit_price"]
_POSITIONS_ROWS = [
    (1, "ART-10001", "Widget A", 100, 50.00),
    (2, "ART-10002", "Widget B", 50, 150.00),
]

_ARTICLE_COLUMNS = [
    "article_id",
    "description",
    "category",
    "stock_quantity",
    "unit",
    "warehouse",
    "unit_price",
    "min_stock",
    "supplier",
    "lead_time_days",
]
_ARTICLE_ROW = (
    "ART-10001",
    "Widget A",
    "Bauteile",
    500,
    "Stueck",
    "Hauptlager",
    50.00,
    100,
    "Lieferant X",
    14,
)

_CUSTOMER_COLUMNS = ["customer_id", "name", "customer_since", "payment_terms", "credit_limit"]
_CUSTOMER_ROW = ("K-1001", "Musterfirma GmbH", "2020-01-01", "30 Tage netto", 50000.00)


def _make_cursor_description(columns):
    """Erzeugt eine cursor.description-Liste aus Spaltennamen."""
    return [(col,) for col in columns]


def _make_connector_mock(main_columns, main_row, sub_columns=None, sub_rows=None):
    """Erzeugt einen SQLConnector-Mock mit connection()-Context-Manager.

    sql_utils ruft:
      1. cur.execute(main_query, (entity_id,))
      2. cur.description -> [('col',), ...]
      3. cur.fetchmany(1) -> [main_row]
      4. (optional) cur.execute(sub_query, (entity_id,))
      5. (optional) cur.description -> sub_columns
      6. (optional) cur.fetchall() -> sub_rows

    Der Cursor gibt die Ergebnisse in der richtigen Reihenfolge zurueck.
    """
    cursor = MagicMock()
    # description-Sequenz: zuerst Haupt-Tabelle, dann ggf. Sub-Tabelle
    if sub_columns is not None:
        cursor.description = _make_cursor_description(main_columns)
        desc_values = [
            _make_cursor_description(main_columns),
            _make_cursor_description(sub_columns),
        ]
        cursor.description = desc_values[0]
        call_count = [0]
        original_execute = cursor.execute

        def execute_side_effect(query, params=None):
            call_count[0] += 1
            if call_count[0] == 2:
                cursor.description = desc_values[1]

        cursor.execute.side_effect = execute_side_effect
        cursor.fetchmany.return_value = [main_row]
        cursor.fetchall.return_value = sub_rows or []
    else:
        cursor.description = _make_cursor_description(main_columns)
        cursor.fetchmany.return_value = [main_row]

    conn = MagicMock()
    conn.cursor.return_value = cursor

    mock_connector = MagicMock()
    mock_connector._driver = "postgresql"

    @contextmanager
    def fake_connection():
        yield conn

    mock_connector.connection = fake_connection
    return mock_connector


def _make_empty_connector():
    """Connector-Mock der keine Zeilen zurueckgibt (NOT FOUND)."""
    cursor = MagicMock()
    cursor.description = []
    cursor.fetchmany.return_value = []

    conn = MagicMock()
    conn.cursor.return_value = cursor

    mock_connector = MagicMock()
    mock_connector._driver = "postgresql"

    @contextmanager
    def fake_connection():
        yield conn

    mock_connector.connection = fake_connection
    return mock_connector


def _make_config():
    """Erzeugt die KIara-Config mit SQL-ERP-Backend (kein echtes DSN noetig)."""
    return {
        "tools": {
            "enabled": True,
            "erp": {
                "backend": "sql",
                "sql_dsn": "mock://erp",
                "sql_driver": "postgresql",
                "mapping": {
                    "orders": {
                        "table": "auftraege",
                        "id_column": "Auftragsnummer",
                        "columns": {
                            "order_id": "Auftragsnummer",
                            "customer_id": "Kundennummer",
                            "customer_name": "Kundenname",
                            "status": "Auftragsstatus",
                            "order_date": "Bestelldatum",
                            "delivery_date": "Liefertermin",
                            "total_amount": "Gesamtbetrag",
                        },
                        "positions_table": "positionen",
                        "positions_fk": "Auftragsnummer",
                        "positions_columns": {
                            "pos": "Position",
                            "article_id": "Artikelnummer",
                            "description": "Bezeichnung",
                            "quantity": "Menge",
                            "unit_price": "Einzelpreis",
                        },
                    },
                    "articles": {
                        "table": "artikel",
                        "id_column": "Artikelnummer",
                        "columns": {
                            "article_id": "Artikelnummer",
                            "description": "Bezeichnung",
                            "category": "Warengruppe",
                            "stock_quantity": "Bestand",
                            "unit": "Einheit",
                            "warehouse": "Lagerort",
                            "unit_price": "VK_Preis",
                            "min_stock": "Mindestbestand",
                            "supplier": "Hauptlieferant",
                            "lead_time_days": "Lieferzeit_Tage",
                        },
                    },
                    "customers": {
                        "table": "kunden",
                        "id_column": "Kundennummer",
                        "columns": {
                            "customer_id": "Kundennummer",
                            "name": "Firmenname",
                            "customer_since": "Kunde_seit",
                            "payment_terms": "Zahlungsbedingung",
                            "credit_limit": "Kreditlimit",
                        },
                    },
                    "production_orders": {
                        "table": "fertigungsauftraege",
                        "id_column": "FA_Nummer",
                        "columns": {
                            "production_order_id": "FA_Nummer",
                            "order_reference": "Bezugsauftrag",
                            "article_id": "Artikelnummer",
                            "article_description": "Artikelbezeichnung",
                            "quantity": "Menge",
                            "status": "Status",
                            "start_date": "Startdatum",
                            "planned_end_date": "Geplantes_Ende",
                        },
                    },
                },
            },
        },
        "ollama": {"url": "http://localhost:11434", "llm_model": "test", "embedding_model": "test"},
        "datasource": {"path": "/tmp"},
        "vectordb": {"path": "/tmp"},
    }


class TestSQLQueryOrder:
    """Auftragsabfragen gegen SQL-Mock."""

    @pytest.fixture
    def backend(self):
        config = _make_config()
        return ERPBackend(config)

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_query_order(self, MockConnCls, backend):
        MockConnCls.return_value = _make_connector_mock(
            _ORDER_COLUMNS,
            _ORDER_ROW,
            sub_columns=_POSITIONS_COLUMNS,
            sub_rows=_POSITIONS_ROWS,
        )

        result = await backend.query_order("A-1001")
        assert result is not None
        assert result["order_id"] == "A-1001"
        assert result["customer_name"] == "Musterfirma GmbH"
        assert result["status"] == "In Bearbeitung"
        assert result["total_amount"] == 12500.00

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_query_order_with_positions(self, MockConnCls, backend):
        MockConnCls.return_value = _make_connector_mock(
            _ORDER_COLUMNS,
            _ORDER_ROW,
            sub_columns=_POSITIONS_COLUMNS,
            sub_rows=_POSITIONS_ROWS,
        )

        result = await backend.query_order("A-1001")
        assert "positions" in result
        assert len(result["positions"]) == 2
        assert result["positions"][0]["article_id"] == "ART-10001"

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_query_order_not_found(self, MockConnCls, backend):
        MockConnCls.return_value = _make_empty_connector()

        result = await backend.query_order("NONEXISTENT")
        assert result is None


class TestSQLQueryStock:
    """Bestandsabfragen gegen SQL-Mock."""

    @pytest.fixture
    def backend(self):
        config = _make_config()
        return ERPBackend(config)

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_query_stock(self, MockConnCls, backend):
        MockConnCls.return_value = _make_connector_mock(_ARTICLE_COLUMNS, _ARTICLE_ROW)

        result = await backend.query_stock("ART-10001")
        assert result is not None
        assert result["article_id"] == "ART-10001"
        assert result["description"] == "Widget A"
        assert result["stock_quantity"] == 500
        assert result["min_stock"] == 100


class TestSQLQueryCustomer:
    """Kundenabfragen gegen SQL-Mock."""

    @pytest.fixture
    def backend(self):
        config = _make_config()
        return ERPBackend(config)

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_query_customer(self, MockConnCls, backend):
        MockConnCls.return_value = _make_connector_mock(_CUSTOMER_COLUMNS, _CUSTOMER_ROW)

        result = await backend.query_customer("K-1001")
        assert result is not None
        assert result["customer_id"] == "K-1001"
        assert result["name"] == "Musterfirma GmbH"
        assert result["credit_limit"] == 50000.00

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_query_customer_not_found(self, MockConnCls, backend):
        MockConnCls.return_value = _make_empty_connector()

        result = await backend.query_customer("K-9999")
        assert result is None


class TestMappingMissing:
    """Fehlendes Mapping -> None (kein Crash)."""

    @pytest.mark.asyncio
    async def test_mapping_missing(self):
        config = _make_config()
        # Mapping fuer production_orders entfernen
        del config["tools"]["erp"]["mapping"]["production_orders"]
        backend = ERPBackend(config)
        result = await backend.query_production("FA-2001")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_mapping(self):
        config = _make_config()
        config["tools"]["erp"]["mapping"] = {}
        backend = ERPBackend(config)
        result = await backend.query_order("A-1001")
        assert result is None


class TestERPBackendBulk:
    """Bulk-API fuer Graph-Sync."""

    @pytest.mark.asyncio
    async def test_mock_list_all(self, tmp_path):
        mock_data = {
            "orders": [{"order_id": "A-1"}],
            "articles": [{"article_id": "ART-1"}],
            "customers": [{"customer_id": "K-1"}],
            "production_orders": [{"production_order_id": "FA-1"}],
        }
        mock_path = tmp_path / "mock.json"
        mock_path.write_text(__import__("json").dumps(mock_data), encoding="utf-8")
        backend = ERPBackend({"tools": {"erp": {"backend": "mock", "mock_data": str(mock_path)}}})

        assert await backend.list_all_customers() == mock_data["customers"]
        assert await backend.list_all_articles() == mock_data["articles"]
        assert await backend.list_all_orders() == mock_data["orders"]
        assert await backend.list_all_production_orders() == mock_data["production_orders"]

    @pytest.mark.asyncio
    async def test_sql_bulk_missing_mapping_fails_closed(self):
        config = _make_config()
        del config["tools"]["erp"]["mapping"]["orders"]
        backend = ERPBackend(config)

        with pytest.raises(ValueError, match="Mapping"):
            await backend.list_all_orders()
