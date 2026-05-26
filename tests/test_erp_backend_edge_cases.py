"""Tests fuer ERP-Backend Edge-Cases (Phase 6 Erweiterung)."""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from kiara.tools.erp_backend import ERPBackend

# -- Vordefinierte Cursor-Daten (simulieren SQLConnector-Verbindung) --
# sql_utils.sql_query_entity() nutzt connector.connection() als Context-Manager
# und ruft cursor.execute() + cursor.description + fetchmany()/fetchall() auf.

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
]

_PRODUCTION_COLUMNS = [
    "production_order_id",
    "order_reference",
    "article_id",
    "article_description",
    "quantity",
    "status",
    "start_date",
    "planned_end_date",
]
_PRODUCTION_ROW = (
    "FA-2001",
    "A-1001",
    "ART-10001",
    "Widget A",
    100,
    "In Fertigung",
    "2024-02-01",
    "2024-02-20",
)

_OPERATIONS_COLUMNS = ["step", "description", "workplace", "duration_minutes"]
_OPERATIONS_ROWS = [
    (1, "Zuschnitt", "AP-100", 30),
    (2, "Montage", "AP-200", 60),
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
      2. cur.description -> Haupt-Spalten
      3. cur.fetchmany(1) -> [main_row]
      4. (optional) cur.execute(sub_query, (entity_id,))
      5. (optional) cur.description -> Sub-Spalten
      6. (optional) cur.fetchall() -> sub_rows
    """
    cursor = MagicMock()

    if sub_columns is not None:
        desc_values = [
            _make_cursor_description(main_columns),
            _make_cursor_description(sub_columns),
        ]
        cursor.description = desc_values[0]
        call_count = [0]

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
                        "operations_table": "arbeitsgaenge",
                        "operations_fk": "FA_Nummer",
                        "operations_columns": {
                            "step": "Schritt",
                            "description": "Beschreibung",
                            "workplace": "Arbeitsplatz",
                            "duration_minutes": "Dauer_Minuten",
                        },
                    },
                },
            },
        },
        "ollama": {"url": "http://localhost:11434", "llm_model": "test", "embedding_model": "test"},
        "datasource": {"path": "/tmp"},
        "vectordb": {"path": "/tmp"},
    }


class TestSQLQueryProductionWithOperations:
    """Fertigungsauftrag mit Arbeitsgaengen (Operations-Subtabelle)."""

    @pytest.fixture
    def backend(self):
        config = _make_config()
        return ERPBackend(config)

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_query_production_with_operations(self, MockConnCls, backend):
        """Fertigungsauftrag FA-2001 -> result['operations'] enthaelt Arbeitsgaenge."""
        MockConnCls.return_value = _make_connector_mock(
            _PRODUCTION_COLUMNS,
            _PRODUCTION_ROW,
            sub_columns=_OPERATIONS_COLUMNS,
            sub_rows=_OPERATIONS_ROWS,
        )

        result = await backend.query_production("FA-2001")
        assert result is not None
        assert result["production_order_id"] == "FA-2001"
        assert result["status"] == "In Fertigung"
        assert "operations" in result
        assert len(result["operations"]) == 2
        # Erster Arbeitsgang: Zuschnitt
        ops = sorted(result["operations"], key=lambda o: o["step"])
        assert ops[0]["description"] == "Zuschnitt"
        assert ops[0]["workplace"] == "AP-100"
        assert ops[0]["duration_minutes"] == 30
        # Zweiter Arbeitsgang: Montage
        assert ops[1]["description"] == "Montage"


class TestUnknownBackendFallbackMock:
    """Unbekannter Backend-Typ -> Fallback auf Mock."""

    def test_unknown_backend_fallback_mock(self, tmp_path):
        """backend='unknown' -> Fallback auf Mock-Backend."""
        config = {
            "tools": {"erp": {"backend": "unknown"}},
            "ollama": {"url": "http://localhost:11434"},
            "datasource": {"path": "/tmp"},
            "vectordb": {"path": "/tmp"},
        }
        backend = ERPBackend(config)
        # Backend-Typ wird intern auf 'mock' geaendert
        assert backend._backend_type == "mock"
        assert backend._mock_data is not None


class TestRESTBackendRaisesNotImplemented:
    """REST-Backend -> NotImplementedError."""

    @pytest.mark.asyncio
    async def test_rest_backend_raises_not_implemented(self, tmp_path):
        """backend='rest' -> query_order() wirft NotImplementedError."""
        config = {
            "tools": {"erp": {"backend": "rest"}},
            "ollama": {"url": "http://localhost:11434"},
            "datasource": {"path": "/tmp"},
            "vectordb": {"path": "/tmp"},
        }
        backend = ERPBackend(config)
        with pytest.raises(NotImplementedError, match="REST-Backend"):
            await backend.query_order("A-1001")


class TestMockBackendOrderNotFound:
    """Mock-Backend: Nicht existierender Auftrag -> None."""

    @pytest.mark.asyncio
    async def test_mock_backend_order_not_found(self, tmp_path):
        """Mock-Backend mit query_order('NONEXISTENT') -> None."""
        # Erstelle Mock-Datei mit leerem Auftrags-Array
        mock_data = {"orders": [], "articles": [], "customers": [], "production_orders": []}
        mock_path = str(tmp_path / "mock_data.json")
        with open(mock_path, "w") as f:
            json.dump(mock_data, f)

        config = {
            "tools": {"erp": {"backend": "mock", "mock_data": mock_path}},
            "ollama": {"url": "http://localhost:11434"},
            "datasource": {"path": "/tmp"},
            "vectordb": {"path": "/tmp"},
        }
        backend = ERPBackend(config)
        result = await backend.query_order("NONEXISTENT")
        assert result is None


class TestMockBackendLoadsFromCustomPath:
    """Mock-Backend laedt aus benutzerdefiniertem Pfad."""

    @pytest.mark.asyncio
    async def test_mock_backend_loads_from_custom_path(self, tmp_path):
        """erp.mock_data='/pfad/zur/datei.json' -> laedt daraus."""
        mock_data = {
            "orders": [
                {"order_id": "CUSTOM-001", "customer_name": "Custom GmbH", "status": "Offen"}
            ],
            "articles": [],
            "customers": [],
            "production_orders": [],
        }
        mock_path = str(tmp_path / "custom_mock.json")
        with open(mock_path, "w") as f:
            json.dump(mock_data, f)

        config = {
            "tools": {"erp": {"backend": "mock", "mock_data": mock_path}},
            "ollama": {"url": "http://localhost:11434"},
            "datasource": {"path": "/tmp"},
            "vectordb": {"path": "/tmp"},
        }
        backend = ERPBackend(config)
        result = await backend.query_order("CUSTOM-001")
        assert result is not None
        assert result["order_id"] == "CUSTOM-001"
        assert result["customer_name"] == "Custom GmbH"


class TestMockBackendMissingFile:
    """Mock-Datei existiert nicht -> leere Listen, kein Crash."""

    @pytest.mark.asyncio
    async def test_mock_backend_missing_file(self, tmp_path):
        """Nicht existierende Mock-Datei -> leere Daten, kein Absturz."""
        config = {
            "tools": {
                "erp": {"backend": "mock", "mock_data": str(tmp_path / "nicht_vorhanden.json")}
            },
            "ollama": {"url": "http://localhost:11434"},
            "datasource": {"path": "/tmp"},
            "vectordb": {"path": "/tmp"},
        }
        backend = ERPBackend(config)
        # Backend hat leere Listen als Fallback
        result = await backend.query_order("A-1001")
        assert result is None
        result = await backend.query_stock("ART-10001")
        assert result is None


class TestSQLBackendEntityWithoutSubtable:
    """Entitaet ohne Sub-Tabelle -> nur Hauptdaten."""

    @pytest.fixture
    def backend(self):
        config = _make_config()
        return ERPBackend(config)

    @pytest.mark.asyncio
    @patch("kiara.connectors.sql_connector.SQLConnector")
    async def test_sql_backend_entity_without_subtable(self, MockConnCls, backend):
        """Artikel-Abfrage (keine positions/operations_table) -> nur Hauptdaten."""
        MockConnCls.return_value = _make_connector_mock(_ARTICLE_COLUMNS, _ARTICLE_ROW)

        result = await backend.query_stock("ART-10001")
        assert result is not None
        assert result["article_id"] == "ART-10001"
        assert result["description"] == "Widget A"
        assert result["stock_quantity"] == 500
        # Keine Sub-Listen vorhanden
        assert "positions" not in result
        assert "operations" not in result
