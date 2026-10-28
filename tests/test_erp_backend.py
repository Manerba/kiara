"""Tests fuer das ERPBackend (kiara.tools.erp_backend)."""

import pytest

from kiara.tools.erp_backend import ERPBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Minimale Config fuer Mock-Backend (Standard mock_data.json)."""
    return {
        "tools": {
            "enabled": True,
            "erp": {
                "backend": "mock",
                "mock_data": "",  # Leerer Pfad → nutzt eingebaute Daten
            },
        },
    }


@pytest.fixture
def backend(mock_config):
    """ERPBackend-Instanz mit Mock-Daten."""
    return ERPBackend(mock_config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestERPBackend:
    def test_mock_backend_loads_data(self, backend):
        """Mock-Daten werden korrekt geladen."""
        assert backend._mock_data is not None
        assert "orders" in backend._mock_data
        assert "articles" in backend._mock_data
        assert "customers" in backend._mock_data
        assert "production_orders" in backend._mock_data
        # Mindestens einige Datensaetze sollten vorhanden sein
        assert len(backend._mock_data["orders"]) > 0
        assert len(backend._mock_data["articles"]) > 0

    @pytest.mark.asyncio
    async def test_query_order_existing(self, backend):
        """Bestehenden Auftrag A-1001 abfragen."""
        order = await backend.query_order("A-1001")
        assert order is not None
        assert order["order_id"] == "A-1001"
        assert "customer_name" in order
        assert "status" in order
        assert "positions" in order

    @pytest.mark.asyncio
    async def test_query_order_not_found(self, backend):
        """Unbekannter Auftrag liefert None."""
        order = await backend.query_order("NICHTEXISTENT-999")
        assert order is None

    @pytest.mark.asyncio
    async def test_query_stock_existing(self, backend):
        """Artikel-Bestand ART-10001 abfragen."""
        article = await backend.query_stock("ART-10001")
        assert article is not None
        assert article["article_id"] == "ART-10001"
        assert "stock_quantity" in article
        assert "unit_price" in article
        assert "warehouse" in article

    @pytest.mark.asyncio
    async def test_query_customer_existing(self, backend):
        """Kundendaten K-1001 abfragen."""
        customer = await backend.query_customer("K-1001")
        assert customer is not None
        assert customer["customer_id"] == "K-1001"
        assert "name" in customer
        assert "address" in customer
        assert "contact" in customer

    @pytest.mark.asyncio
    async def test_query_production_existing(self, backend):
        """Fertigungsauftrag FA-2001 abfragen."""
        production = await backend.query_production("FA-2001")
        assert production is not None
        assert production["production_order_id"] == "FA-2001"
        assert "operations" in production
        assert len(production["operations"]) > 0

    def test_fallback_on_unknown_backend(self):
        """Unbekannter Backend-Typ faellt auf mock zurueck."""
        config = {
            "tools": {
                "erp": {
                    "backend": "unbekannt",
                    "mock_data": "",
                },
            },
        }
        be = ERPBackend(config)
        assert be._backend_type == "mock"
        assert be._mock_data is not None

    def test_erp_bulk_timeout_default_is_explicit(self):
        from kiara.config import DEFAULT_CONFIG, DEFAULT_TIMEOUTS, get_timeout

        assert DEFAULT_CONFIG["timeouts"]["erp_bulk"] == DEFAULT_TIMEOUTS["erp_bulk"]
        assert get_timeout({}, "erp_bulk") == DEFAULT_TIMEOUTS["erp_bulk"]

    def test_erp_query_timeout_default_is_explicit(self):
        from kiara.config import DEFAULT_CONFIG, DEFAULT_TIMEOUTS, get_timeout

        assert DEFAULT_CONFIG["timeouts"]["erp_query"] == DEFAULT_TIMEOUTS["erp_query"]
        assert get_timeout({}, "erp_query") == DEFAULT_TIMEOUTS["erp_query"]

    def test_sql_query_entity_uses_configured_statement_timeout(self, monkeypatch):
        calls = {}

        def fake_sql_query_entity(dsn, driver, mapping, entity_id, *, statement_timeout_seconds):
            calls["args"] = (dsn, driver, mapping, entity_id, statement_timeout_seconds)
            return {"order_id": entity_id}

        monkeypatch.setattr(
            "kiara.tools.erp_connectors.sql_utils.sql_query_entity",
            fake_sql_query_entity,
        )
        config = {
            "timeouts": {"erp_query": 4.0},
            "tools": {
                "erp": {
                    "backend": "sql",
                    "sql_dsn": "postgresql://host/db",
                    "sql_driver": "postgresql",
                },
            },
        }
        backend = ERPBackend(config)

        result = backend._sql_query_entity({"table": "orders"}, "A-1")

        assert result == {"order_id": "A-1"}
        assert calls["args"] == (
            "postgresql://host/db",
            "postgresql",
            {"table": "orders"},
            "A-1",
            4.0,
        )

    @pytest.mark.asyncio
    async def test_registry_bulk_value_error_is_configuration_error(self, mock_config):
        """Connector-Bulk-ValueError wird nicht als Upstream-Fehler maskiert."""

        class Connector:
            async def list_all_orders(self):
                raise ValueError("GenericSQL-Bulk-Mapping fuer 'orders' fehlt.")

        class Registry:
            def get_connector_for_tool(self, tool_name):
                return Connector() if tool_name == "erp_order_status" else None

        backend = ERPBackend(mock_config, Registry())

        with pytest.raises(ValueError, match="Mapping"):
            await backend.list_all_orders()
