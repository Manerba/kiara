"""Tests fuer die konkreten ERP-Tools (kiara.tools.erp_tools)."""

import pytest

from kiara.tools.erp_backend import ERPBackend
from kiara.tools.erp_tools import (
    ERPCustomerInfoTool,
    ERPOrderStatusTool,
    ERPProductionStatusTool,
    ERPStockLevelTool,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    return {
        "tools": {
            "enabled": True,
            "erp": {
                "backend": "mock",
                "mock_data": "",
            },
        },
    }


@pytest.fixture
def backend(mock_config):
    return ERPBackend(mock_config)


@pytest.fixture
def order_tool(backend):
    return ERPOrderStatusTool(backend)


@pytest.fixture
def stock_tool(backend):
    return ERPStockLevelTool(backend)


@pytest.fixture
def customer_tool(backend):
    return ERPCustomerInfoTool(backend)


@pytest.fixture
def production_tool(backend):
    return ERPProductionStatusTool(backend)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestERPOrderStatusTool:
    @pytest.mark.asyncio
    async def test_order_status_tool_success(self, order_tool):
        """Erfolgreicher Auftragsabruf liefert display_text mit Auftragsdaten."""
        result = await order_tool.execute({"order_id": "A-1001"})
        assert result.success is True
        assert result.data is not None
        assert result.data["order_id"] == "A-1001"
        assert "Auftrag: A-1001" in result.display_text
        assert "Status:" in result.display_text
        assert result.error is None

    @pytest.mark.asyncio
    async def test_order_status_tool_not_found(self, order_tool):
        """Auftrag nicht gefunden liefert Fehler."""
        result = await order_tool.execute({"order_id": "FAKE-999"})
        assert result.success is False
        assert result.data is None
        assert "nicht gefunden" in result.error

    def test_order_status_invalid_amounts_do_not_crash(self, order_tool):
        """Fehlerhafte ERP-Zahlenwerte werden als fehlende Werte formatiert."""
        text = order_tool._format_result(
            {
                "order_id": "A-X",
                "positions": [
                    {"pos": 1, "description": "Bad", "quantity": 1, "unit_price": "abc"},
                ],
                "total_amount": "not-a-number",
            },
            "A-X",
        )

        assert "1 x 0.00 EUR" in text
        assert "Gesamtbetrag: 0.00 EUR" in text


class TestERPStockLevelTool:
    @pytest.mark.asyncio
    async def test_stock_level_tool_success(self, stock_tool):
        """Lagerbestand abrufen funktioniert korrekt."""
        result = await stock_tool.execute({"article_id": "ART-10001"})
        assert result.success is True
        assert result.data is not None
        assert result.data["article_id"] == "ART-10001"
        assert "Bestand:" in result.display_text
        assert "Lagerort:" in result.display_text

    @pytest.mark.asyncio
    async def test_stock_level_tool_not_found(self, stock_tool):
        """Unbekannter Artikel liefert Fehler."""
        result = await stock_tool.execute({"article_id": "FAKE-0"})
        assert result.success is False
        assert "nicht gefunden" in result.error

    def test_stock_level_invalid_amounts_do_not_crash(self, stock_tool):
        """Fehlerhafte Bestandswerte aus dem ERP-Backend brechen die Ausgabe nicht."""
        text = stock_tool._format_result(
            {
                "article_id": "ART-X",
                "stock_quantity": "n/a",
                "min_stock": "invalid",
                "unit_price": "bad-price",
            },
            "ART-X",
        )

        assert "Bestand: 0.0 Stueck" in text
        assert "Stueckpreis: 0.00 EUR" in text


class TestERPCustomerInfoTool:
    @pytest.mark.asyncio
    async def test_customer_info_tool_success(self, customer_tool):
        """Kundendaten abrufen funktioniert korrekt."""
        result = await customer_tool.execute({"customer_id": "K-1001"})
        assert result.success is True
        assert result.data is not None
        assert result.data["customer_id"] == "K-1001"
        assert "Kunde:" in result.display_text
        assert "Adresse:" in result.display_text


class TestERPProductionStatusTool:
    @pytest.mark.asyncio
    async def test_production_status_tool_success(self, production_tool):
        """Fertigungsstatus abrufen funktioniert korrekt."""
        result = await production_tool.execute({"order_id": "FA-2001"})
        assert result.success is True
        assert result.data is not None
        assert result.data["production_order_id"] == "FA-2001"
        assert "Fertigungsauftrag:" in result.display_text
        assert "Arbeitsgaenge:" in result.display_text


class TestToolDefinitions:
    def test_tool_definition_correct(self, order_tool, stock_tool, customer_tool, production_tool):
        """Alle Tool-Definitionen haben die richtigen Felder."""
        for tool, expected_name in [
            (order_tool, "erp_order_status"),
            (stock_tool, "erp_stock_level"),
            (customer_tool, "erp_customer_info"),
            (production_tool, "erp_production_status"),
        ]:
            defn = tool.definition()
            assert defn.name == expected_name
            assert defn.description != ""
            assert "type" in defn.parameters
            assert defn.parameters["type"] == "object"
            assert "properties" in defn.parameters
            assert "required" in defn.parameters
            assert len(defn.required_permissions) > 0
