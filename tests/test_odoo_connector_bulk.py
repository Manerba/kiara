"""Regressionstests fuer Odoo-Bulk-Pfade."""

import asyncio
from unittest.mock import patch

from kiara.tools.erp_connectors.base import ConnectionInfo
from kiara.tools.erp_connectors.odoo import OdooConnector


def _make_connector() -> OdooConnector:
    return OdooConnector(
        ConnectionInfo(
            id="conn-1",
            name="Odoo",
            connector_type="odoo",
            config={
                "url": "http://odoo.example.test",
                "database": "kiara",
                "username": "admin",
                "password": "secret",
            },
        )
    )


def test_bulk_product_code_map_falls_back_to_product_id():
    connector = _make_connector()

    with patch.object(
        connector,
        "_search_read",
        return_value=[{"id": 7, "default_code": ""}],
    ):
        result = connector._bulk_product_code_map([7])

    assert result == {7: "7"}


def test_list_all_orders_uses_product_id_fallback_for_missing_default_code():
    connector = _make_connector()
    orders = [
        {
            "id": 1,
            "name": "SO-1",
            "partner_id": [10, "Customer"],
            "state": "sale",
            "date_order": "2026-04-01",
            "commitment_date": False,
            "amount_total": 12.5,
            "order_line": [100],
        }
    ]

    def search_read(model, _domain, _fields, limit=0, offset=0, order=""):
        if model == "sale.order.line":
            return [
                {
                    "id": 100,
                    "sequence": 1,
                    "product_id": [7, "Widget"],
                    "name": "Widget",
                    "product_uom_qty": 2,
                    "price_unit": 6.25,
                }
            ]
        if model == "product.product":
            return [{"id": 7, "default_code": ""}]
        if model == "mrp.production":
            return []
        raise AssertionError(f"unexpected model {model}")

    with (
        patch.object(connector, "_paged_search_read", return_value=orders),
        patch.object(connector, "_bulk_partner_ref_map", return_value={10: "K-10"}),
        patch.object(connector, "_search_read", side_effect=search_read),
    ):
        result = asyncio.run(connector.list_all_orders())

    assert result[0]["positions"][0]["article_id"] == "7"


def test_list_all_production_orders_uses_product_id_fallback_for_missing_default_code():
    connector = _make_connector()
    productions = [
        {
            "id": 1,
            "name": "MO-1",
            "origin": "SO-1",
            "product_id": [7, "Widget"],
            "product_qty": 3,
            "state": "confirmed",
            "date_start": "2026-04-02",
            "date_deadline": "2026-04-03",
            "workorder_ids": [],
        }
    ]

    with (
        patch.object(connector, "_paged_search_read", return_value=productions),
        patch.object(
            connector,
            "_search_read",
            return_value=[{"id": 7, "default_code": ""}],
        ),
    ):
        result = asyncio.run(connector.list_all_production_orders())

    assert result[0]["article_id"] == "7"


def test_list_all_orders_marks_sale_with_active_production_as_in_fertigung():
    connector = _make_connector()
    orders = [
        {
            "id": 1,
            "name": "SO-1",
            "partner_id": [10, "Customer"],
            "state": "sale",
            "date_order": "2026-04-01",
            "commitment_date": False,
            "amount_total": 12.5,
            "order_line": [],
        },
        {
            "id": 2,
            "name": "SO-2",
            "partner_id": [10, "Customer"],
            "state": "sale",
            "date_order": "2026-04-01",
            "commitment_date": False,
            "amount_total": 5.0,
            "order_line": [],
        },
    ]

    def search_read(model, _domain, _fields, limit=0, offset=0, order=""):
        if model == "mrp.production":
            return [{"origin": "SO-1"}]
        raise AssertionError(f"unexpected model {model}")

    with (
        patch.object(connector, "_paged_search_read", return_value=orders),
        patch.object(connector, "_bulk_partner_ref_map", return_value={10: "K-10"}),
        patch.object(connector, "_search_read", side_effect=search_read),
    ):
        result = asyncio.run(connector.list_all_orders())

    by_id = {o["order_id"]: o for o in result}
    assert by_id["SO-1"]["status"] == "in_fertigung"
    assert by_id["SO-2"]["status"] == "offen"
