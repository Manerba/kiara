"""Tests fuer das sicher begrenzte ERP-Auftragsgrunddaten-Tool."""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiara.tools.erp_backend import ERPBackend, ERPUpstreamError
from kiara.tools.erp_connectors.base import ConnectionInfo
from kiara.tools.erp_connectors.odoo import OdooConnector
from kiara.tools.erp_order_query import apply_order_query, build_order_query
from kiara.tools.erp_tools import ERPOrdersTool
from kiara.tools.registry import ToolRegistry


def _query(**overrides):
    params = {
        "filters": [
            {"field": "delivery_status", "operator": "eq", "value": "full"},
        ],
        "fields": ["order_id", "order_date", "actual_delivery_date"],
        "sort_by": "actual_delivery_date",
        "sort_direction": "desc",
        "limit": 10,
    }
    params.update(overrides)
    return build_order_query(params)


def _odoo_connector() -> OdooConnector:
    return OdooConnector(
        ConnectionInfo(
            id="odoo-test",
            name="Odoo Test",
            connector_type="odoo",
            config={
                "url": "http://odoo.example.test",
                "database": "test",
                "username": "user",
                "password": "secret",
            },
        )
    )


class TestOrderQueryValidation:
    def test_accepts_structured_bounded_query(self):
        query = _query(limit=5, offset=2)

        assert query.limit == 5
        assert query.offset == 2
        assert query.filters[0].field == "delivery_status"

    @pytest.mark.parametrize(
        "params, match",
        [
            ({"fields": ["raw_odoo_model"]}, "Nicht erlaubte Felder"),
            (
                {"filters": [{"field": "__sql__", "operator": "eq", "value": "x"}]},
                "Filterfeld",
            ),
            (
                {"filters": [{"field": "total_amount", "operator": "contains", "value": 10}]},
                "Operator",
            ),
            ({"limit": 51}, "limit muss zwischen"),
            ({"offset": 1001}, "offset muss zwischen"),
            ({"raw_domain": [["id", ">", 0]]}, "Nicht erlaubte Parameter"),
            (
                {"filters": [{"field": "total_amount", "operator": "eq", "value": math.inf}]},
                "endlich",
            ),
        ],
    )
    def test_rejects_unbounded_or_unsafe_queries(self, params, match):
        with pytest.raises(ValueError, match=match):
            build_order_query(params)

    def test_rejects_extra_nested_filter_keys(self):
        with pytest.raises(ValueError, match="genau field"):
            build_order_query(
                {
                    "filters": [
                        {
                            "field": "order_id",
                            "operator": "eq",
                            "value": "S1",
                            "domain": [["id", ">", 0]],
                        }
                    ]
                }
            )


class TestOrderQueryApplication:
    def test_filters_sorts_projects_and_keeps_dates_distinct(self):
        rows = [
            {
                "order_id": "S1",
                "delivery_status": "full",
                "order_date": "2026-07-01 10:00:00",
                "delivery_date": "2026-07-10 10:00:00",
                "actual_delivery_date": "2026-07-05 10:00:00",
            },
            {
                "order_id": "S2",
                "delivery_status": "partial",
                "order_date": "2026-07-02 10:00:00",
                "actual_delivery_date": "2026-07-08 10:00:00",
            },
            {
                "order_id": "S3",
                "delivery_status": "full",
                "order_date": "2026-07-03 10:00:00",
                "promised_delivery_date": "2026-07-20 10:00:00",
                "actual_delivery_date": "2026-07-09 10:00:00",
            },
        ]

        result = apply_order_query(rows, _query())

        assert [row["order_id"] for row in result["rows"]] == ["S3", "S1"]
        assert result["rows"][1]["actual_delivery_date"] == "2026-07-05 10:00:00"
        assert set(result["rows"][0]) == {
            "order_id",
            "order_date",
            "actual_delivery_date",
        }

    def test_limit_offset_and_has_more(self):
        rows = [{"order_id": f"S{i}", "order_date": f"2026-07-{i:02d}"} for i in range(1, 6)]
        query = build_order_query(
            {
                "fields": ["order_id"],
                "sort_by": "order_date",
                "sort_direction": "asc",
                "limit": 2,
                "offset": 1,
            }
        )

        result = apply_order_query(rows, query)

        assert result["rows"] == [{"order_id": "S2"}, {"order_id": "S3"}]
        assert result["has_more"] is True


class TestERPOrdersTool:
    @pytest.mark.asyncio
    async def test_executes_validated_query(self):
        backend = MagicMock()
        backend.query_orders = AsyncMock(
            return_value={
                "rows": [{"order_id": "S1"}],
                "returned": 1,
                "has_more": False,
                "limit": 10,
                "offset": 0,
                "fields": ["order_id"],
                "scan_truncated": False,
            }
        )
        tool = ERPOrdersTool(backend)

        result = await tool.execute({"fields": ["order_id"]}, user_id=3)

        assert result.success is True
        assert result.data["rows"] == [{"order_id": "S1"}]
        backend.query_orders.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejects_unknown_field_before_backend_call(self):
        backend = MagicMock()
        backend.query_orders = AsyncMock()
        result = await ERPOrdersTool(backend).execute({"fields": ["password"]})

        assert result.success is False
        backend.query_orders.assert_not_awaited()

    def test_definition_is_read_only_and_bounded(self):
        definition = ERPOrdersTool(MagicMock()).definition()

        assert definition.name == "erp_orders"
        assert definition.requires_confirmation is False
        assert definition.parameters["properties"]["limit"]["maximum"] == 50
        assert "ERP-Leser" in definition.required_permissions

    def test_permission_matrix_exposes_tool_only_to_configured_group(self):
        registry = ToolRegistry(tool_permissions={"GG_100": ["erp_orders"]})
        registry.register(ERPOrdersTool(MagicMock()))

        assert [tool.name for tool in registry.list_tools(["GG_100"])] == ["erp_orders"]
        assert registry.list_tools(["GG_110"]) == []

    def test_ollama_parser_accepts_structured_filter_lists(self):
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        parsed = OllamaToolAdapter._parse_tool_call(
            'TOOL_CALL: erp_orders(filters=[{"field":"delivery_status",'
            '"operator":"eq","value":"full"}], fields=["order_id",'
            '"actual_delivery_date"], sort_by="actual_delivery_date", limit=10)'
        )

        assert parsed == (
            "erp_orders",
            {
                "filters": [{"field": "delivery_status", "operator": "eq", "value": "full"}],
                "fields": ["order_id", "actual_delivery_date"],
                "sort_by": "actual_delivery_date",
                "limit": 10,
            },
        )


class TestERPBackendOrderQuery:
    @pytest.mark.asyncio
    async def test_uses_order_status_connection(self):
        connector = MagicMock()
        connector.query_orders = AsyncMock(
            return_value={
                "rows": [],
                "returned": 0,
                "has_more": False,
                "limit": 10,
                "offset": 0,
                "fields": ["order_id", "order_date", "actual_delivery_date"],
                "scan_truncated": False,
            }
        )
        registry = MagicMock()
        registry.get_connector_for_tool.return_value = connector
        backend = ERPBackend({"tools": {"erp": {"backend": "mock"}}}, registry)

        result = await backend.query_orders(_query())

        assert result["rows"] == []
        registry.get_connector_for_tool.assert_called_once_with("erp_order_status")

    @pytest.mark.asyncio
    async def test_rejects_malformed_connector_result(self):
        connector = MagicMock()
        connector.query_orders = AsyncMock(return_value={"rows": "not-a-list"})
        registry = MagicMock()
        registry.get_connector_for_tool.return_value = connector
        backend = ERPBackend({"tools": {"erp": {"backend": "mock"}}}, registry)

        with pytest.raises(ERPUpstreamError, match="ungueltiges Ergebnis"):
            await backend.query_orders(_query())

    @pytest.mark.asyncio
    async def test_rejects_unprojected_connector_fields(self):
        connector = MagicMock()
        connector.query_orders = AsyncMock(
            return_value={
                "rows": [
                    {
                        "order_id": "S1",
                        "order_date": "2026-07-01",
                        "actual_delivery_date": "2026-07-02",
                        "password": "leak",
                    }
                ],
                "returned": 1,
                "has_more": False,
                "limit": 10,
                "offset": 0,
                "fields": ["order_id", "order_date", "actual_delivery_date"],
                "scan_truncated": False,
            }
        )
        registry = MagicMock()
        registry.get_connector_for_tool.return_value = connector
        backend = ERPBackend({"tools": {"erp": {"backend": "mock"}}}, registry)

        with pytest.raises(ERPUpstreamError, match="Feldprojektion"):
            await backend.query_orders(_query())

    @pytest.mark.asyncio
    async def test_legacy_sql_fails_closed_instead_of_bulk_scan(self):
        backend = ERPBackend({"tools": {"erp": {"backend": "sql"}}})

        with pytest.raises(NotImplementedError, match="hart begrenzte"):
            await backend.query_orders(_query())


class TestOdooOrderGroundData:
    def test_delivery_aggregates_keep_first_and_last_transfer(self):
        connector = _odoo_connector()

        def execute(_model, _method, args, _kwargs):
            aggregate = args[1][1]
            date = "2026-07-01 10:00:00" if aggregate.endswith(":min") else "2026-07-03 12:00:00"
            return [{"sale_id": [7, "S7"], "date_done": date}]

        with patch.object(connector, "_execute", side_effect=execute):
            dates = connector._delivery_dates_by_sale_ids([7])

        assert dates[7] == {
            "first": "2026-07-01 10:00:00",
            "actual": "2026-07-03 12:00:00",
        }

    def test_full_delivery_overrides_sale_state(self):
        connector = _odoo_connector()
        row = connector._build_order_record(
            {
                "id": 7,
                "name": "S7",
                "state": "sale",
                "delivery_status": "full",
                "commitment_date": "2026-07-10 00:00:00",
                "effective_date": "2026-07-01 10:00:00",
                "order_line": [],
            },
            delivery_dates={
                7: {
                    "first": "2026-07-01 10:00:00",
                    "actual": "2026-07-03 12:00:00",
                }
            },
        )

        assert row["status"] == "geliefert"
        assert row["promised_delivery_date"] == "2026-07-10 00:00:00"
        assert row["actual_delivery_date"] == "2026-07-03 12:00:00"

    def test_query_orders_applies_domain_and_actual_delivery_sort(self):
        connector = _odoo_connector()
        query = _query(limit=1)
        orders = [
            {
                "id": 7,
                "name": "S7",
                "partner_id": [4, "Kunde"],
                "state": "sale",
                "delivery_status": "full",
                "date_order": "2026-07-01 09:00:00",
                "commitment_date": False,
                "effective_date": "2026-07-01 10:00:00",
                "amount_total": 100,
                "currency_id": [1, "EUR"],
                "order_line": [],
                "picking_ids": [70, 71],
            }
        ]

        with (
            patch.object(
                connector,
                "_delivery_sorted_candidates",
                return_value=([7], {7: "2026-07-03 12:00:00"}, False),
            ),
            patch.object(connector, "_search_read", return_value=orders) as search_read,
            patch.object(connector, "_bulk_partner_ref_map", return_value={4: "K-4"}),
            patch.object(connector, "_bulk_production_origins", return_value=set()),
            patch.object(
                connector,
                "_delivery_dates_by_sale_ids",
                return_value={
                    7: {
                        "first": "2026-07-01 10:00:00",
                        "actual": "2026-07-03 12:00:00",
                    }
                },
            ),
        ):
            result = asyncio.run(connector.query_orders(query))

        domain = search_read.call_args.args[1]
        assert ["delivery_status", "=", "full"] in domain
        assert result["rows"] == [
            {
                "order_id": "S7",
                "order_date": "2026-07-01 09:00:00",
                "actual_delivery_date": "2026-07-03 12:00:00",
            }
        ]

    def test_first_delivery_filter_is_not_mapped_to_effective_date(self):
        connector = _odoo_connector()
        query = build_order_query(
            {
                "filters": [
                    {
                        "field": "first_delivery_date",
                        "operator": "gte",
                        "value": "2026-07-01",
                    }
                ]
            }
        )

        assert connector._order_domain_from_query(query) == []

    def test_first_delivery_sort_uses_minimum_done_transfer(self):
        connector = _odoo_connector()
        query = build_order_query(
            {
                "fields": ["order_id", "first_delivery_date"],
                "sort_by": "first_delivery_date",
                "sort_direction": "asc",
                "limit": 1,
            }
        )
        order = {
            "id": 7,
            "name": "S7",
            "partner_id": [4, "Kunde"],
            "state": "sale",
            "delivery_status": "partial",
            "date_order": "2026-07-01 09:00:00",
            "commitment_date": False,
            "effective_date": "2026-07-03 12:00:00",
            "amount_total": 100,
            "currency_id": [1, "EUR"],
            "order_line": [],
            "picking_ids": [70, 71],
        }

        with (
            patch.object(
                connector,
                "_delivery_sorted_candidates",
                return_value=([7], {7: "2026-07-01 10:00:00"}, False),
            ) as candidates,
            patch.object(connector, "_search_read", return_value=[order]),
            patch.object(connector, "_bulk_partner_ref_map", return_value={4: "K-4"}),
            patch.object(connector, "_bulk_production_origins", return_value=set()),
            patch.object(
                connector,
                "_delivery_dates_by_sale_ids",
                return_value={
                    7: {
                        "first": "2026-07-01 10:00:00",
                        "actual": "2026-07-03 12:00:00",
                    }
                },
            ) as dates,
        ):
            result = asyncio.run(connector.query_orders(query))

        candidates.assert_called_once_with("asc", "min")
        assert dates.call_args.kwargs["known_first"] == {7: "2026-07-01 10:00:00"}
        assert result["rows"] == [{"order_id": "S7", "first_delivery_date": "2026-07-01 10:00:00"}]


def test_agent_registry_registers_erp_orders():
    from kiara.web.api_chat import QueryRequest, _build_agent_tool_registry

    config = {
        "tools": {"enabled": True, "erp": {"backend": "mock"}},
        "retrieval": {},
    }
    degradation = MagicMock()
    degradation.is_feature_available.return_value = True
    with (
        patch("kiara.retrieval.retriever.HybridRetriever", return_value=MagicMock()),
        patch("kiara.retrieval.reranker.create_reranker", return_value=MagicMock()),
        patch("kiara.web.api_health._get_degradation_manager", return_value=degradation),
        patch("kiara.tools.erp_connectors.ConnectionRegistry.load_from_config"),
    ):
        registry = _build_agent_tool_registry(
            config,
            QueryRequest(question="Test", mode="agent"),
            None,
        )

    assert registry.get_tool("erp_orders") is not None
