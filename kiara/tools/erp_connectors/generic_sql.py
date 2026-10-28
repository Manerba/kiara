"""Generischer SQL-Konnektor fuer ERP-Daten.

Migriert die SQL-Backend-Logik aus ``erp_backend.py`` in einen
eigenstaendigen Konnektor. Nutzt das Column-Mapping-Pattern
inkl. Sub-Tabellen (Positionen, Arbeitsgaenge).
"""

import asyncio
import logging

from kiara.config import DEFAULT_TIMEOUTS
from kiara.tools.erp_connectors.base import (
    ConnectionInfo,
    ConnectorDefinition,
    ERPConnectorBase,
)

logger = logging.getLogger(__name__)


class GenericSQLConnector(ERPConnectorBase):
    """SQL-Konnektor mit konfigurierbarem Column-Mapping."""

    def __init__(self, connection: ConnectionInfo) -> None:
        super().__init__(connection)
        self._dsn: str = connection.config.get("dsn", "")
        self._driver: str = connection.config.get("driver", "auto")
        self._mapping: dict = connection.config.get("mapping", {})
        self._query_timeout = float(
            connection.config.get("query_timeout", DEFAULT_TIMEOUTS["erp_query"])
        )
        self._bulk_timeout = float(
            connection.config.get("bulk_timeout", DEFAULT_TIMEOUTS["erp_bulk"])
        )

    @classmethod
    def connector_definition(cls) -> ConnectorDefinition:
        return ConnectorDefinition(
            type_id="generic_sql",
            display_name="Generisch SQL",
            config_fields=[
                {
                    "name": "dsn",
                    "label": "Verbindungsstring (DSN)",
                    "type": "password",
                    "placeholder": "DRIVER={ODBC Driver 18...};SERVER=...",
                    "required": True,
                },
                {
                    "name": "driver",
                    "label": "Treiber",
                    "type": "select",
                    "options": ["auto", "mssql", "postgresql", "mysql"],
                    "required": False,
                },
                {
                    "name": "mapping",
                    "label": "Tabellen-Mapping (JSON)",
                    "type": "json",
                    "required": False,
                },
            ],
        )

    async def test_connection(self) -> dict:
        """Verbindungstest ueber SQLConnector."""
        try:
            from kiara.connectors.sql_connector import SQLConnector

            connector = SQLConnector(
                self._dsn,
                self._driver,
                statement_timeout=self._query_timeout,
            )
            result = await asyncio.to_thread(connector.test_connection)
            return result
        except Exception as exc:
            return {"success": False, "message": str(exc), "details": {}}

    async def query_order(self, order_id: str) -> dict | None:
        mapping = self._mapping.get("orders")
        if not mapping:
            return None
        return await asyncio.to_thread(self._sql_query_entity, mapping, order_id)

    async def query_stock(self, article_id: str) -> dict | None:
        mapping = self._mapping.get("articles")
        if not mapping:
            return None
        return await asyncio.to_thread(self._sql_query_entity, mapping, article_id)

    async def query_customer(self, customer_id: str, query: str | None = None) -> dict | None:
        mapping = self._mapping.get("customers")
        if not mapping:
            return None
        return await asyncio.to_thread(self._sql_query_entity, mapping, customer_id)

    async def query_production(self, order_id: str) -> dict | None:
        mapping = self._mapping.get("production_orders")
        if not mapping:
            return None
        return await asyncio.to_thread(self._sql_query_entity, mapping, order_id)

    async def list_all_customers(self) -> list[dict]:
        return await self._list_all("customers")

    async def list_all_articles(self) -> list[dict]:
        return await self._list_all("articles")

    async def list_all_orders(self) -> list[dict]:
        return await self._list_all("orders")

    async def list_all_production_orders(self) -> list[dict]:
        return await self._list_all("production_orders")

    async def _list_all(self, entity_type: str) -> list[dict]:
        mapping = self._mapping.get(entity_type)
        if not mapping:
            raise ValueError(f"GenericSQL-Bulk-Mapping fuer '{entity_type}' fehlt.")
        return await asyncio.to_thread(
            self._sql_query_all_entities,
            entity_type,
            mapping,
        )

    def _sql_query_entity(self, mapping: dict, entity_id: str) -> dict | None:
        """Generische SQL-Abfrage mit Spalten-Mapping (synchron).

        Delegiert an die gemeinsame Implementierung in
        ``sql_utils.sql_query_entity()``.
        """
        from kiara.tools.erp_connectors.sql_utils import sql_query_entity

        return sql_query_entity(
            self._dsn,
            self._driver,
            mapping,
            entity_id,
            statement_timeout_seconds=self._query_timeout,
        )

    def _sql_query_all_entities(self, entity_type: str, mapping: dict) -> list[dict]:
        """Generische SQL-Bulk-Abfrage mit explizitem Entity-Kontext."""
        from kiara.tools.erp_connectors.sql_utils import sql_query_all_entities

        return sql_query_all_entities(
            self._dsn,
            self._driver,
            entity_type,
            mapping,
            statement_timeout_seconds=self._bulk_timeout,
        )
