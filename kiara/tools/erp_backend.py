"""ERP-Backend-Abstraktion fuer KIara-Tools.

Unterstuetzt verschiedene Backend-Typen:
- **mock**: Laedt Testdaten aus einer JSON-Datei (Standard fuer Entwicklung).
- **rest**: Anbindung an ein REST-basiertes ERP-System (noch nicht implementiert).
- **sql**: Direkter SQL-Zugriff auf eine ERP-Datenbank via SQLConnector + Spalten-Mapping.

Zusaetzlich kann eine ``ConnectionRegistry`` uebergeben werden, um
Tool-spezifische Konnektoren (Odoo, SQL, ...) zu nutzen. Ohne Registry
oder bei fehlender Tool-Zuordnung wird das Legacy-Backend verwendet.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiara.tools.erp_connectors.registry import ConnectionRegistry

logger = logging.getLogger(__name__)

_DEFAULT_MOCK_DATA_PATH = Path(__file__).parent / "mock_data.json"
_BULK_TOOLS = {
    "customers": "erp_customer_info",
    "articles": "erp_stock_level",
    "orders": "erp_order_status",
    "production_orders": "erp_production_status",
}
_MOCK_COLLECTIONS = ("orders", "articles", "customers", "production_orders")


def _empty_mock_data() -> dict[str, list]:
    return {key: [] for key in _MOCK_COLLECTIONS}


class ERPUpstreamError(RuntimeError):
    """Sanitisierte Fehlerklasse fuer ERP-Bulk-Upstream-Probleme."""


class ERPBackend:
    """Abstrahiert den Zugriff auf ERP-Daten.

    Der Backend-Typ wird ueber die Konfiguration gesteuert::

        tools:
          erp:
            backend: mock          # mock | rest | sql
            sql_dsn: "DRIVER=..."  # Verbindungsstring (nur fuer sql)
            sql_driver: "auto"     # Treiber (auto/mssql/postgresql/mysql)
            mapping:               # Spalten-Mapping pro Entity-Typ
              orders:
                table: AuftragskopfView
                id_column: Auftragsnummer
                columns:
                  order_id: Auftragsnummer
                  ...

    Args:
        config: Das globale KIara-Konfigurationsdict.
        connection_registry: Optionale ConnectionRegistry fuer Konnektor-basierte Abfragen.
    """

    def __init__(
        self,
        config: dict,
        connection_registry: ConnectionRegistry | None = None,
    ) -> None:
        tools_cfg = config.get("tools", {})
        erp_cfg = tools_cfg.get("erp", {})
        self._config = config
        self._registry = connection_registry
        self._backend_type: str = erp_cfg.get("backend", "mock")
        self._mock_data: dict | None = None
        self._sql_dsn: str = erp_cfg.get("sql_dsn", "")
        self._sql_driver: str = erp_cfg.get("sql_driver", "auto")
        from kiara.config import get_timeout

        # Two-Layer-Timeout: SQL-Schicht (sql_utils, statement_timeout) feuert mit
        # diesem Wert; asyncio.wait_for-Aufrufe addieren +5.0s Slack, damit der
        # innere Timeout zuerst greift und eine saubere Fehlermeldung liefert,
        # bevor wait_for unsanft via CancelledError abbricht.
        self._query_timeout = float(get_timeout(config, "erp_query"))
        self._bulk_timeout = float(get_timeout(config, "erp_bulk"))

        if self._backend_type == "mock":
            self._load_mock_data(config)
        elif self._backend_type == "sql":
            # DSN entschluesseln falls Fernet-verschluesselt
            if self._sql_dsn:
                from kiara.shared.crypto import decrypt, is_encrypted

                if is_encrypted(self._sql_dsn):
                    self._sql_dsn = decrypt(self._sql_dsn)
            # DSN kann Credentials enthalten — nur Schema+Host loggen.
            # Achtung: urlparse() auf ODBC-DSNs (z.B. "UID=...;PWD=...;DRIVER=...")
            # interpretiert das erste Feld als path und wuerde Credentials leaken.
            # Daher nur loggen, wenn echte URL-Form (scheme + hostname) erkannt wurde.
            try:
                from urllib.parse import urlparse

                _parsed = urlparse(self._sql_dsn)
                if _parsed.scheme and _parsed.hostname:
                    _safe_dsn = f"{_parsed.scheme}://{_parsed.hostname}:{_parsed.port or '?'}/{_parsed.path.lstrip('/')}"
                    logger.info("ERP-Backend 'sql' konfiguriert (DSN: %s).", _safe_dsn)
                else:
                    logger.info("ERP-Backend 'sql' konfiguriert (ODBC-DSN, Inhalt nicht geloggt).")
            except Exception:
                logger.info("ERP-Backend 'sql' konfiguriert (DSN: nicht parsbar).")
        elif self._backend_type == "rest":
            logger.info("ERP-Backend 'rest' konfiguriert (Verbindung bei Abfrage).")
        else:
            logger.warning(
                "Unbekannter ERP-Backend-Typ: '%s'. Fallback auf 'mock'.", self._backend_type
            )
            self._backend_type = "mock"
            self._load_mock_data(config)

    def _load_mock_data(self, config: dict) -> None:
        """Laedt Mock-Daten aus einer JSON-Datei.

        Sucht den Pfad in der Konfiguration unter ``tools.erp.mock_data``.
        Falls nicht gesetzt, wird die mitgelieferte ``mock_data.json`` verwendet.

        Args:
            config: Das globale KIara-Konfigurationsdict.
        """
        tools_cfg = config.get("tools", {})
        erp_cfg = tools_cfg.get("erp", {})
        mock_path_str = erp_cfg.get("mock_data")

        mock_path = Path(mock_path_str) if mock_path_str else _DEFAULT_MOCK_DATA_PATH

        if not mock_path.exists():
            logger.error("Mock-Datendatei nicht gefunden: %s", mock_path)
            self._mock_data = _empty_mock_data()
            return

        try:
            with open(mock_path, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Fehler beim Laden der Mock-Daten aus %s: %s", mock_path, exc)
            self._mock_data = _empty_mock_data()
            return

        if not isinstance(raw, dict):
            logger.error(
                "Mock-Daten haben falschen Root-Typ (erwartet: dict, erhalten: %s) — Datei: %s",
                type(raw).__name__,
                mock_path,
            )
            self._mock_data = _empty_mock_data()
            return

        self._mock_data = _empty_mock_data()
        for key in _MOCK_COLLECTIONS:
            value = raw.get(key, [])
            if not isinstance(value, list):
                logger.warning(
                    "Mock-Collection '%s' hat falschen Typ (erwartet: list, erhalten: %s) — verwende leere Liste.",
                    key,
                    type(value).__name__,
                )
                continue
            self._mock_data[key] = value

        logger.info(
            "Mock-Daten geladen: %d Auftraege, %d Artikel, %d Kunden, %d Fertigungsauftraege.",
            len(self._mock_data["orders"]),
            len(self._mock_data["articles"]),
            len(self._mock_data["customers"]),
            len(self._mock_data["production_orders"]),
        )

    # ------------------------------------------------------------------
    # Oeffentliche Query-Methoden (pruefen erst Registry, dann Legacy)
    # ------------------------------------------------------------------

    async def query_order(self, order_id: str) -> dict | None:
        """Fragt einen Auftrag anhand der Auftragsnummer ab.

        Args:
            order_id: Die Auftragsnummer, z.B. ``"A-1001"``.

        Returns:
            Auftrags-Dictionary oder None, falls nicht gefunden.
        """
        if self._registry:
            connector = self._registry.get_connector_for_tool("erp_order_status")
            if connector:
                return await asyncio.wait_for(
                    connector.query_order(order_id),
                    timeout=self._query_timeout + 5.0,
                )
        return await self._legacy_query_order(order_id)

    async def query_stock(self, article_id: str) -> dict | None:
        """Fragt den Lagerbestand eines Artikels ab.

        Args:
            article_id: Die Artikelnummer, z.B. ``"ART-10001"``.

        Returns:
            Artikel-Dictionary mit Bestandsinformationen oder None.
        """
        if self._registry:
            connector = self._registry.get_connector_for_tool("erp_stock_level")
            if connector:
                return await asyncio.wait_for(
                    connector.query_stock(article_id),
                    timeout=self._query_timeout + 5.0,
                )
        return await self._legacy_query_stock(article_id)

    async def query_customer(self, customer_id: str, query: str | None = None) -> dict | None:
        """Fragt Kundenstammdaten ab.

        Args:
            customer_id: Die Kundennummer, z.B. ``"K-1001"``.
            query: Optionale Freitext-Suche (nur bei Konnektor-Backend).

        Returns:
            Kunden-Dictionary oder None.
        """
        if self._registry:
            connector = self._registry.get_connector_for_tool("erp_customer_info")
            if connector:
                return await asyncio.wait_for(
                    connector.query_customer(customer_id, query=query),
                    timeout=self._query_timeout + 5.0,
                )
        # Fallback: Legacy (query wird ignoriert, da Legacy ihn nicht unterstuetzt)
        return await self._legacy_query_customer(customer_id)

    async def query_production(self, order_id: str) -> dict | None:
        """Fragt einen Fertigungsauftrag mit Arbeitsgaengen ab.

        Args:
            order_id: Die Fertigungsauftragsnummer, z.B. ``"FA-2001"``.

        Returns:
            Fertigungsauftrags-Dictionary mit Arbeitsgaengen oder None.
        """
        if self._registry:
            connector = self._registry.get_connector_for_tool("erp_production_status")
            if connector:
                return await asyncio.wait_for(
                    connector.query_production(order_id),
                    timeout=self._query_timeout + 5.0,
                )
        return await self._legacy_query_production(order_id)

    # ------------------------------------------------------------------
    # Bulk-API fuer Graph-Sync
    # ------------------------------------------------------------------

    async def list_all_customers(self) -> list[dict]:
        return await self._list_all("customers", "list_all_customers")

    async def list_all_articles(self) -> list[dict]:
        return await self._list_all("articles", "list_all_articles")

    async def list_all_orders(self) -> list[dict]:
        return await self._list_all("orders", "list_all_orders")

    async def list_all_production_orders(self) -> list[dict]:
        return await self._list_all("production_orders", "list_all_production_orders")

    async def _list_all(self, entity_type: str, method_name: str) -> list[dict]:
        if self._registry:
            connector = self._registry.get_connector_for_tool(_BULK_TOOLS[entity_type])
            if connector:
                method = getattr(connector, method_name)
                return await self._await_bulk(method(), method_name)
        return await self._legacy_list_all(entity_type, method_name)

    async def _await_bulk(self, awaitable, method_name: str) -> list[dict]:
        try:
            result = await asyncio.wait_for(
                awaitable,
                timeout=self._bulk_timeout + 5.0,
            )
        except (NotImplementedError, asyncio.TimeoutError, ValueError):
            raise
        except Exception as exc:
            logger.warning("ERP-Bulk-Upstream-Fehler in %s: %s", method_name, exc)
            raise ERPUpstreamError("ERP-Upstream konnte Bulk-Daten nicht laden.") from exc
        # Fail-closed: ``None`` kann bei partiellen Connector-Fehlern auftreten —
        # NICHT als leere Liste behandeln, sonst loescht clear(source='erp_import')
        # bestehende ERP-Graphdaten vor dem Rebuild (ImplScan V28, Finding 6).
        if result is None:
            raise ERPUpstreamError(
                f"ERP-Bulk-Methode '{method_name}' lieferte None "
                "(unvollstaendiger Connector-Fehler)."
            )
        if not isinstance(result, list):
            raise ERPUpstreamError("ERP-Bulk-Methode lieferte keine Liste.")
        return result

    async def _legacy_list_all(self, entity_type: str, method_name: str) -> list[dict]:
        if self._backend_type == "mock":
            return list((self._mock_data or {}).get(entity_type, []))
        if self._backend_type == "sql":
            mapping = self._get_mapping(entity_type)
            if not mapping:
                raise ValueError(f"ERP-SQL-Bulk-Mapping fuer '{entity_type}' fehlt.")
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self._sql_query_all_entities,
                        entity_type,
                        mapping,
                    ),
                    timeout=self._bulk_timeout + 5.0,
                )
            except asyncio.TimeoutError:
                raise
            except ValueError:
                raise
            except Exception as exc:
                logger.warning(
                    "ERP-SQL-Bulk-Upstream-Fehler fuer %s: %s",
                    entity_type,
                    exc,
                )
                raise ERPUpstreamError("ERP-SQL-Upstream konnte Bulk-Daten nicht laden.") from exc
        if self._backend_type == "rest":
            raise NotImplementedError(
                f"REST-Backend unterstuetzt Bulk-Methode {method_name} nicht."
            )
        raise NotImplementedError(
            f"ERP-Backend '{self._backend_type}' unterstuetzt {method_name} nicht."
        )

    # ------------------------------------------------------------------
    # Legacy-Backend (Mock/SQL/REST)
    # ------------------------------------------------------------------

    async def _legacy_query_order(self, order_id: str) -> dict | None:
        """Bisherige query_order()-Logik (Mock/SQL/REST)."""
        if self._backend_type == "mock":
            return self._find_by_id(self._mock_data.get("orders", []), "order_id", order_id)
        if self._backend_type == "sql":
            mapping = self._get_mapping("orders")
            if not mapping:
                return None
            return await asyncio.wait_for(
                asyncio.to_thread(self._sql_query_entity, mapping, order_id),
                timeout=self._query_timeout + 5.0,
            )
        raise NotImplementedError(
            "REST-Backend fuer Auftragsabfragen ist noch nicht implementiert."
        )

    async def _legacy_query_stock(self, article_id: str) -> dict | None:
        """Bisherige query_stock()-Logik (Mock/SQL/REST)."""
        if self._backend_type == "mock":
            return self._find_by_id(self._mock_data.get("articles", []), "article_id", article_id)
        if self._backend_type == "sql":
            mapping = self._get_mapping("articles")
            if not mapping:
                return None
            return await asyncio.wait_for(
                asyncio.to_thread(self._sql_query_entity, mapping, article_id),
                timeout=self._query_timeout + 5.0,
            )
        raise NotImplementedError(
            "REST-Backend fuer Bestandsabfragen ist noch nicht implementiert."
        )

    async def _legacy_query_customer(self, customer_id: str) -> dict | None:
        """Bisherige query_customer()-Logik (Mock/SQL/REST). Kein query-Parameter."""
        if self._backend_type == "mock":
            return self._find_by_id(
                self._mock_data.get("customers", []), "customer_id", customer_id
            )
        if self._backend_type == "sql":
            mapping = self._get_mapping("customers")
            if not mapping:
                return None
            return await asyncio.wait_for(
                asyncio.to_thread(self._sql_query_entity, mapping, customer_id),
                timeout=self._query_timeout + 5.0,
            )
        raise NotImplementedError("REST-Backend fuer Kundenabfragen ist noch nicht implementiert.")

    async def _legacy_query_production(self, order_id: str) -> dict | None:
        """Bisherige query_production()-Logik (Mock/SQL/REST)."""
        if self._backend_type == "mock":
            return self._find_by_id(
                self._mock_data.get("production_orders", []), "production_order_id", order_id
            )
        if self._backend_type == "sql":
            mapping = self._get_mapping("production_orders")
            if not mapping:
                return None
            return await asyncio.wait_for(
                asyncio.to_thread(self._sql_query_entity, mapping, order_id),
                timeout=self._query_timeout + 5.0,
            )
        raise NotImplementedError(
            "REST-Backend fuer Fertigungsabfragen ist noch nicht implementiert."
        )

    # ------------------------------------------------------------------
    # SQL-Backend
    # ------------------------------------------------------------------

    def _get_mapping(self, entity_type: str) -> dict | None:
        """Holt das Spalten-Mapping fuer einen Entity-Typ."""
        tools_cfg = self._config.get("tools", {})
        erp_cfg = tools_cfg.get("erp", {})
        mappings = erp_cfg.get("mapping", {})
        return mappings.get(entity_type)

    def _sql_query_entity(self, mapping: dict, entity_id: str) -> dict | None:
        """Generische SQL-Abfrage mit Spalten-Mapping.

        Delegiert an die gemeinsame Implementierung in
        ``erp_connectors.sql_utils.sql_query_entity()``.
        """
        from kiara.tools.erp_connectors.sql_utils import sql_query_entity

        return sql_query_entity(
            self._sql_dsn,
            self._sql_driver,
            mapping,
            entity_id,
            statement_timeout_seconds=self._query_timeout,
        )

    def _sql_query_all_entities(self, entity_type: str, mapping: dict) -> list[dict]:
        """Generische SQL-Bulk-Abfrage mit explizitem Entity-Kontext."""
        from kiara.tools.erp_connectors.sql_utils import sql_query_all_entities

        return sql_query_all_entities(
            self._sql_dsn,
            self._sql_driver,
            entity_type,
            mapping,
            statement_timeout_seconds=self._bulk_timeout,
        )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _find_by_id(collection: list[dict], id_field: str, id_value: str) -> dict | None:
        """Sucht in einer Liste von Dicts nach einem Eintrag mit passendem ID-Feld."""
        for item in collection:
            if item.get(id_field) == id_value:
                return item
        return None
