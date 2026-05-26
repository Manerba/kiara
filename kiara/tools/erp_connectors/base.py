"""Abstrakte Basisklasse fuer ERP-Konnektoren.

Definiert das Interface, das alle konkreten Konnektoren implementieren
muessen, sowie die Datenklassen fuer Konnektor-Definitionen und
Verbindungsinformationen.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ConnectorDefinition:
    """Beschreibt einen Konnektor-Typ fuer das Admin-UI.

    Attributes:
        type_id: Technischer Identifier, z.B. ``"odoo"``, ``"generic_sql"``.
        display_name: Anzeigename, z.B. ``"Odoo (XML-RPC)"``.
        config_fields: JSON-Schema-artige Feldbeschreibungen fuer die
            typ-spezifische Konfiguration (Name, Label, Typ, Pflichtfeld).
    """

    type_id: str
    display_name: str
    config_fields: list[dict] = field(default_factory=list)


@dataclass
class ConnectionInfo:
    """Gespeicherte ERP-Verbindung.

    Attributes:
        id: Eindeutige ID (UUID-String).
        name: Frei waehlbarer Anzeigename, z.B. ``"Odoo Produktion"``.
        connector_type: Typ-ID des Konnektors, z.B. ``"odoo"``.
        config: Typ-spezifische Konfiguration (URL, Credentials, ...).
        enabled: Ob die Verbindung aktiv ist.
    """

    id: str
    name: str
    connector_type: str
    config: dict = field(default_factory=dict)
    enabled: bool = True


class ERPConnectorBase(ABC):
    """Abstrakte Basis fuer alle ERP-Konnektoren.

    Jeder Konnektor implementiert die 4 Query-Methoden plus einen
    Verbindungstest. Die Instanziierung erfolgt ueber die
    ``ConnectionRegistry``, die eine ``ConnectionInfo`` uebergibt.
    """

    def __init__(self, connection: ConnectionInfo) -> None:
        self._connection = connection

    @classmethod
    @abstractmethod
    def connector_definition(cls) -> ConnectorDefinition:
        """Statische Beschreibung des Konnektor-Typs (fuer UI-Rendering)."""
        ...

    @abstractmethod
    async def test_connection(self) -> dict:
        """Verbindung testen.

        Returns:
            Dict mit ``{"success": bool, "message": str, "details": dict}``.
        """
        ...

    @abstractmethod
    async def query_order(self, order_id: str) -> dict | None:
        """Auftrag abfragen."""
        ...

    @abstractmethod
    async def query_stock(self, article_id: str) -> dict | None:
        """Lagerbestand abfragen."""
        ...

    @abstractmethod
    async def query_customer(self, customer_id: str, query: str | None = None) -> dict | None:
        """Kundenstammdaten abfragen. Optional Freitext-Suche ueber query."""
        ...

    @abstractmethod
    async def query_production(self, order_id: str) -> dict | None:
        """Fertigungsauftrag abfragen."""
        ...

    async def list_all_customers(self) -> list[dict]:
        """Alle Kunden fuer ERP-Graph-Sync auflisten."""
        raise NotImplementedError(self._bulk_not_supported_message("list_all_customers"))

    async def list_all_articles(self) -> list[dict]:
        """Alle Artikel fuer ERP-Graph-Sync auflisten."""
        raise NotImplementedError(self._bulk_not_supported_message("list_all_articles"))

    async def list_all_orders(self) -> list[dict]:
        """Alle Auftraege fuer ERP-Graph-Sync auflisten."""
        raise NotImplementedError(self._bulk_not_supported_message("list_all_orders"))

    async def list_all_production_orders(self) -> list[dict]:
        """Alle Fertigungsauftraege fuer ERP-Graph-Sync auflisten."""
        raise NotImplementedError(self._bulk_not_supported_message("list_all_production_orders"))

    @classmethod
    def _bulk_not_supported_message(cls, method_name: str) -> str:
        type_id = cls.connector_definition().type_id
        return f"ERP-Konnektor '{type_id}' unterstuetzt Bulk-Methode {method_name} nicht."
