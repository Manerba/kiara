"""Konkrete ERP-Tools fuer KIara.

Vier Tools fuer den Zugriff auf ERP-Daten: Auftragsstatus, Lagerbestand,
Kundenstammdaten und Fertigungsstatus. Alle erben von ERPToolBase, die den
gemeinsamen execute()-Boilerplate (Validierung, Backend-Query, Error-Handling,
None-Check) kapselt. Die Subklassen definieren nur definition() und
_format_result().
"""

import logging
from abc import abstractmethod

from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.erp_backend import ERPBackend
from kiara.tools.erp_connectors.registry import ERPAssignmentBrokenError

logger = logging.getLogger(__name__)


def _float_or_default(value, default: float = 0.0) -> float:
    """Konvertiert ERP-Zahlen robust; fehlerhafte Backend-Werte gelten als fehlend."""
    try:
        return float(value or default)
    except (TypeError, ValueError):
        logger.warning("ERP-Zahlenwert nicht numerisch: %r", value)
        return default


class ERPToolBase(BaseTool):
    """Gemeinsame Basisklasse fuer alle ERP-Tools.

    Kapselt den wiederkehrenden execute()-Ablauf:
    1. Parameter validieren
    2. Primaer-ID aus params extrahieren
    3. Backend-Query ausfuehren (mit Error-Handling)
    4. None-Check (Entity nicht gefunden)
    5. Formatierung an Subklasse delegieren

    Subklassen muessen definieren:
    - definition() — ToolDefinition mit Metadaten
    - _primary_param — Name des Pflicht-Parameters (z.B. "order_id")
    - _entity_label — Menschenlesbare Bezeichnung (z.B. "Auftrag")
    - _backend_query() — Coroutine die das Backend aufruft
    - _format_result() — Ergebnis-dict in display_text umwandeln
    """

    def __init__(self, backend: ERPBackend) -> None:
        self._backend = backend

    @property
    @abstractmethod
    def _primary_param(self) -> str:
        """Name des Pflicht-Parameters (z.B. 'order_id')."""
        ...

    @property
    @abstractmethod
    def _entity_label(self) -> str:
        """Menschenlesbare Bezeichnung der Entity (z.B. 'Auftrag')."""
        ...

    @abstractmethod
    async def _backend_query(self, params: dict) -> dict | None:
        """Fuehrt die Backend-Abfrage aus.

        Args:
            params: Die validierten Tool-Parameter.

        Returns:
            Das Ergebnis-Dict vom Backend, oder None wenn nicht gefunden.
        """
        ...

    @abstractmethod
    def _format_result(self, data: dict, primary_id: str) -> str:
        """Formatiert das Backend-Ergebnis als menschenlesbaren Text.

        Args:
            data: Das Ergebnis-Dict vom Backend (garantiert nicht None).
            primary_id: Der Wert des Primaer-Parameters.

        Returns:
            Formatierter display_text.
        """
        ...

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        """Fuehrt die ERP-Abfrage aus (Template-Method-Pattern).

        Validiert Parameter, ruft das Backend auf, prueft auf None
        und delegiert die Formatierung an _format_result().
        """
        if error := self._validate_or_error(params):
            return error

        primary_id = params[self._primary_param]
        try:
            result = await self._backend_query(params)
        except ERPAssignmentBrokenError as exc:
            logger.error(
                "%s-Tool hat defekte Connection-Zuordnung: %s",
                self._entity_label,
                exc,
            )
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                display_text=(
                    f"Die ERP-Anbindung fuer '{self._entity_label}' ist "
                    f"defekt: {exc} Bitte Admin kontaktieren, um die "
                    f"Tool-Zuordnung im ERP-Tab zu pruefen."
                ),
            )
        except Exception as exc:
            logger.error(
                "Fehler bei %s-Abfrage '%s': %s",
                self._entity_label,
                primary_id,
                exc,
            )
            return ToolResult(
                success=False,
                data=None,
                error="ERP-Abfrage fehlgeschlagen. Details siehe Server-Log.",
                display_text="",
            )

        if result is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"{self._entity_label} '{primary_id}' nicht gefunden.",
                display_text=(
                    f"{self._entity_label} '{primary_id}' wurde im ERP-System nicht gefunden."
                ),
            )

        display_text = self._format_result(result, primary_id)
        return ToolResult(
            success=True,
            data=result,
            error=None,
            display_text=display_text,
        )


class ERPOrderStatusTool(ERPToolBase):
    """Auftragsstatus aus dem ERP-System abfragen."""

    _primary_param = "order_id"
    _entity_label = "Auftrag"

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Auftragsabfragen."""
        return ToolDefinition(
            name="erp_order_status",
            description=(
                "Auftragsstatus aus dem ERP-System abfragen. "
                "Gibt Status, Liefertermin und Positionen zurueck."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Auftragsnummer, z.B. A-1001",
                    },
                },
                "required": ["order_id"],
            },
            requires_confirmation=False,
            required_permissions=["ERP-Leser"],
            category="erp",
        )

    async def _backend_query(self, params: dict) -> dict | None:
        return await self._backend.query_order(params["order_id"])

    def _format_result(self, data: dict, primary_id: str) -> str:
        pos_lines = []
        for pos in data.get("positions", []):
            pos_lines.append(
                f"  Pos {pos.get('pos', '?')}: {pos.get('description', '-')} "
                f"({pos.get('quantity', 0)} x {_float_or_default(pos.get('unit_price')):.2f} EUR)"
            )
        positions_text = "\n".join(pos_lines) if pos_lines else "  Keine Positionen"

        return (
            f"Auftrag: {data.get('order_id', primary_id)}\n"
            f"Kunde: {data.get('customer_name', 'unbekannt')}\n"
            f"Status: {data.get('status', 'unbekannt')}\n"
            f"Bestelldatum: {data.get('order_date', '-')}\n"
            f"Liefertermin: {data.get('delivery_date', '-')}\n"
            f"Gesamtbetrag: {_float_or_default(data.get('total_amount')):.2f} EUR\n"
            f"Positionen:\n{positions_text}"
        )


class ERPStockLevelTool(ERPToolBase):
    """Aktuellen Lagerbestand eines Artikels pruefen."""

    _primary_param = "article_id"
    _entity_label = "Artikel"

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Bestandsabfragen."""
        return ToolDefinition(
            name="erp_stock_level",
            description=(
                "Aktuellen Lagerbestand eines Artikels pruefen. "
                "Gibt Menge, Einheit und Lagerort zurueck."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "article_id": {
                        "type": "string",
                        "description": "Artikelnummer, z.B. ART-10001",
                    },
                },
                "required": ["article_id"],
            },
            requires_confirmation=False,
            required_permissions=["ERP-Leser"],
            category="erp",
        )

    async def _backend_query(self, params: dict) -> dict | None:
        return await self._backend.query_stock(params["article_id"])

    def _format_result(self, data: dict, primary_id: str) -> str:
        stock_qty = _float_or_default(data.get("stock_quantity"))
        min_stock = _float_or_default(data.get("min_stock"))
        stock_warning = ""
        if stock_qty < min_stock:
            stock_warning = (
                f"\nACHTUNG: Bestand unter Mindestbestand! "
                f"(Mindestbestand: {min_stock} {data.get('unit', 'Stueck')})"
            )

        return (
            f"Artikel: {data.get('article_id', primary_id)}\n"
            f"Bezeichnung: {data.get('description', '-')}\n"
            f"Kategorie: {data.get('category', '-')}\n"
            f"Bestand: {stock_qty} {data.get('unit', 'Stueck')}\n"
            f"Lagerort: {data.get('warehouse', '-')}\n"
            f"Stueckpreis: {_float_or_default(data.get('unit_price')):.2f} EUR\n"
            f"Lieferant: {data.get('supplier', '-')}\n"
            f"Lieferzeit: {data.get('lead_time_days', '-')} Tage"
            f"{stock_warning}"
        )


class ERPCustomerInfoTool(ERPToolBase):
    """Kundenstammdaten aus dem ERP-System abfragen."""

    _primary_param = "customer_id"
    _entity_label = "Kunde"

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Kundenabfragen."""
        return ToolDefinition(
            name="erp_customer_info",
            description=("Kundenstammdaten abfragen: Adresse, Ansprechpartner, letzte Auftraege."),
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Kundennummer, z.B. K-1001",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optionale Freitext-Suche in Kundendaten",
                    },
                },
                "required": ["customer_id"],
            },
            requires_confirmation=False,
            required_permissions=["ERP-Vertrieb"],
            category="erp",
        )

    async def _backend_query(self, params: dict) -> dict | None:
        query = params.get("query")
        return await self._backend.query_customer(params["customer_id"], query=query)

    def _format_result(self, data: dict, primary_id: str) -> str:
        # Support nested (Odoo) and flat (SQL) result formats
        address = data.get("address")
        if not isinstance(address, dict):
            address = {
                "street": data.get("street", "-"),
                "zip": data.get("zip", ""),
                "city": data.get("city", "-"),
                "country": data.get("country", "-"),
            }
        address_text = (
            f"{address.get('street', '-')}, "
            f"{address.get('zip', '')} {address.get('city', '-')}, "
            f"{address.get('country', '-')}"
        )

        contact = data.get("contact")
        if not isinstance(contact, dict):
            contact = {
                "name": data.get("contact_name", "-"),
                "phone": data.get("phone", "-"),
                "email": data.get("email", "-"),
            }
        contact_text = (
            f"{contact.get('name', '-')} "
            f"(Tel: {contact.get('phone', '-')}, "
            f"E-Mail: {contact.get('email', '-')})"
        )

        last_orders = data.get("last_orders", [])
        orders_text = ", ".join(last_orders) if last_orders else "Keine"

        return (
            f"Kunde: {data.get('name', primary_id)} ({data.get('customer_id', primary_id)})\n"
            f"Adresse: {address_text}\n"
            f"Ansprechpartner: {contact_text}\n"
            f"Kunde seit: {data.get('customer_since', '-')}\n"
            f"Zahlungsbedingungen: {data.get('payment_terms', '-')}\n"
            f"Kreditlimit: {_float_or_default(data.get('credit_limit', 0)):.2f} EUR\n"
            f"Letzte Auftraege: {orders_text}"
        )


class ERPProductionStatusTool(ERPToolBase):
    """Fertigungsstatus aus dem ERP-System abfragen."""

    _primary_param = "order_id"
    _entity_label = "Fertigungsauftrag"

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Fertigungsabfragen."""
        return ToolDefinition(
            name="erp_production_status",
            description=(
                "Fertigungsstatus: Welche Arbeitsgaenge sind abgeschlossen, was steht noch aus?"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Fertigungsauftragsnummer, z.B. FA-2001",
                    },
                },
                "required": ["order_id"],
            },
            requires_confirmation=False,
            required_permissions=["ERP-Fertigung"],
            category="erp",
        )

    async def _backend_query(self, params: dict) -> dict | None:
        return await self._backend.query_production(params["order_id"])

    def _format_result(self, data: dict, primary_id: str) -> str:
        operations = data.get("operations", [])
        op_lines = []
        completed_count = 0
        total_count = len(operations)
        for op in operations:
            status = op.get("status", "unbekannt")
            status_icon = {
                "abgeschlossen": "[OK]",
                "in_bearbeitung": "[>>]",
                "offen": "[  ]",
            }.get(status, "[??]")
            if status == "abgeschlossen":
                completed_count += 1

            hours_text = ""
            actual = op.get("actual_hours")
            planned = op.get("planned_hours")
            if actual is not None and planned is not None:
                hours_text = f" ({float(actual):.1f}h / {float(planned):.1f}h geplant)"
            elif planned is not None:
                hours_text = f" ({float(planned):.1f}h geplant)"

            op_lines.append(
                f"  {status_icon} AG {op.get('op_number', '?')}: "
                f"{op.get('name', '-')} @ {op.get('workstation', '-')}"
                f"{hours_text}"
            )

        operations_text = "\n".join(op_lines) if op_lines else "  Keine Arbeitsgaenge"
        progress_pct = (
            f"{completed_count}/{total_count} ({completed_count * 100 // total_count}%)"
            if total_count > 0
            else "0/0"
        )

        return (
            f"Fertigungsauftrag: {data.get('production_order_id', primary_id)}\n"
            f"Bezugsauftrag: {data.get('order_reference', '-')}\n"
            f"Artikel: {data.get('article_description', '-')} "
            f"({data.get('article_id', '-')})\n"
            f"Menge: {data.get('quantity', '-')}\n"
            f"Status: {data.get('status', 'unbekannt')}\n"
            f"Zeitraum: {data.get('start_date', '-')} bis "
            f"{data.get('planned_end_date', '-')}\n"
            f"Fortschritt: {progress_pct}\n"
            f"Arbeitsgaenge:\n{operations_text}"
        )
