"""Aktions-Tools fuer KIara.

Tools die Aktionen vorschlagen (nicht ausfuehren):
- GenerateStatusMailTool: E-Mail-Entwurf mit Auftragsstatus
- GenerateChecklistTool: Strukturierte Checkliste zu einem Thema
"""

import logging
import re

from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.erp_backend import ERPBackend

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")


def _normalize_email(value: object) -> str:
    """Bereinigt und validiert eine einzelne Empfaengeradresse."""
    recipient = str(value or "").replace("\n", "").replace("\r", "").strip()
    if not recipient:
        return ""
    if not _EMAIL_RE.fullmatch(recipient):
        raise ValueError("Ungueltige Empfaenger-E-Mail-Adresse")
    return recipient


class GenerateStatusMailTool(BaseTool):
    """Erstellt einen E-Mail-Entwurf mit aktuellem Auftragsstatus.

    Holt Auftragsdaten via ERPBackend und generiert einen professionellen
    Mail-Entwurf. Die E-Mail wird NICHT direkt gesendet — der Benutzer
    muss den Entwurf bestaetigen.
    """

    def __init__(self, backend: ERPBackend) -> None:
        self._backend = backend

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Status-Mail-Erstellung."""
        return ToolDefinition(
            name="generate_status_mail",
            description=(
                "Erstellt einen E-Mail-Entwurf mit aktuellem Auftragsstatus fuer den Kunden."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Auftragsnummer, z.B. A-1001",
                    },
                    "recipient": {
                        "type": "string",
                        "description": "E-Mail-Empfaenger (optional, wird aus Kundendaten ermittelt)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Sprache des Mail-Entwurfs",
                        "enum": ["de", "en"],
                    },
                },
                "required": ["order_id"],
            },
            requires_confirmation=True,
            required_permissions=["ERP-Vertrieb"],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        """Generiert einen Status-Mail-Entwurf basierend auf Auftragsdaten."""
        if error := self._validate_or_error(params):
            return error

        order_id = params["order_id"]
        language = params.get("language", "de")

        # Auftragsdaten holen
        try:
            order = await self._backend.query_order(order_id)
        except Exception as exc:
            logger.error("Fehler bei Auftragsabfrage fuer Mail '%s': %s", order_id, exc)
            return ToolResult(
                success=False,
                data=None,
                error=f"ERP-Abfrage fehlgeschlagen: {exc}",
                display_text="",
            )

        if order is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"Auftrag '{order_id}' nicht gefunden.",
                display_text=f"Auftrag '{order_id}' nicht gefunden — Mail kann nicht erstellt werden.",
            )

        # Kundendaten holen fuer Empfaenger-Ermittlung
        customer_id = order.get("customer_id", "")
        recipient = params.get("recipient", "")
        contact_name = order.get("customer_name", "Kunde")

        if customer_id and not recipient:
            try:
                customer = await self._backend.query_customer(customer_id)
                if customer:
                    contact = customer.get("contact", {})
                    recipient = contact.get("email", "")
                    contact_name = contact.get("name", order.get("customer_name", "Kunde"))
            except Exception as exc:
                logger.warning("Kundendaten fuer Mail nicht abrufbar: %s", exc)

        # Status-Text bestimmen
        status = order.get("status", "unbekannt")
        status_texts_de = {
            "offen": "Ihr Auftrag befindet sich in Bearbeitung",
            "in_fertigung": "Ihr Auftrag befindet sich derzeit in der Fertigung",
            "geliefert": "Ihr Auftrag wurde ausgeliefert",
            "storniert": "Ihr Auftrag wurde storniert",
        }
        status_texts_en = {
            "offen": "Your order is currently being processed",
            "in_fertigung": "Your order is currently in production",
            "geliefert": "Your order has been delivered",
            "storniert": "Your order has been cancelled",
        }

        # Positionen formatieren
        positions = order.get("positions", [])
        pos_lines = []
        for pos in positions:
            desc = str(pos.get("description", "-")).replace("\n", " ").replace("\r", " ")[:200]
            raw_qty = pos.get("quantity")
            try:
                qty = float(raw_qty) if raw_qty is not None else 0.0
            except (TypeError, ValueError):
                qty = 0.0
            qty_str = str(int(qty)) if qty == int(qty) else f"{qty}"
            pos_lines.append(f"  - {desc} (Menge: {qty_str})")
        positions_text = "\n".join(pos_lines) if pos_lines else "  Keine Positionen"

        # Header-Injection verhindern: Newlines aus User-Daten entfernen
        order_id = str(order_id).replace("\n", "").replace("\r", "")
        contact_name = str(contact_name).replace("\n", "").replace("\r", "")
        try:
            recipient = _normalize_email(recipient)
        except ValueError as exc:
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                display_text=str(exc),
            )

        if language == "en":
            status_text = status_texts_en.get(
                status, "Current status information will follow shortly"
            )
            subject = f"Order Status Update: {order_id}"
            body = (
                f"Dear {contact_name},\n\n"
                f"We would like to inform you about the current status of your order {order_id}.\n\n"
                f"{status_text}.\n\n"
                f"Order details:\n"
                f"  Order date: {order.get('order_date', '-')}\n"
                f"  Estimated delivery: {order.get('delivery_date', '-')}\n"
                f"  Total amount: {float(order.get('total_amount', 0) or 0):.2f} EUR\n\n"
                f"Items:\n{positions_text}\n\n"
                f"If you have any questions, please do not hesitate to contact us.\n\n"
                f"Best regards,\n[Your Name]"
            )
        else:
            status_text = status_texts_de.get(
                status, "Eine detaillierte Statusinformation folgt in Kuerze"
            )
            subject = f"Statusaktualisierung Auftrag {order_id}"
            body = (
                f"Sehr geehrte/r {contact_name},\n\n"
                f"hiermit informieren wir Sie ueber den aktuellen Stand Ihres Auftrags {order_id}.\n\n"
                f"{status_text}.\n\n"
                f"Auftragsdetails:\n"
                f"  Bestelldatum: {order.get('order_date', '-')}\n"
                f"  Voraussichtliche Lieferung: {order.get('delivery_date', '-')}\n"
                f"  Gesamtbetrag: {float(order.get('total_amount', 0) or 0):.2f} EUR\n\n"
                f"Positionen:\n{positions_text}\n\n"
                f"Bei Rueckfragen stehen wir Ihnen gerne zur Verfuegung.\n\n"
                f"Mit freundlichen Gruessen,\n[Ihr Name]"
            )

        display_text = (
            f"--- E-Mail-Entwurf ---\n"
            f"An: {recipient if recipient else '(nicht ermittelt)'}\n"
            f"Betreff: {subject}\n"
            f"---\n"
            f"{body}\n"
            f"--- Ende des Entwurfs ---"
        )

        data = {
            "recipient": recipient,
            "subject": subject,
            "body": body,
            "order_id": order_id,
            "language": language,
        }

        return ToolResult(
            success=True,
            data=data,
            error=None,
            display_text=display_text,
        )


class GenerateChecklistTool(BaseTool):
    """Erstellt eine strukturierte Checkliste zu einem Thema.

    Generiert eine statische Vorlage basierend auf dem angegebenen Thema.
    Kein LLM-Zugriff innerhalb des Tools — die Checkliste wird aus einer
    Vorlagen-Struktur erzeugt.
    """

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Checklisten-Erstellung."""
        return ToolDefinition(
            name="generate_checklist",
            description=(
                "Erstellt eine strukturierte Checkliste aus der Wissensbasis zu einem Thema."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Thema der Checkliste",
                        "minLength": 2,
                    },
                    "format": {
                        "type": "string",
                        "description": "Ausgabeformat der Checkliste",
                        "enum": ["markdown", "text"],
                    },
                },
                "required": ["topic"],
            },
            category="action",
            requires_confirmation=False,
            required_permissions=[],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        """Generiert eine Checklisten-Vorlage zum angegebenen Thema."""
        if error := self._validate_or_error(params):
            return error

        topic = params["topic"]
        fmt = params.get("format", "markdown")

        # Checklisten-Struktur generieren
        sections = [
            {
                "title": "Vorbereitung",
                "items": [
                    f"Anforderungen fuer '{topic}' definieren",
                    "Zustaendige Personen identifizieren",
                    "Zeitrahmen festlegen",
                    "Ressourcen und Budget pruefen",
                ],
            },
            {
                "title": "Durchfuehrung",
                "items": [
                    f"Dokumentation zu '{topic}' sichten",
                    "Ist-Zustand analysieren",
                    "Soll-Zustand definieren",
                    "Massnahmenplan erstellen",
                    "Umsetzung durchfuehren",
                ],
            },
            {
                "title": "Qualitaetssicherung",
                "items": [
                    "Ergebnis pruefen und dokumentieren",
                    "Tests durchfuehren",
                    "Feedback einholen",
                    "Abnahme durch Verantwortlichen",
                ],
            },
            {
                "title": "Abschluss",
                "items": [
                    "Dokumentation aktualisieren",
                    "Beteiligte informieren",
                    "Lessons Learned festhalten",
                    "Aufgabe als erledigt markieren",
                ],
            },
        ]

        if fmt == "markdown":
            display_text = self._format_markdown(topic, sections)
        else:
            display_text = self._format_text(topic, sections)

        data = {
            "topic": topic,
            "format": fmt,
            "sections": sections,
        }

        return ToolResult(
            success=True,
            data=data,
            error=None,
            display_text=display_text,
        )

    @staticmethod
    def _format_markdown(topic: str, sections: list[dict]) -> str:
        """Formatiert die Checkliste als Markdown."""
        lines = [f"# Checkliste: {topic}", ""]
        for section in sections:
            lines.append(f"## {section['title']}")
            for item in section["items"]:
                lines.append(f"- [ ] {item}")
            lines.append("")
        return "\n".join(lines).rstrip()

    @staticmethod
    def _format_text(topic: str, sections: list[dict]) -> str:
        """Formatiert die Checkliste als reinen Text."""
        lines = [f"Checkliste: {topic}", "=" * (13 + len(topic)), ""]
        for section in sections:
            lines.append(f"{section['title']}:")
            lines.append("-" * (len(section["title"]) + 1))
            for i, item in enumerate(section["items"], 1):
                lines.append(f"  [ ] {i}. {item}")
            lines.append("")
        return "\n".join(lines).rstrip()
