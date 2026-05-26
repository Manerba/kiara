"""Workflow-Tools fuer KIara.

Drei Tools fuer Dokumenten-Workflows:
- SummarizeDocumentTool: Dokument aus der Wissensbasis zusammenfassen
- GenerateFromTemplateTool: Dokument aus Template generieren
- ExportDocumentTool: Dokument in PDF/DOCX/Text exportieren
"""

import logging

from kiara.tools.base import BaseTool, ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class SummarizeDocumentTool(BaseTool):
    """Fasst ein Dokument aus der Wissensbasis zusammen.

    Nutzt den Retriever um Dokument-Chunks zu suchen und den
    DocumentSummarizer fuer die Zusammenfassung.

    Args:
        retriever: Retriever-Instanz fuer die Dokumentensuche.
        summarizer: DocumentSummarizer-Instanz.
    """

    def __init__(
        self, retriever, summarizer, allowed_datasource_ids: list[int] | None = None
    ) -> None:
        self._retriever = retriever
        self._summarizer = summarizer
        self._allowed_datasource_ids = allowed_datasource_ids

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Dokumenten-Zusammenfassung."""
        return ToolDefinition(
            name="summarize_document",
            description=(
                "Fasst ein Dokument oder Thema aus der Wissensbasis zusammen. "
                "Sucht relevante Chunks und erstellt eine strukturierte Zusammenfassung."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Suchbegriff oder Dokumentname, nach dem in der "
                            "Wissensbasis gesucht wird."
                        ),
                        "minLength": 2,
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optionaler inhaltlicher Fokus fuer die Zusammenfassung.",
                    },
                    "format": {
                        "type": "string",
                        "description": "Ausgabeformat der Zusammenfassung.",
                        "enum": ["paragraph", "bullet_points", "structured"],
                    },
                },
                "required": ["source"],
            },
            requires_confirmation=False,
            required_permissions=[],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        """Sucht Chunks via Retriever und fasst sie zusammen."""
        if error := self._validate_or_error(params):
            return error

        source = params["source"]
        focus = params.get("focus")
        output_format = params.get("format", "paragraph")

        # Chunks aus der Wissensbasis suchen
        try:
            results = self._retriever.retrieve(
                query=source,
                k=10,
                filter_datasource_ids=self._allowed_datasource_ids,
            )
        except Exception as exc:
            logger.error("Retrieval fehlgeschlagen fuer '%s': %s", source, exc)
            return ToolResult(
                success=False,
                data=None,
                error=f"Suche fehlgeschlagen: {exc}",
                display_text="",
            )

        if not results:
            return ToolResult(
                success=False,
                data=None,
                error=f"Keine Dokumente zu '{source}' gefunden.",
                display_text=f"Keine relevanten Dokumente zu '{source}' in der Wissensbasis gefunden.",
            )

        # Chunks zu einem Text zusammenfuegen
        combined_text = "\n\n".join(str(r.get("text", "")) for r in results if r.get("text"))

        # Quellen sammeln
        sources = []
        seen = set()
        for r in results:
            src = r.get("metadata", {}).get("source", "")
            if src and src not in seen:
                sources.append(src)
                seen.add(src)

        # Zusammenfassung erstellen
        try:
            summary = await self._summarizer.summarize(
                text=combined_text,
                focus=focus,
                output_format=output_format,
            )
        except Exception as exc:
            logger.error("Zusammenfassung fehlgeschlagen: %s", exc)
            return ToolResult(
                success=False,
                data=None,
                error=f"Zusammenfassung fehlgeschlagen: {exc}",
                display_text="",
            )

        sources_text = "\n".join(f"- {s}" for s in sources) if sources else "(keine)"
        display_text = f"## Zusammenfassung: {source}\n\n{summary}\n\n### Quellen\n{sources_text}"

        return ToolResult(
            success=True,
            data={
                "summary": summary,
                "sources": sources,
                "chunk_count": len(results),
            },
            error=None,
            display_text=display_text,
        )


class GenerateFromTemplateTool(BaseTool):
    """Generiert ein Dokument aus einem Template oder per LLM.

    Args:
        template_engine: TemplateEngine-Instanz.
    """

    # Defensive DoS-Limits: harte Obergrenzen fuer Werte, die als Listen-/Schleifen-
    # Treiber in die Engine fliessen. Verhindert num_items=1_000_000 oder
    # documents=[<500MB-String>] Angriffe.
    _MAX_NUM_ITEMS = 100
    _MIN_MAX_LENGTH = 50
    _MAX_MAX_LENGTH = 10000
    _MAX_DOCUMENTS = 50
    _MAX_DOCUMENT_LEN = 50000
    _MAX_PARTICIPANTS = 100
    _MAX_PARTICIPANT_LEN = 200
    _MAX_CONTEXT_LEN = 50000
    _MAX_FOCUS_LEN = 1000
    _MAX_DATE_LEN = 100

    def __init__(self, template_engine) -> None:
        self._engine = template_engine

    def _validate_variable_limits(self, variables: dict) -> str | None:
        """Erzwingt Ressourcen-Limits fuer DoS-relevante Variablen-Werte.

        Returns:
            Fehlerbeschreibung als String, oder None wenn alle Limits eingehalten.
        """
        num_items = variables.get("num_items")
        if num_items is not None and (
            not isinstance(num_items, int)
            or isinstance(num_items, bool)
            or not 1 <= num_items <= self._MAX_NUM_ITEMS
        ):
            return f"num_items muss eine Ganzzahl zwischen 1 und {self._MAX_NUM_ITEMS} sein"

        max_length = variables.get("max_length")
        if max_length is not None and (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or not self._MIN_MAX_LENGTH <= max_length <= self._MAX_MAX_LENGTH
        ):
            return (
                f"max_length muss eine Ganzzahl zwischen {self._MIN_MAX_LENGTH} "
                f"und {self._MAX_MAX_LENGTH} sein"
            )

        documents = variables.get("documents")
        if documents is not None:
            if not isinstance(documents, list) or len(documents) > self._MAX_DOCUMENTS:
                return f"documents muss eine Liste mit max. {self._MAX_DOCUMENTS} Eintraegen sein"
            for d in documents:
                if not isinstance(d, str) or len(d) > self._MAX_DOCUMENT_LEN:
                    return (
                        f"documents-Eintrag muss ein String mit max. "
                        f"{self._MAX_DOCUMENT_LEN} Zeichen sein"
                    )

        participants = variables.get("participants")
        if participants is not None:
            if not isinstance(participants, list) or len(participants) > self._MAX_PARTICIPANTS:
                return (
                    f"participants muss eine Liste mit max. "
                    f"{self._MAX_PARTICIPANTS} Eintraegen sein"
                )
            for p in participants:
                if not isinstance(p, str) or len(p) > self._MAX_PARTICIPANT_LEN:
                    return (
                        f"participants-Eintrag muss ein String mit max. "
                        f"{self._MAX_PARTICIPANT_LEN} Zeichen sein"
                    )

        context = variables.get("context")
        if context is not None and (
            not isinstance(context, str) or len(context) > self._MAX_CONTEXT_LEN
        ):
            return f"context muss ein String mit max. {self._MAX_CONTEXT_LEN} Zeichen sein"

        focus = variables.get("focus")
        if focus is not None and (not isinstance(focus, str) or len(focus) > self._MAX_FOCUS_LEN):
            return f"focus muss ein String mit max. {self._MAX_FOCUS_LEN} Zeichen sein"

        date = variables.get("date")
        if date is not None and (not isinstance(date, str) or len(date) > self._MAX_DATE_LEN):
            return f"date muss ein String mit max. {self._MAX_DATE_LEN} Zeichen sein"

        return None

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Template-basierte Generierung."""
        return ToolDefinition(
            name="generate_from_template",
            description=(
                "Generiert ein Dokument aus einem Template (Checkliste, Protokoll, "
                "Zusammenfassung) oder rendert ein benutzerdefiniertes Template."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "description": (
                            "Template-Typ: 'checklist', 'protocol', 'summary', "
                            "oder Dateiname eines benutzerdefinierten Templates (z.B. 'status_mail.md.j2')."
                        ),
                        "minLength": 1,
                    },
                    "topic": {
                        "type": "string",
                        "description": "Thema oder Titel fuer das zu generierende Dokument.",
                        "minLength": 1,
                    },
                    "variables": {
                        "type": "object",
                        "description": (
                            "Zusaetzliche Variablen fuer das Template. "
                            "Je nach Template unterschiedlich."
                        ),
                        "maxProperties": 20,
                    },
                },
                "required": ["template", "topic"],
            },
            requires_confirmation=False,
            required_permissions=[],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        """Generiert ein Dokument je nach Template-Typ."""
        if error := self._validate_or_error(params):
            return error

        template = params["template"]
        topic = params["topic"]
        variables = params.get("variables", {})

        max_vars = 20
        if len(variables) > max_vars:
            return ToolResult(
                success=False,
                data=None,
                error=f"Zu viele Template-Variablen: {len(variables)} (max {max_vars})",
                display_text="",
            )

        if msg := self._validate_variable_limits(variables):
            return ToolResult(success=False, data=None, error=msg, display_text="")

        try:
            if template == "checklist":
                content = await self._engine.generate_checklist(
                    topic=topic,
                    context=variables.get("context", ""),
                    num_items=variables.get("num_items", 10),
                )
            elif template == "protocol":
                content = await self._engine.generate_protocol(
                    title=topic,
                    context=variables.get("context", ""),
                    participants=variables.get("participants"),
                    date=variables.get("date"),
                )
            elif template == "summary":
                documents = variables.get("documents", [topic])
                content = await self._engine.generate_summary(
                    documents=documents,
                    focus=variables.get("focus"),
                    max_length=variables.get("max_length", 500),
                )
            else:
                # Benutzerdefiniertes Template rendern
                tpl_vars = dict(variables)
                tpl_vars["title"] = topic
                content = await self._engine.render_template(template, tpl_vars)
        except Exception as exc:
            logger.error("Template-Generierung '%s' fehlgeschlagen: %s", template, exc)
            return ToolResult(
                success=False,
                data=None,
                error=f"Generierung fehlgeschlagen: {exc}",
                display_text="",
            )

        display_text = content

        return ToolResult(
            success=True,
            data={
                "content": content,
                "template": template,
                "topic": topic,
            },
            error=None,
            display_text=display_text,
        )


class ExportDocumentTool(BaseTool):
    """Exportiert ein Dokument in verschiedene Formate.

    Args:
        exporter: DocumentExporter-Instanz.
    """

    def __init__(self, exporter) -> None:
        self._exporter = exporter

    def definition(self) -> ToolDefinition:
        """Liefert die ToolDefinition fuer Dokumenten-Export."""
        return ToolDefinition(
            name="export_document",
            description=(
                "Exportiert ein Dokument (Markdown-Inhalt) als PDF, DOCX, "
                "Markdown-Datei oder reinen Text."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Der zu exportierende Markdown-Inhalt.",
                        "minLength": 1,
                    },
                    "filename": {
                        "type": "string",
                        "description": "Gewuenschter Dateiname (Erweiterung wird automatisch angepasst).",
                        "minLength": 1,
                    },
                    "format": {
                        "type": "string",
                        "description": "Zielformat fuer den Export.",
                        "enum": ["pdf", "docx", "markdown", "text"],
                    },
                    "title": {
                        "type": "string",
                        "description": "Optionaler Dokumenttitel (fuer PDF/DOCX Titelseite).",
                    },
                },
                "required": ["content", "filename"],
            },
            requires_confirmation=True,
            required_permissions=[],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        """Exportiert den Inhalt im gewuenschten Format."""
        if error := self._validate_or_error(params):
            return error

        content = params["content"]
        filename = params["filename"]
        fmt = params.get("format", "markdown")
        title = params.get("title")

        try:
            filepath = await self._exporter.export(
                content=content,
                filename=filename,
                format=fmt,
                title=title,
                metadata={"user_id": user_id} if user_id else None,
            )
        except Exception as exc:
            logger.error("Export fehlgeschlagen ('%s', %s): %s", filename, fmt, exc)
            return ToolResult(
                success=False,
                data=None,
                error=f"Export fehlgeschlagen: {exc}",
                display_text="",
            )

        display_text = (
            f"Dokument erfolgreich exportiert:\n"
            f"  Datei: {filepath}\n"
            f"  Format: {fmt}\n"
            f"  Groesse: {len(content)} Zeichen Quelldokument"
        )

        return ToolResult(
            success=True,
            data={
                "filepath": filepath,
                "format": fmt,
                "filename": filename,
            },
            error=None,
            display_text=display_text,
        )
