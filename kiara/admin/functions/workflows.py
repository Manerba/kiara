"""Admin-Funktionen fuer Dokumenten-Workflows.

Dual-Port-Architektur: Die Original-Endpunkte in api_workflows.py laufen
auf der Main-App (Port 8501) mit get_chat_user-Auth. Diese Admin-Funktionen
ergaenzen zusaetzlichen Admin-Zugang ueber den Auto-Router (Port 8502)
und den Admin-Chat (Port 8503).

AUSGESCHLOSSEN: workflows.exports.download (FileResponse inkompatibel mit
JSON-basiertem Auto-Router).
"""

import logging
import os
from pathlib import Path

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="workflows.templates.list",
    category="workflows",
    description="Dokumenten-Templates auflisten",
    help="Listet alle verfügbaren Dokumenten-Templates mit Name, Beschreibung und Variablen auf.",
    examples=[
        "Welche Dokumenten-Templates gibt es?",
        "Zeige verfügbare Templates",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der Templates mit Details",
    raw_response=True,
    http=("GET", "/api/workflows/templates"),
)
async def workflows_templates_list(user: dict) -> AdminResult:
    from kiara.config import load_config
    from kiara.workflows.template_engine import TemplateEngine

    config = load_config()
    engine = TemplateEngine(config)
    templates = engine.list_templates()

    return AdminResult.ok(
        f"{len(templates)} Templates verfuegbar",
        data={"templates": templates},
    )


@admin_function(
    name="workflows.templates.detail",
    category="workflows",
    description="Template-Details und Inhalt anzeigen",
    help="Zeigt den Inhalt und die Variablen eines bestimmten Templates.",
    examples=[
        "Zeige Template 'checklist'",
        "Template-Details für protocol",
    ],
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name des Templates"},
        },
        "required": ["name"],
    },
    returns="Template mit Name, Beschreibung, Variablen und Inhalt",
    raw_response=True,
    http=("GET", "/api/workflows/templates/{name}"),
)
async def workflows_templates_detail(user: dict, name: str) -> AdminResult:
    from kiara.config import get_template_dir, load_config
    from kiara.workflows.template_engine import TemplateEngine

    config = load_config()
    engine = TemplateEngine(config)
    templates = engine.list_templates()

    tpl_info = None
    for tpl in templates:
        if tpl["name"] == name:
            tpl_info = tpl
            break

    if tpl_info is None:
        return AdminResult.fail(f"Template '{name}' nicht gefunden")

    tpl_dir = get_template_dir(config)
    tpl_path = Path(tpl_dir) / name

    if not tpl_path.exists():
        return AdminResult.fail(f"Template-Datei '{name}' nicht gefunden")

    content = tpl_path.read_text(encoding="utf-8")

    return AdminResult.ok(
        f"Template '{name}'",
        data={
            "name": tpl_info["name"],
            "description": tpl_info["description"],
            "variables": tpl_info["variables"],
            "content": content,
        },
    )


@admin_function(
    name="workflows.generate",
    category="workflows",
    description="Dokument aus Template generieren",
    help="Generiert ein Dokument basierend auf einem Template. "
    "Optional kann direkt in ein Format (pdf, docx, markdown, text) exportiert werden.",
    examples=[
        "Generiere eine Checkliste zum Thema Datensicherung",
        "Erstelle ein Protokoll für Meeting",
        "Generiere Dokument aus Template 'summary'",
    ],
    parameters={
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "description": "Template-Name (z.B. checklist, protocol, summary)",
            },
            "variables": {
                "type": "object",
                "description": "Template-Variablen (template-abhaengig)",
                "default": {},
            },
            "format": {"type": "string", "description": "Exportformat (pdf, docx, markdown, text)"},
        },
        "required": ["template"],
    },
    returns="Generierter Inhalt, optional mit Exportpfad",
    raw_response=True,
    http=("POST", "/api/workflows/generate"),
)
async def workflows_generate(
    user: dict,
    template: str,
    variables: dict | None = None,
    format: str | None = None,
) -> AdminResult:
    from kiara.config import load_config
    from kiara.workflows.template_engine import TemplateEngine

    if variables is None:
        variables = {}

    config = load_config()
    engine = TemplateEngine(config)

    topic = variables.get("title", variables.get("topic", template))

    try:
        if template == "checklist":
            content = await engine.generate_checklist(
                topic=topic,
                context=variables.get("context", ""),
                num_items=variables.get("num_items", 10),
            )
        elif template == "protocol":
            content = await engine.generate_protocol(
                title=topic,
                context=variables.get("context", ""),
                participants=variables.get("participants"),
                date=variables.get("date"),
            )
        elif template == "summary":
            documents = variables.get("documents", [])
            content = await engine.generate_summary(
                documents=documents if documents else [topic],
                focus=variables.get("focus"),
                max_length=variables.get("max_length", 500),
            )
        else:
            tpl_vars = dict(variables)
            tpl_vars.setdefault("title", topic)
            content = await engine.render_template(template, tpl_vars)
    except Exception as exc:
        logger.error("Generierung fehlgeschlagen ('%s'): %s", template, exc)
        return AdminResult.fail("Generierung fehlgeschlagen")

    result = {"content": content, "template": template}

    # Optional exportieren
    if format:
        from kiara.workflows.exporter import DocumentExporter

        exporter = DocumentExporter(config)
        try:
            filepath = await exporter.export(
                content=content,
                filename=f"{template}_{topic[:30]}",
                format=format,
                title=topic,
            )
            result["filepath"] = filepath
            result["export_format"] = format
        except Exception as exc:
            logger.error("Export nach Generierung fehlgeschlagen: %s", exc)
            result["export_error"] = str(exc)

    return AdminResult.ok("Dokument generiert", data=result)


@admin_function(
    name="workflows.summarize",
    category="workflows",
    description="Dokument zusammenfassen",
    help="Sucht Dokumente in der Wissensbasis und erstellt eine "
    "Zusammenfassung mit Quellenangaben.",
    examples=[
        "Fasse Dokumente zum Thema Datenschutz zusammen",
        "Erstelle eine Zusammenfassung der Sicherheitsrichtlinien",
    ],
    parameters={
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Suchquery oder Quellendokument"},
            "focus": {"type": "string", "description": "Fokus/Perspektive der Zusammenfassung"},
            "format": {
                "type": "string",
                "description": "Ausgabeformat (paragraph, bullet-points, outline)",
                "default": "paragraph",
            },
        },
        "required": ["source"],
    },
    returns="Zusammenfassung mit Quellen und Chunk-Anzahl",
    raw_response=True,
    http=("POST", "/api/workflows/summarize"),
)
async def workflows_summarize(
    user: dict,
    source: str,
    focus: str | None = None,
    format: str = "paragraph",
) -> AdminResult:
    from kiara.config import load_config
    from kiara.workflows.summarizer import DocumentSummarizer

    config = load_config()

    # Retriever initialisieren
    try:
        from kiara.retrieval.retriever import Retriever

        retriever = Retriever(config)
    except Exception as exc:
        from kiara.retrieval.index_compatibility import IndexCompatibilityError

        if isinstance(exc, IndexCompatibilityError):
            raise
        logger.error("Retriever konnte nicht initialisiert werden: %s", exc)
        return AdminResult.fail("Retriever nicht verfuegbar")

    # Chunks suchen
    try:
        results = retriever.retrieve(query=source, k=10)
    except Exception as exc:
        from kiara.retrieval.index_compatibility import IndexCompatibilityError

        if isinstance(exc, IndexCompatibilityError):
            raise
        logger.error("Retrieval fehlgeschlagen fuer '%s': %s", source, exc)
        return AdminResult.fail("Suche fehlgeschlagen")

    if not results:
        return AdminResult.fail(f"Keine Dokumente zu '{source}' gefunden")

    combined_text = "\n\n".join(r.get("text", "") for r in results if r.get("text"))

    sources = []
    seen = set()
    for r in results:
        src = r.get("metadata", {}).get("source", "")
        if src and src not in seen:
            sources.append(src)
            seen.add(src)

    llm = None
    try:
        from kiara.config import get_llm_backend
        from kiara.generation.backends.ollama import OllamaBackend

        backend = get_llm_backend(config)
        llm = OllamaBackend(url=backend["url"], model=backend["model"])
    except Exception:
        logger.warning("Ollama-Backend fuer Summarizer nicht verfuegbar")

    summarizer = DocumentSummarizer(config, llm=llm)
    try:
        summary = await summarizer.summarize(
            text=combined_text,
            focus=focus,
            output_format=format,
        )
    except Exception as exc:
        logger.error("Zusammenfassung fehlgeschlagen: %s", exc)
        return AdminResult.fail("Zusammenfassung fehlgeschlagen")

    return AdminResult.ok(
        "Zusammenfassung erstellt",
        data={
            "summary": summary,
            "sources": sources,
            "chunk_count": len(results),
        },
    )


@admin_function(
    name="workflows.export",
    category="workflows",
    description="Dokument exportieren",
    help="Exportiert Inhalt in ein bestimmtes Format (markdown, pdf, docx).",
    examples=[
        "Exportiere diesen Text als PDF",
        "Erstelle einen DOCX-Export",
    ],
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Zu exportierender Inhalt"},
            "filename": {"type": "string", "description": "Dateiname (ohne Erweiterung)"},
            "format": {
                "type": "string",
                "description": "Exportformat (markdown, pdf, docx)",
                "default": "markdown",
            },
            "title": {"type": "string", "description": "Dokument-Titel"},
        },
        "required": ["content", "filename"],
    },
    returns="Exportpfad, Dateiname und Format",
    raw_response=True,
    http=("POST", "/api/workflows/export"),
)
async def workflows_export(
    user: dict,
    content: str,
    filename: str,
    format: str = "markdown",
    title: str | None = None,
) -> AdminResult:
    from kiara.config import load_config
    from kiara.workflows.exporter import DocumentExporter

    config = load_config()
    exporter = DocumentExporter(config)

    try:
        filepath = await exporter.export(
            content=content,
            filename=filename,
            format=format,
            title=title,
            metadata={"user": user["username"]},
        )
    except Exception as exc:
        logger.error("Export fehlgeschlagen: %s", exc)
        return AdminResult.fail("Export fehlgeschlagen")

    return AdminResult.ok(
        "Dokument exportiert",
        data={
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "format": format,
        },
    )


@admin_function(
    name="workflows.exports.list",
    category="workflows",
    description="Exportierte Dateien auflisten",
    help="Listet alle exportierten Dateien mit Dateiname, Format, Größe und Erstellungsdatum auf.",
    examples=[
        "Welche Exporte gibt es?",
        "Zeige exportierte Dateien",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der exportierten Dateien",
    raw_response=True,
    http=("GET", "/api/workflows/exports"),
)
async def workflows_exports_list(user: dict) -> AdminResult:
    from kiara.config import load_config
    from kiara.workflows.exporter import DocumentExporter

    config = load_config()
    exporter = DocumentExporter(config)
    exports = exporter.list_exports()

    return AdminResult.ok(
        f"{len(exports)} exportierte Dateien",
        data={"exports": exports},
    )


@admin_function(
    name="workflows.exports.delete",
    category="workflows",
    description="Exportierte Datei löschen",
    help="Löscht eine exportierte Datei. Prüft gegen Pfad-Traversal.",
    examples=[
        "Lösche Export bericht_2026-03.pdf",
        "Exportierte Datei entfernen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Dateiname des zu loeschenden Exports"},
        },
        "required": ["filename"],
    },
    returns="Bestätigung der Löschung",
    requires_confirmation=True,
    confirm_template="export {filename} löschen",
    pk_param="filename",
    changelog_area="workflows",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/workflows/exports/{filename}"),
)
async def workflows_exports_delete(user: dict, filename: str) -> AdminResult:
    from kiara.config import load_config
    from kiara.workflows.exporter import DocumentExporter

    config = load_config()
    exporter = DocumentExporter(config)
    out_dir = Path(exporter.output_dir)

    # Pfad-Traversal verhindern
    safe_path = (out_dir / filename).resolve()
    if not str(safe_path).startswith(str(out_dir.resolve())):
        return AdminResult.fail("Ungueltiger Dateiname")

    if not safe_path.exists():
        return AdminResult.fail(f"Datei '{filename}' nicht gefunden")

    if not safe_path.is_file():
        return AdminResult.fail("Kein regulaerer Dateipfad")

    try:
        safe_path.unlink()
    except OSError as exc:
        logger.error("Datei konnte nicht geloescht werden: %s", exc)
        return AdminResult.fail("Loeschen fehlgeschlagen")

    logger.info("Export geloescht von %s: %s", user["username"], filename)
    return AdminResult.ok(
        f"Datei '{filename}' geloescht",
        data={"message": f"Datei '{filename}' geloescht."},
    )
