"""
KIara - Workflows API-Router
Dokumenten-Workflows: Templates, Zusammenfassungen, Export.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from kiara.config import get_template_dir, load_config
from kiara.shared.constants import CHARS_PER_TOKEN, DEFAULT_MAX_CONTEXT_TOKENS
from kiara.web.api_utils import lazy_singleton
from kiara.web.auth import get_chat_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

# ---------------------------------------------------------------------------
# Lazy Singletons
# ---------------------------------------------------------------------------


def _init_llm():
    """Ollama-Backend optional initialisieren (fuer Workflows/Summarizer)."""
    config = load_config()
    try:
        from kiara.config import get_llm_backend
        from kiara.generation.backends.ollama import OllamaBackend

        backend = get_llm_backend(config)
        return config, OllamaBackend(url=backend["url"], model=backend["model"])
    except Exception as exc:
        logger.warning("Ollama-Backend fuer Workflows nicht verfuegbar: %s", exc)
        return config, None


@lazy_singleton
def _get_template_engine():
    """Lazy-initialisiert die TemplateEngine."""
    from kiara.workflows.template_engine import TemplateEngine

    config, llm = _init_llm()
    return TemplateEngine(config, llm=llm)


@lazy_singleton
def _get_summarizer():
    """Lazy-initialisiert den DocumentSummarizer."""
    from kiara.workflows.summarizer import DocumentSummarizer

    config, llm = _init_llm()
    return DocumentSummarizer(config, llm=llm)


@lazy_singleton
def _get_exporter():
    """Lazy-initialisiert den DocumentExporter."""
    from kiara.workflows.exporter import DocumentExporter

    config = load_config()
    _exporter = DocumentExporter(config)
    return _exporter


def _summary_input_budget_chars(config: dict) -> int:
    """Maximale Eingabetextlaenge fuer /summarize vor dem LLM-Aufruf."""
    workflow_cfg = config.get("workflows", {})
    max_tokens = workflow_cfg.get(
        "summarize_max_input_tokens",
        workflow_cfg.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS),
    )
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = DEFAULT_MAX_CONTEXT_TOKENS
    if max_tokens <= 0:
        max_tokens = DEFAULT_MAX_CONTEXT_TOKENS
    return max(1, int(max_tokens * CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request zum Generieren eines Dokuments aus einem Template."""

    template: str = Field(..., max_length=200)
    variables: dict = {}
    format: str | None = Field(None, max_length=50)


class SummarizeRequest(BaseModel):
    """Request zum Zusammenfassen eines Dokuments aus der Wissensbasis."""

    source: str = Field(..., max_length=1000)
    focus: str | None = Field(None, max_length=500)
    format: str = Field("paragraph", max_length=50)


class ExportRequest(BaseModel):
    """Request zum Exportieren eines Dokuments in ein bestimmtes Format."""

    content: str = Field(..., max_length=500_000)
    filename: str = Field(..., max_length=255)
    format: str = Field("markdown", max_length=50)
    title: str | None = Field(None, max_length=500)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@router.get("/templates")
async def list_templates(user=Depends(get_chat_user)):
    """Verfuegbare Templates auflisten.

    Returns:
        {"templates": [{"name", "description", "variables"}]}
    """
    engine = _get_template_engine()
    templates = engine.list_templates()
    return {"templates": templates}


@router.get("/templates/{name}")
async def get_template(name: str, user=Depends(get_chat_user)):
    """Template-Inhalt und Variablen abrufen.

    Returns:
        {"name", "description", "variables", "content"}
    """
    engine = _get_template_engine()
    templates = engine.list_templates()

    # Template suchen
    tpl_info = None
    for tpl in templates:
        if tpl["name"] == name:
            tpl_info = tpl
            break

    if tpl_info is None:
        raise HTTPException(status_code=404, detail=f"Template '{name}' nicht gefunden.")

    # Template-Inhalt lesen
    config = load_config()
    config.get("workflows", {})
    tpl_dir = get_template_dir(config)
    tpl_path = (Path(tpl_dir) / name).resolve()

    if not tpl_path.is_relative_to(Path(tpl_dir).resolve()):
        raise HTTPException(status_code=400, detail="Ungueltiger Template-Name.")

    if not tpl_path.exists():
        raise HTTPException(status_code=404, detail=f"Template-Datei '{name}' nicht gefunden.")

    content = tpl_path.read_text(encoding="utf-8")

    return {
        "name": tpl_info["name"],
        "description": tpl_info["description"],
        "variables": tpl_info["variables"],
        "content": content,
    }


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@router.post("/generate")
async def generate_document(data: GenerateRequest, user=Depends(get_chat_user)):
    """Dokument aus Template generieren.

    Body: {"template": str, "variables": dict, "format": str|null}

    Wenn ``format`` gesetzt ist (pdf, docx, text, markdown), wird das
    Ergebnis exportiert und der Dateipfad zurueckgegeben.
    """
    engine = _get_template_engine()

    template = data.template
    variables = data.variables
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
            # Benutzerdefiniertes Template
            tpl_vars = dict(variables)
            tpl_vars.setdefault("title", topic)
            content = await engine.render_template(template, tpl_vars)
    except Exception as exc:
        logger.error("Generierung fehlgeschlagen ('%s'): %s", template, exc)
        raise HTTPException(
            status_code=500,
            detail="Generierung fehlgeschlagen",
        )

    result = {"content": content, "template": template}

    # Optional exportieren
    if data.format:
        exporter = _get_exporter()
        try:
            filepath = await exporter.export(
                content=content,
                filename=f"{template}_{topic[:30]}",
                format=data.format,
                title=topic,
            )
            result["filepath"] = filepath
            result["export_format"] = data.format
        except Exception as exc:
            logger.error("Export nach Generierung fehlgeschlagen: %s", exc)
            result["export_error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


@router.post("/summarize")
async def summarize_document(data: SummarizeRequest, user=Depends(get_chat_user)):
    """Dokument aus der Wissensbasis zusammenfassen.

    Body: {"source": str, "focus": str|null, "format": str}

    Sucht Chunks via Retriever und fasst sie zusammen.
    """
    config = load_config()
    summarizer = _get_summarizer()

    # Retriever initialisieren
    try:
        from kiara.retrieval.retriever import Retriever

        retriever = Retriever(config)
    except Exception as exc:
        logger.error("Retriever konnte nicht initialisiert werden: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Retriever nicht verfuegbar",
        )

    # DS-Berechtigungen (identische Logik wie /api/search und /api/chat)
    from kiara.web.api_chat import _build_datasource_filter

    allowed_ds_ids = _build_datasource_filter(user, None, False)

    if allowed_ds_ids == []:
        raise HTTPException(
            status_code=404,
            detail=f"Keine Dokumente zu '{data.source}' gefunden.",
        )

    # Chunks suchen
    try:
        results = retriever.retrieve(
            query=data.source,
            k=10,
            filter_datasource_ids=allowed_ds_ids,
        )
    except Exception as exc:
        logger.error("Retrieval fehlgeschlagen fuer '%s': %s", data.source, exc)
        raise HTTPException(
            status_code=500,
            detail="Suche fehlgeschlagen",
        )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Keine Dokumente zu '{data.source}' gefunden.",
        )

    combined_text = "\n\n".join(r.get("text", "") for r in results if r.get("text"))
    if not combined_text.strip():
        raise HTTPException(
            status_code=404,
            detail=f"Keine Textinhalte zu '{data.source}' gefunden.",
        )
    input_chars = len(combined_text)
    max_input_chars = _summary_input_budget_chars(config)
    truncated = input_chars > max_input_chars
    if truncated:
        combined_text = combined_text[:max_input_chars]

    # Quellen
    sources = []
    seen = set()
    for r in results:
        src = r.get("metadata", {}).get("source", "")
        if src and src not in seen:
            sources.append(src)
            seen.add(src)

    try:
        summary = await summarizer.summarize(
            text=combined_text,
            focus=data.focus,
            output_format=data.format,
        )
    except Exception as exc:
        logger.error("Zusammenfassung fehlgeschlagen: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Zusammenfassung fehlgeschlagen",
        )

    if truncated:
        summary = (
            f"{summary}\n\n"
            f"[Hinweis: Eingabetext wurde vor der Zusammenfassung auf "
            f"{max_input_chars} Zeichen gekuerzt.]"
        )

    return {
        "summary": summary,
        "sources": sources,
        "chunk_count": len(results),
        "input_chars": input_chars,
        "max_input_chars": max_input_chars,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.post("/export")
async def export_document(data: ExportRequest, user=Depends(get_chat_user)):
    """Dokument exportieren.

    Body: {"content": str, "filename": str, "format": str, "title": str|null}
    """
    exporter = _get_exporter()

    try:
        filepath = await exporter.export(
            content=data.content,
            filename=data.filename,
            format=data.format,
            title=data.title,
            metadata={"user": user["username"]},
        )
    except Exception as exc:
        logger.error("Export fehlgeschlagen: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Export fehlgeschlagen",
        )

    return {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "format": data.format,
    }


@router.get("/exports")
async def list_exports(user=Depends(get_chat_user)):
    """Exportierte Dateien auflisten.

    Returns:
        {"exports": [{"filename", "format", "size_bytes", "created_at"}]}
    """
    exporter = _get_exporter()
    exports = exporter.list_exports()
    return {"exports": exports}


@router.delete("/exports/{filename}")
async def delete_export(filename: str, user=Depends(get_chat_user)):
    """Exportierte Datei loeschen.

    Prueft gegen Pfad-Traversal-Angriffe.
    """
    exporter = _get_exporter()
    out_dir = Path(exporter.output_dir)

    # Pfad-Traversal verhindern
    safe_path = (out_dir / filename).resolve()
    if not safe_path.is_relative_to(out_dir.resolve()):
        raise HTTPException(status_code=400, detail="Ungueltiger Dateiname.")

    if not safe_path.exists():
        raise HTTPException(status_code=404, detail=f"Datei '{filename}' nicht gefunden.")

    if not safe_path.is_file():
        raise HTTPException(status_code=400, detail="Kein regulaerer Dateipfad.")

    try:
        safe_path.unlink()
    except OSError as exc:
        logger.error("Datei konnte nicht geloescht werden: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Loeschen fehlgeschlagen",
        )

    logger.info("Export geloescht von %s: %s", user["username"], filename)
    return {"message": f"Datei '{filename}' geloescht."}


@router.get("/exports/{filename}/download")
async def download_export(filename: str, user=Depends(get_chat_user)):
    """Exportierte Datei herunterladen.

    Prueft gegen Pfad-Traversal-Angriffe.
    """
    exporter = _get_exporter()
    out_dir = Path(exporter.output_dir)

    # Pfad-Traversal verhindern
    safe_path = (out_dir / filename).resolve()
    if not safe_path.is_relative_to(out_dir.resolve()):
        raise HTTPException(status_code=400, detail="Ungueltiger Dateiname.")

    if not safe_path.exists() or not safe_path.is_file():
        raise HTTPException(status_code=404, detail=f"Datei '{filename}' nicht gefunden.")

    # MIME-Types
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
        ".txt": "text/plain",
    }
    ext = safe_path.suffix.lower()
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(safe_path),
        filename=filename,
        media_type=media_type,
    )
