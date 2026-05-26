"""Document loaders for various file formats.

Haupteinstiegspunkt: load_file() und describe_image().
Die Einzel-Loader sind in Submodule aufgeteilt:
- loaders/general.py: Docx, PDF, XLSX, ODT, PPTX, CSV, JSON, Plaintext, Code
- loaders/cad.py: DXF, STEP, IGES
- loaders/vision.py: VLM-Bildbeschreibung
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kiara.config import DEFAULT_TIMEOUTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-Exports aus Submodulen (Rueckwaertskompatibilitaet fuer Tests + Pipeline)
# ---------------------------------------------------------------------------

from kiara_document_loaders.cad import (  # noqa: F401
    _BOM_TAGS,
    _TITLE_BLOCK_TAGS,
    _dxf_clean_mtext,
    _dxf_extract_bom,
    _dxf_extract_dimensions,
    _dxf_extract_leaders,
    _dxf_extract_texts_by_layer,
    _dxf_extract_title_block,
    _load_dxf,
    _load_iges,
    _load_step,
)
from kiara_document_loaders.general import (  # noqa: F401
    _CODE_LANGUAGES,
    _load_code,
    _load_csv_file,
    _load_docx,
    _load_json_file,
    _load_odt,
    _load_pdf,
    _load_plaintext,
    _load_pptx,
    _load_xlsx,
)
from kiara_document_loaders.vision import (  # noqa: F401
    _OLLAMA_NATIVE,
    IMAGE_FORMATS,
    MAX_VLM_IMAGE_SIZE,
    _to_png_bytes,
    _vlm_prompt_de,
    describe_image,
)

# Backward-Compat-Alias
_IMAGE_ONLY_FORMATS = IMAGE_FORMATS


# Docling-faehige Formate
_DOCLING_FORMATS = {".pdf", ".png", ".tiff", ".tif", ".jpg", ".jpeg", ".docx", ".pptx", ".xlsx"}


def _sanitize_text(text: str, source: str, parser: str) -> str | None:
    """Prüft Text auf Binär-Inhalt und entfernt vereinzelte Null-Byte-Artefakte.

    Echte Binärdateien haben einen hohen Null-Byte- oder U+FFFD-Anteil (>5%).
    Parser wie pypdf/docling erzeugen vereinzelt Null-Bytes als Encoding-Artefakte,
    die hier bereinigt werden. _load_plaintext nutzt ``errors="replace"`` auf
    nicht dekodierbaren Bytes — daraus resultierende U+FFFD-Fluten werden
    ebenfalls als Binär-Inhalt erkannt.

    Returns:
        Bereinigter Text, oder None bei echtem Binär-Inhalt.
    """
    if "\x00" in text:
        null_ratio = text.count("\x00") / len(text) if text else 0
        if null_ratio > 0.05:
            logger.warning(
                "Binary content detected in %s (%s, %.1f%% null bytes), skipping",
                source,
                parser,
                null_ratio * 100,
            )
            return None
        logger.debug("Sanitized %d null bytes from %s (%s)", text.count("\x00"), source, parser)
        text = text.replace("\x00", "")

    if "�" in text and text:
        replacement_ratio = text.count("�") / len(text)
        if replacement_ratio > 0.05:
            logger.warning(
                "Binary content detected in %s (%s, %.1f%% U+FFFD), skipping",
                source,
                parser,
                replacement_ratio * 100,
            )
            return None

    return text


def _load_with_docling(
    path: str, ddu_backends: list[dict] | None = None
) -> tuple[str | None, str, str]:
    """Load document using remote Docling-DDU backends with failover.

    Args:
        path: Dateipfad.
        ddu_backends: Liste von Backend-Dicts (sortiert nach Prioritaet).

    Returns:
        Tuple (markdown_text, remote_error, source).
        remote_error: "" bei Erfolg ohne Remote-Fehler,
                      "timeout"/"connection"/"http_error" bei Remote-Fehler.
                      Wird auch bei erfolgreichem Fallback beibehalten,
                      damit _process_single_file() Remote-Fehler tracken kann.
        source: "docling-remote" bei Remote-Erfolg, "" bei totalem Fehlschlag.
    """
    if not ddu_backends:
        return None, "", ""

    first_remote_error = ""
    last_error = ""

    for backend in ddu_backends:
        if backend.get("is_local") or not backend.get("url"):
            logger.info(
                "DDU-Backend ohne URL fuer %s ignoriert: KIara nutzt Docling remote-only",
                path,
            )
            last_error = "missing_url"
            continue

        from kiara.ingestion.ddu_client import convert_document

        result = convert_document(
            path,
            ddu_url=backend["url"],
            table_mode=backend.get("table_mode", "accurate"),
            do_ocr=backend.get("do_ocr", True),
            ocr_lang=backend.get("ocr_lang", ["de", "en"]),
            timeout=backend.get("timeout", DEFAULT_TIMEOUTS["ddu_convert"]),
            concurrency_limit=backend.get("concurrency_limit"),
        )
        if result.content is not None:
            return result.content, "", "docling-remote"

        if not first_remote_error:
            first_remote_error = result.error

        # Timeout: Kein Remote-Failover. Der Standard-Loader darf unten
        # trotzdem versuchen, verwertbaren Text aus der Datei zu holen.
        if result.error == "timeout":
            logger.warning(
                "DDU timeout fuer %s auf %s — kein DDU-Failover, Standard-Loader folgt",
                path,
                backend["url"],
            )
            return None, "timeout", ""

        # Connection/HTTP-Error: Naechstes Backend versuchen
        logger.info(
            "DDU remote fehlgeschlagen fuer %s (%s auf %s), versuche naechstes Backend",
            path,
            result.error,
            backend["url"],
        )
        last_error = result.error
        continue

    # Alle Backends erschoepft
    return None, first_remote_error or last_error, ""


# ---------------------------------------------------------------------------
# Hauptfunktion: load_file
# ---------------------------------------------------------------------------


def load_file(
    file_path: str, ddu_enabled: bool = False, ddu_backends: list[dict] | None = None
) -> dict[str, Any] | None:
    """Load a file and return its text content with metadata.

    Args:
        file_path: Path to the file to load.
        ddu_enabled: If True, use Docling for layout-aware extraction (DDU).
        ddu_backends: Liste von DDU-Backend-Dicts (sortiert nach Prioritaet).

    Returns:
        Dict with 'text' and 'metadata' keys, or None if loading fails.
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    try:
        stat = path.stat()
    except OSError as exc:
        logger.warning("Datei nicht zugreifbar: %s — %s", file_path, exc)
        return None

    metadata = {
        "source": str(path),
        "directory": str(path.parent),
        "filename": path.name,
        "file_type": ext,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
    }

    # DDU-Pfad: Docling fuer layout-aware Extraktion
    if ddu_enabled and ddu_backends and ext in _DOCLING_FORMATS:
        text = None
        ddu_error = ""
        ddu_source = ""
        try:
            text, ddu_error, ddu_source = _load_with_docling(file_path, ddu_backends=ddu_backends)
            if ddu_source:
                metadata["parser"] = ddu_source
        except Exception as exc:
            logger.warning("Docling failed for %s, falling back: %s", file_path, exc)
            text = None

        # Fehlerinfo fuer Stats-Tracking in Pipeline
        # remote_error wird auch bei erfolgreichem Fallback gesetzt
        if ddu_error:
            metadata["ddu_error"] = ddu_error

        if text and text.strip():
            text = _sanitize_text(text, file_path, metadata.get("parser", "docling"))
            if text is None:
                return None
            return {"text": text, "metadata": metadata}

        # Fallback auf Standard-Loader, auch bei DDU-Infrafehlern. So fuehrt ein
        # Timeout nicht zu silent no_content, wenn der klassische Loader reicht.
        metadata["parser_fallback"] = True
        # Reine Bilddateien haben keinen Standard-Loader — aufgeben
        # Dict mit no_content statt None, damit Pipeline "skipped" statt "error" trackt
        if ext in _IMAGE_ONLY_FORMATS:
            metadata["no_content"] = True
            return {"text": "", "metadata": metadata}

    # Bilddateien haben keinen sinnvollen Standard-Loader — _load_plaintext wuerde
    # Binaer-Bytes via errors="replace" zu U+FFFD-Muell dekodieren. Ohne DDU/VLM
    # gibt es keine Text-Extraktion, also als no_content behandeln.
    if ext in IMAGE_FORMATS:
        metadata["no_content"] = True
        metadata["parser"] = "standard"
        return {"text": "", "metadata": metadata}

    # Standard-Pfad (wie bisher)
    loader = _LOADERS.get(ext, _load_plaintext)
    try:
        text = loader(file_path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", file_path, exc)
        if ddu_enabled and ddu_error in ("timeout", "connection", "http_error"):
            metadata["ddu_error"] = ddu_error
            metadata["parser_fallback"] = True
            return {"text": "", "metadata": metadata}
        return None

    if text is None:
        # Loader konnte die Datei nicht verarbeiten (z.B. fehlende Abhaengigkeit)
        return None

    if not text.strip():
        # Loader lief erfolgreich, aber kein extrahierbarer Inhalt
        # (z.B. VBA-only .xlsm, leere Sheets). Kein Fehler, sondern Skip.
        metadata["parser"] = "standard"
        metadata["no_content"] = True
        return {"text": "", "metadata": metadata}

    text = _sanitize_text(text, file_path, "standard")
    if text is None:
        return None

    metadata["parser"] = "standard"
    return {"text": text, "metadata": metadata}


# ---------------------------------------------------------------------------
# Loader-Registry
# ---------------------------------------------------------------------------

_LOADERS = {
    ".docx": _load_docx,
    ".pdf": _load_pdf,
    ".xlsx": _load_xlsx,
    ".xlsm": _load_xlsx,
    ".odt": _load_odt,
    ".pptx": _load_pptx,
    ".csv": _load_csv_file,
    ".json": _load_json_file,
    # Code files
    ".cs": _load_code,
    ".vb": _load_code,
    ".ps1": _load_code,
    ".py": _load_code,
    ".pl": _load_code,
    ".js": _load_code,
    ".bat": _load_code,
    ".sh": _load_code,
    ".css": _load_code,
    ".html": _load_code,
    ".sln": _load_code,
    # Plain text (default handler)
    ".txt": _load_plaintext,
    ".md": _load_plaintext,
    ".xml": _load_plaintext,
    ".config": _load_plaintext,
    ".ini": _load_plaintext,
    # CAD files
    ".dxf": _load_dxf,
    ".step": _load_step,
    ".stp": _load_step,
    ".iges": _load_iges,
    ".igs": _load_iges,
}
