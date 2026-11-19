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

from kiara_document_loaders.content_detect import detect_file_format

from kiara.config import DEFAULT_TIMEOUTS
from kiara.ingestion.ddu_client import is_infrastructure_error

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
    _load_xls,
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


def _load_html_markdown(path: str) -> str:
    """Konvertiert HTML-Dateien mit demselben Profil wie Mailbodies."""
    from kiara_document_loaders.html import convert_html

    return convert_html(_load_plaintext(path)).markdown


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
    path: str, ddu_backends: list[dict] | None = None, *, metadata: dict | None = None
) -> tuple[str | None, str, str]:
    """Load document using remote Docling-DDU backends with failover.

    Args:
        path: Dateipfad.
        ddu_backends: Liste von Backend-Dicts (sortiert nach Prioritaet).
        metadata: Erhaelt bei Fehlschlag den konkreten Grund fuer das Dateilog.

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
    first_remote_detail = ""
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
        from kiara.shared.constants import DDU_IMAGE_MAX_PIXELS, DDU_IMAGE_MIN_SIDE

        result = convert_document(
            path,
            ddu_url=backend["url"],
            table_mode=backend.get("table_mode", "accurate"),
            do_ocr=backend.get("do_ocr", True),
            ocr_lang=backend.get("ocr_lang", ["de", "en"]),
            timeout=backend.get("timeout", DEFAULT_TIMEOUTS["ddu_convert"]),
            concurrency_limit=backend.get("concurrency_limit"),
            image_min_side=backend.get("image_min_side", DDU_IMAGE_MIN_SIDE),
            image_max_pixels=backend.get("image_max_pixels", DDU_IMAGE_MAX_PIXELS),
        )
        if metadata is not None and result.task_id:
            metadata["docling_task_id"] = result.task_id
        if metadata is not None and metadata.get("file_type") == ".pdf":
            from kiara.ingestion.pdf_recovery import PageRecoveryError, has_text, recover_pdf

            if (
                result.error == "document_error"
                or result.missing_pages
                or (result.page_count and not has_text(result.content or ""))
            ):
                detail = result.detail or "Docling lieferte Seiten ohne verwertbaren Text"
                try:
                    text = recover_pdf(path, result, backend, metadata)
                except PageRecoveryError as exc:
                    metadata.update(
                        load_error=str(exc),
                        ddu_error_detail=detail,
                        parser="pdf-recovery",
                        parser_fallback=True,
                    )
                    return None, exc.kind, ""
                parser = "docling-remote"
                if metadata["pdf_ocr_pages"]:
                    parser = "pdf-page-ocr"
                elif metadata["pdf_native_pages"]:
                    parser = "pdf-text-fallback"
                if result.error:
                    metadata["ddu_error_detail"] = detail
                return text, result.error or "", parser

        if result.content is not None:
            return result.content, "", "docling-remote"

        if not first_remote_error:
            first_remote_error = result.error
            first_remote_detail = result.detail

        if not is_infrastructure_error(result.error):
            if metadata is not None:
                if result.error == "image_too_small":
                    metadata["skip_reason"] = result.detail
                else:
                    metadata["ddu_error_detail"] = result.detail
            # Derselbe Inhalt wird auf dem naechsten Backend nicht repariert.
            return None, "" if result.error == "image_too_small" else result.error, ""

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
    if metadata is not None and first_remote_detail:
        metadata["ddu_error_detail"] = first_remote_detail
    return None, first_remote_error or last_error, ""


# ---------------------------------------------------------------------------
# Hauptfunktion: load_file
# ---------------------------------------------------------------------------


def load_file(
    file_path: str,
    ddu_enabled: bool = False,
    ddu_backends: list[dict] | None = None,
    *,
    mail_archive: dict | None = None,
    ddu_ocr_images: bool = True,
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
    source_ext = path.suffix.lower()
    ext = source_ext
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
        "source_extension": source_ext,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
    }

    if ext == ".pkg":
        from kiara.cad.adapter import document, load_pkg

        return document(load_pkg(file_path), metadata)

    try:
        detected = detect_file_format(file_path)
    except Exception as exc:
        metadata.update(
            parser="format-detection", load_error=f"Formaterkennung fehlgeschlagen: {exc}"
        )
        return {"text": "", "metadata": metadata}
    ext = detected.extension
    metadata.update(file_type=ext, detected_format=ext, mime_type=detected.mime_type)
    if detected.encoding:
        metadata["encoding"] = detected.encoding
    aliases = {".jpeg": ".jpg", ".tif": ".tiff", ".htm": ".html"}
    if aliases.get(source_ext, source_ext) != ext:
        metadata["format_mismatch"] = True
        logger.info(
            "Dateiformat-Abweichung fuer %s: Endung %s, Inhalt %s", file_path, source_ext, ext
        )
    if ext in {".doc", ".ppt", ".ods", ".odp", ".ole", ".zip", ".rtf", ".svg", ".encrypted-office"}:
        reason = (
            "Verschluesseltes Office-Dokument; ohne Kennwort nicht lesbar"
            if ext == ".encrypted-office"
            else f"Nicht unterstuetzter Inhalt {ext.lstrip('.').upper()}"
        )
        metadata.update(
            parser="format-detection",
            skip_category="unsupported_format",
            skip_reason=f"{reason} (Endung {source_ext or 'ohne'})",
        )
        return {"text": "", "metadata": metadata}
    if ext == ".binary":
        metadata.update(
            parser="format-detection",
            load_error=f"Binaeres Dateiformat nicht erkennbar (Endung {source_ext or 'ohne'})",
        )
        return {"text": "", "metadata": metadata}
    if ext in IMAGE_FORMATS and not ddu_ocr_images:
        ddu_enabled = False

    # Hausaenderung 02.11.2026: vor dem DDU-Zweig belegen; der Standard-Pfad
    # liest ddu_error im except-Zweig auch fuer Formate ohne Docling (UnboundLocalError).
    ddu_error = ""

    # DDU-Pfad: Docling fuer layout-aware Extraktion
    if ddu_enabled and ddu_backends and ext in _DOCLING_FORMATS:
        text = None
        ddu_source = ""
        try:
            text, ddu_error, ddu_source = _load_with_docling(
                file_path, ddu_backends=ddu_backends, metadata=metadata
            )
            if ddu_source:
                metadata["parser"] = ddu_source
        except Exception as exc:
            logger.warning("Docling failed for %s: %s", file_path, exc)
            text = None
            ddu_error = "parser_error"
            metadata["ddu_error_detail"] = str(exc)

        # Fehlerinfo fuer Stats-Tracking in Pipeline
        # remote_error wird auch bei erfolgreichem Fallback gesetzt
        if ddu_error:
            metadata["ddu_error"] = ddu_error

        if metadata.get("load_error"):
            return {"text": "", "metadata": metadata}

        if metadata.get("skip_reason"):
            metadata.update(parser="image-preflight", no_content=True)
            return {"text": "", "metadata": metadata}

        # Hausaenderung 18.11.2026: Ein endgueltiger Docling-Dokumentfehler sagt
        # nichts ueber pypdf/python-docx aus (36 Plaene in DS 1 las pypdf, Docling
        # nicht). Der Standard-Loader unten versucht es; scheitert auch er, nennt
        # das Dateilog beide Ursachen. Nur Bilder haben keinen Standard-Loader.
        # Client-/Anfragefehler (parser_error, request_error) bleiben Fehler ohne Fallback.
        if (
            ddu_error
            and not is_infrastructure_error(ddu_error)
            and (ddu_error != "document_error" or ext in _IMAGE_ONLY_FORMATS)
        ):
            metadata.setdefault("parser", "docling-remote")
            return {"text": "", "metadata": metadata}

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
    parser_name = "standard"
    try:
        if ext in (".html", ".htm"):
            from kiara_document_loaders.html import convert_html

            conversion = convert_html(_load_plaintext(file_path))
            text = conversion.markdown
            metadata.update(conversion.metadata())
            parser_name = metadata["parser"]
        else:
            text = loader(file_path)
        if source_ext == ".txt" and ext == ".txt" and text is not None:
            from kiara.config import get_mail_archive_config, load_config
            from kiara.ingestion.mail_archive import read_mail_metadata

            archive = (
                mail_archive if mail_archive is not None else get_mail_archive_config(load_config())
            )
            metadata.update(read_mail_metadata(file_path, text, archive))
            parser_name = metadata.get("parser", parser_name)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", file_path, exc)
        metadata.update(
            parser=parser_name,
            load_error=f"{ext.lstrip('.').upper()}-Loader: {type(exc).__name__}: {exc}",
        )
        if ddu_error:
            metadata["ddu_error"] = ddu_error
            metadata["parser_fallback"] = True
        return {"text": "", "metadata": metadata}

    if text is None:
        # Loader konnte die Datei nicht verarbeiten (z.B. fehlende Abhaengigkeit)
        return None

    if not text.strip():
        # Loader lief erfolgreich, aber kein extrahierbarer Inhalt
        # (z.B. VBA-only .xlsm, leere Sheets). Kein Fehler, sondern Skip.
        metadata["parser"] = parser_name
        metadata["no_content"] = True
        return {"text": "", "metadata": metadata}

    text = _sanitize_text(text, file_path, parser_name)
    if text is None:
        return None

    metadata["parser"] = parser_name
    return {"text": text, "metadata": metadata}


# ---------------------------------------------------------------------------
# Loader-Registry
# ---------------------------------------------------------------------------

_LOADERS = {
    ".docx": _load_docx,
    ".pdf": _load_pdf,
    ".xlsx": _load_xlsx,
    ".xls": _load_xls,
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
    ".html": _load_html_markdown,
    ".htm": _load_html_markdown,
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
