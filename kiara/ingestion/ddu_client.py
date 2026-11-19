"""HTTP-Client fuer docling-serve Remote-Backend.

Concurrency-Hinweis:
    ``_DDU_SEMAPHORE`` begrenzt parallele GPU-Requests **pro Python-Prozess**.
    KIara laeuft aktuell als Streamlit Single-Process mit ThreadPool —
    das Semaphore schuetzt die GPU korrekt.

    Bei einem spaeteren Wechsel auf Multi-Worker (z.B. Gunicorn mit N Workern)
    waere das effektive Limit ``N * concurrency_limit``. Loesungen:
    - Named Semaphore (``multiprocessing.Semaphore``) in shared memory
    - Reverse-Proxy Rate-Limit auf docling-serve Port
    - docling-serve eigene ``--max-workers`` Option (wenn verfuegbar)
"""

import logging
import threading
import time
from pathlib import Path
from typing import NamedTuple

import httpx
from httpx import RequestError
from kiara_document_loaders.content_detect import detect_file_format

from kiara.config import DEFAULT_TIMEOUTS
from kiara.shared.constants import (
    DDU_DEFAULT_CONCURRENCY,
    DDU_DEFAULT_OCR_LANGUAGES,
    DDU_DEFAULT_TABLE_MODE,
    DDU_IMAGE_MAX_PIXELS,
    DDU_IMAGE_MIN_SIDE,
)

logger = logging.getLogger(__name__)
_DDU_SEMAPHORE: threading.Semaphore | None = None
_SEMAPHORE_LOCK = threading.Lock()


def _get_semaphore(limit: int | None = None) -> threading.Semaphore:
    """Gibt das Semaphore zurueck, erstellt es lazy beim ersten Aufruf.

    Das Limit wird beim ersten Aufruf gesetzt und danach nicht mehr geaendert
    (Semaphore-Wert laesst sich nicht nachtraeglich aendern). Bei Aenderungen
    in extra_config.concurrency_limit muss der Prozess neu gestartet werden.
    """
    global _DDU_SEMAPHORE
    if _DDU_SEMAPHORE is None:
        with _SEMAPHORE_LOCK:
            if _DDU_SEMAPHORE is None:
                _DDU_SEMAPHORE = threading.Semaphore(limit or DDU_DEFAULT_CONCURRENCY)
    return _DDU_SEMAPHORE


class DDUResult(NamedTuple):
    """Ergebnis einer Remote-DDU-Konvertierung."""

    content: str | None
    error: str  # "" bei Erfolg; Infrastruktur-, Dokument- oder Bildfehler
    detail: str = ""
    page_count: int = 0
    missing_pages: tuple[int, ...] = ()
    task_id: str = ""


def is_infrastructure_error(error: str) -> bool:
    """Nur voruebergehende Backend-Probleme duerfen Recovery ausloesen."""
    return error in {"timeout", "connection", "http_error"}


def check_ddu_health(ddu_backends: list[dict], timeout: float = 120.0) -> tuple[bool, str]:
    """Prueft ob mindestens ein DDU-Backend erreichbar ist (Pre-flight Check).

    Iteriert ueber alle Backends in Prioritaetsreihenfolge. Sobald ein Backend
    antwortet, gilt der Check als bestanden.

    Args:
        ddu_backends: Liste von Backend-Dicts (aus get_ddu_backends()).
        timeout: HTTP-Timeout fuer Remote-Backends (Default 120s wegen
                 docling-proxy Container-Startup + GPU-Laden).

    Returns:
        Tuple (available, error_message).
    """
    if not ddu_backends:
        return False, "Keine DDU-Backends konfiguriert"

    errors = []
    for backend in ddu_backends:
        if backend.get("is_local") or not backend.get("url"):
            errors.append("DDU-Backend ohne URL wird ignoriert (remote-only)")
            continue
        base_url = backend["url"].rstrip("/")
        url = f"{base_url}/health"
        backend_timeout = backend.get("timeout", timeout)
        try:
            resp = httpx.get(url, timeout=backend_timeout)
            if resp.status_code < 400:
                return True, ""
            errors.append(f"{url}: HTTP {resp.status_code}")
            continue
        except httpx.ConnectError:
            errors.append(f"{url}: Verbindung verweigert")
        except httpx.TimeoutException:
            errors.append(f"{url}: Timeout nach {backend_timeout}s")
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    return False, "; ".join(errors)


def convert_document(
    file_path: str,
    ddu_url: str,
    table_mode: str = DDU_DEFAULT_TABLE_MODE,
    do_ocr: bool = True,
    ocr_lang: list[str] | None = None,
    timeout: float | None = None,
    concurrency_limit: int | None = None,
    image_min_side: int = DDU_IMAGE_MIN_SIDE,
    image_max_pixels: int = DDU_IMAGE_MAX_PIXELS,
) -> DDUResult:
    """Sendet Datei an docling-serve und gibt Ergebnis zurueck.

    Args:
        file_path: Lokaler Pfad zur Datei.
        ddu_url: Base-URL des docling-serve Servers.
        table_mode: 'fast' oder 'accurate'.
        do_ocr: OCR aktivieren (Default True).
        ocr_lang: Sprachen fuer OCR (Default: ["de", "en"]).
        timeout: Request-Timeout in Sekunden (Default: DEFAULT_TIMEOUTS["ddu_convert"]).
        concurrency_limit: Max parallele Requests (Default: 2).
        image_min_side: Mindestseitenlaenge aller Bildframes in Pixeln.
        image_max_pixels: Maximale Pixelzahl je Bildframe.

    Returns:
        DDUResult mit content (Markdown oder None) und error-Typ.
    """
    if ocr_lang is None:
        ocr_lang = list(DDU_DEFAULT_OCR_LANGUAGES)
    if timeout is None:
        timeout = DEFAULT_TIMEOUTS["ddu_convert"]

    from kiara.ingestion.ddu_images import check_image

    try:
        detected = detect_file_format(file_path)
    except Exception as exc:
        return DDUResult(None, "document_error", f"Formaterkennung fehlgeschlagen: {exc}")

    image_error, detail = check_image(
        file_path,
        min_side=image_min_side,
        max_pixels=image_max_pixels,
        detected_extension=detected.extension,
    )
    if image_error:
        logger.info("DDU Bildpruefung fuer %s: %s", file_path, detail)
        return DDUResult(None, image_error, detail)

    url = f"{ddu_url.rstrip('/')}/v1/convert/file"
    filename = Path(file_path).with_suffix(detected.extension).name

    # Korrekter MIME-Type statt pauschal application/octet-stream
    mime_type = detected.mime_type

    deadline = time.monotonic() + timeout

    semaphore = _get_semaphore(concurrency_limit)
    if not semaphore.acquire(timeout=timeout):
        logger.warning(
            "DDU-Semaphore-Timeout nach %ds fuer %s (alle Slots belegt)", timeout, file_path
        )
        return DDUResult(content=None, error="timeout")
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("DDU timeout exhausted during semaphore acquire for %s", file_path)
            return DDUResult(content=None, error="timeout")
        try:
            with open(file_path, "rb") as f:
                resp = httpx.post(
                    url,
                    # httpx serialisiert Listen in data als wiederholte Formular-Felder
                    # (ocr_lang=de&ocr_lang=en), was docling-serve erwartet
                    files=[("files", (filename, f, mime_type))],
                    data={
                        "do_ocr": str(do_ocr).lower(),
                        "table_mode": table_mode,
                        "to_formats": "md",
                        "ocr_lang": ocr_lang,
                        # Hausaenderung 01.11.2026: ohne Angabe bettet docling-serve
                        # alle Bilder als base64-Data-URI ins Markdown ein (Default
                        # "embedded"); die landeten als Hunderte Chunks im Index.
                        "image_export_mode": "placeholder",
                    },
                    timeout=remaining,
                )
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                logger.warning("DDU non-JSON response for %s: %s", file_path, resp.text[:200])
                return DDUResult(content=None, error="http_error")
            if not isinstance(data, dict):
                return DDUResult(None, "http_error", "Ungueltige DDU-Antwort: JSON-Objekt fehlt")
            status = data.get("status")
            task_id = str(data.get("task_id") or "")[:80]
            detail = ""
            if status in {"failure", "partial_success", "skipped"}:
                errors = str(data.get("errors", []))[:500]
                detail = f"DDU-Konvertierung nicht vollstaendig: {status}; {errors}"
                if task_id:
                    detail = f"Task {task_id}: {detail}"
                logger.warning("%s: %s", file_path, detail)
                if status != "partial_success" or detected.extension != ".pdf":
                    return DDUResult(None, "document_error", detail, task_id=task_id)
            document = data.get("document")
            if isinstance(document, dict) and isinstance(document.get("md_content"), str):
                content = document["md_content"]
                if detected.extension == ".pdf":
                    count = document.get("source_page_count")
                    pages = document.get("missing_text_pages")
                    if count is not None or pages is not None:
                        if (
                            type(count) is not int
                            or count < 1
                            or not isinstance(pages, list)
                            or any(type(n) is not int or not 1 <= n <= count for n in pages)
                            or len(pages) != len(set(pages))
                        ):
                            return DDUResult(None, "http_error", "Ungueltige PDF-Seitenabdeckung")
                        if status != "partial_success" or pages:
                            return DDUResult(
                                content,
                                "document_error" if detail else "",
                                detail,
                                count,
                                tuple(pages),
                                task_id,
                            )
                if detail:
                    return DDUResult(None, "document_error", detail, task_id=task_id)
                if content.strip() and "<!--" in content:
                    from kiara.ingestion.pdf_recovery import has_text

                    if not has_text(content):
                        return DDUResult(None, "document_error", "Docling lieferte nur Platzhalter")
                return DDUResult(content, "")
            if detail:
                return DDUResult(None, "document_error", detail, task_id=task_id)
            return DDUResult(content=None, error="http_error")
        except httpx.TimeoutException:
            logger.warning("DDU timeout for %s after %ds", file_path, timeout)
            return DDUResult(content=None, error="timeout")
        except httpx.ConnectError:
            logger.warning("DDU connection refused for %s at %s", file_path, ddu_url)
            return DDUResult(content=None, error="connection")
        except RequestError as exc:
            logger.warning("DDU Transportfehler fuer %s: %s", file_path, exc)
            return DDUResult(None, "connection", str(exc))
        except httpx.HTTPStatusError as exc:
            response = exc.response
            body = response.text
            logger.warning("DDU HTTP %s fuer %s: %s", response.status_code, file_path, body)
            error = "http_error"
            if response.status_code == 404:
                try:
                    task_missing = response.json().get("detail") == (
                        "Task result not found. Please wait for a completion status."
                    )
                except (ValueError, AttributeError):
                    task_missing = False
                error = "document_error" if task_missing else "request_error"
            elif 400 <= response.status_code < 500 and response.status_code not in {
                408,
                409,
                425,
                429,
            }:
                error = "request_error"
            return DDUResult(None, error, f"HTTP {response.status_code}: {body[:2000]}")
        except OSError as exc:
            logger.warning("DDU Quelldatei nicht lesbar %s: %s", file_path, exc)
            return DDUResult(None, "document_error", str(exc))
        except Exception as exc:
            logger.exception("DDU Clientfehler fuer %s", file_path)
            return DDUResult(None, "client_error", str(exc))
    finally:
        semaphore.release()
