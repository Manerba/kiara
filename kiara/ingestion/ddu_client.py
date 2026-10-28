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
import mimetypes
import threading
import time
from pathlib import Path
from typing import NamedTuple

import httpx

from kiara.config import DEFAULT_TIMEOUTS
from kiara.shared.constants import (
    DDU_DEFAULT_CONCURRENCY,
    DDU_DEFAULT_OCR_LANGUAGES,
    DDU_DEFAULT_TABLE_MODE,
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
    error: str  # "" bei Erfolg, "timeout", "connection", "http_error"


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

    Returns:
        DDUResult mit content (Markdown oder None) und error-Typ.
    """
    if ocr_lang is None:
        ocr_lang = list(DDU_DEFAULT_OCR_LANGUAGES)
    if timeout is None:
        timeout = DEFAULT_TIMEOUTS["ddu_convert"]

    url = f"{ddu_url.rstrip('/')}/v1/convert/file"
    filename = Path(file_path).name

    # Korrekter MIME-Type statt pauschal application/octet-stream
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

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
                    },
                    timeout=remaining,
                )
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                logger.warning("DDU non-JSON response for %s: %s", file_path, resp.text[:200])
                return DDUResult(content=None, error="http_error")
            document = data.get("document", {})
            if "md_content" in document and document["md_content"] is not None:
                return DDUResult(content=document["md_content"], error="")
            return DDUResult(content=None, error="http_error")
        except httpx.TimeoutException:
            logger.warning("DDU timeout for %s after %ds", file_path, timeout)
            return DDUResult(content=None, error="timeout")
        except httpx.ConnectError:
            logger.warning("DDU connection refused for %s at %s", file_path, ddu_url)
            return DDUResult(content=None, error="connection")
        except httpx.HTTPStatusError as exc:
            logger.warning("DDU HTTP error for %s: %s", file_path, exc)
            return DDUResult(content=None, error="http_error")
        except Exception as exc:
            logger.warning("DDU request failed for %s: %s", file_path, exc)
            return DDUResult(content=None, error="connection")
    finally:
        semaphore.release()
