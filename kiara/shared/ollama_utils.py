"""Ollama-Server-Kommunikation (infrastruktur-unabhaengig).

Prueft Erreichbarkeit und verfuegbare Modelle eines Ollama-Servers.
Wird von Health-Checks, Admin-Dashboard und KI-Backend-Verwaltung genutzt.

Beziehungen:
    - kiara.health.checks_services nutzt check_ollama_models() fuer Service-Checks
    - kiara.web.api_ki nutzt check_ollama_models() fuer Backend-Verwaltung
    - kiara.web.api_dashboard nutzt check_ollama_models() fuer Status-Anzeige
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _parse_ollama_tags_response(
    data: dict,
) -> tuple[bool, list[str], list[dict], str]:
    """Parst die Ollama /api/tags Response in ein einheitliches Ergebnis-Tuple.

    Gemeinsame Logik fuer die async- und sync-Varianten von check_ollama_models().

    Returns:
        Tuple ``(erreichbar, modell_namen, rohe_modell_dicts, fehler_text)``.
    """
    raw_models = data.get("models")
    if not isinstance(raw_models, list):
        msg = f"Ollama /api/tags: 'models' ist {type(raw_models).__name__} statt list"
        logger.warning(msg)
        return False, [], [], msg
    names = [m.get("name", "") for m in raw_models if isinstance(m, dict) and m.get("name")]
    return True, names, raw_models, ""


def _sanitize_exc_message(exc: Exception, url: str) -> str:
    """Erzeugt einen Fehlertext ohne URL-Credentials.

    ``requests.exceptions.InvalidURL`` (und einige andere Exceptions) schreiben
    die komplette URL in ``str(exc)``. Enthaelt die URL ``user:pw@host`` — z.B.
    bei einem Reverse-Proxy mit Basic-Auth vor Ollama —, wuerden Credentials
    sonst in Logs und Health-Reports landen. Bei vorhandenem userinfo geben
    wir daher einen sicheren Platzhalter statt der Original-Message zurueck.
    """
    if isinstance(url, str) and "@" in url:
        after_scheme = url.split("://", 1)[-1]
        if "@" in after_scheme.split("/", 1)[0]:
            return f"{exc.__class__.__name__}: connection failed (URL with credentials redacted)"
    return str(exc)


async def check_ollama_models(
    url: str,
    timeout: float = 5.0,
) -> tuple[bool, list[str], list[dict], str]:
    """Prueft Ollama-Erreichbarkeit via ``/api/tags`` und liefert verfuegbare Modelle.

    Zentrale async-Implementierung — wird von api_ki, api_dashboard,
    checks_services und ueberall genutzt, wo der Ollama-Status geprueft wird.

    Args:
        url:     Basis-URL des Ollama-Servers (z.B. ``http://localhost:11434``).
        timeout: HTTP-Timeout in Sekunden.

    Returns:
        Tuple ``(erreichbar, modell_namen, rohe_modell_dicts, fehler_text)``.
        ``rohe_modell_dicts`` enthaelt die vollstaendigen Modell-Objekte aus
        der Ollama-API (mit ``details``, ``families`` etc.) — noetig fuer
        Kategorie-Klassifizierung in api_ki.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            return _parse_ollama_tags_response(resp.json())
    except Exception as exc:
        return False, [], [], _sanitize_exc_message(exc, url)


def check_ollama_models_sync(
    url: str,
    timeout: float = 5.0,
) -> tuple[bool, list[str], list[dict], str]:
    """Synchrone Variante von :func:`check_ollama_models`.

    Nutzt ``requests`` statt ``httpx`` — fuer Streamlit und andere sync-Kontexte.

    Returns:
        Tuple ``(erreichbar, modell_namen, rohe_modell_dicts, fehler_text)``.
    """
    import requests as _requests

    try:
        resp = _requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return _parse_ollama_tags_response(resp.json())
    except Exception as exc:
        return False, [], [], _sanitize_exc_message(exc, url)
