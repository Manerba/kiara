"""HTTP-Confirmation-Roundtrip fuer Admin-Auto-Router-Tests.

Der Auto-Router erzwingt fuer Funktionen mit ``requires_confirmation=True``
einen Two-Phase-Workflow: erste Anfrage ohne ``x-kiara-confirm-token`` gibt
HTTP 428 mit ``confirm_token``+``confirm_string`` zurueck, eine zweite Anfrage
mit den passenden Headern fuehrt die Aktion aus (siehe
``kiara/admin/router.py::_require_http_confirmation``).

Integrationstests, die nur das Endergebnis pruefen wollen, koennen
``confirmed_request`` als Drop-In fuer ``client.put/delete/patch/post`` nutzen
und erhalten transparent die Endantwort.
"""

from __future__ import annotations

import unicodedata
from typing import Any


def _ascii_fold(value: str) -> str:
    """Foldet Diakritika (NFKD strip) — HTTP-Header verlangen ASCII (RFC 7230)."""
    folded = unicodedata.normalize("NFKD", value)
    return "".join(c for c in folded if ord(c) < 128 and not unicodedata.combining(c))


def confirmed_request(client: Any, method: str, url: str, **kwargs: Any):
    """Sendet ``method`` an ``url`` und durchlaeuft ggf. den 428-Confirm-Workflow.

    Verhaelt sich identisch zu ``client.request(method, url, **kwargs)``,
    fuehrt aber bei einer ``428``-Antwort mit ``requires_confirmation`` einen
    zweiten Request mit den serverseitig erzeugten Confirm-Headern aus.
    """
    response = client.request(method, url, **kwargs)
    if response.status_code != 428:
        return response
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return response
    if not isinstance(payload, dict) or not payload.get("requires_confirmation"):
        return response
    confirmation = payload.get("confirmation") or {}
    token = confirmation.get("confirm_token")
    confirm_string = confirmation.get("confirm_string")
    if not token or not confirm_string:
        return response
    headers = dict(kwargs.pop("headers", None) or {})
    headers.setdefault("x-kiara-confirm-token", token)
    headers.setdefault("x-kiara-confirm-string", _ascii_fold(confirm_string))
    return client.request(method, url, headers=headers, **kwargs)
