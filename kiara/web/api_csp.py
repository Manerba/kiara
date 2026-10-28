"""
KIara - CSP-Report API

Empfaengt CSP-Violation-Reports, die der Browser bei Strict-CSP-Verletzungen
sendet. Wird als Report-URI in der CSP-Header-Konfiguration hinterlegt und
vom Strict-CSP-Report-Only-Rollout (V30 Welle 4) genutzt.

Der Endpoint:
- benoetigt keine Authentifizierung (Browser sendet ohne Credentials)
- ist CSRF-exempt (kein XRW-/Origin-Token erwartbar bei Browser-Auto-POST)
- loggt Reports ueber den Sicherheits-Logger (strukturiert, tail-friendly)
- liefert 204 No Content zurueck (rate-limit-freundlich)

Spec: https://www.w3.org/TR/CSP3/#reporting
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import Response

logger = logging.getLogger("kiara.security.csp")

router = APIRouter()


@router.post("/api/admin/csp-report")
@router.post("/api/csp-report")
async def receive_csp_report(request: Request) -> Response:
    """Empfaengt CSP-Violation-Reports des Browsers und loggt sie.

    Akzeptiert JSON (application/csp-report oder application/reports+json).
    Antwortet mit 204, unabhaengig vom Payload-Format — wir wollen den
    Browser nicht am erneuten Senden hindern oder irrefuehren.
    """
    try:
        raw = await request.body()
        text = raw.decode("utf-8", errors="replace") if raw else ""
        if text:
            # Einzelzeilig loggen (eine Zeile pro Violation) — damit grep/tail
            # auf /var/log/kiara/*.log ergonomisch bleibt. Keine Parse-Kosten,
            # die Browser-Payloads sind immer klein (<2 KiB).
            logger.info("csp-report %s", text.replace("\n", " "))
    except Exception:
        # Nie 4xx/5xx zurueckgeben — der Browser sollte den Report nicht
        # als Fehlschlag interpretieren.
        logger.exception("csp-report handling failed")
    return Response(status_code=204)
