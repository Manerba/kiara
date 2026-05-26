"""Separater FastAPI-Router fuer CSV-Export von system_events.

Der Auto-Router in `kiara/admin/router.py` kann nur JSON-Responses liefern;
CSV-Download benoetigt daher einen eigenen Endpoint. Die Filter-Logik und
das SQL-Statement werden ueber `SystemEventRepository.list(...)` mit dem
gemeinsamen `_coerce_system_event_filters`-Helper aus Shared-Repository
geteilt — keine zweite SQL-Query.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any, Iterable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from kiara.repositories.system_events import (
    SystemEventRepository,
    _coerce_system_event_filters,
)
from kiara.web.auth import get_admin_user
from kiara.web.rate_limit_syslog import check_rate_limit

router = APIRouter()
logger = logging.getLogger(__name__)


_CSV_COLUMNS = (
    "id",
    "ts",
    "level",
    "source",
    "event_type",
    "reason",
    "user_name",
    "ip",
    "message",
    "details",
)
_CSV_LIMIT = 10_000


def _neutralize_csv_value(value: Any) -> str:
    """RFC-4180-konforme Repraesentation + CSV-Injection-Schutz.

    Zellen, die nach Trim mit `=`, `+`, `-` oder `@` starten, werden mit
    fuehrendem Apostroph neutralisiert — sonst koennte Excel/LibreOffice
    die Zelle als Formel interpretieren.
    """
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = str(value)
    stripped = text.lstrip()
    if stripped and stripped[0] in ("=", "+", "-", "@"):
        text = "'" + text
    return text


def _row_to_csv_values(row: dict) -> list[str]:
    return [_neutralize_csv_value(row.get(col)) for col in _CSV_COLUMNS]


def _render_csv_rows(rows: Iterable[dict]) -> str:
    """Erzeugt die komplette CSV vor dem HTTP-Response.

    Der Export ist auf 10.000 Zeilen begrenzt. Vorab-Rendering verhindert, dass
    Generator-Fehler erst nach bereits gesendetem HTTP 200 auftreten.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    writer.writerow(list(_CSV_COLUMNS))
    for row in rows:
        writer.writerow(_row_to_csv_values(row))
    return buf.getvalue()


@router.get("/api/admin/system-events/export.csv")
async def export_system_events_csv(
    request: Request,
    user: dict = Depends(get_admin_user),
):
    user_key = str(user.get("username") or user.get("id") or "unknown")
    allowed, retry_after = check_rate_limit(user_key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit: zu viele Anfragen, bitte kurz warten.",
            headers={"Retry-After": str(int(retry_after or 1))},
        )

    query_params: dict[str, Any] = {}
    for key, value in request.query_params.multi_items():
        if key in query_params:
            existing = query_params[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                query_params[key] = [existing, value]
        else:
            query_params[key] = value

    filters = _coerce_system_event_filters(query_params)
    repo = SystemEventRepository()
    # Shared-Repository: identischer Code-Pfad wie system.events.list — nur
    # limit/limit_cap werden auf 10.000 erhoeht. Keine zweite SQL-Query.
    try:
        rows = repo.list(
            limit=_CSV_LIMIT,
            offset=0,
            limit_cap=_CSV_LIMIT,
            **filters,
        )
        csv_body = _render_csv_rows(rows)
    except Exception as exc:
        logger.exception("System-Events CSV-Export fehlgeschlagen")
        raise HTTPException(
            status_code=500,
            detail="CSV-Export fehlgeschlagen.",
        ) from exc

    response = Response(
        content=csv_body,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="system-events.csv"'},
    )
    return response
