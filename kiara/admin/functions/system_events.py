"""AdminFunctions fuer das System-Log-Feature (system.events.*).

Alle Endpoints sind JSON-only (`raw_response=True`) und laufen durch
denselben SystemEventRepository. CSV-Export lebt in einem separaten
FastAPI-Router (`kiara/web/api_admin_system_events.py`), weil der
Auto-Router nur JSON liefern kann.

Self-Audit: `system.events.list` schreibt bei jedem Aufruf einen
`SYSLOG_VIEW`-Event auf INFO-Level (nicht DEBUG — waere vom Logger
verworfen, Audit-Luecke). Das Frontend blendet `event_type='SYSLOG_VIEW'`
in der Default-Sicht aus; Admins koennen den Filter deaktivieren.
"""

from __future__ import annotations

import hashlib
import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.repositories.system_events import (
    SystemEventRepository,
    _coerce_system_event_filters,
)
from kiara.web.rate_limit_syslog import check_rate_limit

security_log = logging.getLogger("kiara.security")

_FILTER_SCHEMA_KEYS = {
    "since": {"type": "string", "description": "ISO-Zeitstempel (ab)"},
    "until": {"type": "string", "description": "ISO-Zeitstempel (bis)"},
    "min_level": {
        "type": "string",
        "description": "Mindestlevel (INFO, WARNING, ERROR, CRITICAL)",
    },
    "event_types": {
        "type": "string",
        "description": "Komma-separierte Event-Typen (Include-Filter)",
    },
    "exclude_event_types": {
        "type": "string",
        "description": "Komma-separierte Event-Typen (Exclude-Filter)",
    },
    "reason": {"type": "string", "description": "Reason-Kategorie"},
    "search": {"type": "string", "description": "Volltextsuche in message"},
}


def _user_label(user: dict) -> str:
    """Robuste User-Label-Ableitung fuer HTTP- und Admin-Chat-Aufrufer."""
    if not isinstance(user, dict):
        return "unknown"
    return str(user.get("username") or user.get("id") or "unknown")


def _filter_hash(filters: dict) -> str:
    """Stabiler Hash der aktiven Filter fuer Self-Audit-Korrelation."""
    canonical = repr(sorted((k, repr(v)) for k, v in filters.items()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _rate_limit_or_fail(user: dict) -> AdminResult | None:
    """Rate-Limit pruefen. None = Pass, AdminResult = 429-Response."""
    user_key = _user_label(user)
    allowed, retry_after = check_rate_limit(user_key)
    if allowed:
        return None
    # raw_response=True: Router gibt nur result.data aus — Fehlerinfos
    # muessen daher im data-Dict stehen (success/error/message/retry_after).
    return AdminResult.fail(
        "Rate limit",
        data={
            "success": False,
            "error": "Rate limit",
            "message": "Rate limit: zu viele Anfragen, bitte kurz warten.",
            "retry_after": round(retry_after or 0.0, 2),
        },
        http_status=429,
    )


@admin_function(
    name="system.events.distinct_reasons",
    category="system",
    description="Liefert die Liste aller vorkommenden reason-Werte",
    help="Dient dem Filter-Dropdown im System-Log-Sub-Tab. Nur reason IS NOT NULL.",
    examples=[
        "Welche reason-Kategorien gab es in den letzten 7 Tagen?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "since": {"type": "string", "description": "ISO-Zeitstempel (ab)"},
        },
        "required": [],
    },
    returns="Liste von reason-Strings",
    raw_response=True,
    http=("GET", "/api/admin/system-events/distinct-reasons"),
)
async def system_events_distinct_reasons(user: dict, since: str | None = None) -> AdminResult:
    rate = _rate_limit_or_fail(user)
    if rate is not None:
        return rate
    repo = SystemEventRepository()
    reasons = repo.distinct_reasons(since=since)
    return AdminResult.ok("OK", data={"reasons": reasons})


@admin_function(
    name="system.events.distinct_types",
    category="system",
    description="Liefert die Liste aller vorkommenden event_type-Werte",
    help="Dient dem Filter-Dropdown im System-Log-Sub-Tab. Nur event_type IS NOT NULL.",
    examples=[
        "Welche event_type-Kategorien gab es?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "since": {"type": "string", "description": "ISO-Zeitstempel (ab)"},
        },
        "required": [],
    },
    returns="Liste von event_type-Strings",
    raw_response=True,
    http=("GET", "/api/admin/system-events/distinct-types"),
)
async def system_events_distinct_types(user: dict, since: str | None = None) -> AdminResult:
    rate = _rate_limit_or_fail(user)
    if rate is not None:
        return rate
    repo = SystemEventRepository()
    types = repo.distinct_event_types(since=since)
    return AdminResult.ok("OK", data={"event_types": types})


@admin_function(
    name="system.events.list",
    category="system",
    description="Gefilterte Liste strukturierter Security-Events",
    help=(
        "Liefert die Events aus system_events mit Pagination und Filter "
        "(Zeitbereich, Level-Schwelle, Event-Type-Include/Exclude, reason, search). "
        "Jeder Aufruf erzeugt einen SYSLOG_VIEW-Self-Audit-Eintrag."
    ),
    examples=[
        "Zeige die letzten 24h Login-Fehler",
        "Welche WARNING-Events gab es heute?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Seite (1-basiert)"},
            "page_size": {"type": "integer", "description": "Eintraege pro Seite (max 500)"},
            "sort_field": {
                "type": "string",
                "description": "Sortierfeld (ts|level|event_type|reason|user_name|ip)",
            },
            "sort_dir": {"type": "string", "description": "asc oder desc"},
            **_FILTER_SCHEMA_KEYS,
        },
        "required": [],
    },
    returns="{data: [...], pagination: {...}}",
    raw_response=True,
    http=("GET", "/api/admin/system-events"),
)
async def system_events_list(
    user: dict,
    page: int | None = None,
    page_size: int | None = None,
    sort_field: str = "ts",
    sort_dir: str = "desc",
    **raw_filters,
) -> AdminResult:
    rate = _rate_limit_or_fail(user)
    if rate is not None:
        return rate

    # Pagination-Sanitizing vor Offset-Berechnung
    try:
        effective_page = int(page) if page is not None else 1
    except (TypeError, ValueError):
        effective_page = 1
    try:
        effective_page_size = int(page_size) if page_size is not None else 100
    except (TypeError, ValueError):
        effective_page_size = 100
    effective_page = max(1, effective_page)
    effective_page_size = max(1, min(effective_page_size, 500))

    filters = _coerce_system_event_filters(raw_filters)
    repo = SystemEventRepository()
    offset = (effective_page - 1) * effective_page_size
    rows = repo.list(
        limit=effective_page_size,
        offset=offset,
        sort_field=sort_field,
        sort_dir=sort_dir,
        limit_cap=500,
        **filters,
    )
    total = repo.count(**filters)
    total_pages = (total + effective_page_size - 1) // effective_page_size if total > 0 else 0

    # Self-Audit: INFO-Level (DEBUG waere vom Logger verworfen).
    security_log.info(
        "SYSLOG_VIEW user=%s filter_hash=%s page=%s page_size=%s",
        _user_label(user),
        _filter_hash(filters),
        effective_page,
        effective_page_size,
    )

    return AdminResult.ok(
        "OK",
        data={
            "data": _serialize_rows(rows),
            "pagination": {
                "page": effective_page,
                "page_size": effective_page_size,
                "total_items": int(total),
                "total_pages": int(total_pages),
            },
        },
    )


@admin_function(
    name="system.events.get",
    category="system",
    description="Einzelner System-Event",
    help="Gibt einen einzelnen Event aus der system_events-Tabelle zurueck.",
    examples=["Zeige Event 42"],
    parameters={
        "type": "object",
        "properties": {
            "event_id": {"type": "integer", "description": "Event-ID"},
        },
        "required": ["event_id"],
    },
    returns="Event-Dict oder 404",
    raw_response=True,
    http=("GET", "/api/admin/system-events/{event_id}"),
)
async def system_events_get(user: dict, event_id: int) -> AdminResult:
    rate = _rate_limit_or_fail(user)
    if rate is not None:
        return rate
    repo = SystemEventRepository()
    row = repo.get_by_id(int(event_id))
    if row is None:
        return AdminResult.fail("Event nicht gefunden", http_status=404)
    return AdminResult.ok("OK", data=_serialize_row(row))


def _serialize_rows(rows: list[dict]) -> list[dict]:
    return [_serialize_row(r) for r in rows]


def _serialize_row(row: dict) -> dict:
    """JSON-serialisierbare Projektion (datetime → ISO, JSONB bleibt dict)."""
    out = dict(row)
    ts = out.get("ts")
    if ts is not None and hasattr(ts, "isoformat"):
        out["ts"] = ts.isoformat()
    return out
