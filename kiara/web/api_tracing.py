"""Trace Explorer API — Endpunkte fuer Trace-Viewer, Filter, Aggregation,
Vergleich und Export.

Sprint 17.6 — Erweiterter Trace Explorer.
"""

import csv
import io
import re
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from kiara.config import load_config
from kiara.shared.db import db_query
from kiara.web.auth import get_admin_user
from kiara.web.config_changelog import log_config_change

router = APIRouter()

# ---------------------------------------------------------------------------
# SQL-Templates
# ---------------------------------------------------------------------------

SPAN_QUERY_SQL = """
SELECT span_id, parent_span_id, name, start_time_ns, end_time_ns,
       duration_ms, attributes, status
FROM traces
WHERE trace_id = %s
ORDER BY start_time_ns;
"""

TRACE_SCOPE_SQL = """
SELECT
    rt.trace_id,
    rt.root_span,
    rt.start_time_ns,
    rt.start_time,
    rt.duration_ms,
    rt.status,
    rt.attributes
FROM (
    SELECT
        t.trace_id,
        t.name AS root_span,
        t.start_time_ns,
        TO_TIMESTAMP(t.start_time_ns / 1000000000.0) AS start_time,
        t.duration_ms,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM traces te
                WHERE te.trace_id = t.trace_id
                  AND te.status = 'ERROR'
            ) THEN 'ERROR'
            WHEN EXISTS (
                SELECT 1
                FROM traces tu
                WHERE tu.trace_id = t.trace_id
                  AND tu.status = 'UNSET'
            ) THEN 'UNSET'
            ELSE COALESCE(t.status, 'OK')
        END AS status,
        t.attributes
    FROM traces t
    WHERE t.parent_span_id IS NULL
) AS rt
WHERE (%s IS NULL OR rt.root_span = %s)
  AND (%s IS NULL OR rt.duration_ms >= %s)
  AND (%s IS NULL OR rt.status = %s)
  AND (%s IS NULL OR rt.start_time_ns >= (EXTRACT(EPOCH FROM %s::TIMESTAMPTZ) * 1000000000)::BIGINT)
  AND (%s IS NULL OR rt.start_time_ns <= (EXTRACT(EPOCH FROM %s::TIMESTAMPTZ) * 1000000000)::BIGINT)
  AND (%s IS NULL OR EXISTS (
      SELECT 1 FROM traces t3
      WHERE t3.trace_id = rt.trace_id AND t3.name = %s
  ))
  AND (NOT %s OR EXISTS (
      SELECT 1 FROM traces t4
      WHERE t4.trace_id = rt.trace_id
        AND t4.attributes @> jsonb_build_object(%s, %s)
  ))
"""

LIST_TRACES_SELECT_SQL = f"""
SELECT
    s.trace_id,
    s.root_span,
    s.start_time,
    s.duration_ms,
    (SELECT COUNT(*) FROM traces t2 WHERE t2.trace_id = s.trace_id) AS span_count,
    s.status,
    s.attributes
FROM (
    {TRACE_SCOPE_SQL}
) AS s
"""

COUNT_TRACES_SQL = f"""
SELECT COUNT(*) AS total
FROM (
    {TRACE_SCOPE_SQL}
) AS s;
"""

BOUNDS_SQL = f"""
SELECT
    COUNT(*) AS total,
    COALESCE(MIN(duration_ms), 0) AS min_val,
    COALESCE(MAX(duration_ms), 0) AS max_val,
    COALESCE(AVG(duration_ms), 0) AS avg_ms,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) AS p50,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99
FROM (
    {TRACE_SCOPE_SQL}
) AS s
WHERE s.duration_ms IS NOT NULL;
"""

HISTOGRAM_SQL = f"""
SELECT
    width_bucket(s.duration_ms, %s, %s, %s) AS bucket,
    COUNT(*) AS count
FROM (
    {TRACE_SCOPE_SQL}
) AS s
WHERE s.duration_ms IS NOT NULL
GROUP BY bucket
ORDER BY bucket;
"""

SPAN_AVERAGES_SQL = f"""
SELECT
    name,
    COUNT(*) AS count,
    ROUND(AVG(duration_ms)::NUMERIC, 1) AS avg_ms,
    ROUND(MIN(duration_ms)::NUMERIC, 1) AS min_ms,
    ROUND(MAX(duration_ms)::NUMERIC, 1) AS max_ms,
    ROUND(
        (PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms))::NUMERIC,
        1
    ) AS p95_ms
FROM traces spans
WHERE spans.duration_ms IS NOT NULL
  AND spans.trace_id IN (
      SELECT scope.trace_id
      FROM (
          {TRACE_SCOPE_SQL}
      ) AS scope
  )
GROUP BY spans.name
ORDER BY avg_ms DESC;
"""

TRACE_SORT_FIELDS = {
    "trace_id": "s.trace_id",
    "root_span": "s.root_span",
    "start_time": "s.start_time_ns",
    "duration_ms": "s.duration_ms",
    "span_count": "span_count",
    "status": "s.status",
}

MAX_TRACE_QUERY_RANGE_DAYS = 31

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_to_date(to_date: str | None) -> str | None:
    """Erweitert nur echte date-only Werte um 23:59:59.

    Wichtig: Bereits zeitbehaftete Werte wie '2026-04-02 12:00:00' oder
    '2026-04-02T12:00:00' duerfen NICHT veraendert werden.
    """
    if to_date and _DATE_ONLY_RE.fullmatch(to_date):
        return to_date + " 23:59:59"
    return to_date


def _parse_date_param(value: str | None, param_name: str) -> datetime | None:
    """Parst einen Datums-String fuer Trace-Filter oder wirft HTTP 400."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        raise HTTPException(400, f"Ungueltiges Datumsformat fuer {param_name}: {value!r}")


def _validate_trace_date_range(from_date: str | None, to_date: str | None) -> None:
    """Begrenzt explizite Trace-Zeitfenster gegen teure Vollbereichs-Scans."""
    from_dt = _parse_date_param(from_date, "from_date")
    to_dt = _parse_date_param(to_date, "to_date")
    if from_dt is None or to_dt is None:
        return
    if to_dt < from_dt:
        raise HTTPException(400, "to_date muss nach from_date liegen")
    max_range = timedelta(days=MAX_TRACE_QUERY_RANGE_DAYS)
    if to_dt - from_dt > max_range:
        raise HTTPException(
            400,
            f"Trace-Zeitraum ist zu gross; maximal {MAX_TRACE_QUERY_RANGE_DAYS} Tage pro Anfrage",
        )


def _build_trace_scope_params(
    *,
    name: str | None,
    min_duration_ms: float | None,
    status: str | None,
    from_date: str | None,
    to_date: str | None,
    span_name: str | None = None,
    attr_search: str | None = None,
):
    to_date = _normalize_to_date(to_date)
    _validate_trace_date_range(from_date, to_date)

    attr_filter_active = False
    attr_key, attr_value = "", ""
    if attr_search and "=" in attr_search:
        attr_key, attr_value = attr_search.split("=", 1)
        attr_key = attr_key.strip()
        attr_value = attr_value.strip()
        attr_filter_active = True

    return (
        name,
        name,
        min_duration_ms,
        min_duration_ms,
        status,
        status,
        from_date,
        from_date,
        to_date,
        to_date,
        span_name,
        span_name,
        attr_filter_active,
        attr_key,
        attr_value,
    )


def _normalize_trace_sort(sort_field: str | None, sort_dir: str | None) -> tuple[str, str]:
    sort_sql = TRACE_SORT_FIELDS.get(sort_field or "", "s.start_time_ns")
    sort_dir_sql = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
    return sort_sql, sort_dir_sql


def _trace_status_from_spans(rows) -> str:
    """Aggregiert rohe Span-Statuswerte zu einem Trace-Status.

    Prioritaet: ERROR > UNSET > OK.
    """
    statuses = {str((row.get("status") or "OK")).upper() for row in rows}
    if "ERROR" in statuses:
        return "ERROR"
    if "UNSET" in statuses:
        return "UNSET"
    return "OK"


def _serialize_spans(rows):
    """Konvertiert RealDictRow-Span-Objekte in JSON-serialisierbare Dicts.

    Notwendig weil JSONResponse json.dumps() nutzt, das Decimal/datetime nicht
    serialisieren kann. FastAPI's jsonable_encoder (bei dict-Return) wuerde das
    automatisch tun, aber bei explizitem JSONResponse muss manuell konvertiert werden.
    """
    result = []
    for row in rows:
        d = dict(row)
        if d.get("duration_ms") is not None:
            d["duration_ms"] = float(d["duration_ms"])
        result.append(d)
    return result


def _build_diagnostics() -> dict:
    """Best-effort Diagnose-Metadaten fuer leeren Trace-Explorer.

    Wird nur bei total==0 aufgerufen. Jeder Teilschritt ist einzeln
    gegen Fehler abgesichert, damit /api/traces nie deswegen fehlschlaegt.
    """
    diag = {}
    diag_error = None

    try:
        cfg = load_config()
        diag["tracing_enabled"] = cfg.get("tracing", {}).get("enabled", False)
    except Exception as e:
        diag["tracing_enabled"] = None
        diag_error = f"load_config: {e}"

    try:
        with db_query() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM traces").fetchone()
            diag["trace_span_count"] = int(row["cnt"] or 0)
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM traces WHERE parent_span_id IS NULL"
            ).fetchone()
            diag["trace_root_count"] = int(row["cnt"] or 0)
    except Exception as e:
        diag.setdefault("trace_span_count", 0)
        diag.setdefault("trace_root_count", 0)
        diag_error = (diag_error + "; " if diag_error else "") + f"traces: {e}"

    try:
        with db_query() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM query_log").fetchone()
            diag["query_log_count"] = int(row["cnt"] or 0)
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM query_log WHERE trace_id IS NOT NULL"
            ).fetchone()
            diag["query_log_trace_id_count"] = int(row["cnt"] or 0)
    except Exception as e:
        diag.setdefault("query_log_count", 0)
        diag.setdefault("query_log_trace_id_count", 0)
        diag_error = (diag_error + "; " if diag_error else "") + f"query_log: {e}"

    result = {"diagnostics": diag}
    if diag_error:
        result["diagnostics_error"] = diag_error
    return result


def _count_traces(name, min_duration_ms, status, from_date, to_date, span_name, attr_search) -> int:
    params = _build_trace_scope_params(
        name=name,
        min_duration_ms=min_duration_ms,
        status=status,
        from_date=from_date,
        to_date=to_date,
        span_name=span_name,
        attr_search=attr_search,
    )

    with db_query() as conn:
        row = conn.execute(COUNT_TRACES_SQL, params).fetchone()
    return int(row["total"] or 0)


def _query_traces(
    name,
    min_duration_ms,
    status,
    from_date,
    to_date,
    span_name,
    attr_search,
    limit,
    *,
    offset=0,
    sort_field="start_time",
    sort_dir="desc",
):
    """Fuehrt die Trace-Abfrage mit allen Filtern aus.

    Gibt eine Liste von JSON-serialisierbaren dicts zurueck (nicht RealDictRow).
    Wird von list_traces() und export_traces() verwendet.
    """
    sort_sql, sort_dir_sql = _normalize_trace_sort(sort_field, sort_dir)
    sql = LIST_TRACES_SELECT_SQL + f"\nORDER BY {sort_sql} {sort_dir_sql}\nLIMIT %s OFFSET %s"

    params = (
        *_build_trace_scope_params(
            name=name,
            min_duration_ms=min_duration_ms,
            status=status,
            from_date=from_date,
            to_date=to_date,
            span_name=span_name,
            attr_search=attr_search,
        ),
        limit,
        offset,
    )

    with db_query() as conn:
        rows = conn.execute(sql, params).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        d["id"] = d["trace_id"]  # ExpandableTable nutzt row.id als stabile Row-ID
        if d.get("start_time") and hasattr(d["start_time"], "isoformat"):
            d["start_time"] = d["start_time"].isoformat()
        if d.get("duration_ms") is not None:
            d["duration_ms"] = float(d["duration_ms"])
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Endpunkte — Reihenfolge ist KRITISCH (feste Pfade VOR {trace_id})
# ---------------------------------------------------------------------------


# 0. Tracing-Toggle + Status
@router.get("/api/traces/status")
async def tracing_status(user: dict = Depends(get_admin_user)) -> dict:
    """Gibt den aktuellen Tracing-Status zurueck (fuer Frontend-Toggle)."""
    cfg = load_config()
    return {"enabled": cfg.get("tracing", {}).get("enabled", False)}


@router.post("/api/traces/toggle")
async def toggle_tracing(user: dict = Depends(get_admin_user)) -> dict:
    """Schaltet Tracing ein/aus und startet kiara.service neu."""
    import asyncio

    from kiara.config import save_config

    cfg = load_config()
    current = cfg.get("tracing", {}).get("enabled", False)
    new_state = not current
    cfg.setdefault("tracing", {})["enabled"] = new_state
    try:
        save_config(cfg)
    except (PermissionError, OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Config-Fehler: {exc}")
    log_config_change(
        user,
        area="tracing",
        action="update",
        field_name="enabled",
        old_value=str(current),
        new_value=str(new_state),
        source="gui",
    )

    # kiara.service neustarten damit init_tracing() neu laeuft
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "systemctl",
            "restart",
            "kiara.service",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {
                "enabled": new_state,
                "service_restarted": False,
                "error": stderr.decode().strip(),
            }
    except (FileNotFoundError, OSError) as exc:
        return {"enabled": new_state, "service_restarted": False, "error": str(exc)}

    return {"enabled": new_state, "service_restarted": True}


# 1. Basis-Endpunkt: Liste
@router.get("/api/traces")
async def list_traces(
    # --- ExpandableTable / Pagination ---
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
    sort_field: str = Query(default="start_time"),
    sort_dir: str = Query(default="desc"),
    # --- Filter ---
    name: str | None = None,
    min_duration_ms: float | None = Query(default=None, ge=0),
    status: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    span_name: str | None = None,
    attr_search: str | None = None,
    user: dict = Depends(get_admin_user),
) -> dict:
    offset = (page - 1) * page_size
    total = _count_traces(
        name,
        min_duration_ms,
        status,
        from_date,
        to_date,
        span_name,
        attr_search,
    )
    rows = _query_traces(
        name,
        min_duration_ms,
        status,
        from_date,
        to_date,
        span_name,
        attr_search,
        page_size,
        offset=offset,
        sort_field=sort_field,
        sort_dir=sort_dir,
    )
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    response = {
        "data": rows,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total,
            "total_pages": total_pages,
        },
    }
    if total == 0:
        response.update(_build_diagnostics())
    return response


# 2. Feste Pfade (Trace Explorer) — VOR {trace_id}
@router.get("/api/traces/stats/duration-histogram")
async def duration_histogram(
    buckets: int = Query(default=20, ge=2, le=100),
    name: str | None = None,
    min_duration_ms: float | None = None,
    status: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    span_name: str | None = None,
    attr_search: str | None = None,
    user: dict = Depends(get_admin_user),
) -> dict:
    filter_params = _build_trace_scope_params(
        name=name,
        min_duration_ms=min_duration_ms,
        status=status,
        from_date=from_date,
        to_date=to_date,
        span_name=span_name,
        attr_search=attr_search,
    )

    with db_query() as conn:
        stats = conn.execute(BOUNDS_SQL, filter_params).fetchone()

        total = stats["total"]
        if total == 0:
            return {"buckets": [], "total": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0}

        min_val = float(stats["min_val"] or 0)
        max_val = float(stats["max_val"] or 0)

        # Edge Case: Alle Traces gleiche Dauer → kuenstliche Spreizung
        if min_val == max_val:
            max_val = min_val + 1.0

        rows = conn.execute(
            HISTOGRAM_SQL,
            (min_val, max_val + 0.001, buckets, *filter_params),
        ).fetchall()

    bucket_width = (max_val - min_val) / buckets
    bucket_counts = {}
    for row in rows:
        b = row["bucket"] - 1  # width_bucket ist 1-basiert
        b = max(0, min(b, buckets - 1))  # Clamp: Unter-/Ueber-Bereich-Werte
        bucket_counts[b] = bucket_counts.get(b, 0) + row["count"]

    result_buckets = []
    for b in range(buckets):
        result_buckets.append(
            {
                "from_ms": round(min_val + b * bucket_width, 1),
                "to_ms": round(min_val + (b + 1) * bucket_width, 1),
                "count": bucket_counts.get(b, 0),
            }
        )

    return {
        "buckets": result_buckets,
        "total": total,
        "avg_ms": round(float(stats["avg_ms"] or 0), 1),
        "p50_ms": round(float(stats["p50"] or 0), 1),
        "p95_ms": round(float(stats["p95"] or 0), 1),
        "p99_ms": round(float(stats["p99"] or 0), 1),
    }


@router.get("/api/traces/stats/span-averages")
async def span_averages(
    name: str | None = None,
    min_duration_ms: float | None = None,
    status: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    span_name: str | None = None,
    attr_search: str | None = None,
    user: dict = Depends(get_admin_user),
) -> dict:
    filter_params = _build_trace_scope_params(
        name=name,
        min_duration_ms=min_duration_ms,
        status=status,
        from_date=from_date,
        to_date=to_date,
        span_name=span_name,
        attr_search=attr_search,
    )

    with db_query() as conn:
        rows = conn.execute(SPAN_AVERAGES_SQL, filter_params).fetchall()

    spans = []
    for r in rows:
        d = dict(r)
        for key in ("avg_ms", "min_ms", "max_ms", "p95_ms"):
            if d.get(key) is not None:
                d[key] = float(d[key])
        spans.append(d)
    return {"spans": spans}


@router.get("/api/traces/compare")
async def compare_traces(
    trace_a: str = Query(..., description="Erste trace_id"),
    trace_b: str = Query(..., description="Zweite trace_id"),
    user: dict = Depends(get_admin_user),
) -> dict:
    with db_query() as conn:
        spans_a = conn.execute(SPAN_QUERY_SQL, (trace_a,)).fetchall()
        spans_b = conn.execute(SPAN_QUERY_SQL, (trace_b,)).fetchall()

    if not spans_a:
        raise HTTPException(404, f"Trace {trace_a} nicht gefunden")
    if not spans_b:
        raise HTTPException(404, f"Trace {trace_b} nicht gefunden")

    def _aggregate(spans):
        agg = defaultdict(lambda: {"total_ms": 0.0, "count": 0})
        for s in spans:
            agg[s["name"]]["total_ms"] += float(s["duration_ms"] or 0)
            agg[s["name"]]["count"] += 1
        return agg

    agg_a = _aggregate(spans_a)
    agg_b = _aggregate(spans_b)
    all_names = sorted(set(agg_a) | set(agg_b))

    diff = []
    for sname in all_names:
        a_ms = round(agg_a[sname]["total_ms"], 1) if sname in agg_a else None
        b_ms = round(agg_b[sname]["total_ms"], 1) if sname in agg_b else None
        delta_ms = round((b_ms or 0) - (a_ms or 0), 1)
        delta_pct = round(delta_ms / a_ms * 100, 1) if a_ms else None
        diff.append(
            {
                "span_name": sname,
                "a_ms": a_ms,
                "b_ms": b_ms,
                "a_count": agg_a[sname]["count"] if sname in agg_a else 0,
                "b_count": agg_b[sname]["count"] if sname in agg_b else 0,
                "delta_ms": delta_ms,
                "delta_pct": delta_pct,
            }
        )

    root_a = next((s for s in spans_a if not s["parent_span_id"]), spans_a[0])
    root_b = next((s for s in spans_b if not s["parent_span_id"]), spans_b[0])

    return {
        "trace_a": {
            "trace_id": trace_a,
            "root_span": root_a["name"],
            "status": _trace_status_from_spans(spans_a),
            "duration_ms": float(root_a["duration_ms"] or 0),
            "spans": _serialize_spans(spans_a),
        },
        "trace_b": {
            "trace_id": trace_b,
            "root_span": root_b["name"],
            "status": _trace_status_from_spans(spans_b),
            "duration_ms": float(root_b["duration_ms"] or 0),
            "spans": _serialize_spans(spans_b),
        },
        "diff": diff,
    }


@router.get("/api/traces/export")
async def export_traces(
    fmt: str = Query(default="csv", pattern="^(csv|json)$"),
    limit: int = Query(default=5000, ge=1, le=5000),
    sort_field: str = Query(default="start_time"),
    sort_dir: str = Query(default="desc"),
    name: str | None = None,
    min_duration_ms: float | None = None,
    status: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    span_name: str | None = None,
    attr_search: str | None = None,
    user: dict = Depends(get_admin_user),
) -> Response:
    rows = _query_traces(
        name,
        min_duration_ms,
        status,
        from_date,
        to_date,
        span_name,
        attr_search,
        limit + 1,
        offset=0,
        sort_field=sort_field,
        sort_dir=sort_dir,
    )
    if len(rows) > limit:
        raise HTTPException(
            400,
            f"Export auf {limit} Traces begrenzt; aktuelle Filter matchen mehr als {limit}. "
            "Bitte Filter weiter einschraenken.",
        )

    export_rows = [{k: v for k, v in row.items() if k != "id"} for row in rows[:limit]]

    if fmt == "csv":
        output = io.StringIO()
        writer = csv.writer(output, delimiter=";")
        writer.writerow(
            ["trace_id", "root_span", "start_time", "duration_ms", "span_count", "status"]
        )
        for row in export_rows:
            writer.writerow(
                [
                    row["trace_id"],
                    row["root_span"],
                    row["start_time"],
                    row["duration_ms"],
                    row["span_count"],
                    row["status"],
                ]
            )
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=traces.csv"},
        )

    # fmt == "json"
    return JSONResponse(
        content={"data": export_rows},
        headers={"Content-Disposition": "attachment; filename=traces.json"},
    )


def _get_trace_by_id(trace_id: str) -> dict | None:
    """Trace-Daten mit Spans laden. Gibt None zurueck wenn nicht gefunden."""
    with db_query() as conn:
        rows = conn.execute(SPAN_QUERY_SQL, (trace_id,)).fetchall()
    if not rows:
        return None
    root = next((s for s in rows if not s["parent_span_id"]), rows[0])
    return {
        "trace_id": trace_id,
        "root_span": root["name"],
        "duration_ms": float(root["duration_ms"] or 0),
        "span_count": len(rows),
        "status": _trace_status_from_spans(rows),
        "spans": _serialize_spans(rows),
    }


# 3. Parametrische Routen — ZULETZT
@router.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str, user: dict = Depends(get_admin_user)):
    result = _get_trace_by_id(trace_id)
    if not result:
        raise HTTPException(status_code=404, detail="Trace nicht gefunden")
    return result


@router.get("/api/traces/{trace_id}/export")
async def export_single_trace(
    trace_id: str,
    user: dict = Depends(get_admin_user),
) -> JSONResponse:
    result = _get_trace_by_id(trace_id)
    if not result:
        raise HTTPException(status_code=404, detail="Trace nicht gefunden")
    return JSONResponse(
        content=result,
        headers={"Content-Disposition": "attachment; filename=trace_export.json"},
    )
