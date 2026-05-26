"""Admin-Funktionen fuer Tracing (P17)."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# tracing.toggle — Ein-/Ausschalten mit automatischem Service-Restart
# ---------------------------------------------------------------------------


@admin_function(
    name="tracing.toggle",
    category="tracing",
    description="Tracing ein-/ausschalten",
    help="Aktiviert oder deaktiviert OpenTelemetry-Tracing. Aendert tracing.enabled "
    "in der Config und startet kiara.service automatisch neu, damit die "
    "Aenderung wirksam wird.",
    examples=[
        "Aktiviere Tracing",
        "Tracing ausschalten",
        "Schalte Tracing ein",
    ],
    parameters={
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "True = aktivieren, False = deaktivieren",
            },
        },
        "required": ["enabled"],
    },
    returns="Bestaetigug der Aenderung mit Service-Status",
    requires_confirmation=True,
    confirm_template="tracing {enabled}",
    changelog_area="tracing",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/tracing/toggle"),
)
async def tracing_toggle(user: dict, enabled: bool) -> AdminResult:
    import asyncio

    from kiara.config import load_config, save_config

    # 1. Config aendern
    cfg = load_config()
    cfg.setdefault("tracing", {})["enabled"] = enabled
    try:
        save_config(cfg)
    except (PermissionError, OSError, ValueError) as exc:
        return AdminResult.fail(f"Config-Fehler: {exc}")

    # 2. kiara.service neustarten (Tracing wird bei init_tracing() beim Start geladen)
    svc_warning = None
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
            svc_warning = stderr.decode().strip()
            logger.warning("systemctl restart kiara.service fehlgeschlagen: %s", svc_warning)
    except (FileNotFoundError, OSError) as exc:
        svc_warning = str(exc)
        logger.warning("systemctl restart fehlgeschlagen: %s", exc)

    # 3. Status pruefen — Polling statt fixem Sleep
    service_status = "unknown"
    for _ in range(10):
        await asyncio.sleep(0.5)
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                "is-active",
                "kiara.service",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            service_status = stdout.decode().strip()
            if service_status == "active":
                break
        except Exception:
            service_status = "unknown"

    status_text = "aktiviert" if enabled else "deaktiviert"
    msg = f"Tracing {status_text}"
    if svc_warning:
        msg += f" (Config gespeichert, aber Service-Restart fehlgeschlagen: {svc_warning})"
    elif service_status != "active":
        msg += f" (Config gespeichert, Service-Status: {service_status})"
    else:
        msg += ", Service neu gestartet"
    return AdminResult.ok(
        msg,
        data={
            "enabled": enabled,
            "service_status": service_status,
        },
    )


@admin_function(
    name="tracing.list",
    category="tracing",
    description="Traces auflisten",
    help="Listet aufgezeichnete OpenTelemetry-Traces mit Pagination und optionalen Filtern. "
    "Unterstuetzt Filterung nach Root-Span-Name, Mindest-Dauer, Status (OK/ERROR/UNSET), "
    "Zeitraum, Span-Name und JSONB-Attribut-Suche.",
    examples=[
        "Zeig mir die letzten Traces",
        "Liste alle fehlgeschlagenen Traces",
        "Welche Traces gibt es von heute?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Seite (ab 1)", "default": 1},
            "page_size": {
                "type": "integer",
                "description": "Eintraege pro Seite (max 500)",
                "default": 100,
            },
            "name": {"type": "string", "description": "Root-Span-Name-Filter (z.B. 'rag_query')"},
            "min_duration_ms": {"type": "number", "description": "Mindest-Dauer in Millisekunden"},
            "status": {"type": "string", "description": "Trace-Status (OK, ERROR, UNSET)"},
            "from_date": {"type": "string", "description": "Startdatum (YYYY-MM-DD)"},
            "to_date": {"type": "string", "description": "Enddatum (YYYY-MM-DD)"},
        },
        "required": [],
    },
    returns="Paginierte Trace-Liste mit trace_id, root_span, duration_ms, status, span_count",
    raw_response=True,
    http=("GET", "/api/admin/tracing/list"),
)
async def tracing_list(user: dict, **params) -> AdminResult:
    from kiara.web.api_tracing import _count_traces, _query_traces

    page = max(1, params.get("page", 1))
    page_size = max(1, min(params.get("page_size", 100), 500))
    name = params.get("name")
    min_duration_ms = params.get("min_duration_ms")
    status = params.get("status")
    from_date = params.get("from_date")
    to_date = params.get("to_date")

    total = _count_traces(name, min_duration_ms, status, from_date, to_date, None, None)
    rows = _query_traces(
        name,
        min_duration_ms,
        status,
        from_date,
        to_date,
        None,
        None,
        page_size,
        offset=(page - 1) * page_size,
    )
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return AdminResult.ok(
        f"{total} Trace(s) gefunden (Seite {page}/{total_pages})",
        data={
            "data": rows,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
            },
        },
    )


@admin_function(
    name="tracing.detail",
    category="tracing",
    description="Trace-Details mit Spans anzeigen",
    help="Zeigt einen einzelnen Trace mit allen Spans, Timing und Status-Informationen. "
    "Die Span-Hierarchie (Parent/Child) ist ueber parent_span_id nachvollziehbar.",
    examples=[
        "Zeig mir die Details zum Trace abc123",
        "Was passiert im Trace xyz?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "trace_id": {"type": "string", "description": "Die 32-stellige Trace-ID"},
        },
        "required": ["trace_id"],
    },
    returns="Trace-Objekt mit trace_id, root_span, duration_ms, status, span_count und spans-Array",
    raw_response=True,
    http=("GET", "/api/admin/tracing/detail"),
)
async def tracing_detail(user: dict, **params) -> AdminResult:
    from kiara.web.api_tracing import _get_trace_by_id

    trace_id = params.get("trace_id", "")
    if not trace_id:
        return AdminResult.fail("trace_id ist erforderlich")

    result = _get_trace_by_id(trace_id)
    if not result:
        return AdminResult.fail(f"Trace {trace_id} nicht gefunden")

    return AdminResult.ok(
        f"Trace {trace_id[:8]}… — {result['root_span']} "
        f"({result['duration_ms']:.0f}ms, {result['span_count']} Spans)",
        data=result,
    )


@admin_function(
    name="tracing.histogram",
    category="tracing",
    description="Duration-Histogram der Traces anzeigen",
    help="Erstellt eine Bucket-basierte Verteilung der Trace-Dauern inkl. P50/P95/P99-Percentilen. "
    "Die gleichen Filter wie bei tracing.list sind anwendbar.",
    examples=[
        "Zeig mir das Dauer-Histogram der Traces",
        "Wie verteilen sich die Trace-Laufzeiten?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "buckets": {"type": "integer", "description": "Anzahl Buckets (2-100)", "default": 20},
            "name": {"type": "string", "description": "Root-Span-Name-Filter"},
            "status": {"type": "string", "description": "Trace-Status-Filter"},
            "from_date": {"type": "string", "description": "Startdatum (YYYY-MM-DD)"},
            "to_date": {"type": "string", "description": "Enddatum (YYYY-MM-DD)"},
        },
        "required": [],
    },
    returns="Buckets mit from_ms/to_ms/count plus Gesamt-Statistiken (total, avg, P50, P95, P99)",
    raw_response=True,
    http=("GET", "/api/admin/tracing/histogram"),
)
async def tracing_histogram(user: dict, **params) -> AdminResult:
    from kiara.shared.db import db_query
    from kiara.web.api_tracing import BOUNDS_SQL, HISTOGRAM_SQL, _build_trace_scope_params

    buckets = min(max(params.get("buckets", 20), 2), 100)
    filter_params = _build_trace_scope_params(
        name=params.get("name"),
        min_duration_ms=params.get("min_duration_ms"),
        status=params.get("status"),
        from_date=params.get("from_date"),
        to_date=params.get("to_date"),
    )

    with db_query() as conn:
        stats = conn.execute(BOUNDS_SQL, filter_params).fetchone()
        total = stats["total"]
        if total == 0:
            return AdminResult.ok(
                "Keine Traces im gewaehlten Zeitraum",
                data={
                    "buckets": [],
                    "total": 0,
                    "avg_ms": 0,
                    "p50_ms": 0,
                    "p95_ms": 0,
                    "p99_ms": 0,
                },
            )

        min_val = float(stats["min_val"] or 0)
        max_val = float(stats["max_val"] or 0)
        if min_val == max_val:
            max_val = min_val + 1.0

        rows = conn.execute(
            HISTOGRAM_SQL,
            (min_val, max_val + 0.001, buckets, *filter_params),
        ).fetchall()

    bucket_width = (max_val - min_val) / buckets
    bucket_counts = {}
    for row in rows:
        b = max(0, min(row["bucket"] - 1, buckets - 1))
        bucket_counts[b] = bucket_counts.get(b, 0) + row["count"]

    result_buckets = [
        {
            "from_ms": round(min_val + b * bucket_width, 1),
            "to_ms": round(min_val + (b + 1) * bucket_width, 1),
            "count": bucket_counts.get(b, 0),
        }
        for b in range(buckets)
    ]

    data = {
        "buckets": result_buckets,
        "total": total,
        "avg_ms": round(float(stats["avg_ms"] or 0), 1),
        "p50_ms": round(float(stats["p50"] or 0), 1),
        "p95_ms": round(float(stats["p95"] or 0), 1),
        "p99_ms": round(float(stats["p99"] or 0), 1),
    }
    return AdminResult.ok(
        f"Histogram: {total} Traces, P50={data['p50_ms']}ms, P95={data['p95_ms']}ms",
        data=data,
    )


@admin_function(
    name="tracing.span_averages",
    category="tracing",
    description="Durchschnittliche Span-Dauern anzeigen",
    help="Berechnet pro Span-Typ die durchschnittliche Dauer, Aufrufhaeufigkeit und "
    "Percentile. Nuetzlich um Engpaesse zu identifizieren (z.B. langsames Reranking).",
    examples=[
        "Welche Spans sind am langsamsten?",
        "Zeig mir die Span-Averages",
        "Wo verbringen Traces die meiste Zeit?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Root-Span-Name-Filter"},
            "status": {"type": "string", "description": "Trace-Status-Filter"},
            "from_date": {"type": "string", "description": "Startdatum (YYYY-MM-DD)"},
            "to_date": {"type": "string", "description": "Enddatum (YYYY-MM-DD)"},
        },
        "required": [],
    },
    returns="Liste von Span-Typen mit avg_ms, count, P50, P95, P99",
    raw_response=True,
    http=("GET", "/api/admin/tracing/span-averages"),
)
async def tracing_span_averages(user: dict, **params) -> AdminResult:
    from kiara.shared.db import db_query
    from kiara.web.api_tracing import SPAN_AVERAGES_SQL, _build_trace_scope_params

    filter_params = _build_trace_scope_params(
        name=params.get("name"),
        min_duration_ms=params.get("min_duration_ms"),
        status=params.get("status"),
        from_date=params.get("from_date"),
        to_date=params.get("to_date"),
    )

    with db_query() as conn:
        rows = conn.execute(SPAN_AVERAGES_SQL, filter_params).fetchall()

    spans = [dict(r) for r in rows]
    if not spans:
        return AdminResult.ok("Keine Span-Daten im gewaehlten Zeitraum", data={"spans": []})

    return AdminResult.ok(
        f"{len(spans)} Span-Typen analysiert",
        data={"spans": spans},
    )


@admin_function(
    name="tracing.compare",
    category="tracing",
    description="Zwei Traces vergleichen",
    help="Vergleicht zwei Traces anhand ihrer Span-Dauern. Zeigt pro Span-Typ "
    "die Dauer in beiden Traces, die Differenz (absolut und prozentual) "
    "und die Aufrufhaeufigkeit.",
    examples=[
        "Vergleiche Trace abc123 mit def456",
        "Warum ist Trace X langsamer als Trace Y?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "trace_a": {"type": "string", "description": "Erste Trace-ID"},
            "trace_b": {"type": "string", "description": "Zweite Trace-ID"},
        },
        "required": ["trace_a", "trace_b"],
    },
    returns="Vergleichstabelle mit span_name, a_ms, b_ms, delta_ms, delta_pct",
    raw_response=True,
    http=("GET", "/api/admin/tracing/compare"),
)
async def tracing_compare(user: dict, **params) -> AdminResult:
    from collections import defaultdict

    from kiara.shared.db import db_query
    from kiara.web.api_tracing import (
        SPAN_QUERY_SQL,
        _trace_status_from_spans,
    )

    trace_a = params.get("trace_a", "")
    trace_b = params.get("trace_b", "")
    if not trace_a or not trace_b:
        return AdminResult.fail("trace_a und trace_b sind erforderlich")

    with db_query() as conn:
        spans_a = conn.execute(SPAN_QUERY_SQL, (trace_a,)).fetchall()
        spans_b = conn.execute(SPAN_QUERY_SQL, (trace_b,)).fetchall()

    if not spans_a:
        return AdminResult.fail(f"Trace {trace_a} nicht gefunden")
    if not spans_b:
        return AdminResult.fail(f"Trace {trace_b} nicht gefunden")

    def _aggregate(spans):
        agg = defaultdict(lambda: {"total_ms": 0.0, "count": 0})
        for s in spans:
            agg[s["name"]]["total_ms"] += float(s["duration_ms"] or 0)
            agg[s["name"]]["count"] += 1
        return agg

    agg_a = _aggregate(spans_a)
    agg_b = _aggregate(spans_b)

    diff = []
    for sname in sorted(set(agg_a) | set(agg_b)):
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

    return AdminResult.ok(
        f"Vergleich: {root_a['name']} ({float(root_a['duration_ms'] or 0):.0f}ms) "
        f"vs {root_b['name']} ({float(root_b['duration_ms'] or 0):.0f}ms)",
        data={
            "trace_a": {
                "trace_id": trace_a,
                "root_span": root_a["name"],
                "status": _trace_status_from_spans(spans_a),
                "duration_ms": float(root_a["duration_ms"] or 0),
            },
            "trace_b": {
                "trace_id": trace_b,
                "root_span": root_b["name"],
                "status": _trace_status_from_spans(spans_b),
                "duration_ms": float(root_b["duration_ms"] or 0),
            },
            "diff": diff,
        },
    )
