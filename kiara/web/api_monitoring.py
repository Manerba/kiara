"""KIara System-Monitor API.

Endpoints:
    GET  /api/monitoring/stream                 → SSE (1/s, Live-Metriken)
    GET  /api/monitoring/history?range=1h       → Zeitreihe aus DB
    GET  /api/monitoring/services               → Service-Status (via admin_function)
    POST /api/monitoring/services/{name}/restart → Neustart (via admin_function)
    GET  /api/monitoring/current                → Letzter Snapshot (via admin_function)
    GET  /api/monitoring/journal                → Letzte Fehler
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kiara.shared.constants import METRICS_MAX_DATAPOINTS
from kiara.web.auth import get_admin_user

log = logging.getLogger(__name__)
router = APIRouter()


def _downsample_rows(rows: list, max_points: int) -> list:
    """Duennere Zeitreihen aus und erhalte dabei ersten und letzten Punkt."""
    if len(rows) <= max_points:
        return rows
    if max_points <= 1:
        return [rows[-1]]

    last_index = len(rows) - 1
    indices = [round(i * last_index / (max_points - 1)) for i in range(max_points)]
    return [rows[index] for index in indices]


# ──────────────────── SSE-Stream ────────────────────


@router.get("/api/monitoring/stream")
async def metrics_stream(request: Request, user=Depends(get_admin_user)):
    """Server-Sent Events Stream fuer Live-Metriken (1 Event/Sekunde).

    Event-Format:
        event: metrics
        data: {"cpu_percent": 12.5, "ram_used_bytes": 4500000000, ...}

    Der Stream laeuft bis der Client die Verbindung schliesst.
    EventSource im Browser reconnected automatisch bei Verbindungsabbruch.
    """
    from kiara.monitoring.collector import subscribe, unsubscribe

    queue = subscribe()

    async def _event_generator():
        try:
            while True:
                # Client disconnected?
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield f"event: metrics\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive (verhindert Proxy-Timeout)
                    yield ": keepalive\n\n"
        finally:
            unsubscribe(queue)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx: kein Response-Buffering
        },
    )


# ──────────────────── History (DB) ────────────────────


@router.get("/api/monitoring/history")
async def get_metrics_history(
    time_range: str = Query("1h", alias="range", pattern="^(1h|24h|30d|1y)$"),
    metric: str = Query("cpu", pattern="^(cpu|ram|swap|load|net|disk)$"),
    max_points: int = Query(METRICS_MAX_DATAPOINTS, ge=10, le=2000),
    user=Depends(get_admin_user),
):
    """Historische Zeitreihe fuer eine Metrik-Gruppe."""
    from kiara.shared.db import db_query

    range_map = {
        "1h": ("1 hour", ("raw",)),
        "24h": ("24 hours", ("raw",)),
        "30d": ("30 days", ("raw", "5m")),
        "1y": ("365 days", ("raw", "5m", "1h")),
    }
    interval, resolutions = range_map[time_range]
    preferred_res = resolutions[-1]

    metric_columns = {
        "cpu": ["cpu_percent"],
        "ram": ["ram_used_bytes", "ram_total_bytes"],
        "swap": ["swap_used_bytes", "swap_total_bytes"],
        "load": ["load_1m", "load_5m", "load_15m"],
        "net": ["net_sent_bps", "net_recv_bps"],
        "disk": ["disk_used_bytes", "disk_total_bytes"],
    }
    columns = metric_columns[metric]
    col_sql = ", ".join(columns)
    res_clause = ", ".join(f"'{r}'" for r in resolutions)

    with db_query() as conn:
        rows = conn.execute(
            f"SELECT ts, {col_sql} FROM system_metrics "
            f"WHERE resolution IN ({res_clause}) "
            f"  AND ts >= NOW() - INTERVAL '{interval}' "
            f"ORDER BY ts ASC"
        ).fetchall()

    # Ausduennen wenn zu viele Punkte
    rows = _downsample_rows(rows, max_points)

    timestamps = [
        row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else str(row["ts"]) for row in rows
    ]
    series = {col: [row[col] for row in rows] for col in columns}

    return JSONResponse(
        content={
            "timestamps": timestamps,
            "series": series,
            "resolution": preferred_res,
            "range": time_range,
            "count": len(rows),
        }
    )


# ──────────────────── Services ────────────────────
# Service-Endpoints werden NICHT hier definiert, sondern automatisch
# von generate_admin_routes() erzeugt, weil die bestehenden
# @admin_function-Decorators in kiara/admin/functions/services.py
# um http=() Routen erweitert werden (siehe Schicht 4).
#
# Ergebnis-Endpoints (via generate_admin_routes):
#   GET  /api/monitoring/services         → services.list()
#   POST /api/monitoring/services/{service}/restart  → services.restart(service=...)
#   GET  /api/monitoring/current           → system.metrics()
#   GET  /api/monitoring/journal           → system.journal()
