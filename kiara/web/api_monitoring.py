"""KIara System-Monitor API.

Endpoints:
    GET  /api/monitoring/stream                 → SSE (1/s, Live-Metriken)
    GET  /api/monitoring/live?range=1m          → 250-ms-Verlauf aus RAM (1m/10m)
    GET  /api/monitoring/history?range=1h       → Gebuendelte Zeitreihen aus DB
    GET  /api/monitoring/services               → Service-Status (via admin_function)
    POST /api/monitoring/services/{name}/restart → Neustart (via admin_function)
    GET  /api/monitoring/current                → Letzter Snapshot (via admin_function)
    GET  /api/monitoring/journal                → Letzte Fehler
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from kiara.shared.constants import METRICS_MAX_DATAPOINTS
from kiara.web.auth import get_admin_user

log = logging.getLogger(__name__)
router = APIRouter()


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


@router.get("/api/monitoring/live")
async def live_history(
    request: Request,
    time_range: str = Query("1m", alias="range", pattern="^(1m|10m)$"),
    user=Depends(get_admin_user),
):
    """250-ms-Verlauf: vorhandener RAM-Puffer, danach neue Punkte (ohne DB)."""
    from kiara.monitoring.high_res_history import HighResHistoryClosed

    service = getattr(request.app.state, "monitoring_history", None)
    if service is None or not service.running:
        raise HTTPException(status_code=503, detail="Live-Sampler nicht verfügbar")

    async def events():
        frame = service.snapshot_frame(time_range)
        sequence = frame["samples"][-1]["sequence"] if frame["samples"] else 0
        yield "retry: 1000\n\n"
        yield f"event: history\ndata: {json.dumps(frame, allow_nan=False)}\n\n"
        while not await request.is_disconnected():
            try:
                sample = await asyncio.wait_for(service.wait_for_next(sequence), timeout=5)
                sequence = sample["sequence"]
                frame = service.point_frame(sample, time_range)
                yield f"event: history\ndata: {json.dumps(frame, allow_nan=False)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
            except HighResHistoryClosed:
                break

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/monitoring/history")
def get_metrics_history(
    time_range: str = Query("1h", alias="range", pattern="^(1h|6h|24h|7d|30d|90d|1y)$"),
    max_points: int = Query(METRICS_MAX_DATAPOINTS, ge=10, le=2000),
    user=Depends(get_admin_user),
):
    """Alle historischen Kurven gemeinsam; blockierende DB-Arbeit im Threadpool."""
    from kiara.monitoring.history import get_history

    return JSONResponse(content=get_history(time_range, max_points))


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
