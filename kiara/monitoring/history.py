"""Gebuendelte PostgreSQL-Zeitreihen fuer die Langzeitansichten des Monitors.

Wie in Kiron werden alle Kurven gemeinsam in Zeitfenstern gemittelt. Die
vorhandenen KIara-Retentionsstufen bleiben erhalten; ihre unterschiedliche
Dauer wird beim Mitteln gewichtet. Kurze Antwort-Caches teilen Abfragen
zwischen Browsern, die 250-ms-Livehistorie greift ausschliesslich auf RAM zu.
"""

import math
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from kiara.shared.db import db_query

HISTORY_RANGES = {
    "1h": (3600, 10),
    "6h": (21600, 60),
    "24h": (86400, 60),
    "7d": (604800, 300),
    "30d": (2592000, 300),
    "90d": (7776000, 300),
    "1y": (31536000, 300),
}

_METRIC_SQL = {
    "cpu_usage": "cpu_percent",
    "cpu_load_1m": "load_1m",
    "memory_usage": "ram_percent",
    "memory_used_gb": "ram_used_bytes / 1073741824.0",
    "disk_read_mb_s": "disk_read_bps / 1048576.0",
    "disk_write_mb_s": "disk_write_bps / 1048576.0",
    "net_sent_mb_s": "net_sent_bps / 1048576.0",
    "net_recv_mb_s": "net_recv_bps / 1048576.0",
}
_cache: OrderedDict[tuple[str, int], tuple[float, dict]] = OrderedDict()
_cache_lock = threading.Lock()
_CACHE_CAPACITY = 32


def ensure_history_schema(conn) -> None:
    """Ergaenzt die drei bisher nicht persistierten Kurvenwerte."""
    conn.execute(
        "ALTER TABLE public.system_metrics "
        "ADD COLUMN IF NOT EXISTS ram_percent real, "
        "ADD COLUMN IF NOT EXISTS disk_read_bps real, "
        "ADD COLUMN IF NOT EXISTS disk_write_bps real"
    )


def get_history(time_range: str, max_points: int = 500) -> dict:
    if time_range not in HISTORY_RANGES or not 10 <= max_points <= 2000:
        raise ValueError("ungueltiger Monitoring-Zeitraum oder Punktelimit")
    key = (time_range, max_points)
    # Endpoint laeuft im Threadpool. Ein Cache-Miss pro Zeitfenster wird
    # gemeinsam bearbeitet, auch bei gleichzeitigen Requests mehrerer Browser.
    with _cache_lock:
        cached = _cache.get(key)
        if cached and time.monotonic() < cached[0]:
            _cache.move_to_end(key)
            return deepcopy(cached[1])
        result = _query_history(time_range, max_points)
        _cache[key] = (time.monotonic() + HISTORY_RANGES[time_range][1], result)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_CAPACITY:
            _cache.popitem(last=False)
        return deepcopy(result)


def _query_history(time_range: str, max_points: int) -> dict:
    seconds, _ttl = HISTORY_RANGES[time_range]
    end = datetime.now(timezone.utc)
    start = end - timedelta(seconds=seconds)
    expressions = ", ".join(f"{expression} AS {key}" for key, expression in _METRIC_SQL.items())
    averages = ", ".join(
        f"SUM({key} * weight) / NULLIF(SUM(weight) FILTER (WHERE {key} IS NOT NULL), 0) AS {key}"
        for key in _METRIC_SQL
    )
    with db_query() as conn:
        conn.execute("SET LOCAL statement_timeout = '5s'")
        rows = conn.execute(
            f"""WITH source AS (
                SELECT EXTRACT(EPOCH FROM ts)::double precision AS timestamp,
                       CASE resolution WHEN '1h' THEN 3600 WHEN '5m' THEN 300 ELSE 60 END AS weight,
                       {expressions}
                FROM system_metrics
                WHERE resolution IN ('raw', '5m', '1h') AND ts >= %(start)s AND ts <= %(end)s
            )
            SELECT MIN(timestamp) AS timestamp, {averages}, COUNT(*) AS source_points
            FROM source
            GROUP BY LEAST(FLOOR((timestamp - %(epoch)s) / %(width)s)::int, %(last)s)
            ORDER BY timestamp""",
            {
                "start": start,
                "end": end,
                "epoch": start.timestamp(),
                "width": seconds / max_points,
                "last": max_points - 1,
            },
        ).fetchall()

    def number(value, digits):
        if value is None:
            return None
        parsed = float(value)
        return round(parsed, digits) if math.isfinite(parsed) else None

    return {
        "range": time_range,
        "from": start.timestamp(),
        "to": end.timestamp(),
        "timestamps": [float(row["timestamp"]) for row in rows],
        "series": {
            key: [number(row[key], 6 if key.startswith("net_") else 3) for row in rows]
            for key in _METRIC_SQL
        },
        "count": len(rows),
        "source_count": sum(row["source_points"] for row in rows),
    }
