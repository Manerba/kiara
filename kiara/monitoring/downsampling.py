"""Downsampling fuer System-Metriken.

3-Stufen-Komprimierung + Trace-Cleanup:
  raw (1/min)  ──[>24h]──▶  5min-AVG  ──[>30d]──▶  1h-AVG  ──[>1y]──▶  DELETE

Laeuft einmal taeglich als Background-Task (Standard: 03:00 Uhr).
Alle Operationen sind idempotent — wenn keine passenden Daten vorhanden,
wird nichts eingefuegt und nichts geloescht.
"""

import logging

from kiara.shared.constants import (
    METRICS_1H_RETENTION_DAYS,
    METRICS_5MIN_RETENTION_DAYS,
    METRICS_RAW_RETENTION_HOURS,
)
from kiara.shared.db import db_transaction

log = logging.getLogger(__name__)

# Spalten die aggregiert werden (AVG fuer Raten/Prozente, MAX fuer Totals)
_AGG_COLS = """
    AVG(cpu_percent)            AS cpu_percent,
    AVG(ram_used_bytes)::bigint AS ram_used_bytes,
    MAX(ram_total_bytes)        AS ram_total_bytes,
    AVG(swap_used_bytes)::bigint AS swap_used_bytes,
    MAX(swap_total_bytes)       AS swap_total_bytes,
    AVG(load_1m)  AS load_1m,
    AVG(load_5m)  AS load_5m,
    AVG(load_15m) AS load_15m,
    AVG(net_sent_bps) AS net_sent_bps,
    AVG(net_recv_bps) AS net_recv_bps,
    MAX(disk_used_bytes)  AS disk_used_bytes,
    MAX(disk_total_bytes) AS disk_total_bytes,
    AVG(ram_percent) AS ram_percent,
    AVG(disk_read_bps) AS disk_read_bps,
    AVG(disk_write_bps) AS disk_write_bps
"""

_INSERT_COLS = """ts, resolution, cpu_percent, ram_used_bytes, ram_total_bytes,
    swap_used_bytes, swap_total_bytes, load_1m, load_5m, load_15m,
    net_sent_bps, net_recv_bps, disk_used_bytes, disk_total_bytes,
    ram_percent, disk_read_bps, disk_write_bps"""


def _load_retention_config() -> tuple[int, int, int]:
    """Liest monitoring.* Retention-Werte aus Runtime-Config (Fallback: Konstanten)."""
    try:
        from kiara.config import load_config

        cfg = load_config().get("monitoring", {}) or {}
    except Exception:
        cfg = {}
    return (
        int(cfg.get("raw_retention_hours", METRICS_RAW_RETENTION_HOURS)),
        int(cfg.get("retention_5min_days", METRICS_5MIN_RETENTION_DAYS)),
        int(cfg.get("retention_1h_days", METRICS_1H_RETENTION_DAYS)),
    )


def run_downsampling() -> dict:
    """Fuehrt alle Downsampling-/Cleanup-Schritte aus. Gibt Statistik zurueck.

    Jede Phase laeuft isoliert: Faellt eine Phase aus, laufen die anderen
    weiter. Fehler werden geloggt und im Stats-Dict als '<phase>_error' vermerkt.
    Wichtig fuer Retention-Policy: ein transient fehlgeschlagener Schritt
    darf den taeglichen Cleanup der anderen Tabellen nicht blockieren.
    """
    raw_h, r5_d, r1h_d = _load_retention_config()
    stats: dict = {}
    phases = (
        ("raw_to_5m", _compress_raw_to_5min, (raw_h,)),
        ("5m_to_1h", _compress_5min_to_1h, (r5_d,)),
        ("deleted_old", _delete_expired, (r1h_d,)),
        ("traces_deleted", _cleanup_traces, ()),
    )
    for key, fn, args in phases:
        try:
            stats[key] = fn(*args)
        except Exception as exc:
            log.error("Downsampling-Phase %s fehlgeschlagen: %s", key, exc, exc_info=True)
            stats[key] = 0
            stats[f"{key}_error"] = str(exc)
    log.info("Downsampling abgeschlossen: %s", stats)
    return stats


def _compress_raw_to_5min(retention_hours: int = METRICS_RAW_RETENTION_HOURS) -> int:
    """Komprimiert raw-Daten aelter als retention_hours in 5-Minuten-Buckets."""
    with db_transaction() as conn:
        conn.execute(
            f"""
            INSERT INTO system_metrics ({_INSERT_COLS})
            SELECT
                date_trunc('hour', ts)
                    + INTERVAL '5 min' * (EXTRACT(MINUTE FROM ts)::int / 5)
                    AS bucket,
                '5m',
                {_AGG_COLS}
            FROM system_metrics
            WHERE resolution = 'raw'
              AND ts < NOW() - (%s * INTERVAL '1 hour')
            GROUP BY bucket
        """,
            (retention_hours,),
        )

        result = conn.execute(
            """
            DELETE FROM system_metrics
            WHERE resolution = 'raw'
              AND ts < NOW() - (%s * INTERVAL '1 hour')
        """,
            (retention_hours,),
        )
        deleted = result.rowcount if hasattr(result, "rowcount") else 0
    return deleted


def _compress_5min_to_1h(retention_days: int = METRICS_5MIN_RETENTION_DAYS) -> int:
    """Komprimiert 5min-Daten aelter als retention_days in 1-Stunden-Buckets."""
    with db_transaction() as conn:
        conn.execute(
            f"""
            INSERT INTO system_metrics ({_INSERT_COLS})
            SELECT
                date_trunc('hour', ts) AS bucket,
                '1h',
                {_AGG_COLS}
            FROM system_metrics
            WHERE resolution = '5m'
              AND ts < NOW() - (%s * INTERVAL '1 day')
            GROUP BY bucket
        """,
            (retention_days,),
        )

        result = conn.execute(
            """
            DELETE FROM system_metrics
            WHERE resolution = '5m'
              AND ts < NOW() - (%s * INTERVAL '1 day')
        """,
            (retention_days,),
        )
        deleted = result.rowcount if hasattr(result, "rowcount") else 0
    return deleted


def _delete_expired(retention_days: int = METRICS_1H_RETENTION_DAYS) -> int:
    """Loescht 1h-Daten aelter als retention_days."""
    with db_transaction() as conn:
        result = conn.execute(
            """
            DELETE FROM system_metrics
            WHERE resolution = '1h'
              AND ts < NOW() - (%s * INTERVAL '1 day')
        """,
            (retention_days,),
        )
        deleted = result.rowcount if hasattr(result, "rowcount") else 0
    return deleted


def _cleanup_traces() -> int:
    """Loescht Traces aelter als retention_days (Phase 17)."""
    from kiara.shared.constants import TRACING_DEFAULT_RETENTION_DAYS

    try:
        from kiara.config import load_config

        cfg = load_config()
        days = cfg.get("tracing", {}).get("retention_days", TRACING_DEFAULT_RETENTION_DAYS)
    except Exception:
        days = TRACING_DEFAULT_RETENTION_DAYS

    try:
        with db_transaction() as conn:
            result = conn.execute(
                "DELETE FROM traces WHERE created_at < NOW() - (%s * INTERVAL '1 day')",
                (days,),
            )
            deleted = result.rowcount if hasattr(result, "rowcount") else 0
        return deleted
    except Exception as exc:
        log.debug("Trace-Cleanup uebersprungen (Tabelle existiert ggf. noch nicht): %s", exc)
        return 0
