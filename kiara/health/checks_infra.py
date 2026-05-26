"""Infrastruktur Health-Checks (Database, Disk)."""

import asyncio
import logging
import os
import shutil
import time

from kiara.config import get_vectordb_data_path
from kiara.health.models import ComponentHealth
from kiara.shared.constants import DISK_CRIT_BYTES, DISK_WARN_BYTES
from kiara.shared.db import db_query
from kiara.shared.enums import HealthStatus

logger = logging.getLogger(__name__)


def _check_database_sync(config: dict) -> ComponentHealth:
    """Synchroner DB-Check (wird via asyncio.to_thread aufgerufen)."""
    start = time.time()

    try:
        with db_query() as conn:
            # Statement-Timeout setzen damit der Thread bei haengender DB
            # nicht endlos laeuft (asyncio.wait_for cancelt nur die Coroutine,
            # nicht den zugrunde liegenden Thread von asyncio.to_thread)
            conn.execute("SET LOCAL statement_timeout = '8000'")
            # Verify we can read tables
            tables_sql = (
                "SELECT tablename AS name FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
            )
            tables_raw = conn.execute(tables_sql).fetchall()
            table_names = [r["name"] for r in tables_raw]
        latency = (time.time() - start) * 1000

        if len(table_names) > 0:
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 1),
                message=f"DB OK, {len(table_names)} Tabellen",
                details={"backend": "postgresql", "table_count": len(table_names)},
            )
        else:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency, 1),
                message="DB lesbar, aber keine Tabellen vorhanden",
                details={"backend": "postgresql", "table_count": 0},
            )

    except Exception as exc:
        latency = (time.time() - start) * 1000
        logger.error("Database Health-Check Fehler: %s", exc)
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 1),
            message=f"DB nicht lesbar: {exc}",
            details={"backend": "postgresql"},
        )


async def check_database(config: dict) -> ComponentHealth:
    """Datenbank pruefen.

    Healthy: DB lesbar, Tabellen vorhanden.
    Unhealthy: nicht lesbar.
    """
    return await asyncio.to_thread(_check_database_sync, config)


async def check_disk(config: dict) -> ComponentHealth:
    """Festplattenplatz pruefen.

    Healthy: >1GB frei. Degraded: <1GB. Unhealthy: <100MB.
    Basiert auf dem Verzeichnis von vectordb.path.
    """
    start = time.time()
    vdb_path = get_vectordb_data_path(config)

    configured_path = vdb_path
    configured_exists = os.path.exists(configured_path)
    parent_path = os.path.dirname(configured_path) or "/"
    parent_exists = os.path.exists(parent_path)
    fallback_used = False

    if configured_exists:
        check_path = configured_path
    elif parent_exists:
        check_path = parent_path
        fallback_used = True
    else:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="disk",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 1),
            message=f"Storage-Pfad fehlt: {configured_path}",
            details={
                "configured_path": configured_path,
                "path": parent_path,
                "path_exists": False,
                "parent_exists": False,
                "fallback_used": False,
            },
        )

    health_cfg = config.get("health", {})
    warn_bytes = int(health_cfg.get("disk_warning_gb", DISK_WARN_BYTES / (1024**3)) * (1024**3))
    crit_bytes = int(health_cfg.get("disk_critical_gb", DISK_CRIT_BYTES / (1024**3)) * (1024**3))

    try:
        usage = shutil.disk_usage(check_path)
        free_bytes = usage.free
        free_gb = free_bytes / (1024**3)
        total_gb = usage.total / (1024**3)
        used_pct = ((usage.total - usage.free) / usage.total) * 100 if usage.total > 0 else 0

        details = {
            "configured_path": configured_path,
            "path": check_path,
            "path_exists": configured_exists,
            "parent_exists": parent_exists,
            "fallback_used": fallback_used,
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_percent": round(used_pct, 1),
        }

        latency = (time.time() - start) * 1000

        if free_bytes < crit_bytes:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency, 1),
                message=f"Kritisch wenig Speicher: {free_gb:.2f} GB frei",
                details=details,
            )
        elif free_bytes < warn_bytes:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 1),
                message=f"Wenig Speicher: {free_gb:.2f} GB frei",
                details=details,
            )
        elif fallback_used:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 1),
                message=(
                    f"Storage-Pfad fehlt: {configured_path}; Parent hat {free_gb:.2f} GB frei"
                ),
                details=details,
            )
        else:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 1),
                message=f"Speicher OK: {free_gb:.2f} GB frei ({used_pct:.1f}% belegt)",
                details=details,
            )

    except Exception as exc:
        latency = (time.time() - start) * 1000
        logger.error("Disk Health-Check Fehler: %s", exc)
        return ComponentHealth(
            name="disk",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 1),
            message=f"Festplatten-Check fehlgeschlagen: {exc}",
            details={
                "configured_path": configured_path,
                "path": check_path,
                "path_exists": configured_exists,
                "parent_exists": parent_exists,
                "fallback_used": fallback_used,
            },
        )
