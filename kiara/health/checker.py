"""
KIara Health-Check System.

Prueft alle Systemkomponenten (Ollama, VektorDB, PostgreSQL, LDAP, Cloud-KI,
ERP, Festplatte) und liefert einen aggregierten Gesundheitsstatus.

Die einzelnen Checks sind in Submodule aufgeteilt:
- checks_vectordb: Qdrant
- checks_services: Ollama, LDAP, Cloud-KI, ERP
- checks_infra: Database, Disk

Beziehung zum DegradationManager:
    HealthChecker liefert den Ist-Zustand (SystemHealth), der
    DegradationManager (kiara.resilience.degradation) interpretiert ihn
    und leitet daraus den Betriebsmodus ab (FULL, NO_CLOUD, NO_LLM, ...).
    DegradationManager haelt eine HealthChecker-Instanz und ruft
    periodisch check_all() auf, um den Modus ggf. anzupassen.

Config-Reloading:
    _reload_config() wird vor jedem check_all() aufgerufen und laedt
    die YAML-Config frisch von Disk (/etc/kiara/kiara.yaml). Dadurch
    wirken Aenderungen in der Admin-UI (z.B. neuer Ollama-Host, geaenderte
    LDAP-Config) sofort beim naechsten Health-Check, ohne dass der Service
    neugestartet werden muss. Bei Ladefehler wird die bisherige Config
    beibehalten (fail-safe).
"""

import asyncio
import logging
import time
from datetime import datetime

from kiara.health.checks_infra import check_database, check_disk
from kiara.health.checks_services import (
    check_cloud_ki,
    check_colbert,
    check_erp,
    check_ldap,
    check_ollama,
    check_reranker,
)
from kiara.health.checks_vectordb import check_vectordb
from kiara.health.models import ComponentHealth, SystemHealth
from kiara.shared.enums import HealthStatus

logger = logging.getLogger(__name__)


class HealthChecker:
    """Fuehrt Health-Checks fuer alle KIara-Komponenten durch.

    Wird vom DegradationManager (kiara.resilience.degradation) gehalten
    und periodisch abgefragt. Kann auch standalone verwendet werden
    (z.B. vom /health API-Endpoint).

    Alle Einzelchecks laufen parallel via asyncio.gather. Exceptions
    aus einzelnen Checks werden als 'unhealthy'-Komponenten behandelt,
    damit ein fehlgeschlagener Check die anderen nicht blockiert.
    """

    def __init__(self, config: dict):
        self._config = config
        self._start_time = time.monotonic()

    def _reload_config(self) -> None:
        """Laedt die Config frisch von Disk (/etc/kiara/kiara.yaml).

        Wird vor jedem check_all() aufgerufen, damit Aenderungen in der
        Admin-UI (z.B. neuer Ollama-Host) sofort beim naechsten Check
        wirksam werden, ohne Service-Restart.

        Fail-safe: Bei Ladefehler wird die bisherige Config beibehalten.
        """
        try:
            from kiara.config import load_config

            self._config = load_config()
        except Exception as exc:
            logger.warning("Config-Reload fehlgeschlagen, alte Config bleibt aktiv: %s", exc)

    async def _run_component_check(
        self,
        name: str,
        check_coro,
        *,
        timeout: float,
    ) -> ComponentHealth:
        """Fuehrt einen einzelnen Check mit Timeout, Timing und Fehler-Capture aus."""
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(check_coro, timeout=timeout)
            latency = round((time.monotonic() - start) * 1000, 1)
            if not result.latency_ms:
                result.latency_ms = latency
            return result
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round((time.monotonic() - start) * 1000, 1),
                message=f"Timeout nach {timeout:g}s",
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=round((time.monotonic() - start) * 1000, 1),
                message=f"Fehler: {message}",
            )

    async def check_all(self) -> SystemHealth:
        """Prueft alle Komponenten parallel via asyncio.gather.

        Gesamtstatus: healthy wenn alle healthy/unconfigured,
        degraded wenn mind. eine degraded,
        unhealthy wenn mind. eine unhealthy.
        """
        self._reload_config()
        config = self._config  # Snapshot — alle Checks sehen dieselbe Config
        per_check_timeout = 10.0
        checks = [
            self._run_component_check("ollama", check_ollama(config), timeout=per_check_timeout),
            self._run_component_check(
                "vectordb", check_vectordb(config), timeout=per_check_timeout
            ),
            self._run_component_check(
                "database", check_database(config), timeout=per_check_timeout
            ),
            self._run_component_check("ldap", check_ldap(config), timeout=per_check_timeout),
            self._run_component_check(
                "cloud_ki", check_cloud_ki(config), timeout=per_check_timeout
            ),
            self._run_component_check("erp", check_erp(config), timeout=per_check_timeout),
            self._run_component_check(
                "reranker", check_reranker(config), timeout=per_check_timeout
            ),
            self._run_component_check("colbert", check_colbert(config), timeout=per_check_timeout),
            self._run_component_check("disk", check_disk(config), timeout=per_check_timeout),
        ]
        resolved = await asyncio.gather(*checks)

        # Determine overall status (unconfigured counts as healthy)
        statuses = [c.status for c in resolved]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        from kiara import __version__

        return SystemHealth(
            status=overall,
            components=resolved,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=round(time.monotonic() - self._start_time, 1),
            version=__version__,
        )

    async def check_ollama(self) -> ComponentHealth:
        """Delegiert an checks_services.check_ollama."""
        return await check_ollama(self._config)

    async def check_vectordb(self) -> ComponentHealth:
        """Delegiert an checks_vectordb.check_vectordb."""
        return await check_vectordb(self._config)

    async def check_database(self) -> ComponentHealth:
        """Delegiert an checks_infra.check_database."""
        return await check_database(self._config)

    async def check_ldap(self) -> ComponentHealth:
        """Delegiert an checks_services.check_ldap."""
        return await check_ldap(self._config)

    async def check_cloud_ki(self) -> ComponentHealth:
        """Delegiert an checks_services.check_cloud_ki."""
        return await check_cloud_ki(self._config)

    async def check_erp(self) -> ComponentHealth:
        """Delegiert an checks_services.check_erp."""
        return await check_erp(self._config)

    async def check_reranker(self) -> ComponentHealth:
        """Delegiert an checks_services.check_reranker."""
        return await check_reranker(self._config)

    async def check_colbert(self) -> ComponentHealth:
        """Delegiert an checks_services.check_colbert."""
        return await check_colbert(self._config)

    async def check_disk(self) -> ComponentHealth:
        """Delegiert an checks_infra.check_disk."""
        return await check_disk(self._config)
