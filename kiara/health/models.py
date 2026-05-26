"""Datenmodelle und Utilities fuer das Health-Check-System."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from kiara.shared.enums import HealthStatus

logger = logging.getLogger(__name__)


@dataclass
class ComponentHealth:
    """Gesundheitsstatus einer einzelnen Komponente."""

    name: str
    status: str  # HealthStatus-Wert
    latency_ms: float = 0.0
    message: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Aggregierter Gesundheitsstatus des Gesamtsystems."""

    status: str  # HealthStatus-Wert
    components: list[ComponentHealth] = field(default_factory=list)
    timestamp: str = ""
    uptime_seconds: float = 0.0
    version: str = ""


def timed_check(name: str, check_fn: Callable[[], ComponentHealth]) -> ComponentHealth:
    """Fuehrt einen Health-Check mit Zeitmessung und Fehlerbehandlung aus.

    Args:
        name:     Komponentenname (z.B. "database", "ollama").
        check_fn: Funktion, die den eigentlichen Check durchfuehrt
                  und ein ``ComponentHealth`` zurueckgibt.

    Returns:
        ``ComponentHealth`` mit korrekter ``latency_ms``, auch bei Exceptions.
    """
    start = time.time()
    try:
        result = check_fn()
        result.latency_ms = round((time.time() - start) * 1000, 1)
        return result
    except Exception as exc:
        latency = round((time.time() - start) * 1000, 1)
        logger.error("Health-Check %s Fehler: %s", name, exc)
        return ComponentHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            latency_ms=latency,
            message=f"Fehler: {exc}",
        )
