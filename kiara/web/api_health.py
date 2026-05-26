"""
KIara Health API-Router.

Stellt unauthentifizierte Health-Endpoints fuer Monitoring-Tools bereit:
- /api/health     - Voller Systemstatus
- /api/health/ready - Readiness-Probe (Ollama + VektorDB)
- /api/health/live  - Liveness-Probe (Prozess + DB)
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import threading
from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from kiara.config import load_config
from kiara.repositories.health import HealthLogRepository
from kiara.shared.constants import HEALTH_BACKGROUND_LOCK_PATH
from kiara.shared.enums import HealthStatus

if TYPE_CHECKING:
    from kiara.health.checker import HealthChecker
    from kiara.resilience.degradation import DegradationManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

health_log_repo = HealthLogRepository()


def _get_health_checker() -> HealthChecker:
    """Erstellt eine neue HealthChecker-Instanz mit aktueller Config."""
    from kiara.health.checker import HealthChecker

    config = load_config()
    return HealthChecker(config)


_degradation_manager: DegradationManager | None = None
_background_started: bool = False
_background_lock_fh = None
_background_lock = asyncio.Lock()
_init_lock = threading.Lock()


def _get_degradation_manager() -> DegradationManager:
    """Gibt die Singleton-Instanz des DegradationManagers zurueck.

    Nutzt Double-Checked-Locking mit threading.Lock, damit parallele
    Health-Probes (sync- oder thread-pool-Aufrufer) nicht mehrere
    unabhaengige Manager-Instanzen erzeugen koennen.
    """
    global _degradation_manager
    if _degradation_manager is None:
        with _init_lock:
            if _degradation_manager is None:
                from kiara.resilience.degradation import DegradationManager

                config = load_config()
                checker = _get_health_checker()
                _degradation_manager = DegradationManager(config, checker)
    return _degradation_manager


async def ensure_background_checks() -> DegradationManager:
    """Startet Background-Checks beim ersten Aufruf. Idempotent."""
    global _background_lock_fh, _background_started
    dm = _get_degradation_manager()
    async with _background_lock:
        if not _background_started:
            try:
                os.makedirs(os.path.dirname(HEALTH_BACKGROUND_LOCK_PATH), exist_ok=True)
                _background_lock_fh = open(HEALTH_BACKGROUND_LOCK_PATH, "a")
                fcntl.flock(_background_lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if _background_lock_fh is not None:
                    _background_lock_fh.close()
                    _background_lock_fh = None
                logger.info("Health-Background-Checks laufen bereits in einem anderen Prozess")
                _background_started = True
                return dm
            except OSError as exc:
                logger.warning(
                    "Health-Background-Lock nicht verfuegbar; starte lokalen Loop: %s",
                    exc,
                )
            await dm.start_background_checks()
            _background_started = True
    return dm


async def shutdown_degradation_manager() -> None:
    """Stoppt Background-Checks. Aufruf aus Lifespan-Cleanup."""
    global _background_lock_fh, _background_started
    async with _background_lock:
        if _degradation_manager is not None and _background_started:
            await _degradation_manager.stop_background_checks()
            _background_started = False
        if _background_lock_fh is not None:
            try:
                fcntl.flock(_background_lock_fh.fileno(), fcntl.LOCK_UN)
                _background_lock_fh.close()
            except OSError:
                pass
            _background_lock_fh = None


@router.get("/api/health")
async def health_check():
    """Vollstaendiger Systemstatus aller Komponenten.

    Liest den gecachten Status aus dem Background-Check-Loop.
    Nur beim allerersten Aufruf (bevor der Loop lief) wird ein
    einmaliger Check ausgefuehrt.

    Returns:
        200: System healthy oder degraded.
        503: System unhealthy.
    """
    dm = _get_degradation_manager()

    # Nur initialen Check, wenn Background-Loop noch keinen gemacht hat.
    # ensure_initial_check ist race-safe (Double-Checked Locking) — mehrere
    # parallele Erstaufrufe fuehren genau einen Check aus.
    try:
        await dm.ensure_initial_check()
    except Exception as exc:
        logger.warning("Initialer Health-Check fehlgeschlagen: %s", exc)

    health = dm.last_health
    if health is None:
        return JSONResponse(status_code=503, content={"status": "unknown"})

    result = asdict(health)
    result["degradation_mode"] = dm.mode

    # Components von Liste zu Dict konvertieren (Frontend erwartet Dict mit Name als Key)
    if isinstance(result.get("components"), list):
        result["components"] = {c["name"]: c for c in result["components"]}

    status_code = 200 if health.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED) else 503

    # Health-Log schreiben
    try:
        health_log_repo.create(
            status=health.status,
            components_json=json.dumps(result["components"]),
            degradation_mode=dm.mode,
        )
    except Exception as exc:
        logger.warning("Health-Log schreiben fehlgeschlagen: %s", exc)

    return JSONResponse(content=result, status_code=status_code)


@router.get("/api/health/degradation-mode")
async def degradation_mode():
    """Leichtgewichtiger Endpoint fuer Frontend-Polling.

    Kein Health-Check-Trigger — liest nur den letzten bekannten Mode.
    """
    dm = _get_degradation_manager()
    mode = dm.mode
    messages = {
        "FULL": "",
        "NO_LLM": "LLM (Ollama) ist nicht erreichbar. Chat-Funktionen sind eingeschraenkt.",
        "NO_CLOUD": "Cloud-KI ist nicht erreichbar. Nur lokales LLM verfuegbar.",
        "NO_ERP": "ERP-System ist nicht erreichbar. ERP-Tools sind deaktiviert.",
        "READONLY": "Datenbank-, Speicher- oder VektorDB-Probleme. Nur Lesezugriff moeglich.",
        "MAINTENANCE": "Wartungsmodus aktiv. Alle Features sind deaktiviert.",
    }
    return {"mode": mode, "message": messages.get(mode, "")}


@router.get("/api/health/ready")
async def readiness_check():
    """Readiness-Probe: Prueft ob Ollama und VektorDB bereit sind.

    200 wenn beide Komponenten healthy, sonst 503.
    Wird von Load-Balancern und Orchestrierungstools genutzt.
    """
    checker = _get_health_checker()

    ollama = await checker._run_component_check(
        "ollama",
        checker.check_ollama(),
        timeout=10.0,
    )
    vectordb_health = await checker._run_component_check(
        "vectordb",
        checker.check_vectordb(),
        timeout=10.0,
    )

    ollama_dict = asdict(ollama)
    vectordb_dict = asdict(vectordb_health)

    ready = ollama.status == HealthStatus.HEALTHY and vectordb_health.status == HealthStatus.HEALTHY
    status_code = 200 if ready else 503

    return JSONResponse(
        content={
            "ready": ready,
            "components": {
                "ollama": ollama_dict,
                "vectordb": vectordb_dict,
            },
        },
        status_code=status_code,
    )


@router.get("/api/health/live")
async def liveness_check():
    """Liveness-Probe: Prueft ob der Prozess lebt.

    200 wenn der Prozess laeuft und die DB mindestens lesbar ist.
    503 nur wenn die DB korrupt ist.
    """
    checker = _get_health_checker()
    db_health = await checker.check_database()
    db_dict = asdict(db_health)

    alive = db_health.status != HealthStatus.UNHEALTHY
    status_code = 200 if alive else 503

    return JSONResponse(
        content={
            "alive": alive,
            "database": db_dict,
        },
        status_code=status_code,
    )
