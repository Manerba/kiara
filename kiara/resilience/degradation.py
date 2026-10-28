"""
KIara Graceful-Degradation Manager.

Verwaltet den Betriebsmodus basierend auf dem Gesundheitsstatus
der Systemkomponenten. Modi: FULL, NO_CLOUD, NO_LLM, NO_ERP,
READONLY, MAINTENANCE.
"""

import asyncio
import logging
import threading

from kiara.health.checker import HealthChecker, SystemHealth
from kiara.shared.enums import HealthStatus

logger = logging.getLogger(__name__)


class DegradationManager:
    """Steuert den Betriebsmodus von KIara basierend auf Health-Checks.

    Modi und verfuegbare Features:
    - FULL: Alle Features (cloud, llm, erp_tools, logging, full)
    - NO_CLOUD: Ohne Cloud-KI (llm, erp_tools, logging)
    - NO_LLM: Ohne LLM (cloud, erp_tools, logging)
    - NO_ERP: Ohne ERP-Tools (cloud, llm, logging)
    - READONLY: Nur Leseoperationen (cloud, llm, erp_tools)
    - MAINTENANCE: Wartungsmodus, keine Features verfuegbar
    """

    # Mapping Betriebsmodus -> verfuegbare Features.
    # Jeder Modus deaktiviert bestimmte Subsysteme:
    #   FULL:        Alles verfuegbar (Normalbetrieb)
    #   NO_CLOUD:    Cloud-KI ausgefallen → nur lokales LLM + ERP + Logging
    #   NO_LLM:      Ollama ausgefallen → Cloud-KI + ERP-Tools + Logging (kein lokales LLM)
    #   NO_ERP:      ERP-Backend ausgefallen → Cloud + LLM + Logging, keine ERP-Tools
    #   READONLY:    Persistenz-Probleme -> kein Logging/Schreiben, Lese-Features bleiben
    #   MAINTENANCE: Manuell gesetzt → alle Features deaktiviert
    FEATURE_MAP: dict[str, set[str]] = {
        "FULL": {"cloud", "llm", "erp_tools", "logging", "full"},
        "NO_CLOUD": {"llm", "erp_tools", "logging"},
        "NO_LLM": {"cloud", "erp_tools", "logging"},
        "NO_ERP": {"cloud", "llm", "logging"},
        "READONLY": {"cloud", "llm", "erp_tools"},
        "MAINTENANCE": set(),
    }

    # Mapping: Komponente -> Features die bei Ausfall deaktiviert werden.
    # Wird von is_feature_available() genutzt um bei Mehrfachausfaellen
    # korrekte Feature-Flags zu liefern (nicht nur den priorisierten Modus).
    _COMPONENT_DISABLES: dict[str, set[str]] = {
        "database": {"logging", "full"},
        "vectordb": {"full"},
        "disk": {"logging", "full"},
        "ollama": {"llm", "full"},
        "cloud_ki": {"cloud", "full"},
        "erp": {"erp_tools", "full"},
    }

    def __init__(self, config: dict, health_checker: HealthChecker):
        self._mode = "FULL"
        self._config = config
        self._health_checker = health_checker
        self._check_interval = config.get("health", {}).get("check_interval_seconds", 60)
        self._background_task: asyncio.Task | None = None
        self._last_health: SystemHealth | None = None
        self._fail_counts: dict[str, int] = {}
        self._down_components: set[str] = set()
        self._update_lock = asyncio.Lock()
        self._state_lock = threading.RLock()

        # Welche HealthStatus-Werte als "fehlgeschlagen" zaehlen (pro Komponente).
        # database/ollama/vectordb nur bei UNHEALTHY, cloud_ki/erp/disk
        # auch bei DEGRADED.
        self._trigger_statuses: dict[str, set] = {
            "disk": {HealthStatus.UNHEALTHY, HealthStatus.DEGRADED},
            "cloud_ki": {HealthStatus.UNHEALTHY, HealthStatus.DEGRADED},
            "erp": {HealthStatus.UNHEALTHY, HealthStatus.DEGRADED},
        }
        self._default_trigger = {HealthStatus.UNHEALTHY}

    @property
    def mode(self) -> str:
        """Aktueller Betriebsmodus."""
        with self._state_lock:
            return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        """Setzt den Betriebsmodus (nur gueltige Modi akzeptiert)."""
        if value in self.FEATURE_MAP:
            with self._state_lock:
                old = self._mode
                self._mode = value
            if old != value:
                logger.warning("Degradation-Modus geaendert: %s -> %s", old, value)
        else:
            logger.warning("Ungueltiger Degradation-Modus ignoriert: %s", value)

    @property
    def last_health(self) -> SystemHealth | None:
        """Letztes Health-Check-Ergebnis."""
        return self._last_health

    async def update_status(self) -> None:
        """Fuehrt Health-Check durch und setzt Modus basierend auf Ergebnissen.

        Hysterese: Eine Komponente gilt erst nach 2 konsekutiven Fehlern als
        ausgefallen. Recovery nach 1 Erfolg.
        MAINTENANCE-Modus wird nicht automatisch geaendert (manueller Modus).
        Serialisiert mit asyncio.Lock gegen parallele Aufrufe (Race-Condition
        auf _fail_counts/_mode).
        """
        async with self._update_lock:
            await self._update_status_locked()

    async def ensure_initial_check(self) -> None:
        """Fuehrt einen Health-Check aus, falls noch keiner gemacht wurde.

        Race-safe: Holt _update_lock und prueft _last_health innerhalb des Locks
        erneut (Double-Checked Locking). Mehrere parallele Erstaufrufe fuehren
        damit genau einen Check aus, anstatt die Hysterese-Counter zu vervielfachen.
        """
        async with self._update_lock:
            if self._last_health is not None:
                return
            await self._update_status_locked()

    async def _update_status_locked(self) -> None:
        """Eigentliche Update-Logik. Aufrufer MUSS _update_lock halten."""
        health = await self._health_checker.check_all()
        self._last_health = health

        # Hysterese-Counter aktualisieren (per Komponente)
        for comp in health.components:
            triggers = self._trigger_statuses.get(comp.name, self._default_trigger)
            if comp.status in triggers:
                self._fail_counts[comp.name] = self._fail_counts.get(comp.name, 0) + 1
            else:
                self._fail_counts[comp.name] = 0

        def _is_down(name: str) -> bool:
            """Erst ab 2 konsekutiven Fehlern als ausgefallen werten."""
            return self._fail_counts.get(name, 0) >= 2

        # MAINTENANCE ist ein manueller Modus, wird nicht auto-geaendert
        if self._mode == "MAINTENANCE":
            return

        old_mode = self._mode
        down_components = {name for name in self._COMPONENT_DISABLES if _is_down(name)}
        with self._state_lock:
            self._down_components = down_components

            # Modus bestimmen basierend auf Hysterese-gefiltertem Status
            if _is_down("database") or _is_down("vectordb") or _is_down("disk"):
                self.mode = "READONLY"
            elif _is_down("ollama"):
                self.mode = "NO_LLM"
            elif _is_down("cloud_ki"):
                self.mode = "NO_CLOUD"
            elif _is_down("erp"):
                self.mode = "NO_ERP"
            else:
                self.mode = "FULL"

        # Cache invalidieren bei JEDEM Mode-Wechsel (stale Ergebnisse vermeiden)
        if old_mode != self._mode:
            try:
                from kiara.shared.cache import get_query_cache

                get_query_cache().invalidate()
            except Exception as exc:
                logger.warning("Cache-Invalidierung bei Mode-Wechsel fehlgeschlagen: %s", exc)

        # Mail-Notification NUR bei Wechsel von FULL weg (non-blocking im Thread-Pool)
        if old_mode == "FULL" and self._mode != "FULL":
            asyncio.get_running_loop().run_in_executor(
                None, self._notify_mode_change, old_mode, self._mode
            )

    def is_feature_available(self, feature: str) -> bool:
        """Prueft ob ein Feature im aktuellen Modus verfuegbar ist.

        Beruecksichtigt ALLE ausgefallenen Komponenten, nicht nur den
        priorisierten Modus. Beispiel: ollama+cloud_ki down → weder
        "llm" noch "cloud" verfuegbar.

        Args:
            feature: Name des Features (z.B. "cloud", "llm", "erp_tools",
                     "logging", "full")

        Returns:
            True wenn das Feature im aktuellen Modus verfuegbar ist.
        """
        with self._state_lock:
            mode = self._mode
            down_components = set(self._down_components)

        available = self.FEATURE_MAP.get(mode, set())
        if feature not in available:
            return False
        # Zusaetzlich Features deaktivieren fuer ALLE ausgefallenen Komponenten
        for comp in down_components:
            if feature in self._COMPONENT_DISABLES.get(comp, set()):
                return False
        return True

    def _notify_mode_change(self, old_mode: str, new_mode: str) -> None:
        """Sendet Mail-Benachrichtigung bei Mode-Wechsel (best-effort).

        Liest die vollstaendige mail_config aus der DB (SMTP-Zugangsdaten +
        Empfaenger), entschluesselt das Passwort und ruft send_mail() auf.
        """
        try:
            from kiara.shared.crypto import decrypt, is_encrypted
            from kiara.shared.db import db_query
            from kiara.web.mail_service import send_mail

            with db_query() as db:
                row = db.execute("SELECT * FROM mail_config WHERE id = 1").fetchone()
            if not row or not row["notify_degradation"]:
                return

            password = row["password"]
            if password and is_encrypted(password):
                password = decrypt(password)

            mail_config = {
                "smtp_server": row["smtp_server"],
                "smtp_port": row["smtp_port"],
                "encryption": row["encryption"],
                "username": row["username"],
                "password": password,
                "sender_address": row["sender_address"],
                "recipient_addresses": row["recipient_addresses"],
            }

            subject = f"KIara Degradation: {old_mode} -> {new_mode}"
            body = (
                f"<h3>KIara Mode-Wechsel</h3>"
                f"<p><b>Alter Modus:</b> {old_mode}</p>"
                f"<p><b>Neuer Modus:</b> {new_mode}</p>"
                f"<p>Bitte pruefen Sie den Systemstatus in der Admin-UI.</p>"
            )
            send_mail(mail_config, subject, body)
        except Exception as exc:
            logger.warning("Degradation-Mail konnte nicht gesendet werden: %s", exc)

    async def start_background_checks(self) -> None:
        """Startet periodische Health-Checks als asyncio-Task."""
        if self._background_task is not None:
            return
        self._background_task = asyncio.create_task(self._check_loop())

    async def stop_background_checks(self) -> None:
        """Stoppt die periodischen Health-Checks."""
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

    async def _check_loop(self) -> None:
        """Endlosschleife fuer periodische Health-Checks."""
        while True:
            try:
                await self.update_status()
            except Exception as exc:
                logger.error("Health-Check Fehler: %s", exc)
            await asyncio.sleep(self._check_interval)
