"""DB-Handler fuer strukturierte Security-Events.

Der Handler schreibt jeden `kiara.security`-Log-Record synchron in die
`system_events`-Tabelle. Standardbetrieb ist bewusst **fail-safe**: jede
Exception wird stumm geschluckt, damit ein DB-Ausfall nie einen Login- oder
API-Request platzen laesst. Fuer forensische CLI-Pfade kann der Handler im
strict mode laufen; dann propagieren DB-Fehler und der CLI-Aufruf liefert
rc!=0.

Parser-Vertrag: Messages wie `LOGIN_FAILED user=admin ip=192.0.2.10` werden
in `event_type` + KV-Details zerlegt. Unparsbare Messages landen mit
`event_type=NULL` und voller Message in `details.raw`.

Design-Vertrag: `message`, `details.raw` und `details.traceback` werden
absichtlich nicht anonymisiert oder redigiert gespeichert. Diese Tabelle ist
der forensische Audit-Pfad; Sanitizing muss, falls fachlich benoetigt, vor dem
Log-Aufruf passieren. Bugscans duerfen die Raw-Persistenz in diesem Handler
nicht als Fehler melden, solange diese Projektentscheidung gilt.
"""

from __future__ import annotations

import logging
import os
import re
import traceback
from typing import Any

_EVENT_TYPE_RE = re.compile(r"^([A-Z][A-Z0-9_]{2,})\b")
_KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def _parse_kv_message(msg: str) -> dict[str, Any]:
    """Zerlegt strukturierte Log-Messages in event_type + KV-Paare.

    - `event_type`: erstes UPPERCASE-Token (mind. 3 Zeichen) am Anfang.
    - `kv`: dict aller erkannten `key=value`- oder `key="quoted value"`-Paare.
    - `raw`: nicht als KV parsbarer Rest (nach event_type). Space-haltige
      unquoted Werte fallen hier rein. Leer, wenn alles sauber geparst wurde.
    """
    result: dict[str, Any] = {"event_type": None, "kv": {}, "raw": msg}
    if not msg:
        result["raw"] = msg
        return result

    event_match = _EVENT_TYPE_RE.match(msg)
    remainder = msg
    if event_match:
        result["event_type"] = event_match.group(1)
        remainder = msg[event_match.end() :].lstrip()

    kv: dict[str, str] = {}
    residual_parts: list[str] = []
    cursor = 0
    for match in _KV_RE.finditer(remainder):
        # Alles zwischen letztem Match-Ende und aktuellem Start ist Rest-Text.
        gap = remainder[cursor : match.start()].strip()
        if gap:
            residual_parts.append(gap)
        raw_val = match.group(2)
        if raw_val.startswith('"') and raw_val.endswith('"'):
            raw_val = raw_val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        kv[match.group(1)] = raw_val
        cursor = match.end()

    tail = remainder[cursor:].strip()
    if tail:
        residual_parts.append(tail)

    result["kv"] = kv
    if event_match and not residual_parts:
        result["raw"] = ""
    elif not event_match:
        # Keine event_type-Extraktion -> gesamte Message als forensischer Rest.
        result["raw"] = msg
    else:
        result["raw"] = " ".join(residual_parts)
    return result


def _write_system_event(
    *,
    level_name: str,
    source: str,
    message: str,
    extra_details: dict[str, Any] | None = None,
    exc_info=None,
) -> None:
    """Schreibt ein einzelnes system_events-Event und wirft bei Fehlern."""
    parsed = _parse_kv_message(message)
    kv = parsed["kv"]
    details: dict[str, Any] = {}
    for key, value in kv.items():
        if key in ("user", "ip", "reason"):
            continue
        details[key] = value
    if parsed["raw"]:
        details["raw"] = parsed["raw"]

    if exc_info:
        try:
            details["traceback"] = "".join(traceback.format_exception(*exc_info))
        except Exception:
            pass

    if isinstance(extra_details, dict):
        for key, value in extra_details.items():
            if value is None:
                continue
            details.setdefault(key, value)

    from psycopg2.extras import Json

    from kiara.shared.db import db_transaction

    with db_transaction() as conn:
        conn.execute(
            """
            INSERT INTO system_events
                (level, source, event_type, reason, user_name, ip, message, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                level_name,
                source,
                parsed["event_type"],
                kv.get("reason"),
                kv.get("user"),
                kv.get("ip"),
                message,
                Json(details),
            ),
        )


class SystemEventDBHandler(logging.Handler):
    """Haengt Security-Log-Records an die `system_events`-Tabelle.

    Standard: jede Exception im emit() wird geschluckt. Mit ``strict=True``
    propagieren DB-Fehler an den Aufrufer. Kein `logger.*`-Call, damit keine
    Rekursion entstehen kann (propagate=False auf kiara.security wird
    zusaetzlich in logging_config.setup_security_logger gesetzt).
    """

    def __init__(self, *, strict: bool = False) -> None:
        super().__init__()
        self._min_level_cache: int | None = None
        self.strict = strict

    def _resolve_min_level(self) -> int:
        """Liest `system_events.min_level` lazy und cached das Ergebnis."""
        if self._min_level_cache is not None:
            return self._min_level_cache
        try:
            from kiara.config import get_system_events_config, load_config

            level_name = get_system_events_config(load_config())["min_level"]
        except Exception:
            level_name = "INFO"
        level = logging.getLevelName(level_name)
        if not isinstance(level, int):
            level = logging.INFO
        self._min_level_cache = level
        return level

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Escape-Hatch fuer pytest-Selbsttest-Runner: verhindert Deadlocks,
            # wenn ein Test security_log.*-Events triggert und der Test-Prozess
            # keine stabile DB-Connection-Initialisierung hat (kiara-admin setzt
            # die Variable in api_testsuite._run_tests, damit der Prod-Service
            # den DB-Handler weiterhin voll aktiv nutzt).
            if os.environ.get("KIARA_SKIP_DB_LOG_HANDLER") == "1":
                if self.strict:
                    raise RuntimeError(
                        "Strict-SystemEventDBHandler ist per KIARA_SKIP_DB_LOG_HANDLER deaktiviert"
                    )
                return

            if not self.strict and record.levelno < self._resolve_min_level():
                return

            try:
                message = record.getMessage()
            except Exception:
                message = str(record.msg)

            source = record.name
            if source == "kiara.security":
                source = "security"

            level_name = record.levelname or logging.getLevelName(record.levelno)
            _write_system_event(
                level_name=level_name,
                source=source,
                message=message,
                extra_details=getattr(record, "kiara_details", None),
                exc_info=record.exc_info,
            )
        except Exception:
            if self.strict:
                raise
            # fail-safe: DB-Ausfall darf nie einen Request zum Platzen bringen.
            pass
