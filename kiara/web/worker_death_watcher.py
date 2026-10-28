"""Watcher-Service fuer uvicorn-Sub-Worker-Tode in KIara-Services.

Hintergrund:
    systemd's `OnFailure=`-Hook (vgl. kiara-failure-logger@.service) triggert
    nur bei abnormalem Tod des Master-Prozesses einer Unit. uvicorn-Workers
    sind Sub-Prozesse innerhalb einer einzigen Unit — wenn ein Worker stirbt,
    sieht systemd nichts (Master lebt), `OnFailure=` schweigt, und die
    Diagnose-Lücke ist offen.

    Der uvicorn-Master loggt jeden Worker-Tod via stdout-Linie der Form
    "INFO:     Child process [PID] died". Diese Lines landen in journald
    unter der Service-Unit. Dieser Watcher tailt journald, matched per Regex,
    und ruft pro Match den CLI-Befehl `kiara log-worker-death` auf — der den
    Tod als WORKER_DIED-Event in `system_events` persistiert und einen
    Kontext-Snapshot beilegt (dmesg, CIFS-Mount-Status, Memory, journal).

Multi-Service:
    Der Watcher ueberwacht alle KIara-Services parallel via Threading. Jeder
    Service bekommt einen eigenen journalctl-Tail-Subprocess in einem Daemon-
    Thread. Bei Beendigung (SIGTERM/SIGINT) werden alle Tails gestoppt.

Edge-Cases:
    - Service-Stop/Restart: Master killt Worker beim Stop. Heuristik:
      Wenn der Service laut `systemctl is-active` gestoppt wird, wird das
      Death-Event unterdrueckt.
    - Crash-Loops: Rate-Limit pro Service und Per-PID-Dedup verhindern
      Event-Flood, falls ein Worker im Loop respawnt.
    - Watcher-Restart: journalctl wird mit `--cursor`-Datei pro Service
      gestartet, damit gespeicherter Last-Position keine doppelten Events
      nach Restart erzeugt.
    - Trigger-Fail: Das betroffene Death-Event wird als Pending-State
      persistiert und vor jedem weiteren journald-Resume erneut versucht.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kiara.worker_death_watcher")

_TRACKED_SERVICES = (
    "kiara.service",
    "kiara-admin.service",
    "kiara-admin-chat.service",
    "kiara-worker.service",
    "kiara-mcp.service",
)
_DIED_RE = re.compile(r"Child process \[(\d+)\] died")
_STOPPING_RE = re.compile(r"\b(Stopping|Shutting down|SIGTERM)\b", re.IGNORECASE)
_CURSOR_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9_]*=[^;=\s]+)"
    r"(?:;[A-Za-z][A-Za-z0-9_]*=[^;=\s]+)*$"
)

_GLOBAL_RATE_WINDOW = 60.0
_GLOBAL_RATE_LIMIT = 6
_PER_PID_DEDUP_SECONDS = 30.0

_CURSOR_DIR = Path("/var/lib/kiara/worker_death_watcher")


class _RateLimiter:
    def __init__(self, window: float, limit: int) -> None:
        self._window = window
        self._limit = limit
        self._events: deque[float] = deque()

    def allow(self, now: float) -> bool:
        while self._events and now - self._events[0] > self._window:
            self._events.popleft()
        if len(self._events) >= self._limit:
            return False
        self._events.append(now)
        return True


def _is_service_stopping(service: str) -> bool:
    """Erkennt Service-Stop/Restart-Phasen, in denen Worker-Tode normal sind."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True,
            timeout=5,
        )
        state = result.stdout.strip()
        if state in ("inactive", "failed", "deactivating", "activating"):
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return False


_KIARA_CLI = "/opt/kiara/venv/bin/kiara"
_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX = 30.0
_THREAD_SUPERVISION_INTERVAL = 5.0
_BACKOFF_STABILIZE_SECONDS = 30.0
_CONSECUTIVE_FAIL_LIMIT = 5  # konsekutive Trigger-Fails -> Thread terminieren


class _LineStatus(str, Enum):
    """Terminaler Zustand einer journalctl-Line."""

    NO_MATCH = "no_match"
    TRIGGERED_OK = "triggered_ok"
    TRIGGERED_FAIL = "triggered_fail"
    DEDUPED = "deduped"
    RATE_LIMITED = "rate_limited"
    SUPPRESSED_STOP = "suppressed_stop"


class _PendingAction(str, Enum):
    """Naechste Aktion nach Bearbeitung eines Pending-Death-Events."""

    PROCEED = "proceed"
    RETRY_LATER = "retry_later"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class _LineResult:
    """Ergebnis der Line-Verarbeitung samt PID fuer Pending-Persistenz."""

    status: _LineStatus
    pid: str | None = None


@dataclass(frozen=True)
class _PendingWorkerDeath:
    """Durable Retry-Zustand fuer ein noch nicht geloggtes Worker-Death-Event."""

    service: str
    pid: str
    message: str
    cursor: str
    attempts: int = 1
    logged: bool = False


class _PendingStateError(RuntimeError):
    """Signalisiert korrupten oder unlesbaren Pending-State."""


class _CursorStateError(RuntimeError):
    """Signalisiert korrupten oder nicht persistierbaren Cursor-State."""


# Status-Codes fuer _handle_line: Cursor-Persistenz haengt davon ab, ob die
# Death-Line erfolgreich gemeldet, bewusst unterdrueckt oder verloren ging.
LINE_NO_MATCH = _LineStatus.NO_MATCH  # keine Death-Line
LINE_TRIGGERED_OK = _LineStatus.TRIGGERED_OK  # Match -> CLI rc=0
LINE_TRIGGERED_FAIL = _LineStatus.TRIGGERED_FAIL  # Match -> CLI Fail
LINE_DEDUPED = _LineStatus.DEDUPED  # gleiche PID < 30s
LINE_RATE_LIMITED = _LineStatus.RATE_LIMITED  # globaler Rate-Limit hit
LINE_SUPPRESSED_STOP = _LineStatus.SUPPRESSED_STOP  # Service-Stop-Phase

# Cursor wird fortgeschrieben nur wenn die Line entweder kein Match war oder
# das Match einen wohldefinierten Endzustand hat. Bei TRIGGERED_FAIL lassen
# wir den Cursor stehen — beim Watcher-Restart wird die Line erneut gesehen
# und der Trigger erneut versucht. Audit-Run-2 Finding #2.
_LINE_STATUSES_ADVANCE_CURSOR = frozenset(
    {
        LINE_NO_MATCH,
        LINE_TRIGGERED_OK,
        LINE_DEDUPED,
        LINE_RATE_LIMITED,
        LINE_SUPPRESSED_STOP,
    }
)


def _trigger_log_command(service: str, pid: str) -> bool:
    """Ruft `kiara log-worker-death PID SERVICE` auf.

    Returns:
        True bei rc=0, False bei rc!=0/Timeout/OSError.
        Loggt non-zero returncode mit gekuerztem stderr/stdout-Snippet.
    """
    try:
        result = subprocess.run(
            [_KIARA_CLI, "log-worker-death", pid, service],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("log-worker-death-Aufruf fehlgeschlagen (%s pid=%s): %s", service, pid, exc)
        return False
    if result.returncode != 0:
        stderr_snippet = (result.stderr or "")[:500].replace("\n", " ")
        stdout_snippet = (result.stdout or "")[:200].replace("\n", " ")
        logger.warning(
            "log-worker-death CLI rc=%s service=%s pid=%s stderr=%r stdout=%r",
            result.returncode,
            service,
            pid,
            stderr_snippet,
            stdout_snippet,
        )
        return False
    return True


def _is_valid_cursor(cursor: str) -> bool:
    if not _CURSOR_RE.fullmatch(cursor):
        return False
    parts = {part.split("=", 1)[0] for part in cursor.split(";")}
    return {"s", "i"}.issubset(parts)


def _read_cursor(path: Path) -> str | None:
    """Liest die persistierte Cursor-Position fail-closed."""
    try:
        cursor = path.read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _CursorStateError(f"Cursor nicht lesbar {path}: {exc}") from exc
    if not cursor:
        raise _CursorStateError(f"Cursor-Datei ist leer: {path}")
    if not _is_valid_cursor(cursor):
        raise _CursorStateError(f"Cursor-Datei ist ungueltig: {path}")
    return cursor


def _write_cursor_atomic(path: Path, cursor: str) -> bool:
    """Schreibt Cursor atomar via tmpfile + rename."""
    if not _is_valid_cursor(cursor):
        logger.error("Cursor-Schreibabbruch %s: ungueltiger Cursor %r", path, cursor)
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(cursor)
        tmp.replace(path)
        return True
    except OSError as exc:
        logger.error("Cursor-Schreibfehler %s: %s", path, exc)
        return False


def _pending_file_for(cursor_file: Path) -> Path:
    return cursor_file.with_suffix(cursor_file.suffix + ".pending.json")


def _write_pending_death_atomic(path: Path, pending: _PendingWorkerDeath) -> bool:
    """Persistiert ein fehlgeschlagenes Death-Event fuer durable Retries."""
    if not _is_valid_cursor(pending.cursor):
        logger.error("Pending-Death-Schreibabbruch %s: ungueltiger Cursor %r", path, pending.cursor)
        return False
    payload = {
        "version": 1,
        "service": pending.service,
        "pid": pending.pid,
        "message": pending.message,
        "cursor": pending.cursor,
        "attempts": pending.attempts,
        "logged": pending.logged,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        tmp.replace(path)
        return True
    except OSError as exc:
        logger.error("Pending-Death-Schreibfehler %s: %s", path, exc)
        return False


def _read_pending_death(path: Path) -> _PendingWorkerDeath | None:
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _PendingStateError(f"Pending-Death nicht lesbar {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _PendingStateError(f"Pending-Death korrupt {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise _PendingStateError(f"Pending-Death {path} ist kein JSON-Objekt")

    service = data.get("service")
    pid = data.get("pid")
    if not isinstance(service, str) or not service:
        raise _PendingStateError(f"Pending-Death {path} hat keinen Service")
    if pid is None:
        raise _PendingStateError(f"Pending-Death {path} hat keine PID")

    message = data.get("message", "")
    cursor = data.get("cursor")
    if not isinstance(cursor, str) or not cursor or not _is_valid_cursor(cursor):
        raise _PendingStateError(f"Pending-Death {path} hat keinen gueltigen Cursor")
    attempts = data.get("attempts", 0)
    try:
        attempts = int(attempts)
    except (TypeError, ValueError) as exc:
        raise _PendingStateError(f"Pending-Death {path} hat ungueltige attempts") from exc

    return _PendingWorkerDeath(
        service=service,
        pid=str(pid),
        message=message if isinstance(message, str) else str(message),
        cursor=cursor,
        attempts=max(attempts, 0),
        logged=bool(data.get("logged", False)),
    )


def _delete_pending_death(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Pending-Death-Loeschfehler %s: %s", path, exc)


def _process_pending_death(
    service: str,
    pending_file: Path,
    cursor_file: Path,
) -> _PendingAction:
    """Retry vor journalctl-Resume, solange ein Death-Event unpersistiert ist."""
    try:
        pending = _read_pending_death(pending_file)
    except _PendingStateError as exc:
        logger.error("%s — Watcher-Thread terminiert fail-closed", exc)
        return _PendingAction.TERMINATE
    if pending is None:
        return _PendingAction.PROCEED
    if pending.service != service:
        logger.error(
            "Pending-Death-Service-Mismatch file=%s pending=%s watcher=%s",
            pending_file,
            pending.service,
            service,
        )
        return _PendingAction.TERMINATE

    if pending.logged:
        logger.info(
            "Pending Worker-Death bereits geloggt, schreibe Cursor (service=%s pid=%s)",
            service,
            pending.pid,
        )
    else:
        logger.info(
            "Retry pending log-worker-death (service=%s pid=%s attempts=%d/%d)",
            service,
            pending.pid,
            pending.attempts,
            _CONSECUTIVE_FAIL_LIMIT,
        )
        if not _trigger_log_command(service, pending.pid):
            attempts = pending.attempts + 1
            next_pending = _PendingWorkerDeath(
                service=pending.service,
                pid=pending.pid,
                message=pending.message,
                cursor=pending.cursor,
                attempts=attempts,
                logged=False,
            )
            if not _write_pending_death_atomic(pending_file, next_pending):
                return _PendingAction.TERMINATE
            if attempts >= _CONSECUTIVE_FAIL_LIMIT:
                logger.error(
                    "Pending Worker-Death fuer %s pid=%s nach %d Versuchen weiterhin "
                    "ungeloggt — Thread terminiert fuer systemd-Restart",
                    service,
                    pending.pid,
                    attempts,
                )
                return _PendingAction.TERMINATE
            logger.warning(
                "Pending Worker-Death retry failed (service=%s pid=%s attempts=%d/%d)",
                service,
                pending.pid,
                attempts,
                _CONSECUTIVE_FAIL_LIMIT,
            )
            return _PendingAction.RETRY_LATER

        pending = _PendingWorkerDeath(
            service=pending.service,
            pid=pending.pid,
            message=pending.message,
            cursor=pending.cursor,
            attempts=pending.attempts,
            logged=True,
        )
        if not _write_pending_death_atomic(pending_file, pending):
            return _PendingAction.TERMINATE

    if _write_cursor_atomic(cursor_file, pending.cursor):
        _delete_pending_death(pending_file)
        logger.info(
            "Pending Worker-Death erfolgreich geloggt (service=%s pid=%s)", service, pending.pid
        )
        return _PendingAction.PROCEED
    logger.error(
        "Pending Worker-Death fuer %s pid=%s geloggt, aber Cursor "
        "nicht persistiert — Thread terminiert fail-closed",
        service,
        pending.pid,
    )
    return _PendingAction.TERMINATE


def _start_journalctl(service: str, cursor_file: Path) -> subprocess.Popen:
    """Startet einen journalctl-Tail-Subprocess mit JSON-Output.

    --output=json liefert pro Line ein vollstaendiges JSON-Objekt mit
    `__CURSOR`-Feld. Damit kann der Watcher inkrementell selbst
    persistieren — journalctl's eigenes `--cursor-file` schreibt erst
    beim sauberen Beenden, wir wollen aber Persistenz bei jedem Match
    fuer SIGKILL/OOM-Resilienz.

    Beim ersten Start (Cursor-Datei fehlt): --lines=0, damit nicht der
    gesamte Service-Journal-Verlauf reingeladen wird. Bei vorhandenem
    Cursor: --after-cursor, damit verpasste Events seit Watcher-Downtime
    nachgezogen werden.
    """
    cmd = [
        "journalctl",
        "-u",
        service,
        "--output=json",
        "--no-pager",
        "--follow",
    ]
    cursor = _read_cursor(cursor_file)
    if cursor:
        # --no-tail ist Pflicht bei Cursor-Resume in Follow-Mode: ohne diesen
        # Flag begrenzt journalctl die Replay-Menge auf die letzten N Lines
        # (default 10 bei --follow), Worker-Deaths waehrend Watcher-Downtime
        # gehen verloren sobald seit dem Cursor mehr als der Tail angefallen
        # ist. Audit-Run-2 Finding #1.
        cmd.extend(["--after-cursor", cursor, "--no-tail"])
    else:
        cmd.append("--lines=0")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # gekuerzt geloggt bei rc != 0; vorher DEVNULL
        text=True,
    )


def _stop_proc(proc: subprocess.Popen) -> None:
    """Beendet einen journalctl-Subprocess fail-safe."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception:
        pass


def _handle_line(
    line: str,
    service: str,
    last_seen_pid: dict[str, float],
    rate_limiter: _RateLimiter,
) -> _LineResult:
    """Verarbeitet eine einzelne journalctl-Line — Match, Dedup, Trigger.

    Returns:
        _LineResult mit LINE_*-Status-Code und optionaler Worker-PID. Caller
        entscheidet anhand des Status, ob der Cursor fortgeschrieben werden
        darf (siehe _LINE_STATUSES_ADVANCE_CURSOR).
    """
    match = _DIED_RE.search(line)
    if not match:
        return _LineResult(LINE_NO_MATCH)

    now = time.monotonic()
    pid = match.group(1)

    last = last_seen_pid.get(pid)
    if last is not None and (now - last) < _PER_PID_DEDUP_SECONDS:
        return _LineResult(LINE_DEDUPED, pid)
    last_seen_pid[pid] = now

    if _is_service_stopping(service):
        logger.info("Worker-Tod waehrend Stop ignoriert (service=%s pid=%s)", service, pid)
        return _LineResult(LINE_SUPPRESSED_STOP, pid)

    if not rate_limiter.allow(now):
        logger.warning(
            "Rate-Limit erreicht — Worker-Tod-Event verworfen (service=%s pid=%s)", service, pid
        )
        return _LineResult(LINE_RATE_LIMITED, pid)

    logger.info("Trigger log-worker-death (service=%s pid=%s)", service, pid)
    if _trigger_log_command(service, pid):
        return _LineResult(LINE_TRIGGERED_OK, pid)
    return _LineResult(LINE_TRIGGERED_FAIL, pid)


def _watch_service(service: str, max_iterations: int | None = None) -> None:
    """Tailt journalctl pro Service mit eingebautem Restart-Loop.

    Wenn der journalctl-Subprocess endet (Stream-EOF, Crash, journalctl-Restart
    durch journald-Vacuum) ODER eine unerwartete Exception fliegt, wird nach
    exponential backoff ein neuer Subprocess gestartet. Bei dauerhaft
    fehlschlagendem `log-worker-death` terminiert der Thread fail-closed; die
    Hauptprozess-Supervision (siehe `main`) beendet dann den Prozess, damit
    systemd neu startet.

    Args:
        service: Name der zu beobachtenden Service-Unit.
        max_iterations: Test-Hilfsparameter — limitiert die Anzahl der
            Watcher-Loop-Iterationen. None = unbegrenzt (Produktiv).
    """
    cursor_file = _CURSOR_DIR / f"{service}.cursor"
    pending_file = _pending_file_for(cursor_file)
    rate_limiter = _RateLimiter(_GLOBAL_RATE_WINDOW, _GLOBAL_RATE_LIMIT)
    last_seen_pid: dict[str, float] = {}
    backoff = _BACKOFF_INITIAL
    iterations = 0

    while True:
        if max_iterations is not None and iterations >= max_iterations:
            return
        iterations += 1

        pending_action = _process_pending_death(service, pending_file, cursor_file)
        if pending_action == _PendingAction.TERMINATE:
            return
        if pending_action == _PendingAction.RETRY_LATER:
            time.sleep(backoff)
            backoff = min(backoff * 2.0, _BACKOFF_MAX)
            continue

        proc: subprocess.Popen | None = None
        loop_started_at = time.monotonic()
        had_processed_entry = False
        try:
            proc = _start_journalctl(service, cursor_file)
            logger.info("Watcher tailt service=%s journalctl-pid=%s", service, proc.pid)

            stream_aborted_after_fail = False
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    # Defekte Line einfach ueberspringen, Cursor nicht persistieren
                    continue
                if not isinstance(entry, dict):
                    continue

                message = entry.get("MESSAGE", "")
                if not isinstance(message, str):
                    message = str(message)

                result = _handle_line(message, service, last_seen_pid, rate_limiter)
                status = result.status
                had_processed_entry = True

                # Backoff erst nach erfolgreichem Match-Output ODER stabiler
                # Laufzeit zuruecksetzen — Audit-Run-2 Finding #4: vorher
                # wurde reset direkt nach Popen, wodurch ein sofort-EOF-
                # journalctl in 1s-Loop endlos respawnt waere.
                backoff = _BACKOFF_INITIAL

                if status == LINE_TRIGGERED_FAIL:
                    if not result.pid:
                        logger.error(
                            "Trigger-Fail ohne extrahierte PID (service=%s) — "
                            "Thread terminiert fail-closed",
                            service,
                        )
                        return
                    cursor = entry.get("__CURSOR")
                    if not isinstance(cursor, str) or not _is_valid_cursor(cursor):
                        logger.error(
                            "Death-Line ohne gueltigen journald-Cursor "
                            "(service=%s pid=%s) — Thread terminiert fail-closed",
                            service,
                            result.pid,
                        )
                        return
                    pending = _PendingWorkerDeath(
                        service=service,
                        pid=result.pid,
                        message=message,
                        cursor=cursor,
                        attempts=1,
                    )
                    if not _write_pending_death_atomic(pending_file, pending):
                        logger.error(
                            "Pending-Death konnte nicht persistiert werden "
                            "(service=%s pid=%s) — Thread terminiert fail-closed",
                            service,
                            result.pid,
                        )
                        return
                    # Ohne Pending-State koennte eine spaetere NO_MATCH-Line
                    # den Cursor hinter das unpersistierte Death-Event ziehen.
                    # Deshalb brechen wir den Stream sofort ab und verarbeiten
                    # erst den durable Retry, bevor journald weitergelesen wird.
                    logger.warning(
                        "Trigger-Fail (service=%s) — Stream wird beendet, "
                        "Pending-Death persistiert (pid=%s attempts=1/%d)",
                        service,
                        result.pid,
                        _CONSECUTIVE_FAIL_LIMIT,
                    )
                    stream_aborted_after_fail = True
                    break

                # Cursor nur fortschreiben wenn die Line einen wohldefinierten
                # Endzustand hat. Bei TRIGGERED_FAIL wird ein Pending-Death
                # geschrieben und der Stream vor dem Cursor-Write abgebrochen.
                if status in _LINE_STATUSES_ADVANCE_CURSOR:
                    cursor = entry.get("__CURSOR")
                    if not isinstance(cursor, str) or not _is_valid_cursor(cursor):
                        logger.error(
                            "Journal-Line ohne gueltigen Cursor (service=%s "
                            "status=%s) — Thread terminiert fail-closed",
                            service,
                            status.value,
                        )
                        return
                    if not _write_cursor_atomic(cursor_file, cursor):
                        logger.error(
                            "Cursor konnte nicht persistiert werden "
                            "(service=%s status=%s) — Thread terminiert "
                            "fail-closed",
                            service,
                            status.value,
                        )
                        return

            # for-loop sauber verlassen — entweder Stream-EOF oder break-aus-FAIL.
            if stream_aborted_after_fail:
                logger.info(
                    "Stream fuer %s nach TRIGGERED_FAIL beendet — neuer Versuch in %.1fs",
                    service,
                    backoff,
                )
            else:
                stderr_snippet = ""
                if proc and proc.stderr:
                    try:
                        stderr_snippet = proc.stderr.read()[:500].replace("\n", " ")
                    except (OSError, ValueError):
                        pass
                returncode = None
                if proc:
                    try:
                        returncode = proc.poll()
                    except (OSError, ValueError):
                        returncode = proc.returncode
                    if returncode is None:
                        returncode = proc.returncode
                if returncode not in (None, 0) and "cursor" in stderr_snippet.lower():
                    logger.error(
                        "journalctl fuer %s scheiterte mit Cursor-Fehler "
                        "(rc=%s, stderr=%r) — Thread terminiert fail-closed",
                        service,
                        returncode,
                        stderr_snippet,
                    )
                    return
                uptime = time.monotonic() - loop_started_at
                logger.warning(
                    "journalctl-Stream fuer %s endete unerwartet (rc=%s, "
                    "uptime=%.1fs, stderr=%r) — Restart in %.1fs",
                    service,
                    returncode,
                    uptime,
                    stderr_snippet,
                    backoff,
                )
        except _CursorStateError as exc:
            logger.error("%s — Watcher-Thread terminiert fail-closed", exc)
            return
        except Exception:
            logger.exception("Watcher-Loop %s Fehler — Restart in %.1fs", service, backoff)
        finally:
            if proc is not None:
                _stop_proc(proc)

        # Backoff-Eskalation: nur wenn keine Line verarbeitet wurde, also
        # journalctl quasi-sofort beendet. Bei stabiler Laufzeit (>=30s)
        # reset auf Initial. Audit-Run-2 Finding #4.
        uptime = time.monotonic() - loop_started_at
        if had_processed_entry or uptime >= _BACKOFF_STABILIZE_SECONDS:
            backoff = _BACKOFF_INITIAL
        else:
            backoff = min(backoff * 2.0, _BACKOFF_MAX)

        time.sleep(backoff)


def _signal_handler(signum, _frame):
    logger.info("Signal %s empfangen, beende Watcher", signum)
    # os._exit (statt sys.exit) damit Daemon-Threads sofort enden,
    # auch wenn sie in subprocess.stdout.read blockiert sind.
    os._exit(0)


def main() -> None:
    if os.geteuid() != 0:
        logger.warning(
            "Watcher laeuft nicht als root — journalctl/dmesg koennen eingeschraenkt sein"
        )

    try:
        _CURSOR_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cursor-Verzeichnis %s nicht erstellbar: %s", _CURSOR_DIR, exc)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Multi-Service: ein Daemon-Thread pro Service. _watch_service hat einen
    # eigenen Restart-Loop und sollte nie sauber returnen — deshalb ist
    # `thread.is_alive()` False ein Indikator fuer einen Bug, der den Thread
    # zerlegt hat (z.B. ein KeyError ausserhalb des try/except).
    threads: list[threading.Thread] = []
    for service in _TRACKED_SERVICES:
        thread = threading.Thread(
            target=_watch_service,
            args=(service,),
            name=f"watcher-{service}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        logger.info("Watcher-Thread gestartet fuer %s", service)

    # Thread-Supervision: alle 5s pruefen, ob jeder Service-Watcher noch laeuft.
    # Wenn nicht: harter Prozess-Exit, damit systemd Restart=on-failure greift.
    # Vorher hatte signal.pause() den Hauptprozess wachgehalten, auch wenn
    # einzelne Threads tot waren — Service erschien grun, Coverage war aber
    # teilblind (Audit-Finding High, 0.0.1+447-Audit-Run).
    while True:
        time.sleep(_THREAD_SUPERVISION_INTERVAL)
        for thread in threads:
            if not thread.is_alive():
                logger.error(
                    "Watcher-Thread %s ist tot — harter Exit fuer systemd Restart=on-failure",
                    thread.name,
                )
                os._exit(1)


if __name__ == "__main__":
    main()
