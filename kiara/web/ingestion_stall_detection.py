"""Diagnostische Erkennung blockierter CIFS-Reads im Ingestion-Worker.

Die Erkennung ist absichtlich rein beobachtend: Sie sendet keine Signale,
veraendert keine Mounts und greift nicht in den Jobablauf ein. Ein fehlender
Datei- oder Chunk-Fortschritt ist nur Kontext. Als Hänger gilt erst ein
durchgehend im Linux-D-State befindlicher nativer Verarbeitungsthread mit
unveraenderter Scheduler- und Pipeline-Aktivitaet sowie CIFS-/filemap-
Wartekontext.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class IngestionActivitySnapshot:
    """Unveraenderlicher Snapshot der fachlichen Pipeline-Aktivitaet."""

    job_id: int | None = None
    datasource_id: int | None = None
    connector_type: str = ""
    file_name: str = ""
    file_path: str = ""
    file_started_at: str = ""
    phase: str = ""
    phase_started_at: str = ""
    last_event: str = ""
    last_activity_at: str = ""
    last_activity_monotonic: float = 0.0
    activity_sequence: int = 0
    chunks_created_current_file: int = 0
    chunks_written_current_file: int = 0

    def for_persistence(self) -> dict:
        """Serialisierbare Aktivitaet ohne prozesslokale Monotonic-Zeit."""
        data = asdict(self)
        data.pop("last_activity_monotonic", None)
        return data


class IngestionActivityTracker:
    """Thread-sicherer Aktivitaetszaehler fuer genau einen seriellen Job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = IngestionActivitySnapshot()

    def snapshot(self) -> IngestionActivitySnapshot:
        with self._lock:
            return self._snapshot

    def begin_job(self, job_id: int, datasource_id: int, connector_type: str) -> None:
        now_iso = _utc_now_iso()
        connector_value = getattr(connector_type, "value", connector_type)
        with self._lock:
            sequence = self._snapshot.activity_sequence + 1
            self._snapshot = IngestionActivitySnapshot(
                job_id=job_id,
                datasource_id=datasource_id,
                connector_type=str(connector_value or "").lower(),
                phase="initializing",
                phase_started_at=now_iso,
                last_event="job_started",
                last_activity_at=now_iso,
                last_activity_monotonic=time.monotonic(),
                activity_sequence=sequence,
            )

    def finish_job(self, job_id: int) -> None:
        with self._lock:
            if self._snapshot.job_id != job_id:
                return
            self._snapshot = IngestionActivitySnapshot(
                activity_sequence=self._snapshot.activity_sequence + 1
            )

    def record(
        self,
        job_id: int,
        event: str,
        *,
        phase: str | None = None,
        file_name: str | None = None,
        file_path: str | None = None,
        chunks_created: int | None = None,
        chunks_written_delta: int = 0,
    ) -> IngestionActivitySnapshot | None:
        """Zaehlt einen fachlichen Checkpoint und liefert den neuen Snapshot."""
        now_iso = _utc_now_iso()
        now_monotonic = time.monotonic()
        with self._lock:
            current = self._snapshot
            if current.job_id != job_id:
                return None

            is_file_start = event == "file_started"
            next_phase = phase if phase is not None else current.phase
            phase_started_at = current.phase_started_at
            if next_phase != current.phase:
                phase_started_at = now_iso

            next_file_name = current.file_name if file_name is None else str(file_name)
            next_file_path = current.file_path if file_path is None else str(file_path)
            file_started_at = current.file_started_at
            created = current.chunks_created_current_file
            written = current.chunks_written_current_file
            if is_file_start:
                file_started_at = now_iso
                created = 0
                written = 0
            if chunks_created is not None:
                created = max(0, int(chunks_created))
            if chunks_written_delta:
                written += max(0, int(chunks_written_delta))

            self._snapshot = replace(
                current,
                file_name=next_file_name,
                file_path=next_file_path,
                file_started_at=file_started_at,
                phase=next_phase,
                phase_started_at=phase_started_at,
                last_event=str(event),
                last_activity_at=now_iso,
                last_activity_monotonic=now_monotonic,
                activity_sequence=current.activity_sequence + 1,
                chunks_created_current_file=created,
                chunks_written_current_file=written,
            )
            return self._snapshot


@dataclass(frozen=True)
class ThreadProcSnapshot:
    """Relevante Werte aus ``/proc/<pid>/task/<tid>``."""

    pid: int
    tid: int
    state: str = ""
    state_label: str = ""
    wchan: str = ""
    kernel_stack: str = ""
    syscall: str = ""
    sched: str = ""
    io: str = ""
    user_ticks: int | None = None
    system_ticks: int | None = None
    voluntary_context_switches: int | None = None
    nonvoluntary_context_switches: int | None = None
    read_errors: dict[str, str] = field(default_factory=dict)

    def for_persistence(self) -> dict:
        return asdict(self)


def _parse_thread_status(raw: str) -> dict:
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key] = value.strip()

    state_label = fields.get("State", "")
    state = state_label.split(maxsplit=1)[0] if state_label else ""

    def _optional_int(key: str) -> int | None:
        value = fields.get(key)
        if value is None:
            return None
        try:
            return int(value.split()[0])
        except (TypeError, ValueError, IndexError):
            return None

    return {
        "state": state,
        "state_label": state_label,
        "voluntary_context_switches": _optional_int("voluntary_ctxt_switches"),
        "nonvoluntary_context_switches": _optional_int("nonvoluntary_ctxt_switches"),
    }


def _parse_thread_stat(raw: str) -> dict:
    """Parst ``stat`` robust gegen Leerzeichen und Klammern im Thread-Namen."""
    comm_end = raw.rfind(")")
    if comm_end < 0:
        return {}
    fields = raw[comm_end + 1 :].split()
    # fields[0] ist Feld 3 (state), fields[11]/[12] sind utime/stime.
    if len(fields) <= 12:
        return {}
    try:
        return {
            "user_ticks": int(fields[11]),
            "system_ticks": int(fields[12]),
        }
    except (TypeError, ValueError):
        return {}


def _read_limited_text(path: Path, max_chars: int) -> tuple[str, str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            value = handle.read(max_chars + 1)
    except OSError as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if len(value) > max_chars:
        value = value[:max_chars] + "\n[truncated]"
    return value.strip(), ""


def read_thread_proc_snapshot(
    pid: int,
    tid: int,
    *,
    proc_root: str | os.PathLike[str] = "/proc",
) -> ThreadProcSnapshot | None:
    """Liest einen nativen Thread-Snapshot; verschwundene Threads ergeben ``None``."""
    task_dir = Path(proc_root) / str(pid) / "task" / str(tid)
    status, status_error = _read_limited_text(task_dir / "status", 32_768)
    if status_error:
        return None

    status_values = _parse_thread_status(status)
    errors: dict[str, str] = {}

    stat, stat_error = _read_limited_text(task_dir / "stat", 8_192)
    if stat_error:
        errors["stat"] = stat_error
    stat_values = _parse_thread_stat(stat)

    wchan, wchan_error = _read_limited_text(task_dir / "wchan", 4_096)
    if wchan_error:
        errors["wchan"] = wchan_error

    kernel_stack, stack_error = _read_limited_text(task_dir / "stack", 32_768)
    if stack_error:
        errors["stack"] = stack_error

    syscall, syscall_error = _read_limited_text(task_dir / "syscall", 8_192)
    if syscall_error:
        errors["syscall"] = syscall_error

    sched, sched_error = _read_limited_text(task_dir / "sched", 65_536)
    if sched_error:
        errors["sched"] = sched_error

    io, io_error = _read_limited_text(task_dir / "io", 16_384)
    if io_error:
        errors["io"] = io_error

    return ThreadProcSnapshot(
        pid=int(pid),
        tid=int(tid),
        state=status_values.get("state", ""),
        state_label=status_values.get("state_label", ""),
        wchan=wchan,
        kernel_stack=kernel_stack,
        syscall=syscall,
        sched=sched,
        io=io,
        user_ticks=stat_values.get("user_ticks"),
        system_ticks=stat_values.get("system_ticks"),
        voluntary_context_switches=status_values.get("voluntary_context_switches"),
        nonvoluntary_context_switches=status_values.get("nonvoluntary_context_switches"),
        read_errors=errors,
    )


def read_python_thread_snapshot(native_tid: int) -> dict:
    """Liefert den Python-Stack eines nativen Threads ohne dessen Mitarbeit.

    ``sys._current_frames`` funktioniert auch dann, wenn der Zielthread gerade
    in einem blockierenden C-Syscall steckt. Damit bleibt die Python-Aufrufkette
    oberhalb von ``read(2)`` im selben strukturierten Stall-Event erhalten.
    """
    result: dict = {
        "available": False,
        "native_tid": int(native_tid),
        "thread_name": "",
        "python_ident": None,
        "stack": "",
        "error": "",
    }
    try:
        target = next(
            (thread for thread in threading.enumerate() if thread.native_id == native_tid),
            None,
        )
        if target is None or target.ident is None:
            result["error"] = "Nativer Thread ist keinem Python-Thread zugeordnet"
            return result
        result["thread_name"] = target.name
        result["python_ident"] = target.ident
        frame = sys._current_frames().get(target.ident)
        if frame is None:
            result["error"] = "Python-Frame ist nicht mehr verfuegbar"
            return result
        stack = "".join(traceback.format_stack(frame))
        if len(stack) > 65_536:
            stack = "[truncated to tail]\n" + stack[-65_536:]
        result["stack"] = stack
        result["available"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_cifs_proc_diagnostics(
    *,
    proc_root: str | os.PathLike[str] = "/proc",
) -> dict[str, dict[str, str | bool]]:
    """Liest begrenzte CIFS-Kerneldiagnostik fuer ein forensisches Event."""
    cifs_root = Path(proc_root) / "fs" / "cifs"
    diagnostics: dict[str, dict[str, str | bool]] = {}
    for name, max_chars in (("DebugData", 65_536), ("Stats", 32_768), ("open_files", 32_768)):
        value, error = _read_limited_text(cifs_root / name, max_chars)
        diagnostics[name] = {
            "available": not bool(error),
            "raw": value,
            "error": error,
        }
    return diagnostics


@dataclass(frozen=True)
class DStateEvent:
    event_type: str
    level: str
    state: str
    reason: str
    duration_seconds: float
    details: dict


@dataclass
class _DStateCandidate:
    key: tuple[int, str, int]
    first_seen_monotonic: float
    first_seen_at: str
    activity: IngestionActivitySnapshot
    initial_thread: ThreadProcSnapshot
    latest_thread: ThreadProcSnapshot
    warning_emitted: bool = False
    confirmed: bool = False
    classification: str = ""


_CIFS_WAIT_MARKERS = (
    "cifs",
    "smb2",
    "smb3",
)
_GENERIC_FILE_WAIT_MARKERS = (
    "filemap",
    "folio_wait",
    "wait_on_page",
    "netfs",
)


def _classify_cifs_wait(
    activity: IngestionActivitySnapshot,
    thread: ThreadProcSnapshot,
) -> str:
    wait_context = f"{thread.wchan}\n{thread.kernel_stack}".lower()
    if any(marker in wait_context for marker in _CIFS_WAIT_MARKERS):
        return "cifs_kernel_wait"
    if (
        activity.connector_type == "smb"
        and activity.phase == "loading"
        and any(marker in wait_context for marker in _GENERIC_FILE_WAIT_MARKERS)
    ):
        return "smb_filemap_wait"
    return ""


def _scheduler_signature(thread: ThreadProcSnapshot) -> tuple[int | None, ...]:
    return (
        thread.user_ticks,
        thread.system_ticks,
        thread.voluntary_context_switches,
        thread.nonvoluntary_context_switches,
    )


def _scheduler_evidence_available(thread: ThreadProcSnapshot) -> bool:
    signature = _scheduler_signature(thread)
    return any(value is not None for value in signature[:2]) and any(
        value is not None for value in signature[2:]
    )


class CIFSDStateDetector:
    """Zustandsautomat fuer deduplizierte D-State-Systemereignisse."""

    def __init__(self, *, warning_after_seconds: float, timeout_seconds: float) -> None:
        self.warning_after_seconds = float(warning_after_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._candidate: _DStateCandidate | None = None

    def observe(
        self,
        activity: IngestionActivitySnapshot,
        thread: ThreadProcSnapshot | None,
        *,
        now_monotonic: float | None = None,
        observed_at: str | None = None,
    ) -> list[DStateEvent]:
        """Nimmt einen Poll-Snapshot entgegen und liefert neue Zustandsereignisse."""
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        wall_time = observed_at or _utc_now_iso()
        events: list[DStateEvent] = []

        qualifies = bool(
            thread is not None
            and thread.state == "D"
            and activity.job_id is not None
            and activity.connector_type == "smb"
            and activity.file_path
        )
        if not qualifies:
            reason = "thread_left_d_state"
            if activity.job_id is None or not activity.file_path:
                reason = "job_or_file_finished"
            elif thread is None:
                reason = "thread_unavailable"
            events.extend(self._clear_visible_candidate(now, wall_time, reason))
            return events

        assert thread is not None
        assert activity.job_id is not None
        key = (activity.job_id, activity.file_path, activity.activity_sequence)
        if self._candidate is None:
            self._candidate = self._new_candidate(key, activity, thread, now, wall_time)
            return events

        candidate = self._candidate
        scheduler_changed = _scheduler_signature(candidate.initial_thread) != _scheduler_signature(
            thread
        )
        if candidate.key != key or scheduler_changed:
            reason = (
                "pipeline_activity_resumed"
                if candidate.key != key
                else "scheduler_activity_resumed"
            )
            events.extend(self._clear_visible_candidate(now, wall_time, reason))
            self._candidate = self._new_candidate(key, activity, thread, now, wall_time)
            return events

        candidate.activity = activity
        candidate.latest_thread = thread
        candidate.classification = _classify_cifs_wait(activity, thread) or candidate.classification
        duration = max(0.0, now - candidate.first_seen_monotonic)

        if duration >= self.warning_after_seconds and not candidate.warning_emitted:
            candidate.warning_emitted = True
            events.append(
                self._build_event(
                    candidate,
                    event_type="INGESTION_DSTATE_DETECTED",
                    level="WARNING",
                    state="detected",
                    reason=candidate.classification or "persistent_d_state",
                    duration=duration,
                    observed_at=wall_time,
                )
            )

        if (
            duration >= self.timeout_seconds
            and not candidate.confirmed
            and candidate.classification
            and _scheduler_evidence_available(candidate.initial_thread)
        ):
            candidate.confirmed = True
            events.append(
                self._build_event(
                    candidate,
                    event_type="INGESTION_DSTATE_STALL_CONFIRMED",
                    level="ERROR",
                    state="confirmed",
                    reason=candidate.classification,
                    duration=duration,
                    observed_at=wall_time,
                )
            )

        return events

    def _new_candidate(
        self,
        key: tuple[int, str, int],
        activity: IngestionActivitySnapshot,
        thread: ThreadProcSnapshot,
        now: float,
        wall_time: str,
    ) -> _DStateCandidate:
        return _DStateCandidate(
            key=key,
            first_seen_monotonic=now,
            first_seen_at=wall_time,
            activity=activity,
            initial_thread=thread,
            latest_thread=thread,
            classification=_classify_cifs_wait(activity, thread),
        )

    def _clear_visible_candidate(
        self,
        now: float,
        wall_time: str,
        reason: str,
    ) -> list[DStateEvent]:
        candidate = self._candidate
        self._candidate = None
        if candidate is None or not (candidate.warning_emitted or candidate.confirmed):
            return []
        duration = max(0.0, now - candidate.first_seen_monotonic)
        return [
            self._build_event(
                candidate,
                event_type="INGESTION_DSTATE_CLEARED",
                level="INFO",
                state="cleared",
                reason=reason,
                duration=duration,
                observed_at=wall_time,
            )
        ]

    @staticmethod
    def _build_event(
        candidate: _DStateCandidate,
        *,
        event_type: str,
        level: str,
        state: str,
        reason: str,
        duration: float,
        observed_at: str,
    ) -> DStateEvent:
        details = {
            "state": state,
            "reason": reason,
            "duration_seconds": round(duration, 3),
            "d_state_first_seen_at": candidate.first_seen_at,
            "observed_at": observed_at,
            "activity": candidate.activity.for_persistence(),
            "thread": candidate.latest_thread.for_persistence(),
            "initial_scheduler": {
                "user_ticks": candidate.initial_thread.user_ticks,
                "system_ticks": candidate.initial_thread.system_ticks,
                "voluntary_context_switches": (candidate.initial_thread.voluntary_context_switches),
                "nonvoluntary_context_switches": (
                    candidate.initial_thread.nonvoluntary_context_switches
                ),
            },
            "classification": candidate.classification or "unclassified",
            "automatic_recovery": False,
        }
        return DStateEvent(
            event_type=event_type,
            level=level,
            state=state,
            reason=reason,
            duration_seconds=round(duration, 3),
            details=details,
        )
