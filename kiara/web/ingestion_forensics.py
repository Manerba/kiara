"""Privilegierte Forensik fuer blockierte SMB-Ingestion-Threads.

Der Ingestion-Worker selbst darf viele fuer einen D-State relevante Kernel-
Dateien nicht lesen. Dieses Modul stellt deshalb zwei eng begrenzte Bausteine
bereit:

* einen kleinen CIFS-Tracefs-Ringpuffer, der bereits vor einem Hänger laeuft;
* einen Root-Collector, der ausschliesslich Threads des systemd-Services
  ``kiara-worker.service`` akzeptiert.

Der Collector veraendert weder Mounts noch Jobs. Der Tracepuffer wird fuer den
Snapshot nur kurz angehalten und danach in seinen vorherigen Zustand versetzt.
"""

from __future__ import annotations

import json
import os
import platform
import pwd
import re
import stat
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_TRACE_INSTANCE_NAME = "kiara-cifs"
_TRACE_BUFFER_KB_PER_CPU = 256
_TRACE_EVENTS = (
    "smb3_cmd_enter",
    "smb3_cmd_done",
    "smb3_cmd_err",
    "smb3_read_enter",
    "smb3_read_done",
    "smb3_read_err",
    "smb3_slow_rsp",
    "smb3_reconnect",
    "smb3_reconnect_detected",
    "smb3_partial_send_reconnect",
    "smb3_credit_timeout",
    "smb3_wait_credits",
    "smb3_insufficient_credits",
    "smb3_pend_credits",
    "smb3_hdr_credits",
    "smb3_add_credits",
    "smb3_reconnect_with_invalid_credits",
    "smb3_overflow_credits",
    "smb3_too_many_credits",
)
_KRETPROBE_GROUP = "kiara_cifs"
_KRETPROBE_EVENT = "decrypt_raw_data_ret"
_KRETPROBE_ID = f"{_KRETPROBE_GROUP}/{_KRETPROBE_EVENT}"
_KRETPROBE_DEFINITION = f"r128:{_KRETPROBE_ID} decrypt_raw_data ret=$retval:s32"
_KRETPROBE_FILTER = "ret < 0"
_TRACE_TAIL_LINES = 600
_TRACE_MID_MATCH_LIMIT = 200
_TRACE_DIAGNOSTIC_MATCH_LIMIT = 200
_PCAP_SERVICE = "kiara-smb-pcap-recorder.service"
_PCAP_DIR = Path("/var/lib/kiara-smb-forensics")
_PCAP_PREFIX = "client-smb.pcap"
_PCAP_FILE_LIMIT = 16
_CLI_PATH = "/opt/kiara/venv/bin/kiara"
_SUDO_PATH = "/usr/bin/sudo"
_WORKER_UNIT = "kiara-worker.service"
_MAX_PID = 4_194_304
_READ_SYSCALLS = {
    0: "read",
    17: "pread64",
    19: "readv",
    295: "preadv",
    327: "preadv2",
}
_MID_RE = re.compile(
    r"State:\s*(?P<state>\d+)\s+com:\s*(?P<command>\d+)\s+"
    r"pid:\s*(?P<pid>\d+).*?\bmid\s+(?P<mid>\d+)"
)


class ForensicsTargetError(ValueError):
    """Der angeforderte PID/TID gehoert nicht zum KIara-Ingestion-Worker."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _read_limited(path: Path, max_chars: int) -> dict[str, str | bool]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read(max_chars + 1)
    except OSError as exc:
        return {"available": False, "raw": "", "error": _error_text(exc)}
    truncated = len(raw) > max_chars
    if truncated:
        raw = raw[:max_chars] + "\n[truncated]"
    return {
        "available": True,
        "raw": raw.strip(),
        "error": "",
        "truncated": truncated,
    }


def _write_control(path: Path, value: str) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write(value)


def _append_control(path: Path, value: str) -> None:
    # kprobe_events ist eine Kommando-Schnittstelle: neue Definitionen muessen
    # als einzelner unbuffered write angehaengt werden. TextIOWrapper wird von
    # tracefs hier mit EINVAL abgewiesen; O_TRUNC wuerde zudem fremde Probes
    # entfernen.
    payload = value.encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CLOEXEC)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError(f"Kurzer tracefs-Control-Write: {written}/{len(payload)}")
    finally:
        os.close(fd)


def _ensure_decrypt_kretprobe(trace_root: Path) -> dict:
    """Definiert eine gefilterte Rueckkehrprobe, ohne fremde Probes anzufassen."""
    definition_path = trace_root / "kprobe_events"
    result: dict = {
        "available": False,
        "defined": False,
        "created": False,
        "enabled": False,
        "event": _KRETPROBE_ID,
        "definition": _KRETPROBE_DEFINITION,
        "filter": _KRETPROBE_FILTER,
        "error": "",
    }
    try:
        definitions = definition_path.read_text(encoding="ascii")
        matching = []
        for line in definitions.splitlines():
            fields = line.split(maxsplit=1)
            if fields and fields[0].endswith(f":{_KRETPROBE_ID}"):
                matching.append(line.strip())
        if matching and _KRETPROBE_DEFINITION not in matching:
            result["error"] = f"Kprobe-Namenskollision fuer {_KRETPROBE_ID}: {matching[0]}"
            return result
        if not matching:
            _append_control(definition_path, f"{_KRETPROBE_DEFINITION}\n")
            result["created"] = True
        result["defined"] = True
        result["available"] = True
    except OSError as exc:
        result["error"] = _error_text(exc)
    return result


def _control_is(path: Path, expected: str) -> bool:
    try:
        return path.read_text(encoding="ascii").strip() == expected
    except OSError:
        return False


def configure_cifs_flight_recorder(
    *,
    trace_root: str | os.PathLike[str] = "/sys/kernel/tracing",
) -> dict:
    """Initialisiert einen begrenzten CIFS-Trace-Ringpuffer best-effort."""
    root = Path(trace_root)
    instance = root / "instances" / _TRACE_INSTANCE_NAME
    instance_preexisting = instance.exists()
    result: dict = {
        "available": False,
        "instance": str(instance),
        "buffer_kb_per_cpu": _TRACE_BUFFER_KB_PER_CPU,
        "preserved_existing": False,
        "enabled_events": [],
        "missing_events": [],
        "kretprobe": {},
        "errors": {},
    }
    try:
        instance.mkdir(parents=False, exist_ok=True)
    except OSError as exc:
        result["errors"]["instance"] = _error_text(exc)
        return result
    kretprobe = _ensure_decrypt_kretprobe(root)
    result["kretprobe"] = kretprobe
    probe_event_dir = instance / "events" / _KRETPROBE_GROUP / _KRETPROBE_EVENT
    probe_filter_path = probe_event_dir / "filter"
    probe_enable_path = probe_event_dir / "enable"

    enabled = [
        event
        for event in _TRACE_EVENTS
        if _control_is(instance / "events" / "cifs" / event / "enable", "1")
    ]
    probe_complete = bool(
        kretprobe.get("defined")
        and _control_is(probe_enable_path, "1")
        and _control_is(probe_filter_path, _KRETPROBE_FILTER)
    )

    # Service-Restarts duerfen gerade im Incident-Fall keinen bereits laufenden
    # Ringpuffer leeren. Ist die vollstaendige Konfiguration aktiv, bleibt sie
    # deshalb unangetastet. Eine unvollstaendige bestehende Instanz wird unten
    # nur pausiert und erweitert; buffer_size_kb und trace bleiben unveraendert.
    if instance_preexisting:
        try:
            tracing_on = (instance / "tracing_on").read_text(encoding="ascii").strip()
            if (
                tracing_on == "1"
                and len(enabled) == len(_TRACE_EVENTS)
                and (probe_complete or not kretprobe.get("defined"))
            ):
                result["available"] = True
                result["preserved_existing"] = True
                result["enabled_events"] = enabled
                kretprobe["enabled"] = probe_complete
                return result
        except OSError:
            pass

    previous = ""
    try:
        previous = (instance / "tracing_on").read_text(encoding="ascii").strip()
        _write_control(instance / "tracing_on", "0\n")
        if instance_preexisting:
            result["preserved_existing"] = True
        else:
            _write_control(instance / "buffer_size_kb", f"{_TRACE_BUFFER_KB_PER_CPU}\n")
            # O_TRUNC auf tracefs leert nur den neuen Instanz-Ringpuffer.
            _write_control(instance / "trace", "")
    except OSError as exc:
        result["errors"]["initialization"] = _error_text(exc)
        if previous == "1":
            try:
                _write_control(instance / "tracing_on", "1\n")
            except OSError as resume_exc:
                result["errors"]["resume"] = _error_text(resume_exc)
        return result

    for event in _TRACE_EVENTS:
        enable_path = instance / "events" / "cifs" / event / "enable"
        if not enable_path.exists():
            result["missing_events"].append(event)
            continue
        try:
            _write_control(enable_path, "1\n")
            result["enabled_events"].append(event)
        except OSError as exc:
            result["errors"][event] = _error_text(exc)

    if kretprobe.get("defined"):
        if not probe_event_dir.exists():
            kretprobe["available"] = False
            kretprobe["error"] = "Kretprobe-Event fehlt in der Trace-Instanz"
        else:
            try:
                _write_control(probe_filter_path, f"{_KRETPROBE_FILTER}\n")
                _write_control(probe_enable_path, "1\n")
                kretprobe["enabled"] = True
            except OSError as exc:
                kretprobe["available"] = False
                kretprobe["error"] = _error_text(exc)

    if not result["enabled_events"]:
        result["errors"].setdefault("events", "Keine CIFS-Tracepoints verfuegbar")
        if previous == "1":
            try:
                _write_control(instance / "tracing_on", "1\n")
            except OSError as exc:
                result["errors"]["resume"] = _error_text(exc)
        return result
    try:
        _write_control(instance / "tracing_on", "1\n")
    except OSError as exc:
        result["errors"]["tracing_on"] = _error_text(exc)
        return result
    result["available"] = True
    return result


def _parse_pending_mids(debug_data: str) -> list[dict[str, int]]:
    pending: list[dict[str, int]] = []
    for match in _MID_RE.finditer(debug_data):
        pending.append({key: int(value) for key, value in match.groupdict().items()})
    return pending


def _trace_line_matches_mid(line: str, mids: set[int]) -> bool:
    return any(
        re.search(rf"(?:\bmid|\bcurrent_mid)(?:=|\s+){mid}\b", line) is not None for mid in mids
    )


def read_cifs_flight_recorder(
    pending_mids: list[dict[str, int]],
    *,
    trace_root: str | os.PathLike[str] = "/sys/kernel/tracing",
) -> dict:
    """Friert den CIFS-Ringpuffer kurz ein und liefert ein begrenztes Ende."""
    instance = Path(trace_root) / "instances" / _TRACE_INSTANCE_NAME
    tracing_on_path = instance / "tracing_on"
    trace_path = instance / "trace"
    result: dict = {
        "available": False,
        "instance": str(instance),
        "tracing_was_on": False,
        "line_count": 0,
        "tail": "",
        "pending_mid_matches": [],
        "pending_mid_analysis": [],
        "pending_thread_matches": [],
        "decrypt_failures": [],
        "credit_events": [],
        "kretprobe_filter": _read_limited(
            instance / "events" / _KRETPROBE_GROUP / _KRETPROBE_EVENT / "filter",
            4_096,
        ),
        "kretprobe_profile": _read_limited(
            Path(trace_root) / "kprobe_profile",
            16_384,
        ),
        "trace_stats": _read_limited(instance / "trace_stat", 16_384),
        "error": "",
    }
    previous = ""
    try:
        previous = tracing_on_path.read_text(encoding="ascii").strip()
        result["tracing_was_on"] = previous == "1"
        _write_control(tracing_on_path, "0\n")

        tail: deque[str] = deque(maxlen=_TRACE_TAIL_LINES)
        matches: deque[str] = deque(maxlen=_TRACE_MID_MATCH_LIMIT)
        thread_matches: deque[str] = deque(maxlen=_TRACE_MID_MATCH_LIMIT)
        decrypt_failures: deque[str] = deque(maxlen=_TRACE_DIAGNOSTIC_MATCH_LIMIT)
        credit_events: deque[str] = deque(maxlen=_TRACE_DIAGNOSTIC_MATCH_LIMIT)
        mids = {item["mid"] for item in pending_mids}
        thread_ids = {item["pid"] for item in pending_mids}
        with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                result["line_count"] += 1
                clean = line.rstrip("\n")
                tail.append(clean)
                if mids and _trace_line_matches_mid(clean, mids):
                    matches.append(clean)
                if any(
                    re.search(rf"(?:-|\s){thread_id}\s+\[", clean) is not None
                    or f"pid={thread_id}" in clean
                    for thread_id in thread_ids
                ):
                    thread_matches.append(clean)
                if f"{_KRETPROBE_EVENT}:" in clean:
                    decrypt_failures.append(clean)
                if any(
                    f"{event}:" in clean
                    for event in (
                        "smb3_pend_credits",
                        "smb3_hdr_credits",
                        "smb3_add_credits",
                        "smb3_reconnect_with_invalid_credits",
                        "smb3_overflow_credits",
                        "smb3_too_many_credits",
                    )
                ):
                    credit_events.append(clean)
        result["tail"] = "\n".join(tail)
        result["pending_mid_matches"] = list(matches)
        result["pending_thread_matches"] = list(thread_matches)
        result["decrypt_failures"] = list(decrypt_failures)
        result["credit_events"] = list(credit_events)
        result["pending_mid_analysis"] = [
            {
                **item,
                "trace_lines": [
                    line for line in matches if _trace_line_matches_mid(line, {item["mid"]})
                ],
                "command_enter_seen": any(
                    "smb3_cmd_enter:" in line
                    for line in matches
                    if _trace_line_matches_mid(line, {item["mid"]})
                ),
                "command_done_seen": any(
                    "smb3_cmd_done:" in line
                    for line in matches
                    if _trace_line_matches_mid(line, {item["mid"]})
                ),
                "command_error_seen": any(
                    "smb3_cmd_err:" in line
                    for line in matches
                    if _trace_line_matches_mid(line, {item["mid"]})
                ),
            }
            for item in pending_mids
        ]
        result["available"] = True
    except OSError as exc:
        result["error"] = _error_text(exc)
    finally:
        if previous == "1":
            try:
                _write_control(tracing_on_path, "1\n")
            except OSError as exc:
                resume_error = _error_text(exc)
                result["resume_error"] = resume_error
                result["available"] = False
    return result


def _parse_uid(status: str) -> int | None:
    for line in status.splitlines():
        if line.startswith("Uid:"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _process_start_time(stat_raw: str) -> str:
    comm_end = stat_raw.rfind(")")
    if comm_end < 0:
        return ""
    fields = stat_raw[comm_end + 1 :].split()
    return fields[19] if len(fields) > 19 else ""


def _read_cmdline(path: Path) -> str:
    try:
        raw = path.read_bytes()[:32_769]
    except OSError as exc:
        raise ForensicsTargetError(f"cmdline nicht lesbar: {_error_text(exc)}") from exc
    if len(raw) > 32_768:
        raise ForensicsTargetError("cmdline ist unerwartet gross")
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _validate_worker_target(
    pid: int,
    tid: int,
    *,
    proc_root: Path,
    expected_uid: int,
) -> dict[str, str | int]:
    if not (0 < pid <= _MAX_PID and 0 < tid <= _MAX_PID):
        raise ForensicsTargetError("PID/TID ausserhalb des gueltigen Bereichs")
    process_dir = proc_root / str(pid)
    task_dir = process_dir / "task" / str(tid)
    if not task_dir.is_dir():
        raise ForensicsTargetError("TID gehoert nicht zum angegebenen Prozess")

    status_result = _read_limited(process_dir / "status", 32_768)
    if not status_result["available"]:
        raise ForensicsTargetError(f"Prozessstatus nicht lesbar: {status_result['error']}")
    status = str(status_result["raw"])
    actual_uid = _parse_uid(status)
    if actual_uid != expected_uid:
        raise ForensicsTargetError(
            f"Prozess-UID {actual_uid!r} entspricht nicht KIara-UID {expected_uid}"
        )

    cgroup_result = _read_limited(process_dir / "cgroup", 16_384)
    cgroup = str(cgroup_result["raw"])
    if not cgroup_result["available"] or not any(
        line.rstrip().endswith(f"/{_WORKER_UNIT}") for line in cgroup.splitlines()
    ):
        raise ForensicsTargetError(f"Prozess gehoert nicht zu {_WORKER_UNIT}")

    cmdline = _read_cmdline(process_dir / "cmdline")
    if "serve-worker" not in cmdline:
        raise ForensicsTargetError("Prozess ist kein KIara serve-worker")

    stat_result = _read_limited(process_dir / "stat", 8_192)
    if not stat_result["available"]:
        raise ForensicsTargetError(f"Prozess-stat nicht lesbar: {stat_result['error']}")
    start_time = _process_start_time(str(stat_result["raw"]))
    if not start_time:
        raise ForensicsTargetError("Prozess-Startzeit konnte nicht gelesen werden")
    return {
        "pid": pid,
        "tid": tid,
        "uid": actual_uid,
        "cgroup": cgroup,
        "cmdline": cmdline,
        "start_time_ticks": start_time,
    }


def _decode_syscall(raw: str) -> dict:
    result: dict = {"raw": raw, "number": None, "name": "", "fd": None}
    fields = raw.split()
    if not fields:
        return result
    try:
        number = int(fields[0], 0)
    except ValueError:
        return result
    result["number"] = number
    result["name"] = _READ_SYSCALLS.get(number, "")
    if number in _READ_SYSCALLS and len(fields) > 1:
        try:
            result["fd"] = int(fields[1], 0)
        except ValueError:
            pass
    if number in (0, 17) and len(fields) > 3:
        try:
            result["requested_bytes"] = int(fields[3], 0)
        except ValueError:
            pass
    return result


def _capture_fd_context(process_dir: Path, syscall: dict) -> dict:
    fd = syscall.get("fd")
    if not isinstance(fd, int) or fd < 0:
        return {"available": False, "error": "Aktueller Syscall hat keinen lesbaren Datei-FD"}
    fd_path = process_dir / "fd" / str(fd)
    context: dict = {
        "available": False,
        "fd": fd,
        "target": "",
        "fdinfo": _read_limited(process_dir / "fdinfo" / str(fd), 16_384),
        "error": "",
    }
    try:
        context["target"] = os.readlink(fd_path)
        file_stat = os.stat(fd_path)
        context["stat"] = {
            "mode": stat.filemode(file_stat.st_mode),
            "size": file_stat.st_size,
            "inode": file_stat.st_ino,
            "device": file_stat.st_dev,
            "mtime_ns": file_stat.st_mtime_ns,
        }
        context["available"] = True
    except OSError as exc:
        context["error"] = _error_text(exc)
    return context


def _capture_mountinfo(process_dir: Path, fd_context: dict) -> dict:
    fdinfo = fd_context.get("fdinfo") or {}
    raw_fdinfo = str(fdinfo.get("raw") or "") if isinstance(fdinfo, dict) else ""
    match = re.search(r"^mnt_id:\s*(\d+)\s*$", raw_fdinfo, flags=re.MULTILINE)
    mount_id = match.group(1) if match else ""
    mountinfo = _read_limited(process_dir / "mountinfo", 1_048_576)
    result: dict = {
        "available": bool(mountinfo["available"]),
        "mount_id": int(mount_id) if mount_id else None,
        "matching_lines": [],
        "error": mountinfo["error"],
    }
    if mountinfo["available"] and mount_id:
        result["matching_lines"] = [
            line
            for line in str(mountinfo["raw"]).splitlines()
            if line.split(maxsplit=1)[0] == mount_id
        ]
    return result


def _run_command(
    args: list[str],
    *,
    timeout: float = 8.0,
    max_chars: int = 131_072,
) -> dict:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "raw": "", "error": _error_text(exc)}
    raw = completed.stdout or ""
    if completed.stderr:
        raw = f"{raw}\n[stderr]\n{completed.stderr}" if raw else completed.stderr
    truncated = len(raw) > max_chars
    if truncated:
        raw = raw[-max_chars:]
        raw = "[truncated to tail]\n" + raw
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "raw": raw.strip(),
        "error": "" if completed.returncode == 0 else f"returncode={completed.returncode}",
        "truncated": truncated,
    }


def _capture_smb_sockets() -> dict:
    result = _run_command(["ss", "-tinpe"], max_chars=262_144)
    if not result["available"]:
        return result
    lines = str(result["raw"]).splitlines()
    selected: list[str] = []
    if lines:
        selected.append(lines[0])
    include_continuation = False
    for line in lines[1:]:
        is_connection = bool(line and not line[0].isspace())
        if is_connection:
            include_continuation = ":445" in line
        if include_continuation:
            selected.append(line)
    result["raw"] = "\n".join(selected)
    return result


def _capture_relevant_dmesg() -> dict:
    result = _run_command(["dmesg", "--ctime"], max_chars=1_048_576)
    if not result["available"]:
        return result
    markers = (
        "cifs",
        "smb",
        "netfs",
        "hung task",
        "blocked for more than",
        "tcp",
        "virtio",
        "i/o error",
    )
    relevant = [
        line for line in str(result["raw"]).splitlines() if any(m in line.lower() for m in markers)
    ]
    result["raw"] = "\n".join(relevant[-400:])
    return result


def _capture_pcap_ring(
    *,
    pcap_dir: str | os.PathLike[str] = _PCAP_DIR,
) -> dict:
    """Inventarisiert den root-only SMB-PCAP-Ring ohne Paketdaten zu kopieren."""
    directory = Path(pcap_dir)
    result: dict = {
        "available": False,
        "service": _PCAP_SERVICE,
        "directory": str(directory),
        "files": [],
        "total_bytes": 0,
        "error": "",
        "service_status": _run_command(
            [
                "systemctl",
                "show",
                _PCAP_SERVICE,
                "--property=ActiveState,SubState,MainPID,ExecMainStartTimestamp",
            ],
            timeout=4,
            max_chars=16_384,
        ),
    }
    try:
        directory_stat = directory.lstat()
        if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
            result["error"] = "PCAP-Pfad ist kein echtes Verzeichnis"
            return result

        files: list[dict] = []
        for path in directory.iterdir():
            if not path.name.startswith(_PCAP_PREFIX):
                continue
            file_stat = path.lstat()
            if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
                continue
            files.append(
                {
                    "path": str(path),
                    "size": file_stat.st_size,
                    "mtime_ns": file_stat.st_mtime_ns,
                    "inode": file_stat.st_ino,
                }
            )
        files.sort(key=lambda item: (item["mtime_ns"], item["path"]), reverse=True)
        result["files"] = files[:_PCAP_FILE_LIMIT]
        result["total_bytes"] = sum(item["size"] for item in files)
        result["available"] = bool(files)
        if not files:
            result["error"] = "Noch keine Client-SMB-PCAP-Datei vorhanden"
    except OSError as exc:
        result["error"] = _error_text(exc)
    return result


def collect_ingestion_stall_forensics(
    pid: int,
    tid: int,
    *,
    proc_root: str | os.PathLike[str] = "/proc",
    trace_root: str | os.PathLike[str] = "/sys/kernel/tracing",
    expected_uid: int | None = None,
) -> dict:
    """Sammelt Root-Kontext fuer genau einen validierten Worker-Thread."""
    if expected_uid is None:
        try:
            expected_uid = pwd.getpwnam("kiara").pw_uid
        except KeyError as exc:
            raise ForensicsTargetError("System-User kiara existiert nicht") from exc

    proc = Path(proc_root)
    identity = _validate_worker_target(
        int(pid),
        int(tid),
        proc_root=proc,
        expected_uid=int(expected_uid),
    )
    process_dir = proc / str(pid)
    task_dir = process_dir / "task" / str(tid)
    thread_files = {
        name: _read_limited(task_dir / name, max_chars)
        for name, max_chars in (
            ("status", 32_768),
            ("stat", 8_192),
            ("wchan", 4_096),
            ("stack", 32_768),
            ("syscall", 8_192),
            ("sched", 65_536),
            ("io", 16_384),
        )
    }
    syscall = _decode_syscall(str(thread_files["syscall"].get("raw") or ""))
    fd_context = _capture_fd_context(process_dir, syscall)

    from kiara.web.ingestion_stall_detection import read_cifs_proc_diagnostics

    cifs_proc = read_cifs_proc_diagnostics(proc_root=proc)
    debug_data = str(cifs_proc.get("DebugData", {}).get("raw") or "")
    pending_mids = _parse_pending_mids(debug_data)
    pending_for_thread = [item for item in pending_mids if item["pid"] == int(tid)]

    result = {
        "available": True,
        "captured_at": _utc_now_iso(),
        "kernel": platform.uname()._asdict(),
        "boot_id": _read_limited(proc / "sys" / "kernel" / "random" / "boot_id", 256),
        "uptime": _read_limited(proc / "uptime", 256),
        "target": identity,
        "thread_files": thread_files,
        "syscall": syscall,
        "fd": fd_context,
        "mountinfo": _capture_mountinfo(process_dir, fd_context),
        "cifs_proc": cifs_proc,
        "pending_mids": pending_mids,
        "pending_mids_for_thread": pending_for_thread,
        "cifs_trace": read_cifs_flight_recorder(pending_mids, trace_root=trace_root),
        "smb_pcap_ring": _capture_pcap_ring(),
        "smb_sockets": _capture_smb_sockets(),
        "kernel_messages": _capture_relevant_dmesg(),
    }

    final_stat = _read_limited(process_dir / "stat", 8_192)
    final_start_time = (
        _process_start_time(str(final_stat["raw"])) if final_stat["available"] else ""
    )
    result["target_stable"] = bool(
        final_start_time and final_start_time == identity["start_time_ticks"]
    )
    if not result["target_stable"]:
        result["available"] = False
        result["target_stability_error"] = "Prozess verschwand oder PID wurde wiederverwendet"
    return result


def request_privileged_ingestion_forensics(
    pid: int,
    tid: int,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict:
    """Ruft den eng freigegebenen Root-Collector aus dem Worker heraus auf."""
    args = [
        _SUDO_PATH,
        "-n",
        _CLI_PATH,
        "collect-ingestion-stall-forensics",
        str(int(pid)),
        str(int(tid)),
    ]
    try:
        completed = runner(
            args,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": _error_text(exc), "command": args[2:]}
    if completed.returncode != 0:
        return {
            "available": False,
            "error": f"Collector returncode={completed.returncode}",
            "stderr": (completed.stderr or "")[-4_096:],
            "stdout": (completed.stdout or "")[-4_096:],
            "command": args[2:],
        }
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            "available": False,
            "error": f"Ungueltige Collector-Antwort: {_error_text(exc)}",
            "stdout": (completed.stdout or "")[-4_096:],
            "command": args[2:],
        }
    if not isinstance(payload, dict):
        return {"available": False, "error": "Collector-Antwort ist kein JSON-Objekt"}
    return payload
