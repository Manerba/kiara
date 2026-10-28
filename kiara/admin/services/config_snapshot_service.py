"""Config-Snapshots: versionierte Staende von /etc/kiara/kiara.yaml.

Leichtgewichtige Ergaenzung zum Voll-Backup (kiara/backup.py): sichert nur die
Konfigurationsdatei, dafuer sekundenschnell, mit benennbaren Staenden, Diff und
Restore. Benchmark-Ergebnisse verknuepfen sich automatisch ueber den
config_file_sha256 im reproducibility-Block der Summary-Artefakte — derselbe
Hash identifiziert den Snapshot (kein Fremdschluessel, keine Doppelpflege).
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path("/var/lib/kiara/config-snapshots")
INDEX_FILE = SNAPSHOT_DIR / "index.jsonl"
BENCHMARK_ROOT = Path("/var/lib/kiara/benchmarks")

_SECRET_KEY_PATTERN = re.compile(r"password|secret|api_key|apikey|token|dsn", re.IGNORECASE)
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_index_lock = threading.Lock()


class ConfigSnapshotError(RuntimeError):
    """Fehler in der Snapshot-Verwaltung (ungueltige Referenz, Validierung, IO)."""


def _config_path() -> Path:
    from kiara.config import get_config_file_path

    return Path(get_config_file_path())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_dirs() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(SNAPSHOT_DIR, 0o750)


def _read_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    entries: list[dict] = []
    with INDEX_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Config-Snapshot-Index: ungueltige Zeile uebersprungen")
    return entries


def _write_index(entries: list[dict]) -> None:
    tmp = INDEX_FILE.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.chmod(tmp, 0o640)
    tmp.replace(INDEX_FILE)


def _append_index(entry: dict) -> None:
    with INDEX_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.chmod(INDEX_FILE, 0o640)


def _kiara_version() -> str:
    try:
        import kiara

        return getattr(kiara, "__version__", "unbekannt")
    except Exception:
        return "unbekannt"


def create_snapshot(
    label: str = "", comment: str = "", trigger: str = "manual", user: str = ""
) -> dict:
    """Sichert den aktuellen Config-Stand als Snapshot.

    Bei trigger="auto"/"pre_restore" wird dedupliziert: Ist der Inhalt identisch
    mit dem neuesten Index-Eintrag, entsteht kein neuer Snapshot. Manuelle
    Snapshots werden immer angelegt (der Nutzer will einen benannten Marker).
    """
    if label and not _LABEL_PATTERN.match(label):
        raise ConfigSnapshotError(
            "Ungueltiges Label (erlaubt: Buchstaben, Ziffern, . _ -, max 64 Zeichen)"
        )
    path = _config_path()
    if not path.exists():
        raise ConfigSnapshotError(f"Config-Datei nicht gefunden: {path}")
    data = path.read_bytes()
    sha = _sha256_bytes(data)

    with _index_lock:
        _ensure_dirs()
        entries = _read_index()
        if trigger != "manual" and entries and entries[-1].get("config_sha256") == sha:
            return {**entries[-1], "deduplicated": True}

        ts = _utc_ts()
        suffix = f"_{label}" if label else ""
        filename = f"{ts}_{sha[:8]}{suffix}.yaml"
        target = SNAPSHOT_DIR / filename
        target.write_bytes(data)
        os.chmod(target, 0o640)
        entry = {
            "timestamp": ts,
            "label": label,
            "file": filename,
            "config_sha256": sha,
            "kiara_version": _kiara_version(),
            "trigger": trigger,
            "comment": comment,
            "user": user,
        }
        _append_index(entry)
        _enforce_retention_locked([*entries, entry])
    logger.info("Config-Snapshot angelegt: %s (trigger=%s)", filename, trigger)
    return entry


def auto_snapshot(trigger: str = "auto") -> dict | None:
    """Best-effort-Snapshot vor Config-Schreiboperationen (blockiert nie)."""
    try:
        return create_snapshot(trigger=trigger)
    except Exception as exc:
        logger.warning("Auto-Config-Snapshot fehlgeschlagen (Schreiben laeuft weiter): %s", exc)
        return None


def _max_auto_snapshots() -> int:
    try:
        from kiara.config import load_config

        raw = (load_config().get("backup") or {}).get("max_config_snapshots", 50)
        return max(1, int(raw))
    except Exception:
        return 50


def _enforce_retention_locked(entries: list[dict]) -> None:
    """Entfernt aelteste UNBENANNTE Snapshots ueber dem Limit (Lock gehalten)."""
    unnamed = [e for e in entries if not e.get("label")]
    excess = len(unnamed) - _max_auto_snapshots()
    if excess <= 0:
        return
    to_remove = {e["file"] for e in unnamed[:excess]}
    for entry in entries:
        if entry["file"] in to_remove:
            try:
                (SNAPSHOT_DIR / entry["file"]).unlink(missing_ok=True)
            except OSError:
                logger.warning("Snapshot-Datei nicht loeschbar: %s", entry["file"])
    _write_index([e for e in entries if e["file"] not in to_remove])


def _resolve(ref: str, entries: list[dict] | None = None) -> dict:
    """Loest eine Referenz (Label, sha256-Praefix ab 8 Zeichen oder Dateiname)."""
    ref = (ref or "").strip()
    if not ref:
        raise ConfigSnapshotError("Leere Snapshot-Referenz")
    entries = entries if entries is not None else _read_index()
    matches = [e for e in entries if e.get("label") == ref or e.get("file") == ref]
    if not matches and len(ref) >= 8:
        matches = [e for e in entries if e.get("config_sha256", "").startswith(ref.lower())]
    if not matches:
        raise ConfigSnapshotError(f"Kein Snapshot gefunden fuer Referenz: {ref!r}")
    # Bei Mehrdeutigkeit (gleicher Hash mehrfach) gewinnt der neueste Eintrag.
    return matches[-1]


def _benchmark_runs_by_sha() -> dict[str, list[str]]:
    """Mappt config_file_sha256 -> Benchmark-run_labels (Scan der Summaries)."""
    runs: dict[str, list[str]] = {}
    if not BENCHMARK_ROOT.exists():
        return runs
    for summary in BENCHMARK_ROOT.glob("*/results/*_summary.json"):
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sha = (payload.get("reproducibility") or {}).get("config_file_sha256")
        if not sha:
            continue
        label = (payload.get("run") or {}).get("run_label") or summary.stem.removesuffix("_summary")
        runs.setdefault(sha, []).append(label)
    return runs


def list_snapshots(with_runs: bool = True) -> list[dict]:
    """Alle Snapshots, neueste zuerst; optional mit verknuepften Benchmark-Laeufen."""
    entries = _read_index()
    current_sha = None
    try:
        current_sha = _sha256_bytes(_config_path().read_bytes())
    except OSError:
        pass
    runs = _benchmark_runs_by_sha() if with_runs else {}
    result = []
    for entry in reversed(entries):
        item = dict(entry)
        item["is_current"] = entry.get("config_sha256") == current_sha
        if with_runs:
            item["benchmark_runs"] = sorted(set(runs.get(entry.get("config_sha256", ""), [])))
        result.append(item)
    return result


def set_label(ref: str, label: str, comment: str | None = None) -> dict:
    """Setzt Label (und optional Kommentar) eines bestehenden Snapshots."""
    if label and not _LABEL_PATTERN.match(label):
        raise ConfigSnapshotError(
            "Ungueltiges Label (erlaubt: Buchstaben, Ziffern, . _ -, max 64 Zeichen)"
        )
    with _index_lock:
        entries = _read_index()
        entry = _resolve(ref, entries)
        entry["label"] = label
        if comment is not None:
            entry["comment"] = comment
        _write_index(entries)
    return entry


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), sub, out)
    else:
        out[prefix] = value


def _masked(key: str, value) -> str:
    if _SECRET_KEY_PATTERN.search(key):
        return "***" if value not in (None, "", []) else ""
    return json.dumps(value, ensure_ascii=False, default=str)


def diff_snapshots(ref: str, against: str = "current") -> list[dict]:
    """Key-Diff zweier Staende (Werte von Secret-Feldern maskiert).

    against="current" vergleicht gegen die aktive Config-Datei.
    """
    import yaml

    entry = _resolve(ref)
    a = yaml.safe_load((SNAPSHOT_DIR / entry["file"]).read_text(encoding="utf-8")) or {}
    if against == "current":
        b = yaml.safe_load(_config_path().read_text(encoding="utf-8")) or {}
        b_name = "current"
    else:
        other = _resolve(against)
        b = yaml.safe_load((SNAPSHOT_DIR / other["file"]).read_text(encoding="utf-8")) or {}
        b_name = other["file"]

    flat_a: dict = {}
    flat_b: dict = {}
    _flatten("", a, flat_a)
    _flatten("", b, flat_b)
    changes = []
    for key in sorted(set(flat_a) | set(flat_b)):
        va, vb = flat_a.get(key, "<fehlt>"), flat_b.get(key, "<fehlt>")
        if va != vb:
            changes.append(
                {
                    "key": key,
                    "snapshot": _masked(key, va) if va != "<fehlt>" else "<fehlt>",
                    "against": _masked(key, vb) if vb != "<fehlt>" else "<fehlt>",
                }
            )
    return [{"against": b_name}, *changes] if changes else []


def restore_snapshot(ref: str, user: str = "") -> dict:
    """Stellt einen Snapshot als aktive Config wieder her (fail-closed).

    Ablauf: Ziel-Stand mit der vollen Config-Validierung laden (inkompatible
    Altstaende werden abgewiesen statt halb geschrieben), aktuellen Stand als
    pre_restore-Snapshot sichern, dann via save_config schreiben. Ein noetiger
    Service-Restart ist Sache des Aufrufers.
    """
    from kiara.config import load_config, save_config

    entry = _resolve(ref)
    snapshot_path = SNAPSHOT_DIR / entry["file"]
    if not snapshot_path.exists():
        raise ConfigSnapshotError(f"Snapshot-Datei fehlt: {entry['file']}")

    try:
        target_config = load_config(str(snapshot_path))
    except Exception as exc:
        raise ConfigSnapshotError(
            f"Snapshot besteht die Config-Validierung nicht (Restore abgelehnt): {exc}"
        ) from exc

    pre = create_snapshot(trigger="pre_restore", user=user, comment=f"vor Restore von {ref}")
    save_config(target_config)
    logger.info("Config-Restore: %s durch %s (pre_restore=%s)", entry["file"], user, pre["file"])
    return {"restored": entry, "pre_restore_snapshot": pre["file"]}


_RESTART_SERVICES = [
    "kiara.service",
    "kiara-admin.service",
    "kiara-worker.service",
    "kiara-mcp.service",
]
_restart_last = 0.0


def trigger_service_restart(user: str = "") -> dict:
    """Detachter systemctl-Restart der Kern-Services (Muster admin_app)."""
    global _restart_last
    import subprocess

    now = time.monotonic()
    if now - _restart_last < 10:
        raise ConfigSnapshotError("Restart bereits ausgeloest, bitte kurz warten.")
    _restart_last = now
    proc = subprocess.Popen(["systemctl", "restart", *_RESTART_SERVICES])
    logger.info("Service-Restart nach Config-Restore durch %s (pid=%s)", user, proc.pid)
    return {"restarting": _RESTART_SERVICES}
