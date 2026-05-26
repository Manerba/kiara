"""KIara - Cron-Schedule-Verwaltung.

Verwaltet /etc/cron.d/kiara fuer automatische Ingestion-Schedules.
Jede Datenquelle kann einen eigenen Cron-Ausdruck haben.

Dieses Modul liegt im shared-Paket, damit es sowohl von Core-Modulen
(z.B. ingestion_worker) als auch von Web-Modulen importiert werden kann,
ohne eine Layer-Verletzung zu erzeugen.
"""

import fcntl
import logging
import os
import re
import tempfile
import threading
from contextlib import contextmanager

from kiara.shared.constants import CRON_LOCK_PATH

logger = logging.getLogger(__name__)

_cron_lock = threading.Lock()

CRON_FILE = "/etc/cron.d/kiara"
MARKER_PREFIX = "# kiara-ds-"
KIARA_BIN = "/opt/kiara/venv/bin/kiara"
CRON_USER = "kiara"


@contextmanager
def _cross_process_lock():
    """Blockierendes Inter-Process-Lock via fcntl.flock.

    Ergaenzt den prozess-internen ``_cron_lock`` (threading.Lock) um
    Synchronisation zwischen mehreren Gunicorn/uvicorn-Workern. Ohne
    dieses Lock koennten zwei Worker-Prozesse ``write_schedule`` parallel
    ausfuehren und Schedules ueberschreiben (read-modify-write-Race).
    """
    os.makedirs(os.path.dirname(CRON_LOCK_PATH), exist_ok=True)
    with open(CRON_LOCK_PATH, "a") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


INTERVAL_PRESETS = {
    "30min": "*/30 * * * *",
    "1h": "0 * * * *",
    "2h": "0 */2 * * *",
    "6h": "0 */6 * * *",
    "12h": "0 */12 * * *",
    "24h": "0 3 * * *",
}


_CRON_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]


def _valid_cron_field(token: str, lo: int, hi: int) -> bool:
    """Prueft ob ein Cron-Feld (Zahl, Range, Step, Liste) im Wertebereich liegt."""
    for item in token.split(","):
        step_parts = item.split("/", 1)
        has_step = len(step_parts) == 2
        if has_step and (not step_parts[1].isdigit() or int(step_parts[1]) < 1):
            return False
        base = step_parts[0]
        if base == "*":
            continue
        if "-" in base:
            rp = base.split("-", 1)
            if not rp[0].isdigit() or not rp[1].isdigit():
                return False
            a, b = int(rp[0]), int(rp[1])
            if a < lo or b > hi or a > b:
                return False
        elif not base.isdigit() or not (lo <= int(base) <= hi):
            return False
        elif has_step:
            # Cron-Spec: M/N ist ungueltig — nur */N oder M-N/S erlaubt.
            # Vixie-Cron interpretiert M/N implizit als M-(field-max)/N,
            # andere cron-Implementierungen lehnen es stillschweigend ab.
            return False
    return True


def validate_cron_expr(expr: str) -> bool:
    """Validiert einen Cron-Ausdruck (5 Felder).

    Prueft Struktur (5 Felder), erlaubte Zeichen und semantische
    Gueltigkeit (Zahlenbereiche: minute 0-59, hour 0-23, dom 1-31,
    month 1-12, dow 0-7).
    """
    if not expr or not expr.strip():
        return False
    if "\n" in expr or "\r" in expr:
        return False
    parts = expr.strip().split()
    if len(parts) != 5:
        return False
    pattern = re.compile(r"^[\d\*\/\-\,]+$")
    if not all(pattern.match(p) for p in parts):
        return False
    return all(
        _valid_cron_field(p, lo, hi) for p, (lo, hi) in zip(parts, _CRON_RANGES, strict=False)
    )


def read_schedules() -> dict[int, dict]:
    """Liest alle Cron-Schedules aus der Cron-Datei.

    Returns:
        Dict {ds_id: {"name": str, "cron_expr": str, "line": str}}.
    """
    schedules = {}
    if not os.path.isfile(CRON_FILE):
        return schedules

    try:
        with open(CRON_FILE, "r") as f:
            lines = f.readlines()
    except OSError as exc:
        logger.error("Cron-Datei nicht lesbar: %s", exc)
        return schedules

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith(MARKER_PREFIX):
            # Format: # kiara-ds-{id}: {name}
            try:
                rest = line[len(MARKER_PREFIX) :]
                id_str, name = rest.split(":", 1)
                ds_id = int(id_str.strip())
                name = name.strip()
                # Naechste Zeile ist der Cron-Eintrag
                if i + 1 < len(lines) and not lines[i + 1].startswith("#"):
                    cron_line = lines[i + 1].rstrip()
                    # Cron-Ausdruck extrahieren (erste 5 Felder)
                    parts = cron_line.split()
                    if len(parts) >= 6:
                        cron_expr = " ".join(parts[:5])
                        schedules[ds_id] = {
                            "name": name,
                            "cron_expr": cron_expr,
                            "line": cron_line,
                        }
                    i += 2
                    continue
            except (ValueError, IndexError):
                pass
        i += 1

    return schedules


def _get_ds_embedding_model(ds_id: int) -> str:
    """Ermittelt das zuletzt verwendete Embedding-Modell einer Datenquelle.

    Returns:
        Modellname oder leerer String wenn keine Ingestion bekannt.
    """
    try:
        from kiara.repositories.datasource import DatasourceEmbeddingRepository

        ds_emb_repo = DatasourceEmbeddingRepository()
        return ds_emb_repo.get_last_model_for_datasource(ds_id)
    except Exception as exc:
        logger.warning(
            "Embedding-Modell fuer DS %s konnte nicht ermittelt werden: %s",
            ds_id,
            exc,
        )
        return ""


def _get_default_embedding_model() -> str:
    """Ermittelt das aktuell konfigurierte Embedding-Modell fuer neue DS."""
    try:
        from kiara.config import get_embedding_model, load_config

        return get_embedding_model(load_config())
    except Exception as exc:
        logger.warning(
            "Default-Embedding-Modell konnte nicht ermittelt werden: %s",
            exc,
        )
        return ""


def write_schedule(ds_id: int, name: str, cron_expr: str) -> None:
    """Setzt oder aktualisiert den Cron-Schedule fuer eine Datenquelle.

    Args:
        ds_id: Datenquellen-ID.
        name: Name der Datenquelle (fuer den Kommentar).
        cron_expr: Cron-Ausdruck (5 Felder) oder Preset-Name.
    """
    import shlex

    # Preset auflösen
    if cron_expr in INTERVAL_PRESETS:
        cron_expr = INTERVAL_PRESETS[cron_expr]

    if not validate_cron_expr(cron_expr):
        raise ValueError(f"Ungueltiger Cron-Ausdruck: {cron_expr}")

    em = _get_ds_embedding_model(ds_id)
    if not em:
        em = _get_default_embedding_model()
        if em:
            logger.info(
                "Kein zuletzt verwendetes Embedding-Modell fuer DS %s bekannt; "
                "Cron-Schedule nutzt Default-Modell %s",
                ds_id,
                em,
            )
        else:
            raise ValueError(
                f"Kein Embedding-Modell fuer Datenquelle {ds_id} bekannt "
                "und kein aktives Default-Embedding-Modell konfiguriert."
            )
    # shlex.quote() schuetzt vor Shell-Metazeichen, aber NICHT vor CR/LF —
    # ein Modellname mit eingebettetem Newline wuerde eine zusaetzliche
    # aktive Zeile in /etc/cron.d/kiara erzeugen (Cron parst zeilenweise).
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in em):
        raise ValueError(f"Embedding-Modellname enthaelt unerlaubte Steuerzeichen: {em!r}")

    marker = f"{MARKER_PREFIX}{ds_id}: {name.replace(chr(10), '').replace(chr(13), '')}"
    # HINWEIS: Dieses Format ist gekoppelt an das Parsing in
    # _finalize_success() (ingestion_worker.py). Aenderungen am Flag-Namen
    # oder der Reihenfolge muessen dort synchron nachgezogen werden.
    cron_line = (
        f"{cron_expr} {CRON_USER} {shlex.quote(KIARA_BIN)} queue-ingest "
        f"--ds-id {shlex.quote(str(ds_id))} --triggered-by cron "
        f"--embedding-model {shlex.quote(em)}"
    )

    with _cron_lock, _cross_process_lock():
        lines = _read_cron_lines()
        new_lines = _remove_ds_lines(lines, ds_id)

        # Am Ende anfuegen
        new_lines.append(f"\n{marker}\n")
        new_lines.append(f"{cron_line}\n")

        _write_cron_lines(new_lines)
    logger.info("Cron-Schedule gesetzt fuer DS %s: %s", ds_id, cron_expr)


def remove_schedule(ds_id: int) -> None:
    """Entfernt den Cron-Schedule fuer eine Datenquelle."""
    with _cron_lock, _cross_process_lock():
        lines = _read_cron_lines()
        new_lines = _remove_ds_lines(lines, ds_id)
        _write_cron_lines(new_lines)
    logger.info("Cron-Schedule entfernt fuer DS %s", ds_id)


def _read_cron_lines() -> list[str]:
    """Liest die Cron-Datei oder erstellt den Header."""
    if not os.path.isfile(CRON_FILE):
        return [
            "# Managed by KIara - do not edit manually\n",
            "SHELL=/bin/bash\n",
        ]
    try:
        with open(CRON_FILE, "r") as f:
            return f.readlines()
    except OSError:
        return [
            "# Managed by KIara - do not edit manually\n",
            "SHELL=/bin/bash\n",
        ]


def _run_sudo_command(cmd: list[str], *, input_text: str | None = None) -> None:
    import subprocess

    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise OSError(f"{' '.join(cmd)} fehlgeschlagen: {result.stderr.strip()}")


def _write_cron_lines_direct(content: str) -> None:
    """Schreibt CRON_FILE atomar ohne sudo (Tests/root-Fallback)."""
    cron_dir = os.path.dirname(CRON_FILE)
    os.makedirs(cron_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(CRON_FILE)}.",
        suffix=".tmp",
        dir=cron_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        try:
            os.chown(tmp_path, 0, 0)
        except PermissionError:
            pass
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, CRON_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _write_cron_lines(lines: list[str]) -> None:
    """Schreibt die Cron-Datei atomar via sudo tmp + mv."""
    import subprocess

    # Trailing Leerzeilen entfernen
    while lines and lines[-1].strip() == "":
        lines.pop()
    lines.append("\n")  # Cron erwartet Newline am Ende

    content = "".join(lines)
    tmp_file = f"{CRON_FILE}.tmp.{os.getpid()}.{threading.get_ident()}"

    try:
        _run_sudo_command(["sudo", "tee", tmp_file], input_text=content)
        _run_sudo_command(["sudo", "chown", "root:root", tmp_file])
        _run_sudo_command(["sudo", "chmod", "0644", tmp_file])
        _run_sudo_command(["sudo", "mv", tmp_file, CRON_FILE])
    except FileNotFoundError:
        # Fallback: direkt schreiben (z.B. in Tests oder als root)
        _write_cron_lines_direct(content)
    except Exception:
        # Temp-Datei best-effort entfernen; finale Cron-Datei bleibt unveraendert.
        try:
            subprocess.run(
                ["sudo", "rm", "-f", tmp_file],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            pass
        raise


def _remove_ds_lines(lines: list[str], ds_id: int) -> list[str]:
    """Entfernt Marker + folgende Cron-Zeile fuer eine bestimmte DS-ID.

    skip_next darf erst durch die tatsaechliche Cron-Zeile (Nicht-Kommentar)
    zurueckgesetzt werden — bei einem Kommentar zwischen Marker und Cron-Zeile
    wuerde ein vorzeitiges Reset die eigentliche Cron-Zeile erhalten lassen.
    """
    marker = f"{MARKER_PREFIX}{ds_id}:"
    new_lines = []
    skip_next = False
    for line in lines:
        if skip_next and not line.startswith("#"):
            skip_next = False
            continue
        if line.rstrip().startswith(marker):
            skip_next = True
            continue
        new_lines.append(line)
    return new_lines
