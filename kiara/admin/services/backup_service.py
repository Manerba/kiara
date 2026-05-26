"""
Backup-Service — extrahiert aus api_backup.py.
Verwaltet Background-Backup-Erstellung, Status-Tracking und Restore.
"""

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Shared state fuer Background-Backup-Thread
_backup_status: dict = {"running": False, "result": None, "error": None}
_backup_lock = threading.Lock()


def is_running() -> bool:
    """Prüft ob ein Backup-Vorgang läuft."""
    with _backup_lock:
        return _backup_status["running"]


def get_status() -> dict:
    """Gibt den aktuellen Backup-Status zurück."""
    with _backup_lock:
        return dict(_backup_status)


def list_backups() -> list[dict]:
    """Listet alle verfügbaren Backups auf, sortiert nach Datum (neueste zuerst).

    Feldnamen identisch zu api_backup.py list_backups (Frontend-Kompatibilität).
    """
    from kiara.config import get_backup_dir, load_config

    cfg = load_config()
    backup_dir = get_backup_dir(cfg)

    if not os.path.isdir(backup_dir):
        return []

    import datetime

    backups = []
    for f in sorted(
        Path(backup_dir).glob("*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        stat = f.stat()
        backups.append(
            {
                "filename": f.name,
                "size_bytes": stat.st_size,
                "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        )
    return backups


def start_backup(*, include_vectordb_data: bool = False) -> None:
    """Startet ein Backup in einem Background-Thread.

    Raises: RuntimeError wenn bereits ein Backup läuft.
    """
    with _backup_lock:
        if _backup_status["running"]:
            raise RuntimeError("Ein Backup läuft bereits")
        _backup_status.update({"running": True, "result": None, "error": None})

    thread = threading.Thread(
        target=_run_backup,
        kwargs={"include_vectordb_data": include_vectordb_data},
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        with _backup_lock:
            _backup_status["running"] = False
        raise


def _run_backup(*, include_vectordb_data: bool) -> None:
    """Background-Thread: Erstellt ein tar.gz-Backup.

    Logik extrahiert aus api_backup.py create_backup_endpoint (Zeile 38,
    Thread-Funktion run_backup Zeile 49-61).
    """
    try:
        from kiara.backup import create_backup
        from kiara.config import load_config

        cfg = load_config()
        result_path = create_backup(cfg, include_vectordb_data=include_vectordb_data)
        stat = Path(result_path).stat()

        with _backup_lock:
            _backup_status["result"] = {
                "filename": Path(result_path).name,
                "size_bytes": stat.st_size,
            }
    except Exception as exc:
        logger.exception("Backup fehlgeschlagen")
        with _backup_lock:
            _backup_status["error"] = str(exc)
            _backup_status["result"] = None
    finally:
        with _backup_lock:
            _backup_status["running"] = False


def restore_backup(*, filename: str, force: bool = False) -> dict:
    """Stellt ein Backup wieder her.

    Raises: ValueError bei ungültigem Dateinamen oder fehlender Datei.
    Logik extrahiert aus api_backup.py restore_backup_endpoint (Zeile 74,
    Validierung + Restore Zeile 86-92).
    """
    from kiara.config import get_backup_dir, load_config

    cfg = load_config()
    backup_dir = get_backup_dir(cfg)

    # Path-Traversal-Schutz
    backup_path = Path(backup_dir) / filename
    if not backup_path.resolve().is_relative_to(Path(backup_dir).resolve()):
        raise ValueError("Ungültiger Dateiname (Path Traversal)")

    if not backup_path.exists():
        raise ValueError(f"Backup-Datei nicht gefunden: {filename}")

    # Validierung + Restore (aus kiara.backup — wie api_backup.py Zeile 86-92)
    from kiara.backup import restore_backup as do_restore
    from kiara.backup import validate_backup

    valid, _manifest, errors = validate_backup(str(backup_path))
    if not valid:
        raise ValueError(f"Validierung fehlgeschlagen: {'; '.join(errors)}")

    result = do_restore(str(backup_path), cfg, force=force)
    if not result.get("success"):
        detail = "; ".join(result.get("issues", []))
        if detail:
            raise ValueError(f"{result.get('message', 'Restore fehlgeschlagen')}: {detail}")
        raise ValueError(result.get("message", "Restore fehlgeschlagen"))
    return {"message": "Backup wiederhergestellt", "restored": result.get("restored", [])}
