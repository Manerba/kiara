"""Backup und Restore fuer KIara."""

import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from kiara.config import (
    get_backup_dir,
    get_config_file_path,
    get_template_dir,
    get_vectordb_data_path,
)
from kiara.shared.constants import BACKUP_TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)


# pg_dump/pg_restore Timeout — grosse DBs brauchen Zeit, aber nicht unendlich
_PG_DUMP_TIMEOUT_SECS = 600
_PG_RESTORE_TIMEOUT_SECS = 600


class BackupError(RuntimeError):
    """Fehler beim Erstellen oder Wiederherstellen eines Backups."""


def _parse_dsn(dsn: str) -> dict:
    """Parst einen PostgreSQL-DSN in die Bestandteile fuer pg_dump/pg_restore.

    Raises:
        BackupError: DSN leer oder nicht parsebar.
    """
    if not dsn:
        raise BackupError(
            "database.postgresql.dsn ist nicht konfiguriert — Backup ohne DB-Dump ist nutzlos"
        )
    parsed = urlparse(dsn)
    if not parsed.hostname or not parsed.path:
        raise BackupError(f"database.postgresql.dsn ist ungueltig: {dsn}")
    return {
        "host": parsed.hostname,
        "port": str(parsed.port or 5432),
        "user": parsed.username or "kiara",
        "password": parsed.password or "",
        "dbname": parsed.path.lstrip("/") or "kiara",
    }


def _check_pg_tool(binary: str) -> None:
    """Prueft ob pg_dump/pg_restore verfuegbar ist.

    Fail-closed: Backup ist ohne DB-Dump nutzlos — fehlendes Tool = harter Fehler.
    """
    if shutil.which(binary) is None:
        raise BackupError(
            f"{binary} nicht gefunden. Bitte postgresql-client installieren "
            f"(z.B. apt install postgresql-client). Backup ohne DB-Dump ist nutzlos."
        )


def _run_pg_dump(dsn_parts: dict, output_path: str) -> None:
    """Fuehrt pg_dump aus und schreibt Custom-Format-Dump nach output_path."""
    _check_pg_tool("pg_dump")
    env = os.environ.copy()
    env["PGPASSWORD"] = dsn_parts["password"]
    cmd = [
        "pg_dump",
        "-h",
        dsn_parts["host"],
        "-p",
        dsn_parts["port"],
        "-U",
        dsn_parts["user"],
        "-d",
        dsn_parts["dbname"],
        "--format=custom",  # kompakt + selektives Restore moeglich
        "--no-owner",  # keine ALTER OWNER TO …
        "--no-privileges",  # keine GRANT/REVOKE
        "-f",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=_PG_DUMP_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackupError(f"pg_dump Timeout nach {_PG_DUMP_TIMEOUT_SECS}s — DB zu gross?") from exc
    if result.returncode != 0:
        raise BackupError(
            f"pg_dump fehlgeschlagen (rc={result.returncode}): {result.stderr.strip()}"
        )


def _run_pg_restore(dsn_parts: dict, dump_path: str) -> None:
    """Stellt einen pg_dump-Custom-Format-Dump wieder her."""
    _check_pg_tool("pg_restore")
    env = os.environ.copy()
    env["PGPASSWORD"] = dsn_parts["password"]
    cmd = [
        "pg_restore",
        "-h",
        dsn_parts["host"],
        "-p",
        dsn_parts["port"],
        "-U",
        dsn_parts["user"],
        "-d",
        dsn_parts["dbname"],
        "--clean",  # DROP vor CREATE
        "--if-exists",  # keine Fehler bei fehlenden Objekten
        "--no-owner",
        "--no-privileges",
        dump_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=_PG_RESTORE_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackupError(
            f"pg_restore Timeout nach {_PG_RESTORE_TIMEOUT_SECS}s — Dump zu gross?"
        ) from exc
    # pg_restore kann non-zero zurueckgeben bei harmlosen Warnings — nur
    # unten definierte Patterns als echten Fehler werten.
    stderr = result.stderr.strip()
    if result.returncode != 0 and "errors ignored on restore" not in stderr.lower():
        raise BackupError(f"pg_restore fehlgeschlagen (rc={result.returncode}): {stderr}")
    if stderr:
        logger.info("pg_restore stderr (nicht-fatale Meldungen): %s", stderr)


def _vectordb_backend(config: dict) -> str:
    return str((config.get("vectordb") or {}).get("backend") or "qdrant").lower()


def _file_vectordb_backup_path(config: dict, include_vectordb_data: bool) -> str | None:
    """Liefert den Dateipfad fuer file-backed VectorDB-Backups oder bricht ab."""
    if not include_vectordb_data:
        return None

    backend = _vectordb_backend(config)
    if backend == "qdrant":
        raise BackupError(
            "VectorDB-Dateibackup ist fuer Qdrant deaktiviert. "
            "Qdrant laeuft als Server-Backend; der alte ChromaDB-Dateipfad ist "
            "kein gueltiges Backup der Qdrant-Collections. Backup ohne "
            "VectorDB-Daten erstellen und Collections nach Restore per Re-Ingest "
            "oder separatem Qdrant-Snapshot wiederherstellen."
        )

    vectordb_data_path = get_vectordb_data_path(config)
    if not vectordb_data_path:
        raise BackupError("vectordb.path ist leer — VectorDB-Dateibackup nicht moeglich")
    if not os.path.isdir(vectordb_data_path):
        raise BackupError(f"VectorDB-Datenpfad existiert nicht: {vectordb_data_path}")
    return vectordb_data_path


def create_backup(
    config: dict, output_path: str = None, include_vectordb_data: bool = False
) -> str:
    """Erstellt ein Backup als tar.gz.

    Inhalt:
    - database.dump (PostgreSQL pg_dump custom format, PFLICHT)
    - kiara.yaml (Config)
    - templates/ (benutzerdefinierte Templates)
    - vectordb_data/ (optional)
    - manifest.json (SHA256-Checksummen, database_included-Flag)

    Fail-closed: Ohne installiertes pg_dump oder erreichbare DB schlaegt
    das Backup fehl — ein Backup ohne DB-Dump ist nutzlos.

    Raises:
        BackupError: pg_dump nicht gefunden, DSN ungueltig oder DB-Fehler.

    Returns:
        Pfad zur erstellten Backup-Datei.
    """
    bak_dir = get_backup_dir(config)
    os.makedirs(bak_dir, exist_ok=True)

    timestamp = datetime.now().strftime(BACKUP_TIMESTAMP_FORMAT)
    if output_path is None:
        output_path = os.path.join(bak_dir, f"kiara-backup-{timestamp}.tar.gz")

    config_path = get_config_file_path()
    template_dir = get_template_dir(config)
    vectordb_data_path = _file_vectordb_backup_path(config, include_vectordb_data)

    # DSN & pg_dump-Verfuegbarkeit VOR dem temp dir pruefen, damit wir bei
    # Fehlern nicht erst halb-fertige Backups bauen.
    dsn_parts = _parse_dsn(config.get("database", {}).get("postgresql", {}).get("dsn", ""))
    _check_pg_tool("pg_dump")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "database_included": True,
        "database_name": dsn_parts["dbname"],
        "vectordb_backend": _vectordb_backend(config),
        "files": {},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # PostgreSQL-Dump (Pflicht)
        dump_path = os.path.join(tmpdir, "database.dump")
        _run_pg_dump(dsn_parts, dump_path)
        manifest["files"]["database.dump"] = _sha256(dump_path)

        # Copy config
        if os.path.exists(config_path):
            dest = os.path.join(tmpdir, "kiara.yaml")
            shutil.copy2(config_path, dest)
            manifest["files"]["kiara.yaml"] = _sha256(dest)

        # Copy templates
        if os.path.isdir(template_dir):
            dest = os.path.join(tmpdir, "templates")
            shutil.copytree(template_dir, dest, symlinks=True)
            for root, _dirs, files in os.walk(dest):
                for f in files:
                    fpath = os.path.join(root, f)
                    if os.path.islink(fpath):
                        continue
                    relpath = os.path.relpath(fpath, tmpdir)
                    manifest["files"][relpath] = _sha256(fpath)

        # Copy file-backed VectorDB data. Qdrant is deliberately excluded above:
        # copying the old ChromaDB path would create a false full-backup signal.
        if vectordb_data_path:
            dest = os.path.join(tmpdir, "vectordb_data")
            shutil.copytree(vectordb_data_path, dest, symlinks=True)
            manifest["include_vectordb_data"] = True
            for root, _dirs, files in os.walk(dest):
                for f in files:
                    fpath = os.path.join(root, f)
                    if os.path.islink(fpath):
                        continue
                    relpath = os.path.relpath(fpath, tmpdir)
                    manifest["files"][relpath] = _sha256(fpath)
        else:
            manifest["include_vectordb_data"] = False

        # Write manifest
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Create tar.gz (Symlinks ueberspringen — kein Grund fuer Symlinks in Backups)
        def _no_symlinks(tarinfo):
            return None if tarinfo.issym() or tarinfo.islnk() else tarinfo

        with tarfile.open(output_path, "w:gz") as tar:
            for item in os.listdir(tmpdir):
                tar.add(os.path.join(tmpdir, item), arcname=item, filter=_no_symlinks)

    logger.info("Backup erstellt: %s", output_path)

    # Cleanup old backups
    max_backups = config.get("backup", {}).get("max_backups", 5)
    _cleanup_old_backups(bak_dir, max_backups)

    return output_path


def _validate_backup_tar(tar: tarfile.TarFile) -> tuple[bool, dict, list[str]]:
    """Validiert ein bereits geoeffnetes Backup-Tar anhand der Checksummen."""
    issues = []

    # Security check
    for member in tar.getmembers():
        if member.name.startswith("/") or ".." in member.name:
            return False, {}, ["Sicherheitsproblem: Backup enthaelt absolute Pfade"]

    # Extract manifest
    try:
        manifest_file = tar.extractfile("manifest.json")
        if manifest_file is None:
            return False, {}, ["manifest.json nicht im Backup"]
        manifest = json.load(manifest_file)
    except (KeyError, json.JSONDecodeError) as exc:
        return False, {}, [f"manifest.json fehlerhaft: {exc}"]

    # Verify checksums
    for filename, expected_hash in manifest.get("files", {}).items():
        try:
            member_file = tar.extractfile(filename)
            if member_file is None:
                issues.append(f"Datei '{filename}' nicht im Backup")
                continue
            actual_hash = hashlib.sha256(member_file.read()).hexdigest()
            if not hmac.compare_digest(actual_hash, expected_hash):
                issues.append(f"Checksumme fuer '{filename}' stimmt nicht")
        except KeyError:
            issues.append(f"Datei '{filename}' nicht im Backup")

    # Reverse-Check: Dateien im Archiv die nicht im Manifest stehen
    manifest_files = set(manifest.get("files", {}).keys())
    manifest_files.add("manifest.json")
    for member in tar.getmembers():
        if member.isfile() and member.name not in manifest_files:
            issues.append(f"Datei '{member.name}' im Archiv aber nicht im Manifest")

    if "database.dump" not in manifest.get("files", {}):
        issues.append(
            "Backup ohne PostgreSQL-Dump ist ungueltig — Restore wuerde DB-Metadaten verlieren"
        )

    return len(issues) == 0, manifest, issues


def validate_backup(backup_path: str) -> tuple[bool, dict, list[str]]:
    """Validiert ein Backup anhand der Checksummen.

    Returns: (is_valid, manifest, issues)
    """
    if not os.path.exists(backup_path):
        return False, {}, [f"Backup-Datei nicht gefunden: {backup_path}"]

    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            return _validate_backup_tar(tar)
    except tarfile.TarError as exc:
        return False, {}, [f"Backup-Datei beschaedigt: {exc}"]


def restore_backup(
    backup_path: str, config: dict, force: bool = False, dry_run: bool = False
) -> dict:
    """Stellt ein Backup wieder her.

    Returns: dict mit restore-Details
    """
    config_path = get_config_file_path()
    template_dir = get_template_dir(config)
    vectordb_data_path = get_vectordb_data_path(config)

    restored = []

    if not os.path.exists(backup_path):
        return {
            "success": False,
            "message": "Backup-Validierung fehlgeschlagen",
            "issues": [f"Backup-Datei nicht gefunden: {backup_path}"],
        }

    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            is_valid, manifest, issues = _validate_backup_tar(tar)

            if not is_valid:
                return {
                    "success": False,
                    "message": "Backup-Validierung fehlgeschlagen",
                    "issues": issues,
                }

            if dry_run:
                # Just report what would be restored
                members = tar.getnames()
                result = {
                    "success": True,
                    "dry_run": True,
                    "would_restore": members,
                    "issues": issues,
                }
                return result

            with tempfile.TemporaryDirectory() as tmpdir:
                tar.extractall(tmpdir, filter="data")

                # Unmanifestierte Dateien entfernen (Defense-in-Depth)
                manifest_files = set(manifest.get("files", {}).keys())
                manifest_files.add("manifest.json")
                for root, _dirs, files in os.walk(tmpdir):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        relpath = os.path.relpath(fpath, tmpdir)
                        if relpath not in manifest_files:
                            os.remove(fpath)
                            logger.warning(
                                "Unmanifestierte Datei aus Backup entfernt: %s",
                                relpath,
                            )

                src_dump = os.path.join(tmpdir, "database.dump")
                src_config = os.path.join(tmpdir, "kiara.yaml")
                src_templates = os.path.join(tmpdir, "templates")
                src_vdb = os.path.join(tmpdir, "vectordb_data")

                restore_blockers: list[str] = []
                if os.path.isdir(src_vdb) and _vectordb_backend(config) == "qdrant":
                    restore_blockers.append(
                        "Backup enthaelt file-backed vectordb_data/, Zielbackend ist "
                        "Qdrant. Qdrant-Collections duerfen nicht aus dem alten "
                        "Dateipfad-Restore hergestellt werden."
                    )
                if (
                    os.path.isdir(src_vdb)
                    and _vectordb_backend(config) != "qdrant"
                    and not get_vectordb_data_path(config)
                ):
                    restore_blockers.append(
                        "Backup enthaelt vectordb_data/, aber vectordb.path ist leer"
                    )
                if restore_blockers:
                    return {
                        "success": False,
                        "dry_run": False,
                        "message": "Backup-Restore blockiert",
                        "restored": [],
                        "issues": issues + restore_blockers,
                    }

                conflicts: list[str] = []
                if os.path.exists(src_dump):
                    conflicts.append(
                        "database.dump wuerde die bestehende PostgreSQL-Datenbank ueberschreiben"
                    )
                if os.path.exists(src_config) and os.path.exists(config_path):
                    conflicts.append("kiara.yaml existiert bereits")
                if (
                    os.path.isdir(src_templates)
                    and os.path.isdir(template_dir)
                    and os.listdir(template_dir)
                ):
                    conflicts.append("templates/ existiert bereits")
                if os.path.isdir(src_vdb):
                    vectordb_data_path = get_vectordb_data_path(config)
                    if os.path.isdir(vectordb_data_path) and os.listdir(vectordb_data_path):
                        conflicts.append("vectordb_data/ existiert bereits")
                if conflicts and not force:
                    return {
                        "success": False,
                        "dry_run": False,
                        "message": "Restore wuerde bestehende Daten ueberschreiben",
                        "restored": [],
                        "issues": issues + [f"{c} — --force erforderlich" for c in conflicts],
                    }

                # Restore PostgreSQL-Dump (erst NACH Extraktion, aber vor den Datei-
                # Operationen, damit bei DB-Fehler keine halb-restaurierte Umgebung
                # mit neuer kiara.yaml aber alter DB zurueckbleibt).
                if os.path.exists(src_dump):
                    dsn_parts = _parse_dsn(
                        config.get("database", {}).get("postgresql", {}).get("dsn", "")
                    )
                    _run_pg_restore(dsn_parts, src_dump)
                    restored.append("database.dump")
                else:
                    return {
                        "success": False,
                        "dry_run": False,
                        "message": "Backup enthaelt keinen PostgreSQL-Dump",
                        "restored": restored,
                        "issues": [*issues, "database.dump fehlt im Archiv"],
                    }

                # Restore config
                if os.path.exists(src_config):
                    if os.path.exists(config_path) and not force:
                        issues.append(
                            "kiara.yaml existiert bereits — uebersprungen (--force zum Ueberschreiben)"
                        )
                    else:
                        os.makedirs(os.path.dirname(config_path), exist_ok=True)
                        shutil.copy2(src_config, config_path)
                        restored.append("kiara.yaml")

                # Restore templates
                if os.path.isdir(src_templates):
                    if os.path.isdir(template_dir) and os.listdir(template_dir) and not force:
                        issues.append(
                            "templates/ existiert bereits — uebersprungen (--force zum Ueberschreiben)"
                        )
                    else:
                        os.makedirs(template_dir, exist_ok=True)
                        shutil.copytree(
                            src_templates, template_dir, dirs_exist_ok=True, symlinks=True
                        )
                        restored.append("templates/")

                # Restore file-backed VectorDB data. Qdrant-Targets were blocked in
                # the preflight above; no legacy ChromaDB fallback is supported.
                if os.path.isdir(src_vdb):
                    vectordb_data_path = get_vectordb_data_path(config)
                    if (
                        os.path.isdir(vectordb_data_path)
                        and os.listdir(vectordb_data_path)
                        and not force
                    ):
                        issues.append(
                            "vectordb_data/ existiert bereits — uebersprungen (--force zum Ueberschreiben)"
                        )
                    else:
                        if os.path.isdir(vectordb_data_path) and force:
                            try:
                                shutil.rmtree(vectordb_data_path)
                            except OSError as exc:
                                logger.warning(
                                    "VectorDB-Verzeichnis konnte nicht geloescht werden: %s", exc
                                )
                        try:
                            shutil.copytree(
                                src_vdb, vectordb_data_path, symlinks=True, dirs_exist_ok=True
                            )
                            restored.append("vectordb_data/")
                        except OSError as exc:
                            raise BackupError(f"VectorDB-Restore fehlgeschlagen: {exc}") from exc
    except tarfile.TarError as exc:
        return {
            "success": False,
            "message": "Backup-Validierung fehlgeschlagen",
            "issues": [f"Backup-Datei beschaedigt: {exc}"],
        }

    return {"success": True, "dry_run": False, "restored": restored, "issues": issues}


def _sha256(file_path: str) -> str:
    """SHA256-Hash einer Datei."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cleanup_old_backups(backup_dir: str, max_backups: int):
    """Loescht die aeltesten Backups wenn mehr als max_backups vorhanden."""
    backups = sorted(
        [f for f in Path(backup_dir).glob("kiara-backup-*.tar.gz")],
        key=lambda f: f.stat().st_mtime,
    )
    while len(backups) > max_backups:
        oldest = backups.pop(0)
        try:
            oldest.unlink()
        except FileNotFoundError:
            logger.warning("Backup bereits geloescht: %s", oldest)
            continue
        logger.info("Altes Backup geloescht: %s", oldest)
