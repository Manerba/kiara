"""Admin-Funktionen fuer Backup/Restore."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="backup.list",
    category="backup",
    description="Verfügbare Backups auflisten",
    help="Listet alle vorhandenen Backup-Dateien mit Größe und Erstellungsdatum auf.",
    examples=[
        "Welche Backups gibt es?",
        "Zeig mir die Backup-Liste",
        "Wann wurde das letzte Backup erstellt?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der Backups mit Dateiname, Größe und Datum",
    raw_response=True,
    http=("GET", "/api/admin/backup/list"),
)
async def backup_list(user: dict) -> AdminResult:
    from kiara.admin.services.backup_service import list_backups

    backups = list_backups()
    return AdminResult.ok(
        f"{len(backups)} Backup(s) gefunden",
        data={"backups": backups},
    )


@admin_function(
    name="backup.create",
    category="backup",
    description="Neues Backup erstellen",
    help="Erstellt ein neues Backup der Datenbank. File-backed VektorDB-Daten "
    "koennen optional eingeschlossen werden; Qdrant-Collections werden nicht "
    "per altem Dateipfad gesichert. Der Vorgang läuft im Hintergrund — Status mit "
    "backup.status abfragen.",
    examples=[
        "Erstelle ein Backup",
        "Backup ohne VektorDB-Daten erstellen",
        "Sichere die Datenbank",
    ],
    parameters={
        "type": "object",
        "properties": {
            "include_vectordb_data": {
                "type": "boolean",
                "description": "File-backed VektorDB-Daten einschliessen (Standard: false; Qdrant nicht per Dateipfad)",
                "default": False,
            },
        },
        "required": [],
    },
    returns="Bestätigung dass Backup gestartet wurde",
    raw_response=True,
    http=("POST", "/api/admin/backup/create"),
)
async def backup_create(user: dict, include_vectordb_data: bool = False) -> AdminResult:
    from kiara.admin.services.backup_service import is_running, start_backup

    if is_running():
        return AdminResult.fail("Ein Backup läuft bereits")
    try:
        start_backup(include_vectordb_data=include_vectordb_data)
    except RuntimeError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Backup gestartet",
        data={"message": "Backup gestartet"},
    )


@admin_function(
    name="backup.status",
    category="backup",
    description="Backup-Status abfragen",
    help="Gibt den aktuellen Status eines laufenden oder abgeschlossenen Backup-Vorgangs zurück.",
    examples=[
        "Wie ist der Backup-Status?",
        "Ist das Backup fertig?",
        "Backup-Fortschritt?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Backup-Status (running, result, error)",
    raw_response=True,
    http=("GET", "/api/admin/backup/status"),
)
async def backup_status(user: dict) -> AdminResult:
    from kiara.admin.services.backup_service import get_status

    status = get_status()
    return AdminResult.ok("Backup-Status", data=status)


@admin_function(
    name="backup.restore",
    category="backup",
    description="Backup wiederherstellen",
    help="ACHTUNG: Stellt die Datenbank aus einem Backup wieder her. "
    "Alle seit dem Backup gemachten Änderungen gehen verloren!",
    examples=[
        "Stelle das Backup von gestern wieder her",
        "Restore kiara_backup_2026-03-27.tar.gz",
    ],
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Name der Backup-Datei (z.B. kiara_backup_2026-03-27_14-30.tar.gz)",
            },
            "force": {
                "type": "boolean",
                "description": "Bestehende Daten nach erfolgreicher Integritätsprüfung überschreiben",
                "default": False,
            },
        },
        "required": ["filename"],
    },
    returns="Bestätigung der Wiederherstellung",
    requires_confirmation=True,
    confirm_template="backup {filename} wiederherstellen",
    pk_param="filename",
    changelog_area="backup",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/backup/restore"),
)
async def backup_restore(user: dict, filename: str, force: bool = False) -> AdminResult:
    import asyncio

    from kiara.admin.services.backup_service import restore_backup

    try:
        result = await asyncio.to_thread(restore_backup, filename=filename, force=force)
    except ValueError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        f"Backup {filename} wiederhergestellt",
        data=result,
    )
