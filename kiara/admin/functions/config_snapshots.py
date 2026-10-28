"""Admin-Funktionen fuer Config-Snapshots (versionierte kiara.yaml-Staende)."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


def _username(user: dict) -> str:
    return str(user.get("username") or user.get("id") or "")


@admin_function(
    name="config.snapshot.list",
    category="backup",
    description="Config-Snapshots auflisten",
    help="Listet versionierte Konfigurationsstaende (kiara.yaml) mit Label, "
    "Hash und den darueber verknuepften Benchmark-Laeufen (gleicher "
    "config_file_sha256 in den Summary-Artefakten).",
    examples=[
        "Welche Config-Snapshots gibt es?",
        "Zeig mir die Konfigurationsstaende",
        "Mit welcher Config lief der letzte Golden-Benchmark?",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste der Snapshots mit Label, Hash, Trigger und Benchmark-Laeufen",
    raw_response=True,
    http=("GET", "/api/admin/config-snapshots/list"),
)
async def config_snapshot_list(user: dict) -> AdminResult:
    import asyncio

    from kiara.admin.services.config_snapshot_service import list_snapshots

    snapshots = await asyncio.to_thread(list_snapshots)
    return AdminResult.ok(
        f"{len(snapshots)} Config-Snapshot(s)",
        data={"snapshots": snapshots},
    )


@admin_function(
    name="config.snapshot.create",
    category="backup",
    description="Config-Snapshot erstellen",
    help="Sichert den aktuellen Stand von kiara.yaml als benannten Snapshot.",
    examples=[
        "Erstelle einen Config-Snapshot mit dem Label reranker-live",
        "Sichere die aktuelle Konfiguration",
    ],
    parameters={
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "Kurzname (Buchstaben, Ziffern, . _ -; optional)",
            },
            "comment": {"type": "string", "description": "Freitext-Kommentar (optional)"},
        },
        "required": [],
    },
    returns="Angelegter Snapshot-Eintrag",
    changelog_area="config_snapshot",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/config-snapshots/create"),
)
async def config_snapshot_create(user: dict, label: str = "", comment: str = "") -> AdminResult:
    import asyncio

    from kiara.admin.services.config_snapshot_service import (
        ConfigSnapshotError,
        create_snapshot,
    )

    try:
        entry = await asyncio.to_thread(
            create_snapshot, label=label, comment=comment, trigger="manual", user=_username(user)
        )
    except ConfigSnapshotError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(f"Snapshot angelegt: {entry['file']}", data={"snapshot": entry})


@admin_function(
    name="config.snapshot.label",
    category="backup",
    description="Config-Snapshot benennen",
    help="Setzt Label und optional Kommentar eines bestehenden Snapshots.",
    examples=["Benenne den Snapshot 5556a40c als reranker-abstain-live"],
    parameters={
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Snapshot-Referenz: Label, Dateiname oder sha256-Praefix (>=8 Zeichen)",
            },
            "label": {"type": "string", "description": "Neues Label"},
            "comment": {"type": "string", "description": "Neuer Kommentar (optional)"},
        },
        "required": ["ref", "label"],
    },
    returns="Aktualisierter Snapshot-Eintrag",
    changelog_area="config_snapshot",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/config-snapshots/label"),
)
async def config_snapshot_label(
    user: dict, ref: str, label: str, comment: str | None = None
) -> AdminResult:
    import asyncio

    from kiara.admin.services.config_snapshot_service import ConfigSnapshotError, set_label

    try:
        entry = await asyncio.to_thread(set_label, ref, label, comment)
    except ConfigSnapshotError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(f"Snapshot beschriftet: {label}", data={"snapshot": entry})


@admin_function(
    name="config.snapshot.diff",
    category="backup",
    description="Config-Snapshot vergleichen",
    help="Zeigt die Unterschiede eines Snapshots zur aktuellen Config (oder zu "
    "einem zweiten Snapshot). Werte sensibler Felder werden maskiert.",
    examples=[
        "Was unterscheidet den Snapshot reranker-abstain-live von der aktuellen Config?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Snapshot-Referenz"},
            "against": {
                "type": "string",
                "description": "Vergleichsziel: 'current' (Default) oder zweite Snapshot-Referenz",
                "default": "current",
            },
        },
        "required": ["ref"],
    },
    returns="Liste geaenderter Schluessel mit maskierten Werten",
    raw_response=True,
    http=("POST", "/api/admin/config-snapshots/diff"),
)
async def config_snapshot_diff(user: dict, ref: str, against: str = "current") -> AdminResult:
    import asyncio

    from kiara.admin.services.config_snapshot_service import ConfigSnapshotError, diff_snapshots

    try:
        changes = await asyncio.to_thread(diff_snapshots, ref, against)
    except ConfigSnapshotError as exc:
        return AdminResult.fail(str(exc))
    count = max(0, len(changes) - 1)
    return AdminResult.ok(f"{count} Unterschied(e)", data={"changes": changes})


@admin_function(
    name="config.snapshot.restore",
    category="backup",
    description="Config-Snapshot wiederherstellen",
    help="ACHTUNG: Ersetzt die aktive kiara.yaml durch den Snapshot-Stand. Der "
    "aktuelle Stand wird vorher automatisch gesichert (pre_restore). Optional "
    "werden die KIara-Services danach neu gestartet.",
    examples=[
        "Stelle den Config-Snapshot reranker-abstain-live wieder her",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Snapshot-Referenz"},
            "restart": {
                "type": "boolean",
                "description": "Services nach dem Restore neu starten (Default: true)",
                "default": True,
            },
        },
        "required": ["ref"],
    },
    returns="Restore-Ergebnis inkl. pre_restore-Snapshot",
    requires_confirmation=True,
    confirm_template="Config-Snapshot {ref} wiederherstellen",
    pk_param="ref",
    changelog_area="config_snapshot",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/config-snapshots/restore"),
)
async def config_snapshot_restore(user: dict, ref: str, restart: bool = True) -> AdminResult:
    import asyncio

    from kiara.admin.services.config_snapshot_service import (
        ConfigSnapshotError,
        restore_snapshot,
        trigger_service_restart,
    )

    username = _username(user)
    try:
        result = await asyncio.to_thread(restore_snapshot, ref, username)
    except ConfigSnapshotError as exc:
        return AdminResult.fail(str(exc))
    if restart:
        try:
            result["restart"] = trigger_service_restart(username)
        except ConfigSnapshotError as exc:
            result["restart_error"] = str(exc)
    return AdminResult.ok(
        f"Config-Snapshot wiederhergestellt: {result['restored']['file']}"
        + (" — Services starten neu" if restart and "restart" in result else ""),
        data=result,
    )
