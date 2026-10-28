"""Admin-Funktionen fuer Integritaetspruefung."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.retrieval.index_compatibility import IndexCompatibilityError

logger = logging.getLogger(__name__)


# --- 1. Vollpruefung starten (Background) ---


@admin_function(
    name="integrity.run",
    category="integrity",
    description="Integritaetspruefung starten",
    help="Startet eine vollstaendige Integritaetspruefung aller Datenschichten "
    "(Qdrant, Elasticsearch, chunk_datasources, file_hashes). "
    "Der Vorgang laeuft im Hintergrund — Status ueber integrity.status abfragen.",
    examples=[
        "Starte Integritaetspruefung",
        "Pruefe die Datenkonsistenz",
        "Sind die Datenschichten konsistent?",
        "System-Integritaet pruefen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass Pruefung gestartet wurde",
    raw_response=True,
    http=("POST", "/api/admin/integrity/run"),
)
async def integrity_run(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_check

    if is_running():
        return AdminResult.fail("Eine Integritaetspruefung laeuft bereits")
    try:
        start_check()
    except IndexCompatibilityError:
        raise
    except RuntimeError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Integritaetspruefung gestartet",
        data={"message": "Integritaetspruefung gestartet"},
    )


# --- 2. Status abfragen (Polling) ---


@admin_function(
    name="integrity.status",
    category="integrity",
    description="Integritaetspruefung-Status abfragen",
    help="Gibt den Status der laufenden oder letzten Integritaetspruefung zurueck. "
    "Enthaelt Fortschritt und Ergebnisse.",
    examples=[
        "Wie weit ist die Integritaetspruefung?",
        "Integritaets-Status?",
        "Ist die Pruefung fertig?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Task-Status (running, progress, result mit Einzel-Check-Ergebnissen)",
    raw_response=True,
    http=("GET", "/api/admin/integrity/status"),
)
async def integrity_status(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import get_task_status

    status = get_task_status()
    return AdminResult.ok("Integritaets-Status", data=status)


# --- 3. Schnell-Check (synchron, nur Layer-Counts) ---


@admin_function(
    name="integrity.quick",
    category="integrity",
    description="Schnelle Schicht-Zaehlung",
    help="Zaehlt Chunks in allen 4 Datenschichten (Qdrant, ES, chunk_datasources, "
    "datasource_embeddings) und vergleicht. Laeuft synchron (wenige Sekunden).",
    examples=[
        "Wie viele Chunks sind in Qdrant vs ES?",
        "Schnelle Integritaetspruefung",
        "Schicht-Counts vergleichen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Chunk-Counts pro Datenschicht mit Vergleich",
    raw_response=True,
    http=("GET", "/api/admin/integrity/quick"),
)
async def integrity_quick(user: dict) -> AdminResult:
    from kiara.config import load_config
    from kiara.integrity.checks import check_layer_counts

    try:
        config = load_config()
        result = check_layer_counts(config)
    except Exception as exc:
        return AdminResult.fail(f"Schnell-Check fehlgeschlagen: {exc}")
    return AdminResult.ok("Schicht-Zaehlung", data=result)


# --- 4. Repair: ds_ids in Qdrant synchronisieren ---


@admin_function(
    name="integrity.repair_ds_ids",
    category="integrity",
    description="DS-IDs in Qdrant synchronisieren",
    help="Synchronisiert das ds_ids-Payload-Feld aller Chunks in Qdrant mit den "
    "tatsaechlichen Zuordnungen aus chunk_datasources. Laeuft im Hintergrund.",
    examples=[
        "Synchronisiere ds_ids in Qdrant",
        "DS-IDs reparieren",
        "Qdrant ds_ids aktualisieren",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass Repair gestartet wurde",
    requires_confirmation=True,
    confirm_template="ds_ids in qdrant synchronisieren",
    changelog_area="integrity",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/integrity/repair/ds-ids"),
)
async def integrity_repair_ds_ids(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_repair

    if is_running():
        return AdminResult.fail("Ein Integritaets-Task laeuft bereits")
    try:
        start_repair("repair_ds_ids")
    except IndexCompatibilityError:
        raise
    except (RuntimeError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "ds_ids-Sync gestartet",
        data={"message": "ds_ids-Sync gestartet"},
    )


# --- 5. Repair: Verwaiste Chunks loeschen ---


@admin_function(
    name="integrity.repair_orphans",
    category="integrity",
    description="Verwaiste Chunks aus Qdrant loeschen",
    help="Scannt ALLE Chunks in Qdrant und loescht diejenigen, die keine "
    "Zuordnung in chunk_datasources haben. DESTRUKTIV — vollstaendiger Scan. "
    "Laeuft im Hintergrund.",
    examples=[
        "Loesche verwaiste Chunks",
        "Orphan-Cleanup ausfuehren",
        "Verwaiste Chunks bereinigen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass Orphan-Cleanup gestartet wurde",
    requires_confirmation=True,
    confirm_template="verwaiste chunks aus qdrant löschen",
    changelog_area="integrity",
    changelog_action="delete",
    raw_response=True,
    http=("POST", "/api/admin/integrity/repair/orphans"),
)
async def integrity_repair_orphans(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_repair

    if is_running():
        return AdminResult.fail("Ein Integritaets-Task laeuft bereits")
    try:
        start_repair("repair_orphans")
    except IndexCompatibilityError:
        raise
    except (RuntimeError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Orphan-Cleanup gestartet",
        data={"message": "Orphan-Cleanup gestartet"},
    )


# --- 6. Repair: Statistiken korrigieren ---


@admin_function(
    name="integrity.repair_stats",
    category="integrity",
    description="Chunk-Statistiken in datasource_embeddings korrigieren",
    help="Zaehlt die tatsaechlichen Chunks pro Datenquelle in chunk_datasources "
    "und aktualisiert chunk_count in datasource_embeddings. Laeuft im Hintergrund.",
    examples=[
        "Statistiken korrigieren",
        "Chunk-Counts aktualisieren",
        "datasource_embeddings reparieren",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass Stats-Korrektur gestartet wurde",
    requires_confirmation=True,
    confirm_template="chunk-statistiken korrigieren",
    changelog_area="integrity",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/integrity/repair/stats"),
)
async def integrity_repair_stats(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_repair

    if is_running():
        return AdminResult.fail("Ein Integritaets-Task laeuft bereits")
    try:
        start_repair("repair_stats")
    except IndexCompatibilityError:
        raise
    except (RuntimeError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Stats-Korrektur gestartet",
        data={"message": "Stats-Korrektur gestartet"},
    )


# --- 7. Repair: Metadaten-Reset (nur wenn ALLE Collections leer) ---


@admin_function(
    name="integrity.repair_reset_meta",
    category="integrity",
    description="Verwaiste Metadaten loeschen (alle Qdrant-Collections leer)",
    help="Loescht alle Eintraege in chunk_datasources, file_hashes und "
    "datasource_embeddings. NUR ausfuehren wenn ALLE Qdrant-Collections "
    "leer sind — das System prueft dies automatisch und bricht ab falls "
    "noch Chunks in irgendeiner Collection vorhanden sind. "
    "WICHTIG: Wenn Qdrant noch Daten enthaelt, stattdessen "
    "integrity.repair_reconstruct verwenden — das rekonstruiert die "
    "Metadaten aus den vorhandenen Qdrant-Daten ohne Datenverlust.",
    examples=[
        "Verwaiste Metadaten loeschen",
        "Qdrant ist komplett leer, Metadaten bereinigen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass Metadaten-Reset gestartet wurde",
    requires_confirmation=True,
    confirm_template="alle metadaten löschen (chunk_datasources, file_hashes, datasource_embeddings)",
    changelog_area="integrity",
    changelog_action="delete",
    raw_response=True,
    http=("POST", "/api/admin/integrity/repair/reset-meta"),
)
async def integrity_repair_reset_meta(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_repair

    if is_running():
        return AdminResult.fail("Ein Integritaets-Task laeuft bereits")
    try:
        start_repair("repair_reset_meta")
    except IndexCompatibilityError:
        raise
    except (RuntimeError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Metadaten-Reset gestartet",
        data={"message": "Metadaten-Reset gestartet"},
    )


# --- 8. Repair: Metadaten aus Qdrant rekonstruieren ---


@admin_function(
    name="integrity.repair_reconstruct",
    category="integrity",
    description="Metadaten aus Qdrant-Daten rekonstruieren",
    help="Rekonstruiert chunk_datasources und datasource_embeddings aus den "
    "vorhandenen Qdrant-Chunks. Fuer jeden Chunk wird der source-Pfad "
    "gegen die Datasource-Pfade gematcht. Anschliessend werden die "
    "ds_ids-Payloads in Qdrant aktualisiert. "
    "Durchsucht ALLE Qdrant-Collections (Multi-Modell). "
    "Kein Datenverlust — file_hashes werden bei der naechsten "
    "Indexierung automatisch wieder aufgebaut.",
    examples=[
        "Metadaten aus Qdrant rekonstruieren",
        "chunk_datasources aus Qdrant wiederherstellen",
        "DS-Zuordnungen aus Vektordaten rekonstruieren",
        "Metadaten wiederherstellen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass Rekonstruktion gestartet wurde",
    requires_confirmation=True,
    confirm_template="metadaten aus qdrant-daten rekonstruieren",
    changelog_area="integrity",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/integrity/repair/reconstruct"),
)
async def integrity_repair_reconstruct(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_repair

    if is_running():
        return AdminResult.fail("Ein Integritaets-Task laeuft bereits")
    try:
        start_repair("repair_reconstruct")
    except IndexCompatibilityError:
        raise
    except (RuntimeError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Metadaten-Rekonstruktion gestartet",
        data={"message": "Metadaten-Rekonstruktion gestartet"},
    )


# --- 9. Repair: Collection-Rebuild (ersetzt Defragmentierung) ---


@admin_function(
    name="integrity.repair_rebuild",
    category="integrity",
    description="Fragmentierte Qdrant-Collections neu aufbauen",
    help="Erstellt einen Snapshot, loescht die Collection und stellt sie "
    "aus dem Snapshot wieder her. Entfernt leere Segmente und gibt "
    "Speicherplatz frei. Nur fuer Collections mit >10.000 Chunks und "
    ">5 Segmenten (konsistent mit check_qdrant_fragmentation).",
    examples=[
        "Qdrant-Collection rebuilden",
        "Collection neu aufbauen",
        "Leere Segmente entfernen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass Rebuild gestartet wurde",
    requires_confirmation=True,
    confirm_template="fragmentierte qdrant-collections neu aufbauen (snapshot/restore)",
    changelog_area="integrity",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/integrity/repair/rebuild"),
)
async def integrity_repair_rebuild(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_repair

    if is_running():
        return AdminResult.fail("Ein Integritaets-Task laeuft bereits")
    try:
        start_repair("repair_rebuild")
    except IndexCompatibilityError:
        raise
    except (RuntimeError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Collection-Rebuild gestartet",
        data={"message": "Collection-Rebuild gestartet"},
    )


# --- 10. Repair: Stale ColBERT-Collections bereinigen ---


@admin_function(
    name="integrity.cleanup_colbert",
    category="integrity",
    description="Stale ColBERT-Collections bereinigen",
    help="Startet den manuellen Cleanup veralteter ColBERT-Collections und "
    "verwaister colbert::*-DB-Eintraege im Hintergrund.",
    examples=[
        "Stale ColBERT-Collections bereinigen",
        "ColBERT-Cleanup starten",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestaetigung dass ColBERT-Cleanup gestartet wurde",
    requires_confirmation=True,
    confirm_template="stale colbert-collections bereinigen",
    changelog_area="integrity",
    changelog_action="delete",
    raw_response=True,
    http=("POST", "/api/admin/integrity/cleanup/colbert"),
)
async def integrity_cleanup_colbert(user: dict) -> AdminResult:
    from kiara.admin.services.integrity_service import is_running, start_repair

    if is_running():
        return AdminResult.fail("Ein Integritaets-Task laeuft bereits")
    try:
        start_repair("repair_cleanup_colbert")
    except IndexCompatibilityError:
        raise
    except (RuntimeError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "ColBERT-Cleanup gestartet",
        data={"message": "ColBERT-Cleanup gestartet"},
    )
