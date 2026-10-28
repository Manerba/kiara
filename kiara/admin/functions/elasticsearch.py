"""Admin-Funktionen fuer Elasticsearch-Management."""

import asyncio
import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="elasticsearch.status",
    category="elasticsearch",
    description="Elasticsearch-Status anzeigen",
    help="Zeigt den Verbindungsstatus, Index-Informationen und Chunk-Counts "
    "(physische ES-Kopien pro Quelle und eindeutige Qdrant-Chunks getrennt) an.",
    examples=[
        "Wie ist der Elasticsearch-Status?",
        "Ist der ES-Index aktuell?",
        "Zeig mir die ES-Statistiken",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="ES-Status mit Verbindung, Index-Info und Chunk-Counts",
    raw_response=True,
    http=("GET", "/api/admin/elasticsearch/status"),
)
async def elasticsearch_status(user: dict) -> AdminResult:
    from kiara.admin.services.elasticsearch_service import get_es_status

    try:
        status = await asyncio.to_thread(get_es_status)
    except Exception as exc:
        return AdminResult.fail(f"ES-Status-Abfrage fehlgeschlagen: {exc}")
    return AdminResult.ok("Elasticsearch-Status", data=status)


@admin_function(
    name="elasticsearch.rebuild",
    category="elasticsearch",
    description="Elasticsearch-Index neu aufbauen",
    help="Baut den Elasticsearch-Index komplett neu auf. Alle Chunks werden "
    "aus der VektorDB gelesen und in ES re-indexiert. Erfordert Offline-Wartung "
    "nach docs/operations/elasticsearch-s4.md; kein Online-Start.",
    examples=[
        "Baue den ES-Index neu auf",
        "Elasticsearch-Index rebuilden",
        "ES-Index neu erstellen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung dass Rebuild gestartet wurde",
    requires_confirmation=True,
    confirm_template="elasticsearch-index neu aufbauen",
    changelog_area="elasticsearch",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/elasticsearch/rebuild"),
)
async def elasticsearch_rebuild(user: dict) -> AdminResult:
    from kiara.admin.services.elasticsearch_service import is_running, start_rebuild

    if is_running():
        return AdminResult.fail("Ein ES-Task läuft bereits")
    try:
        start_rebuild()
    except RuntimeError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Index-Rebuild gestartet",
        data={"message": "Index-Rebuild gestartet"},
    )


@admin_function(
    name="elasticsearch.delete",
    category="elasticsearch",
    description="Elasticsearch-Index löschen",
    help="Löscht den Elasticsearch-Index komplett. Danach ist die "
    "Volltextsuche deaktiviert bis ein Rebuild durchgeführt wird.",
    examples=[
        "Lösche den ES-Index",
        "Elasticsearch-Index entfernen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung der Löschung",
    requires_confirmation=True,
    confirm_template="elasticsearch-index löschen",
    changelog_area="elasticsearch",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/elasticsearch/index"),
)
async def elasticsearch_delete(user: dict) -> AdminResult:
    from kiara.admin.services.elasticsearch_service import delete_index

    try:
        result = await asyncio.to_thread(delete_index)
    except Exception as exc:
        return AdminResult.fail(f"Index-Löschung fehlgeschlagen: {exc}")
    return AdminResult.ok("Index gelöscht", data=result)


@admin_function(
    name="elasticsearch.verify",
    category="elasticsearch",
    description="Elasticsearch-Index-Integrität prüfen",
    help="Prüft die Konsistenz zwischen Elasticsearch-Index und VektorDB. "
    "Der Vorgang läuft im Hintergrund — Status über elasticsearch.task abfragen.",
    examples=[
        "Prüfe die ES-Index-Integrität",
        "Ist der ES-Index konsistent?",
        "Elasticsearch verifizieren",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung dass Verify gestartet wurde",
    raw_response=True,
    http=("POST", "/api/admin/elasticsearch/verify"),
)
async def elasticsearch_verify(user: dict) -> AdminResult:
    from kiara.admin.services.elasticsearch_service import is_running, start_verify

    if is_running():
        return AdminResult.fail("Ein ES-Task läuft bereits")
    try:
        start_verify()
    except RuntimeError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        "Integritätsprüfung gestartet",
        data={"message": "Integritätsprüfung gestartet"},
    )


@admin_function(
    name="elasticsearch.task",
    category="elasticsearch",
    description="ES-Task-Status abfragen",
    help="Gibt den Status des laufenden ES-Tasks (Rebuild oder Verify) zurück. "
    "Enthält Action, Fortschritt und Ergebnis.",
    examples=[
        "Wie weit ist der ES-Rebuild?",
        "ES-Task-Status?",
        "Ist der ES-Rebuild fertig?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Task-Status (running, action, progress, result)",
    raw_response=True,
    http=("GET", "/api/admin/elasticsearch/task"),
)
async def elasticsearch_task(user: dict) -> AdminResult:
    from kiara.admin.services.elasticsearch_service import get_task_status

    status = get_task_status()
    return AdminResult.ok("ES-Task-Status", data=status)
