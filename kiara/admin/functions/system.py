import json

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult


@admin_function(
    name="system.ping",
    category="system",
    description="Prüft ob der Admin-Chat-Service antwortet",
    help="Einfacher Health-Check. Gibt 'pong' zurück.",
    examples=["Ping", "Bist du da?"],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="pong",
    raw_response=True,
    http=("GET", "/api/admin/chat/ping"),
)
async def system_ping(user: dict) -> AdminResult:
    return AdminResult.ok("pong")


# ---------------------------------------------------------------------------
# Health-Checks (Chat-only — HTTP bleibt unauthentifiziert in api_health.py)
# ---------------------------------------------------------------------------


@admin_function(
    name="system.health",
    category="system",
    description="Vollständiger Systemstatus (Degradation-Mode)",
    help="Zeigt den Gesundheitszustand aller Subsysteme: DB, Qdrant, Ollama, "
    "Embeddings. Gibt auch den Degradation-Modus an.",
    examples=[
        "Wie gesund ist das System?",
        "Systemstatus prüfen",
        "Health-Check",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Health-Status aller Subsysteme",
)
async def system_health(user: dict) -> AdminResult:
    from kiara.web.api_health import health_check

    response = await health_check()
    try:
        data = json.loads(response.body)
    except (ValueError, TypeError):
        return AdminResult.fail("Health-Check-Response ungueltig")
    return AdminResult.ok("Health-Status", data=data)


@admin_function(
    name="system.health_ready",
    category="system",
    description="Readiness-Probe (alle Subsysteme bereit?)",
    help="Prüft ob alle kritischen Subsysteme bereit sind. "
    "Wird von Orchestrierern wie Kubernetes genutzt.",
    examples=[
        "Ist das System bereit?",
        "Readiness prüfen",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Readiness-Status mit Details zu jedem Subsystem",
)
async def system_health_ready(user: dict) -> AdminResult:
    from kiara.web.api_health import readiness_check

    response = await readiness_check()
    try:
        data = json.loads(response.body)
    except (ValueError, TypeError):
        return AdminResult.fail("Readiness-Check-Response ungueltig")
    return AdminResult.ok("Readiness-Status", data=data)


@admin_function(
    name="system.health_live",
    category="system",
    description="Liveness-Probe (Service läuft?)",
    help="Einfacher Check ob der Service-Prozess noch antwortet.",
    examples=[
        "Lebt der Service?",
        "Liveness prüfen",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liveness-Status",
)
async def system_health_live(user: dict) -> AdminResult:
    from kiara.web.api_health import liveness_check

    response = await liveness_check()
    try:
        data = json.loads(response.body)
    except (ValueError, TypeError):
        return AdminResult.fail("Liveness-Check-Response ungueltig")
    return AdminResult.ok("Liveness-Status", data=data)


# ---------------------------------------------------------------------------
# Config-Changelog
# ---------------------------------------------------------------------------


@admin_function(
    name="system.changelog",
    category="system",
    description="Config-Changelog (paginiert)",
    help="Zeigt die Änderungshistorie aller Konfigurationsänderungen. "
    "Filterbar nach Bereich, paginiert und sortierbar.",
    examples=[
        "Zeig mir das Changelog",
        "Letzte Konfigurationsänderungen",
        "Changelog für datasources",
    ],
    parameters={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Seite (Default: 1)"},
            "page_size": {
                "type": "integer",
                "description": "Einträge pro Seite (Default: 50, max: 200)",
            },
            "area": {"type": "string", "description": "Filter nach Bereich"},
            "source": {"type": "string", "description": "Filter nach Quelle (gui, chat)"},
            "sort_field": {"type": "string", "description": "Sortierfeld (Default: created_at)"},
            "sort_dir": {
                "type": "string",
                "description": "Sortierrichtung: asc/desc (Default: desc)",
            },
        },
        "required": [],
    },
    returns="Paginierte Changelog-Liste mit Metadaten",
    raw_response=True,
    http=("GET", "/api/admin/system/changelog"),
)
async def system_changelog(
    user: dict,
    page: int = 1,
    page_size: int = 50,
    area: str = None,
    source: str = None,
    sort_field: str = "created_at",
    sort_dir: str = "desc",
) -> AdminResult:
    from kiara.web.config_changelog import query_changelog

    data = query_changelog(
        area=area,
        source=source,
        page=page,
        page_size=page_size,
        sort_field=sort_field,
        sort_dir=sort_dir,
    )
    return AdminResult.ok("Changelog", data=data)


@admin_function(
    name="system.changelog_areas",
    category="system",
    description="Verfügbare Changelog-Bereiche",
    help="Listet alle vorhandenen Bereiche im Config-Changelog auf "
    "(z.B. datasources, ki, users). Nützlich für Filter-Dropdowns.",
    examples=[
        "Welche Changelog-Bereiche gibt es?",
        "Changelog-Kategorien",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste der Bereiche",
    raw_response=True,
    http=("GET", "/api/admin/system/changelog/areas"),
)
async def system_changelog_areas(user: dict) -> AdminResult:
    from kiara.repositories.changelog import ConfigChangelogRepository

    changelog_repo = ConfigChangelogRepository()
    areas = changelog_repo.list_distinct_areas()
    return AdminResult.ok("Bereiche", data={"areas": areas})


# ──────────────────────────────────────────────
# Sprint 08: Settings + VectorDB-Test
# ──────────────────────────────────────────────


@admin_function(
    name="system.settings.read",
    category="system",
    description="Zentrale Konfiguration laden",
    help="Liest die aktuelle kiara.yaml-Konfiguration und gibt sie als "
    "strukturiertes Objekt zurück.",
    examples=[
        "Zeig mir die Konfiguration",
        "Welche Einstellungen sind aktiv?",
        "Was steht in der kiara.yaml?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Vollständige Konfiguration als dict",
    raw_response=True,
    http=("GET", "/api/admin/settings"),
)
async def system_settings_read(user: dict) -> AdminResult:
    from kiara.config import load_config, mask_config_secrets

    config = mask_config_secrets(load_config())
    return AdminResult.ok("Konfiguration geladen", data={"config": config})


@admin_function(
    name="system.settings.update",
    category="system",
    description="Zentrale Konfiguration speichern",
    help="Ändert die zentrale Konfigurationsdatei /etc/kiara/kiara.yaml. "
    "Manche Änderungen erfordern einen Service-Neustart.",
    examples=[
        "Ändere die Konfiguration",
        "Setze den Ollama-URL auf http://gpu-server.local:11434",
    ],
    parameters={
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "description": "Vollständiges Konfigurations-Objekt",
            },
        },
        "required": ["config"],
    },
    returns="Bestätigung der Speicherung",
    requires_confirmation=True,
    confirm_template="konfiguration speichern",
    changelog_area="settings",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/settings"),
)
async def system_settings_update(user: dict, config: dict) -> AdminResult:
    if not isinstance(config, dict):
        return AdminResult.fail("Ungültiges Konfigurations-Objekt")

    from kiara.config import load_config, restore_config_secrets, save_config

    config = restore_config_secrets(config, load_config())
    try:
        save_config(config)
    except ValueError as exc:
        return AdminResult.fail(f"Validierungsfehler: {exc}")
    except (PermissionError, OSError) as exc:
        return AdminResult.fail(f"Fehler beim Speichern: {exc}")
    return AdminResult.ok(
        "Konfiguration gespeichert",
        data={"message": "Konfiguration gespeichert"},
    )


@admin_function(
    name="system.vectordb.test",
    category="system",
    description="VektorDB-Verbindungstest",
    help="Testet die Verbindung zur Vektor-Datenbank (Qdrant). "
    "Prüft Erreichbarkeit und listet vorhandene Collections.",
    examples=[
        "Teste die VektorDB-Verbindung",
        "Ist Qdrant erreichbar?",
        "Prüfe die Qdrant-Verbindung auf localhost:6333",
    ],
    parameters={
        "type": "object",
        "properties": {
            "backend": {
                "type": "string",
                "description": "VektorDB-Backend (aktuell nur 'qdrant')",
                "default": "qdrant",
            },
            "path": {
                "type": "string",
                "description": "Pfad (für lokale Backends)",
                "default": "",
            },
            "qdrant_url": {
                "type": "string",
                "description": "Qdrant-URL (z.B. http://localhost:6333)",
                "default": "",
            },
            "qdrant_api_key": {
                "type": "string",
                "description": "Qdrant API-Key (optional)",
                "default": "",
            },
            "vector_size": {
                "type": "integer",
                "description": "Vektordimension",
            },
        },
        "required": [],
    },
    returns="Verbindungsstatus und Collection-Anzahl",
    raw_response=True,
    http=("POST", "/api/admin/vectordb/test-connection"),
)
async def system_vectordb_test(
    user: dict,
    backend: str = "qdrant",
    path: str = "",
    qdrant_url: str = "",
    qdrant_api_key: str = "",
    vector_size: int = 768,
) -> AdminResult:
    if backend != "qdrant":
        return AdminResult.fail(f"Unbekanntes Backend: {backend}")
    try:
        from qdrant_client import QdrantClient

        from kiara.config import DEFAULT_TIMEOUTS
        from kiara.shared.constants import DEFAULT_QDRANT_URL

        url = qdrant_url or DEFAULT_QDRANT_URL
        client = QdrantClient(
            url=url, api_key=qdrant_api_key or None, timeout=DEFAULT_TIMEOUTS["db_connect"]
        )
        colls = client.get_collections()
        return AdminResult.ok(
            f"Qdrant erreichbar, {len(colls.collections)} Collection(s)",
            data={
                "success": True,
                "message": f"Qdrant erreichbar, {len(colls.collections)} Collection(s)",
            },
        )
    except Exception as exc:
        return AdminResult.fail(f"Verbindung fehlgeschlagen: {exc}")
