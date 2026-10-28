import json
import os

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.repositories.cache import CacheRepository
from kiara.repositories.datasource import DatasourceRepository
from kiara.repositories.permission import PermissionRepository
from kiara.shared.constants import PASSWORD_MASK
from kiara.web.api_utils import count_files_limited

ds_repo = DatasourceRepository()
perm_repo = PermissionRepository()
cache_repo = CacheRepository()

# Schema-Inkonsistenz: einige ds_*-Spalten sind boolean (z.B. ds_ddu_enabled),
# andere historisch integer (0/1). Das JS-Frontend sendet fuer beide bool.
# Hier explizit auf int casten, damit PostgreSQL den UPDATE akzeptiert.
# TODO(schema): Migration auf einheitlich boolean (Greenfield).
_INT_DS_FIELDS = frozenset(
    {
        "ds_ddu_ocr_images",
        "ds_accept_all_text",
        "ds_index_attachments",
        "ds_contextual_retrieval",
    }
)


@admin_function(
    name="datasource.list",
    category="datasources",
    description="Alle Datenquellen auflisten",
    help="Listet alle konfigurierten Datenquellen mit Status, Typ und "
    "Ingestion-Informationen auf. Liest aus der Cache-Tabelle.",
    examples=[
        "Zeig mir alle Datenquellen",
        "Welche Datenquellen gibt es?",
        "Liste der Datenquellen",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste aller Datenquellen mit Metadaten",
    raw_response=True,
    http=("GET", "/api/admin/datasources"),
)
async def datasource_list(user: dict) -> AdminResult:
    cached = cache_repo.get_cached_datasources()
    if not cached:
        from kiara.web.cache_refresh import refresh_datasource_cache

        refresh_datasource_cache()
        cached = cache_repo.get_cached_datasources()
    return AdminResult.ok("Datenquellen", data={"data": cached})


@admin_function(
    name="datasource.status",
    category="datasources",
    description="Status einer einzelnen Datenquelle",
    help="Zeigt Pfad, Erreichbarkeit und Dateianzahl einer Datenquelle.",
    examples=[
        "Status von Datenquelle 1",
        "Ist DS 5 erreichbar?",
    ],
    parameters={
        "type": "object",
        "properties": {"ds_id": {"type": "integer", "description": "ID der Datenquelle"}},
        "required": ["ds_id"],
    },
    returns="Pfad, Erreichbarkeit und Dateianzahl",
    raw_response=True,
    http=("GET", "/api/admin/datasources/{ds_id}/status"),
)
async def datasource_status(user: dict, ds_id: int) -> AdminResult:
    from kiara.web.api_datasource import _effective_path

    row = ds_repo.get_by_id(ds_id)
    if not row:
        return AdminResult.fail("Datenquelle nicht gefunden")

    path = _effective_path(row)
    if not path:
        return AdminResult.ok("Status", data={"path": "", "accessible": False, "file_count": 0})

    accessible = os.path.isdir(path)
    file_count = count_files_limited(path) if accessible else 0
    return AdminResult.ok(
        "Status", data={"path": path, "accessible": accessible, "file_count": file_count}
    )


@admin_function(
    name="datasource.groups",
    category="datasources",
    description="Zugeordnete Gruppen einer Datenquelle",
    help="Zeigt welche Benutzergruppen Zugriff auf eine Datenquelle haben.",
    examples=[
        "Welche Gruppen hat DS 3?",
        "Gruppen von Datenquelle 1",
    ],
    parameters={
        "type": "object",
        "properties": {"ds_id": {"type": "integer", "description": "ID der Datenquelle"}},
        "required": ["ds_id"],
    },
    returns="Liste der zugeordneten Gruppen",
    raw_response=True,
    http=("GET", "/api/admin/datasources/{ds_id}/groups"),
)
async def datasource_groups(user: dict, ds_id: int) -> AdminResult:
    if not ds_repo.exists(ds_id):
        return AdminResult.fail("Datenquelle nicht gefunden")
    groups = perm_repo.get_datasource_groups(ds_id)
    return AdminResult.ok("Gruppen", data={"groups": groups})


@admin_function(
    name="datasource.users",
    category="datasources",
    description="Zugeordnete Benutzer einer Datenquelle",
    help="Zeigt welche Benutzer direkt einer Datenquelle zugeordnet sind.",
    examples=[
        "Welche Benutzer hat DS 2?",
        "User von Datenquelle 5",
    ],
    parameters={
        "type": "object",
        "properties": {"ds_id": {"type": "integer", "description": "ID der Datenquelle"}},
        "required": ["ds_id"],
    },
    returns="Liste der zugeordneten Benutzer",
    raw_response=True,
    http=("GET", "/api/admin/datasources/{ds_id}/users"),
)
async def datasource_users(user: dict, ds_id: int) -> AdminResult:
    if not ds_repo.exists(ds_id):
        return AdminResult.fail("Datenquelle nicht gefunden")
    users = perm_repo.get_datasource_users(ds_id)
    return AdminResult.ok("Benutzer", data={"users": users})


@admin_function(
    name="datasource.connector_types",
    category="datasources",
    description="Verfügbare Konnektor-Typen mit Schema",
    help="Listet alle registrierten Konnektor-Typen (local, smb, webdav, imap, "
    "exchange_online, sql, ...) mit ihrem Konfigurationsschema auf.",
    examples=[
        "Welche Konnektor-Typen gibt es?",
        "Verfügbare Datenquellen-Typen",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste der Konnektor-Typen mit Label und Schema",
    raw_response=True,
    http=("GET", "/api/admin/connector-types"),
)
async def datasource_connector_types(user: dict) -> AdminResult:
    from kiara.connectors.registry import all_connectors, ensure_loaded

    ensure_loaded()
    types = []
    for _name, cls in sorted(all_connectors().items(), key=lambda x: x[1].label):
        types.append(
            {
                "name": cls.name,
                "label": cls.label,
                "badge_class": cls.badge_class,
                "schema": cls.config_schema(),
                "default_config": cls.default_config(),
                "supports_browse": getattr(cls, "supports_browse", False) is True,
            }
        )
    return AdminResult.ok("Konnektor-Typen", data={"data": types})


@admin_function(
    name="datasource.ingestion_status",
    category="datasources",
    description="Ingestion-Status einer Datenquelle",
    help="Zeigt den aktuellen Ingestion-Status (idle, running, paused, error), "
    "Fortschritt und letzte Fehler einer Datenquelle.",
    examples=[
        "Ingestion-Status von DS 1",
        "Läuft bei Datenquelle 3 eine Indexierung?",
    ],
    parameters={
        "type": "object",
        "properties": {"ds_id": {"type": "integer", "description": "ID der Datenquelle"}},
        "required": ["ds_id"],
    },
    returns="Ingestion-Status mit Fortschritt, Fehler und Zeitstempeln",
    raw_response=True,
    http=("GET", "/api/admin/datasources/{ds_id}/ingestion-status"),
)
async def datasource_ingestion_status(user: dict, ds_id: int) -> AdminResult:
    from kiara.shared.db import db_query

    with db_query() as db:
        row = db.execute(
            "SELECT payload_json FROM cache_datasources WHERE datasource_id = %s",
            (ds_id,),
        ).fetchone()
    if not row:
        from kiara.web.cache_refresh import refresh_datasource_cache

        refresh_datasource_cache(ds_id)
        with db_query() as db:
            row = db.execute(
                "SELECT payload_json FROM cache_datasources WHERE datasource_id = %s",
                (ds_id,),
            ).fetchone()
    if not row:
        return AdminResult.fail("Datenquelle nicht gefunden")

    # JSONB: psycopg2 liefert dict direkt; Fallback fuer Alt-Pfade
    payload = row["payload_json"]
    if isinstance(payload, dict):
        data = payload
    else:
        try:
            data = json.loads(payload) if payload else {}
        except (ValueError, TypeError):
            return AdminResult.fail("Cache-Daten beschaedigt (ungültiges JSON)")
    embeddings = data.get("embeddings", [])
    result = {
        "ingestion_status": data.get("ingestion_status", "idle"),
        "last_ingested_at": max(
            (e.get("last_ingested_at") for e in embeddings if e.get("last_ingested_at")),
            default=None,
        ),
        "last_error": data.get("last_error", ""),
        "chunk_count": sum(e.get("chunk_count", 0) for e in embeddings),
        "file_count": sum(e.get("file_count", 0) for e in embeddings),
        "processed_files": data.get("processed_files", 0),
        "progress_total": data.get("progress_total", 0),
        "progress_processed": data.get("progress_processed", 0),
        "started_at": data.get("started_at"),
        "phase": data.get("phase"),
        "job_id": data.get("job_id"),
    }
    return AdminResult.ok("Ingestion-Status", data=result)


# ---------------------------------------------------------------------------
# Write-Funktionen (Sprint 05)
# ---------------------------------------------------------------------------


@admin_function(
    name="datasource.create",
    category="datasources",
    description="Neue Datenquelle erstellen",
    help="Erstellt eine neue Datenquelle mit Konnektor-Konfiguration und Indexierungs-Einstellungen. "
    "Konnektor-Typen und ihre Pflichtfelder sind via datasource.connector_types abrufbar. "
    "Passwörter werden automatisch verschlüsselt gespeichert.",
    examples=[
        "Erstelle eine neue WebDAV-Datenquelle",
        "Neue lokale Datenquelle für /data/dokumente",
        "SMB-Share als Datenquelle hinzufügen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Anzeigename der Datenquelle"},
            "type": {
                "type": "string",
                "description": "Konnektor-Typ",
                "enum": ["local", "smb", "webdav", "imap", "exchange_online", "sql"],
            },
            "connector_config": {
                "type": "object",
                "description": "Konnektor-spezifische Konfiguration "
                "(Schema via datasource.connector_types)",
            },
            "ds_chunk_strategy": {
                "type": "string",
                "description": "Chunking-Strategie (Default: auto)",
            },
            "ds_chunk_size": {"type": "integer", "description": "Chunk-Größe in Zeichen"},
            "ds_chunk_overlap": {"type": "integer", "description": "Chunk-Überlappung in Zeichen"},
            "ds_file_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Erlaubte Dateitypen (leer = alle)",
            },
            "ds_exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ausschluss-Muster (Glob-Syntax)",
            },
            "ds_ddu_enabled": {
                "type": "boolean",
                "description": "Document Understanding aktivieren",
            },
            "ds_ddu_ocr_images": {
                "type": "boolean",
                "description": "OCR für Bilder (Default: true)",
            },
            "ds_vlm_enabled": {
                "type": "boolean",
                "description": "Vision Language Model aktivieren",
            },
            "ds_accept_all_text": {
                "type": "boolean",
                "description": "Alle Textdateien akzeptieren",
            },
            "ds_index_attachments": {
                "type": "boolean",
                "description": "E-Mail-Anhänge indexieren (Default: true)",
            },
            "sync_interval": {"type": "string", "description": "Sync-Intervall (Default: global)"},
            "restriction_level": {
                "type": "string",
                "description": "Restriction-Level (public, anonymize, blocked)",
                "enum": ["public", "anonymize", "blocked"],
                "default": "anonymize",
            },
            "ds_contextual_retrieval": {
                "type": "boolean",
                "description": "Contextual Retrieval (Enrichment) aktivieren",
            },
            "ds_enrichment_prompt_id": {
                "type": "integer",
                "description": "Enrichment-Prompt-Template-ID (NULL = Standard)",
            },
            "ds_enrichment_backend_id": {
                "type": "integer",
                "description": "Enrichment-Backend-ID (Pflicht wenn CR aktiv)",
            },
            "ds_colbert_enabled": {
                "type": "boolean",
                "description": "ColBERT Multi-Vector-Index aktivieren",
            },
        },
        "required": ["name", "type", "connector_config"],
    },
    returns="Erstellte Datenquelle mit ID, Name und Typ",
    pk_param="name",
    changelog_area="datasources",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/datasources"),
)
async def datasource_create(
    user: dict,
    name: str,
    type: str,
    connector_config: dict,
    ds_chunk_strategy: str = "auto",
    ds_chunk_size: int | None = None,
    ds_chunk_overlap: int | None = None,
    ds_file_types: list[str] | None = None,
    ds_exclude_patterns: list[str] | None = None,
    ds_ddu_enabled: bool = False,
    ds_ddu_ocr_images: bool = True,
    ds_vlm_enabled: bool = False,
    ds_accept_all_text: bool = False,
    ds_index_attachments: bool = True,
    sync_interval: str = "global",
    restriction_level: str = "anonymize",
    ds_contextual_retrieval: bool = False,
    ds_enrichment_prompt_id: int | None = None,
    ds_enrichment_backend_id: int | None = None,
    ds_colbert_enabled: bool = False,
) -> AdminResult:
    try:
        from kiara.admin.services.datasource_service import create_datasource

        result = await create_datasource(
            name=name,
            ds_type=type,
            connector_config=connector_config,
            ds_chunk_strategy=ds_chunk_strategy,
            ds_chunk_size=ds_chunk_size,
            ds_chunk_overlap=ds_chunk_overlap,
            ds_file_types=ds_file_types,
            ds_exclude_patterns=ds_exclude_patterns,
            ds_ddu_enabled=ds_ddu_enabled,
            ds_ddu_ocr_images=ds_ddu_ocr_images,
            ds_vlm_enabled=ds_vlm_enabled,
            ds_accept_all_text=ds_accept_all_text,
            ds_index_attachments=ds_index_attachments,
            sync_interval=sync_interval,
            restriction_level=restriction_level,
            ds_contextual_retrieval=ds_contextual_retrieval,
            ds_enrichment_prompt_id=ds_enrichment_prompt_id,
            ds_enrichment_backend_id=ds_enrichment_backend_id,
            ds_colbert_enabled=ds_colbert_enabled,
        )
        # raw_response=True → data muss bestehendes Frontend-Format haben
        return AdminResult.ok(f"Datenquelle '{name}' erstellt (ID {result['id']})", data=result)
    except ValueError as exc:
        return AdminResult.fail(str(exc))


@admin_function(
    name="datasource.update",
    category="datasources",
    description="Datenquelle aktualisieren",
    help="Aktualisiert Name, Konnektor-Konfiguration oder Indexierungs-Einstellungen einer "
    "bestehenden Datenquelle. Nur übergebene Felder werden geändert (Partial Update). "
    "Passwort-Felder mit dem Wert '********' behalten den bestehenden Wert bei.",
    examples=[
        "Ändere den Namen von Datenquelle 5 zu 'Buchhaltung'",
        "Aktualisiere das Passwort der WebDAV-Quelle 12",
        "Deaktiviere DDU für Datenquelle 8",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "ID der Datenquelle"},
            "name": {"type": "string", "description": "Neuer Anzeigename"},
            "is_active": {"type": "boolean", "description": "Datenquelle aktivieren/deaktivieren"},
            "connector_config": {"type": "object", "description": "Neue Konnektor-Konfiguration"},
            "ds_chunk_strategy": {"type": "string", "description": "Chunking-Strategie"},
            "ds_chunk_size": {"type": "integer", "description": "Chunk-Größe"},
            "ds_chunk_overlap": {"type": "integer", "description": "Chunk-Überlappung"},
            "ds_file_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dateifilter",
            },
            "ds_exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ausschluss-Muster",
            },
            "ds_ddu_enabled": {"type": "boolean", "description": "DDU aktivieren/deaktivieren"},
            "ds_ddu_ocr_images": {"type": "boolean", "description": "OCR für Bilder"},
            "ds_vlm_enabled": {"type": "boolean", "description": "VLM aktivieren/deaktivieren"},
            "ds_accept_all_text": {
                "type": "boolean",
                "description": "Alle Textdateien akzeptieren",
            },
            "ds_index_attachments": {"type": "boolean", "description": "E-Mail-Anhänge indexieren"},
            "sync_interval": {"type": "string", "description": "Sync-Intervall"},
            "restriction_level": {
                "type": "string",
                "description": "Restriction-Level (public, anonymize, blocked)",
                "enum": ["public", "anonymize", "blocked"],
            },
            "ds_contextual_retrieval": {
                "type": "boolean",
                "description": "Contextual Retrieval aktivieren",
            },
            "ds_enrichment_prompt_id": {
                "type": "integer",
                "description": "Enrichment-Prompt-Template-ID",
            },
            "ds_enrichment_backend_id": {
                "type": "integer",
                "description": "Enrichment-Backend-ID (Pflicht wenn CR aktiv)",
            },
            "ds_colbert_enabled": {
                "type": "boolean",
                "description": "ColBERT Multi-Vector-Index aktivieren",
            },
        },
        "required": ["ds_id"],
    },
    returns="Aktualisierte Datenquelle",
    pk_param="ds_id",
    changelog_area="datasources",
    changelog_action="update",
    raw_response=True,
    http=("PUT", "/api/admin/datasources/{ds_id}"),
)
async def datasource_update(
    user: dict, ds_id: int, name: str | None = None, connector_config: dict | None = None, **kwargs
) -> AdminResult:
    ds_repo_local = DatasourceRepository()
    ds = ds_repo_local.get_by_id(ds_id)
    if not ds:
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")

    fields = {}
    if name is not None:
        fields["name"] = name
    if "is_active" in kwargs and kwargs["is_active"] is not None:
        fields["is_active"] = bool(kwargs["is_active"])

    # connector_config: PASSWORD_MASK aufloesen
    if connector_config is not None:
        from kiara.admin.services.datasource_service import resolve_masked_config
        from kiara.connectors.registry import ensure_loaded
        from kiara.connectors.registry import get as get_connector

        ensure_loaded()
        connector_cls = get_connector(ds["type"])
        if not connector_cls:
            return AdminResult.fail(f"Unbekannter Konnektor-Typ: {ds['type']}")
        try:
            existing_config = json.loads(ds.get("connector_config", "{}"))
        except (ValueError, TypeError):
            existing_config = {}
        try:
            fields["connector_config"] = json.dumps(
                resolve_masked_config(connector_config, existing_config, connector_cls)
            )
        except Exception as exc:
            return AdminResult.fail(f"Config-Verarbeitung fehlgeschlagen: {exc}")

    # ds_*-Felder: nur gesetzte Werte uebernehmen
    ds_fields = [
        "ds_chunk_strategy",
        "ds_chunk_size",
        "ds_chunk_overlap",
        "ds_file_types",
        "ds_exclude_patterns",
        "ds_ddu_enabled",
        "ds_ddu_ocr_images",
        "ds_vlm_enabled",
        "ds_accept_all_text",
        "ds_index_attachments",
        "sync_interval",
        "restriction_level",
        "ds_contextual_retrieval",
        "ds_colbert_enabled",
    ]
    for f in ds_fields:
        if f in kwargs and kwargs[f] is not None:
            val = kwargs[f]
            # bool bleibt bool fuer echte boolean-Spalten (ds_ddu_enabled,
            # ds_vlm_enabled, ds_colbert_enabled). Fuer integer-Spalten in
            # _INT_DS_FIELDS wird der vom JS gesendete bool nach int gecastet.
            # list wird JSON-serialisiert (Text-Spalten).
            if isinstance(val, list):
                try:
                    val = json.dumps(val) if val else ""
                except (TypeError, ValueError) as exc:
                    return AdminResult.fail(f"'{f}' enthält nicht-serialisierbare Werte: {exc}")
            elif f in _INT_DS_FIELDS and isinstance(val, bool):
                val = int(val)
            fields[f] = val

    # Chunk-Parameter validieren (gleiche Regeln wie bei datasource.create)
    new_chunk_size = fields.get("ds_chunk_size")
    new_chunk_overlap = fields.get("ds_chunk_overlap")
    if new_chunk_size is not None and new_chunk_size <= 0:
        return AdminResult.fail("ds_chunk_size muss groesser als 0 sein")
    if new_chunk_overlap is not None and new_chunk_overlap < 0:
        return AdminResult.fail("ds_chunk_overlap darf nicht negativ sein")
    eff_size = new_chunk_size if new_chunk_size is not None else ds.get("ds_chunk_size")
    eff_overlap = new_chunk_overlap if new_chunk_overlap is not None else ds.get("ds_chunk_overlap")
    if eff_size is not None and eff_overlap is not None and eff_overlap >= eff_size:
        return AdminResult.fail("ds_chunk_overlap muss kleiner als ds_chunk_size sein")

    # Nullable DS-Felder: explizites None -> NULL in DB
    _nullable_ds = {"ds_enrichment_prompt_id", "ds_enrichment_backend_id"}
    for nf in _nullable_ds:
        if nf in kwargs:
            val = kwargs[nf]
            if val is not None and (not isinstance(val, int) or val <= 0):
                return AdminResult.fail(f"'{nf}' muss eine positive Ganzzahl oder null sein")
            fields[nf] = val

    if not fields:
        return AdminResult.fail("Keine Änderungen angegeben")

    # Validierung: CR aktiv → Backend muss gesetzt sein
    cr_active = fields.get("ds_contextual_retrieval", ds.get("ds_contextual_retrieval", 0))
    backend_set = fields.get("ds_enrichment_backend_id", ds.get("ds_enrichment_backend_id"))
    if cr_active and not backend_set:
        return AdminResult.fail(
            "Contextual Retrieval erfordert ein explizites Enrichment-Backend. "
            "Bitte ein LLM- oder Cloud-Backend waehlen."
        )

    ds_repo_local.update(ds_id, **fields)
    from kiara.web.cache_refresh import refresh_datasource_cache

    refresh_datasource_cache()

    # raw_response=True → Format muss bestehendem Endpunkt entsprechen
    return AdminResult.ok(
        f"Datenquelle {ds_id} aktualisiert", data={"message": "Datenquelle aktualisiert"}
    )


@admin_function(
    name="datasource.delete",
    category="datasources",
    description="Datenquelle und alle zugehörigen Daten löschen",
    help="Löscht eine Datenquelle vollständig: Qdrant-Vektoren (alle Collections), "
    "Elasticsearch-Einträge, File-Hash-Store, DB-Cascade (11 Tabellen: "
    "ingestion_file_logs, file_hashes, hash_mode_meta, chunk_datasources, "
    "datasource_embeddings, datasource_groups, datasource_users, ingestion_jobs, "
    "cache_datasources, cache_jobs, datasources), Mount-Points und Cron-Schedules. "
    "IRREVERSIBEL — erfordert Bestätigung.",
    examples=[
        "Lösche Datenquelle 5",
        "Datenquelle 'Altdaten' entfernen",
    ],
    parameters={
        "type": "object",
        "properties": {"ds_id": {"type": "integer", "description": "ID der Datenquelle"}},
        "required": ["ds_id"],
    },
    returns="Löschbestätigung mit bereinigten Abhängigkeiten",
    requires_confirmation=True,
    confirm_template="ds {ds_id} löschen",
    pk_param="ds_id",
    changelog_area="datasources",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/datasources/{ds_id}"),
)
async def datasource_delete(user: dict, ds_id: int) -> AdminResult:
    try:
        from kiara.admin.services.datasource_service import delete_datasource_full

        result = await delete_datasource_full(ds_id)
        # raw_response=True → {"message": "..."} wie bestehender Endpunkt
        return AdminResult.ok("Datenquelle gelöscht", data=result)
    except ValueError as exc:
        return AdminResult.fail(str(exc))


@admin_function(
    name="datasource.test_connection",
    category="datasources",
    description="Verbindungstest für eine Datenquelle",
    help="Testet die Verbindung zu einer Datenquelle. Erfordert immer type + config. "
    "Wird zusätzlich ds_id angegeben, werden PASSWORD_MASK-Felder in config "
    "automatisch aus der bestehenden Datenquelle aufgelöst.",
    examples=[
        "Teste die Verbindung zu Datenquelle 5",
        "Teste WebDAV-Verbindung zu https://cloud.firma.de",
    ],
    parameters={
        "type": "object",
        "properties": {
            "type": {"type": "string", "description": "Konnektor-Typ"},
            "config": {"type": "object", "description": "Verbindungsdaten"},
            "ds_id": {
                "type": "integer",
                "description": "ID einer bestehenden Datenquelle (optional, löst PASSWORD_MASK-Felder auf)",
            },
        },
        "required": ["type", "config"],
    },
    returns="Verbindungsstatus (success/fail) mit Details",
    raw_response=True,
    http=("POST", "/api/admin/datasources/test-connection"),
)
async def datasource_test_connection(
    user: dict, type: str, config: dict, ds_id: int | None = None
) -> AdminResult:
    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector

    ensure_loaded()
    connector_cls = get_connector(type)
    if not connector_cls:
        return AdminResult.fail(f"Unbekannter Konnektor-Typ: {type}")

    # Bei bestehender DS: PASSWORD_MASK aufloesen
    test_config = dict(config)
    if ds_id:
        from kiara.admin.services.datasource_service import resolve_masked_config

        ds_repo_local = DatasourceRepository()
        ds = ds_repo_local.get_by_id(ds_id)
        if ds:
            if ds.get("type") and ds.get("type") != type:
                return AdminResult.fail("PASSWORD_MASK passt nicht zum Konnektor-Typ")
            try:
                existing = json.loads(ds.get("connector_config", "{}"))
            except (ValueError, TypeError):
                existing = {}
            test_config = resolve_masked_config(test_config, existing, connector_cls)

    # resolve_masked_config gibt bei PASSWORD_MASK den verschluesselten DB-Wert zurueck.
    # safe_test_connection braucht aber Klartext → entschluesseln.
    from kiara.shared.crypto import decrypt, is_encrypted

    for field in connector_cls.secret_fields():
        if field in test_config and test_config[field] and is_encrypted(test_config[field]):
            test_config[field] = decrypt(test_config[field])

    # safe_test_connection nimmt CONNECTOR config (nicht App-Config!) und ein Row-Dict
    row = {"id": ds_id or "neu", "name": "(Verbindungstest)", "type": type}
    try:
        result = connector_cls.safe_test_connection(test_config, row)
    finally:
        for field in connector_cls.secret_fields():
            if field in test_config:
                test_config[field] = ""
    if result.get("success"):
        return AdminResult.ok("Verbindung erfolgreich", data=result)
    return AdminResult.fail(result.get("message", "Verbindung fehlgeschlagen"))


def _connector_secret_fields(connector_cls) -> list[str]:
    fields = connector_cls.secret_fields()
    if isinstance(fields, (list, tuple, set)):
        return list(fields)
    return []


@admin_function(
    name="datasource.browse",
    category="datasources",
    description="Verzeichnis-Inhalt einer Datenquelle auflisten",
    help="Listet Dateien und Ordner an einem Pfad auf. Dispatcht intern nach Typ: "
    "SMB, WebDAV, IMAP/Exchange (list-folders), "
    "Lokal (Dateisystem). Nützlich vor datasource.create zum Pfad-Erkunden.",
    examples=[
        "Zeig mir die Ordner auf dem SMB-Share",
        "Was liegt im WebDAV-Verzeichnis /files/hr?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "description": "Konnektor-Typ (smb, webdav, imap, exchange_online, local)",
            },
            "config": {"type": "object", "description": "Verbindungsdaten"},
            "path": {"type": "string", "description": "Pfad zum Browsen (Default: Root)"},
            "ds_id": {
                "type": "integer",
                "minimum": 1,
                "description": "ID einer bestehenden Datenquelle (optional, loest PASSWORD_MASK-Felder auf)",
            },
        },
        "required": ["type", "config"],
    },
    returns="Liste von Dateien/Ordnern am angegebenen Pfad",
)
async def datasource_browse(
    user: dict,
    type: str,
    config: dict,
    path: str = "",
    ds_id: int | None = None,
) -> AdminResult:
    if ds_id is not None and ds_id <= 0:
        return AdminResult.fail("Ungueltige Datenquellen-ID")

    from kiara.connectors.browse import ConnectorBrowseError, normalize_browse_path, parent_path
    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector

    ensure_loaded()
    try:
        connector_cls = get_connector(type)
    except KeyError:
        # Registry wirft KeyError bei unbekanntem Typ (Auto-Fix-Haertung).
        return AdminResult.fail(f"Browse nicht unterstützt für Typ: {type}")
    if not connector_cls:
        return AdminResult.fail(f"Browse nicht unterstützt für Typ: {type}")

    effective_config = dict(config or {})
    secret_fields = _connector_secret_fields(connector_cls)

    if ds_id is not None:
        ds_repo_local = DatasourceRepository()
        ds = ds_repo_local.get_by_id(ds_id)
        if not ds:
            return AdminResult.fail("Datenquelle nicht gefunden")
        if ds.get("type") and ds.get("type") != type:
            return AdminResult.fail("PASSWORD_MASK passt nicht zum Konnektor-Typ")
        try:
            raw_existing = ds.get("connector_config", "{}")
            if isinstance(raw_existing, dict):
                existing = dict(raw_existing)
            else:
                existing = json.loads(raw_existing) if raw_existing else {}
        except (ValueError, TypeError):
            existing = {}

        for field in secret_fields:
            if effective_config.get(field) == PASSWORD_MASK and not existing.get(field):
                return AdminResult.fail("Passwort erforderlich")

        from kiara.admin.services.datasource_service import resolve_masked_config

        try:
            effective_config = resolve_masked_config(effective_config, existing, connector_cls)
        except Exception as exc:
            return AdminResult.fail(f"Config-Verarbeitung fehlgeschlagen: {exc}")

        from kiara.shared.crypto import decrypt, is_encrypted

        for field in secret_fields:
            value = effective_config.get(field)
            if value and is_encrypted(value):
                effective_config[field] = decrypt(value)

    for field in secret_fields:
        if effective_config.get(field) == PASSWORD_MASK:
            return AdminResult.fail("Passwort erforderlich")

    if type in ("imap", "exchange_online"):
        try:
            folders = connector_cls.list_folders(effective_config)
        except Exception as exc:
            return AdminResult.fail(str(exc))
        return AdminResult.ok("Ordner-Liste", data={"folders": folders})

    if getattr(connector_cls, "supports_browse", False) is True:
        try:
            normalized = normalize_browse_path(path)
            result = connector_cls.list_directory(
                effective_config,
                normalized,
                directories_only=False,
            )
        except ConnectorBrowseError as exc:
            return AdminResult.fail(exc.message)
        except Exception as exc:
            return AdminResult.fail(str(exc))
        return AdminResult.ok(
            "Verzeichnis-Inhalt",
            data={
                "entries": result["entries"],
                "path": normalized,
                "parent": parent_path(normalized),
            },
        )

    return AdminResult.fail(f"Browse nicht unterstützt für Typ: {type}")


@admin_function(
    name="datasource.assign_groups",
    category="datasources",
    description="Gruppen einer Datenquelle zuweisen",
    help="Setzt die Gruppen-Zuordnung einer Datenquelle. ACHTUNG: Ersetzt ALLE "
    "bestehenden Zuordnungen (Replace-All, nicht Append). Leere Liste "
    "entfernt alle Gruppenzuordnungen.",
    examples=[
        "Weise Datenquelle 5 den Gruppen 1 und 3 zu",
        "Entferne alle Gruppenzuordnungen von DS 8",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "ID der Datenquelle"},
            "group_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Gruppen-IDs (ersetzt alle bestehenden)",
            },
        },
        "required": ["ds_id", "group_ids"],
    },
    returns="Bestätigung der Gruppenzuordnung",
    requires_confirmation=True,
    confirm_template="ds {ds_id} gruppen zuweisen",
    pk_param="ds_id",
    changelog_area="datasources",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/datasources/{ds_id}/groups"),
)
async def datasource_assign_groups(user: dict, ds_id: int, group_ids: list[int]) -> AdminResult:
    from kiara.repositories.permission import PermissionRepository

    ds_repo_local = DatasourceRepository()
    if not ds_repo_local.exists(ds_id):
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")
    perm_repo_local = PermissionRepository()
    try:
        perm_repo_local.set_datasource_groups(ds_id, group_ids)
    except ValueError as exc:
        return AdminResult.fail(str(exc))
    # raw_response=True → Format wie bestehender Endpunkt
    return AdminResult.ok(
        f"{len(group_ids)} Gruppen zugewiesen", data={"message": "Gruppen zugewiesen"}
    )


@admin_function(
    name="datasource.assign_users",
    category="datasources",
    description="User einer Datenquelle zuweisen",
    help="Setzt die User-Zuordnung einer Datenquelle. ACHTUNG: Ersetzt ALLE "
    "bestehenden Zuordnungen (Replace-All, nicht Append).",
    examples=[
        "Weise Datenquelle 5 den Usern 2 und 7 zu",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "ID der Datenquelle"},
            "user_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "User-IDs (ersetzt alle bestehenden)",
            },
        },
        "required": ["ds_id", "user_ids"],
    },
    returns="Bestätigung der User-Zuordnung",
    requires_confirmation=True,
    confirm_template="ds {ds_id} user zuweisen",
    pk_param="ds_id",
    changelog_area="datasources",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/datasources/{ds_id}/users"),
)
async def datasource_assign_users(user: dict, ds_id: int, user_ids: list[int]) -> AdminResult:
    from kiara.repositories.permission import PermissionRepository

    ds_repo_local = DatasourceRepository()
    if not ds_repo_local.exists(ds_id):
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")
    perm_repo_local = PermissionRepository()
    try:
        perm_repo_local.set_datasource_users(ds_id, user_ids)
    except ValueError as exc:
        return AdminResult.fail(str(exc))
    # raw_response=True → Format wie bestehender Endpunkt
    return AdminResult.ok(
        f"{len(user_ids)} User zugewiesen", data={"message": "Benutzer zugewiesen"}
    )
