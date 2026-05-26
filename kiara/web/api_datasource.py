"""KIara Admin - Multi-Datasource API.

CRUD, Verbindungstest, Status, Gruppen-Zuweisung, Konnektor-Typen.
Nutzt die Registry-basierte Konnektor-Plugin-Architektur mit
``connector_config`` JSON-Spalte fuer konnektor-spezifische Settings.

Modulstruktur:
    - api_datasource.py         — CRUD, Test, Status, Gruppen, Konnektor-Typen (dieses Modul)
    - api_datasource_browse.py  — Browse-Endpunkte (Datei-/Verzeichnis-Picker)
    - api_datasource_ingest.py  — Ingestion-Endpunkte (Start, Status, Pause, Resume)
    - api_datasource_features.py — Feature-Checks (DDU, VLM, Embedding-Modelle)

Beziehungen:
    - kiara.connectors.registry → Konnektor-Klassen laden/abrufen
    - kiara.web.config_utils    → safe_json_loads, resolve_masked_field
    - kiara.shared.constants    → PASSWORD_MASK
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from kiara.connectors.registry import (
    ensure_loaded,
)
from kiara.connectors.registry import (
    get as get_connector,
)
from kiara.repositories.cache import CacheRepository
from kiara.repositories.datasource import DatasourceRepository
from kiara.repositories.permission import PermissionRepository
from kiara.shared.constants import PASSWORD_MASK
from kiara.shared.crypto import decrypt, encrypt_secret_fields, is_encrypted
from kiara.shared.enums import ConnectorType
from kiara.web.auth import get_admin_user
from kiara.web.config_utils import safe_json_loads

logger = logging.getLogger(__name__)

router = APIRouter(tags=["datasource"])

# ---------------------------------------------------------------------------
# Repository-Instanzen (Module-Level Singletons)
# ---------------------------------------------------------------------------
ds_repo = DatasourceRepository()
perm_repo = PermissionRepository()
cache_repo = CacheRepository()

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DatasourceRequest(BaseModel):
    """Request-Body fuer Erstellen/Aktualisieren einer Datenquelle.

    Enthaelt konnektor-spezifische Config (connector_config als JSON-Dict)
    und per-Datasource Ingestion-Einstellungen (Chunk-Strategie, Dateitypen etc.).
    """

    name: str = ""
    type: str = "local"
    connector_config: dict = {}
    # Per-DS Config (nicht konnektor-spezifisch, bleibt als Einzelfelder)
    ds_chunk_strategy: str = "auto"
    ds_chunk_size: int | None = None
    ds_chunk_overlap: int | None = None
    ds_file_types: list[str] | None = None
    ds_exclude_patterns: list[str] | None = None
    ds_ddu_enabled: bool = False
    ds_ddu_ocr_images: bool = True
    ds_vlm_enabled: bool = False
    ds_accept_all_text: bool = False
    ds_index_attachments: bool = True
    sync_interval: str = "global"
    restriction_level: str = "anonymize"
    ds_contextual_retrieval: bool = False
    ds_enrichment_prompt_id: int | None = None
    ds_enrichment_backend_id: int | None = None
    ds_colbert_enabled: bool = False


class GroupAssignRequest(BaseModel):
    """Request-Body fuer Gruppen-Zuweisung an eine Datenquelle."""

    group_ids: list[int] = []


class UserAssignRequest(BaseModel):
    """Request-Body fuer User-Zuweisung an eine Datenquelle."""

    user_ids: list[int] = []


class ShareBrowseRequest(BaseModel):
    """SMB-Folder-Picker — eigene Felder fuer nicht-gespeicherte Datasources."""

    server: str = ""
    share: str = ""
    username: str = ""
    password: str = ""
    domain: str = ""
    path: str = ""
    ds_id: int | None = None


class WebDAVBrowseRequest(BaseModel):
    """WebDAV-Folder-Picker — Verzeichnisse per PROPFIND auflisten."""

    url: str = ""
    username: str = ""
    password: str = ""
    path: str = ""
    ds_id: int | None = None


class ConnectorActionRequest(BaseModel):
    """Request-Body fuer test-connection und list-folders Endpoints."""

    type: str
    config: dict
    ds_id: int | None = None


# ---------------------------------------------------------------------------
# Connector-Config Hilfsfunktionen
# ---------------------------------------------------------------------------


def _get_connector_config(row: dict) -> dict:
    """Liest und parst connector_config aus einer DB-Row.

    Die connector_config-Spalte enthaelt konnektor-spezifische Einstellungen
    als JSON-String (z.B. SMB-Server/Share/Credentials, IMAP-Host/Port/Folders).
    Gibt ein leeres Dict zurueck wenn die Spalte fehlt oder ungueltig ist.

    Args:
        row: Datasource-Row aus der DB (dict-artiges Row-Objekt oder dict).

    Returns:
        Geparstes Config-Dict, oder {} bei fehlendem/ungueltigen JSON.
    """
    raw = row.get("connector_config", "{}")
    return safe_json_loads(raw, {})


def _decrypt_config(config: dict, connector_cls: type) -> dict:
    """Entschluesselt Secret-Felder in der Connector-Config.

    Args:
        config:        Connector-Config-Dict mit verschluesselten Secret-Feldern.
        connector_cls: Konnektor-Klasse mit ``secret_fields()``-Methode.

    Returns:
        Neues Dict mit entschluesselten Secret-Feldern.
    """
    decrypted = dict(config)
    for field in connector_cls.secret_fields():
        value = decrypted.get(field, "")
        if value and is_encrypted(value):
            try:
                decrypted[field] = decrypt(value)
            except Exception as exc:
                logger.warning("Entschluesselung fehlgeschlagen fuer Feld '%s': %s", field, exc)
                decrypted[field] = ""
    return decrypted


def _resolve_masked_config(
    config: dict,
    connector_cls,
    ds_id: int | None,
    expected_type: str | None = None,
) -> dict:
    """Loest PASSWORD_MASK-Werte gegen gespeicherte Secrets auf.

    Im Frontend werden Secret-Felder (Passwoerter, DSNs, API-Keys) als
    PASSWORD_MASK ('********') angezeigt. Wenn der User einen Verbindungstest
    ausfuehrt ohne das Passwort erneut einzugeben, kommt der Mask-Wert
    im Request an. Diese Methode ersetzt ihn durch den gespeicherten,
    entschluesselten Wert.

    Wird von test-connection und list-folders gemeinsam genutzt.

    Args:
        config:        Connector-Config aus dem Request-Body.
        connector_cls: Konnektor-Klasse (liefert secret_fields()).
        ds_id:         ID der bestehenden Datasource (fuer Lookup der
                       gespeicherten Secrets). None bei neuer Datasource.

    Raises:
        HTTPException 400: Wenn ein Mask-Wert ohne ds_id oder ohne
                           gespeicherten Wert aufgeloest werden soll.

    Returns:
        Config-Dict mit aufgeloesten Secret-Feldern.
    """
    config = dict(config)
    stored_config = None
    if ds_id:
        stored_row = ds_repo.get_by_id(ds_id)
        if stored_row:
            if expected_type and "type" in stored_row and stored_row["type"] != expected_type:
                raise HTTPException(400, "PASSWORD_MASK passt nicht zum Konnektor-Typ")
            stored_config = _decrypt_config(
                _get_connector_config(stored_row),
                connector_cls,
            )

    for field in connector_cls.secret_fields():
        if config.get(field) != PASSWORD_MASK:
            continue
        if not stored_config:
            raise HTTPException(400, f"Feld '{field}' ist erforderlich")
        resolved = stored_config.get(field)
        if not resolved:
            raise HTTPException(500, f"Gespeichertes Feld '{field}' ist leer")
        config[field] = resolved

    return config


def _encrypt_config(config: dict, connector_cls: type, existing_config: dict | None = None) -> dict:
    """Verschluesselt Secret-Felder in der Connector-Config fuer DB-Persistierung.

    Delegiert an ``encrypt_secret_fields()`` aus ``shared.crypto``.
    """
    return encrypt_secret_fields(config, connector_cls.secret_fields(), existing_config)


# ---------------------------------------------------------------------------
# Display- und Pfad-Funktionen (Registry-basiert)
# ---------------------------------------------------------------------------

_OLD_CONNECTOR_FIELDS = {
    "smb_server",
    "smb_share",
    "smb_username",
    "smb_password",
    "smb_domain",
    "smb_mount_point",
    "webdav_url",
    "webdav_username",
    "webdav_password",
    "webdav_base_path",
    "imap_server",
    "imap_port",
    "imap_ssl",
    "imap_username",
    "imap_password",
    "imap_folders",
    "imap_since_days",
    "imap_download_attachments",
    "imap_max_attachment_mb",
    "sql_dsn",
    "sql_driver",
    "sql_query",
    "sql_text_columns",
    "sql_id_column",
    "sql_timestamp_column",
    "sql_text_template",
}


def _mask_row(row: dict) -> dict:
    """Convert a datasource row to a dict with masked passwords."""
    ensure_loaded()
    d = dict(row)

    config = _get_connector_config(row)
    connector_cls = get_connector(row["type"])
    if connector_cls:
        for field in connector_cls.secret_fields():
            if config.get(field):
                config[field] = PASSWORD_MASK
    d["connector_config"] = config

    # path fuer local-Typ aus connector_config spiegeln
    if d.get("type") == ConnectorType.LOCAL:
        d["path"] = config.get("path", "")

    # Alte Konnektor-Spalten aus dem Response entfernen
    for field in _OLD_CONNECTOR_FIELDS:
        d.pop(field, None)

    return d


def _source_display(row: dict) -> str:
    """Format the remote source for display via Registry."""
    ensure_loaded()
    connector_cls = get_connector(row["type"])
    if not connector_cls:
        try:
            return row["path"] or ""
        except (KeyError, IndexError):
            return ""
    config = _get_connector_config(row)
    return connector_cls.source_display(config, dict(row))


def _effective_path(row: dict) -> str:
    """Compute the effective filesystem path via Registry."""
    ensure_loaded()
    connector_cls = get_connector(row["type"])
    if not connector_cls:
        return ""
    config = _get_connector_config(row)
    return connector_cls.effective_path(config, dict(row))


# ---------------------------------------------------------------------------
# CRUD Endpoints
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: datasource.list (Sprint 04) ---
# @router.get("/api/admin/datasources")
# async def list_datasources(user: dict = Depends(get_admin_user)) -> dict:
#     """List all datasources — liest aus vorberechneter Cache-Tabelle."""
#     cached = cache_repo.get_cached_datasources()
#     if not cached:
#         from kiara.web.cache_refresh import refresh_datasource_cache
#         refresh_datasource_cache()
#         cached = cache_repo.get_cached_datasources()
#     return {"data": cached}

# --- Migriert zu Admin-Registry: datasource.create (Sprint 05) ---
# @router.post("/api/admin/datasources")
# async def create_datasource(data: DatasourceRequest, user: dict = Depends(get_admin_user)) -> dict:
#     """Create a new datasource."""
#     ensure_loaded()
#     if not data.name.strip():
#         raise HTTPException(status_code=400, detail="Name ist erforderlich")
#     connector_cls = get_connector(data.type)
#     if not connector_cls:
#         raise HTTPException(status_code=400, detail=f"Unbekannter Datenquellen-Typ: {data.type}")
#     config = _encrypt_config(data.connector_config, connector_cls)
#     config_json = json.dumps(config, ensure_ascii=False)
#     file_types_json = json.dumps(data.ds_file_types) if data.ds_file_types else None
#     exclude_json = json.dumps(data.ds_exclude_patterns) if data.ds_exclude_patterns else None
#     new_id = ds_repo.create(
#         name=data.name.strip(), type=data.type, connector_config=config_json,
#         ds_chunk_strategy=data.ds_chunk_strategy, ds_chunk_size=data.ds_chunk_size,
#         ds_chunk_overlap=data.ds_chunk_overlap, ds_file_types=file_types_json,
#         ds_exclude_patterns=exclude_json, ds_ddu_enabled=int(data.ds_ddu_enabled),
#         ds_ddu_ocr_images=int(data.ds_ddu_ocr_images), ds_vlm_enabled=int(data.ds_vlm_enabled),
#         ds_accept_all_text=int(data.ds_accept_all_text),
#         ds_index_attachments=int(data.ds_index_attachments), sync_interval=data.sync_interval,
#     )
#     logger.info("Datasource created: id=%s name='%s' by %s", new_id, data.name, user["username"])
#     from kiara.web.cache_refresh import refresh_datasource_cache
#     safe_refresh(refresh_datasource_cache, new_id)
#     return {"id": new_id, "message": "Datenquelle erstellt"}

# --- Migriert zu Admin-Registry: datasource.update (Sprint 05) ---
# @router.put("/api/admin/datasources/{ds_id}")
# async def update_datasource(ds_id: int, data: DatasourceRequest, user: dict = Depends(get_admin_user)) -> dict:
#     """Update an existing datasource."""
#     ensure_loaded()
#     existing = ds_repo.get_by_id(ds_id)
#     if not existing:
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     connector_cls = get_connector(data.type)
#     if not connector_cls:
#         raise HTTPException(status_code=400, detail=f"Unbekannter Datenquellen-Typ: {data.type}")
#     existing_config = _get_connector_config(existing)
#     config = _encrypt_config(data.connector_config, connector_cls, existing_config)
#     config_json = json.dumps(config, ensure_ascii=False)
#     ds_repo.update(ds_id, name=data.name.strip() or existing["name"], type=data.type,
#         connector_config=config_json, ds_chunk_strategy=data.ds_chunk_strategy,
#         ds_chunk_size=data.ds_chunk_size, ds_chunk_overlap=data.ds_chunk_overlap,
#         ds_ddu_enabled=int(data.ds_ddu_enabled), ds_ddu_ocr_images=int(data.ds_ddu_ocr_images),
#         ds_vlm_enabled=int(data.ds_vlm_enabled), ds_accept_all_text=int(data.ds_accept_all_text),
#         ds_index_attachments=int(data.ds_index_attachments), sync_interval=data.sync_interval,
#         updated_at=RawSQL("NOW()"),
#     )
#     logger.info("Datasource updated: id=%s by %s", ds_id, user["username"])
#     from kiara.web.cache_refresh import refresh_datasource_cache
#     safe_refresh(refresh_datasource_cache, ds_id)
#     return {"message": "Datenquelle aktualisiert"}

# ---------------------------------------------------------------------------
# Test / Status
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: datasource.test_connection (Sprint 05) ---
# @router.post("/api/admin/datasources/test-connection")
# def test_datasource_connection(body: ConnectorActionRequest, user: dict = Depends(get_admin_user)) -> dict:
#     """Test connectivity with ad-hoc config (Create- und Edit-Dialog)."""
#     ds_type = body.type
#     config = body.config
#     ds_id = body.ds_id
#     ensure_loaded()
#     connector_cls = get_connector(ds_type)
#     if not connector_cls:
#         raise HTTPException(400, f"Unbekannter Konnektor-Typ: {ds_type}")
#     if not isinstance(config, dict):
#         raise HTTPException(400, "config muss ein Dict sein")
#     config = _resolve_masked_config(config, connector_cls, ds_id)
#     row = {"id": ds_id or "neu", "name": "(Verbindungstest)", "type": ds_type}
#     result = connector_cls.safe_test_connection(config, row)
#     if not result.get("success"):
#         raise HTTPException(502, result.get("message", "Verbindungstest fehlgeschlagen"))
#     return result


@router.post("/api/admin/datasources/list-folders")
def list_datasource_folders(
    body: ConnectorActionRequest, user: dict = Depends(get_admin_user)
) -> dict:
    """Ordnerliste fuer Folder-Picker (IMAP/Exchange)."""
    ds_type = body.type
    config = body.config
    ds_id = body.ds_id

    ensure_loaded()

    connector_cls = get_connector(ds_type)
    if not connector_cls:
        raise HTTPException(400, f"Unbekannter Konnektor-Typ: {ds_type}")

    if not hasattr(connector_cls, "list_folders"):
        raise HTTPException(400, "Ordnerauswahl wird fuer diesen Typ nicht unterstuetzt")

    if not isinstance(config, dict):
        raise HTTPException(400, "config muss ein Dict sein")

    config = _resolve_masked_config(config, connector_cls, ds_id, expected_type=ds_type)

    try:
        folders = connector_cls.list_folders(config)
        return {"success": True, "folders": folders}
    except Exception as exc:
        logger.warning("list-folders fehlgeschlagen fuer %s: %s", ds_type, exc)
        raise HTTPException(502, str(exc))


# --- Migriert zu Admin-Registry: datasource.status (Sprint 04) ---
# @router.get("/api/admin/datasources/{ds_id}/status")
# async def get_datasource_status(ds_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     """Get datasource status (path, accessibility, file count)."""
#     row = ds_repo.get_by_id(ds_id)
#     if not row:
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     path = _effective_path(row)
#     if not path:
#         return {"path": "", "accessible": False, "file_count": 0}
#     accessible = os.path.isdir(path)
#     file_count = count_files_limited(path) if accessible else 0
#     return {"path": path, "accessible": accessible, "file_count": file_count}

# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: datasource.groups (Sprint 04) ---
# @router.get("/api/admin/datasources/{ds_id}/groups")
# async def get_datasource_groups(ds_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     """Get groups assigned to a datasource."""
#     if not ds_repo.exists(ds_id):
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     groups = perm_repo.get_datasource_groups(ds_id)
#     return {"groups": groups}

# --- Migriert zu Admin-Registry: datasource.assign_groups (Sprint 05) ---
# @router.post("/api/admin/datasources/{ds_id}/groups")
# async def set_datasource_groups(ds_id: int, data: GroupAssignRequest, user: dict = Depends(get_admin_user)) -> dict:
#     """Set groups for a datasource (replace all)."""
#     if not ds_repo.exists(ds_id):
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     perm_repo.set_datasource_groups(ds_id, data.group_ids)
#     logger.info("Datasource %s groups set to %s by %s", ds_id, data.group_ids, user["username"])
#     from kiara.web.cache_refresh import refresh_datasource_cache
#     safe_refresh(refresh_datasource_cache, ds_id)
#     return {"message": "Gruppen zugewiesen"}

# ---------------------------------------------------------------------------
# User assignment
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: datasource.users (Sprint 04) ---
# @router.get("/api/admin/datasources/{ds_id}/users")
# async def get_datasource_users(ds_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     """Get users directly assigned to a datasource."""
#     if not ds_repo.exists(ds_id):
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     users = perm_repo.get_datasource_users(ds_id)
#     return {"users": users}

# --- Migriert zu Admin-Registry: datasource.assign_users (Sprint 05) ---
# @router.post("/api/admin/datasources/{ds_id}/users")
# async def set_datasource_users(ds_id: int, data: UserAssignRequest, user: dict = Depends(get_admin_user)) -> dict:
#     """Set users for a datasource (replace all)."""
#     if not ds_repo.exists(ds_id):
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     perm_repo.set_datasource_users(ds_id, data.user_ids)
#     logger.info("Datasource %s users set to %s by %s", ds_id, data.user_ids, user["username"])
#     from kiara.web.cache_refresh import refresh_datasource_cache
#     safe_refresh(refresh_datasource_cache, ds_id)
#     return {"message": "Benutzer zugewiesen"}

# ---------------------------------------------------------------------------
# Connector Types Endpoint
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: datasource.connector_types (Sprint 04) ---
# @router.get("/api/admin/connector-types")
# async def list_connector_types(user: dict = Depends(get_admin_user)) -> dict:
#     """Gibt alle registrierten Konnektor-Typen mit Schema zurueck."""
#     ensure_loaded()
#     types = []
#     for name, cls in sorted(all_connectors().items(), key=lambda x: x[1].label):
#         types.append({
#             "name": cls.name, "label": cls.label,
#             "badge_class": cls.badge_class, "schema": cls.config_schema(),
#             "default_config": cls.default_config(),
#         })
#     return {"data": types}

# ---------------------------------------------------------------------------
# Sub-Router einbetten (Browse, Ingestion, Feature-Checks)
# Code lebt in eigenen Dateien, Endpoints laufen ueber diesen Router.
# ---------------------------------------------------------------------------

from kiara.web.api_datasource_browse import router as _browse_router
from kiara.web.api_datasource_features import router as _features_router
from kiara.web.api_datasource_ingest import router as _ingest_router

router.include_router(_browse_router)
router.include_router(_ingest_router)
router.include_router(_features_router)

# Re-Exports fuer Rueckwaertskompatibilitaet (Tests + Imports)
from kiara.web.api_datasource_features import (  # noqa: F401
    ddu_available,
    list_embedding_models,
    vlm_available,
)
from kiara.web.api_datasource_ingest import IngestRequest, _prepare_ds_config  # noqa: F401
