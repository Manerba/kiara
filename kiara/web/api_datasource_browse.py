"""KIara Admin - Datasource File Browser.

Browse-Endpunkte fuer Datenquellen-Verzeichnisse und SMB-Shares.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from kiara.config import load_config
from kiara.connectors.browse import (
    ConnectorBrowseError,
    list_local_directory,
    normalize_browse_path,
    parent_path,
)
from kiara.connectors.registry import auto_mount_point, ensure_loaded
from kiara.connectors.registry import get as get_connector
from kiara.connectors.smb_connector import SMBConnector
from kiara.connectors.webdav_connector import WebDAVConnector
from kiara.repositories.datasource import DatasourceRepository
from kiara.repositories.permission import PermissionRepository
from kiara.shared.constants import PASSWORD_MASK
from kiara.shared.enums import ConnectorType
from kiara.web.api_datasource import (
    ShareBrowseRequest,
    WebDAVBrowseRequest,
    _decrypt_config,
    _effective_path,
    _get_connector_config,
)
from kiara.web.api_utils import ERR_DATASOURCE_NOT_FOUND
from kiara.web.auth import get_admin_user, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["datasource-browse"])

ds_repo = DatasourceRepository()
perm_repo = PermissionRepository()

_BROWSE_MAX_ENTRIES = 1000


def _row_is_active(row: dict) -> bool:
    try:
        value = row["is_active"]
    except (KeyError, IndexError):
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _require_active_datasource(row: dict) -> None:
    if not _row_is_active(row):
        raise HTTPException(status_code=409, detail="Datenquelle ist inaktiv")


def _stored_connector_config(ds_id: int, expected_type: str) -> dict:
    row = ds_repo.get_by_id(ds_id)
    if not row:
        raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
    _require_active_datasource(row)
    if row["type"] != expected_type:
        raise HTTPException(
            status_code=400, detail="Gespeicherte Datenquelle passt nicht zum Konnektor-Typ"
        )

    ensure_loaded()
    connector_cls = get_connector(expected_type)
    if not connector_cls:
        raise HTTPException(status_code=400, detail="Unbekannter Konnektor-Typ")
    return _decrypt_config(_get_connector_config(row), connector_cls)


def _resolve_masked_password(
    *,
    password: str,
    ds_id: int | None,
    expected_type: str,
    request_config: dict,
    identity_fields: list[str],
) -> str:
    if password != PASSWORD_MASK:
        return password
    if ds_id is None:
        raise HTTPException(status_code=400, detail="Passwort erforderlich")

    stored = _stored_connector_config(ds_id, expected_type)
    for field in identity_fields:
        if str(request_config.get(field) or "") != str(stored.get(field) or ""):
            raise HTTPException(
                status_code=400,
                detail="PASSWORD_MASK darf nur fuer unveraenderte Verbindungsdaten genutzt werden",
            )
    resolved = stored.get("password") or ""
    if not resolved:
        raise HTTPException(status_code=500, detail="Gespeichertes Passwort ist leer")
    return resolved


def _normalize_or_raise(path: str | None) -> str:
    try:
        return normalize_browse_path(path)
    except ConnectorBrowseError as exc:
        status_code = 403 if exc.reason == "forbidden" or exc.status_code == 403 else 400
        raise HTTPException(status_code=status_code, detail=exc.message) from exc


def _connector_error_response(exc: ConnectorBrowseError, normalized_path: str) -> dict:
    if exc.reason in {"bad_input", "forbidden", "unsupported"} or 400 <= exc.status_code < 500:
        status_code = 403 if exc.reason == "forbidden" or exc.status_code == 403 else 400
        raise HTTPException(status_code=status_code, detail=exc.message) from exc
    return {"path": normalized_path, "entries": [], "error": exc.message}


def _browse_http_status(status_code: int) -> int:
    if status_code == 403:
        return 403
    if status_code == 404:
        return 404
    if 400 <= status_code < 500:
        return 400
    return 502


@router.get("/api/datasources/{ds_id}/browse")
async def browse_datasource(ds_id: int, path: str = "", user=Depends(get_current_user)):
    """Browse a datasource directory."""
    ensure_loaded()
    from kiara.web.smb_utils import is_mounted, mount_smb
    from kiara.web.webdav_utils import mount_webdav

    row = ds_repo.get_by_id(ds_id)
    if not row:
        raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
    _require_active_datasource(row)

    # Berechtigungspruefung: User muss Zugriff auf diese Datasource haben (Admin-Bypass)
    if not user.get("is_admin"):
        user_ds_ids = set(user.get("datasource_ids", []))
        public_ds_ids = set(perm_repo.get_public_datasource_ids())
        if ds_id not in user_ds_ids and ds_id not in public_ds_ids:
            raise HTTPException(status_code=403, detail="Kein Zugriff auf diese Datenquelle")

    normalized = _normalize_or_raise(path)

    try:
        base_path = _effective_path(row)
    except ValueError as exc:
        logger.warning(
            "browse_datasource: _effective_path fuer ds%s fehlgeschlagen: %s", ds_id, exc
        )
        raise HTTPException(
            status_code=400, detail="Ungueltige Datenquellen-Konfiguration"
        ) from exc
    if not base_path:
        base_path = load_config()["datasource"]["path"]

    conn_config = _get_connector_config(row)
    connector_cls = get_connector(row["type"])

    # SMB: ensure mounted
    if row["type"] == ConnectorType.SMB and connector_cls:
        mount_point = auto_mount_point(row)
        if not is_mounted(mount_point):
            dec_config = _decrypt_config(conn_config, connector_cls)
            ok, msg = mount_smb(
                server=dec_config.get("server", ""),
                share=dec_config.get("share", ""),
                username=dec_config.get("username", ""),
                password=dec_config.get("password", ""),
                domain=dec_config.get("domain", ""),
                mount_point=mount_point,
            )
            if not ok:
                raise HTTPException(status_code=502, detail=f"SMB-Mount fehlgeschlagen: {msg}")
        base_path = _effective_path(row)

    # WebDAV: ensure mounted
    if row["type"] == ConnectorType.WEBDAV and connector_cls:
        mount_point = auto_mount_point(row)
        if not is_mounted(mount_point):
            dec_config = _decrypt_config(conn_config, connector_cls)
            _bp = dec_config.get("base_path") or ""
            url = (dec_config.get("url") or "").rstrip("/") + "/" + _bp.lstrip("/")
            ok, msg = mount_webdav(
                url=url,
                username=dec_config.get("username", ""),
                password=dec_config.get("password", ""),
                mount_point=mount_point,
            )
            if not ok:
                raise HTTPException(status_code=502, detail=f"WebDAV-Mount fehlgeschlagen: {msg}")
        base_path = mount_point
    try:
        result = list_local_directory(
            base_path,
            normalized,
            directories_only=False,
            allow_absolute=False,
            max_entries=_BROWSE_MAX_ENTRIES,
        )
    except ConnectorBrowseError as exc:
        raise HTTPException(
            status_code=_browse_http_status(exc.status_code), detail=exc.message
        ) from exc

    return {
        "path": normalized,
        "parent": parent_path(normalized),
        "entries": result["entries"],
        "truncated": bool(result.get("truncated")),
        "max_entries": _BROWSE_MAX_ENTRIES,
    }


@router.post("/api/admin/datasources/browse-share")
async def browse_share(data: ShareBrowseRequest, user=Depends(get_admin_user)):
    """Browse an SMB share for folder picker (via smbclient, no mount needed)."""
    if data.ds_id is not None and data.ds_id <= 0:
        raise HTTPException(status_code=400, detail="Ungueltige Datenquellen-ID")
    if not data.server or not data.share:
        raise HTTPException(status_code=400, detail="Server und Freigabe erforderlich")

    request_config = {
        "server": data.server,
        "share": data.share,
        "username": data.username,
        "domain": data.domain,
    }
    if data.ds_id is not None:
        _stored_connector_config(data.ds_id, ConnectorType.SMB)
    password = _resolve_masked_password(
        password=data.password,
        ds_id=data.ds_id,
        expected_type=ConnectorType.SMB,
        request_config=request_config,
        identity_fields=["server", "share", "username", "domain"],
    )

    connector_config = {
        "server": data.server,
        "share": data.share,
        "username": data.username,
        "domain": data.domain,
        "password": password,
    }
    normalized = _normalize_or_raise(data.path)
    try:
        result = SMBConnector.list_directory(
            connector_config,
            normalized,
            directories_only=True,
        )
    except ConnectorBrowseError as exc:
        return _connector_error_response(exc, normalized)

    return {
        "path": normalized,
        "parent": parent_path(normalized),
        "entries": result["entries"],
    }


@router.post("/api/admin/datasources/browse-webdav")
async def browse_webdav(data: WebDAVBrowseRequest, user=Depends(get_admin_user)):
    """Browse a WebDAV share for folder picker (via PROPFIND, no mount needed)."""
    if data.ds_id is not None and data.ds_id <= 0:
        raise HTTPException(status_code=400, detail="Ungueltige Datenquellen-ID")
    if not data.url:
        raise HTTPException(status_code=400, detail="WebDAV-URL erforderlich")

    request_config = {
        "url": data.url,
        "username": data.username,
    }
    if data.ds_id is not None:
        _stored_connector_config(data.ds_id, ConnectorType.WEBDAV)
    password = _resolve_masked_password(
        password=data.password,
        ds_id=data.ds_id,
        expected_type=ConnectorType.WEBDAV,
        request_config=request_config,
        identity_fields=["url", "username"],
    )

    connector_config = {
        "url": data.url,
        "username": data.username,
        "password": password,
    }
    normalized = _normalize_or_raise(data.path)
    try:
        result = WebDAVConnector.list_directory(
            connector_config,
            normalized,
            directories_only=True,
        )
    except ConnectorBrowseError as exc:
        return _connector_error_response(exc, normalized)

    return {
        "path": normalized,
        "parent": parent_path(normalized),
        "entries": result["entries"],
    }
