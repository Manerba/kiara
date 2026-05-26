"""
KIara - OnlyOffice API
Config, WOPI endpoints, Editor.
"""

import hashlib
import ipaddress
import logging
import os
import re
import secrets
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from kiara.config import DEFAULT_TIMEOUTS, load_config
from kiara.repositories.config import ConfigRepository
from kiara.repositories.datasource import DatasourceRepository
from kiara.repositories.permission import PermissionRepository
from kiara.repositories.wopi import WopiRepository
from kiara.shared.crypto import decrypt, is_encrypted
from kiara.shared.db import db_transaction
from kiara.web.auth import get_admin_user, get_chat_user
from kiara.web.config_changelog import log_config_change
from kiara.web.config_utils import get_masked_config, resolve_masked_field
from kiara.web.wopi_proof import WopiProofValidationError, WopiProofValidator

logger = logging.getLogger(__name__)

admin_router = APIRouter(tags=["onlyoffice-admin"])
editor_router = APIRouter(tags=["onlyoffice"])
wopi_router = APIRouter(tags=["wopi"])
callback_router = APIRouter(tags=["onlyoffice-callback"])
router = APIRouter(tags=["onlyoffice"])

WOPI_TOKEN_EXPIRY = 3600 * 8  # 8 hours
MAX_WOPI_FILE_SIZE = 100 * 1024 * 1024
JWT_ALGORITHM = "HS256"
_WOPI_LOCK_MAX_LEN = 1024
_WOPI_LOCK_SAFE_RE = re.compile(r"^[\x20-\x7e]{1,1024}$")

config_repo = ConfigRepository()
wopi_repo = WopiRepository()
ds_repo = DatasourceRepository()
perm_repo = PermissionRepository()
wopi_proof_validator = WopiProofValidator()


def _validate_wopi_lock_id(lock_id: str, header_name: str = "X-WOPI-Lock") -> str:
    """Validiert WOPI-Locks gegen Header- und Speicher-Missbrauch."""
    value = str(lock_id or "")
    if not value:
        raise HTTPException(status_code=400, detail=f"{header_name} erforderlich")
    if len(value) > _WOPI_LOCK_MAX_LEN or not _WOPI_LOCK_SAFE_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail=f"{header_name} ungueltig")
    return value


def _safe_wopi_lock_header(lock_id: str | None) -> str:
    """Gibt nur reflektierbare Lock-Header-Werte zurueck."""
    value = str(lock_id or "")
    if not value:
        return ""
    if len(value) > _WOPI_LOCK_MAX_LEN or not _WOPI_LOCK_SAFE_RE.fullmatch(value):
        logger.warning("Ungueltiger gespeicherter WOPI-Lock unterdrueckt")
        return ""
    return value


def _lock_wopi_file_for_update(conn, file_id: str) -> None:
    """Serialisiert WOPI-Dateimutationen pro file_id transaktional."""
    digest = hashlib.sha256(f"wopi-file:{file_id}".encode("utf-8")).digest()
    lock_key = int.from_bytes(digest[:8], "big", signed=True)
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))


def _load_wopi_user_with_acl(user_id: int) -> dict | None:
    """Lade aktuellen User-Stand inkl. datasource_ids fuer WOPI-ACL-Pruefung.

    Token speichert nur user_id+username — Berechtigungen koennen sich
    waehrend der 8h-Token-Lebensdauer aendern. Dieser Helper laedt den
    aktuellen Zustand pro Request, damit entzogene ACLs sofort wirken
    (Bug #1850).
    """
    from kiara.shared.db import db_query
    from kiara.web.config_utils import load_user_permissions

    with db_query() as db:
        row = db.execute(
            "SELECT id, username FROM users WHERE id = %s AND is_active = true",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        _, datasource_ids = load_user_permissions(db, user_id)
    return {
        "id": row["id"],
        "username": row["username"],
        "datasource_ids": datasource_ids,
    }


def _check_datasource_acl(full_path: str, user: dict) -> None:
    """Prueft dass full_path unter dem Pfad einer fuer den User zugelassenen
    Datasource liegt.

    Zugelassen = direkte Zuweisung (user["datasource_ids"]) ODER is_public=1.
    Bei Mismatch wird HTTPException 403 geworfen (Bug #876).
    """
    user_ds_ids = set(user.get("datasource_ids", []))
    public_ds_ids = set(perm_repo.get_public_datasource_ids())
    allowed_ids = user_ds_ids | public_ds_ids
    if not allowed_ids:
        raise HTTPException(status_code=403, detail="Kein Zugriff auf diese Datei")

    full_resolved = os.path.realpath(full_path)
    for ds in ds_repo.get_active():
        if ds["id"] not in allowed_ids:
            continue
        ds_path = ds.get("path") or ""
        if not ds_path:
            continue
        ds_resolved = os.path.realpath(ds_path)
        if full_resolved == ds_resolved or full_resolved.startswith(ds_resolved + os.sep):
            return  # Zugriff erlaubt
    raise HTTPException(status_code=403, detail="Kein Zugriff auf diese Datei")


class OnlyOfficeConfigRequest(BaseModel):
    """Request zum Speichern der OnlyOffice-Document-Server-Konfiguration."""

    document_server_url: str
    jwt_secret: str
    wopi_external_base_url: str = ""
    proof_validation_enabled: bool = True


def _normalize_document_server_url(raw_url: str) -> str:
    """Normalize the configured OnlyOffice Document Server URL."""
    url = str(raw_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError("OnlyOffice Document Server URL muss http(s) mit Host sein")
    if parsed.username or parsed.password:
        raise ValueError("OnlyOffice Document Server URL darf keine Credentials enthalten")
    if parsed.query or parsed.fragment:
        raise ValueError("OnlyOffice Document Server URL darf keine Query/Fragment enthalten")
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def _normalize_external_base_url(raw_url: str | None) -> str:
    """Normalize optional external KIara base URL used for WOPI signatures."""
    url = str(raw_url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError("Externe WOPI-Basis-URL muss http(s) mit Host sein")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Externe WOPI-Basis-URL darf keine Credentials/Query/Fragment enthalten")
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def _ip_allowed_for_document_server(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
        return False
    if ip.is_private:
        return True
    return ip.is_global


def _assert_document_server_request_allowed(document_server_url: str) -> None:
    """Reject URL shapes and resolved targets that are unsafe for server-side tests."""
    normalized = _normalize_document_server_url(document_server_url)
    host = urlparse(normalized).hostname or ""
    if host.lower() in {"localhost", "localhost.localdomain"}:
        raise ValueError("Loopback-Hosts sind fuer den Verbindungstest nicht erlaubt")

    try:
        literal_ip = ipaddress.ip_address(host.strip("[]"))
        if not _ip_allowed_for_document_server(literal_ip):
            raise ValueError("OnlyOffice-Ziel-IP ist nicht erlaubt")
        return
    except ValueError as exc:
        if "OnlyOffice-Ziel-IP" in str(exc):
            raise

    try:
        addrinfos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("OnlyOffice-Host konnte nicht aufgeloest werden") from exc
    if not addrinfos:
        raise ValueError("OnlyOffice-Host konnte nicht aufgeloest werden")
    for *_, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not _ip_allowed_for_document_server(ip):
            raise ValueError(f"OnlyOffice-Ziel-IP ist nicht erlaubt: {ip}")


def _get_jwt_secret(oo_config: dict | None) -> str:
    """Return the decrypted OnlyOffice JWT secret or fail closed."""
    if not oo_config:
        raise ValueError("OnlyOffice nicht konfiguriert")
    secret = str(oo_config.get("jwt_secret") or "")
    if is_encrypted(secret):
        secret = decrypt(secret)
    secret = secret.strip()
    if not secret:
        raise ValueError("OnlyOffice JWT-Secret fehlt")
    return secret


def _sign_jwt(payload: dict, secret: str) -> str:
    return pyjwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def _verify_jwt(token: str, secret: str) -> dict:
    try:
        data = pyjwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except pyjwt.PyJWTError as exc:
        raise PermissionError("OnlyOffice JWT ungueltig") from exc
    if not isinstance(data, dict):
        raise PermissionError("OnlyOffice JWT-Payload ungueltig")
    return data


def _extract_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _verified_callback_payload(request: Request, data: dict, secret: str) -> dict:
    token = str(data.get("token") or "").strip()
    if not token:
        token = _extract_bearer_token(request)
    if not token:
        raise PermissionError("OnlyOffice Callback ohne JWT")
    verified = _verify_jwt(token, secret)
    payload = verified.get("payload")
    if isinstance(payload, dict):
        return payload
    return verified


def _request_external_base_url(request: Request, oo_config: dict | None = None) -> str:
    configured_base = _normalize_external_base_url((oo_config or {}).get("wopi_external_base_url"))
    if configured_base:
        return configured_base
    proto = request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}".rstrip("/")


def _request_external_url(request: Request, oo_config: dict | None = None) -> str:
    base = _request_external_base_url(request, oo_config)
    query = f"?{request.url.query}" if request.url.query else ""
    return f"{base}{request.url.path}{query}"


def _proof_validation_enabled(oo_config: dict | None) -> bool:
    if not oo_config:
        return True
    return bool(oo_config.get("proof_validation_enabled", True))


# ─── Admin Config ───────────────────────────────────────────────


@admin_router.get("/api/admin/onlyoffice/config")
async def get_onlyoffice_config(user=Depends(get_admin_user)):
    """Get OnlyOffice configuration."""
    config = get_masked_config("onlyoffice_config", mask_fields=["jwt_secret"])
    return {"config": config}


@admin_router.post("/api/admin/onlyoffice/config")
async def save_onlyoffice_config(data: OnlyOfficeConfigRequest, user=Depends(get_admin_user)):
    """Save OnlyOffice configuration."""
    try:
        document_server_url = _normalize_document_server_url(data.document_server_url)
        _assert_document_server_request_allowed(document_server_url)
        wopi_external_base_url = _normalize_external_base_url(data.wopi_external_base_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not data.jwt_secret:
        raise HTTPException(status_code=400, detail="JWT-Secret erforderlich")

    from kiara.shared.db import db_query

    with db_query() as db:
        jwt_secret = resolve_masked_field(db, "onlyoffice_config", "jwt_secret", data.jwt_secret)

    config_repo.set(
        "onlyoffice_config",
        {
            "document_server_url": document_server_url,
            "jwt_secret": jwt_secret,
            "wopi_external_base_url": wopi_external_base_url,
            "proof_validation_enabled": bool(data.proof_validation_enabled),
        },
    )
    log_config_change(user, area="onlyoffice_config", action="update", source="gui")

    logger.info("OnlyOffice config saved by %s", user["username"])
    return {"message": "OnlyOffice-Konfiguration gespeichert"}


@admin_router.delete("/api/admin/onlyoffice/config")
async def delete_onlyoffice_config(user=Depends(get_admin_user)):
    """Delete OnlyOffice configuration."""
    config_repo.delete("onlyoffice_config")
    log_config_change(user, area="onlyoffice_config", action="delete", source="gui")
    return {"message": "OnlyOffice-Konfiguration geloescht"}


@admin_router.post("/api/admin/onlyoffice/test")
async def test_onlyoffice(user=Depends(get_admin_user)):
    """Test OnlyOffice Document Server connectivity."""
    row = config_repo.get("onlyoffice_config")
    if not row:
        raise HTTPException(status_code=404, detail="OnlyOffice nicht konfiguriert")

    import httpx

    try:
        url = _normalize_document_server_url(row["document_server_url"])
        _assert_document_server_request_allowed(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUTS["http_api"], follow_redirects=False
        ) as client:
            resp = await client.get(f"{url}/healthcheck")
            if resp.status_code == 200:
                return {"success": True, "message": "Document Server erreichbar"}
            raise HTTPException(status_code=502, detail=f"HTTP {resp.status_code}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@admin_router.post("/api/admin/onlyoffice/refresh-discovery")
async def refresh_onlyoffice_discovery(user=Depends(get_admin_user)):
    """Refresh WOPI discovery proof keys from the configured Document Server."""
    row = config_repo.get("onlyoffice_config")
    if not row:
        raise HTTPException(status_code=404, detail="OnlyOffice nicht konfiguriert")
    try:
        url = _normalize_document_server_url(row["document_server_url"])
        _assert_document_server_request_allowed(url)
        keys = await wopi_proof_validator.refresh(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except WopiProofValidationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    logger.info("OnlyOffice WOPI discovery refreshed by %s", user["username"])
    return {
        "success": True,
        "message": "WOPI-Discovery aktualisiert",
        "current_key": True,
        "old_key": keys.old_key is not None,
    }


# ─── Editor / Token ─────────────────────────────────────────────


def _file_id_for_path(full_path: str) -> str:
    """Erzeugt die WOPI-Datei-ID aus dem kanonischen Dateipfad."""
    return hashlib.sha256(os.path.realpath(full_path).encode()).hexdigest()[:32]


@editor_router.get("/api/onlyoffice/open")
async def open_editor(file_path: str, request: Request, user=Depends(get_chat_user)):
    """Get editor configuration for a file.

    Prueft neben dem Path-Traversal-Schutz auch die Datasource-ACL:
    die angeforderte Datei muss unter dem Pfad einer Datasource liegen,
    auf die der User Zugriff hat (direkte Zuweisung oder public) —
    Bug #876.
    """
    config = load_config()
    base_path = config["datasource"]["path"]

    # Validate path
    full_path = os.path.realpath(os.path.join(base_path, file_path))
    if not Path(full_path).is_relative_to(Path(base_path).resolve()):
        raise HTTPException(status_code=403, detail="Zugriff verweigert")
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="Datei nicht gefunden")

    # ACL: Datei muss in einer vom User zugelassenen Datasource liegen
    _check_datasource_acl(full_path, user)

    # Get OnlyOffice config
    oo_config = config_repo.get("onlyoffice_config")
    if not oo_config:
        raise HTTPException(status_code=503, detail="OnlyOffice nicht konfiguriert")
    try:
        jwt_secret = _get_jwt_secret(oo_config)
        doc_server_url = _normalize_document_server_url(oo_config["document_server_url"])
        external_base_url = _request_external_base_url(request, oo_config)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Create WOPI token
    file_id = _file_id_for_path(full_path)
    token = secrets.token_hex(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=WOPI_TOKEN_EXPIRY)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    wopi_repo.create_token(token, file_id, user["id"], expires_at, file_path=full_path)

    ext = os.path.splitext(file_path)[1].lower()

    doc_type_map = {
        ".docx": "word",
        ".doc": "word",
        ".odt": "word",
        ".txt": "word",
        ".rtf": "word",
        ".xlsx": "cell",
        ".xls": "cell",
        ".ods": "cell",
        ".csv": "cell",
        ".pptx": "slide",
        ".ppt": "slide",
        ".odp": "slide",
    }
    doc_type = doc_type_map.get(ext, "word")

    file_type_map = {
        ".docx": "docx",
        ".doc": "doc",
        ".odt": "odt",
        ".txt": "txt",
        ".rtf": "rtf",
        ".xlsx": "xlsx",
        ".xls": "xls",
        ".ods": "ods",
        ".csv": "csv",
        ".pptx": "pptx",
        ".ppt": "ppt",
        ".odp": "odp",
    }
    file_type = file_type_map.get(ext, ext.lstrip("."))

    doc_url = f"{external_base_url}/api/wopi/files/{file_id}/contents?access_token={token}"
    callback_url = (
        f"{external_base_url}/api/onlyoffice/callback?file_id={file_id}&access_token={token}"
    )
    editor_config = {
        "document": {
            "fileType": file_type,
            "key": f"{file_id}_{int(datetime.now(timezone.utc).timestamp())}",
            "title": os.path.basename(file_path),
            "url": doc_url,
            "permissions": {
                "download": True,
                "edit": True,
                "print": True,
                "review": True,
            },
        },
        "documentType": doc_type,
        "editorConfig": {
            "callbackUrl": callback_url,
            "lang": "de",
            "mode": "edit",
            "user": {
                "id": str(user["id"]),
                "name": str(user.get("username") or "Benutzer"),
            },
        },
        "type": "desktop",
        "width": "100%",
        "height": "100%",
    }
    editor_config["token"] = _sign_jwt(editor_config, jwt_secret)

    return {
        "document_server_url": doc_server_url,
        "config": editor_config,
        "file_id": file_id,
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_type": file_type,
        "document_type": doc_type,
        "token": token,
        "token_ttl": WOPI_TOKEN_EXPIRY,
    }


# ─── WOPI Protocol ──────────────────────────────────────────────


def _validate_wopi_token(token: str, file_id: str) -> dict | None:
    """Validate a WOPI access token."""
    if not token:
        return None
    from kiara.shared.db import db_query

    with db_query() as db:
        row = db.execute(
            """SELECT wt.*, u.username FROM wopi_tokens wt
               JOIN users u ON wt.user_id = u.id
               WHERE wt.token = %s AND wt.file_id = %s AND wt.expires_at::timestamp > NOW()""",
            (token, file_id),
        ).fetchone()
    if not row:
        return None
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "file_id": row["file_id"],
        "file_path": row["file_path"],
    }


def _get_file_path_from_id(file_id: str) -> str | None:
    """Resolve file_id to path via DB-Lookup (gespeichert bei Token-Erstellung)."""
    from kiara.shared.db import db_query

    with db_query() as db:
        row = db.execute(
            "SELECT file_path FROM wopi_tokens WHERE file_id = %s AND file_path IS NOT NULL LIMIT 1",
            (file_id,),
        ).fetchone()
    return row["file_path"] if row else None


def _get_file_path_for_wopi_user(wopi_user: dict) -> str | None:
    """Liefert den an den validierten Token gebundenen Dateipfad."""
    return wopi_user.get("file_path") or _get_file_path_from_id(wopi_user["file_id"])


async def _validate_wopi_proof(request: Request, access_token: str) -> None:
    """Validate WOPI proof headers for a token-authenticated WOPI request."""
    oo_config = config_repo.get("onlyoffice_config")
    if not oo_config:
        raise HTTPException(status_code=500, detail="WOPI proof validation unavailable")
    if not _proof_validation_enabled(oo_config):
        return
    try:
        document_server_url = _normalize_document_server_url(oo_config["document_server_url"])
    except (KeyError, ValueError) as exc:
        logger.warning("WOPI proof validation: OnlyOffice-Konfiguration ungueltig: %s", exc)
        raise HTTPException(status_code=500, detail="WOPI proof validation unavailable")

    proof = request.headers.get("X-WOPI-Proof", "")
    proof_old = request.headers.get("X-WOPI-ProofOld", "")
    timestamp = request.headers.get("X-WOPI-TimeStamp", "")
    request_url = _request_external_url(request, oo_config)
    try:
        valid = await wopi_proof_validator.validate(
            document_server_url=document_server_url,
            access_token=access_token,
            request_url=request_url,
            timestamp=timestamp,
            proof=proof,
            proof_old=proof_old,
        )
    except WopiProofValidationError as exc:
        logger.warning("WOPI proof validation nicht moeglich: %s", exc)
        raise HTTPException(status_code=500, detail="Invalid WOPI proof")
    if not valid:
        logger.warning("WOPI proof validation fehlgeschlagen: url=%s", request_url)
        raise HTTPException(status_code=500, detail="Invalid WOPI proof")


@wopi_router.get("/api/wopi/files/{file_id}")
async def wopi_check_file_info(file_id: str, request: Request, access_token: str = ""):
    """WOPI CheckFileInfo."""
    wopi_user = _validate_wopi_token(access_token, file_id)
    if not wopi_user:
        raise HTTPException(status_code=401, detail="Invalid token")
    await _validate_wopi_proof(request, access_token)

    full_path = _get_file_path_for_wopi_user(wopi_user)
    if not full_path or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    # ACL pro Request: entzogene Datasource-Berechtigungen wirken sofort,
    # statt erst nach Token-Ablauf (Bug #1850).
    acl_user = _load_wopi_user_with_acl(wopi_user["user_id"])
    if not acl_user:
        raise HTTPException(status_code=401, detail="User inaktiv oder geloescht")
    _check_datasource_acl(full_path, acl_user)

    stat = os.stat(full_path)
    return {
        "BaseFileName": os.path.basename(full_path),
        "Size": stat.st_size,
        "OwnerId": str(wopi_user["user_id"]),
        "UserId": str(wopi_user["user_id"]),
        "UserFriendlyName": wopi_user["username"],
        "Version": str(int(stat.st_mtime)),
        "UserCanWrite": True,
        "UserCanNotWriteRelative": True,
        "SupportsLocks": True,
        "SupportsUpdate": True,
    }


@wopi_router.get("/api/wopi/files/{file_id}/contents")
async def wopi_get_file(file_id: str, request: Request, access_token: str = ""):
    """WOPI GetFile."""
    wopi_user = _validate_wopi_token(access_token, file_id)
    if not wopi_user:
        raise HTTPException(status_code=401, detail="Invalid token")
    await _validate_wopi_proof(request, access_token)

    full_path = _get_file_path_for_wopi_user(wopi_user)
    if not full_path or not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    # ACL pro Request: entzogene Datasource-Berechtigungen wirken sofort,
    # statt erst nach Token-Ablauf (Bug #1850).
    acl_user = _load_wopi_user_with_acl(wopi_user["user_id"])
    if not acl_user:
        raise HTTPException(status_code=401, detail="User inaktiv oder geloescht")
    _check_datasource_acl(full_path, acl_user)

    try:
        chunks: list[bytes] = []
        total = 0
        with open(full_path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_WOPI_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(f"Datei zu gross (max {MAX_WOPI_FILE_SIZE // (1024 * 1024)} MB)"),
                    )
                chunks.append(chunk)
        content = b"".join(chunks)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return Response(content=content, media_type="application/octet-stream")


@wopi_router.post("/api/wopi/files/{file_id}/contents")
async def wopi_put_file(file_id: str, request: Request, access_token: str = ""):
    """WOPI PutFile."""
    wopi_user = _validate_wopi_token(access_token, file_id)
    if not wopi_user:
        raise HTTPException(status_code=401, detail="Invalid token")
    await _validate_wopi_proof(request, access_token)

    full_path = _get_file_path_for_wopi_user(wopi_user)
    if not full_path:
        raise HTTPException(status_code=404, detail="File not found")

    # ACL pro Request: entzogene Datasource-Berechtigungen wirken sofort,
    # statt erst nach Token-Ablauf (Bug #1850).
    acl_user = _load_wopi_user_with_acl(wopi_user["user_id"])
    if not acl_user:
        raise HTTPException(status_code=401, detail="User inaktiv oder geloescht")
    _check_datasource_acl(full_path, acl_user)

    # X-WOPI-Override darf falls gesetzt nur "PUT" sein (Defense-in-Depth, WOPI-Spec)
    override = request.headers.get("X-WOPI-Override", "").upper()
    if override and override != "PUT":
        raise HTTPException(status_code=400, detail="Invalid X-WOPI-Override")

    try:
        content_length = int(request.headers.get("content-length", 0))
    except (ValueError, TypeError):
        content_length = 0
    if content_length > MAX_WOPI_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail=f"Datei zu gross (max {MAX_WOPI_FILE_SIZE // (1024 * 1024)} MB)"
        )

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_WOPI_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Datei zu gross (max {MAX_WOPI_FILE_SIZE // (1024 * 1024)} MB)",
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    # Lock-Pruefung: WOPI-Spezifikation verlangt Lock-Validierung bei PutFile.
    # Der Advisory-Lock umfasst Lock-Read, File-State-Check und atomaren Replace,
    # damit parallele PutFile/Callback-Schreiber keinen TOCTOU-Spalt haben.
    request_lock = request.headers.get("X-WOPI-Lock", "")
    if request_lock:
        request_lock = _validate_wopi_lock_id(request_lock)
    with db_transaction() as conn:
        _lock_wopi_file_for_update(conn, file_id)
        existing_lock = wopi_repo.get_lock(file_id, for_update=True, conn=conn)
        if existing_lock and existing_lock.get("expires_at"):
            try:
                exp = existing_lock["expires_at"]
                if isinstance(exp, str):
                    exp = datetime.strptime(exp, "%Y-%m-%d %H:%M:%S")
                if not exp.tzinfo:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp < datetime.now(timezone.utc):
                    existing_lock = None
            except (ValueError, TypeError):
                pass

        # WOPI-Spec: Bei nicht-leerer Datei oder existierendem Lock muss ein
        # nicht-leerer X-WOPI-Lock gesendet werden, der mit dem aktuellen Lock matcht.
        # Initial-Create (file_size == 0, kein Lock) darf ohne Lock-Header passieren.
        file_size = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
        if not request_lock:
            if file_size > 0 or existing_lock:
                resp = JSONResponse(status_code=409, content={"status": "lock required"})
                resp.headers["X-WOPI-Lock"] = _safe_wopi_lock_header(
                    existing_lock["lock_id"] if existing_lock else ""
                )
                return resp
        elif not existing_lock or existing_lock["lock_id"] != request_lock:
            resp = JSONResponse(status_code=409, content={"status": "lock conflict"})
            resp.headers["X-WOPI-Lock"] = _safe_wopi_lock_header(
                existing_lock["lock_id"] if existing_lock else ""
            )
            return resp

        try:
            _write_file_atomic(full_path, content, allow_create=True)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="File not found")
        except PermissionError:
            raise HTTPException(status_code=403, detail="Permission denied")

    logger.info("WOPI PutFile: %s by user %s", full_path, wopi_user["username"])
    return JSONResponse(content={"status": "ok"})


@wopi_router.post("/api/wopi/files/{file_id}")
async def wopi_post_action(file_id: str, request: Request, access_token: str = ""):
    """WOPI Lock/Unlock/RefreshLock."""
    wopi_user = _validate_wopi_token(access_token, file_id)
    if not wopi_user:
        raise HTTPException(status_code=401, detail="Invalid token")
    await _validate_wopi_proof(request, access_token)

    # ACL pro Request: entzogene Datasource-Berechtigungen wirken sofort,
    # statt erst nach Token-Ablauf (Bug #1850).
    full_path = _get_file_path_for_wopi_user(wopi_user)
    if not full_path:
        raise HTTPException(status_code=404, detail="File not found")
    acl_user = _load_wopi_user_with_acl(wopi_user["user_id"])
    if not acl_user:
        raise HTTPException(status_code=401, detail="User inaktiv oder geloescht")
    _check_datasource_acl(full_path, acl_user)

    override = request.headers.get("X-WOPI-Override", "").upper()
    lock_id = request.headers.get("X-WOPI-Lock", "")
    old_lock = request.headers.get("X-WOPI-OldLock", "")
    if lock_id:
        lock_id = _validate_wopi_lock_id(lock_id)
    if old_lock:
        old_lock = _validate_wopi_lock_id(old_lock, "X-WOPI-OldLock")

    if override == "LOCK":
        return _handle_lock(file_id, lock_id, old_lock, wopi_user)
    elif override == "UNLOCK":
        return _handle_unlock(file_id, lock_id)
    elif override == "REFRESH_LOCK":
        return _handle_refresh_lock(file_id, lock_id)
    elif override == "GET_LOCK":
        return _handle_get_lock(file_id)

    return JSONResponse(content={"status": "ok"})


def _handle_lock(file_id: str, lock_id: str, old_lock: str, wopi_user: dict):
    """Handle LOCK and UNLOCK_AND_RELOCK."""
    # WOPI-Spec: X-WOPI-Lock muss nicht-leer sein, sonst wird ein "leerer Lock"
    # gespeichert, der den PutFile-Lock-Check umgehen koennte.
    lock_id = _validate_wopi_lock_id(lock_id)
    with db_transaction() as conn:
        existing = wopi_repo.get_lock(file_id, for_update=True, conn=conn)
        # Filter out expired locks
        if existing and existing.get("expires_at"):
            try:
                exp = existing["expires_at"]
                if isinstance(exp, str):
                    exp = datetime.strptime(exp, "%Y-%m-%d %H:%M:%S")
                if not exp.tzinfo:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp < datetime.now(timezone.utc):
                    existing = None
            except (ValueError, TypeError):
                pass

        if existing:
            if existing["lock_id"] == lock_id:
                # Refresh
                expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                wopi_repo.set_lock(file_id, lock_id, existing["user_id"], expires, conn=conn)
                return JSONResponse(content={"status": "ok"})
            if old_lock and existing["lock_id"] == old_lock:
                # Unlock and relock
                expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                wopi_repo.set_lock(file_id, lock_id, wopi_user["user_id"], expires, conn=conn)
                return JSONResponse(content={"status": "ok"})
            # Lock mismatch
            resp = JSONResponse(status_code=409, content={"status": "lock conflict"})
            resp.headers["X-WOPI-Lock"] = _safe_wopi_lock_header(existing["lock_id"])
            return resp

        # New lock
        expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        wopi_repo.set_lock(file_id, lock_id, wopi_user["user_id"], expires, conn=conn)
        return JSONResponse(content={"status": "ok"})


def _handle_unlock(file_id: str, lock_id: str):
    lock_id = _validate_wopi_lock_id(lock_id)
    with db_transaction() as conn:
        existing = wopi_repo.get_lock(file_id, for_update=True, conn=conn)
        if existing and existing["lock_id"] != lock_id:
            resp = JSONResponse(status_code=409, content={"status": "lock conflict"})
            resp.headers["X-WOPI-Lock"] = _safe_wopi_lock_header(existing["lock_id"])
            return resp
        wopi_repo.release_lock(file_id, conn=conn)
        return JSONResponse(content={"status": "ok"})


def _handle_refresh_lock(file_id: str, lock_id: str):
    lock_id = _validate_wopi_lock_id(lock_id)
    with db_transaction() as conn:
        existing = wopi_repo.get_lock(file_id, for_update=True, conn=conn)
        if not existing or existing["lock_id"] != lock_id:
            resp = JSONResponse(status_code=409, content={"status": "lock conflict"})
            if existing:
                resp.headers["X-WOPI-Lock"] = _safe_wopi_lock_header(existing["lock_id"])
            return resp
        expires = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
        wopi_repo.set_lock(file_id, lock_id, existing["user_id"], expires, conn=conn)
        return JSONResponse(content={"status": "ok"})


def _handle_get_lock(file_id: str):
    existing = wopi_repo.get_lock(file_id)
    # Filter expired
    if existing and existing.get("expires_at"):
        try:
            exp = existing["expires_at"]
            if isinstance(exp, str):
                exp = datetime.strptime(exp, "%Y-%m-%d %H:%M:%S")
            if not exp.tzinfo:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp < datetime.now(timezone.utc):
                existing = None
        except (ValueError, TypeError):
            pass
    resp = JSONResponse(content={"status": "ok"})
    resp.headers["X-WOPI-Lock"] = _safe_wopi_lock_header(existing["lock_id"] if existing else "")
    return resp


# ─── OnlyOffice Callback ────────────────────────────────────────


def _assert_onlyoffice_download_url(download_url: str, oo_config: dict) -> None:
    """Erlaubt Callback-Downloads nur vom konfigurierten Document-Server."""
    configured_url = _normalize_document_server_url(str(oo_config.get("document_server_url") or ""))
    expected = urlparse(configured_url)
    candidate = urlparse(download_url)
    if not expected.scheme or not expected.netloc:
        raise ValueError("OnlyOffice Document Server URL ist ungueltig")
    if candidate.scheme not in {"http", "https"} or not candidate.netloc:
        raise ValueError("OnlyOffice Callback-Download-URL ist ungueltig")
    if (candidate.scheme, candidate.netloc) != (expected.scheme, expected.netloc):
        raise ValueError("OnlyOffice Callback-Download-URL passt nicht zur Konfiguration")


async def _download_onlyoffice_callback_content(download_url: str, oo_config: dict) -> bytes:
    """Laedt den von OnlyOffice bereitgestellten Speicherstand."""
    _assert_onlyoffice_download_url(download_url, oo_config)
    import httpx

    async with httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUTS["http_api"], follow_redirects=False
    ) as client:
        resp = await client.get(download_url)
    resp.raise_for_status()

    try:
        content_length = int(resp.headers.get("content-length") or 0)
    except (ValueError, TypeError):
        content_length = 0
    if content_length > MAX_WOPI_FILE_SIZE or len(resp.content) > MAX_WOPI_FILE_SIZE:
        raise ValueError(
            f"OnlyOffice Callback-Datei zu gross (max {MAX_WOPI_FILE_SIZE // (1024 * 1024)} MB)"
        )
    return resp.content


def _write_file_atomic(full_path: str, content: bytes, allow_create: bool = False) -> None:
    """Schreibt eine Datei atomar im Zielverzeichnis."""
    if not allow_create and not os.path.isfile(full_path):
        raise FileNotFoundError(full_path)

    tmp_path = f"{full_path}.onlyoffice-{secrets.token_hex(8)}.tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, full_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _save_onlyoffice_callback_document(
    file_id: str, access_token: str, download_url: str
) -> str:
    """Validiert Callback-Kontext und persistiert den OnlyOffice-Speicherstand."""
    wopi_user = _validate_wopi_token(access_token, file_id)
    if not wopi_user:
        raise PermissionError("Invalid token")
    full_path = _get_file_path_for_wopi_user(wopi_user)
    if not full_path:
        raise FileNotFoundError(file_id)

    # ACL pro Request: entzogene Datasource-Berechtigungen wirken sofort,
    # statt erst nach Token-Ablauf (Bug #1850).
    acl_user = _load_wopi_user_with_acl(wopi_user["user_id"])
    if not acl_user:
        raise PermissionError("User inaktiv oder geloescht")
    _check_datasource_acl(full_path, acl_user)

    oo_config = config_repo.get("onlyoffice_config")
    if not oo_config:
        raise RuntimeError("OnlyOffice nicht konfiguriert")

    content = await _download_onlyoffice_callback_content(download_url, oo_config)
    with db_transaction() as conn:
        _lock_wopi_file_for_update(conn, file_id)
        _write_file_atomic(full_path, content)
    return full_path


@callback_router.post("/api/onlyoffice/callback")
async def onlyoffice_callback(request: Request, file_id: str = "", access_token: str = ""):
    """OnlyOffice callback for document save events."""
    oo_config = config_repo.get("onlyoffice_config")
    try:
        jwt_secret = _get_jwt_secret(oo_config)
    except ValueError as exc:
        logger.warning("OnlyOffice callback ohne gueltige Konfiguration: %s", exc)
        return JSONResponse(content={"error": 1})

    try:
        data = await request.json()
    except Exception as exc:
        logger.warning("OnlyOffice callback JSON ungueltig: %s", exc)
        return JSONResponse(content={"error": 1})
    if not isinstance(data, dict):
        logger.warning("OnlyOffice callback JSON ist kein Objekt")
        return JSONResponse(content={"error": 1})

    try:
        data = _verified_callback_payload(request, data, jwt_secret)
    except PermissionError as exc:
        logger.warning("OnlyOffice callback JWT ungueltig: %s", exc)
        return JSONResponse(content={"error": 1}, status_code=401)

    status = data.get("status", 0)
    # Status 2 = document ready for saving, 6 = document being saved
    if status in (2, 6):
        url = data.get("url")
        key = data.get("key", "")
        if not file_id or not access_token or not url:
            logger.warning("OnlyOffice callback unvollstaendig: status=%s, key=%s", status, key)
            return JSONResponse(content={"error": 1})
        try:
            full_path = await _save_onlyoffice_callback_document(file_id, access_token, url)
        except Exception as exc:
            logger.warning(
                "OnlyOffice callback konnte Datei nicht speichern: status=%s, key=%s, error=%s",
                status,
                key,
                exc,
            )
            return JSONResponse(content={"error": 1})
        logger.info(
            "OnlyOffice callback gespeichert: status=%s, key=%s, file=%s", status, key, full_path
        )

    return JSONResponse(content={"error": 0})


router.include_router(admin_router)
router.include_router(editor_router)
router.include_router(wopi_router)
router.include_router(callback_router)
