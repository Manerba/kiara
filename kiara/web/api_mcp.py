"""MCP API-Key-Verwaltung — Admin-only Endpunkte."""

import asyncio
import hashlib
import secrets

from fastapi import APIRouter
from pydantic import BaseModel

from kiara.repositories.api_key import ApiKeyRepository
from kiara.repositories.user import UserRepository

router = APIRouter(prefix="/api/admin/mcp", tags=["mcp"])

api_key_repo = ApiKeyRepository()
user_repo = UserRepository()


# ---------------------------------------------------------------------------
# Shared Helper — wird von toggle + status genutzt
# ---------------------------------------------------------------------------

_PRIVILEGED_ACTIONS = {"enable", "disable", "start", "stop", "restart"}


async def _systemctl_exec(*args: str) -> tuple[int, str, str]:
    """Fuehrt systemctl-Befehl async aus. Gibt (returncode, stdout, stderr) zurueck.

    Privilegierte Aktionen (enable/disable/start/stop/restart) werden ueber
    sudo ausgefuehrt — eine passende NOPASSWD-Regel in /etc/sudoers.d/kiara-mcp
    wird vorausgesetzt. Reine Abfragen (is-active, show) laufen ohne sudo.
    """
    use_sudo = args and args[0] in _PRIVILEGED_ACTIONS
    cmd = ["sudo", "systemctl", *args] if use_sudo else ["systemctl", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode,
        stdout.decode(errors="replace").strip(),
        stderr.decode(errors="replace").strip(),
    )


# ---------------------------------------------------------------------------
# MCP-Server Toggle (Phase 2)
# ---------------------------------------------------------------------------


class MCPToggleRequest(BaseModel):
    """Request zum Ein-/Ausschalten des MCP-Servers."""

    enabled: bool


# --- Migriert zu Admin-Registry: mcp.toggle (Sprint 08) ---
# @router.post("/toggle")
# async def toggle_mcp(req: MCPToggleRequest, user=Depends(get_admin_user)):
#     """Schaltet den MCP-Server ein/aus (Config + systemd Service)."""
#     try:
#         config = load_config()
#         config["web"]["mcp_enabled"] = req.enabled
#         save_config(config)
#     except (PermissionError, OSError, ValueError) as exc:
#         raise HTTPException(status_code=500, detail=f"Config-Fehler: {exc}")
#     try:
#         action = ["enable", "--now"] if req.enabled else ["disable", "--now"]
#         rc, _, stderr = await _systemctl_exec(*action, "kiara-mcp")
#         if rc != 0:
#             raise HTTPException(status_code=500, detail=f"systemctl fehlgeschlagen: {stderr}")
#     except FileNotFoundError:
#         raise HTTPException(status_code=500, detail="systemctl nicht gefunden")
#     except OSError as exc:
#         raise HTTPException(status_code=500, detail=f"Service-Steuerung fehlgeschlagen: {exc}")
#     await asyncio.sleep(1.5)
#     rc, status, _ = await _systemctl_exec("is-active", "kiara-mcp")
#     return {
#         "enabled": req.enabled,
#         "service_status": status,
#         "message": f"MCP-Server {'aktiviert' if req.enabled else 'deaktiviert'}",
#     }


# ---------------------------------------------------------------------------
# MCP-Service Live-Status (Phase 3)
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: mcp.status (Sprint 08) ---
# @router.get("/status")
# async def mcp_status(user=Depends(get_admin_user)):
#     """Gibt den Live-Status des kiara-mcp.service zurueck."""
#     config = load_config()
#     config_enabled = config["web"].get("mcp_enabled", False)
#     from kiara.shared.constants import MCP_PORT_DEFAULT
#     port = config["web"].get("mcp_port", MCP_PORT_DEFAULT)
#     service_status = "unknown"
#     service_running = False
#     pid = None
#     uptime = None
#     try:
#         rc, status, _ = await _systemctl_exec("is-active", "kiara-mcp")
#         service_status = status
#         service_running = (rc == 0)
#         rc, props, _ = await _systemctl_exec("show", "kiara-mcp",
#             "--property=MainPID,ActiveEnterTimestampMonotonic")
#         prop_dict = {}
#         for line in props.split("\n"):
#             if "=" in line:
#                 k, v = line.split("=", 1)
#                 prop_dict[k] = v
#         raw_pid = int(prop_dict.get("MainPID") or "0")
#         pid = raw_pid if raw_pid > 0 else None
#         ts_mono = prop_dict.get("ActiveEnterTimestampMonotonic", "0")
#         if pid and ts_mono and ts_mono != "0":
#             try:
#                 service_start_us = int(ts_mono)
#                 with open("/proc/uptime") as f:
#                     system_uptime_s = float(f.read().split()[0])
#                 system_uptime_us = int(system_uptime_s * 1_000_000)
#                 service_uptime_s = (system_uptime_us - service_start_us) // 1_000_000
#                 if service_uptime_s >= 0:
#                     hours, remainder = divmod(service_uptime_s, 3600)
#                     minutes = remainder // 60
#                     uptime = f"{hours}h {minutes}min"
#             except (ValueError, TypeError, OSError):
#                 uptime = None
#     except (FileNotFoundError, OSError, ValueError):
#         pass
#     return {
#         "config_enabled": config_enabled,
#         "service_running": service_running,
#         "service_status": service_status,
#         "port": port,
#         "uptime": uptime,
#         "pid": pid,
#     }


def _generate_api_key() -> tuple[str, str, str]:
    """Generiert einen API-Key.

    Returns:
        (full_key, key_hash, key_prefix)
    """
    raw = secrets.token_hex(24)  # 48 Hex-Zeichen = 192 Bit
    full_key = f"orag_{raw}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:20] + "..."
    return full_key, key_hash, key_prefix


class CreateKeyRequest(BaseModel):
    """Request zum Erstellen eines neuen MCP-API-Keys."""

    name: str
    user_id: int
    expires_at: str | None = None  # ISO-Datum oder None


# --- Migriert zu Admin-Registry: mcp.keys.create (Sprint 08) ---
# @router.post("/keys")
# async def create_key(data: CreateKeyRequest, admin=Depends(get_admin_user)):
#     """Erstellt einen neuen API-Key. Gibt den Key EINMALIG im Klartext zurueck."""
#     if not user_repo.exists(data.user_id):
#         raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
#     full_key, key_hash, key_prefix = _generate_api_key()
#     api_key_repo.create(
#         name=data.name,
#         key_hash=key_hash,
#         key_prefix=key_prefix,
#         user_id=data.user_id,
#         expires_at=data.expires_at,
#     )
#     return {"key": full_key, "prefix": key_prefix, "message": "Key nur jetzt sichtbar — sicher aufbewahren!"}

# --- Migriert zu Admin-Registry: mcp.keys.list (Sprint 08) ---
# @router.get("/keys")
# async def list_keys(admin=Depends(get_admin_user)):
#     """Listet alle API-Keys (ohne Klartext)."""
#     from kiara.shared.db import db_query
#     with db_query() as db:
#         rows = db.execute(
#             """SELECT ak.id, ak.name, ak.key_prefix, ak.is_active, ak.expires_at,
#                       ak.last_used_at, ak.created_at, ak.user_id, u.username
#                FROM api_keys ak
#                JOIN users u ON ak.user_id = u.id
#                ORDER BY ak.created_at DESC"""
#         ).fetchall()
#     return {"data": [dict(r) for r in rows]}

# --- Migriert zu Admin-Registry: mcp.keys.delete (Sprint 08) ---
# @router.delete("/keys/{key_id}")
# async def delete_key(key_id: int, admin=Depends(get_admin_user)):
#     """Loescht einen API-Key."""
#     if not api_key_repo.exists(key_id):
#         raise HTTPException(status_code=404, detail="Key nicht gefunden")
#     api_key_repo.delete(key_id)
#     return {"message": "API-Key geloescht"}

# --- Migriert zu Admin-Registry: mcp.keys.toggle (Sprint 08) ---
# @router.put("/keys/{key_id}/toggle")
# async def toggle_key(key_id: int, admin=Depends(get_admin_user)):
#     """Aktiviert/deaktiviert einen API-Key."""
#     row = api_key_repo.get_by_id(key_id)
#     if not row:
#         raise HTTPException(status_code=404, detail="Key nicht gefunden")
#     new_status = 0 if row["is_active"] else 1
#     api_key_repo.update(key_id, is_active=new_status)
#     return {"is_active": bool(new_status)}
