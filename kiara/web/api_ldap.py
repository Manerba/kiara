"""
KIara Admin - LDAP API-Router
LDAP-Konfiguration, Gruppen, Synchronisierung.
Orientiert am Tareas-Pattern: Config speichern testet automatisch,
Gruppe auswaehlen triggert sofort Sync.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from kiara.repositories.config import ConfigRepository
from kiara.repositories.user import UserRepository
from kiara.web.config_utils import build_ldap_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/ldap", tags=["ldap"])

config_repo = ConfigRepository()
user_repo = UserRepository()

# ============================================================
# Pydantic-Modelle
# ============================================================


class LdapConfigRequest(BaseModel):
    """Request zum Speichern der LDAP-Verbindungskonfiguration."""

    server_url: str
    port: int = 389
    use_ssl: int = 0
    bind_dn: str
    bind_password: str
    search_base: str


class GroupSelectRequest(BaseModel):
    """Request zum Auswaehlen einer AD-Gruppe fuer die Benutzer-Synchronisierung."""

    group_dn: str
    group_name: str


# ============================================================
# Helpers
# ============================================================


def _get_connection_config() -> dict | None:
    """Build LDAP connection config dict from DB."""
    row = config_repo.get("ldap_config")
    if not row:
        return None
    return build_ldap_config(row)


# ============================================================
# Config CRUD
# ============================================================

# --- Migriert zu Admin-Registry: ldap.config.read (Sprint 08) ---
# @router.get("/config")
# async def get_ldap_config(user=Depends(get_admin_user)):
#     """Aktuelle LDAP-Konfiguration laden (bind_password maskiert)."""
#     config = get_masked_config("ldap_config", mask_fields=["bind_password"])
#     return {"config": config}

# --- Migriert zu Admin-Registry: ldap.config.update (Sprint 08) ---
# @router.post("/config")
# async def save_ldap_config(data: LdapConfigRequest, user=Depends(get_admin_user)):
#     """LDAP-Konfiguration speichern (nach automatischem Verbindungstest)."""
#     from kiara.web.ldap_utils import test_connection
#     if not data.server_url.strip():
#         raise HTTPException(status_code=400, detail="Server ist Pflichtfeld")
#     if not data.bind_dn.strip():
#         raise HTTPException(status_code=400, detail="Bind-DN ist Pflichtfeld")
#     if not data.bind_password.strip():
#         raise HTTPException(status_code=400, detail="Bind-Passwort ist Pflichtfeld")
#     if not data.search_base.strip():
#         raise HTTPException(status_code=400, detail="Suchpfad ist Pflichtfeld")
#     with db_query() as db:
#         actual_password = resolve_masked_field(db, "ldap_config", "bind_password", data.bind_password)
#     clear_password = decrypt(actual_password) if is_encrypted(actual_password) else actual_password
#     config = {
#         "server": data.server_url.strip(),
#         "port": data.port,
#         "use_ssl": data.use_ssl,
#         "bind_dn": data.bind_dn.strip(),
#         "bind_password": clear_password,
#         "search_base": data.search_base.strip(),
#     }
#     success, message = await asyncio.to_thread(test_connection, config)
#     if not success:
#         raise HTTPException(status_code=400, detail=f"Verbindungstest fehlgeschlagen: {message}")
#     config_repo.set("ldap_config", {
#         "server_url": data.server_url.strip(),
#         "port": data.port,
#         "use_ssl": data.use_ssl,
#         "bind_dn": data.bind_dn.strip(),
#         "bind_password": actual_password,
#         "search_base": data.search_base.strip(),
#     })
#     logger.info("LDAP config saved by %s", user["username"])
#     return {"message": "LDAP-Konfiguration gespeichert"}

# --- Migriert zu Admin-Registry: ldap.config.delete (Sprint 08) ---
# @router.delete("/config")
# async def delete_ldap_config(user=Depends(get_admin_user)):
#     """LDAP-Konfiguration loeschen und alle LDAP-Benutzer deaktivieren."""
#     with db_transaction() as db:
#         db.execute("DELETE FROM ldap_config WHERE id = 1")
#         db.execute("UPDATE users SET is_active = false WHERE auth_source = 'ldap'")
#     logger.info("LDAP config deleted by %s", user["username"])
#     return {"message": "LDAP-Konfiguration geloescht, LDAP-Benutzer deaktiviert"}

# ============================================================
# Gruppen
# ============================================================

# --- Migriert zu Admin-Registry: ldap.groups (Sprint 08) ---
# @router.get("/groups")
# async def get_ldap_groups(user=Depends(get_admin_user)):
#     """Verfuegbare AD-Gruppen auflisten."""
#     from kiara.web.ldap_utils import get_groups
#     config = _get_connection_config()
#     if not config:
#         raise HTTPException(status_code=400, detail="Keine LDAP-Konfiguration vorhanden")
#     try:
#         groups = await asyncio.to_thread(get_groups, config)
#         return {"groups": groups}
#     except Exception as exc:
#         logger.exception("Fehler beim Laden der LDAP-Gruppen")
#         raise HTTPException(status_code=500, detail="LDAP-Gruppen konnten nicht geladen werden")

# --- Migriert zu Admin-Registry: ldap.select_group (Sprint 08) ---
# @router.post("/group")
# async def select_group(data: GroupSelectRequest, user=Depends(get_admin_user)):
#     """Gruppe auswaehlen und sofort Sync ausfuehren."""
#     from kiara.web.ldap_sync import sync_ldap_users
#     row = config_repo.get("ldap_config")
#     if not row:
#         raise HTTPException(status_code=400, detail="Keine LDAP-Konfiguration vorhanden")
#     config_repo.set("ldap_config", {
#         "server_url": row["server_url"],
#         "port": row["port"],
#         "use_ssl": row.get("use_ssl", 0),
#         "bind_dn": row["bind_dn"],
#         "bind_password": row["bind_password"],
#         "search_base": row["search_base"],
#         "group_dn": data.group_dn,
#         "group_name": data.group_name,
#     })
#     result = await asyncio.to_thread(sync_ldap_users)
#     if "error" in result:
#         raise HTTPException(status_code=500, detail=result["error"])
#     logger.info("LDAP group selected: '%s' by %s", data.group_name, user["username"])
#     return {
#         "message": f"Gruppe '{data.group_name}' ausgewaehlt und synchronisiert",
#         "sync": result,
#     }

# --- Migriert zu Admin-Registry: ldap.deselect_group (Sprint 08) ---
# @router.delete("/group")
# async def deselect_group(user=Depends(get_admin_user)):
#     """Gruppenauswahl aufheben und alle LDAP-Benutzer deaktivieren."""
#     with db_transaction() as db:
#         db.execute(
#             "UPDATE ldap_config SET group_dn = NULL, group_name = NULL WHERE id = 1"
#         )
#         db.execute("UPDATE users SET is_active = false WHERE auth_source = 'ldap'")
#     logger.info("LDAP group deselected by %s", user["username"])
#     return {"message": "Gruppenauswahl aufgehoben, LDAP-Benutzer deaktiviert"}

# ============================================================
# Sync
# ============================================================

# --- Migriert zu Admin-Registry: ldap.sync (Sprint 08) ---
# @router.post("/sync")
# async def manual_sync(user=Depends(get_admin_user)):
#     """Manueller Sync."""
#     from kiara.web.ldap_sync import sync_ldap_users
#     result = await asyncio.to_thread(sync_ldap_users)
#     if "error" in result:
#         raise HTTPException(status_code=500, detail=result["error"])
#     logger.info("LDAP manual sync by %s: %s", user["username"], result)
#     return {"message": "Synchronisierung abgeschlossen", "sync": result}
