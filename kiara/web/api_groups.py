"""
KIara Admin - Groups API
LDAP-Gruppen importieren, verwalten, Mitgliedschaften synchronisieren.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from kiara.repositories.config import ConfigRepository
from kiara.repositories.group import GroupRepository
from kiara.web.config_utils import build_ldap_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["groups"])

group_repo = GroupRepository()
config_repo = ConfigRepository()


class GroupImportRequest(BaseModel):
    """Request zum Importieren von LDAP-Gruppen in die lokale Datenbank."""

    base_dn: str = ""
    group_dns: list[str] = []


# ---------------------------------------------------------------------------
# List / Delete
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: groups.list (Sprint 06) ---
# @router.get("/api/admin/groups")
# async def list_groups(user=Depends(get_admin_user)):
#     ...

# --- Migriert zu Admin-Registry: groups.members (Sprint 06) ---
# @router.get("/api/admin/groups/{group_id}/members")
# async def get_group_members(group_id: int, user=Depends(get_admin_user)):
#     ...

# --- Migriert zu Admin-Registry: groups.delete (Sprint 06) ---
# @router.delete("/api/admin/groups/{group_id}")
# async def delete_group(group_id: int, user=Depends(get_admin_user)):
#     ...

# ---------------------------------------------------------------------------
# Import from LDAP
# ---------------------------------------------------------------------------


def _get_ldap_config() -> dict | None:
    """Load LDAP config from DB."""
    row = config_repo.get("ldap_config")
    if not row:
        return None
    return build_ldap_config(row)


# --- Migriert zu Admin-Registry: groups.import_ldap (Sprint 06) ---
# @router.post("/api/admin/groups/import")
# async def import_groups(data: GroupImportRequest, user=Depends(get_admin_user)):
#     ...

# ---------------------------------------------------------------------------
# Sync group memberships for users
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: groups.sync_members (Sprint 06) ---
# @router.post("/api/admin/groups/sync-members")
# async def sync_group_members(user=Depends(get_admin_user)):
#     ...

# ---------------------------------------------------------------------------
# LDAP group listing (for import preview, not saved)
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: groups.ldap_preview (Sprint 06) ---
# @router.get("/api/admin/groups/ldap")
# async def list_ldap_groups(base_dn: str = "", user=Depends(get_admin_user)):
#     ...
