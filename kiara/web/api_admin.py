"""
KIara Admin - User Management API
User CRUD operations.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["users"])


class CreateUserRequest(BaseModel):
    """Request zum Erstellen eines neuen lokalen Benutzers."""

    username: str
    password: str = ""
    vorname: str = ""
    nachname: str = ""
    email: str = ""
    is_admin: bool = False


class UpdateUserRequest(BaseModel):
    """Request zum Aktualisieren eines Benutzerkontos."""

    vorname: str | None = None
    nachname: str | None = None
    email: str | None = None
    is_admin: bool | None = None
    is_active: bool | None = None
    password: str | None = None


# --- Migriert zu Admin-Registry: users.list (Sprint 06) ---
# @router.get("/users")
# async def list_users(user=Depends(get_admin_user)):
#     ...

# --- Migriert zu Admin-Registry: users.create (Sprint 06) ---
# @router.post("/users")
# async def create_user(data: CreateUserRequest, user=Depends(get_admin_user)):
#     ...

# --- Migriert zu Admin-Registry: users.update (Sprint 06) ---
# @router.put("/users/{user_id}")
# async def update_user(user_id: int, data: UpdateUserRequest, user=Depends(get_admin_user)):
#     ...

# --- Migriert zu Admin-Registry: users.delete (Sprint 06) ---
# @router.delete("/users/{user_id}")
# async def delete_user(user_id: int, user=Depends(get_admin_user)):
#     ...
