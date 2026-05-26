"""Config-Changelog API — Paginierte Aenderungshistorie."""

from fastapi import APIRouter

from kiara.repositories.changelog import ConfigChangelogRepository

router = APIRouter(prefix="/api/admin/system", tags=["system"])

changelog_repo = ConfigChangelogRepository()


# --- Migriert zu Admin-Registry: system.changelog (Sprint 04) ---
# @router.get("/changelog")
# async def get_changelog(
#     user=Depends(get_admin_user),
#     area: str | None = Query(None),
#     page: int = Query(1, ge=1),
#     page_size: int = Query(50, ge=1, le=200),
#     sort_field: str = Query("created_at"),
#     sort_dir: str = Query("desc"),
# ):
#     """Paginierte Changelog-Liste mit optionalem Bereichs-Filter."""
#     return query_changelog(
#         area=area, page=page, page_size=page_size,
#         sort_field=sort_field, sort_dir=sort_dir,
#     )


# --- Migriert zu Admin-Registry: system.changelog_areas (Sprint 04) ---
# @router.get("/changelog/areas")
# async def get_changelog_areas(user=Depends(get_admin_user)):
#     """Gibt alle vorhandenen Bereiche fuer den Filter-Dropdown zurueck."""
#     return {"areas": changelog_repo.list_distinct_areas()}
