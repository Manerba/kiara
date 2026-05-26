"""KIara Admin - Cache API"""

from fastapi import APIRouter

from kiara.config import load_config
from kiara.web.api_utils import lazy_singleton

cache_router = APIRouter(prefix="/api/admin/cache", tags=["cache"])


@lazy_singleton
def _get_cache():
    """Lazy-initialisiert und cached die QueryCache-Instanz."""
    from kiara.cache.query_cache import QueryCache

    return QueryCache(load_config())


# --- Migriert zu Admin-Registry: cache.stats (Sprint 08) ---
# @cache_router.get("/stats")
# async def cache_stats(user=Depends(get_admin_user)):
#     return _get_cache().stats()

# --- Migriert zu Admin-Registry: cache.clear (Sprint 08) ---
# @cache_router.post("/clear")
# async def cache_clear(user=Depends(get_admin_user)):
#     _get_cache().invalidate()
#     return {"message": "Cache geleert"}
