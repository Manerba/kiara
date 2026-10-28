"""Admin-Funktionen fuer Cache-Management."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="cache.stats",
    category="cache",
    description="Cache-Statistiken anzeigen",
    help="Zeigt Statistiken des Query-Caches an (Größe, Treffer, Fehlschläge).",
    examples=[
        "Zeig mir die Cache-Statistiken",
        "Wie groß ist der Cache?",
        "Cache-Status?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Cache-Statistiken",
    raw_response=True,
    http=("GET", "/api/admin/cache/stats"),
)
async def cache_stats(user: dict) -> AdminResult:
    from kiara.shared.cache import get_query_cache

    stats = get_query_cache().stats()
    return AdminResult.ok("Cache-Statistiken", data=stats)


@admin_function(
    name="cache.clear",
    category="cache",
    description="Cache leeren",
    help="Leert den gesamten Query-Cache. Nachfolgende Anfragen werden "
    "ohne Cache-Treffer beantwortet bis der Cache sich wieder füllt.",
    examples=[
        "Leere den Cache",
        "Cache invalidieren",
        "Cache löschen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung der Löschung",
    requires_confirmation=True,
    confirm_template="cache leeren",
    changelog_area="cache",
    changelog_action="delete",
    raw_response=True,
    http=("POST", "/api/admin/cache/clear"),
)
async def cache_clear(user: dict) -> AdminResult:
    from kiara.shared.cache import get_query_cache

    get_query_cache().invalidate()
    return AdminResult.ok("Cache geleert", data={"message": "Cache geleert"})
