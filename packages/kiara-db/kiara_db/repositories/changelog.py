"""Config-Changelog-Repository.

Ersetzt SQL in: web/api_config_changelog.py, web/config_changelog.py
"""

from __future__ import annotations

from kiara_db.base import BaseRepository
from kiara_db.repositories.audit import _AuditTrailMixin


class ConfigChangelogRepository(_AuditTrailMixin, BaseRepository):
    table = "config_changelog"
    default_sort = "created_at DESC"

    def list_by_area(
        self,
        area: str | None = None,
        limit: int = 100,
        offset: int = 0,
        *,
        conn=None,
    ) -> tuple[list[dict], int]:
        """Changelog-Eintraege, optional nach Bereich gefiltert."""
        filters = {}
        if area:
            filters["area"] = area
        return self.list_filtered(
            filters,
            sort="created_at DESC",
            limit=limit,
            offset=offset,
            conn=conn,
        )

    def list_distinct_areas(self, *, conn=None) -> list[str]:
        """Alle vorhandenen Bereiche (fuer Filter-Dropdown)."""
        with self._read(conn) as db:
            rows = db.execute("SELECT DISTINCT area FROM config_changelog ORDER BY area").fetchall()
            return [r["area"] for r in rows]
