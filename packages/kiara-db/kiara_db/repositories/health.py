"""Health-Log-Repository.

Ersetzt SQL in: web/api_health.py
"""

from __future__ import annotations

from kiara_db.base import BaseRepository


class HealthLogRepository(BaseRepository):
    table = "health_log"
    default_sort = "created_at DESC"

    def list_recent(self, limit: int = 100, *, conn=None) -> list[dict]:
        """Letzte N Health-Check-Eintraege."""
        rows, _ = self.list_filtered(
            sort="created_at DESC",
            limit=limit,
            offset=0,
            conn=conn,
        )
        return rows

    def cleanup_old(self, days: int = 30, *, conn=None) -> int:
        """Loescht Health-Logs aelter als N Tage."""
        days = max(1, days)
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM health_log "
                "WHERE created_at::timestamp < NOW() - make_interval(days => %s)",
                (days,),
            )
            return cursor.rowcount
