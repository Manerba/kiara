"""API-Key-Repository.

Ersetzt SQL in: web/api_mcp.py, web/api_key_auth.py
"""

from __future__ import annotations

from kiara_db.base import BaseRepository, RawSQL


class ApiKeyRepository(BaseRepository):
    table = "api_keys"
    default_sort = "created_at DESC"

    def get_by_hash(self, key_hash: str, *, conn=None) -> dict | None:
        """API-Key anhand des Hashes."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT * FROM api_keys WHERE key_hash = %s AND is_active = true",
                (key_hash,),
            ).fetchone()
            return dict(row) if row else None

    def touch_last_used(self, key_id: int, *, conn=None) -> None:
        """Aktualisiert last_used_at."""
        self.update(key_id, conn=conn, last_used_at=RawSQL("NOW()"))

    def list_by_user(self, user_id: int, *, conn=None) -> list[dict]:
        """Alle Keys eines Users."""
        rows, _ = self.list_filtered(
            {"user_id": user_id},
            sort="created_at DESC",
            limit=100,
            offset=0,
            conn=conn,
        )
        return rows

    def revoke(self, key_id: int, *, conn=None) -> bool:
        """Key deaktivieren (soft-delete)."""
        return self.update(key_id, conn=conn, is_active=False)
