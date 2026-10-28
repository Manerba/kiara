"""KI-Backend-Repository.

Ersetzt SQL in: web/api_ki_backends.py (12 execute-Aufrufe)
"""

from __future__ import annotations

from kiara_db.base import BaseRepository, RawSQL


class KiBackendRepository(BaseRepository):
    table = "ki_backends"
    default_sort = "category, priority ASC, id ASC"

    _LIST_COLUMNS = (
        "id, category, provider, name, url, base_url, model, api_key, "
        "is_active, priority, trusted, extra_config, "
        "last_test_at, last_test_ok, last_test_error, created_at, updated_at"
    )

    def list_ordered(self, *, conn=None) -> list[dict]:
        """Alle Backends sortiert nach Kategorie und Prioritaet."""
        with self._read(conn) as db:
            rows = db.execute(
                f"SELECT {self._LIST_COLUMNS} FROM ki_backends "
                "ORDER BY category, priority ASC, id ASC",
            ).fetchall()
            return [dict(r) for r in rows]

    def list_by_category(
        self,
        category: str,
        active_only: bool = True,
        *,
        conn=None,
    ) -> list[dict]:
        """Backends einer Kategorie (embedding/llm/cloud/vlm/ddu)."""
        filters: dict = {"category": category}
        if active_only:
            filters["is_active"] = True
        rows, _ = self.list_filtered(
            filters,
            sort="priority ASC, id ASC",
            limit=100,
            offset=0,
            conn=conn,
        )
        return rows

    def update_test_result(
        self,
        backend_id: int,
        ok: bool,
        error: str = "",
        extra_config: str | None = None,
        *,
        conn=None,
    ) -> bool:
        """Speichert das Ergebnis eines Backend-Tests."""
        fields = {
            "last_test_at": RawSQL("NOW()"),
            "last_test_ok": 1 if ok else 0,
            "last_test_error": error,
            "updated_at": RawSQL("NOW()"),
        }
        if extra_config is not None:
            fields["extra_config"] = extra_config
        return self.update(backend_id, conn=conn, **fields)

    def deactivate(self, backend_id: int, *, conn=None) -> bool:
        """Backend deaktivieren."""
        return self.update(
            backend_id,
            conn=conn,
            is_active=False,
            updated_at=RawSQL("NOW()"),
        )

    def set_active(
        self,
        backend_id: int,
        active: bool,
        *,
        conn=None,
    ) -> bool:
        """Aktiv-Status setzen."""
        return self.update(
            backend_id,
            conn=conn,
            is_active=bool(active),
            updated_at=RawSQL("NOW()"),
        )

    def get_active_by_id(self, backend_id: int, *, conn=None) -> dict | None:
        """Gibt ein aktives Backend anhand der ID zurueck."""
        with self._read(conn) as db:
            row = db.execute(
                f"SELECT {self._LIST_COLUMNS} FROM ki_backends WHERE id = %s AND is_active = true",
                (backend_id,),
            ).fetchone()
            return dict(row) if row else None
