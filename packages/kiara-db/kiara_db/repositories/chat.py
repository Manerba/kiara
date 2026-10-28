"""Chat- und Message-Repositories.

Ersetzt SQL in: web/api_chat.py (25 execute-Aufrufe)
"""

from __future__ import annotations

import json

from kiara_db.base import BaseRepository, RawSQL


class ChatRepository(BaseRepository):
    table = "chats"
    default_sort = "updated_at DESC"

    def list_by_user(
        self,
        user_id: int,
        status: str | None = None,
        *,
        conn=None,
    ) -> list[dict]:
        """Alle Chats eines Users mit message_count, optional nach Status."""
        conditions = ["c.user_id = %s"]
        params: list = [user_id]
        if status:
            conditions.append("c.status = %s")
            params.append(status)
        where = " WHERE " + " AND ".join(conditions)

        with self._read(conn) as db:
            rows = db.execute(
                f"""SELECT c.id, c.user_id, c.title, c.status,
                           c.created_at, c.updated_at,
                           COUNT(m.id) AS message_count
                    FROM chats c
                    LEFT JOIN chat_messages m ON c.id = m.chat_id
                    {where}
                    GROUP BY c.id, c.user_id, c.title, c.status,
                             c.created_at, c.updated_at
                    ORDER BY c.updated_at DESC""",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def get_owned(
        self,
        chat_id: int,
        user_id: int,
        *,
        conn=None,
    ) -> dict | None:
        """Chat nur zurueckgeben wenn der User der Owner ist."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT * FROM chats WHERE id = %s AND user_id = %s",
                (chat_id, user_id),
            ).fetchone()
            return dict(row) if row else None

    def archive(self, chat_id: int, *, conn=None) -> bool:
        return self.update(chat_id, conn=conn, status="archived")

    def restore(self, chat_id: int, *, conn=None) -> bool:
        return self.update(chat_id, conn=conn, status="active")

    def touch(self, chat_id: int, *, conn=None) -> bool:
        """Setzt updated_at auf NOW()."""
        return self.update(chat_id, conn=conn, updated_at=RawSQL("NOW()"))


class MessageRepository(BaseRepository):
    table = "chat_messages"
    default_sort = "created_at ASC"

    def list_by_chat(self, chat_id: int, *, conn=None) -> list[dict]:
        """Alle Nachrichten eines Chats, chronologisch."""
        rows, _ = self.list_filtered(
            {"chat_id": chat_id},
            sort="created_at ASC",
            limit=10000,
            offset=0,
            conn=conn,
        )
        return rows

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        sources: list | dict | None = None,
        *,
        warnings: list[dict] | None = None,
        conn=None,
    ) -> int:
        """Nachricht hinzufuegen. Sources werden als JSON gespeichert."""
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        return self.create(
            conn=conn,
            chat_id=chat_id,
            role=role,
            content=content,
            sources=sources_json,
            warnings=json.dumps(warnings or [], ensure_ascii=False),
        )

    def delete_by_chat(self, chat_id: int, *, conn=None) -> int:
        """Loescht alle Nachrichten eines Chats."""
        return self.delete_where({"chat_id": chat_id}, conn=conn)

    def load_history(
        self,
        chat_id: int,
        limit: int = 20,
        *,
        conn=None,
    ) -> list[dict]:
        """Letzte N Nachrichten (ohne Kontext-Chunks), chronologisch (aelteste zuerst)."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT role, content FROM chat_messages
                   WHERE chat_id = %s AND role != 'context'
                   ORDER BY created_at DESC, id DESC LIMIT %s""",
                (chat_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]
