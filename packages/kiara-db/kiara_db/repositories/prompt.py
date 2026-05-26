"""Prompt-Template-Repository.

Ersetzt SQL in: web/api_chat.py (Template-Lookup), web/api_ki.py
"""

from __future__ import annotations

import hashlib

from kiara_db.base import BaseRepository


class PromptTemplateRepository(BaseRepository):
    table = "prompt_templates"
    default_sort = "sort_order ASC, id ASC"

    def list_chat_modes(self, *, conn=None) -> list[dict]:
        """Aktive Templates die im Chat als Modus angezeigt werden."""
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT id, name, label, chain_type, badge_color, sort_order
                   FROM prompt_templates
                   WHERE show_in_chat = 1
                     AND is_active = true
                     AND is_cloud = false
                     AND chain_type IN ('standard', 'unchained', 'agent', 'rewriter')
                   ORDER BY sort_order, id""",
            ).fetchall()
            return [dict(r) for r in rows]

    def save_history(
        self,
        template_id: int,
        content: str,
        cloud_system_prompt: str = "",
        changed_by: str = "",
        change_reason: str = "",
        *,
        conn=None,
    ) -> int:
        """Speichert die aktuelle Version als History-Eintrag.

        Nutzt self._write(conn) fuer INSERT in prompt_template_history.
        """
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._write(conn) as db:
            row = db.execute(
                """INSERT INTO prompt_template_history
                   (template_id, content, content_hash, cloud_system_prompt,
                    changed_by, change_reason)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    template_id,
                    content,
                    content_hash,
                    cloud_system_prompt,
                    changed_by,
                    change_reason,
                ),
            ).fetchone()
            return row["id"]

    def get_history(self, template_id: int, limit: int = 50, *, conn=None) -> list[dict]:
        """Alle History-Eintraege fuer ein Template, neueste zuerst.

        Nutzt self._read(conn) fuer SELECT auf prompt_template_history.
        """
        with self._read(conn) as db:
            rows = db.execute(
                """SELECT id, template_id, content, content_hash,
                          cloud_system_prompt, changed_by, change_reason, created_at
                   FROM prompt_template_history
                   WHERE template_id = %s
                   ORDER BY created_at DESC, id DESC
                   LIMIT %s""",
                (template_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_name(self, name: str, *, conn=None) -> dict | None:
        """Template anhand des Namens (aktiv)."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT * FROM prompt_templates
                   WHERE name = %s AND is_active = true""",
                (name,),
            ).fetchone()
            return dict(row) if row else None

    def get_active(self, template_id: int, *, conn=None) -> dict | None:
        """Aktives Template anhand der ID."""
        with self._read(conn) as db:
            row = db.execute(
                """SELECT * FROM prompt_templates
                   WHERE id = %s AND is_active = true""",
                (template_id,),
            ).fetchone()
            return dict(row) if row else None
