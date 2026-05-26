import logging

from kiara_db.base import BaseRepository, RawSQL

logger = logging.getLogger(__name__)


class AdminChatRepository(BaseRepository):
    table = "admin_chats"
    default_sort = "updated_at DESC"

    def list_by_user(self, user_id: int, status: str | None = None, *, conn=None) -> list[dict]:
        """Alle Chats eines Users, optional nach Status gefiltert."""
        with self._read(conn) as c:
            if status:
                rows = c.execute(
                    f"SELECT c.*, "
                    f"(SELECT COUNT(*) FROM admin_chat_messages WHERE chat_id = c.id) as message_count "
                    f"FROM {self.table} c WHERE c.user_id = %s AND c.status = %s "
                    f"ORDER BY {self.default_sort}",
                    (user_id, status),
                ).fetchall()
            else:
                rows = c.execute(
                    f"SELECT c.*, "
                    f"(SELECT COUNT(*) FROM admin_chat_messages WHERE chat_id = c.id) as message_count "
                    f"FROM {self.table} c WHERE c.user_id = %s "
                    f"ORDER BY {self.default_sort}",
                    (user_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_owned(self, chat_id: int, user_id: int, *, conn=None) -> dict | None:
        """Chat nur zurueckgeben wenn User der Owner ist."""
        with self._read(conn) as c:
            row = c.execute(
                f"SELECT * FROM {self.table} WHERE id = %s AND user_id = %s",
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


class AdminChatMessageRepository(BaseRepository):
    table = "admin_chat_messages"
    default_sort = "created_at ASC"

    def list_by_chat(self, chat_id: int, *, conn=None) -> list[dict]:
        with self._read(conn) as c:
            rows = c.execute(
                f"SELECT * FROM {self.table} WHERE chat_id = %s ORDER BY {self.default_sort}",
                (chat_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        sources: list | dict | None = None,
        tool_calls: list | None = None,
        metadata: dict | None = None,
        *,
        conn=None,
    ) -> int:
        """Fuegt eine Nachricht hinzu und gibt die neue ID zurueck.

        Analog zu MessageRepository.add_message(): JSON-Serialisierung
        erfolgt im Repository, nicht im Aufrufer.
        """
        import json

        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        return self.create(
            conn=conn,
            chat_id=chat_id,
            role=role,
            content=content,
            sources=sources_json,
            tool_calls=tool_calls_json,
            metadata=metadata_json,
        )

    def load_history(self, chat_id: int, limit: int = 20, *, conn=None) -> list[dict]:
        """Letzte N Nachrichten (chronologisch, aelteste zuerst), ohne 'system'-Role.

        Liefert role, content und metadata — letzteres wird vom History-Filter
        in _stream_admin_query benoetigt, um Setup-Messages auszuschliessen.
        """
        with self._read(conn) as c:
            rows = c.execute(
                f"SELECT role, content, metadata FROM {self.table} "
                f"WHERE chat_id = %s AND role != 'system' "
                f"ORDER BY id DESC LIMIT %s",
                (chat_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def delete_by_chat(self, chat_id: int, *, conn=None) -> int:
        with self._write(conn) as c:
            cursor = c.execute(
                f"DELETE FROM {self.table} WHERE chat_id = %s",
                (chat_id,),
            )
            return cursor.rowcount


class AdminConfirmationRepository(BaseRepository):
    table = "admin_chat_confirmations"
    default_sort = "created_at ASC"

    def get_pending(self, chat_id: int) -> dict | None:
        """Holt die aelteste pending Confirmation fuer einen Chat."""
        self.expire_pending(chat_id)
        with self._read() as db:
            rows = db.execute(
                f"SELECT * FROM {self.table} "
                "WHERE chat_id = %s AND status = 'pending' AND expires_at > NOW() "
                "ORDER BY created_at ASC LIMIT 1",
                (chat_id,),
            ).fetchall()
        return dict(rows[0]) if rows else None

    def list_pending(self, chat_id: int) -> list[dict]:
        """Holt alle pending Confirmations fuer einen Chat (FIFO-Reihenfolge)."""
        self.expire_pending(chat_id)
        with self._read() as db:
            rows = db.execute(
                f"SELECT * FROM {self.table} "
                "WHERE chat_id = %s AND status = 'pending' AND expires_at > NOW() "
                "ORDER BY created_at ASC",
                (chat_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def expire_pending(self, chat_id: int | None = None) -> int:
        """Persistiert abgelaufene pending Confirmations als dismissed."""
        sql = (
            f"UPDATE {self.table} SET status = 'dismissed' "
            "WHERE status = 'pending' AND expires_at <= NOW()"
        )
        params: tuple = ()
        if chat_id is not None:
            sql += " AND chat_id = %s"
            params = (chat_id,)
        with self._write() as db:
            cursor = db.execute(sql, params)
            return cursor.rowcount

    def confirm(self, action_id: str) -> bool:
        """Atomar: Setzt Status nur fuer nicht abgelaufene pending Confirmations."""
        with self._write() as db:
            cursor = db.execute(
                f"UPDATE {self.table} SET status = 'confirmed' "
                "WHERE action_id = %s AND status = 'pending' AND expires_at > NOW()",
                (action_id,),
            )
            return cursor.rowcount > 0

    def dismiss(self, action_id: str) -> bool:
        """Atomar: Setzt Status auf 'dismissed' nur wenn noch 'pending'."""
        return (
            self.update_where(
                {"action_id": action_id, "status": "pending"},
                status="dismissed",
            )
            > 0
        )

    def dismiss_all(self, chat_id: int) -> int:
        """Verwirft alle pending Confirmations eines Chats (atomar)."""
        with self._write() as db:
            cursor = db.execute(
                f"UPDATE {self.table} SET status = 'dismissed' "
                f"WHERE chat_id = %s AND status = 'pending'",
                (chat_id,),
            )
            return cursor.rowcount


_WIZARD_SECRET_KEY_PATTERNS: tuple[str, ...] = (
    "password",
    "secret",
    "api_key",
    "apikey",
    "dsn",
    "token",
    "bind_password",
    "access_key",
    "client_secret",
    "pat",
)
"""Substring-Patterns fuer Parameter-Keys, deren Wert als Secret gilt.

Wird bei Wizard-Persistenz (#558) eingesetzt: Werte dieser Keys werden vor
DB-Schreibzugriff per Fernet verschluesselt und beim Lesen transparent
wieder entschluesselt. Konservativ weit — lieber ein Nicht-Secret
verschluesseln als ein Secret im Klartext persistieren.
"""


def _is_secret_key(key: str) -> bool:
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in _WIZARD_SECRET_KEY_PATTERNS)


def _encrypt_wizard_params(params: dict) -> dict:
    """Verschluesselt Secret-Felder im Wizard-Parameter-Dict (#558).

    Nur String-Werte werden verschluesselt (non-strings unveraendert).
    Bereits verschluesselte Werte (`is_encrypted`) bleiben unveraendert,
    damit ein Read-Modify-Write-Zyklus nicht doppelt verschluesselt.
    """
    from kiara.shared.crypto import encrypt, is_encrypted

    result: dict = {}
    for key, value in params.items():
        if isinstance(value, str) and value and _is_secret_key(key) and not is_encrypted(value):
            result[key] = encrypt(value)
        else:
            result[key] = value
    return result


def _decrypt_wizard_params(params: dict) -> dict:
    """Entschluesselt Fernet-verschluesselte Werte im Wizard-Parameter-Dict.

    Nicht verschluesselte Werte bleiben unveraendert (Abwaerts-Kompatibilitaet
    zu Altdaten aus der Zeit vor #558). Fehlgeschlagene Entschluesselung
    liefert den Rohwert zurueck plus Warnung.
    """
    from kiara.shared.crypto import decrypt, is_encrypted

    result: dict = {}
    for key, value in params.items():
        if isinstance(value, str) and is_encrypted(value):
            try:
                result[key] = decrypt(value)
            except Exception as exc:
                logger.warning(
                    "Wizard-Param '%s' Entschluesselung fehlgeschlagen: %s",
                    key,
                    exc,
                )
                result[key] = value
        else:
            result[key] = value
    return result


def mask_wizard_secret_params(params: dict) -> dict:
    """Maskiert Secret-Felder fuer Frontend-/API-Auslieferung (#1053).

    Tool-Ausfuehrung nutzt entschluesselte Originalwerte server-intern;
    SSE-Events und REST-Antworten duerfen jedoch keine Klartext-Secrets
    transportieren.
    """
    from kiara.shared.constants import PASSWORD_MASK

    result: dict = {}
    for key, value in params.items():
        if isinstance(value, str) and value and _is_secret_key(key):
            result[key] = PASSWORD_MASK
        else:
            result[key] = value
    return result


def _deep_merge_dict(base: dict, updates: dict) -> dict:
    """Rekursiver Dict-Merge fuer Wizard-Parameter."""
    merged = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged


class AdminWizardRepository(BaseRepository):
    table = "admin_chat_wizards"
    default_sort = "created_at DESC"

    def create_wizard(
        self, *, chat_id: int, user_id: int, function_name: str, collected_params: dict
    ) -> int:
        """Legt einen neuen Wizard-Datensatz an mit verschluesselten Secrets (#558).

        Erwartet ``collected_params`` als dict; verschluesselt Secret-Felder
        vor der Persistierung und ruft die Basis-``create`` mit dem
        JSON-String auf. Transparent fuer Aufrufer ausserhalb des Wizards.
        """
        import json

        encrypted = _encrypt_wizard_params(collected_params)
        return self.create(
            chat_id=chat_id,
            user_id=user_id,
            function_name=function_name,
            collected_params=json.dumps(encrypted, default=str),
        )

    def get_active(self, chat_id: int) -> dict | None:
        """Holt den aktiven Wizard fuer einen Chat; entschluesselt Secret-Felder."""
        import json

        rows, _ = self.list_filtered(
            {"chat_id": chat_id, "status": "collecting"},
            sort="created_at DESC",
            limit=1,
        )
        if not rows:
            return None
        row = rows[0]
        try:
            raw = json.loads(row.get("collected_params") or "{}")
            if isinstance(raw, dict):
                row["collected_params"] = json.dumps(
                    _decrypt_wizard_params(raw),
                    default=str,
                )
        except (ValueError, TypeError):
            pass
        return row

    def update_params(self, wizard_id: int, new_params: dict, *, conn=None) -> None:
        """Atomarer JSONB-Merge neuer Parameter in collected_params (PostgreSQL).

        Secret-Felder werden vor dem Merge verschluesselt (#558), damit sie
        nie im Klartext in ``admin_chat_wizards`` landen.
        """
        import json

        encrypted = _encrypt_wizard_params(new_params)
        try:
            params_json = json.dumps(encrypted, default=str)
        except (TypeError, ValueError) as exc:
            logger.error("Wizard-Params nicht serialisierbar (id=%d): %s", wizard_id, exc)
            return
        with self._write(conn) as db:
            row = db.execute(
                f"SELECT collected_params FROM {self.table} WHERE id = %s FOR UPDATE",
                (wizard_id,),
            ).fetchone()
            if not row:
                return
            try:
                current = json.loads(row.get("collected_params") or "{}")
            except (TypeError, ValueError):
                current = {}
            if not isinstance(current, dict):
                current = {}
            params_json = json.dumps(_deep_merge_dict(current, encrypted), default=str)
            db.execute(
                f"UPDATE {self.table} "
                f"SET collected_params = %s, "
                f"    updated_at = NOW() "
                f"WHERE id = %s",
                (params_json, wizard_id),
            )

    def complete(self, wizard_id: int) -> None:
        """Markiert Wizard als ausgefuehrt."""
        self.update(wizard_id, status="executed", updated_at=RawSQL("NOW()"))

    def cancel(self, wizard_id: int) -> None:
        """Bricht Wizard ab."""
        self.update(wizard_id, status="cancelled", updated_at=RawSQL("NOW()"))
