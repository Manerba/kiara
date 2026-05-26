"""User- und Session-Repositories.

Ersetzt SQL in: web/api_admin.py (11), web/api_auth.py (6)
"""

from __future__ import annotations

from kiara_db.base import BaseRepository
from kiara_db.connection import db_query, db_transaction  # noqa: F401


class UserRepository(BaseRepository):
    table = "users"
    default_sort = "username ASC"

    _LIST_COLUMNS = (
        "id, username, vorname, nachname, email, "
        "is_admin, auth_source, ldap_dn, is_active, created_at"
    )

    def list_safe(self, *, conn=None) -> list[dict]:
        """Alle User OHNE password_hash."""
        with self._read(conn) as db:
            rows = db.execute(
                f"SELECT {self._LIST_COLUMNS} FROM users ORDER BY username",
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_username(self, username: str, *, conn=None) -> dict | None:
        """User anhand des Usernames (case-insensitive)."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT * FROM users WHERE LOWER(username) = LOWER(%s)",
                (username,),
            ).fetchone()
            return dict(row) if row else None

    def username_exists(self, username: str, *, conn=None) -> bool:
        """Prueft ob ein Username bereits vergeben ist."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT id FROM users WHERE LOWER(username) = LOWER(%s)",
                (username,),
            ).fetchone()
            return row is not None

    def set_password(
        self,
        user_id: int,
        password_hash: str,
        *,
        conn=None,
    ) -> bool:
        """Setzt neues Passwort-Hash."""
        return self.update(
            user_id,
            conn=conn,
            password_hash=password_hash,
            must_change_password=False,
        )

    def delete_with_sessions(self, user_id: int, *, conn=None) -> bool:
        """Loescht User; DB-FKs entfernen abhaengige Sessions/Berechtigungen."""
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM users WHERE id = %s",
                (user_id,),
            )
            return cursor.rowcount > 0


class SessionRepository(BaseRepository):
    table = "sessions"
    default_sort = "created_at DESC"

    def get_by_token(self, token: str, *, conn=None) -> dict | None:
        """Session anhand des Tokens."""
        with self._read(conn) as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE token = %s",
                (token,),
            ).fetchone()
            return dict(row) if row else None

    def get_valid_by_token(self, token: str, *, conn=None) -> dict | None:
        """Session anhand des Tokens, nur wenn nicht abgelaufen.

        Fuer CSRF-Middleware-Lookup: abgelaufene Rows duerfen den Token-Hash
        nicht mehr zurueckgeben, sonst koennte ein CSRF-Vergleich auf einer
        leichen Session bestehen.
        """
        with self._read(conn) as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE token = %s AND expires_at > NOW()",
                (token,),
            ).fetchone()
            return dict(row) if row else None

    def set_csrf_hash(
        self,
        token: str,
        csrf_hash: str,
        *,
        conn=None,
    ) -> bool:
        """Rotiert csrf_token_hash auf einer bereits CSRF-haltigen, unexpired Row.

        Kein Lazy-Create: nur Sessions mit bestehendem csrf_token_hash werden
        rotiert (Spec §6, §7 + ImplScan-Finding 3). Alte NULL-Hash-Rows sollen
        Re-Login erzwingen, statt sich transparent nachzurusten.

        Returns True wenn genau eine passende unexpired Row getroffen wurde.
        False (rowcount == 0) bedeutet: Session existiert nicht mehr, ist
        abgelaufen oder hat keinen csrf_token_hash — Aufrufer darf dann KEINEN
        neuen Klartext-Token ausliefern.
        """
        with self._write(conn) as db:
            cursor = db.execute(
                "UPDATE sessions SET csrf_token_hash = %s "
                "WHERE token = %s "
                "  AND expires_at > NOW() "
                "  AND csrf_token_hash IS NOT NULL",
                (csrf_hash, token),
            )
            return cursor.rowcount > 0

    def delete_by_token(self, token: str, *, conn=None) -> bool:
        """Loescht eine Session anhand des Tokens."""
        return self.delete_where({"token": token}, conn=conn) > 0

    def delete_by_user(
        self,
        user_id: int,
        *,
        except_token: str | None = None,
        conn=None,
    ) -> int:
        """Loescht alle Sessions eines Users, optional eine ausgenommen."""
        with self._write(conn) as db:
            if except_token:
                cursor = db.execute(
                    "DELETE FROM sessions WHERE user_id = %s AND token != %s",
                    (user_id, except_token),
                )
            else:
                cursor = db.execute(
                    "DELETE FROM sessions WHERE user_id = %s",
                    (user_id,),
                )
            return cursor.rowcount

    def cleanup_expired(self, *, conn=None) -> int:
        """Loescht abgelaufene Sessions."""
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM sessions WHERE expires_at < NOW()",
            )
            return cursor.rowcount
