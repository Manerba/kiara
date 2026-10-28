"""WOPI-Repository (Tokens + Locks).

Verwaltet: wopi_tokens, wopi_locks.
Erbt NICHT von BaseRepository (2 Tabellen).

Ersetzt SQL in: web/api_onlyoffice.py (12 execute-Aufrufe)
"""

from __future__ import annotations

from contextlib import contextmanager

from kiara_db.connection import db_query, db_transaction


class WopiRepository:
    """Verwaltet OnlyOffice WOPI-Tokens und Dokument-Locks."""

    @contextmanager
    def _read(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_query() as db:
                yield db

    @contextmanager
    def _write(self, conn=None):
        if conn is not None:
            yield conn
        else:
            with db_transaction() as db:
                yield db

    # ----- Tokens -----

    def create_token(
        self,
        token: str,
        file_id: str,
        user_id: int,
        expires_at: str,
        file_path: str | None = None,
        *,
        conn=None,
    ) -> None:
        with self._write(conn) as db:
            db.execute(
                """INSERT INTO wopi_tokens (token, file_id, file_path, user_id, expires_at)
                   VALUES (%s, %s, %s, %s, %s)""",
                (token, file_id, file_path, user_id, expires_at),
            )

    def get_token(self, token: str, *, conn=None) -> dict | None:
        with self._read(conn) as db:
            row = db.execute(
                "SELECT * FROM wopi_tokens WHERE token = %s",
                (token,),
            ).fetchone()
            return dict(row) if row else None

    def cleanup_tokens(self, *, conn=None) -> int:
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM wopi_tokens WHERE expires_at < NOW()",
            )
            return cursor.rowcount

    # ----- Locks -----

    def get_lock(self, file_id: str, *, for_update: bool = False, conn=None) -> dict | None:
        with self._read(conn) as db:
            sql = "SELECT * FROM wopi_locks WHERE file_id = %s"
            if for_update:
                sql += " FOR UPDATE"
            row = db.execute(sql, (file_id,)).fetchone()
            return dict(row) if row else None

    def set_lock(
        self,
        file_id: str,
        lock_id: str,
        user_id: int,
        expires_at: str,
        *,
        conn=None,
    ) -> None:
        with self._write(conn) as db:
            db.execute(
                """INSERT INTO wopi_locks (file_id, lock_id, user_id, expires_at)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (file_id) DO UPDATE SET
                       lock_id = EXCLUDED.lock_id,
                       user_id = EXCLUDED.user_id,
                       expires_at = EXCLUDED.expires_at""",
                (file_id, lock_id, user_id, expires_at),
            )

    def release_lock(self, file_id: str, *, conn=None) -> bool:
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM wopi_locks WHERE file_id = %s",
                (file_id,),
            )
            return cursor.rowcount > 0

    def cleanup_locks(self, *, conn=None) -> int:
        with self._write(conn) as db:
            cursor = db.execute(
                "DELETE FROM wopi_locks WHERE expires_at < NOW()",
            )
            return cursor.rowcount
