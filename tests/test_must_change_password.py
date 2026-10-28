"""Tests fuer F-07 — must_change_password Flag.

Deckt ab:
  - Default-Admin wird mit must_change_password=True erstellt
  - Login-Response enthaelt must_change_password-Feld
  - Passwortaenderung setzt must_change_password auf 0
"""

from contextlib import contextmanager
from unittest.mock import patch

import bcrypt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Hilfsklassen
# ---------------------------------------------------------------------------
from kiara.shared.db import get_connection
from kiara.web.api_auth import router as auth_router
from kiara.web.auth import get_current_user

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT,
        vorname TEXT DEFAULT '',
        nachname TEXT DEFAULT '',
        email TEXT DEFAULT '',
        is_admin BOOLEAN DEFAULT false,
        auth_source TEXT DEFAULT 'local',
        ldap_dn TEXT DEFAULT '',
        is_active BOOLEAN DEFAULT true,
        must_change_password BOOLEAN DEFAULT false,
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS'))
    );
    CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token TEXT NOT NULL UNIQUE,
        csrf_token_hash CHAR(64),
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS')),
        expires_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE IF NOT EXISTS groups (
        id SERIAL PRIMARY KEY,
        cn TEXT NOT NULL,
        dn TEXT NOT NULL UNIQUE,
        member_count INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS user_groups (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL,
        group_id INTEGER NOT NULL,
        UNIQUE(user_id, group_id)
    );
    CREATE TABLE IF NOT EXISTS datasource_groups (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        group_id INTEGER NOT NULL,
        UNIQUE(datasource_id, group_id)
    );
    CREATE TABLE IF NOT EXISTS datasource_users (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        UNIQUE(datasource_id, user_id)
    );
    CREATE TABLE IF NOT EXISTS ai_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        embedding_url TEXT NOT NULL DEFAULT 'http://localhost:11434',
        embedding_model TEXT NOT NULL DEFAULT 'nomic-embed-text',
        llm_url TEXT NOT NULL DEFAULT 'http://localhost:11434',
        llm_model TEXT NOT NULL DEFAULT 'qwen3:8b',
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS'))
    );
"""


def _setup_schema(conn):
    """Erstellt das Test-Schema in PostgreSQL (drop+create fuer Robustheit)."""
    # Tabellen droppen (abhaengige zuerst) um Schema-Inkonsistenzen durch andere Tests zu vermeiden
    conn.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS user_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS sessions CASCADE")
    conn.execute("DROP TABLE IF EXISTS groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS users CASCADE")
    conn.executescript(_SCHEMA)
    conn.commit()


def _cleanup_data(conn):
    """Loescht alle Testdaten aus den Tabellen (FK-sichere Reihenfolge)."""
    conn.execute("DELETE FROM datasource_users")
    conn.execute("DELETE FROM datasource_groups")
    conn.execute("DELETE FROM user_groups")
    conn.execute("DELETE FROM sessions")
    # Truncate users mit CASCADE um alle FK-abhaengigen Zeilen zu entfernen
    conn.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE")
    conn.commit()


def _mock_get_db():
    """Factory fuer get_db-Mock, der eine PG-Verbindung zurueckgibt."""

    def _get():
        return get_connection()

    return _get


# ---------------------------------------------------------------------------
# Test: Default-Admin hat must_change_password=True
# ---------------------------------------------------------------------------


class TestDefaultAdminMustChangePassword:
    """Prueft dass init_db() den Admin mit must_change_password=True erstellt."""

    def test_default_admin_has_must_change_password_flag(self):
        """Nach init_db() hat der Default-Admin must_change_password=True."""
        mock_config = {
            "ollama": {
                "url": "http://localhost:11434",
                "embedding_model": "nomic-embed-text",
                "llm_model": "qwen3:8b",
            }
        }

        with patch("kiara.config.load_config", return_value=mock_config):
            from kiara.web.database import init_db

            init_db()

        # Ergebnis pruefen
        conn = get_connection()
        row = conn.execute(
            "SELECT must_change_password FROM users WHERE username = 'admin'"
        ).fetchone()
        conn.close()

        assert row is not None, "Admin-User wurde nicht erstellt"
        assert row["must_change_password"], (
            "must_change_password sollte 1 sein fuer den Default-Admin"
        )


# ---------------------------------------------------------------------------
# Test: Login-Response enthaelt must_change_password
# ---------------------------------------------------------------------------


class TestLoginResponseMustChangePassword:
    """Prueft dass die Login-Response das must_change_password-Feld enthaelt."""

    @pytest.fixture
    def client(self):
        """FastAPI TestClient mit Auth-Router und PostgreSQL-DB."""
        conn = get_connection()
        _setup_schema(conn)
        _cleanup_data(conn)

        # Admin mit must_change_password=True erstellen
        pw_hash = bcrypt.hashpw(b"admin123pw", bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, password_hash, vorname, nachname, is_admin, must_change_password) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            ("admin", pw_hash, "Administrator", "", True, True),
        )
        conn.commit()
        conn.close()

        app = FastAPI()
        app.include_router(auth_router)

        get_db_fn = _mock_get_db()

        @contextmanager
        def query_ctx():
            c = get_connection()
            try:
                yield c
            finally:
                c.close()

        @contextmanager
        def tx_ctx():
            c = get_connection()
            try:
                yield c
                c.commit()
            except Exception:
                c.rollback()
                raise
            finally:
                c.close()

        import kiara_db.base as repo_base
        import kiara_db.repositories.config as repo_config
        import kiara_db.repositories.user as repo_user

        import kiara.shared.db as shared_db
        import kiara.web.auth as auth_mod

        with (
            patch.object(shared_db, "db_query", side_effect=query_ctx),
            patch.object(shared_db, "db_transaction", side_effect=tx_ctx),
            patch.object(repo_base, "db_query", side_effect=query_ctx),
            patch.object(repo_base, "db_transaction", side_effect=tx_ctx),
            patch.object(repo_user, "db_query", side_effect=query_ctx),
            patch.object(repo_user, "db_transaction", side_effect=tx_ctx),
            patch.object(repo_config, "db_query", side_effect=query_ctx),
            patch.object(repo_config, "db_transaction", side_effect=tx_ctx),
            patch.object(auth_mod, "db_query", side_effect=query_ctx),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

        # Cleanup
        conn = get_connection()
        _cleanup_data(conn)
        conn.close()

    def test_login_response_includes_must_change_password(self, client):
        """Login-Response JSON enthaelt das must_change_password-Feld."""
        resp = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin123pw"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200, f"Login fehlgeschlagen: {resp.text}"
        data = resp.json()
        assert "user" in data
        assert "must_change_password" in data["user"], (
            "must_change_password fehlt in der Login-Response"
        )
        assert data["user"]["must_change_password"]


# ---------------------------------------------------------------------------
# Test: Passwortaenderung setzt must_change_password auf 0
# ---------------------------------------------------------------------------


class TestChangePasswordResetsFlag:
    """Prueft dass nach Passwortaenderung must_change_password=False ist."""

    @pytest.fixture
    def client_and_db(self):
        """FastAPI TestClient mit Auth-Router."""
        conn = get_connection()
        _setup_schema(conn)
        _cleanup_data(conn)

        # Nicht-Admin User mit must_change_password=True erstellen
        pw_hash = bcrypt.hashpw(b"oldpass88", bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, password_hash, vorname, nachname, is_admin, must_change_password) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            ("testuser", pw_hash, "Test", "User", False, True),
        )
        conn.commit()
        # Ermittle die tatsaechliche ID
        row = conn.execute("SELECT id FROM users WHERE username = 'testuser'").fetchone()
        user_id = row["id"]
        conn.close()

        app = FastAPI()
        app.include_router(auth_router)

        # get_current_user override: simuliere eingeloggten User
        async def mock_current_user():
            return {
                "id": user_id,
                "username": "testuser",
                "is_admin": False,
                "vorname": "Test",
                "nachname": "User",
                "email": "",
                "must_change_password": True,
                "group_ids": [],
                "datasource_ids": [],
            }

        app.dependency_overrides[get_current_user] = mock_current_user

        get_db_fn = _mock_get_db()

        @contextmanager
        def query_ctx():
            c = get_connection()
            try:
                yield c
            finally:
                c.close()

        @contextmanager
        def tx_ctx():
            c = get_connection()
            try:
                yield c
                c.commit()
            except Exception:
                c.rollback()
                raise
            finally:
                c.close()

        import kiara_db.base as repo_base
        import kiara_db.repositories.config as repo_config
        import kiara_db.repositories.user as repo_user

        import kiara.shared.db as shared_db
        import kiara.web.auth as auth_mod

        with (
            patch.object(shared_db, "db_query", side_effect=query_ctx),
            patch.object(shared_db, "db_transaction", side_effect=tx_ctx),
            patch.object(repo_base, "db_query", side_effect=query_ctx),
            patch.object(repo_base, "db_transaction", side_effect=tx_ctx),
            patch.object(repo_user, "db_query", side_effect=query_ctx),
            patch.object(repo_user, "db_transaction", side_effect=tx_ctx),
            patch.object(repo_config, "db_query", side_effect=query_ctx),
            patch.object(repo_config, "db_transaction", side_effect=tx_ctx),
            patch.object(auth_mod, "db_query", side_effect=query_ctx),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

        # Cleanup
        conn = get_connection()
        _cleanup_data(conn)
        conn.close()

    def test_change_password_resets_flag(self, client_and_db):
        """Nach Passwortaenderung ist must_change_password=False in der DB."""
        client = client_and_db

        resp = client.put(
            "/api/auth/password",
            json={"old_password": "oldpass88", "new_password": "NewPass1234"},
        )
        assert resp.status_code == 200, f"Passwortaenderung fehlgeschlagen: {resp.text}"

        # DB pruefen
        conn = get_connection()
        row = conn.execute(
            "SELECT must_change_password FROM users WHERE username = 'testuser'"
        ).fetchone()
        conn.close()

        assert row is not None
        assert not row["must_change_password"], (
            "must_change_password sollte nach Passwortaenderung 0 sein"
        )
