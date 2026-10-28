"""Tests fuer die in der Entwicklungsumgebung deaktivierte Passwort-Policy.

Deckt ab:
  - create_user: Kurzes Passwort wird akzeptiert
  - create_user: Passwort >= 8 Zeichen wird akzeptiert
  - change_password: Kurzes neues Passwort wird akzeptiert
  - change_password: Neues Passwort >= 8 Zeichen wird akzeptiert
"""

from contextlib import contextmanager
from unittest.mock import patch

import bcrypt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection
from kiara.web.api_admin import router as admin_router
from kiara.web.api_auth import router as auth_router
from kiara.web.auth import get_admin_user, get_current_user

# ---------------------------------------------------------------------------
# Hilfsklassen
# ---------------------------------------------------------------------------

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
        created_at TEXT DEFAULT (NOW())
    );
    CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token TEXT NOT NULL UNIQUE,
        csrf_token_hash CHAR(64),
        created_at TEXT DEFAULT (NOW()),
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
"""


def _make_db():
    """Erstellt die Test-Tabellen in PostgreSQL (drop+create fuer Robustheit)."""
    conn = get_connection()
    # Tabellen droppen um Schema-Inkonsistenzen durch andere Tests zu vermeiden
    conn.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS user_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS sessions CASCADE")
    conn.execute("DROP TABLE IF EXISTS groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS users CASCADE")
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()


def _mock_get_db():
    """Factory fuer get_db-Mock, der auf PostgreSQL zeigt."""

    def _get():
        return get_connection()

    return _get


def _make_query_ctx():
    """Erstellt einen db_query Context-Manager fuer PostgreSQL."""

    @contextmanager
    def query_ctx():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    return query_ctx


def _make_tx_ctx():
    """Erstellt einen db_transaction Context-Manager fuer PostgreSQL."""

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

    return tx_ctx


# ===================================================================
# CREATE USER — Passwort-Mindestlaenge
# ===================================================================


class TestCreateUserPasswordPolicy:
    """Tests fuer die Passwort-Mindestlaenge bei User-Erstellung."""

    @pytest.fixture
    def client(self, tmp_path):
        """FastAPI TestClient mit Admin-Router und Mock-Auth."""
        _make_db()

        # Admin-User anlegen (ID=1)
        conn = get_connection()
        conn.execute("DELETE FROM sessions")
        conn.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE")
        pw_hash = bcrypt.hashpw(b"adminpass", bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) "
            "VALUES (%s, %s, %s, %s, %s)",
            ("admin", pw_hash, "Administrator", "", True),
        )
        conn.commit()
        conn.close()

        from kiara.admin.registry import admin_registry, load_admin_functions
        from kiara.admin.router import generate_admin_routes

        load_admin_functions()

        app = FastAPI()
        app.include_router(admin_router)
        generate_admin_routes(app, admin_registry)

        # Admin-Auth ueberspringen
        async def mock_admin():
            return {"id": 1, "username": "admin", "is_admin": True}

        app.dependency_overrides[get_admin_user] = mock_admin

        import kiara_db.base as repo_base
        import kiara_db.repositories.user as repo_user

        import kiara.shared.db as shared_db

        with (
            patch.object(shared_db, "db_query", side_effect=_make_query_ctx()),
            patch.object(shared_db, "db_transaction", side_effect=_make_tx_ctx()),
            patch.object(repo_base, "db_query", side_effect=_make_query_ctx()),
            patch.object(repo_base, "db_transaction", side_effect=_make_tx_ctx()),
            patch.object(repo_user, "db_query", side_effect=_make_query_ctx()),
            patch.object(repo_user, "db_transaction", side_effect=_make_tx_ctx()),
            patch("kiara.shared.db.get_db", _mock_get_db()),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_create_user_short_password_accepted_in_dev(self, client):
        """Passwort 'abc' wird bei deaktivierter Dev-Policy akzeptiert."""
        resp = client.post(
            "/api/admin/users",
            json={
                "username": "shortpw_user",
                "password": "abc",
                "vorname": "Test",
                "nachname": "User",
            },
        )
        assert resp.status_code == 200, f"Erwartet 200, bekam {resp.status_code}: {resp.text}"
        assert resp.json()["message"] == "Benutzer erstellt"

    def test_create_user_valid_password_accepted(self, client):
        """Passwort 'Test1234' (8 Zeichen, 3 Kategorien) wird akzeptiert."""
        resp = client.post(
            "/api/admin/users",
            json={
                "username": "validpw_user",
                "password": "Test1234",
                "vorname": "Test",
                "nachname": "User",
            },
        )
        assert resp.status_code == 200, f"Erwartet 200, bekam {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "id" in data
        assert data["message"] == "Benutzer erstellt"


# ===================================================================
# CHANGE PASSWORD — Passwort-Mindestlaenge
# ===================================================================


class TestChangePasswordPolicy:
    """Tests fuer die Passwort-Mindestlaenge bei Passwortaenderung."""

    @pytest.fixture
    def client(self, tmp_path):
        """FastAPI TestClient mit Auth-Router und eingeloggtem User."""
        _make_db()

        # User mit bekanntem Passwort anlegen
        conn = get_connection()
        conn.execute("DELETE FROM sessions")
        conn.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE")
        pw_hash = bcrypt.hashpw(b"oldpass88", bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) "
            "VALUES (%s, %s, %s, %s, %s)",
            ("testuser", pw_hash, "Test", "User", False),
        )
        conn.commit()
        conn.close()

        app = FastAPI()
        app.include_router(auth_router)

        # User-Auth simulieren
        async def mock_current_user():
            return {
                "id": 1,
                "username": "testuser",
                "is_admin": False,
                "vorname": "Test",
                "nachname": "User",
                "email": "",
                "must_change_password": False,
                "group_ids": [],
                "datasource_ids": [],
            }

        app.dependency_overrides[get_current_user] = mock_current_user

        get_db_fn = _mock_get_db()

        import kiara_db.base as repo_base
        import kiara_db.repositories.config as repo_config
        import kiara_db.repositories.user as repo_user

        import kiara.shared.db as shared_db

        with (
            patch.object(shared_db, "db_query", side_effect=_make_query_ctx()),
            patch.object(shared_db, "db_transaction", side_effect=_make_tx_ctx()),
            patch.object(repo_base, "db_query", side_effect=_make_query_ctx()),
            patch.object(repo_base, "db_transaction", side_effect=_make_tx_ctx()),
            patch.object(repo_user, "db_query", side_effect=_make_query_ctx()),
            patch.object(repo_user, "db_transaction", side_effect=_make_tx_ctx()),
            patch.object(repo_config, "db_query", side_effect=_make_query_ctx()),
            patch.object(repo_config, "db_transaction", side_effect=_make_tx_ctx()),
        ):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_change_password_short_accepted_in_dev(self, client):
        """Ein kurzes neues Passwort wird bei deaktivierter Dev-Policy akzeptiert."""
        resp = client.put(
            "/api/auth/password",
            json={
                "old_password": "oldpass88",
                "new_password": "short",
            },
        )
        assert resp.status_code == 200, f"Erwartet 200, bekam {resp.status_code}: {resp.text}"
        assert resp.json()["message"] == "Passwort geaendert"

    def test_change_password_valid_accepted(self, client):
        """Neues Passwort >= 8 Zeichen mit 3 Kategorien wird akzeptiert."""
        resp = client.put(
            "/api/auth/password",
            json={
                "old_password": "oldpass88",
                "new_password": "NewPass1234",
            },
        )
        assert resp.status_code == 200, f"Erwartet 200, bekam {resp.status_code}: {resp.text}"
        assert resp.json()["message"] == "Passwort geaendert"
