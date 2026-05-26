"""Tests fuer Auth-Tiering: Admin-UI vs. User-UI Zugriffssteuerung.

Stellt sicher, dass die drei Auth-Dependencies korrekt trennen:
- get_current_user: Tier-neutral (jeder angemeldete User)
- get_chat_user:    User-Tier (nur Non-Admins, fuer Chat/Suche/Workflows)
- get_admin_user:   Admin-Tier (nur Admins, fuer Admin-Konfiguration)

Regressionstest fuer den Admin-UI-Redirect-Bug: /api/auth/me wies Admins
mit 403 ab, weil die Chat-Sperre auf der Auth-Ebene statt der Chat-Ebene sass.
"""

import bcrypt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import kiara.web.middleware as _kiara_middleware
from kiara.shared.db import get_connection
from kiara.web.auth import (
    COOKIE_NAME,
    create_session,
    get_admin_user,
    get_chat_user,
    get_current_user,
    hash_token,
    sign_token,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def auth_db():
    """PostgreSQL-DB mit Admin- und Regular-User + benoetigten Tabellen."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS user_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS sessions CASCADE")
    conn.execute("DROP TABLE IF EXISTS groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS users CASCADE")
    conn.executescript("""
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            vorname TEXT DEFAULT '',
            nachname TEXT DEFAULT '',
            email TEXT DEFAULT '',
            is_admin BOOLEAN DEFAULT false,
            is_active BOOLEAN DEFAULT true,
            auth_source TEXT DEFAULT 'local',
            must_change_password BOOLEAN DEFAULT false,
            created_at TEXT DEFAULT (NOW())
        );
        CREATE TABLE sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            csrf_token_hash CHAR(64),
            expires_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE groups (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE user_groups (user_id INTEGER, group_id INTEGER);
        CREATE TABLE datasource_groups (datasource_id INTEGER, group_id INTEGER);
        CREATE TABLE datasource_users (
            id SERIAL PRIMARY KEY,
            datasource_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            UNIQUE(datasource_id, user_id)
        );
    """)

    admin_hash = bcrypt.hashpw(b"admin", bcrypt.gensalt()).decode()
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) "
        "VALUES (%s, %s, %s, %s, %s)",
        ("admin", admin_hash, "Administrator", "", True),
    )

    user_hash = bcrypt.hashpw(b"userpass", bcrypt.gensalt()).decode()
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) "
        "VALUES (%s, %s, %s, %s, %s)",
        ("chatuser", user_hash, "Chat", "User", False),
    )

    conn.commit()

    yield get_connection

    conn.rollback()
    conn.close()

    # Produktions-Schema wiederherstellen (DROP CASCADE hat andere Tabellen zerstoert)
    restore = get_connection()
    try:
        restore.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
        restore.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
        restore.execute("DROP TABLE IF EXISTS user_groups CASCADE")
        restore.execute("DROP TABLE IF EXISTS sessions CASCADE")
        restore.execute("DROP TABLE IF EXISTS groups CASCADE")
        restore.execute("DROP TABLE IF EXISTS users CASCADE")
        from kiara.web.schema import create_tables

        create_tables(restore)
        restore.commit()
    except Exception:
        restore.rollback()
    restore.close()


def _build_app_with_endpoints():
    """Erstellt eine Test-App mit je einem Endpoint pro Tier."""
    app = FastAPI()

    @app.get("/tier-neutral")
    async def tier_neutral(user=Depends(get_current_user)):
        return {"user": user["username"], "is_admin": user["is_admin"]}

    @app.get("/user-only")
    async def user_only(user=Depends(get_chat_user)):
        return {"user": user["username"]}

    @app.get("/admin-only")
    async def admin_only(user=Depends(get_admin_user)):
        return {"user": user["username"]}

    return app


def _login_client(app, token):
    """Erstellt einen TestClient mit gesetztem Session-Cookie."""
    client = TestClient(app)
    signed = sign_token(token)
    client.cookies.set(COOKIE_NAME, signed)
    return client


# ---------------------------------------------------------------------------
# Tier-neutral: get_current_user (Admin + User erlaubt)
# ---------------------------------------------------------------------------


class TestTierNeutral:
    """get_current_user erlaubt sowohl Admins als auch regulaere User."""

    def test_admin_can_access_tier_neutral(self, auth_db):
        """Admin-User erhaelt 200 auf tier-neutralem Endpoint."""
        token, _ = create_session(1)  # admin
        app = _build_app_with_endpoints()
        client = _login_client(app, token)

        resp = client.get("/tier-neutral")
        assert resp.status_code == 200
        assert resp.json()["user"] == "admin"
        assert resp.json()["is_admin"]

    def test_regular_user_can_access_tier_neutral(self, auth_db):
        """Regulaerer User erhaelt 200 auf tier-neutralem Endpoint."""
        token, _ = create_session(2)  # chatuser
        app = _build_app_with_endpoints()
        client = _login_client(app, token)

        resp = client.get("/tier-neutral")
        assert resp.status_code == 200
        assert resp.json()["user"] == "chatuser"
        assert not resp.json()["is_admin"]

    def test_unauthenticated_gets_401(self, auth_db):
        """Ohne Session-Cookie erhaelt man 401."""
        app = _build_app_with_endpoints()
        client = TestClient(app)

        resp = client.get("/tier-neutral")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# User-Tier: get_chat_user (nur Non-Admins)
# ---------------------------------------------------------------------------


class TestUserTier:
    """get_chat_user erlaubt nur regulaere User, weist Admins mit 403 ab."""

    def test_regular_user_can_access_chat(self, auth_db):
        """Regulaerer User erhaelt 200 auf User-Tier-Endpoint."""
        token, _ = create_session(2)  # chatuser
        app = _build_app_with_endpoints()
        client = _login_client(app, token)

        resp = client.get("/user-only")
        assert resp.status_code == 200
        assert resp.json()["user"] == "chatuser"

    def test_admin_rejected_from_chat(self, auth_db):
        """Admin-User erhaelt 403 auf User-Tier-Endpoint."""
        token, _ = create_session(1)  # admin
        app = _build_app_with_endpoints()
        client = _login_client(app, token)

        resp = client.get("/user-only")
        assert resp.status_code == 403
        assert "Chat-Zugriff" in resp.json()["detail"]

    def test_unauthenticated_gets_401(self, auth_db):
        """Ohne Session erhaelt man 401 (nicht 403)."""
        app = _build_app_with_endpoints()
        client = TestClient(app)

        resp = client.get("/user-only")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Admin-Tier: get_admin_user (nur Admins)
# ---------------------------------------------------------------------------


class TestAdminTier:
    """get_admin_user erlaubt nur Admins, weist regulaere User mit 403 ab."""

    def test_admin_can_access_admin_endpoint(self, auth_db):
        """Admin-User erhaelt 200 auf Admin-Tier-Endpoint."""
        token, _ = create_session(1)  # admin
        app = _build_app_with_endpoints()
        client = _login_client(app, token)

        resp = client.get("/admin-only")
        assert resp.status_code == 200
        assert resp.json()["user"] == "admin"

    def test_regular_user_rejected_from_admin(self, auth_db):
        """Regulaerer User erhaelt 403 auf Admin-Tier-Endpoint."""
        token, _ = create_session(2)  # chatuser
        app = _build_app_with_endpoints()
        client = _login_client(app, token)

        resp = client.get("/admin-only")
        assert resp.status_code == 403

    def test_unauthenticated_gets_401(self, auth_db):
        """Ohne Session erhaelt man 401 (nicht 403)."""
        app = _build_app_with_endpoints()
        client = TestClient(app)

        resp = client.get("/admin-only")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Regressionstests: Admin-UI Endpoints (api_auth.py)
# ---------------------------------------------------------------------------


class TestAdminUiRegression:
    """Regressionstest: Admin kann /api/auth/me und /api/auth/password nutzen.

    Vor dem Fix gab /api/auth/me 403 fuer Admins zurueck, weil get_current_user
    die Chat-Sperre enthielt. Das fuehrte dazu, dass die Admin-UI nach dem Login
    sofort zur Anmeldeseite zurueck-redirectete.
    """

    def test_admin_can_call_auth_me(self, auth_db):
        """Admin-User erhaelt 200 auf /api/auth/me (Regressions-Bug)."""
        from kiara.web.api_auth import router as auth_router

        token, _ = create_session(1)  # admin
        app = FastAPI()
        app.include_router(auth_router)
        client = _login_client(app, token)

        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.json()["user"]
        assert data["username"] == "admin"
        assert data["is_admin"]

    def test_regular_user_can_call_auth_me(self, auth_db):
        """Regulaerer User erhaelt ebenfalls 200 auf /api/auth/me."""
        from kiara.web.api_auth import router as auth_router

        token, _ = create_session(2)  # chatuser
        app = FastAPI()
        app.include_router(auth_router)
        client = _login_client(app, token)

        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.json()["user"]
        assert data["username"] == "chatuser"
        assert not data["is_admin"]

    def test_csrf_refresh_invalid_session_returns_code(self, auth_db):
        """GET /api/auth/csrf liefert auch im Dependency-401 den kanonischen Body."""
        from kiara.web.api_auth import router as auth_router

        token, _ = create_session(1)
        conn = get_connection()
        conn.execute(
            "UPDATE sessions SET csrf_token_hash = NULL WHERE token = %s",
            (hash_token(token),),
        )
        conn.commit()
        conn.close()

        app = FastAPI()
        original_add_csrf = getattr(
            _kiara_middleware,
            "_ORIGINAL_ADD_CSRF_MIDDLEWARE",
            _kiara_middleware.add_csrf_middleware,
        )
        original_add_csrf(
            app,
            exempt_prefixes=[],
            token_exempt_paths={"/api/auth/login"},
            trusted_origins=["http://localhost"],
        )
        app.include_router(auth_router)
        client = _login_client(app, token)

        resp = client.get("/api/auth/csrf")
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Nicht angemeldet", "code": "not_authenticated"}
