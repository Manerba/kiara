"""Tests fuer die CSRF-Middleware aus kiara.web.middleware.

Deckt die Plan-Testfaelle aus TODO_BUGFIX_CSRF_DEFENSE_IN_DEPTH_V18.md ab:
Synchronizer-Token-Check, Origin-/Referer-Validation, X-Requested-With als
Zusatzschicht, Token-Exempts, Session-Rotation und Streaming-Retry-Ausschluss.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import kiara.web.middleware as _kiara_middleware
from kiara.shared.db import get_connection
from kiara.web.auth import (
    create_session,
    hash_password,
    hash_token,
    set_cookie_name,
    sign_token,
    unsign_token,
    verify_password,
)
from kiara.web.cors import compute_trusted_origins

TRUSTED_ORIGINS = ["http://localhost:8502", "http://127.0.0.1:8502"]

# conftest.py hat add_csrf_middleware durch No-Op ersetzt, damit der Rest der
# Test-Suite ohne CSRF-Token arbeiten kann. Diese Datei testet explizit die
# echte Middleware — also hier den Original-Verweis holen.
add_csrf_middleware = getattr(
    _kiara_middleware,
    "_ORIGINAL_ADD_CSRF_MIDDLEWARE",
    _kiara_middleware.add_csrf_middleware,
)
add_security_headers_middleware = _kiara_middleware.add_security_headers_middleware


@pytest.fixture(autouse=True)
def _enable_csrf():
    """Setzt den Cookie-Namen auf den Default zurueck.

    Notwendig, weil `test_api_*`-Module die admin_app importieren und damit
    `set_cookie_name("kiara_admin_session")` prozess-global auslosen
    (siehe Plan V7 §13 Cookie-Name-Constraint).
    """
    set_cookie_name("kiara_session")


@pytest.fixture()
def csrf_app(auth_db):
    """App mit CSRF-Middleware und drei Minimal-Routes fuer POST/GET/Login."""
    app = FastAPI()
    add_csrf_middleware(
        app,
        exempt_prefixes=["/api/wopi/"],
        token_exempt_paths={"/api/auth/login"},
        trusted_origins=TRUSTED_ORIGINS,
    )

    @app.post("/api/foo")
    async def foo():
        return {"ok": True}

    @app.get("/api/foo")
    async def foo_get():
        return {"ok": True}

    @app.post("/api/auth/login")
    async def login():
        return {"ok": "login"}

    @app.post("/api/wopi/callback")
    async def wopi():
        return {"ok": "wopi"}

    return app


@pytest.fixture()
def csrf_security_app(auth_db):
    """App mit echter CSRF- und Security-Header-Middleware."""
    app = FastAPI()
    add_csrf_middleware(
        app,
        exempt_prefixes=["/api/wopi/"],
        token_exempt_paths={"/api/auth/login"},
        trusted_origins=TRUSTED_ORIGINS,
    )
    add_security_headers_middleware(app, csp_strict=True, hsts_max_age=3600)

    @app.post("/api/foo")
    async def foo():
        return {"ok": True}

    return app


@pytest.fixture()
def auth_db():
    """PostgreSQL-Tabellen fuer Session-basierte CSRF-Tests."""
    conn = get_connection()
    for t in (
        "datasource_users",
        "datasource_groups",
        "user_groups",
        "sessions",
        "groups",
        "users",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
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
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token TEXT UNIQUE NOT NULL,
            csrf_token_hash CHAR(64),
            expires_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE IF NOT EXISTS groups (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_groups (
            user_id INTEGER,
            group_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS datasource_groups (
            datasource_id INTEGER,
            group_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS datasource_users (
            datasource_id INTEGER,
            user_id INTEGER
        );
    """)
    conn.execute(
        "INSERT INTO users (username, password_hash, is_active, is_admin) "
        "VALUES (%s, %s, true, false)",
        ("alice", hash_password("OldPass1234")),
    )
    conn.execute(
        "INSERT INTO users (username, password_hash, is_active, is_admin) "
        "VALUES (%s, %s, true, true)",
        ("admin", hash_password("AdminPass1234")),
    )
    conn.commit()
    user_row = conn.execute("SELECT id FROM users WHERE username='alice'").fetchone()
    admin_row = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()
    conn.close()
    yield {
        "user_id": user_row["id"],
        "username": "alice",
        "password": "OldPass1234",
        "admin_user_id": admin_row["id"],
        "admin_username": "admin",
        "admin_password": "AdminPass1234",
    }
    # Truncate-Cleanup: test_must_change_password::TestDefaultAdminMustChangePassword
    # ruft init_db()+run_seeds() ohne eigenen Setup und erwartet eine leere
    # users-Tabelle. Ohne Cleanup bleibt hier `alice` liegen → ensure_admin_user
    # ueberspringt die Admin-Anlage, Folgetest scheitert. DROP wuerde init_db()
    # mit schema-Inkonsistenz brechen (Tabellen-Count > 0, aber users fehlt),
    # deshalb TRUNCATE CASCADE (FK-Schichten bleiben strukturell).
    conn = get_connection()
    conn.execute("TRUNCATE TABLE sessions RESTART IDENTITY CASCADE")
    conn.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()


@pytest.fixture()
def auth_router_app(auth_db):
    """Separate App mit echter CSRF-Middleware und echtem Auth-Router."""
    from kiara.web.api_auth import router as auth_router

    app = FastAPI()
    add_csrf_middleware(
        app,
        token_exempt_paths={"/api/auth/login"},
        trusted_origins=TRUSTED_ORIGINS,
    )
    app.include_router(auth_router)
    return app


@pytest.fixture()
def admin_chat_csrf_app(auth_db):
    """Admin-Chat-App-Reihenfolge: CSRF innen, CORS aussen."""
    from kiara.web.api_admin_chat import router as admin_chat_router

    admin_origin = "http://localhost:8502"
    set_cookie_name("kiara_admin_session")
    app = FastAPI()
    add_csrf_middleware(
        app,
        token_exempt_paths={"/api/auth/login"},
        trusted_origins=[admin_origin],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[admin_origin],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-Requested-With", "X-CSRF-Token"],
    )
    app.include_router(admin_chat_router)
    yield app
    set_cookie_name("kiara_session")


@pytest.fixture()
def graph_csrf_app(auth_db):
    """Graph-Admin-Router mit echter CSRF-Middleware."""
    from kiara.web.api_graph import router as graph_router
    from kiara.web.auth import get_admin_user

    app = FastAPI()
    add_csrf_middleware(
        app,
        token_exempt_paths={"/api/auth/login"},
        trusted_origins=TRUSTED_ORIGINS,
    )

    async def admin_user():
        return {"id": auth_db["admin_user_id"], "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = admin_user
    app.include_router(graph_router)
    return app


def _auth_cookies(token):
    return {"kiara_session": sign_token(token)}


def _client_with_cookie(app, token):
    """TestClient mit Session-Cookie auf Client-Instanz (httpx-Best-Practice)."""
    client = TestClient(app)
    client.cookies.set("kiara_session", sign_token(token))
    return client


def _client_with_named_cookie(app, token, cookie_name="kiara_session"):
    client = TestClient(app)
    client.cookies.set(cookie_name, sign_token(token))
    return client


def _std_headers(origin="http://localhost:8502"):
    return {"Origin": origin, "X-Requested-With": "XMLHttpRequest"}


def _session_row(token):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM sessions WHERE token = %s",
        (hash_token(token),),
    ).fetchone()
    conn.close()
    return row


def _session_rows_for_user(user_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE user_id = %s ORDER BY id",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


def _password_hash_for_user(user_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE id = %s",
        (user_id,),
    ).fetchone()
    conn.close()
    return row["password_hash"]


def _login(client, auth_db):
    resp = client.post(
        "/api/auth/login",
        json={"username": auth_db["username"], "password": auth_db["password"]},
        headers=_std_headers(),
    )
    assert resp.status_code == 200, resp.text
    signed = resp.cookies.get("kiara_session") or client.cookies.get("kiara_session")
    token = unsign_token(signed)
    assert token
    return resp, token, resp.json()["csrf_token"]


# ---------------------------------------------------------------------------
# 1. Fehlender X-CSRF-Token → 403 csrf_invalid
# ---------------------------------------------------------------------------


def test_missing_csrf_token_returns_403(csrf_app, auth_db):
    token, _ = create_session(auth_db["user_id"])
    client = _client_with_cookie(csrf_app, token)
    resp = client.post("/api/foo", headers=_std_headers())
    assert resp.status_code == 403
    assert resp.json().get("code") == "csrf_invalid"


def test_csrf_403_includes_security_headers(csrf_security_app, auth_db):
    token, _ = create_session(auth_db["user_id"])
    client = _client_with_cookie(csrf_security_app, token)
    resp = client.post("/api/foo", headers=_std_headers())
    assert resp.status_code == 403
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Strict-Transport-Security"].startswith("max-age=3600")
    assert "Content-Security-Policy" not in resp.headers


def test_csrf_401_includes_security_headers(csrf_security_app):
    with TestClient(csrf_security_app) as client:
        resp = client.post("/api/foo", headers={**_std_headers(), "X-CSRF-Token": "deadbeef"})
    assert resp.status_code == 401
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Strict-Transport-Security"].startswith("max-age=3600")
    assert "Content-Security-Policy" not in resp.headers


def test_sync_erp_requires_csrf_token(graph_csrf_app, auth_db):
    token, _ = create_session(auth_db["admin_user_id"])
    client = _client_with_cookie(graph_csrf_app, token)

    resp = client.post("/api/admin/graph/sync-erp", headers=_std_headers())

    assert resp.status_code == 403
    assert resp.json() == {
        "detail": "CSRF-Token fehlt oder ungueltig",
        "code": "csrf_invalid",
    }


def test_sync_erp_accepts_valid_csrf_token(graph_csrf_app, auth_db):
    token, csrf_plain = create_session(auth_db["admin_user_id"])
    client = _client_with_cookie(graph_csrf_app, token)
    headers = {**_std_headers(), "X-CSRF-Token": csrf_plain}

    with (
        patch("kiara.web.api_graph._get_graph_store"),
        patch("kiara.graph.erp_sync.ERPGraphSync") as mock_sync,
    ):
        mock_sync.return_value.sync_all = AsyncMock(
            return_value={
                "nodes_created": 0,
                "edges_created": 0,
                "duration_ms": 0,
            }
        )
        resp = client.post("/api/admin/graph/sync-erp", headers=headers)

    assert resp.status_code == 200
    assert resp.json()["message"] == "ERP-Sync abgeschlossen"


# 2. Ungueltiges Token (Hash-Mismatch) → 403
def test_invalid_csrf_token_returns_403(csrf_app, auth_db):
    token, _ = create_session(auth_db["user_id"])
    client = _client_with_cookie(csrf_app, token)
    headers = {**_std_headers(), "X-CSRF-Token": "deadbeef" * 8}
    resp = client.post("/api/foo", headers=headers)
    assert resp.status_code == 403
    assert resp.json().get("code") == "csrf_invalid"


# 3. Gueltiges Token aber fehlendes Origin/Referer → 403
def test_missing_origin_returns_403(csrf_app, auth_db):
    token, csrf_plain = create_session(auth_db["user_id"])
    client = _client_with_cookie(csrf_app, token)
    resp = client.post(
        "/api/foo",
        headers={"X-Requested-With": "XMLHttpRequest", "X-CSRF-Token": csrf_plain},
    )
    assert resp.status_code == 403


# 4. Gueltiges Token + Origin → 200
def test_valid_token_and_origin(csrf_app, auth_db):
    token, csrf_plain = create_session(auth_db["user_id"])
    client = _client_with_cookie(csrf_app, token)
    headers = {**_std_headers(), "X-CSRF-Token": csrf_plain}
    resp = client.post("/api/foo", headers=headers)
    assert resp.status_code == 200


# 5. Referer-Fallback (kein Origin)
def test_referer_fallback_accepted(csrf_app, auth_db):
    token, csrf_plain = create_session(auth_db["user_id"])
    client = _client_with_cookie(csrf_app, token)
    headers = {
        "Referer": "http://localhost:8502/some/path",
        "X-Requested-With": "XMLHttpRequest",
        "X-CSRF-Token": csrf_plain,
    }
    resp = client.post("/api/foo", headers=headers)
    assert resp.status_code == 200


# 6. /api/auth/login ist token-exempt (aber Origin + XRW pflicht)
def test_login_is_token_exempt(csrf_app):
    with TestClient(csrf_app) as client:
        resp = client.post("/api/auth/login", headers=_std_headers())
    assert resp.status_code == 200


# 7. GET-Request ohne Token → 200
def test_get_without_token(csrf_app):
    with TestClient(csrf_app) as client:
        resp = client.get("/api/foo")
    assert resp.status_code == 200


# 8. Login legt Session und CSRF-Hash atomar an
def test_fall_8_login_creates_session_and_csrf_hash_atomically(auth_router_app, auth_db):
    with TestClient(auth_router_app, raise_server_exceptions=False) as client:
        resp, token, csrf_token = _login(client, auth_db)

    data = resp.json()
    assert data["csrf_token"] == csrf_token
    assert resp.headers["cache-control"] == "no-store"
    rows = _session_rows_for_user(auth_db["user_id"])
    assert len(rows) == 1
    row = _session_row(token)
    assert row is not None
    assert row["csrf_token_hash"]
    assert row["csrf_token_hash"].strip() == hash_token(csrf_token)

    conn = get_connection()
    conn.execute("TRUNCATE TABLE sessions RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()

    def _create_then_fail(user_id, *, conn=None):
        conn.execute(
            "INSERT INTO sessions (user_id, token, csrf_token_hash, expires_at) "
            "VALUES (%s, %s, %s, to_char(NOW() + INTERVAL '1 hour', 'YYYY-MM-DD HH24:MI:SS'))",
            (user_id, hash_token("rolled-back-session"), hash_token("rolled-back-csrf")),
        )
        raise RuntimeError("simulated create_session failure")

    with patch("kiara.web.api_auth.create_session", side_effect=_create_then_fail):
        with TestClient(auth_router_app, raise_server_exceptions=False) as client:
            failed = client.post(
                "/api/auth/login",
                json={"username": auth_db["username"], "password": auth_db["password"]},
                headers=_std_headers(),
            )

    assert failed.status_code == 500
    assert _session_rows_for_user(auth_db["user_id"]) == []


# 9. Logout loescht Session inklusive CSRF-Hash
def test_fall_9_logout_deletes_session_and_cookie(auth_router_app, auth_db):
    with TestClient(auth_router_app) as client:
        _, token, csrf_token = _login(client, auth_db)
        resp = client.post(
            "/api/auth/logout",
            headers={**_std_headers(), "X-CSRF-Token": csrf_token},
        )

    assert resp.status_code == 200
    assert _session_row(token) is None
    set_cookie = resp.headers.get("set-cookie", "")
    assert "kiara_session=" in set_cookie
    assert "Max-Age=0" in set_cookie or "expires=" in set_cookie.lower()


# 11. GET /api/auth/csrf rotiert und ist no-store
def test_fall_11_csrf_endpoint_rotates_token_and_invalidates_old(auth_router_app, auth_db):
    with TestClient(auth_router_app) as client:
        _, token, old_csrf = _login(client, auth_db)
        old_hash = _session_row(token)["csrf_token_hash"].strip()

        refresh = client.get("/api/auth/csrf")
        assert refresh.status_code == 200
        assert refresh.headers["cache-control"] == "no-store"
        new_csrf = refresh.json()["csrf_token"]

        row = _session_row(token)
        assert row["csrf_token_hash"].strip() != old_hash
        assert row["csrf_token_hash"].strip() == hash_token(new_csrf)

        old_logout = client.post(
            "/api/auth/logout",
            headers={**_std_headers(), "X-CSRF-Token": old_csrf},
        )
        assert old_logout.status_code == 403
        assert old_logout.json()["code"] == "csrf_invalid"
        assert _session_row(token) is not None

        new_logout = client.post(
            "/api/auth/logout",
            headers={**_std_headers(), "X-CSRF-Token": new_csrf},
        )
        assert new_logout.status_code == 200
        assert _session_row(token) is None


# 14. PUT /api/auth/password rotiert CSRF und behaelt aktuelle Session
def test_fall_14_password_change_rotates_csrf_and_preserves_current_session(
    auth_router_app,
    auth_db,
):
    with TestClient(auth_router_app) as client:
        _, current_token, old_csrf = _login(client, auth_db)
        other_token, _ = create_session(auth_db["user_id"])
        old_password_hash = _password_hash_for_user(auth_db["user_id"])

        resp = client.put(
            "/api/auth/password",
            json={
                "old_password": auth_db["password"],
                "new_password": "NewPass1234",
            },
            headers={**_std_headers(), "X-CSRF-Token": old_csrf},
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["cache-control"] == "no-store"
        new_csrf = resp.json()["csrf_token"]

        new_password_hash = _password_hash_for_user(auth_db["user_id"])
        assert new_password_hash != old_password_hash
        assert verify_password("NewPass1234", new_password_hash)
        assert not verify_password(auth_db["password"], new_password_hash)

        rows = _session_rows_for_user(auth_db["user_id"])
        assert len(rows) == 1
        assert rows[0]["token"] == hash_token(current_token)
        assert _session_row(other_token) is None

        old_logout = client.post(
            "/api/auth/logout",
            headers={**_std_headers(), "X-CSRF-Token": old_csrf},
        )
        assert old_logout.status_code == 403
        assert old_logout.json()["code"] == "csrf_invalid"
        assert _session_row(current_token) is not None

        new_logout = client.post(
            "/api/auth/logout",
            headers={**_std_headers(), "X-CSRF-Token": new_csrf},
        )
        assert new_logout.status_code == 200
        assert _session_row(current_token) is None


# 18. Admin-Chat Cross-Origin-403 ist fuer erlaubte Origin lesbar
def test_fall_18_admin_chat_csrf_403_is_readable_for_allowed_origin(
    admin_chat_csrf_app,
    auth_db,
):
    admin_origin = "http://localhost:8502"
    token, _ = create_session(auth_db["admin_user_id"])
    client = _client_with_named_cookie(
        admin_chat_csrf_app,
        token,
        cookie_name="kiara_admin_session",
    )

    resp = client.post(
        "/api/admin-chat/chats",
        json={},
        headers={
            "Origin": admin_origin,
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRF-Token": "wrong-token",
        },
    )

    assert resp.status_code == 403
    assert resp.json()["code"] == "csrf_invalid"
    assert resp.headers.get("access-control-allow-origin") == admin_origin
    assert resp.headers.get("access-control-allow-credentials") == "true"


# 23. set_csrf_hash-Race liefert 401 und rollt zurueck
def test_fall_23_set_csrf_hash_race_returns_401_and_rolls_back(
    auth_router_app,
    auth_db,
):
    with TestClient(auth_router_app) as client:
        _, current_token, csrf_token = _login(client, auth_db)
        original_csrf_hash = _session_row(current_token)["csrf_token_hash"].strip()

        with patch("kiara.web.api_auth.session_repo.set_csrf_hash", return_value=False):
            refresh = client.get("/api/auth/csrf")

        assert refresh.status_code == 401
        assert refresh.json() == {"detail": "Nicht angemeldet", "code": "not_authenticated"}
        assert "csrf_token" not in refresh.json()
        assert _session_row(current_token)["csrf_token_hash"].strip() == original_csrf_hash

        other_token, _ = create_session(auth_db["user_id"])
        tokens_before = {row["token"] for row in _session_rows_for_user(auth_db["user_id"])}
        old_password_hash = _password_hash_for_user(auth_db["user_id"])

        with patch("kiara.web.api_auth.session_repo.set_csrf_hash", return_value=False):
            changed = client.put(
                "/api/auth/password",
                json={
                    "old_password": auth_db["password"],
                    "new_password": "NewPass1234",
                },
                headers={**_std_headers(), "X-CSRF-Token": csrf_token},
            )

    assert changed.status_code == 401
    assert changed.json() == {"detail": "Nicht angemeldet", "code": "not_authenticated"}
    assert "csrf_token" not in changed.json()
    assert _password_hash_for_user(auth_db["user_id"]) == old_password_hash
    assert _session_row(other_token) is not None
    tokens_after = {row["token"] for row in _session_rows_for_user(auth_db["user_id"])}
    assert tokens_after == tokens_before


# 10. XRW fehlt → 403 csrf_invalid (auch wenn Origin stimmt)
def test_missing_xrequested_with(csrf_app, auth_db):
    token, csrf_plain = create_session(auth_db["user_id"])
    client = _client_with_cookie(csrf_app, token)
    headers = {"Origin": "http://localhost:8502", "X-CSRF-Token": csrf_plain}
    resp = client.post("/api/foo", headers=headers)
    assert resp.status_code == 403


# 13. compute_trusted_origins ohne netifaces crash-sicher
def test_compute_trusted_origins_without_netifaces():
    """Simulierter ImportError darf keine leere Liste liefern."""
    import builtins

    original_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "netifaces":
            raise ImportError("simulated")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        origins = compute_trusted_origins({}, ports=[8501])
    assert "http://localhost:8501" in origins
    assert "http://127.0.0.1:8501" in origins


# 17. Login ohne XRW oder mit untrusted Origin → 403
def test_login_requires_origin_and_xrw(csrf_app):
    with TestClient(csrf_app) as client:
        # untrusted origin
        resp = client.post(
            "/api/auth/login",
            headers={"Origin": "https://evil.example.com", "X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 403
        # ohne XRW
        resp = client.post("/api/auth/login", headers={"Origin": "http://localhost:8502"})
        assert resp.status_code == 403


# 19. compute_trusted_origins port-spezifisch
def test_compute_trusted_origins_port_specific():
    web = compute_trusted_origins({}, ports=[8501])
    admin = compute_trusted_origins({}, ports=[8502])
    assert "http://localhost:8501" in web
    assert "http://localhost:8502" in admin
    assert "http://localhost:8501" not in admin
    assert "http://localhost:8502" not in web

    with_extra = compute_trusted_origins(
        {"web": {"trusted_origins": ["https://proxy.example.com"]}},
        ports=[8501],
    )
    assert "https://proxy.example.com" in with_extra


# 22. Session ohne csrf_token_hash → 401
def test_session_without_csrf_hash_returns_401(csrf_app, auth_db):
    token, _ = create_session(auth_db["user_id"])
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET csrf_token_hash = NULL WHERE token = %s",
        (hash_token(token),),
    )
    conn.commit()
    conn.close()
    client = _client_with_cookie(csrf_app, token)
    headers = {**_std_headers(), "X-CSRF-Token": "abc"}
    resp = client.post("/api/foo", headers=headers)
    assert resp.status_code == 401


# 24. Abgelaufene Session wird wie nicht existent behandelt (401)
def test_expired_session_returns_401(csrf_app, auth_db):
    from datetime import datetime, timedelta, timezone

    token, csrf_plain = create_session(auth_db["user_id"])
    expired = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET expires_at = %s WHERE token = %s",
        (expired, hash_token(token)),
    )
    conn.commit()
    conn.close()
    client = _client_with_cookie(csrf_app, token)
    headers = {**_std_headers(), "X-CSRF-Token": csrf_plain}
    resp = client.post("/api/foo", headers=headers)
    assert resp.status_code == 401


# WOPI-Exempt bleibt vollstaendig offen
def test_wopi_full_exempt(csrf_app):
    with TestClient(csrf_app) as client:
        resp = client.post("/api/wopi/callback")
    assert resp.status_code == 200
