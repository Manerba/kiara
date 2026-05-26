"""Tests fuer Phase 12 — MCP-Server, API-Key-Verwaltung, Migration v8.

Deckt ab:
  - Migration v8 (api_keys Tabelle + Indizes)
  - API-Key-Validierung (validate_api_key)
  - Admin-CRUD-Endpunkte (api_mcp.py)
  - MCP-Server Auth-Middleware
  - MCP-Server Config-Defaults
  - Key-Generierung und Hash-Konsistenz
  - Berechtigungsdurchgriff (Gruppen → Datasources)
"""

import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection
from kiara.web.api_mcp import _generate_api_key
from kiara.web.auth import get_admin_user
from tests.admin_confirm import confirmed_request

_MCP_SCHEMA = """
    DROP TABLE IF EXISTS query_log CASCADE;
    DROP TABLE IF EXISTS api_keys CASCADE;
    DROP TABLE IF EXISTS datasource_users CASCADE;
    DROP TABLE IF EXISTS datasource_groups CASCADE;
    DROP TABLE IF EXISTS user_groups CASCADE;
    DROP TABLE IF EXISTS datasources CASCADE;
    DROP TABLE IF EXISTS groups CASCADE;
    DROP TABLE IF EXISTS users CASCADE;

    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT,
        vorname TEXT DEFAULT '',
        nachname TEXT DEFAULT '',
        email TEXT DEFAULT '',
        is_admin BOOLEAN DEFAULT false,
        is_active BOOLEAN DEFAULT true,
        created_at TEXT DEFAULT (NOW())
    );
    CREATE TABLE IF NOT EXISTS groups (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        cn TEXT NOT NULL,
        dn TEXT NOT NULL UNIQUE,
        member_count INTEGER DEFAULT 0,
        imported_at TEXT DEFAULT (NOW()),
        updated_at TEXT DEFAULT (NOW())
    );
    CREATE TABLE IF NOT EXISTS user_groups (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        UNIQUE(user_id, group_id)
    );
    CREATE TABLE IF NOT EXISTS datasources (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'local',
        path TEXT DEFAULT '',
        is_active BOOLEAN DEFAULT true,
        created_at TEXT DEFAULT (NOW()),
        updated_at TEXT DEFAULT (NOW())
    );
    CREATE TABLE IF NOT EXISTS datasource_groups (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
        group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
        UNIQUE(datasource_id, group_id)
    );
    CREATE TABLE IF NOT EXISTS datasource_users (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        UNIQUE(datasource_id, user_id)
    );
    CREATE TABLE IF NOT EXISTS api_keys (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        key_hash TEXT NOT NULL UNIQUE,
        key_prefix TEXT NOT NULL,
        user_id INTEGER NOT NULL,
        is_active BOOLEAN DEFAULT true,
        expires_at TEXT,
        last_used_at TEXT,
        created_at TEXT DEFAULT (NOW()),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
    CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
    CREATE TABLE IF NOT EXISTS query_log (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        question_hash TEXT NOT NULL,
        question_excerpt TEXT,
        answer_excerpt TEXT,
        routed_to TEXT NOT NULL DEFAULT 'local',
        model_used TEXT,
        sources_count INTEGER DEFAULT 0,
        tools_used TEXT,
        chunks_retrieved INTEGER DEFAULT 0,
        entities_anonymized INTEGER DEFAULT 0,
        duration_ms INTEGER DEFAULT 0,
        chat_id INTEGER,
        chain_type TEXT,
        full_prompt TEXT,
        full_answer TEXT,
        tool_calls TEXT,
        context_sources TEXT,
        correlation_id TEXT,
        created_at TEXT DEFAULT (NOW()),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
"""

_MCP_CLEANUP = """
    DROP TABLE IF EXISTS query_log CASCADE;
    DROP TABLE IF EXISTS api_keys CASCADE;
    DROP TABLE IF EXISTS datasource_users CASCADE;
    DROP TABLE IF EXISTS datasource_groups CASCADE;
    DROP TABLE IF EXISTS user_groups CASCADE;
    DROP TABLE IF EXISTS datasources CASCADE;
    DROP TABLE IF EXISTS groups CASCADE;
    DROP TABLE IF EXISTS users CASCADE;
"""


def _make_db():
    """Erzeugt die Test-Tabellen in PostgreSQL und gibt None zurueck (kein Pfad noetig)."""
    conn = get_connection()
    conn.executescript(_MCP_SCHEMA)
    # Default-User: Admin
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, email, is_admin) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        ("admin", "hash", "Admin", "User", "admin@test.local", True),
    )
    # Normaler User
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, email, is_admin) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        ("user1", "hash", "Normal", "User", "user@test.local", False),
    )
    conn.commit()
    conn.close()


def _cleanup_db():
    """Raeume Test-Tabellen auf und stelle Produktions-Schema wieder her."""
    conn = get_connection()
    conn.executescript(_MCP_CLEANUP)
    conn.commit()
    conn.close()
    # Produktions-Schema wiederherstellen (DROP hat es zerstoert)
    restore = get_connection()
    try:
        # Minimal-Tabellen droppen, damit create_tables() sie korrekt neu anlegt
        restore.execute("DROP TABLE IF EXISTS query_log CASCADE")
        restore.execute("DROP TABLE IF EXISTS api_keys CASCADE")
        restore.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
        restore.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
        restore.execute("DROP TABLE IF EXISTS user_groups CASCADE")
        restore.execute("DROP TABLE IF EXISTS datasources CASCADE")
        restore.execute("DROP TABLE IF EXISTS groups CASCADE")
        restore.execute("DROP TABLE IF EXISTS users CASCADE")
        from kiara.web.schema import create_tables

        create_tables(restore)
        restore.commit()
    except Exception:
        restore.rollback()
    restore.close()


def _insert_api_key(
    user_id=1,
    name="Test-Key",
    key="orag_testkey123456789012345678901234567890abcdef",
    is_active=True,
    expires_at=None,
):
    """Fuegt einen API-Key in die DB ein und gibt (full_key, key_id) zurueck."""
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    key_prefix = key[:12] + "..."
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO api_keys (name, key_hash, key_prefix, user_id, is_active, expires_at) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (name, key_hash, key_prefix, user_id, is_active, expires_at),
    )
    row = cursor.fetchone()
    key_id = row["id"]
    conn.commit()
    conn.close()
    return key, key_id


def _mock_db_query():
    """Context-Manager fuer db_query Mock mit PostgreSQL."""

    @contextmanager
    def ctx():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    return ctx()


def _mock_db_transaction():
    """Context-Manager fuer db_transaction Mock mit PostgreSQL."""

    @contextmanager
    def ctx():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    return ctx()


# ===================================================================
# MIGRATION v8
# ===================================================================

_BASELINE_SCHEMA = """
    DROP TABLE IF EXISTS schema_version CASCADE;
    DROP TABLE IF EXISTS query_log CASCADE;
    DROP TABLE IF EXISTS api_keys CASCADE;
    DROP TABLE IF EXISTS cloud_audit_log CASCADE;
    DROP TABLE IF EXISTS user_groups CASCADE;
    DROP TABLE IF EXISTS ai_config CASCADE;
    DROP TABLE IF EXISTS chat_messages CASCADE;
    DROP TABLE IF EXISTS chats CASCADE;
    DROP TABLE IF EXISTS sessions CASCADE;
    DROP TABLE IF EXISTS datasources CASCADE;
    DROP TABLE IF EXISTS groups CASCADE;
    DROP TABLE IF EXISTS users CASCADE;

    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        created_at TEXT DEFAULT (NOW())
    );
    CREATE TABLE IF NOT EXISTS sessions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER, token TEXT,
        csrf_token_hash CHAR(64),
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS chats (
        id SERIAL PRIMARY KEY,
        user_id INTEGER, title TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        chat_id INTEGER, role TEXT, content TEXT,
        FOREIGN KEY (chat_id) REFERENCES chats(id)
    );
    CREATE TABLE IF NOT EXISTS datasources (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL, path TEXT
    );
    CREATE TABLE IF NOT EXISTS groups (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    );
    CREATE TABLE IF NOT EXISTS user_groups (
        user_id INTEGER, group_id INTEGER,
        PRIMARY KEY (user_id, group_id)
    );
    CREATE TABLE IF NOT EXISTS ai_config (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS cloud_audit_log (
        id SERIAL PRIMARY KEY,
        action TEXT, created_at TEXT DEFAULT (NOW())
    );
    CREATE TABLE IF NOT EXISTS query_log (
        id SERIAL PRIMARY KEY,
        query TEXT, created_at TEXT DEFAULT (NOW())
    );
"""


def _create_baseline_db():
    """Baseline-DB mit Tabellen fuer Migration 1-3 (No-Ops)."""
    conn = get_connection()
    conn.executescript(_BASELINE_SCHEMA)
    conn.commit()
    conn.close()


def _cleanup_baseline_db():
    """Stellt das vollstaendige Test-Schema nach destruktiven Tests wieder her."""
    from db_helpers import restore_test_schema

    restore_test_schema()


# TestMigrationV8 entfernt — Migration v8 ist obsolet, api_keys-Tabelle ist
# seit dem Schema-Reset-Sprint direkt in schema.sql deklariert.
# Die Tabellen-Existenz-Checks laufen jetzt implizit ueber die Live-Tests
# in TestAdminMcpApi / TestValidateApiKey.


# ===================================================================
# KEY-GENERIERUNG
# ===================================================================


class TestKeyGeneration:
    """Tests fuer _generate_api_key()."""

    def test_key_format(self):
        """Key hat Prefix orag_ und 48 Hex-Zeichen."""
        full_key, key_hash, key_prefix = _generate_api_key()
        assert full_key.startswith("orag_")
        raw = full_key[5:]
        assert len(raw) == 48
        # Nur Hex-Zeichen
        int(raw, 16)

    def test_key_hash_is_sha256(self):
        """key_hash ist SHA-256 des vollen Keys."""
        full_key, key_hash, key_prefix = _generate_api_key()
        expected = hashlib.sha256(full_key.encode()).hexdigest()
        assert key_hash == expected

    def test_key_prefix_format(self):
        """key_prefix ist erste 20 Zeichen + '...'."""
        full_key, key_hash, key_prefix = _generate_api_key()
        assert key_prefix == full_key[:20] + "..."

    def test_keys_are_unique(self):
        """Zwei generierte Keys sind verschieden."""
        k1, _, _ = _generate_api_key()
        k2, _, _ = _generate_api_key()
        assert k1 != k2


# ===================================================================
# API-KEY-VALIDIERUNG (validate_api_key)
# ===================================================================


class TestValidateApiKey:
    """Tests fuer kiara.web.api_key_auth.validate_api_key()."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        """Setup MCP-Schema und Teardown nach jedem Test."""
        _make_db()
        yield
        _cleanup_db()

    def test_valid_key_returns_user(self):
        """Gueltiger Key gibt User-Dict mit allen Feldern zurueck."""
        key, _ = _insert_api_key(user_id=1)

        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            result = validate_api_key(key)

        assert result is not None
        assert result["id"] == 1
        assert result["username"] == "admin"
        assert result["is_admin"]
        assert result["api_key_id"] == 1
        assert "group_ids" in result
        assert "datasource_ids" in result

    def test_invalid_key_returns_none(self):
        """Ungueltiger Key gibt None zurueck."""
        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            result = validate_api_key("orag_doesnotexist12345678901234567890abcdef1234")

        assert result is None

    def test_inactive_key_returns_none(self):
        """Deaktivierter Key gibt None zurueck."""
        key, _ = _insert_api_key(user_id=1, is_active=False)

        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            result = validate_api_key(key)

        assert result is None

    def test_inactive_user_returns_none(self):
        """Key eines deaktivierten Users gibt None zurueck."""
        # User deaktivieren
        conn = get_connection()
        conn.execute("UPDATE users SET is_active = false WHERE id = 1")
        conn.commit()
        conn.close()

        key, _ = _insert_api_key(user_id=1)

        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            result = validate_api_key(key)

        assert result is None

    def test_expired_key_returns_none(self):
        """Abgelaufener Key gibt None zurueck."""
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        key, _ = _insert_api_key(user_id=1, expires_at=past)

        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            result = validate_api_key(key)

        assert result is None

    def test_future_expiry_is_valid(self):
        """Key mit Ablaufdatum in der Zukunft ist gueltig."""
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        key, _ = _insert_api_key(user_id=1, expires_at=future)

        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            result = validate_api_key(key)

        assert result is not None
        assert result["username"] == "admin"

    def test_group_and_datasource_ids_populated(self):
        """Gruppen- und Datasource-IDs werden korrekt geladen."""
        conn = get_connection()
        # Gruppe + User-Zuordnung
        conn.execute(
            "INSERT INTO groups (name, cn, dn) VALUES (%s, %s, %s)", ("IT", "IT", "cn=IT,dc=test")
        )
        conn.execute("INSERT INTO user_groups (user_id, group_id) VALUES (%s, %s)", (2, 1))
        # Datasource + Gruppen-Zuordnung
        conn.execute("INSERT INTO datasources (name, type) VALUES (%s, %s)", ("Wiki", "local"))
        conn.execute(
            "INSERT INTO datasource_groups (datasource_id, group_id) VALUES (%s, %s)", (1, 1)
        )
        conn.commit()
        conn.close()

        key, _ = _insert_api_key(
            user_id=2, name="User-Key", key="orag_userkey1234567890123456789012345678901234ab"
        )

        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            result = validate_api_key(key)

        assert result is not None
        assert result["username"] == "user1"
        assert 1 in result["group_ids"]
        assert 1 in result["datasource_ids"]

    def test_last_used_at_updated(self):
        """last_used_at wird bei Validierung aktualisiert."""
        key, key_id = _insert_api_key(user_id=1)

        with (
            patch("kiara.web.api_key_auth.db_query", side_effect=_mock_db_query),
            patch("kiara_db.base.db_transaction", side_effect=_mock_db_transaction),
        ):
            from kiara.web.api_key_auth import validate_api_key

            validate_api_key(key)

        # Pruefen ob last_used_at gesetzt wurde
        conn = get_connection()
        row = conn.execute("SELECT last_used_at FROM api_keys WHERE id=%s", (key_id,)).fetchone()
        conn.close()
        assert row["last_used_at"] is not None


# ===================================================================
# ADMIN-CRUD-API (api_mcp.py)
# ===================================================================


class TestAdminMcpApi:
    """Tests fuer die Admin-CRUD-Endpunkte (migriert zu Admin-Registry)."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        """Setup MCP-Schema und Teardown nach jedem Test."""
        _make_db()
        yield
        _cleanup_db()

    @pytest.fixture
    def client(self):
        """FastAPI TestClient mit Admin-Registry-Routen und Mock-Auth."""
        from kiara.admin.registry import admin_registry, load_admin_functions
        from kiara.admin.router import generate_admin_routes

        load_admin_functions()
        app = FastAPI()
        generate_admin_routes(app, admin_registry)

        async def mock_admin():
            return {"id": 1, "username": "admin", "is_admin": True}

        app.dependency_overrides[get_admin_user] = mock_admin

        with TestClient(app) as c:
            yield c

    @pytest.fixture
    def client_no_auth(self):
        """FastAPI TestClient OHNE Auth-Override."""
        from kiara.admin.registry import admin_registry, load_admin_functions
        from kiara.admin.router import generate_admin_routes

        load_admin_functions()
        app = FastAPI()
        generate_admin_routes(app, admin_registry)

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    # --- Create ---

    def test_create_key(self, client):
        """POST /api/admin/mcp/keys erstellt einen neuen Key."""
        resp = client.post(
            "/api/admin/mcp/keys",
            json={
                "name": "VS Code Test",
                "user_id": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"].startswith("orag_")
        assert "prefix" in data
        assert "message" in data

    def test_create_key_returns_unique_keys(self, client):
        """Zwei erstellte Keys sind verschieden."""
        r1 = client.post("/api/admin/mcp/keys", json={"name": "Key1", "user_id": 1}).json()
        r2 = client.post("/api/admin/mcp/keys", json={"name": "Key2", "user_id": 1}).json()
        assert r1["key"] != r2["key"]

    def test_create_key_with_expiry(self, client):
        """Key mit Ablaufdatum kann erstellt werden."""
        resp = client.post(
            "/api/admin/mcp/keys",
            json={
                "name": "Temp-Key",
                "user_id": 1,
                "expires_at": "2099-12-31",
            },
        )
        assert resp.status_code == 200

    def test_create_key_invalid_user(self, client):
        """Key fuer nicht-existierenden User gibt 404."""
        resp = client.post(
            "/api/admin/mcp/keys",
            json={
                "name": "Bad-Key",
                "user_id": 9999,
            },
        )
        assert resp.status_code == 404

    # --- List ---

    def test_list_keys_empty(self, client):
        """GET /api/admin/mcp/keys gibt leere Liste bei keinen Keys."""
        resp = client.get("/api/admin/mcp/keys")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_keys_after_create(self, client):
        """Nach Erstellung erscheint Key in der Liste."""
        client.post("/api/admin/mcp/keys", json={"name": "MyKey", "user_id": 1})
        resp = client.get("/api/admin/mcp/keys")
        data = resp.json()["data"]
        assert len(data) == 1
        assert data[0]["name"] == "MyKey"
        assert data[0]["username"] == "admin"
        # Klartext-Key darf NICHT in der Liste sein
        assert "key" not in data[0] or not data[0].get("key", "").startswith("orag_")

    def test_list_keys_no_plaintext(self, client):
        """Liste enthaelt nur Prefix, nicht den vollen Key."""
        create_resp = client.post("/api/admin/mcp/keys", json={"name": "Secret", "user_id": 1})
        full_key = create_resp.json()["key"]

        list_resp = client.get("/api/admin/mcp/keys")
        data = list_resp.json()["data"][0]
        assert data["key_prefix"].endswith("...")
        # Voller Key darf nirgends auftauchen
        for v in data.values():
            if isinstance(v, str):
                assert full_key not in v

    # --- Delete ---

    def test_delete_key(self, client):
        """DELETE /api/admin/mcp/keys/{id} entfernt den Key."""
        client.post("/api/admin/mcp/keys", json={"name": "ToDelete", "user_id": 1})
        keys = client.get("/api/admin/mcp/keys").json()["data"]
        key_id = keys[0]["id"]

        resp = confirmed_request(client, "DELETE", f"/api/admin/mcp/keys/{key_id}")
        assert resp.status_code == 200

        keys_after = client.get("/api/admin/mcp/keys").json()["data"]
        assert len(keys_after) == 0

    def test_delete_nonexistent_key(self, client):
        """DELETE auf nicht-existierenden Key gibt 404."""
        resp = confirmed_request(client, "DELETE", "/api/admin/mcp/keys/9999")
        assert resp.status_code == 404

    # --- Toggle ---

    def test_toggle_key_deactivates(self, client):
        """PUT /toggle deaktiviert einen aktiven Key."""
        client.post("/api/admin/mcp/keys", json={"name": "Toggle", "user_id": 1})
        keys = client.get("/api/admin/mcp/keys").json()["data"]
        key_id = keys[0]["id"]

        resp = confirmed_request(client, "PUT", f"/api/admin/mcp/keys/{key_id}/toggle")
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

    def test_toggle_key_reactivates(self, client):
        """Zweimaliges Toggle aktiviert den Key wieder."""
        client.post("/api/admin/mcp/keys", json={"name": "ReToggle", "user_id": 1})
        keys = client.get("/api/admin/mcp/keys").json()["data"]
        key_id = keys[0]["id"]

        confirmed_request(client, "PUT", f"/api/admin/mcp/keys/{key_id}/toggle")  # deaktivieren
        resp = confirmed_request(
            client, "PUT", f"/api/admin/mcp/keys/{key_id}/toggle"
        )  # reaktivieren
        assert resp.json()["is_active"] is True

    def test_toggle_nonexistent_key(self, client):
        """Toggle auf nicht-existierenden Key gibt 404."""
        resp = confirmed_request(client, "PUT", "/api/admin/mcp/keys/9999/toggle")
        assert resp.status_code == 404

    # --- Auth ---

    def test_endpoints_require_auth(self, client_no_auth):
        """Alle Endpunkte erfordern Admin-Authentifizierung."""
        assert client_no_auth.get("/api/admin/mcp/keys").status_code in (401, 403)
        assert client_no_auth.post(
            "/api/admin/mcp/keys", json={"name": "x", "user_id": 1}
        ).status_code in (401, 403, 422)
        assert client_no_auth.delete("/api/admin/mcp/keys/1").status_code in (401, 403)
        assert client_no_auth.put("/api/admin/mcp/keys/1/toggle").status_code in (401, 403)

    # --- Status Route (Sprint 1.4) ---

    def test_status_route_shape(self, client):
        """GET /api/admin/mcp/status liefert dokumentierte Success-Shape."""
        with (
            patch("kiara.config.load_config") as mock_config,
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_sub,
        ):
            mock_config.return_value = {"web": {"mcp_enabled": True, "mcp_port": 8503}}

            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"active\n", b""))
            proc.returncode = 0
            mock_sub.return_value = proc

            resp = client.get("/api/admin/mcp/status")

        assert resp.status_code == 200
        data = resp.json()
        for key in ("config_enabled", "service_running", "service_status", "port"):
            assert key in data, f"Feld '{key}' fehlt in Status-Response"
        assert data["config_enabled"] is True
        assert data["port"] == 8503

    # --- Toggle Route (Sprint 1.4) ---

    def test_toggle_route_shape(self, client):
        """POST /api/admin/mcp/toggle liefert dokumentierte Success-Shape."""
        with (
            patch("kiara.config.load_config") as mock_config,
            patch("kiara.config.save_config"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_sub,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_config.return_value = {"web": {"mcp_enabled": False}}

            enable_proc = AsyncMock()
            enable_proc.communicate = AsyncMock(return_value=(b"", b""))
            enable_proc.returncode = 0

            status_proc = AsyncMock()
            status_proc.communicate = AsyncMock(return_value=(b"active\n", b""))
            status_proc.returncode = 0

            mock_sub.side_effect = [enable_proc] + [status_proc] * 12

            resp = confirmed_request(
                client, "POST", "/api/admin/mcp/toggle", json={"enabled": True}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert "service_status" in data
        assert "message" in data


# ===================================================================
# MCP-SERVER AUTH-MIDDLEWARE
# ===================================================================

try:
    from fastmcp import FastMCP  # noqa: F401

    _has_fastmcp = True
except (ImportError, AttributeError):
    _has_fastmcp = False

_skip_no_fastmcp = pytest.mark.skipif(
    not _has_fastmcp,
    reason="fastmcp nicht installiert",
)


@_skip_no_fastmcp
class TestMcpAuthMiddleware:
    """Tests fuer die Auth-Middleware des MCP-Servers."""

    @pytest.fixture
    def mcp_client(self, tmp_path):
        """TestClient fuer den MCP-Server mit gemockter validate_api_key."""
        # validate_api_key wird lazy importiert in _create_app() —
        # daher muss das Quellmodul gepatcht werden, NICHT kiara.mcp_server.
        from kiara.mcp_server import _create_app

        with patch("kiara.web.api_key_auth.validate_api_key") as mock_validate:
            mock_validate.return_value = None  # Default: ungueltig
            app = _create_app()
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_validate

    def test_health_without_auth(self, mcp_client):
        """GET /health funktioniert ohne Auth."""
        client, _ = mcp_client
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["service"] == "kiara-mcp"

    def test_no_auth_header(self, mcp_client):
        """Anfrage ohne Authorization-Header gibt 401."""
        client, _ = mcp_client
        resp = client.post("/mcp", json={"jsonrpc": "2.0", "method": "tools/list", "id": 1})
        assert resp.status_code == 401
        assert "Bearer" in resp.json()["error"]

    def test_wrong_key_format(self, mcp_client):
        """Key ohne orag_-Prefix gibt 401."""
        client, _ = mcp_client
        resp = client.post(
            "/mcp",
            headers={"Authorization": "Bearer wrongformat123"},
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )
        assert resp.status_code == 401
        assert "orag_" in resp.json()["error"]

    def test_invalid_key(self, mcp_client):
        """Ungueltiger orag_-Key gibt 401."""
        client, mock_validate = mcp_client
        mock_validate.return_value = None

        resp = client.post(
            "/mcp",
            headers={"Authorization": "Bearer orag_invalid123456789012345678901234567890abcdef"},
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )
        assert resp.status_code == 401
        assert "abgelaufener" in resp.json()["error"]

    def test_valid_key_passes_middleware(self, mcp_client):
        """Gueltiger Key laesst die Anfrage durch die Middleware."""
        client, mock_validate = mcp_client
        mock_validate.return_value = {
            "id": 1,
            "username": "admin",
            "is_admin": True,
            "group_ids": [],
            "datasource_ids": [],
            "api_key_id": 1,
        }

        resp = client.post(
            "/mcp",
            headers={
                "Authorization": "Bearer orag_valid1234567890123456789012345678901234abcdef",
                "Accept": "application/json, text/event-stream",
            },
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )
        # Sollte durchgehen (200 = MCP antwortet, nicht 401)
        assert resp.status_code == 200

    def test_options_without_auth(self, mcp_client):
        """OPTIONS-Anfragen benoetigen Auth wie andere MCP-Requests."""
        client, _ = mcp_client
        resp = client.options("/mcp")
        assert resp.status_code == 401


# ===================================================================
# MCP-SERVER CONFIG
# ===================================================================


@_skip_no_fastmcp
class TestMcpConfig:
    """Tests fuer MCP-Konfiguration."""

    def test_default_config_has_mcp_fields(self):
        """DEFAULT_CONFIG enthaelt mcp_port und mcp_enabled."""
        from kiara.config import DEFAULT_CONFIG

        assert "mcp_port" in DEFAULT_CONFIG["web"]
        assert "mcp_enabled" in DEFAULT_CONFIG["web"]
        assert DEFAULT_CONFIG["web"]["mcp_port"] == 8504
        assert DEFAULT_CONFIG["web"]["mcp_enabled"] is False

    def test_run_mcp_server_disabled(self):
        """run_mcp_server() bricht ab wenn mcp_enabled=False."""
        from kiara.mcp_server import run_mcp_server

        # load_config und uvicorn werden lazy importiert in run_mcp_server() —
        # daher Quellmodule patchen, nicht kiara.mcp_server.
        mock_config = {"web": {"mcp_enabled": False, "mcp_port": 8504, "host": "0.0.0.0"}}
        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("uvicorn.run") as mock_uvicorn_run,
        ):
            run_mcp_server()
            mock_uvicorn_run.assert_not_called()

    def test_run_mcp_server_enabled(self):
        """run_mcp_server() startet uvicorn wenn mcp_enabled=True."""
        mock_config = {"web": {"mcp_enabled": True, "mcp_port": 9999, "host": "127.0.0.1"}}
        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.mcp_server._create_app") as mock_create,
            patch("uvicorn.run") as mock_uvicorn_run,
        ):
            mock_create.return_value = MagicMock()
            from kiara.mcp_server import run_mcp_server

            run_mcp_server()
            mock_uvicorn_run.assert_called_once()
            call_kwargs = mock_uvicorn_run.call_args
            assert call_kwargs[1]["port"] == 9999
            assert call_kwargs[1]["host"] == "127.0.0.1"


# ===================================================================
# AUDIT-LOG
# ===================================================================


@_skip_no_fastmcp
class TestMcpAuditLog:
    """Tests fuer _log_mcp_query()."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        """Setup MCP-Schema und Teardown nach jedem Test."""
        _make_db()
        yield
        _cleanup_db()

    def test_log_mcp_query_inserts_record(self):
        """_log_mcp_query schreibt einen Eintrag in query_log."""
        with patch("kiara.shared.db.db_transaction", side_effect=_mock_db_transaction):
            from kiara.mcp_server import _log_mcp_query

            _log_mcp_query(
                user={"id": 1},
                question="Test-Frage",
                answer="Test-Antwort",
                sources=[{"source": "file.txt", "score": 0.9}],
                route="local",
                duration_ms=1234,
                model="qwen3:8b",
            )

        conn = get_connection()
        row = conn.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        assert row is not None
        assert row["user_id"] == 1
        assert row["routed_to"] == "mcp:local"
        assert row["model_used"] == "qwen3:8b"
        assert row["sources_count"] == 1
        assert row["duration_ms"] == 1234
        assert row["question_excerpt"] == "Test-Frage"
        assert row["answer_excerpt"] == "Test-Antwort"
        assert len(row["question_hash"]) == 64  # SHA-256 Hex

    def test_log_mcp_query_truncates_long_text(self):
        """Lange Fragen und Antworten werden auf 200 Zeichen gekuerzt."""
        with patch("kiara.shared.db.db_transaction", side_effect=_mock_db_transaction):
            from kiara.mcp_server import _log_mcp_query

            _log_mcp_query(
                user={"id": 1},
                question="A" * 500,
                answer="B" * 500,
                sources=[],
                route="local",
                duration_ms=0,
                model="test",
            )

        conn = get_connection()
        row = conn.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        assert len(row["question_excerpt"]) == 200
        assert len(row["answer_excerpt"]) == 200

    def test_log_mcp_query_route_prefix(self):
        """routed_to hat das Prefix 'mcp:' (unterscheidbar von Web-Queries)."""
        with patch("kiara.shared.db.db_transaction", side_effect=_mock_db_transaction):
            from kiara.mcp_server import _log_mcp_query

            _log_mcp_query(
                user={"id": 1},
                question="Q",
                answer="A",
                sources=[],
                route="cloud",
                duration_ms=0,
                model="test",
            )

        conn = get_connection()
        row = conn.execute("SELECT routed_to FROM query_log ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        assert row["routed_to"] == "mcp:cloud"


# ===================================================================
# ADMIN-UI ROUTER-INTEGRATION
# ===================================================================


class TestAdminAppIntegration:
    """Tests fuer die Integration des MCP-Routers in admin_app.py."""

    def test_mcp_router_registered(self):
        """admin_app.py hat den MCP-Router eingebunden."""
        from kiara.web.admin_app import app

        routes = [r.path for r in app.routes if hasattr(r, "path")]
        # Der Router hat Prefix /api/admin/mcp
        mcp_paths = [p for p in routes if "/api/admin/mcp" in p]
        assert len(mcp_paths) > 0

    def test_mcp_key_endpoints_exist(self):
        """Alle 4 MCP-Endpunkte sind registriert."""
        from kiara.web.admin_app import app

        route_info = []
        for route in app.routes:
            if hasattr(route, "path") and hasattr(route, "methods"):
                route_info.append((route.path, route.methods))

        paths_methods = {(p, tuple(sorted(m))) for p, m in route_info}

        assert ("/api/admin/mcp/keys", ("GET",)) in paths_methods
        assert ("/api/admin/mcp/keys", ("POST",)) in paths_methods
        assert ("/api/admin/mcp/keys/{key_id}", ("DELETE",)) in paths_methods
        assert ("/api/admin/mcp/keys/{key_id}/toggle", ("PUT",)) in paths_methods


# ===================================================================
# F-57: MCP QUERY ERROR — GENERISCHE FEHLERMELDUNG
# ===================================================================


@_skip_no_fastmcp
class TestMcpQueryErrorHandling:
    """Tests fuer F-57 — Bei Pipeline-Exception wird generische Meldung zurueckgegeben.

    Stellt sicher, dass interne Exception-Details NICHT an den Client
    durchgereicht werden (Information Disclosure Prevention).
    """

    def test_mcp_query_error_returns_generic_message(self):
        """Bei Pipeline-Exception wird eine generische Fehlermeldung zurueckgegeben."""
        import json

        user = {
            "id": 1,
            "username": "admin",
            "is_admin": True,
            "group_ids": [],
            "datasource_ids": [],
            "api_key_id": 1,
        }

        with patch("kiara.mcp_server._run_rag_query") as mock_rag:
            mock_rag.side_effect = RuntimeError(
                "Interner DB-Fehler: Connection refused to gpu-server.local:5432"
            )

            from kiara.mcp_server import _create_app

            with patch("kiara.web.api_key_auth.validate_api_key") as mock_validate:
                mock_validate.return_value = user
                app = _create_app()

                with TestClient(app, raise_server_exceptions=False) as client:
                    resp = client.post(
                        "/mcp",
                        headers={
                            "Authorization": "Bearer orag_testkey1234567890123456789012345678901234abcdef",
                            "Accept": "application/json, text/event-stream",
                        },
                        json={
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "query",
                                "arguments": {"question": "Was ist KIara?", "k": 3},
                            },
                            "id": 1,
                        },
                    )

                    assert resp.status_code == 200

                    # MCP antwortet im SSE-Format — JSON aus "data:"-Zeilen extrahieren
                    body = resp.text
                    data = None
                    for line in body.splitlines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            break

                    assert data is not None, f"Keine data-Zeile in SSE-Response: {body[:500]}"
                    assert "result" in data

                    content_list = data["result"].get("content", [])
                    assert len(content_list) > 0

                    answer_text = content_list[0].get("text", "")
                    answer_data = json.loads(answer_text)

                    assert answer_data["route"] == "error"
                    assert "spaeter erneut versuchen" in answer_data["answer"]

    def test_mcp_query_error_does_not_leak_exception_details(self):
        """Exception-Text darf NICHT in der MCP-Antwort stehen."""
        secret_error = (
            "psycopg2.OperationalError: FATAL password authentication failed for user 'kiara'"
        )

        user = {
            "id": 1,
            "username": "admin",
            "is_admin": True,
            "group_ids": [],
            "datasource_ids": [],
            "api_key_id": 1,
        }

        with patch("kiara.mcp_server._run_rag_query") as mock_rag:
            mock_rag.side_effect = Exception(secret_error)

            from kiara.mcp_server import _create_app

            with patch("kiara.web.api_key_auth.validate_api_key") as mock_validate:
                mock_validate.return_value = user
                app = _create_app()

                with TestClient(app, raise_server_exceptions=False) as client:
                    resp = client.post(
                        "/mcp",
                        headers={
                            "Authorization": "Bearer orag_testkey1234567890123456789012345678901234abcdef",
                            "Accept": "application/json, text/event-stream",
                        },
                        json={
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "query",
                                "arguments": {"question": "Testfrage", "k": 3},
                            },
                            "id": 2,
                        },
                    )

                    assert resp.status_code == 200
                    full_response = resp.text

                    # Der geheime Fehlertext darf NIRGENDS in der Antwort auftauchen
                    assert "psycopg2" not in full_response
                    assert "password authentication failed" not in full_response
                    assert secret_error not in full_response
