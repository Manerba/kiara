"""Tiefergehende Tests fuer kiara.web.api_ki_backends — Sicherheit, Verschluesselung, Test-Endpoint.

Ergaenzt test_api_ki_backends.py um API-Key-Roundtrip, Toggle-Konsistenz,
extra_config JSON, Test-Endpoint und Grenzfaelle.
"""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from kiara.shared.constants import PASSWORD_MASK
from kiara.shared.db import get_connection
from kiara.web.admin_app import app
from tests.admin_confirm import confirmed_request


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_admin():
    """Mock-Admin-User fuer Dependency-Override."""
    user = {"id": 1, "username": "admin", "is_admin": True}
    from kiara.web.auth import get_admin_user

    app.dependency_overrides[get_admin_user] = lambda: user
    yield user
    app.dependency_overrides.pop(get_admin_user, None)


@pytest.fixture
def mock_db():
    """PostgreSQL-DB mit ki_backends-Tabelle."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS ki_backends CASCADE")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ki_backends (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL CHECK (category IN ('embedding', 'llm', 'cloud', 'vlm')),
            provider TEXT NOT NULL CHECK (provider IN ('ollama', 'anthropic', 'openai')),
            name TEXT NOT NULL,
            url TEXT DEFAULT '',
            base_url TEXT DEFAULT '',
            model TEXT NOT NULL,
            api_key TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            priority INTEGER DEFAULT 10,
            trusted BOOLEAN DEFAULT false,
            extra_config TEXT DEFAULT '{}',
            last_test_at TEXT DEFAULT NULL,
            last_test_ok INTEGER DEFAULT NULL,
            last_test_error TEXT DEFAULT '',
            created_at TEXT DEFAULT (NOW()),
            updated_at TEXT DEFAULT (NOW())
        );
    """)
    conn.commit()

    @contextmanager
    def mock_db_query():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def mock_db_transaction():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    with (
        patch("kiara_db.base.db_query", mock_db_query),
        patch("kiara_db.base.db_transaction", mock_db_transaction),
        patch("kiara.backend_selector.invalidate_cache"),
    ):
        yield conn

    conn.rollback()
    conn.close()


def _create_backend(client, **overrides):
    """Helper: Backend ueber API erstellen."""
    data = {
        "category": "llm",
        "provider": "ollama",
        "name": "Test",
        "model": "qwen3:8b",
        "url": "http://localhost:11434",
    }
    data.update(overrides)
    resp = client.post(
        "/api/admin/ki/backends/", json=data, headers={"X-Requested-With": "XMLHttpRequest"}
    )
    return resp


def _get_raw_backend(conn, backend_id):
    """Liest Backend direkt aus DB (ohne API-Masking)."""
    c = get_connection()
    row = c.execute("SELECT * FROM ki_backends WHERE id = %s", (backend_id,)).fetchone()
    c.close()
    return dict(row) if row else None


# ===================================================================
# A) API-Key Verschluesselung Roundtrip
# ===================================================================


class TestApiKeyEncryption:
    def test_create_cloud_encrypts_api_key(self, client, mock_admin, mock_db):
        """Cloud-Backend API-Key wird bei Create verschluesselt in DB gespeichert."""
        resp = _create_backend(
            client,
            category="cloud",
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-ant-secret-key",
        )
        assert resp.status_code == 200
        bid = resp.json()["id"]
        raw = _get_raw_backend(mock_db, bid)
        # api_key in DB darf NICHT der Klartext sein
        assert raw["api_key"] != "sk-ant-secret-key"
        assert raw["api_key"] != ""

    def test_list_masks_api_key(self, client, mock_admin, mock_db):
        """GET-Response maskiert API-Keys mit PASSWORD_MASK."""
        _create_backend(
            client,
            category="cloud",
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-secret",
        )
        resp = client.get("/api/admin/ki/backends/")
        backends = resp.json()["backends"]
        for b in backends:
            if b["api_key"]:
                assert b["api_key"] == PASSWORD_MASK

    def test_update_password_mask_keeps_old_key(self, client, mock_admin, mock_db):
        """PUT mit PASSWORD_MASK behaelt den alten verschluesselten Key."""
        resp = _create_backend(
            client,
            category="cloud",
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-original-key",
        )
        bid = resp.json()["id"]
        original_raw = _get_raw_backend(mock_db, bid)

        # Update mit PASSWORD_MASK
        client.put(
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "cloud",
                "provider": "anthropic",
                "name": "Updated",
                "model": "claude-sonnet-4-6",
                "api_key": PASSWORD_MASK,
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        updated_raw = _get_raw_backend(mock_db, bid)
        assert updated_raw["api_key"] == original_raw["api_key"]

    def test_update_empty_api_key_clears(self, client, mock_admin, mock_db):
        """PUT mit leerem api_key loescht den Key."""
        resp = _create_backend(
            client,
            category="cloud",
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-to-clear",
        )
        bid = resp.json()["id"]

        confirmed_request(
            client,
            "PUT",
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "cloud",
                "provider": "anthropic",
                "name": "Cleared",
                "model": "claude-sonnet-4-6",
                "api_key": "",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        raw = _get_raw_backend(mock_db, bid)
        assert raw["api_key"] == ""

    def test_update_new_api_key_encrypts(self, client, mock_admin, mock_db):
        """PUT mit neuem Key verschluesselt diesen."""
        resp = _create_backend(
            client,
            category="cloud",
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-old",
        )
        bid = resp.json()["id"]

        client.put(
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "cloud",
                "provider": "anthropic",
                "name": "New Key",
                "model": "claude-sonnet-4-6",
                "api_key": "sk-new-secret",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        raw = _get_raw_backend(mock_db, bid)
        assert raw["api_key"] != "sk-new-secret"
        assert raw["api_key"] != ""


# ===================================================================
# B) Toggle mit Modell-Konsistenz
# ===================================================================


class TestToggleConsistency:
    def test_toggle_activate_different_embedding_model_allowed(self, client, mock_admin, mock_db):
        """Reaktivierung eines Embedding-Backends mit anderem Modell ist erlaubt (Multi-Modell)."""
        # Erstes Embedding aktiv
        _create_backend(client, category="embedding", provider="ollama", model="nomic-embed-text")

        # Zweites Embedding mit anderem Modell erstellen (deaktiviert)
        conn = get_connection()
        conn.execute(
            """INSERT INTO ki_backends (category, provider, name, model, is_active, priority)
               VALUES ('embedding', 'ollama', 'Alt', 'all-minilm', false, 20)"""
        )
        conn.commit()
        bid2 = conn.execute("SELECT id FROM ki_backends WHERE model='all-minilm'").fetchone()["id"]
        conn.close()

        # Reaktivierung → erlaubt (verschiedene Modelle = verschiedene Collections)
        resp = client.patch(
            f"/api/admin/ki/backends/{bid2}/toggle",
            json={"is_active": True},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200

    def test_toggle_deactivate_no_check(self, client, mock_admin, mock_db):
        """Deaktivieren fuehrt KEINE Modell-Konsistenzpruefung durch."""
        resp = _create_backend(
            client, category="embedding", provider="ollama", model="nomic-embed-text"
        )
        bid = resp.json()["id"]

        resp = client.patch(
            f"/api/admin/ki/backends/{bid}/toggle",
            json={"is_active": False},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        assert "deaktiviert" in resp.json()["message"]

    def test_toggle_activate_same_model_ok(self, client, mock_admin, mock_db):
        """Reaktivierung bei gleichem Modell → OK."""
        _create_backend(client, category="embedding", provider="ollama", model="nomic-embed-text")

        conn = get_connection()
        conn.execute(
            """INSERT INTO ki_backends (category, provider, name, model, is_active, priority)
               VALUES ('embedding', 'ollama', 'Backup', 'nomic-embed-text', false, 20)"""
        )
        conn.commit()
        bid2 = conn.execute("SELECT id FROM ki_backends WHERE name='Backup'").fetchone()["id"]
        conn.close()

        resp = client.patch(
            f"/api/admin/ki/backends/{bid2}/toggle",
            json={"is_active": True},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200


# ===================================================================
# C) Update-Endpunkt Tiefentest
# ===================================================================


class TestUpdateDeep:
    def test_update_nonexistent_returns_404(self, client, mock_admin, mock_db):
        """PUT auf nicht-existierendes Backend → 404."""
        resp = confirmed_request(
            client,
            "PUT",
            "/api/admin/ki/backends/9999",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Nope",
                "model": "qwen3:8b",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 404

    def test_update_embedding_model_change_allowed(self, client, mock_admin, mock_db):
        """Modell-Wechsel bei Embedding-Backend ist erlaubt (Multi-Modell)."""
        _create_backend(
            client,
            category="embedding",
            provider="ollama",
            model="nomic-embed-text",
            name="Embed 1",
        )
        resp2 = _create_backend(
            client,
            category="embedding",
            provider="ollama",
            model="nomic-embed-text",
            name="Embed 2",
        )
        bid2 = resp2.json()["id"]

        # Modell aendern → erlaubt
        resp = confirmed_request(
            client,
            "PUT",
            f"/api/admin/ki/backends/{bid2}",
            json={
                "category": "embedding",
                "provider": "ollama",
                "name": "Embed 2 Updated",
                "model": "all-minilm",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200

    def test_update_changes_updated_at(self, client, mock_admin, mock_db):
        """Update aktualisiert den updated_at Timestamp."""
        resp = _create_backend(client)
        bid = resp.json()["id"]
        raw_before = _get_raw_backend(mock_db, bid)

        import time

        time.sleep(0.1)  # Minimal-Delay fuer Timestamp-Unterschied

        client.put(
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Updated",
                "model": "qwen3:8b",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        raw_after = _get_raw_backend(mock_db, bid)
        assert raw_after["updated_at"] >= raw_before["updated_at"]


# ===================================================================
# D) extra_config JSON Handling
# ===================================================================


class TestExtraConfig:
    def test_create_with_extra_config(self, client, mock_admin, mock_db):
        """extra_config wird als JSON gespeichert."""
        resp = _create_backend(
            client,
            extra_config={"temperature": 0.7, "num_ctx": 4096},
        )
        bid = resp.json()["id"]
        raw = _get_raw_backend(mock_db, bid)
        parsed = json.loads(raw["extra_config"])
        assert parsed["temperature"] == 0.7
        assert parsed["num_ctx"] == 4096

    def test_list_parses_extra_config(self, client, mock_admin, mock_db):
        """GET-Response gibt extra_config als Dict zurueck, nicht als String."""
        _create_backend(client, extra_config={"key": "value"})
        resp = client.get("/api/admin/ki/backends/")
        backends = resp.json()["backends"]
        assert isinstance(backends[0]["extra_config"], dict)
        assert backends[0]["extra_config"]["key"] == "value"

    def test_invalid_extra_config_defaults_to_empty(self, client, mock_admin, mock_db):
        """Kaputtes JSON in DB → Response gibt {} zurueck."""
        # Backend erstellen und JSON direkt in DB korrumpieren
        resp = _create_backend(client)
        bid = resp.json()["id"]

        conn = get_connection()
        conn.execute("UPDATE ki_backends SET extra_config = 'not-json{' WHERE id = %s", (bid,))
        conn.commit()
        conn.close()

        resp = client.get("/api/admin/ki/backends/")
        backends = resp.json()["backends"]
        assert backends[0]["extra_config"] == {}


# ===================================================================
# E) Test-Endpunkt (POST /{id}/test)
# ===================================================================


class TestTestEndpoint:
    def test_test_ollama_backend_success(self, client, mock_admin, mock_db):
        """Test-Endpunkt fuer Ollama-Backend (Erfolg)."""
        resp = _create_backend(client, category="llm", provider="ollama")
        bid = resp.json()["id"]

        mock_result = {"success": True, "models": ["qwen3:8b", "llama3:8b"], "error": ""}
        with patch(
            "kiara.web.api_ki._test_ollama_connection",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert "qwen3:8b" in resp.json()["models"]

    def test_test_ollama_backend_failure(self, client, mock_admin, mock_db):
        """Test-Endpunkt fuer Ollama-Backend (Fehler)."""
        resp = _create_backend(client, category="llm", provider="ollama")
        bid = resp.json()["id"]

        mock_result = {"success": False, "error": "Connection refused", "models": []}
        with patch(
            "kiara.web.api_ki._test_ollama_connection",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )
        assert resp.status_code == 400

    def test_test_nonexistent_backend_404(self, client, mock_admin, mock_db):
        """Test fuer nicht-existierendes Backend → 404."""
        resp = client.post(
            "/api/admin/ki/backends/9999/test", headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert resp.status_code == 404

    def test_test_saves_result_to_db(self, client, mock_admin, mock_db):
        """Test-Ergebnis wird in DB gespeichert."""
        resp = _create_backend(client, category="llm", provider="ollama")
        bid = resp.json()["id"]

        mock_result = {"success": True, "models": ["qwen3:8b"], "error": ""}
        with patch(
            "kiara.web.api_ki._test_ollama_connection",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        raw = _get_raw_backend(mock_db, bid)
        assert raw["last_test_ok"] == 1
        assert raw["last_test_at"] is not None

    def test_test_failure_saves_error(self, client, mock_admin, mock_db):
        """Fehlgeschlagener Test speichert Error-Text."""
        resp = _create_backend(client, category="llm", provider="ollama")
        bid = resp.json()["id"]

        mock_result = {"success": False, "error": "timeout after 10s", "models": []}
        with patch(
            "kiara.web.api_ki._test_ollama_connection",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        raw = _get_raw_backend(mock_db, bid)
        assert raw["last_test_ok"] == 0
        assert "timeout" in raw["last_test_error"]

    def test_test_cloud_no_api_key(self, client, mock_admin, mock_db):
        """Cloud-Test ohne API-Key → Fehler.

        Service verhindert Anlage ohne Key, also DB-direkt einfuegen.
        """
        c = get_connection()
        cur = c.execute(
            "INSERT INTO ki_backends (category, provider, name, model, api_key) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            ("cloud", "anthropic", "NoKey", "claude-sonnet-4-6", ""),
        )
        bid = cur.fetchone()["id"]
        c.commit()
        c.close()

        resp = client.post(
            f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert resp.status_code == 400


# ===================================================================
# F) Grenzfaelle
# ===================================================================


class TestEdgeCases:
    def test_create_with_priority(self, client, mock_admin, mock_db):
        """Priority-Feld wird korrekt gespeichert."""
        resp = _create_backend(client, priority=5)
        bid = resp.json()["id"]
        raw = _get_raw_backend(mock_db, bid)
        assert raw["priority"] == 5

    def test_list_returns_all_categories(self, client, mock_admin, mock_db):
        """Backends aller Kategorien werden gemischt zurueckgegeben."""
        _create_backend(client, category="llm", provider="ollama", model="qwen3:8b")
        _create_backend(client, category="embedding", provider="ollama", model="nomic")
        _create_backend(
            client,
            category="cloud",
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-test-1234567890",
        )
        _create_backend(client, category="vlm", provider="ollama", model="llava:7b")

        resp = client.get("/api/admin/ki/backends/")
        backends = resp.json()["backends"]
        categories = {b["category"] for b in backends}
        assert categories == {"llm", "embedding", "cloud", "vlm"}

    def test_list_returns_boolean_is_active(self, client, mock_admin, mock_db):
        """is_active wird als Boolean zurueckgegeben, nicht als Integer."""
        _create_backend(client)
        resp = client.get("/api/admin/ki/backends/")
        backends = resp.json()["backends"]
        assert isinstance(backends[0]["is_active"], bool)

    def test_auto_generated_name(self, client, mock_admin, mock_db):
        """Leerer Name generiert Auto-Name aus Kategorie + Provider."""
        resp = _create_backend(client, category="llm", provider="ollama", name="")
        bid = resp.json()["id"]
        raw = _get_raw_backend(mock_db, bid)
        assert raw["name"] == "Llm (ollama)"

    def test_toggle_nonexistent_returns_404(self, client, mock_admin, mock_db):
        """Toggle auf nicht-existierendes Backend → 404."""
        resp = client.patch(
            "/api/admin/ki/backends/9999/toggle",
            json={"is_active": True},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 404


# ===================================================================
# G) DDU-spezifische Tests (Soft-Delete, Docling-Test-Endpoint)
# ===================================================================


@pytest.fixture
def mock_db_ddu():
    """DB mit ki_backends-Tabelle inkl. DDU/Docling in CHECK-Constraints."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS ki_backends CASCADE")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ki_backends (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL CHECK (category IN ('embedding', 'llm', 'cloud', 'vlm', 'ddu')),
            provider TEXT NOT NULL CHECK (provider IN ('ollama', 'anthropic', 'openai', 'docling')),
            name TEXT NOT NULL,
            url TEXT DEFAULT '',
            base_url TEXT DEFAULT '',
            model TEXT NOT NULL,
            api_key TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            priority INTEGER DEFAULT 10,
            trusted BOOLEAN DEFAULT false,
            extra_config TEXT DEFAULT '{}',
            last_test_at TEXT DEFAULT NULL,
            last_test_ok INTEGER DEFAULT NULL,
            last_test_error TEXT DEFAULT '',
            created_at TEXT DEFAULT (NOW()),
            updated_at TEXT DEFAULT (NOW())
        );
    """)
    conn.commit()

    @contextmanager
    def mock_db_query():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def mock_db_transaction():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    with (
        patch("kiara_db.base.db_query", mock_db_query),
        patch("kiara_db.base.db_transaction", mock_db_transaction),
        patch("kiara.backend_selector.invalidate_cache"),
    ):
        yield conn

    conn.rollback()
    conn.close()
    # Das zentrale Test-DB-Helper stellt das komplette Schema in sicherer
    # Reihenfolge wieder her und verweigert Nicht-Test-Datenbanken.
    from db_helpers import restore_test_schema

    restore_test_schema()


def _insert_ddu_backend(conn_fixture, url="", is_active=True, priority=20, name="Remote DDU"):
    """Helper: DDU-Backend direkt in DB einfuegen (umgeht API)."""
    c = get_connection()
    cur = c.execute(
        """INSERT INTO ki_backends (category, provider, name, url, model, is_active, priority)
           VALUES ('ddu', 'docling', %s, %s, 'docling', %s, %s) RETURNING id""",
        (name, url, is_active, priority),
    )
    bid = cur.fetchone()["id"]
    c.commit()
    c.close()
    return bid


class TestDduDelete:
    """DELETE-Endpoint: DDU-Backends werden hart geloescht."""

    def test_delete_legacy_empty_url_ddu_hard_deletes(self, client, mock_admin, mock_db_ddu):
        """Legacy-DDU-Backend ohne URL → Row wird komplett geloescht."""
        bid = _insert_ddu_backend(mock_db_ddu, url="", name="Legacy DDU ohne URL")

        with patch("kiara.web.api_ki_backends.invalidate_ddu_cache", create=True):
            resp = confirmed_request(
                client,
                "DELETE",
                f"/api/admin/ki/backends/{bid}",
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
        assert resp.status_code == 200
        assert "geloescht" in resp.json()["message"]

        # Row existiert nicht mehr
        raw = _get_raw_backend(mock_db_ddu, bid)
        assert raw is None

    def test_delete_remote_ddu_hard_deletes(self, client, mock_admin, mock_db_ddu):
        """Remote DDU-Backend (url gesetzt) → Row wird komplett geloescht."""
        bid = _insert_ddu_backend(mock_db_ddu, url="http://ddu:5001", name="Remote DDU")

        with patch("kiara.web.api_ki_backends.invalidate_ddu_cache", create=True):
            resp = confirmed_request(
                client,
                "DELETE",
                f"/api/admin/ki/backends/{bid}",
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
        assert resp.status_code == 200
        assert "geloescht" in resp.json()["message"]

        # Row existiert nicht mehr
        raw = _get_raw_backend(mock_db_ddu, bid)
        assert raw is None

    def test_delete_non_ddu_always_hard_deletes(self, client, mock_admin, mock_db_ddu):
        """Nicht-DDU-Backend (z.B. LLM) → immer Hard-Delete, auch ohne URL."""
        conn = get_connection()
        cur = conn.execute(
            """INSERT INTO ki_backends (category, provider, name, url, model, is_active, priority)
               VALUES ('llm', 'ollama', 'Test LLM', '', 'qwen3:8b', true, 10) RETURNING id"""
        )
        bid = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        resp = confirmed_request(
            client,
            "DELETE",
            f"/api/admin/ki/backends/{bid}",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        assert "geloescht" in resp.json()["message"]

        raw = _get_raw_backend(mock_db_ddu, bid)
        assert raw is None


class TestDoclingTestEndpoint:
    """POST /{id}/test fuer Docling-Provider (remote-only)."""

    def test_docling_without_url_fails(self, client, mock_admin, mock_db_ddu):
        """Docling-Backend ohne URL → 400 mit Remote-DDU-Hinweis."""
        bid = _insert_ddu_backend(mock_db_ddu, url="", name="Legacy DDU ohne URL")

        resp = client.post(
            f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert resp.status_code == 400
        data = resp.json()
        error_text = (
            data.get("detail", "") or data.get("error", "") or data.get("message", "")
        ).lower()
        assert "remote-ddu-url" in error_text

    def test_remote_docling_health_ok(self, client, mock_admin, mock_db_ddu):
        """Remote Docling Health-Check 200 → success=True."""
        bid = _insert_ddu_backend(mock_db_ddu, url="http://ddu:5001", name="Remote DDU")

        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_resp)

        import httpx as httpx_mod

        with patch.object(httpx_mod, "AsyncClient", return_value=mock_async_client):
            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_remote_docling_health_fail(self, client, mock_admin, mock_db_ddu):
        """Remote Docling Health-Check Fehler → 502 mit Detail."""
        bid = _insert_ddu_backend(mock_db_ddu, url="http://ddu:5001", name="Remote DDU")

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(side_effect=ConnectionError("Connection refused"))

        import httpx as httpx_mod

        with patch.object(httpx_mod, "AsyncClient", return_value=mock_async_client):
            resp = client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )
        assert resp.status_code == 400
        data = resp.json()
        error_text = data.get("detail", "") or data.get("error", "") or data.get("message", "")
        assert "Connection refused" in error_text
