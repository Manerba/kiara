"""Tests fuer Session-Token-Hashing (F-68/F-83) und Cloud-API-Key-Verschluesselung (F-72).

Teil 1: Session-Token-Hashing
    - create_session speichert SHA-256-Hash in DB, gibt Klartext zurueck
    - validate_session liefert Klartext-Token im User-Dict
    - delete_session findet gehashten Token
    - Passwort-Aenderung invalidiert andere Sessions ueber Hash

Teil 2: Cloud-API-Key-Verschluesselung
    - save_cloud_config verschluesselt neue Keys, PASSWORD_MASK bewahrt bestehende
    - get_ki_config maskiert API-Keys, save_ki_config verschluesselt

Migration-Tests fuer _migrate_session_token_hashing und
_migrate_cloud_api_key_encryption entfernt — beide Funktionen sind Teil des
geloeschten db_migrations-Moduls (Schema-Reset-Sprint).
"""

import hashlib
from unittest.mock import patch

import bcrypt
import pytest

from kiara.shared.constants import PASSWORD_MASK
from kiara.web.auth import (
    create_session,
    delete_session,
    validate_session,
)
from tests.admin_confirm import confirmed_request

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _tmp_secret_key(tmp_path):
    """Temporaerer Secret-Key fuer Fernet-Operationen (kein /var/lib/kiara noetig)."""
    key_file = tmp_path / "secret.key"
    key_file.write_text("deadbeef" * 8)
    key_file.chmod(0o600)
    with patch("kiara.shared.crypto.SECRET_KEY_PATH", key_file):
        yield key_file


@pytest.fixture()
def auth_db():
    """Erstellt Tabellen in PostgreSQL mit users-, sessions- und Berechtigungs-Tabellen + Admin-User."""
    from kiara.shared.db import get_connection

    conn = get_connection()

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
            created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS'))
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
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
            id SERIAL PRIMARY KEY,
            datasource_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            UNIQUE(datasource_id, user_id)
        );
    """)
    conn.commit()

    # Bestehende Testdaten aufraeumen
    conn.execute("DELETE FROM datasource_users")
    conn.execute("DELETE FROM datasource_groups")
    conn.execute("DELETE FROM user_groups")
    conn.execute("DELETE FROM sessions")
    conn.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE")
    conn.commit()

    admin_hash = bcrypt.hashpw(b"admin", bcrypt.gensalt()).decode()
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) VALUES (%s, %s, %s, %s, %s)",
        ("admin", admin_hash, "Administrator", "", True),
    )

    user_hash = bcrypt.hashpw(b"userpass", bcrypt.gensalt()).decode()
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) VALUES (%s, %s, %s, %s, %s)",
        ("testuser", user_hash, "Test", "User", False),
    )

    conn.commit()

    # Admin-User-ID ermitteln
    admin_row = conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
    admin_id = admin_row["id"]

    conn.close()

    def _get_db():
        return get_connection()

    yield admin_id, _get_db

    # Cleanup
    conn = get_connection()
    conn.execute("DELETE FROM sessions")
    conn.execute("TRUNCATE TABLE users RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()


# =========================================================================
# Teil 1: Session-Token-Hashing
# =========================================================================


class TestSessionTokenHashing:
    """Tests fuer F-68/F-83: Session-Tokens werden als SHA-256-Hash in der DB gespeichert."""

    def test_session_token_stored_as_hash(self, auth_db):
        """Token in der DB ist SHA-256-Hash, nicht Klartext."""
        admin_id, get_db = auth_db
        token, _ = create_session(admin_id)

        db = get_db()
        row = db.execute("SELECT token FROM sessions").fetchone()
        db.close()

        # DB-Wert ist NICHT der Klartext-Token
        assert row["token"] != token
        # DB-Wert IST der SHA-256-Hash des Tokens
        expected_hash = hashlib.sha256(token.encode()).hexdigest()
        assert row["token"] == expected_hash

    def test_session_token_in_user_dict_is_plaintext(self, auth_db):
        """validate_session gibt den Klartext-Token im User-Dict zurueck (fuer Cookie-Roundtrip)."""
        admin_id, _get_db = auth_db
        token, _ = create_session(admin_id)
        user = validate_session(token)

        assert user is not None
        assert user["_session_token"] == token
        # Sicherheitscheck: _session_token ist nicht der Hash
        assert user["_session_token"] != hashlib.sha256(token.encode()).hexdigest()

    def test_delete_session_finds_hashed_token(self, auth_db):
        """delete_session findet und loescht die Session ueber den gehashten Token."""
        admin_id, get_db = auth_db
        token, _ = create_session(admin_id)

        # Session existiert
        assert validate_session(token) is not None

        # Loeschen mit Klartext-Token
        delete_session(token)

        # Session ist weg
        assert validate_session(token) is None

        # DB ist leer
        db = get_db()
        count = db.execute("SELECT COUNT(*) AS cnt FROM sessions").fetchone()["cnt"]
        db.close()
        assert count == 0

    def test_change_password_deletes_other_sessions_by_hash(self, auth_db):
        """Passwort-Aenderung loescht andere Sessions des Users (ueber gehashte Tokens).

        Simuliert die Logik aus api_auth.py:
        DELETE FROM sessions WHERE user_id = ? AND token != ? (mit gehashem eigenen Token)
        """
        admin_id, get_db = auth_db

        # 3 Sessions fuer den Admin erstellen
        token_a, _ = create_session(admin_id)
        token_b, _ = create_session(admin_id)
        token_c, _ = create_session(admin_id)

        # Alle 3 sind gueltig
        assert validate_session(token_a) is not None
        assert validate_session(token_b) is not None
        assert validate_session(token_c) is not None

        # Passwort-Aenderung: Behalte token_a, loesche alle anderen
        token_a_hash = hashlib.sha256(token_a.encode()).hexdigest()
        db = get_db()
        db.execute(
            "DELETE FROM sessions WHERE user_id = %s AND token != %s",
            (admin_id, token_a_hash),
        )
        db.commit()
        db.close()

        # token_a funktioniert noch
        assert validate_session(token_a) is not None
        # token_b und token_c sind geloescht
        assert validate_session(token_b) is None
        assert validate_session(token_c) is None


# ENTFERNT: Klassen ab Zeile 222 (Migration-Tests obsolet)
class TestSaveCloudConfig:
    """Tests fuer den POST /api/admin/cloud/config Endpoint (API-Key-Verschluesselung)."""

    def _make_cloud_app(self):
        """Erstellt eine FastAPI-App mit Admin-Registry-Routen fuer Cloud-Endpoints."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiara.admin.registry import admin_registry, load_admin_functions
        from kiara.admin.router import generate_admin_routes
        from kiara.web.auth import get_admin_user

        load_admin_functions()
        app = FastAPI()
        generate_admin_routes(app, admin_registry)
        app.dependency_overrides[get_admin_user] = lambda: {
            "id": 1,
            "username": "admin",
            "is_admin": True,
        }
        return TestClient(app)

    def test_save_cloud_config_encrypts_new_key(self, _tmp_secret_key):
        """Neuer API-Key wird beim Speichern verschluesselt (Fernet, gAAAAA-Prefix)."""
        from kiara.shared.crypto import is_encrypted

        saved_config = {}

        def mock_save(cfg):
            saved_config.update(cfg)

        mock_config = {"cloud": {}}

        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.config.save_config", side_effect=mock_save),
        ):
            client = self._make_cloud_app()
            resp = confirmed_request(
                client,
                "POST",
                "/api/admin/cloud/config",
                json={
                    "enabled": True,
                    "provider": "anthropic",
                    "api_key": "sk-ant-my-secret-key-12345",
                    "model": "claude-sonnet-4-6",
                },
            )

            assert resp.status_code == 200

        # Gespeicherter Key ist verschluesselt
        stored_key = saved_config.get("cloud", {}).get("api_key", "")
        assert stored_key.startswith("gAAAAA"), "API-Key muss als Fernet-Token gespeichert werden"
        assert stored_key != "sk-ant-my-secret-key-12345"
        assert is_encrypted(stored_key)

    def test_save_cloud_config_password_mask_keeps_encrypted_key(self, _tmp_secret_key):
        """PASSWORD_MASK bei bereits verschluesseltem Key: bestehender Key bleibt unveraendert."""
        from kiara.shared.crypto import encrypt

        existing_encrypted = encrypt("sk-ant-existing-key")
        saved_config = {}

        def mock_save(cfg):
            saved_config.update(cfg)

        mock_config = {"cloud": {"api_key": existing_encrypted}}

        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.config.save_config", side_effect=mock_save),
        ):
            client = self._make_cloud_app()
            resp = confirmed_request(
                client,
                "POST",
                "/api/admin/cloud/config",
                json={
                    "enabled": True,
                    "provider": "anthropic",
                    "api_key": PASSWORD_MASK,
                    "model": "claude-sonnet-4-6",
                },
            )

            assert resp.status_code == 200

        # Key bleibt unveraendert (bereits verschluesselt)
        stored_key = saved_config.get("cloud", {}).get("api_key", "")
        assert stored_key == existing_encrypted

    def test_save_cloud_config_password_mask_encrypts_plaintext_key(self, _tmp_secret_key):
        """PASSWORD_MASK bei altem Klartext-Key: opportunistische Verschluesselung."""
        from kiara.shared.crypto import is_encrypted

        saved_config = {}

        def mock_save(cfg):
            saved_config.update(cfg)

        mock_config = {"cloud": {"api_key": "sk-ant-plaintext-key"}}

        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.config.save_config", side_effect=mock_save),
        ):
            client = self._make_cloud_app()
            resp = confirmed_request(
                client,
                "POST",
                "/api/admin/cloud/config",
                json={
                    "enabled": True,
                    "provider": "anthropic",
                    "api_key": PASSWORD_MASK,
                    "model": "claude-sonnet-4-6",
                },
            )

            assert resp.status_code == 200

        # Klartext-Key wurde opportunistisch verschluesselt
        stored_key = saved_config.get("cloud", {}).get("api_key", "")
        assert is_encrypted(stored_key), (
            "Klartext-Key haette bei PASSWORD_MASK verschluesselt werden muessen"
        )
        assert stored_key != "sk-ant-plaintext-key"

    def test_save_cloud_config_empty_key(self, _tmp_secret_key):
        """Leerer API-Key bleibt leer (kein Verschluesselungsversuch)."""
        saved_config = {}

        def mock_save(cfg):
            saved_config.update(cfg)

        mock_config = {"cloud": {}}

        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.config.save_config", side_effect=mock_save),
        ):
            client = self._make_cloud_app()
            resp = confirmed_request(
                client,
                "POST",
                "/api/admin/cloud/config",
                json={
                    "enabled": False,
                    "provider": "anthropic",
                    "api_key": "",
                    "model": "claude-sonnet-4-6",
                },
            )

            assert resp.status_code == 200

        stored_key = saved_config.get("cloud", {}).get("api_key", "")
        assert stored_key == ""

    def test_partial_cloud_config_update_preserves_existing_secret_via_http(
        self,
        _tmp_secret_key,
    ):
        """Partieller REST-Body darf bestehenden API-Key nicht loeschen."""
        from kiara.shared.crypto import encrypt

        existing_encrypted = encrypt("sk-ant-existing-key")
        saved_config = {}

        def mock_save(cfg):
            saved_config.update(cfg)

        mock_config = {
            "cloud": {
                "enabled": True,
                "provider": "anthropic",
                "api_key": existing_encrypted,
                "model": "claude-sonnet-4-6",
                "mode": "cloud",
                "routing": {"mode": "cloud", "cloud_keywords": ["extern"]},
                "budget": {"monthly_limit_eur": 50, "alert_threshold_pct": 70},
            }
        }

        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.config.save_config", side_effect=mock_save),
        ):
            client = self._make_cloud_app()
            resp = confirmed_request(
                client,
                "POST",
                "/api/admin/cloud/config",
                json={
                    "monthly_limit_eur": 75,
                },
            )

        assert resp.status_code == 200
        saved_cloud = saved_config["cloud"]
        assert saved_cloud["enabled"] is True
        assert saved_cloud["provider"] == "anthropic"
        assert saved_cloud["api_key"] == existing_encrypted
        assert saved_cloud["model"] == "claude-sonnet-4-6"
        assert saved_cloud["mode"] == "cloud"
        assert saved_cloud["routing"]["mode"] == "cloud"
        assert saved_cloud["routing"]["cloud_keywords"] == ["extern"]
        assert saved_cloud["budget"]["monthly_limit_eur"] == 75
        assert saved_cloud["budget"]["alert_threshold_pct"] == 70


class TestKiConfigEndpointEncryption:
    """Tests fuer GET/POST /api/admin/ki/config: API-Key-Maskierung und -Verschluesselung."""

    def _make_ki_app(self):
        """Erstellt eine FastAPI-Test-App mit KI-Router + Admin-Registry-Routen."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiara.admin.registry import admin_registry, load_admin_functions
        from kiara.admin.router import generate_admin_routes
        from kiara.web.api_ki import router
        from kiara.web.auth import get_admin_user

        load_admin_functions()
        app = FastAPI()
        app.include_router(router)
        generate_admin_routes(app, admin_registry)
        app.dependency_overrides[get_admin_user] = lambda: {
            "id": 1,
            "username": "admin",
            "is_admin": True,
        }
        return TestClient(app)

    def test_get_ki_config_masks_api_key(self, _tmp_secret_key):
        """GET /api/admin/ki/config gibt PASSWORD_MASK statt echtem Key zurueck."""
        from kiara.shared.crypto import encrypt

        encrypted_key = encrypt("sk-ant-secret")

        mock_row = {
            "id": 1,
            "embedding_url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "llm_url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
            "prompt_template": "",
            "updated_at": "2026-03-19",
            "cloud_enabled": True,
            "cloud_provider": "anthropic",
            "cloud_api_key": encrypted_key,
            "cloud_model": "claude-sonnet-4-6",
            "cloud_base_url": "",
            "cloud_mode": "auto",
            "anon_enabled": True,
            "anon_erp_enabled": False,
            "anon_erp_path": "/var/lib/kiara/stammdaten.csv",
            "anon_ldap_enabled": True,
            "anon_word_boundaries": 1,
            "anon_min_term_length": 3,
            "anon_excluded_terms": "[]",
            "anon_custom_patterns": "[]",
            "anon_audit_log": 1,
            "anon_custom_entries": "[]",
            "anon_llm_enabled": False,
            "anon_llm_model": "",
            "anon_llm_temperature": "0.1",
            "anon_llm_confidence": "0.7",
            "anon_llm_timeout": 10,
            "anon_llm_cache_size": 256,
            "anon_llm_num_ctx": 4096,
            "agent_system_prompt": "",
        }

        # MagicMock der sich wie ein DB-Row verhaelt
        class FakeRow(dict):
            pass

        fake_row = FakeRow(mock_row)

        with patch("kiara.web.api_ki.config_repo") as mock_config_repo_inner:
            mock_config_repo_inner.get.return_value = fake_row

            client = self._make_ki_app()
            resp = client.get("/api/admin/ki/config")

        assert resp.status_code == 200
        data = resp.json()["config"]
        assert data["cloud_api_key"] == PASSWORD_MASK
        # Echter Key darf NICHT im Response stehen
        assert encrypted_key not in str(data)

    def test_save_ki_config_encrypts_new_key(self, _tmp_secret_key):
        """POST mit neuem cloud_api_key verschluesselt den Key in der DB."""
        from kiara.shared.crypto import is_encrypted

        stored_fields = {}

        def mock_upsert(db, table, fields):
            stored_fields.update(fields)

        with (
            patch("kiara.web.api_ki.config_repo") as mock_cr,
            patch("kiara.web.api_ki.load_config", return_value={"ollama": {}, "cloud": {}}),
            patch("kiara.web.api_ki.save_config"),
        ):

            def capture_set(table, fields):
                stored_fields.update(fields)

            mock_cr.set.side_effect = capture_set
            mock_cr.get.return_value = None

            client = self._make_ki_app()
            resp = client.post(
                "/api/admin/ki/config",
                json={
                    "embedding_url": "http://localhost:11434",
                    "embedding_model": "nomic-embed-text",
                    "llm_url": "http://localhost:11434",
                    "llm_model": "qwen3:8b",
                    "cloud_api_key": "sk-ant-brand-new-key",
                },
            )

        assert resp.status_code == 200
        stored_key = stored_fields.get("cloud_api_key", "")
        assert is_encrypted(stored_key), (
            "Neuer Cloud-API-Key muss verschluesselt gespeichert werden"
        )
        assert stored_key != "sk-ant-brand-new-key"

    def test_save_ki_config_cloud_mode_uses_runtime_values(self, _tmp_secret_key):
        """cloud_mode speichert die Runtime-Werte auto/local/cloud in DB und YAML."""
        stored_fields = {}
        saved_config = {}

        with (
            patch("kiara.web.api_ki.config_repo") as mock_cr,
            patch("kiara.web.api_ki.load_config", return_value={"ollama": {}, "cloud": {}}),
            patch("kiara.web.api_ki.save_config", side_effect=lambda cfg: saved_config.update(cfg)),
        ):
            mock_cr.get.return_value = None
            mock_cr.set.side_effect = lambda table, fields: stored_fields.update(fields)

            client = self._make_ki_app()
            resp = client.post(
                "/api/admin/ki/config",
                json={
                    "embedding_url": "http://localhost:11434",
                    "embedding_model": "nomic-embed-text",
                    "llm_url": "http://localhost:11434",
                    "llm_model": "qwen3:8b",
                    "cloud_enabled": True,
                    "cloud_mode": "cloud",
                },
            )

        assert resp.status_code == 200
        assert stored_fields["cloud_mode"] == "cloud"
        assert saved_config["cloud"]["mode"] == "cloud"

    @pytest.mark.parametrize("old_mode", ["always", "fallback"])
    def test_save_ki_config_rejects_old_cloud_modes(self, old_mode):
        """Alte API-Werte werden nicht mehr akzeptiert."""
        client = self._make_ki_app()
        resp = client.post(
            "/api/admin/ki/config",
            json={
                "embedding_url": "http://localhost:11434",
                "embedding_model": "nomic-embed-text",
                "llm_url": "http://localhost:11434",
                "llm_model": "qwen3:8b",
                "cloud_mode": old_mode,
            },
        )

        assert resp.status_code == 422

    def test_save_ki_config_password_mask_keeps_key(self, _tmp_secret_key):
        """POST mit PASSWORD_MASK behaelt den bestehenden verschluesselten Key."""
        from kiara.shared.crypto import encrypt

        existing_encrypted = encrypt("sk-ant-keep-me")
        stored_fields = {}

        def mock_upsert(db, table, fields):
            stored_fields.update(fields)

        class FakeRow(dict):
            pass

        mock_existing = FakeRow({"cloud_api_key": existing_encrypted})

        with (
            patch("kiara.web.api_ki.config_repo") as mock_cr,
            patch("kiara.web.api_ki.load_config", return_value={"ollama": {}, "cloud": {}}),
            patch("kiara.web.api_ki.save_config"),
        ):
            mock_cr.get.return_value = mock_existing

            def capture_set(table, fields):
                stored_fields.update(fields)

            mock_cr.set.side_effect = capture_set

            client = self._make_ki_app()
            resp = client.post(
                "/api/admin/ki/config",
                json={
                    "embedding_url": "http://localhost:11434",
                    "embedding_model": "nomic-embed-text",
                    "llm_url": "http://localhost:11434",
                    "llm_model": "qwen3:8b",
                    "cloud_api_key": PASSWORD_MASK,
                },
            )

        assert resp.status_code == 200
        stored_key = stored_fields.get("cloud_api_key", "")
        assert stored_key == existing_encrypted


# ENTFERNT: Klassen ab Zeile 632 (Migration-Tests obsolet)
class TestCryptoRoundtrip:
    """Encrypt → Decrypt Roundtrip-Validierung."""

    def test_decrypt_recovers_original_from_saved_key(self, _tmp_secret_key):
        """encrypt() → decrypt() liefert den Original-Key zurueck (Roundtrip)."""
        from kiara.shared.crypto import decrypt, encrypt, is_encrypted

        original = "sk-ant-api123-roundtrip-test"
        encrypted = encrypt(original)

        assert is_encrypted(encrypted)
        assert encrypted != original
        assert decrypt(encrypted) == original

    def test_decrypt_recovers_empty_string(self, _tmp_secret_key):
        """Leerer Wert bleibt leer (encrypt/decrypt bypass bei leerem Input)."""
        from kiara.shared.crypto import decrypt, encrypt

        assert encrypt("") == ""
        assert decrypt("") == ""


# ENTFERNT: Klassen ab Zeile 779 (Migration-Tests obsolet)
class TestSaveKiConfigOpportunisticEncryption:
    """save_ki_config: PASSWORD_MASK + Klartext-Key → opportunistische Verschluesselung."""

    def _make_ki_app(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiara.web.api_ki import router
        from kiara.web.auth import get_admin_user

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_admin_user] = lambda: {
            "id": 1,
            "username": "admin",
            "is_admin": True,
        }
        return TestClient(app)

    def test_save_ki_config_password_mask_encrypts_plaintext_key(self, _tmp_secret_key):
        """PASSWORD_MASK mit Klartext-Key in DB → opportunistische Verschluesselung."""
        from kiara.shared.crypto import is_encrypted

        stored_fields = {}

        def mock_upsert(db, table, fields):
            stored_fields.update(fields)

        class FakeRow(dict):
            pass

        # DB hat Klartext-Key (vor Migration)
        mock_existing = FakeRow({"cloud_api_key": "sk-ant-old-plaintext-key"})

        with (
            patch("kiara.web.api_ki.config_repo") as mock_cr,
            patch("kiara.web.api_ki.load_config", return_value={"ollama": {}, "cloud": {}}),
            patch("kiara.web.api_ki.save_config"),
        ):
            mock_cr.get.return_value = mock_existing

            def capture_set(table, fields):
                stored_fields.update(fields)

            mock_cr.set.side_effect = capture_set

            client = self._make_ki_app()
            resp = client.post(
                "/api/admin/ki/config",
                json={
                    "embedding_url": "http://localhost:11434",
                    "embedding_model": "nomic-embed-text",
                    "llm_url": "http://localhost:11434",
                    "llm_model": "qwen3:8b",
                    "cloud_api_key": PASSWORD_MASK,
                },
            )

        assert resp.status_code == 200
        stored_key = stored_fields.get("cloud_api_key", "")
        assert is_encrypted(stored_key), (
            "Klartext-Key haette opportunistisch verschluesselt werden muessen"
        )
        assert stored_key != "sk-ant-old-plaintext-key"

    def test_save_ki_config_sync_to_yaml_gets_encrypted_key(self, _tmp_secret_key):
        """save_ki_config uebergibt den verschluesselten Key an _sync_to_yaml (nicht Klartext)."""
        from kiara.shared.crypto import is_encrypted

        yaml_cloud_key = []

        original_sync = None

        # Fange den cloud_api_key-Parameter ab der an _sync_to_yaml uebergeben wird
        def capture_sync(*args, **kwargs):
            yaml_cloud_key.append(kwargs.get("cloud_api_key", ""))

        stored_fields = {}

        def mock_upsert(db, table, fields):
            stored_fields.update(fields)

        with (
            patch("kiara.web.api_ki.config_repo") as mock_cr,
            patch("kiara.web.api_ki._sync_to_yaml", side_effect=capture_sync),
            patch("kiara.web.api_ki.load_config", return_value={"ollama": {}, "cloud": {}}),
        ):
            mock_cr.get.return_value = None

            def capture_set(table, fields):
                stored_fields.update(fields)

            mock_cr.set.side_effect = capture_set

            client = self._make_ki_app()
            resp = client.post(
                "/api/admin/ki/config",
                json={
                    "embedding_url": "http://localhost:11434",
                    "embedding_model": "nomic-embed-text",
                    "llm_url": "http://localhost:11434",
                    "llm_model": "qwen3:8b",
                    "cloud_api_key": "sk-ant-new-key-for-yaml",
                },
            )

        assert resp.status_code == 200
        assert len(yaml_cloud_key) == 1
        # _sync_to_yaml muss den verschluesselten Key bekommen, nicht den Klartext
        assert is_encrypted(yaml_cloud_key[0]), "_sync_to_yaml muss verschluesselten Key erhalten"
        assert yaml_cloud_key[0] != "sk-ant-new-key-for-yaml"


class TestGetKiConfigEmptyKey:
    """get_ki_config: Leerer Key → leerer String (nicht PASSWORD_MASK)."""

    def _make_ki_app_with_registry(self):
        """Erstellt eine FastAPI-App mit KI-Router + Admin-Registry-Routen."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiara.admin.registry import admin_registry, load_admin_functions
        from kiara.admin.router import generate_admin_routes
        from kiara.web.api_ki import router
        from kiara.web.auth import get_admin_user

        load_admin_functions()
        app = FastAPI()
        app.include_router(router)
        generate_admin_routes(app, admin_registry)
        app.dependency_overrides[get_admin_user] = lambda: {
            "id": 1,
            "username": "admin",
            "is_admin": True,
        }
        return TestClient(app)

    def test_get_ki_config_empty_key_returns_empty_string(self, _tmp_secret_key):
        """Kein cloud_api_key → Antwort ist leerer String, nicht PASSWORD_MASK."""

        class FakeRow(dict):
            pass

        mock_row = FakeRow(
            {
                "id": 1,
                "embedding_url": "",
                "embedding_model": "",
                "llm_url": "",
                "llm_model": "",
                "prompt_template": "",
                "updated_at": "",
                "cloud_enabled": False,
                "cloud_provider": "anthropic",
                "cloud_api_key": "",  # Leer!
                "cloud_model": "",
                "cloud_base_url": "",
                "cloud_mode": "auto",
                "anon_enabled": True,
                "anon_erp_enabled": False,
                "anon_erp_path": "/var/lib/kiara/stammdaten.csv",
                "anon_ldap_enabled": True,
                "anon_word_boundaries": 1,
                "anon_min_term_length": 3,
                "anon_excluded_terms": "[]",
                "anon_custom_patterns": "[]",
                "anon_audit_log": 1,
                "anon_custom_entries": "[]",
                "anon_llm_enabled": False,
                "anon_llm_model": "",
                "anon_llm_temperature": "0.1",
                "anon_llm_confidence": "0.7",
                "anon_llm_timeout": 10,
                "anon_llm_cache_size": 256,
                "anon_llm_num_ctx": 4096,
                "agent_system_prompt": "",
            }
        )

        with patch("kiara.web.api_ki.config_repo") as mock_cr:
            mock_cr.get.return_value = mock_row

            client = self._make_ki_app_with_registry()
            resp = client.get("/api/admin/ki/config")

        assert resp.status_code == 200
        assert resp.json()["config"]["cloud_api_key"] == ""

    def test_get_ki_config_none_key_returns_empty_string(self, _tmp_secret_key):
        """cloud_api_key=None → Antwort ist leerer String, nicht PASSWORD_MASK."""

        class FakeRow(dict):
            pass

        mock_row = FakeRow(
            {
                "id": 1,
                "embedding_url": "",
                "embedding_model": "",
                "llm_url": "",
                "llm_model": "",
                "prompt_template": "",
                "updated_at": "",
                "cloud_enabled": False,
                "cloud_provider": "anthropic",
                "cloud_api_key": None,  # None!
                "cloud_model": "",
                "cloud_base_url": "",
                "cloud_mode": "auto",
                "anon_enabled": True,
                "anon_erp_enabled": False,
                "anon_erp_path": "/var/lib/kiara/stammdaten.csv",
                "anon_ldap_enabled": True,
                "anon_word_boundaries": 1,
                "anon_min_term_length": 3,
                "anon_excluded_terms": "[]",
                "anon_custom_patterns": "[]",
                "anon_audit_log": 1,
                "anon_custom_entries": "[]",
                "anon_llm_enabled": False,
                "anon_llm_model": "",
                "anon_llm_temperature": "0.1",
                "anon_llm_confidence": "0.7",
                "anon_llm_timeout": 10,
                "anon_llm_cache_size": 256,
                "anon_llm_num_ctx": 4096,
                "agent_system_prompt": "",
            }
        )

        with patch("kiara.web.api_ki.config_repo") as mock_cr:
            mock_cr.get.return_value = mock_row

            client = self._make_ki_app_with_registry()
            resp = client.get("/api/admin/ki/config")

        assert resp.status_code == 200
        assert resp.json()["config"]["cloud_api_key"] == ""


class TestSessionTokenHashFormat:
    """Session-Token Hash-Format-Validierung."""

    def test_stored_hash_is_64_hex_chars(self, auth_db):
        """Gespeicherter Hash ist exakt 64 Hex-Zeichen (SHA-256 Hexdigest)."""
        import re

        _db_path, get_db = auth_db
        token, _ = create_session(1)

        db = get_db()
        row = db.execute("SELECT token FROM sessions").fetchone()
        db.close()

        stored = row["token"]
        assert len(stored) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", stored), "Hash muss 64 lowercase Hex-Zeichen sein"

    def test_token_and_hash_both_64_chars_but_different(self, auth_db):
        """Token (token_hex(32)) und Hash (sha256.hexdigest) sind beide 64 Zeichen — aber verschieden."""
        _db_path, get_db = auth_db
        token, _ = create_session(1)

        db = get_db()
        row = db.execute("SELECT token FROM sessions").fetchone()
        db.close()

        # Beide sind 64 Zeichen
        assert len(token) == 64
        assert len(row["token"]) == 64
        # Aber NICHT identisch (Hash != Klartext)
        assert token != row["token"]


class TestMultiUserSessionIndependence:
    """Session-Hashing mit mehreren Usern — keine Cross-User-Effekte."""

    def test_sessions_independent_across_users(self, auth_db):
        """Sessions verschiedener User sind nach Hashing vollstaendig unabhaengig."""
        _db_path, get_db = auth_db

        # Sessions fuer User 1 (admin) und User 2 (testuser)
        token_user1, _ = create_session(1)
        token_user2, _ = create_session(2)

        # Beide Sessions sind gueltig
        user1 = validate_session(token_user1)
        user2 = validate_session(token_user2)
        assert user1 is not None
        assert user2 is not None
        assert user1["id"] == 1
        assert user2["id"] == 2

        # Loeschen von User 1 Session beeinflusst User 2 nicht
        delete_session(token_user1)
        assert validate_session(token_user1) is None
        assert validate_session(token_user2) is not None

    def test_change_password_only_affects_own_sessions(self, auth_db):
        """Session-Invalidierung bei PW-Aenderung loescht NUR eigene anderen Sessions."""
        _db_path, get_db = auth_db

        # User 1: 2 Sessions, User 2: 1 Session
        token_u1_a, _ = create_session(1)
        token_u1_b, _ = create_session(1)
        token_u2, _ = create_session(2)

        # PW-Aenderung User 1: behalte token_u1_a, loesche andere von User 1
        token_u1_a_hash = hashlib.sha256(token_u1_a.encode()).hexdigest()
        db = get_db()
        db.execute(
            "DELETE FROM sessions WHERE user_id = %s AND token != %s",
            (1, token_u1_a_hash),
        )
        db.commit()
        db.close()

        # User 1: nur token_a bleibt
        assert validate_session(token_u1_a) is not None
        assert validate_session(token_u1_b) is None
        # User 2: komplett unbeeinflusst
        assert validate_session(token_u2) is not None
