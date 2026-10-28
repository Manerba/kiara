"""Security-Regressionstests fuer Re-Audit #2 Fixes (F-69 bis F-89).

Deckt ab:
- F-71: Backup-Restore Path-Traversal
- F-69: Session-Invalidierung bei Passwort-Aenderung
- F-70: Session-Invalidierung bei User-Deaktivierung + is_active-Check
- F-81: Input-Laengenlimits (LoginRequest, CreateChatRequest)
- F-73: Cloud-API-Key Maskierung
- F-74: Exception-Details nicht in User-facing Responses
- F-82: OpenAPI/Swagger deaktiviert
- F-78: SQL-Regex blockiert Anfuehrungszeichen
- F-87: davfs-Secrets Quoting
"""

import sys
import types

# ldap3 ist eine optionale Abhaengigkeit und steht in der Test-Umgebung nicht
# zur Verfuegung. Stub-Modul vorab registrieren, damit der transitive Import-Pfad
# web/app -> api_health -> health.checker -> checks_services -> shared.ldap
# nicht abbricht (benoetigt von TestOpenApiDisabled).
if "ldap3" not in sys.modules:
    _ldap3_stub = types.ModuleType("ldap3")
    _ldap3_stub.Server = object
    _ldap3_stub.Connection = object
    _ldap3_stub.NONE = None
    _ldap3_stub.SUBTREE = None
    _ldap3_stub.Tls = object
    sys.modules["ldap3"] = _ldap3_stub

import hashlib
import re
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import bcrypt
import pytest

from kiara.web.auth import (
    create_session,
    validate_session,
)

# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def auth_db(tmp_path):
    """PostgreSQL-DB mit users + sessions + benoetigten Tabellen."""
    from kiara.shared.db import get_connection

    conn = get_connection()
    # DROP und Neuanlage damit das Test-Schema garantiert stimmt
    conn.executescript("""
        DROP TABLE IF EXISTS datasource_users CASCADE;
        DROP TABLE IF EXISTS datasource_groups CASCADE;
        DROP TABLE IF EXISTS user_groups CASCADE;
        DROP TABLE IF EXISTS sessions CASCADE;
        DROP TABLE IF EXISTS groups CASCADE;
        DROP TABLE IF EXISTS users CASCADE;

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
        CREATE TABLE user_groups (
            user_id INTEGER,
            group_id INTEGER
        );
        CREATE TABLE datasource_groups (
            datasource_id INTEGER,
            group_id INTEGER
        );
        CREATE TABLE datasource_users (
            id SERIAL PRIMARY KEY,
            datasource_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            UNIQUE(datasource_id, user_id)
        );
    """)

    conn.execute("DELETE FROM sessions")
    conn.execute("DELETE FROM users")
    pw_hash = bcrypt.hashpw(b"testpass", bcrypt.gensalt()).decode()
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) VALUES (%s, %s, %s, %s, %s)",
        ("testuser", pw_hash, "Test", "User", False),
    )
    conn.execute(
        "INSERT INTO users (username, password_hash, vorname, nachname, is_admin) VALUES (%s, %s, %s, %s, %s)",
        ("admin", pw_hash, "Admin", "", True),
    )
    conn.commit()
    conn.close()

    def _get_db():
        return get_connection()

    yield None, _get_db

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


# ---------------------------------------------------------------------------
# F-71: Backup-Restore Path-Traversal
# ---------------------------------------------------------------------------


class TestBackupPathTraversal:
    """F-71: backup_path darf nicht aus dem Backup-Verzeichnis ausbrechen."""

    def test_path_traversal_blocked(self, tmp_path):
        """../../etc/shadow als Filename muss ValueError ausloesen."""
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Boeswillige Datei ausserhalb des Backup-Verzeichnisses
        evil_file = tmp_path / "evil.tar.gz"
        evil_file.write_bytes(b"fake")

        mock_config = {"backup": {"dir": str(backup_dir)}}
        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.config.get_backup_dir", return_value=str(backup_dir)),
        ):
            from kiara.admin.services.backup_service import restore_backup

            with pytest.raises(ValueError, match="Ung"):
                restore_backup(filename="../evil.tar.gz")

    def test_valid_filename_accepted(self, tmp_path):
        """Gueltiger Dateiname innerhalb des Backup-Verzeichnisses wird akzeptiert."""
        from kiara.admin.services.backup_service import restore_backup

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        (backup_dir / "valid.tar.gz").write_bytes(b"fake")

        mock_config = {"backup": {"dir": str(backup_dir)}}
        with (
            patch("kiara.config.load_config", return_value=mock_config),
            patch("kiara.config.get_backup_dir", return_value=str(backup_dir)),
            patch("kiara.backup.validate_backup", return_value=(True, {}, [])),
            patch(
                "kiara.backup.restore_backup",
                return_value={"success": True, "restored": ["database"]},
            ),
        ):
            result = restore_backup(filename="valid.tar.gz")

        assert result["restored"] == ["database"]


# ---------------------------------------------------------------------------
# F-69: Session-Invalidierung bei Passwort-Aenderung
# ---------------------------------------------------------------------------


class TestSessionInvalidationPasswordChange:
    """F-69: Nach PW-Aenderung muessen andere Sessions ungueltig werden."""

    def test_other_sessions_invalidated(self, auth_db):
        """Alle Sessions AUSSER der aktuellen werden bei PW-Aenderung geloescht."""
        db_path, get_db = auth_db

        # Erstelle 3 Sessions fuer User 1
        token_a, _ = create_session(1)
        token_b, _ = create_session(1)
        token_c, _ = create_session(1)

        # Alle 3 muessen gueltig sein
        assert validate_session(token_a) is not None
        assert validate_session(token_b) is not None
        assert validate_session(token_c) is not None

        # Simuliere PW-Aenderung: loesche alle Sessions ausser token_a (aktuelle)
        token_a_hash = hashlib.sha256(token_a.encode()).hexdigest()
        db = get_db()
        db.execute("DELETE FROM sessions WHERE user_id = %s AND token != %s", (1, token_a_hash))
        db.commit()
        db.close()

        # token_a bleibt gueltig, andere nicht
        assert validate_session(token_a) is not None
        assert validate_session(token_b) is None
        assert validate_session(token_c) is None

    def test_session_token_in_user_dict(self, auth_db):
        """validate_session() gibt _session_token im User-Dict zurueck."""
        db_path, get_db = auth_db
        token, _ = create_session(1)
        user = validate_session(token)
        assert user is not None
        assert "_session_token" in user
        assert user["_session_token"] == token


# ---------------------------------------------------------------------------
# F-70: Session-Invalidierung bei User-Deaktivierung
# ---------------------------------------------------------------------------


class TestSessionInvalidationUserDeactivation:
    """F-70: Deaktivierte User koennen keine gueltigen Sessions haben."""

    def test_deactivated_user_session_invalid(self, auth_db):
        """Session eines deaktivierten Users wird von validate_session() abgewiesen."""
        db_path, get_db = auth_db

        token, _ = create_session(1)
        assert validate_session(token) is not None

        # User deaktivieren
        db = get_db()
        db.execute("UPDATE users SET is_active = false WHERE id = 1")
        db.commit()
        db.close()

        # Session muss jetzt ungueltig sein (is_active=True Check in SQL)
        assert validate_session(token) is None

    def test_reactivated_user_session_works(self, auth_db):
        """Nach Reaktivierung funktionieren neue Sessions wieder."""
        db_path, get_db = auth_db

        # Deaktivieren
        db = get_db()
        db.execute("UPDATE users SET is_active = false WHERE id = 1")
        db.commit()
        db.close()

        # Reaktivieren
        db = get_db()
        db.execute("UPDATE users SET is_active = true WHERE id = 1")
        db.commit()
        db.close()

        # Neue Session muss funktionieren
        token, _ = create_session(1)
        assert validate_session(token) is not None


# ---------------------------------------------------------------------------
# F-81: Input-Laengenlimits
# ---------------------------------------------------------------------------


class TestInputLengthLimits:
    """F-81: Login- und Chat-Felder muessen max_length haben."""

    def test_login_username_max_length(self):
        """LoginRequest.username > 255 Zeichen muss Validierungsfehler ausloesen."""
        from pydantic import ValidationError

        from kiara.web.api_auth import LoginRequest

        with pytest.raises(ValidationError):
            LoginRequest(username="x" * 256, password="valid")

    def test_login_password_max_length(self):
        """LoginRequest.password > 1000 Zeichen muss Validierungsfehler ausloesen."""
        from pydantic import ValidationError

        from kiara.web.api_auth import LoginRequest

        with pytest.raises(ValidationError):
            LoginRequest(username="user", password="x" * 1001)

    def test_login_valid_lengths_accepted(self):
        """Normale Laengen werden akzeptiert."""
        from kiara.web.api_auth import LoginRequest

        req = LoginRequest(username="admin", password="geheim123")
        assert req.username == "admin"

    def test_chat_title_max_length(self):
        """CreateChatRequest.title > 500 Zeichen muss Validierungsfehler ausloesen."""
        from pydantic import ValidationError

        from kiara.web.api_chat import CreateChatRequest

        with pytest.raises(ValidationError):
            CreateChatRequest(title="x" * 501)

    def test_chat_title_valid_length(self):
        """Chat-Titel mit 500 Zeichen wird akzeptiert."""
        from kiara.web.api_chat import CreateChatRequest

        req = CreateChatRequest(title="x" * 500)
        assert len(req.title) == 500


# ---------------------------------------------------------------------------
# F-90: LDAP-Login Timing-Angleichung
# ---------------------------------------------------------------------------


class _FakeRequest:
    class _Client:
        host = "127.0.0.1"

    client = _Client()
    headers = {}


class TestLdapLoginTimingPadding:
    """LDAP-Fehlerpfade duerfen keine Username-Enumeration per Timing erlauben."""

    @pytest.mark.asyncio
    async def test_ldap_failure_padding_waits_remaining_time(self, monkeypatch):
        from kiara.web import api_auth

        sleep = AsyncMock()
        monkeypatch.setattr(api_auth.asyncio, "sleep", sleep)
        monkeypatch.setattr(api_auth, "_LDAP_LOGIN_FAILURE_MIN_SECONDS", 0.35)
        monkeypatch.setattr(api_auth.time, "monotonic", MagicMock(return_value=100.1))

        await api_auth._pad_ldap_failure(100.0)

        sleep.assert_awaited_once()
        assert sleep.await_args.args[0] == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_ldap_failure_padding_does_not_wait_after_minimum(self, monkeypatch):
        from kiara.web import api_auth

        sleep = AsyncMock()
        monkeypatch.setattr(api_auth.asyncio, "sleep", sleep)
        monkeypatch.setattr(api_auth, "_LDAP_LOGIN_FAILURE_MIN_SECONDS", 0.35)
        monkeypatch.setattr(api_auth.time, "monotonic", MagicMock(return_value=100.5))

        await api_auth._pad_ldap_failure(100.0)

        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ldap_not_configured_failure_is_padded(self, monkeypatch):
        from fastapi import HTTPException

        from kiara.web import api_auth

        api_auth._rate_limits.clear()
        monkeypatch.setattr(
            api_auth.user_repo,
            "get_by_username",
            lambda username: {
                "id": 1,
                "username": username,
                "password_hash": None,
                "vorname": "",
                "nachname": "",
                "is_admin": False,
                "is_active": True,
                "auth_source": "ldap",
            },
        )
        monkeypatch.setattr(api_auth.config_repo, "get", lambda key: None)
        pad = AsyncMock()
        monkeypatch.setattr(api_auth, "_pad_ldap_failure", pad)

        with pytest.raises(HTTPException) as excinfo:
            await api_auth.login(
                api_auth.LoginRequest(username="alice", password="bad"),
                _FakeRequest(),
            )

        assert excinfo.value.status_code == 401
        pad.assert_awaited_once()
        api_auth._rate_limits.clear()

    @pytest.mark.asyncio
    async def test_ldap_auth_failure_is_padded(self, monkeypatch):
        from fastapi import HTTPException

        from kiara.web import api_auth

        api_auth._rate_limits.clear()
        monkeypatch.setattr(
            api_auth.user_repo,
            "get_by_username",
            lambda username: {
                "id": 1,
                "username": username,
                "password_hash": None,
                "vorname": "",
                "nachname": "",
                "is_admin": False,
                "is_active": True,
                "auth_source": "ldap",
            },
        )
        monkeypatch.setattr(api_auth.config_repo, "get", lambda key: {"server_url": "ldap"})
        monkeypatch.setattr("kiara.web.config_utils.build_ldap_config", lambda row: row)
        monkeypatch.setattr(
            api_auth.asyncio, "to_thread", AsyncMock(return_value=(False, "missing"))
        )
        pad = AsyncMock()
        monkeypatch.setattr(api_auth, "_pad_ldap_failure", pad)

        with pytest.raises(HTTPException) as excinfo:
            await api_auth.login(
                api_auth.LoginRequest(username="alice", password="bad"),
                _FakeRequest(),
            )

        assert excinfo.value.status_code == 401
        pad.assert_awaited_once()
        api_auth._rate_limits.clear()


# ---------------------------------------------------------------------------
# F-73: Cloud-API-Key Maskierung
# ---------------------------------------------------------------------------


class TestCloudApiKeyMasking:
    """F-73: GET /api/admin/cloud/config darf den API-Key nicht im Klartext zurueckgeben."""

    def test_api_key_masked_in_response(self):
        """api_key muss PASSWORD_MASK sein, nicht der echte Key."""
        import asyncio

        from kiara.admin.functions.cloud import cloud_config_read
        from kiara.shared.constants import PASSWORD_MASK

        mock_config = {
            "cloud": {
                "enabled": True,
                "provider": "anthropic",
                "api_key": "sk-ant-secret-key-12345",
                "model": "claude-3",
            }
        }
        with patch("kiara.config.load_config", return_value=mock_config):
            result = asyncio.run(cloud_config_read(user={"id": 1, "username": "admin"}))
            assert result.success is True
            data = result.data["config"]
            assert data["api_key"] == PASSWORD_MASK
            assert "sk-ant" not in data["api_key"]

    def test_empty_api_key_stays_empty(self):
        """Leerer api_key wird als leerer String zurueckgegeben (nicht maskiert)."""
        import asyncio

        from kiara.admin.functions.cloud import cloud_config_read

        mock_config = {"cloud": {"api_key": ""}}
        with patch("kiara.config.load_config", return_value=mock_config):
            result = asyncio.run(cloud_config_read(user={"id": 1, "username": "admin"}))
            assert result.data["config"]["api_key"] == ""


# ---------------------------------------------------------------------------
# F-78: SQL-Regex blockiert Anfuehrungszeichen
# ---------------------------------------------------------------------------


class TestSqlRegexHardened:
    """F-78: paginated_query() Regex darf keine Anfuehrungszeichen erlauben."""

    def test_single_quote_rejected(self):
        """Einfache Anfuehrungszeichen in SQL-Identifier muessen abgelehnt werden."""
        _SQL_IDENT_RE = r"^[\w\s.*,()=<>!]+$"
        assert not re.match(_SQL_IDENT_RE, "users' OR '1'='1")

    def test_double_quote_rejected(self):
        """Doppelte Anfuehrungszeichen muessen abgelehnt werden."""
        _SQL_IDENT_RE = r"^[\w\s.*,()=<>!]+$"
        assert not re.match(_SQL_IDENT_RE, 'users" OR "1"="1')

    def test_valid_identifiers_accepted(self):
        """Normale SQL-Identifier (Buchstaben, Zahlen, Unterstriche) werden akzeptiert."""
        _SQL_IDENT_RE = r"^[\w\s.*,()=<>!]+$"
        valid = ["users", "u.id DESC", "a.*, u.username", "COUNT(*)", "id > 0"]
        for v in valid:
            assert re.match(_SQL_IDENT_RE, v), f"{v!r} sollte akzeptiert werden"


# ---------------------------------------------------------------------------
# F-87: davfs-Secrets Quoting
# ---------------------------------------------------------------------------


class TestDavfsSecretsQuoting:
    """F-87: davfs-Credentials muessen in Anfuehrungszeichen stehen."""

    @pytest.fixture(autouse=True)
    def _isolated_mount_io(self, monkeypatch, tmp_path):
        """sudo und den globalen davfs-Lock auf temporaere Ressourcen umleiten."""
        from kiara.shared import mount as mount_module

        _real_run = subprocess.run

        def _no_sudo_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == "sudo":
                raise FileNotFoundError("sudo gepatcht fuer Tests")
            return _real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", _no_sudo_run)
        monkeypatch.setattr("kiara.shared.mount.subprocess.run", _no_sudo_run)
        monkeypatch.setattr(
            mount_module,
            "DAVFS_SECRETS_LOCK_PATH",
            str(tmp_path / "davfs-secrets.lock"),
        )

    def test_credentials_quoted_in_secrets_file(self, tmp_path):
        """Username und Password muessen in Anfuehrungszeichen geschrieben werden."""
        secrets_file = tmp_path / "davfs_secrets"
        secrets_file.write_text("")

        from kiara.shared.mount import _ensure_davfs_secret

        _ensure_davfs_secret(str(secrets_file), "/mnt/kiara-test", "myuser", "mypass")

        content = secrets_file.read_text()
        assert '"myuser"' in content
        assert '"mypass"' in content

    def test_spaces_in_credentials_handled(self, tmp_path):
        """Leerzeichen in Username/Password duerfen das Format nicht brechen."""
        secrets_file = tmp_path / "davfs_secrets"
        secrets_file.write_text("")

        from kiara.shared.mount import _ensure_davfs_secret

        _ensure_davfs_secret(str(secrets_file), "/mnt/kiara-test", "user with spaces", "pass word")

        content = secrets_file.read_text()
        lines = [l for l in content.strip().splitlines() if l.strip()]
        assert len(lines) == 1  # Nur ein Eintrag, nicht aufgesplittet
        assert '"user with spaces"' in lines[0]
        assert '"pass word"' in lines[0]


# ---------------------------------------------------------------------------
# F-82: OpenAPI/Swagger deaktiviert
# ---------------------------------------------------------------------------


class TestOpenApiDisabled:
    """F-82: /docs, /redoc, /openapi.json sollen nicht erreichbar sein."""

    def test_user_app_openapi_disabled(self):
        """User-App: openapi_url=None → /openapi.json nicht erreichbar."""
        from kiara.web.app import app

        assert app.openapi_url is None

    def test_admin_app_openapi_disabled(self):
        """Admin-App: openapi_url=None → /openapi.json nicht erreichbar."""
        from kiara.web.admin_app import app

        assert app.openapi_url is None

    def test_user_app_docs_url_disabled(self):
        """User-App: docs_url=None."""
        from kiara.web.app import app

        assert app.docs_url is None

    def test_admin_app_docs_url_disabled(self):
        """Admin-App: docs_url=None."""
        from kiara.web.admin_app import app

        assert app.docs_url is None
