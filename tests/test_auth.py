"""Tests fuer Authentifizierung: Login, Session, Logout."""

import hashlib

import bcrypt
import pytest

from kiara.shared.db import get_connection
from kiara.web.auth import (
    create_session,
    delete_session,
    hash_password,
    sign_token,
    unsign_token,
    validate_session,
    verify_password,
)


@pytest.fixture()
def auth_db():
    """Erstellt PostgreSQL-Tabellen mit users- und sessions-Tabelle + Admin-User."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS user_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS sessions CASCADE")
    conn.execute("DROP TABLE IF EXISTS groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS users CASCADE")

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
    conn.close()

    yield None, get_connection

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


class TestPasswordHashing:
    """Tests fuer hash_password und verify_password."""

    def test_hash_and_verify(self):
        hashed = hash_password("geheim123")
        assert verify_password("geheim123", hashed)

    def test_wrong_password(self):
        hashed = hash_password("richtig")
        assert not verify_password("falsch", hashed)

    def test_hash_is_unique(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # bcrypt Salt macht jeden Hash einzigartig


class TestSessions:
    """Tests fuer create_session, validate_session, delete_session."""

    def test_create_session_returns_token(self, auth_db):
        _db_path, _get_db = auth_db
        token, _ = create_session(1)
        assert isinstance(token, str)
        assert len(token) == 64  # secrets.token_hex(32) = 64 hex chars

    def test_create_session_persists_in_db(self, auth_db):
        _db_path, get_db = auth_db
        token, _ = create_session(1)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        db = get_db()
        row = db.execute("SELECT * FROM sessions WHERE token = %s", (token_hash,)).fetchone()
        db.close()
        assert row is not None
        assert row["user_id"] == 1

    def test_validate_session_returns_user(self, auth_db):
        _db_path, _get_db = auth_db
        token, _ = create_session(1)
        user = validate_session(token)
        assert user is not None
        assert user["id"] == 1
        assert user["username"] == "admin"
        assert user["is_admin"]

    def test_validate_session_invalid_token(self, auth_db):
        user = validate_session("invalid_token_that_does_not_exist")
        assert user is None

    def test_delete_session(self, auth_db):
        _db_path, _get_db = auth_db
        token, _ = create_session(1)
        assert validate_session(token) is not None
        delete_session(token)
        assert validate_session(token) is None

    def test_non_admin_session(self, auth_db):
        _db_path, _get_db = auth_db
        token, _ = create_session(2)  # testuser (id=2)
        user = validate_session(token)
        assert user is not None
        assert user["username"] == "testuser"
        assert not user["is_admin"]


class TestTokenSigning:
    """Tests fuer sign_token und unsign_token."""

    def test_sign_and_unsign(self):
        original = "my_secret_token_123"
        signed = sign_token(original)
        assert signed != original
        recovered = unsign_token(signed)
        assert recovered == original

    def test_unsign_invalid_token(self):
        result = unsign_token("this.is.not.valid")
        assert result is None

    def test_unsign_tampered_token(self):
        signed = sign_token("original")
        tampered = signed[:-3] + "xxx"
        result = unsign_token(tampered)
        assert result is None


class TestLoginFlow:
    """Integrations-Test: Passwort-Prüfung → Session → Validierung → Logout."""

    def test_full_login_flow(self, auth_db):
        db_path, get_db = auth_db

        # 1. Passwort prüfen (wie api_auth.py login() es tut)
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE username = %s", ("admin",)).fetchone()
        db.close()
        assert row is not None
        assert verify_password("admin", row["password_hash"])

        # 2. Session erstellen
        token, _ = create_session(row["id"])
        assert token

        # 3. Session validieren
        user = validate_session(token)
        assert user["username"] == "admin"
        assert user["is_admin"]

        # 4. Token signieren/unsignen (Cookie-Roundtrip)
        signed = sign_token(token)
        recovered = unsign_token(signed)
        assert recovered == token

        # 5. Logout
        delete_session(token)
        assert validate_session(token) is None

    def test_wrong_password_rejected(self, auth_db):
        _db_path, get_db = auth_db
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE username = %s", ("admin",)).fetchone()
        db.close()
        assert not verify_password("wrong_password", row["password_hash"])
