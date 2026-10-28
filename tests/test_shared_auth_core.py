"""Tests fuer kiara_auth — Shared Auth Library.

Prueft Core-Logik, Session-Service mit Fakes, FastAPI-Adapter,
Token-Primitives, Re-Export-Kompatibilitaet und Patch-Stabilitaet.
"""

import hashlib
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 1. Core: Passwort-Policy
# ---------------------------------------------------------------------------
from kiara_auth.core import (
    _PW_MIN_LENGTH,
    AuthenticatedPrincipal,
    has_role,
    has_scope,
    hash_password,
    hash_token,
    validate_password,
    verify_password,
)


class TestPasswordPolicy:
    """Passwort-Policy: min 8 Zeichen, 3 von 4 Kategorien."""

    def test_valid_password(self):
        assert validate_password("Secure1!x") is None

    def test_too_short(self):
        err = validate_password("Sh1!")
        assert err is not None
        assert str(_PW_MIN_LENGTH) in err

    def test_empty(self):
        err = validate_password("")
        assert err is not None
        assert str(_PW_MIN_LENGTH) in err

    def test_missing_categories(self):
        err = validate_password("abcdefgh")  # nur Kleinbuchstaben
        assert err is not None
        assert "3 der 4 Kategorien" in err

    def test_two_categories_fail(self):
        err = validate_password("abcdef12")  # Klein + Zahl = 2
        assert err is not None

    def test_three_categories_ok(self):
        assert validate_password("Abcdef12") is None  # Gross + Klein + Zahl = 3

    def test_four_categories_ok(self):
        assert validate_password("Abc1ef!x") is None

    def test_exact_error_text_length(self):
        """Exakte Fehlertexte (werden von API/Admin direkt weitergereicht)."""
        err = validate_password("short")
        assert err == f"Passwort muss mindestens {_PW_MIN_LENGTH} Zeichen haben"

    def test_exact_error_text_categories(self):
        err = validate_password("abcdefgh")
        assert err == (
            "Passwort muss mindestens 3 der 4 Kategorien enthalten: "
            "Grossbuchstabe, Kleinbuchstabe, Zahl, Sonderzeichen"
        )


# ---------------------------------------------------------------------------
# 2. Core: Passwort-Hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_returns_string(self):
        h = hash_password("test1234")
        assert isinstance(h, str)
        assert h.startswith("$2")  # bcrypt prefix

    def test_different_salts(self):
        h1 = hash_password("SamePass1!")
        h2 = hash_password("SamePass1!")
        assert h1 != h2  # unterschiedliche Salts

    def test_verify_correct(self):
        h = hash_password("Correct1!")
        assert verify_password("Correct1!", h) is True

    def test_verify_wrong(self):
        h = hash_password("Correct1!")
        assert verify_password("Wrong1!", h) is False

    def test_verify_invalid_hash(self):
        assert verify_password("anything", "not-a-hash") is False

    def test_verify_type_error(self):
        assert verify_password("anything", "") is False


# ---------------------------------------------------------------------------
# 3. Core: Token-Hashing
# ---------------------------------------------------------------------------


class TestTokenHashing:
    def test_sha256_hex(self):
        result = hash_token("test-token")
        expected = hashlib.sha256("test-token".encode()).hexdigest()
        assert result == expected

    def test_deterministic(self):
        assert hash_token("abc") == hash_token("abc")

    def test_different_inputs(self):
        assert hash_token("token-a") != hash_token("token-b")


# ---------------------------------------------------------------------------
# 4. Core: AuthenticatedPrincipal
# ---------------------------------------------------------------------------


class TestAuthenticatedPrincipal:
    def test_basic_creation(self):
        p = AuthenticatedPrincipal(id=1, username="admin")
        assert p.id == 1
        assert p.username == "admin"
        assert p.roles == []
        assert p.scopes == []
        assert p.is_active is True

    def test_with_roles_and_scopes(self):
        p = AuthenticatedPrincipal(
            id=2, username="user", roles=["editor"], scopes=["read", "write"]
        )
        assert has_role(p, "editor") is True
        assert has_role(p, "admin") is False
        assert has_scope(p, "read") is True
        assert has_scope(p, "delete") is False

    def test_metadata(self):
        p = AuthenticatedPrincipal(id=1, username="u", metadata={"tenant": "acme"})
        assert p.metadata["tenant"] == "acme"


# ---------------------------------------------------------------------------
# 5. Session-Service mit Fakes
# ---------------------------------------------------------------------------

from kiara_auth.sessions import SessionService
from kiara_auth.testing import FakeSessionWriter, FakeUserSessionResolver


class TestSessionService:
    def _make_service(self):
        writer = FakeSessionWriter()
        users = {
            1: {"id": 1, "username": "alice", "is_active": True},
            2: {"id": 2, "username": "bob", "is_active": False},
        }
        resolver = FakeUserSessionResolver(users, writer.sessions)
        service = SessionService(
            session_writer=writer,
            session_max_age=3600,
            user_session_resolver=resolver,
        )
        return service, writer

    def test_create_returns_plaintext_token(self):
        svc, writer = self._make_service()
        token, _ = svc.create(user_id=1)
        assert len(token) == 64  # 32 bytes hex

    def test_create_stores_hash(self):
        svc, writer = self._make_service()
        token, _ = svc.create(user_id=1)
        token_hash = hash_token(token)
        assert token_hash in writer.sessions
        assert token not in writer.sessions  # Klartext nicht gespeichert

    def test_validate_returns_user(self):
        svc, writer = self._make_service()
        token, _ = svc.create(user_id=1)
        result = svc.validate(token)
        assert result is not None
        assert result["username"] == "alice"

    def test_validate_inactive_user(self):
        svc, writer = self._make_service()
        token, _ = svc.create(user_id=2)
        result = svc.validate(token)
        assert result is None  # Bob ist inaktiv

    def test_validate_unknown_token(self):
        svc, _ = self._make_service()
        result = svc.validate("unknown-token-hex-string")
        assert result is None

    def test_delete_removes_session(self):
        svc, writer = self._make_service()
        token, _ = svc.create(user_id=1)
        assert len(writer.sessions) == 1
        svc.delete(token)
        assert len(writer.sessions) == 0

    def test_delete_user_sessions_except_current(self):
        svc, writer = self._make_service()
        t1, _ = svc.create(user_id=1)
        t2, _ = svc.create(user_id=1)
        t3, _ = svc.create(user_id=1)
        assert len(writer.sessions) == 3
        svc.delete_user_sessions(1, except_token=t2)
        assert len(writer.sessions) == 1
        # Die ausgenommene Session existiert noch
        assert hash_token(t2) in writer.sessions

    def test_validate_without_resolver_raises(self):
        writer = FakeSessionWriter()
        svc = SessionService(session_writer=writer, session_max_age=3600)
        svc.create(user_id=1)
        with pytest.raises(NotImplementedError):
            svc.validate("any-token")

    def test_conn_passthrough(self):
        svc, writer = self._make_service()
        # conn wird an create/delete durchgereicht (FakeWriter ignoriert es)
        token, _ = svc.create(user_id=1, conn="fake-conn")
        svc.delete(token, conn="fake-conn")
        assert len(writer.sessions) == 0


# ---------------------------------------------------------------------------
# 6. Token-Primitives
# ---------------------------------------------------------------------------

from kiara_auth.tokens import generate_api_key, make_key_prefix


class TestTokenPrimitives:
    def test_generate_api_key_format(self):
        full_key, key_hash = generate_api_key()
        assert full_key.startswith("orag_")
        assert len(full_key) == 5 + 48  # "orag_" + 48 hex chars

    def test_generate_api_key_hash(self):
        full_key, key_hash = generate_api_key()
        expected = hashlib.sha256(full_key.encode()).hexdigest()
        assert key_hash == expected

    def test_generate_api_key_unique(self):
        k1, _ = generate_api_key()
        k2, _ = generate_api_key()
        assert k1 != k2

    def test_make_key_prefix_20(self):
        """api_mcp.py Prefix-Format: 20 Zeichen + ..."""
        key = "orag_" + "a" * 48
        assert make_key_prefix(key, 20) == "orag_" + "a" * 15 + "..."

    def test_make_key_prefix_12(self):
        """admin/functions/mcp.py Prefix-Format: 12 Zeichen + ..."""
        key = "orag_" + "b" * 48
        assert make_key_prefix(key, 12) == "orag_" + "b" * 7 + "..."


# ---------------------------------------------------------------------------
# 7. FastAPI-Adapter (Cookie)
# ---------------------------------------------------------------------------

from kiara_auth.fastapi import CookieAuthAdapter


class TestCookieAuthAdapter:
    def _make_adapter(self):
        return CookieAuthAdapter(
            secret_key_provider=lambda: "test-secret-key-32bytes!abcdefgh",
            cookie_name="test_session",
            session_max_age=3600,
        )

    def test_sign_unsign_roundtrip(self):
        adapter = self._make_adapter()
        token = "my-session-token-abc123"
        signed = adapter.sign_token(token)
        assert signed != token
        assert adapter.unsign_token(signed) == token

    def test_unsign_invalid(self):
        adapter = self._make_adapter()
        assert adapter.unsign_token("tampered.garbage") is None

    def test_cookie_name_dynamic(self):
        adapter = self._make_adapter()
        assert adapter.cookie_name == "test_session"
        adapter.cookie_name = "admin_session"
        assert adapter.cookie_name == "admin_session"

    def test_different_secrets_incompatible(self):
        a1 = CookieAuthAdapter(secret_key_provider=lambda: "secret-A-32bytes!!!abcdefghijklm")
        a2 = CookieAuthAdapter(secret_key_provider=lambda: "secret-B-32bytes!!!abcdefghijklm")
        signed = a1.sign_token("token")
        assert a2.unsign_token(signed) is None


# ---------------------------------------------------------------------------
# 8. Testing-Helpers
# ---------------------------------------------------------------------------

from kiara_auth.testing import make_test_principal


class TestTestingHelpers:
    def test_make_test_principal_defaults(self):
        p = make_test_principal()
        assert p.username == "testuser"
        assert p.id == 1

    def test_make_test_principal_overrides(self):
        p = make_test_principal(username="alice", roles=["admin"])
        assert p.username == "alice"
        assert "admin" in p.roles


# ---------------------------------------------------------------------------
# 9. Re-Export-Kompatibilitaet
# ---------------------------------------------------------------------------


class TestReExportCompat:
    """Sichert, dass kiara.web.auth alle bisherigen Namen bereitstellt."""

    def test_pure_reexports_are_same_objects(self):
        import kiara.web.auth as auth
        import kiara_auth.core as core

        assert auth.hash_password is core.hash_password
        assert auth.verify_password is core.verify_password
        assert auth.validate_password is core.validate_password
        assert auth.hash_token is core.hash_token

    def test_all_names_importable(self):
        from kiara.web.auth import (
            create_session,
            get_admin_user,
            hash_password,
        )

        assert callable(hash_password)
        assert callable(create_session)
        assert callable(get_admin_user)

    def test_patch_hash_password_works(self):
        """@patch('kiara.web.auth.hash_password') muss weiterhin greifen."""
        with patch("kiara.web.auth.hash_password", return_value="mocked-hash"):
            from kiara.web import auth

            assert auth.hash_password("any") == "mocked-hash"

    def test_patch_validate_password_works(self):
        with patch("kiara.web.auth.validate_password", return_value="mock-error"):
            from kiara.web import auth

            assert auth.validate_password("any") == "mock-error"

    def test_import_smoke(self):
        """Packaging-Smoke: beide Packages importierbar."""
        import kiara_auth

        assert hasattr(kiara_auth, "__version__")


# ---------------------------------------------------------------------------
# 10. Protocol-Kompatibilitaet
# ---------------------------------------------------------------------------

from kiara_auth.repositories import SessionWriter


class TestProtocolCompat:
    """Prueft dass bestehende KIara-Repositories die Protocols erfuellen."""

    def test_fake_session_writer_is_protocol(self):
        writer = FakeSessionWriter()
        assert isinstance(writer, SessionWriter)


# ---------------------------------------------------------------------------
# 11. Session-Ablauf
# ---------------------------------------------------------------------------


class TestSessionExpiry:
    def test_cleanup_expired_removes_old_sessions(self):
        writer = FakeSessionWriter()
        # Abgelaufene Session manuell einfuegen
        writer.sessions["expired-hash"] = {
            "id": 99,
            "user_id": 1,
            "token": "expired-hash",
            "expires_at": "2020-01-01 00:00:00",
        }
        # Aktive Session
        writer.sessions["active-hash"] = {
            "id": 100,
            "user_id": 1,
            "token": "active-hash",
            "expires_at": "2099-12-31 23:59:59",
        }
        removed = writer.cleanup_expired()
        assert removed == 1
        assert "expired-hash" not in writer.sessions
        assert "active-hash" in writer.sessions


# ---------------------------------------------------------------------------
# 12. Passwort-Policy-HTTP-Kompatibilitaet
# ---------------------------------------------------------------------------


class TestPasswordPolicyHTTPCompat:
    """Fehlermeldungen werden als HTTP-detail und AdminResult.fail weitergereicht."""

    def test_none_for_valid(self):
        assert validate_password("ValidPass1!") is None

    def test_error_for_short(self):
        err = validate_password("")
        assert err is not None
        assert isinstance(err, str)

    def test_error_for_weak(self):
        err = validate_password("abcdefgh")
        assert err is not None
        assert isinstance(err, str)
