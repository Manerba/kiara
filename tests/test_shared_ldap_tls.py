"""Tests fuer kiara.shared.ldap TLS-Verhalten.

Kernverhalten:
- use_ssl=False nutzt den historischen STARTTLS-Pfad auf Port 389.
- Zertifikatspruefung bleibt per Default aus, damit interne Self-Signed-ADs funktionieren.
"""

import ssl
from unittest.mock import MagicMock, patch

from kiara.shared.ldap import _create_server


class TestPlainLdapUsesStartTls:
    """Regression: use_ssl=False bleibt beim historischen STARTTLS-Pfad."""

    def test_plain_ldap_bind_calls_start_tls(self):
        from kiara.shared import ldap as ldap_mod

        fake_conn = MagicMock()
        with patch.object(ldap_mod, "Connection", return_value=fake_conn):
            conn = ldap_mod._bind_connection(
                server=MagicMock(),
                bind_dn="CN=svc,DC=example,DC=com",
                bind_password="secret",
                use_ssl=False,
            )

        assert conn is fake_conn
        fake_conn.open.assert_called_once()
        fake_conn.start_tls.assert_called_once()
        fake_conn.bind.assert_called_once()


class TestCreateServerCertValidation:
    """Zertifikatspruefung ist opt-in, nicht Default."""

    def test_default_uses_cert_none(self):
        srv = _create_server("ldap.example.com", 636, use_ssl=True)
        assert srv.tls.validate == ssl.CERT_NONE

    def test_explicit_true_cert_required(self):
        srv = _create_server("ldap.example.com", 636, use_ssl=True, verify_cert=True)
        assert srv.tls.validate == ssl.CERT_REQUIRED

    def test_explicit_false_cert_none(self):
        """Self-Signed-Setups duerfen explizit auf CERT_NONE schalten."""
        srv = _create_server("ldap.example.com", 636, use_ssl=True, verify_cert=False)
        assert srv.tls.validate == ssl.CERT_NONE


class TestLdapConnectionCertDefault:
    """Der Connection-Pfad nutzt den Default ohne Zertifikatszwang."""

    def _base_config(self, **overrides):
        cfg = {
            "server": "ldap.example.com",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "CN=svc,DC=example,DC=com",
            "bind_password": "secret",
        }
        cfg.update(overrides)
        return cfg

    def test_connection_does_not_pass_verify_cert_kwarg(self):
        from kiara.shared import ldap as ldap_mod

        with (
            patch.object(ldap_mod, "_create_server") as mock_create,
            patch.object(ldap_mod, "_bind_connection") as mock_bind,
        ):
            mock_bind.return_value = MagicMock()
            with ldap_mod._ldap_connection(self._base_config()):
                pass
            mock_create.assert_called_once_with("ldap.example.com", 636, 1)


class TestAuthUserCertDefault:
    """Auch der User-Bind-Pfad nutzt den Default ohne Zertifikatszwang."""

    def _base_config(self, **overrides):
        cfg = {
            "server": "ldap.example.com",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "CN=svc,DC=example,DC=com",
            "bind_password": "secret",
            "search_base": "DC=example,DC=com",
        }
        cfg.update(overrides)
        return cfg

    def test_auth_user_user_bind_does_not_pass_verify_cert_kwarg(self):
        from kiara.shared import ldap as ldap_mod

        # _ldap_connection-Pfad: Search nach user_dn liefert einen Treffer
        fake_entry = MagicMock()
        fake_entry.distinguishedName = "CN=alice,DC=example,DC=com"
        fake_conn = MagicMock()
        fake_conn.entries = [fake_entry]

        # Wir wollen nur den ZWEITEN _create_server-Call (User-Bind) inspizieren —
        # er nutzt direkt _create_server, nicht den Context-Manager.
        from contextlib import contextmanager

        @contextmanager
        def fake_ldap_conn(*a, **kw):
            yield fake_conn

        with (
            patch.object(ldap_mod, "_ldap_connection", fake_ldap_conn),
            patch.object(ldap_mod, "_create_server") as mock_create,
            patch.object(ldap_mod, "_bind_connection") as mock_bind,
        ):
            mock_bind.return_value = MagicMock()
            ok, _msg = ldap_mod.authenticate_user(self._base_config(), "alice", "pw")
            assert ok is True
            mock_create.assert_called_once_with("ldap.example.com", 636, 1)
