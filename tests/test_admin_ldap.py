"""Tests fuer Sprint 08: LDAP Admin-Funktionen.

Testet ldap.config.read, .update, .delete, .groups, .select_group,
.deselect_group, .sync — 15 Unit-Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ldap3 ist Pflicht-Dependency fuer LDAP-Tests. Wenn nicht installiert,
# wird das gesamte Modul uebersprungen — keine Stubs, die andere Tests
# kontaminieren oder sinnlose Ergebnisse liefern.
# ---------------------------------------------------------------------------
try:
    from ldap3 import Server as _Server  # noqa: F401
except (ImportError, AttributeError):
    pytest.skip("ldap3 nicht (korrekt) installiert", allow_module_level=True)

from kiara.shared.constants import PASSWORD_MASK


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# ldap.config.read
# ---------------------------------------------------------------------------


class TestLdapConfigRead:
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_config_read_empty(self, mock_get):
        """Test 30: Keine Config → data.config = None."""
        mock_get.return_value = None

        from kiara.admin.functions.ldap import ldap_config_read

        result = _run(ldap_config_read(MOCK_USER))

        assert result.success is True
        assert result.data["config"] is None

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_config_read_masked(self, mock_get):
        """Test 31: bind_password = PASSWORD_MASK."""
        mock_get.return_value = {
            "server_url": "ldaps://ad.firma.de",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "cn=kiara,dc=firma,dc=de",
            "bind_password": "gAAAAABsecret...",
            "search_base": "ou=users,dc=firma,dc=de",
        }

        from kiara.admin.functions.ldap import ldap_config_read

        result = _run(ldap_config_read(MOCK_USER))

        assert result.success is True
        assert result.data["config"]["bind_password"] == PASSWORD_MASK
        assert result.data["config"]["server_url"] == "ldaps://ad.firma.de"


# ---------------------------------------------------------------------------
# ldap.config.update
# ---------------------------------------------------------------------------


class TestLdapConfigUpdate:
    @patch("kiara.shared.ldap.test_connection")
    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_config_update(self, mock_get, mock_set, mock_encrypt, mock_test):
        """Test 32: Speichern mit Confirmation + Encryption."""
        mock_get.return_value = None  # Kein bestehendes Config
        mock_encrypt.return_value = "gAAAAABencrypted..."
        mock_test.return_value = (True, None)

        from kiara.admin.functions.ldap import ldap_config_update

        result = _run(
            ldap_config_update(
                MOCK_USER,
                server_url="ldaps://ad.firma.de",
                bind_dn="cn=kiara,dc=firma,dc=de",
                bind_password="secret123",
                search_base="ou=users,dc=firma,dc=de",
                port=636,
                use_ssl=1,
            )
        )

        assert result.success is True
        assert "gespeichert" in result.message
        mock_encrypt.assert_called_once_with("secret123")
        mock_set.assert_called_once()
        call_args = mock_set.call_args[0]
        assert call_args[1]["bind_password"] == "gAAAAABencrypted..."

    @patch("kiara.shared.ldap.test_connection")
    @patch("kiara.shared.crypto.is_encrypted")
    @patch("kiara.shared.crypto.decrypt")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_config_update_mask(
        self, mock_get, mock_set, mock_decrypt, mock_is_enc, mock_test
    ):
        """Test 33: PASSWORD_MASK → bestehendes PW beibehalten."""
        mock_get.return_value = {
            "server_url": "ldaps://ad.firma.de",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "cn=kiara,dc=firma,dc=de",
            "bind_password": "gAAAAABexisting...",
            "search_base": "ou=users,dc=firma,dc=de",
        }
        mock_is_enc.return_value = True
        mock_decrypt.return_value = "original_password"
        mock_test.return_value = (True, None)

        from kiara.admin.functions.ldap import ldap_config_update

        result = _run(
            ldap_config_update(
                MOCK_USER,
                server_url="ldaps://ad.firma.de",
                bind_dn="cn=kiara,dc=firma,dc=de",
                bind_password=PASSWORD_MASK,
                search_base="ou=users,dc=firma,dc=de",
            )
        )

        assert result.success is True
        # Bestehendes Passwort beibehalten
        call_args = mock_set.call_args[0]
        assert call_args[1]["bind_password"] == "gAAAAABexisting..."


# ---------------------------------------------------------------------------
# ldap.config.delete
# ---------------------------------------------------------------------------


class TestLdapConfigDelete:
    @patch("kiara.shared.db.db_transaction")
    def test_ldap_config_delete(self, mock_tx):
        """Test 34: Config geloescht + User deaktiviert."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_tx.return_value = mock_ctx

        from kiara.admin.functions.ldap import ldap_config_delete

        result = _run(ldap_config_delete(MOCK_USER))

        assert result.success is True
        assert "gelöscht" in result.message
        assert mock_ctx.execute.call_count == 2  # DELETE + UPDATE


# ---------------------------------------------------------------------------
# ldap.groups
# ---------------------------------------------------------------------------


class TestLdapGroups:
    @patch("kiara.shared.ldap.get_groups")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_groups(self, mock_get, mock_is_enc, mock_groups):
        """Test 35: Gruppen-Abfrage (Mock LDAP)."""
        mock_get.return_value = {
            "server_url": "ldaps://ad.firma.de",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "cn=kiara,dc=firma,dc=de",
            "bind_password": "cleartext",
            "search_base": "ou=users,dc=firma,dc=de",
        }
        mock_groups.return_value = [
            {"dn": "CN=Group1,DC=firma,DC=de", "name": "Group1"},
            {"dn": "CN=Group2,DC=firma,DC=de", "name": "Group2"},
        ]

        from kiara.admin.functions.ldap import ldap_groups

        result = _run(ldap_groups(MOCK_USER))

        assert result.success is True
        assert len(result.data["groups"]) == 2


# ---------------------------------------------------------------------------
# ldap.select_group
# ---------------------------------------------------------------------------


class TestLdapSelectGroup:
    @patch("kiara.web.ldap_sync.sync_ldap_users")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_select_group(self, mock_get, mock_set, mock_sync):
        """Test 36: Gruppe auswaehlen mit Confirmation."""
        mock_get.return_value = {
            "server_url": "ldaps://ad.firma.de",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "cn=kiara,dc=firma,dc=de",
            "bind_password": "gAAAAABsecret...",
            "search_base": "ou=users,dc=firma,dc=de",
        }
        mock_sync.return_value = {"created": 5, "updated": 0, "deactivated": 0}

        from kiara.admin.functions.ldap import ldap_select_group

        result = _run(
            ldap_select_group(
                MOCK_USER, group_dn="CN=KIara-Users,DC=firma,DC=de", group_name="KIara-Users"
            )
        )

        assert result.success is True
        assert "synchronisiert" in result.message


# ---------------------------------------------------------------------------
# ldap.deselect_group
# ---------------------------------------------------------------------------


class TestLdapDeselectGroup:
    @patch("kiara.shared.db.db_transaction")
    def test_ldap_deselect_group(self, mock_tx):
        """Test 37: Gruppenauswahl aufheben."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_tx.return_value = mock_ctx

        from kiara.admin.functions.ldap import ldap_deselect_group

        result = _run(ldap_deselect_group(MOCK_USER))

        assert result.success is True
        assert "aufgehoben" in result.message


# ---------------------------------------------------------------------------
# ldap.sync
# ---------------------------------------------------------------------------


class TestLdapSync:
    @patch("kiara.web.ldap_sync.sync_ldap_users")
    def test_ldap_sync(self, mock_sync):
        """Test 38: Manueller Sync mit Confirmation."""
        mock_sync.return_value = {"created": 2, "updated": 1, "deactivated": 0}

        from kiara.admin.functions.ldap import ldap_sync

        result = _run(ldap_sync(MOCK_USER))

        assert result.success is True
        assert "abgeschlossen" in result.message
        assert result.data["sync"]["created"] == 2


class TestLdapSyncBackend:
    @patch("kiara.web.ldap_sync.db_transaction")
    @patch("kiara.web.ldap_sync.hash_password", return_value="hashed")
    @patch("kiara.web.ldap_sync.user_repo.create")
    @patch("kiara.web.ldap_sync.user_repo.get_by_username", return_value=None)
    @patch("kiara.web.ldap_sync.get_group_members")
    @patch("kiara.web.ldap_sync.build_ldap_config", return_value={"server_url": "ldap"})
    @patch("kiara.web.ldap_sync.config_repo.get")
    def test_sync_creates_ldap_users_with_boolean_flags(
        self,
        mock_get,
        mock_build,
        mock_members,
        mock_get_user,
        mock_create,
        mock_hash,
        mock_tx,
    ):
        """PostgreSQL-BOOLEAN-Felder werden als bool uebergeben, nicht als 0/1."""
        mock_get.return_value = {"group_dn": "CN=KIara,DC=firma,DC=de"}
        mock_members.return_value = [
            {
                "sAMAccountName": "andy",
                "givenName": "Andreas",
                "sn": "Admin",
                "mail": "andy@example.test",
                "dn": "CN=andy,DC=firma,DC=de",
            }
        ]

        db = MagicMock()
        db.execute.return_value.fetchall.return_value = []
        ctx = MagicMock()
        ctx.__enter__.return_value = db
        ctx.__exit__.return_value = False
        mock_tx.return_value = ctx

        from kiara.web.ldap_sync import sync_ldap_users

        result = sync_ldap_users()

        assert result["created"] == 1
        kwargs = mock_create.call_args.kwargs
        assert kwargs["is_active"] is True
        assert kwargs["is_admin"] is False

    @patch("kiara.web.ldap_sync.db_transaction")
    @patch("kiara.web.ldap_sync.user_repo.update")
    @patch("kiara.web.ldap_sync.user_repo.get_by_username")
    @patch("kiara.web.ldap_sync.get_group_members")
    @patch("kiara.web.ldap_sync.build_ldap_config", return_value={"server_url": "ldap"})
    @patch("kiara.web.ldap_sync.config_repo.get")
    def test_sync_updates_and_deactivates_with_boolean_flags(
        self,
        mock_get,
        mock_build,
        mock_members,
        mock_get_user,
        mock_update,
        mock_tx,
    ):
        mock_get.return_value = {"group_dn": "CN=KIara,DC=firma,DC=de"}
        mock_members.return_value = [
            {
                "sAMAccountName": "newldap",
                "givenName": "New",
                "sn": "User",
                "mail": "new@example.test",
                "dn": "CN=newldap,DC=firma,DC=de",
            }
        ]
        mock_get_user.return_value = {"id": 10, "username": "newldap", "auth_source": "ldap"}

        active_users = MagicMock()
        active_users.fetchall.return_value = [
            {"id": 10, "username": "newldap"},
            {"id": 22, "username": "oldldap"},
        ]

        db = MagicMock()
        db.execute.side_effect = lambda sql, params=None: (
            active_users if "SELECT id, username FROM users" in sql else MagicMock()
        )
        ctx = MagicMock()
        ctx.__enter__.return_value = db
        ctx.__exit__.return_value = False
        mock_tx.return_value = ctx

        from kiara.web.ldap_sync import sync_ldap_users

        result = sync_ldap_users()

        assert result["updated"] == 1
        assert result["deactivated"] == 1
        assert mock_update.call_args_list[0].kwargs["is_active"] is True
        assert mock_update.call_args_list[1].kwargs["is_active"] is False


# ---------------------------------------------------------------------------
# PASSWORD_MASK Edge Cases
# ---------------------------------------------------------------------------


class TestLdapPasswordMaskEdgeCases:
    @patch("kiara.shared.ldap.test_connection")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_empty_password_new_config_allowed(self, mock_get, mock_set, mock_test):
        """Leerer String bei neuer Config → leeres Passwort, kein Fehler."""
        mock_get.return_value = None
        mock_test.return_value = (True, None)

        from kiara.admin.functions.ldap import ldap_config_update

        result = _run(
            ldap_config_update(
                MOCK_USER,
                server_url="ldaps://test.de",
                bind_dn="cn=test",
                bind_password="",
                search_base="ou=users",
            )
        )

        assert result.success is True
        call_args = mock_set.call_args[0]
        assert call_args[1]["bind_password"] == ""

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mask_without_existing_config_fails(self, mock_get):
        """PASSWORD_MASK bei nicht-existierender Config → Fehler."""
        mock_get.return_value = None  # Kein ldap_config in DB

        from kiara.admin.functions.ldap import ldap_config_update

        result = _run(
            ldap_config_update(
                MOCK_USER,
                server_url="ldaps://test.de",
                bind_dn="cn=test",
                bind_password=PASSWORD_MASK,
                search_base="ou=users",
            )
        )

        assert result.success is False
        assert "passwort" in result.message.lower()

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mask_with_empty_existing_password_fails(self, mock_get):
        """PASSWORD_MASK bei Config OHNE gespeichertes PW → Fehler."""
        mock_get.return_value = {
            "server_url": "ldaps://old.de",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "cn=old",
            "bind_password": "",
            "search_base": "ou=users",
        }

        from kiara.admin.functions.ldap import ldap_config_update

        result = _run(
            ldap_config_update(
                MOCK_USER,
                server_url="ldaps://test.de",
                bind_dn="cn=test",
                bind_password=PASSWORD_MASK,
                search_base="ou=users",
            )
        )

        # bind_password ist "" → .get("bind_password") truthy-Check schlaegt fehl
        # → elif PASSWORD_MASK → fail
        assert result.success is False

    @patch("kiara.shared.ldap.test_connection")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_connection_test_fails(self, mock_get, mock_set, mock_test):
        """Verbindungstest schlaegt fehl → kein Speichern."""
        mock_get.return_value = None
        mock_test.return_value = (False, "Connection timed out")

        from kiara.admin.functions.ldap import ldap_config_update

        result = _run(
            ldap_config_update(
                MOCK_USER,
                server_url="ldaps://bad.de",
                bind_dn="cn=test",
                bind_password="secret",
                search_base="ou=users",
            )
        )

        assert result.success is False
        assert "fehlgeschlagen" in result.message
        mock_set.assert_not_called()


# ---------------------------------------------------------------------------
# LDAP Decrypt Failure
# ---------------------------------------------------------------------------


class TestLdapDecryptFailure:
    @patch("kiara.shared.crypto.decrypt")
    @patch("kiara.shared.crypto.is_encrypted")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_groups_decrypt_failure(self, mock_get, mock_is_enc, mock_decrypt):
        """Korruptes verschluesseltes Passwort → Fehler bei Gruppenabfrage."""
        mock_get.return_value = {
            "server_url": "ldaps://ad.firma.de",
            "port": 636,
            "use_ssl": 1,
            "bind_dn": "cn=kiara",
            "bind_password": "gAAAAABcorrupted_data",
            "search_base": "ou=users",
        }
        mock_is_enc.return_value = True
        mock_decrypt.side_effect = Exception("InvalidToken")

        from kiara.admin.functions.ldap import ldap_groups

        result = _run(ldap_groups(MOCK_USER))

        assert result.success is False


# ---------------------------------------------------------------------------
# DB Transaction Failure
# ---------------------------------------------------------------------------


class TestLdapTransactionFailure:
    @patch("kiara.shared.db.db_transaction")
    def test_ldap_config_delete_db_error(self, mock_tx):
        """DB-Exception waehrend Transaktion → unbehandelter Fehler."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.execute.side_effect = Exception("database is locked")
        mock_tx.return_value = mock_ctx

        from kiara.admin.functions.ldap import ldap_config_delete

        with pytest.raises(Exception, match="database is locked"):
            _run(ldap_config_delete(MOCK_USER))
