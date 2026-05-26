"""Tests fuer Sprint 06: Group-CRUD Admin-Funktionen.

Testet groups.list, .members, .delete, .import_ldap, .sync_members,
.ldap_preview — 9 Unit-Tests.
"""

import asyncio
from unittest.mock import patch

import pytest

# ldap3 muss importierbar sein, damit @patch("kiara.web.ldap_utils.get_groups")
# das Modul auflösen kann (kiara.shared.ldap importiert ldap3 auf Top-Level).
try:
    from ldap3 import Server  # noqa: F401

    _has_ldap3 = True
except (ImportError, AttributeError):
    _has_ldap3 = False

_skip_no_ldap3 = pytest.mark.skipif(not _has_ldap3, reason="ldap3 nicht (korrekt) installiert")


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# groups.list
# ---------------------------------------------------------------------------


class TestGroupsList:
    @patch("kiara.repositories.group.GroupRepository.list_with_counts")
    def test_list_groups(self, mock_list):
        """Test 14: list_with_counts() Aufruf."""
        mock_list.return_value = [
            {"id": 1, "cn": "IT", "dn": "cn=IT,dc=test", "member_count": 5, "datasource_count": 2},
        ]
        from kiara.admin.functions.groups import groups_list

        result = _run(groups_list(user=MOCK_USER))

        assert result.success is True
        assert len(result.data["data"]) == 1
        mock_list.assert_called_once()


# ---------------------------------------------------------------------------
# groups.members
# ---------------------------------------------------------------------------


class TestGroupsMembers:
    @patch("kiara.repositories.group.GroupRepository.get_members")
    @patch("kiara.repositories.group.GroupRepository.exists")
    def test_group_members(self, mock_exists, mock_members):
        """Test 15: exists + get_members()."""
        mock_exists.return_value = True
        mock_members.return_value = [{"id": 1, "username": "user1"}]

        from kiara.admin.functions.groups import groups_members

        result = _run(groups_members(user=MOCK_USER, group_id=3))

        assert result.success is True
        assert len(result.data["members"]) == 1

    @patch("kiara.repositories.group.GroupRepository.exists")
    def test_group_members_nonexistent(self, mock_exists):
        """Test 16: nicht vorhanden → fail."""
        mock_exists.return_value = False

        from kiara.admin.functions.groups import groups_members

        result = _run(groups_members(user=MOCK_USER, group_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# groups.delete
# ---------------------------------------------------------------------------


class TestGroupsDelete:
    @patch("kiara.repositories.group.GroupRepository.delete_cascade")
    @patch("kiara.repositories.group.GroupRepository.exists")
    def test_delete_group(self, mock_exists, mock_delete):
        """Test 17: delete_cascade() aufgerufen."""
        mock_exists.return_value = True

        from kiara.admin.functions.groups import groups_delete

        result = _run(groups_delete(user=MOCK_USER, group_id=5))

        assert result.success is True
        mock_delete.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# groups.import_ldap
# ---------------------------------------------------------------------------


@_skip_no_ldap3
class TestGroupsImportLdap:
    @patch("kiara.repositories.group.GroupRepository.upsert_ldap")
    @patch("kiara.repositories.group.GroupRepository.get_by_sid")
    @patch("kiara.web.ldap_utils.get_groups")
    @patch("kiara.web.config_utils.build_ldap_config")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_import_ldap_groups(
        self, mock_config_get, mock_build, mock_get_groups, mock_get_by_sid, mock_upsert
    ):
        """Test 18: upsert_ldap fuer jede Gruppe."""
        mock_config_get.return_value = {"data": "..."}
        mock_build.return_value = {"server": "ldap://test", "search_base": "dc=test"}
        mock_get_groups.return_value = [
            {"sid": "S-1-5-21-1", "cn": "IT", "dn": "cn=IT,dc=test", "member_count": 5},
            {"sid": "S-1-5-21-2", "cn": "HR", "dn": "cn=HR,dc=test", "member_count": 3},
        ]
        mock_get_by_sid.return_value = None  # Beide neu

        from kiara.admin.functions.groups import groups_import_ldap

        result = _run(groups_import_ldap(user=MOCK_USER))

        assert result.success is True
        assert result.data["imported"] == 2
        assert mock_upsert.call_count == 2
        mock_upsert.assert_any_call(
            sid="S-1-5-21-1",
            cn="IT",
            dn="cn=IT,dc=test",
            member_count=5,
        )

    @patch("kiara.repositories.group.GroupRepository.upsert_ldap")
    @patch("kiara.repositories.group.GroupRepository.get_by_sid")
    @patch("kiara.web.ldap_utils.get_groups")
    @patch("kiara.web.config_utils.build_ldap_config")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_import_ldap_filter_dns(
        self, mock_config_get, mock_build, mock_get_groups, mock_get_by_sid, mock_upsert
    ):
        """Test 19: nur spezifizierte DNs importiert."""
        mock_config_get.return_value = {"data": "..."}
        mock_build.return_value = {"server": "ldap://test", "search_base": "dc=test"}
        mock_get_groups.return_value = [
            {"sid": "S-1-5-21-1", "cn": "IT", "dn": "cn=IT,dc=test", "member_count": 5},
            {"sid": "S-1-5-21-2", "cn": "HR", "dn": "cn=HR,dc=test", "member_count": 3},
        ]
        mock_get_by_sid.return_value = None

        from kiara.admin.functions.groups import groups_import_ldap

        result = _run(
            groups_import_ldap(
                user=MOCK_USER,
                group_dns=["cn=IT,dc=test"],
            )
        )

        assert result.success is True
        assert mock_upsert.call_count == 1  # Nur IT

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_import_ldap_no_config(self, mock_config_get):
        """Test 20: LDAP nicht konfiguriert → fail."""
        mock_config_get.return_value = None

        from kiara.admin.functions.groups import groups_import_ldap

        result = _run(groups_import_ldap(user=MOCK_USER))

        assert result.success is False
        assert "LDAP nicht konfiguriert" in result.message

    @patch("kiara.repositories.group.GroupRepository.upsert_ldap")
    @patch("kiara.web.ldap_utils.get_groups")
    @patch("kiara.web.config_utils.build_ldap_config")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_import_ldap_missing_sid_fails_closed(
        self, mock_config_get, mock_build, mock_get_groups, mock_upsert
    ):
        """LDAP-Gruppen ohne objectSid werden nicht importiert."""
        mock_config_get.return_value = {"data": "..."}
        mock_build.return_value = {"server": "ldap://test", "search_base": "dc=test"}
        mock_get_groups.return_value = [
            {"cn": "IT", "dn": "cn=IT,dc=test", "member_count": 5},
        ]

        from kiara.admin.functions.groups import groups_import_ldap

        result = _run(groups_import_ldap(user=MOCK_USER))

        assert result.success is False
        assert "objectSid" in result.message
        mock_upsert.assert_not_called()


# ---------------------------------------------------------------------------
# groups.sync_members
# ---------------------------------------------------------------------------


@_skip_no_ldap3
class TestGroupsSyncMembers:
    @patch("kiara.web.ldap_sync.sync_user_groups")
    @patch("kiara.web.config_utils.build_ldap_config")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_sync_members(self, mock_config_get, mock_build, mock_sync):
        """Test 21: sync_user_groups aufgerufen."""
        mock_config_get.return_value = {"data": "..."}
        mock_build.return_value = {"server": "ldap://test"}
        mock_sync.return_value = {"synced": 5, "errors": 0}

        from kiara.admin.functions.groups import groups_sync_members

        result = _run(groups_sync_members(user=MOCK_USER))

        assert result.success is True
        mock_sync.assert_called_once()


# ---------------------------------------------------------------------------
# groups.ldap_preview
# ---------------------------------------------------------------------------


@_skip_no_ldap3
class TestGroupsLdapPreview:
    @patch("kiara.repositories.group.GroupRepository.get_all_sids")
    @patch("kiara.web.ldap_utils.get_groups")
    @patch("kiara.web.config_utils.build_ldap_config")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_preview_marks_imported(
        self, mock_config_get, mock_build, mock_get_groups, mock_get_sids
    ):
        """Test 22: imported-Flag gesetzt."""
        mock_config_get.return_value = {"data": "..."}
        mock_build.return_value = {"server": "ldap://test", "search_base": "dc=test"}
        mock_get_groups.return_value = [
            {"sid": "S-1-5-21-1", "cn": "IT", "dn": "cn=IT,dc=test", "member_count": 5},
            {"sid": "S-1-5-21-2", "cn": "HR", "dn": "cn=HR,dc=test", "member_count": 3},
        ]
        mock_get_sids.return_value = ["S-1-5-21-1"]

        from kiara.admin.functions.groups import groups_ldap_preview

        result = _run(groups_ldap_preview(user=MOCK_USER))

        assert result.success is True
        groups = result.data["groups"]
        assert groups[0]["imported"] is True
        assert groups[1]["imported"] is False


# ---------------------------------------------------------------------------
# Ergaenzende Edge-Case Tests (Coverage-Erweiterung)
# ---------------------------------------------------------------------------


@_skip_no_ldap3
class TestGroupsEdgeCases:
    @patch("kiara.web.ldap_utils.get_groups")
    @patch("kiara.web.config_utils.build_ldap_config")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_import_ldap_get_groups_returns_none(
        self, mock_config_get, mock_build, mock_get_groups
    ):
        """get_groups() gibt None zurueck → graceful handling."""
        mock_config_get.return_value = {"data": "..."}
        mock_build.return_value = {"server": "ldap://test", "search_base": "dc=test"}
        mock_get_groups.return_value = None

        from kiara.admin.functions.groups import groups_import_ldap

        result = _run(groups_import_ldap(user=MOCK_USER))

        assert result.success is True
        assert result.data["imported"] == 0

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_ldap_preview_no_config(self, mock_config_get):
        """LDAP nicht konfiguriert bei Preview → fail."""
        mock_config_get.return_value = None

        from kiara.admin.functions.groups import groups_ldap_preview

        result = _run(groups_ldap_preview(user=MOCK_USER))

        assert result.success is False
        assert "LDAP nicht konfiguriert" in result.message

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_sync_members_no_config(self, mock_config_get):
        """LDAP nicht konfiguriert bei Sync → fail."""
        mock_config_get.return_value = None

        from kiara.admin.functions.groups import groups_sync_members

        result = _run(groups_sync_members(user=MOCK_USER))

        assert result.success is False
        assert "LDAP nicht konfiguriert" in result.message

    @patch("kiara.repositories.group.GroupRepository.exists")
    def test_delete_group_not_found(self, mock_exists):
        """Gruppe nicht gefunden → fail."""
        mock_exists.return_value = False

        from kiara.admin.functions.groups import groups_delete

        result = _run(groups_delete(user=MOCK_USER, group_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message
