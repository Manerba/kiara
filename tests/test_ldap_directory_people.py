"""Anonymisierung: LDAP-Stammdaten aus dem ganzen Verzeichnis statt nur der Benutzergruppe."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kiara.anonymization import stammdaten
from kiara.shared import ldap


class _FakeConn:
    def __init__(self, entries):
        self.calls = []
        self.extend = SimpleNamespace(standard=SimpleNamespace(paged_search=self._paged))
        self._entries = entries

    def _paged(self, base, search_filter, **kwargs):
        self.calls.append((base, search_filter, kwargs))
        yield from self._entries


def _patch_connection(conn):
    @contextmanager
    def fake(config, receive_timeout=None):
        yield conn

    return patch.object(ldap, "_ldap_connection", fake)


def test_get_directory_people_searches_domain_root_for_users_and_contacts():
    conn = _FakeConn(
        [
            {
                "type": "searchResEntry",
                "dn": "CN=Anna Schmidt,OU=Alt,DC=example,DC=local",
                "attributes": {
                    "distinguishedName": "CN=Anna Schmidt,OU=Alt,DC=example,DC=local",
                    "sAMAccountName": "as",
                    "sn": "Schmidt",
                    "givenName": "Anna",
                    "mail": ["anna@example.de"],
                    "displayName": "Schmidt, Anna",
                },
            },
            {"type": "searchResRef", "uri": ["ldap://DomainDnsZones.example.local/"]},
            {"type": "searchResEntry", "dn": "CN=Kontakt", "attributes": {"sn": "Meier"}},
        ]
    )
    config = {"search_base": "OU=Benutzer,DC=example,DC=local", "group_dn": "CN=IT,DC=x"}

    with _patch_connection(conn):
        people = ldap.get_directory_people(config)

    base, search_filter, kwargs = conn.calls[0]
    assert base == "DC=example,DC=local"
    assert "objectClass=contact" in search_filter and "objectCategory=person" in search_filter
    assert kwargs["paged_size"] > 0
    assert people[0]["sn"] == "Schmidt" and people[0]["mail"] == "anna@example.de"
    assert people[1] == {
        "dn": "CN=Kontakt",
        "sAMAccountName": "",
        "sn": "Meier",
        "givenName": "",
        "mail": "",
        "displayName": "",
    }


def test_get_directory_people_falls_back_to_group_dn_and_requires_base():
    conn = _FakeConn([])
    with _patch_connection(conn):
        ldap.get_directory_people({"search_base": "", "group_dn": "CN=IT,OU=G,DC=example,DC=local"})
        assert conn.calls[0][0] == "DC=example,DC=local"
        with pytest.raises(ValueError):
            ldap.get_directory_people({"search_base": "", "group_dn": ""})


def test_load_ldap_entries_uses_whole_directory_without_group():
    people = [
        {"dn": "CN=a", "sAMAccountName": "a", "sn": "Schmidt", "givenName": "Anna", "mail": ""},
        {"dn": "CN=b", "sAMAccountName": "b", "sn": "", "givenName": "", "mail": "info@x.de"},
    ]
    with (
        patch("kiara.repositories.config.ConfigRepository.get", return_value={"x": 1}),
        patch("kiara.web.config_utils.build_ldap_config", return_value={"group_dn": None}),
        patch("kiara.shared.ldap.get_directory_people", return_value=people) as directory,
        patch("kiara.shared.ldap.get_group_members") as group_members,
    ):
        entries = stammdaten._load_ldap_entries({})

    directory.assert_called_once()
    group_members.assert_not_called()
    values = {e["value"] for e in entries}
    assert {"Anna Schmidt", "Schmidt", "Anna", "Schmidt, Anna", "info@x.de"} <= values


def test_load_ldap_entries_fails_closed_without_config():
    with patch("kiara.repositories.config.ConfigRepository.get", return_value=None):
        with pytest.raises(RuntimeError):
            stammdaten._load_ldap_entries({})
