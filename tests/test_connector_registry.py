"""Tests fuer ConnectorRegistry (Phase 11.1)."""

import pytest

from kiara.connectors.base import ConnectorBase
from kiara.connectors.registry import (
    all_connectors,
    auto_mount_point,
    connector_names,
    ensure_loaded,
    get,
)


@pytest.fixture(autouse=True)
def _load_connectors():
    ensure_loaded()


class TestEnsureLoaded:
    """ensure_loaded() registriert alle Built-in-Konnektoren."""

    def test_all_five_connectors_registered(self):
        names = connector_names()
        for expected in ("local", "smb", "webdav", "imap", "sql"):
            assert expected in names, f"Konnektor '{expected}' nicht registriert"

    def test_get_returns_connector_class(self):
        cls = get("imap")
        assert cls is not None
        assert cls.name == "imap"
        assert issubclass(cls, ConnectorBase)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get("ftp")

    def test_all_connectors_returns_dict(self):
        conns = all_connectors()
        assert isinstance(conns, dict)
        assert len(conns) >= 5

    def test_all_connectors_returns_copy(self):
        """all_connectors() gibt eine Kopie zurueck, kein Alias."""
        a = all_connectors()
        b = all_connectors()
        assert a is not b

    def test_connector_names_list(self):
        names = connector_names()
        assert isinstance(names, list)
        assert len(names) >= 5


class TestAutoMountPoint:
    """auto_mount_point() erzeugt deterministische Pfade."""

    def test_simple_name(self):
        row = {"name": "Firma", "id": 3}
        assert auto_mount_point(row) == "/mnt/kiara-ds-3"

    def test_ignores_datasource_name(self):
        row = {"name": "Firma (Daten!)", "id": 7}
        renamed = {"name": "Umbenannt", "id": 7}
        assert auto_mount_point(row) == "/mnt/kiara-ds-7"
        assert auto_mount_point(row) == auto_mount_point(renamed)

    def test_deterministic(self):
        row = {"name": "Test DS", "id": 42}
        assert auto_mount_point(row) == auto_mount_point(row)


class TestConnectorBaseInterface:
    """Alle registrierten Konnektoren implementieren das ConnectorBase-Interface."""

    def test_all_have_required_attributes(self):
        for name, cls in all_connectors().items():
            assert hasattr(cls, "name"), f"{name} hat kein 'name'"
            assert hasattr(cls, "label"), f"{name} hat kein 'label'"
            assert hasattr(cls, "badge_class"), f"{name} hat kein 'badge_class'"

    def test_all_have_config_schema(self):
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            assert isinstance(schema, dict), f"{name}.config_schema() ist kein dict"

    def test_all_have_secret_fields(self):
        for name, cls in all_connectors().items():
            fields = cls.secret_fields()
            assert isinstance(fields, list), f"{name}.secret_fields() ist keine Liste"

    def test_all_have_default_config(self):
        for name, cls in all_connectors().items():
            defaults = cls.default_config()
            assert isinstance(defaults, dict), f"{name}.default_config() ist kein dict"
