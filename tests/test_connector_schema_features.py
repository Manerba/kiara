"""Tests fuer die neuen Schema-Features (Phase 11.2)."""

import sys

import pytest

from kiara.connectors import registry
from kiara.connectors.registry import all_connectors, ensure_loaded
from kiara.connectors.registry import get as get_connector

_CONNECTOR_MODULE_PREFIXES = (
    "kiara.connectors.local_connector",
    "kiara.connectors.smb_connector",
    "kiara.connectors.webdav_connector",
    "kiara.connectors.imap_connector",
    "kiara.connectors.exchange_connector",
    "kiara.connectors.sql_ds_connector",
)


@pytest.fixture(autouse=True)
def _load():
    # Falls vorherige Tests (z.B. test_connector_entry_points) die Registry
    # geleert und Module aus sys.modules entfernt haben, muessen wir frisch laden.
    if not registry._registry:
        for key in list(sys.modules):
            if key.startswith(_CONNECTOR_MODULE_PREFIXES):
                del sys.modules[key]
        registry._loaded = False
        ensure_loaded()
    elif not registry._loaded:
        ensure_loaded()


class TestBrowseAttribute:
    def test_smb_sub_path_has_browse(self):
        cls = get_connector("smb")
        schema = cls.config_schema()
        assert "browse" in schema["sub_path"]
        browse = schema["sub_path"]["browse"]
        assert "endpoint" in browse
        assert "depends_on" in browse
        assert isinstance(browse["depends_on"], list)
        assert len(browse["depends_on"]) > 0

    def test_webdav_base_path_has_browse(self):
        cls = get_connector("webdav")
        schema = cls.config_schema()
        browse = schema["base_path"]["browse"]
        assert browse["endpoint"] == "/api/admin/datasources/browse-webdav"
        assert set(("url", "username", "password")).issubset(set(browse["depends_on"]))

    def test_browse_depends_on_fields_exist_in_schema(self):
        """Alle Felder in depends_on muessen im selben Schema existieren."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field_name, spec in schema.items():
                if "browse" in spec:
                    for dep in spec["browse"]["depends_on"]:
                        assert dep in schema, (
                            f"{name}.{field_name}.browse.depends_on "
                            f"referenziert '{dep}', aber das Feld existiert nicht im Schema"
                        )


class TestOptionsAttribute:
    def test_sql_driver_has_options(self):
        cls = get_connector("sql")
        schema = cls.config_schema()
        assert "options" in schema["driver"]
        options = schema["driver"]["options"]
        assert isinstance(options, list)
        assert len(options) >= 4
        values = [o["value"] for o in options]
        assert "auto" in values
        assert "mssql" in values
        assert "postgresql" in values
        assert "sqlite" not in values

    def test_options_have_value_and_label(self):
        """Alle Options-Eintraege haben value und label."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field_name, spec in schema.items():
                if "options" in spec:
                    for i, opt in enumerate(spec["options"]):
                        assert "value" in opt, f"{name}.{field_name}.options[{i}] hat kein 'value'"
                        assert "label" in opt, f"{name}.{field_name}.options[{i}] hat kein 'label'"

    def test_options_values_unique(self):
        """Option-Values muessen eindeutig sein."""
        for name, cls in all_connectors().items():
            schema = cls.config_schema()
            for field_name, spec in schema.items():
                if "options" in spec:
                    values = [o["value"] for o in spec["options"]]
                    assert len(values) == len(set(values)), (
                        f"{name}.{field_name}.options hat doppelte Values: {values}"
                    )


class TestMultilineAttribute:
    def test_sql_query_has_multiline(self):
        cls = get_connector("sql")
        schema = cls.config_schema()
        assert schema["query"].get("multiline") is True


class TestHintAttribute:
    def test_imap_folders_has_hint(self):
        cls = get_connector("imap")
        schema = cls.config_schema()
        assert "hint" in schema["folders"]
        assert isinstance(schema["folders"]["hint"], str)

    def test_local_path_has_hint(self):
        cls = get_connector("local")
        schema = cls.config_schema()
        assert "hint" in schema["path"]


class TestConnectorTypesEndpoint:
    def test_all_connectors_have_default_config(self):
        """default_config() gibt ein dict zurueck fuer jeden Konnektor."""
        for name, cls in all_connectors().items():
            defaults = cls.default_config()
            assert isinstance(defaults, dict), f"{name}.default_config() ist kein dict"
            # Alle Schema-Felder muessen im Default enthalten sein
            for field in cls.config_schema():
                assert field in defaults, f"{name}.default_config() fehlt '{field}'"
