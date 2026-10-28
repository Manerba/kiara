"""Tests fuer ConnectorBase ABC-Enforcement (Phase 11.1).

Stellt sicher, dass unvollstaendige Implementierungen nicht
instanziiert werden koennen und minimale Implementierungen funktionieren.
"""

import pytest

from kiara.connectors.base import ConnectorBase
from kiara.connectors.browse import ConnectorBrowseError


class TestConnectorBaseABC:
    def test_cannot_instantiate_without_all_methods(self):
        """Unvollstaendige Implementierung: TypeError bei Instanziierung."""

        class IncompleteConnector(ConnectorBase):
            name = "incomplete"
            label = "Test"
            badge_class = "badge"
            # Keine abstractmethods implementiert

        with pytest.raises(TypeError):
            IncompleteConnector()

    def test_minimal_connector_implementation(self):
        """Minimale vollstaendige Implementierung: Instanziierung erfolgreich."""

        class MinimalConnector(ConnectorBase):
            name = "minimal"
            label = "Minimal"
            badge_class = "badge-secondary"

            @staticmethod
            def config_schema():
                return {"key": {"type": "str", "label": "Key"}}

            @staticmethod
            def source_display(config, row):
                return config.get("key", "")

            @staticmethod
            def effective_path(config, row):
                return "/tmp/minimal"

            @staticmethod
            def test_connection(config, row):
                return {"success": True}

            @staticmethod
            def prepare_and_scan(config, row, ds_config):
                return "/tmp/minimal"

        # Instanziierung muss funktionieren
        instance = MinimalConnector()
        assert instance is not None

        # Alle Methoden aufrufbar
        assert MinimalConnector.config_schema() == {"key": {"type": "str", "label": "Key"}}
        assert MinimalConnector.source_display({"key": "v"}, {}) == "v"
        assert MinimalConnector.effective_path({}, {}) == "/tmp/minimal"
        assert MinimalConnector.test_connection({}, {}) == {"success": True}
        assert MinimalConnector.prepare_and_scan({}, {}, {}) == "/tmp/minimal"

        # Default-Methoden von ConnectorBase
        assert MinimalConnector.secret_fields() == []
        assert MinimalConnector.default_config() == {"key": ""}
        assert MinimalConnector.supports_browse is False
        with pytest.raises(ConnectorBrowseError) as exc_info:
            MinimalConnector.list_directory({})
        assert exc_info.value.reason == "unsupported"

        descriptor = ConnectorBase.__dict__["list_directory"]
        assert isinstance(descriptor, staticmethod)

    def test_partial_implementation_missing_one_method(self):
        """Fehlende einzelne Methode: TypeError."""

        class AlmostComplete(ConnectorBase):
            name = "almost"
            label = "Almost"
            badge_class = "badge"

            @staticmethod
            def config_schema():
                return {}

            @staticmethod
            def source_display(config, row):
                return ""

            @staticmethod
            def effective_path(config, row):
                return ""

            @staticmethod
            def test_connection(config, row):
                return {}

            # prepare_and_scan fehlt!

        with pytest.raises(TypeError):
            AlmostComplete()

    def test_secret_fields_from_schema(self):
        """secret_fields() wird automatisch aus config_schema() abgeleitet."""

        class SecretConnector(ConnectorBase):
            name = "secret_test"
            label = "Secret Test"
            badge_class = "badge"

            @staticmethod
            def config_schema():
                return {
                    "username": {"type": "str", "label": "User"},
                    "password": {"type": "str", "label": "Pass", "secret": True},
                    "token": {"type": "str", "label": "Token", "secret": True},
                }

            @staticmethod
            def source_display(config, row):
                return ""

            @staticmethod
            def effective_path(config, row):
                return ""

            @staticmethod
            def test_connection(config, row):
                return {}

            @staticmethod
            def prepare_and_scan(config, row, ds_config):
                return ""

        secrets = SecretConnector.secret_fields()
        assert "password" in secrets
        assert "token" in secrets
        assert "username" not in secrets
        assert len(secrets) == 2

    def test_default_config_types(self):
        """default_config() leitet Defaults korrekt ab."""

        class TypedConnector(ConnectorBase):
            name = "typed"
            label = "Typed"
            badge_class = "badge"

            @staticmethod
            def config_schema():
                return {
                    "name": {"type": "str", "default": "test"},
                    "port": {"type": "int", "default": 8080},
                    "ssl": {"type": "bool", "default": True},
                    "no_default": {"type": "str"},  # kein default angegeben
                }

            @staticmethod
            def source_display(config, row):
                return ""

            @staticmethod
            def effective_path(config, row):
                return ""

            @staticmethod
            def test_connection(config, row):
                return {}

            @staticmethod
            def prepare_and_scan(config, row, ds_config):
                return ""

        defaults = TypedConnector.default_config()
        assert defaults["name"] == "test"
        assert defaults["port"] == 8080
        assert defaults["ssl"] is True
        assert defaults["no_default"] == ""  # str-Type Default = ""
