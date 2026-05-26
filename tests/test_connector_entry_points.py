"""Tests fuer Entry-Point-basierte Konnektor-Discovery (Phase 11.2)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from kiara.connectors import registry
from kiara.connectors.base import ConnectorBase

# Module die beim Reset aus sys.modules entfernt werden muessen,
# damit @register beim Re-Import erneut feuert.
_CONNECTOR_MODULE_PREFIXES = (
    "kiara.connectors.local_connector",
    "kiara.connectors.smb_connector",
    "kiara.connectors.webdav_connector",
    "kiara.connectors.imap_connector",
    "kiara.connectors.exchange_connector",
    "kiara.connectors.sql_ds_connector",
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """Registry UND Module-Cache vor jedem Test zuruecksetzen.

    @register ist ein Class-Decorator der nur beim ersten Import feuert.
    Wenn ein Modul bereits in sys.modules gecacht ist, wird __import__()
    das Modul zurueckgeben OHNE den Code erneut auszufuehren.
    Deshalb muessen wir die Module aus sys.modules entfernen.

    Im Teardown wird der VOLLSTAENDIGE Zustand wiederhergestellt,
    damit nachfolgende Tests nicht von einem leeren Registry ausgehen.
    """
    # Zustand sichern
    saved_registry = dict(registry._registry)
    saved_loaded = registry._loaded
    saved_modules = {}

    # Registry leeren + Module aus sys.modules entfernen
    registry._registry.clear()
    registry._loaded = False
    for key in list(sys.modules):
        if key.startswith(_CONNECTOR_MODULE_PREFIXES):
            saved_modules[key] = sys.modules.pop(key)

    yield

    # Vollstaendigen Zustand wiederherstellen
    registry._registry.clear()
    registry._registry.update(saved_registry)
    registry._loaded = saved_loaded
    sys.modules.update(saved_modules)


class TestEntryPointDiscovery:
    def test_ensure_loaded_loads_builtins(self):
        """Built-in-Konnektoren werden immer geladen."""
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.ensure_loaded()

        names = registry.connector_names()
        assert len(names) >= 6
        for name in ("local", "smb", "webdav", "imap", "exchange_online", "sql"):
            assert name in names

    def test_entry_points_load_external_plugins(self):
        """Externe Plugins werden via Entry-Points zusaetzlich geladen."""
        # Dummy-Konnektor als "externes Plugin"
        mock_ep = MagicMock()
        mock_ep.name = "test_external"

        class FakeConnector(ConnectorBase):
            name = "test_external"
            label = "Test"
            badge_class = "default"

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
                return {"success": True}

            @staticmethod
            def prepare_and_scan(config, row, ds_config):
                return None

        mock_ep.load.side_effect = lambda: registry.register(FakeConnector)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        names = registry.connector_names()
        # Built-ins UND externes Plugin
        assert "local" in names
        assert "test_external" in names

    def test_entry_point_skips_already_registered(self):
        """Bereits registrierte Konnektoren werden nicht nochmal per EP geladen."""
        mock_ep = MagicMock()
        mock_ep.name = "local"  # Bereits als Built-in registriert

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        # EP.load() sollte NICHT aufgerufen worden sein
        mock_ep.load.assert_not_called()
        assert "local" in registry.connector_names()

    def test_partial_entry_point_failure(self):
        """Ein Entry-Point schlaegt fehl -> andere werden trotzdem geladen."""
        mock_fail = MagicMock()
        mock_fail.name = "broken_plugin"
        mock_fail.load.side_effect = ImportError("broken")

        with patch("importlib.metadata.entry_points", return_value=[mock_fail]):
            registry.ensure_loaded()

        # Built-ins wurden trotzdem geladen
        assert "local" in registry.connector_names()
        assert len(registry.connector_names()) >= 6

    def test_ensure_loaded_idempotent(self):
        """Zweimaliger Aufruf aendert nichts."""
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.ensure_loaded()
            first_names = set(registry.connector_names())
            registry.ensure_loaded()
            assert set(registry.connector_names()) == first_names

    def test_loaded_true_after_success(self):
        """V2-Fix K1: _loaded wird am Ende auf True gesetzt."""
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.ensure_loaded()
        assert registry._loaded is True
        assert len(registry.connector_names()) >= 6

    def test_entry_point_discovery_exception(self):
        """Entry-Point-Discovery schlaegt fehl -> Built-ins laden trotzdem."""
        with patch("importlib.metadata.entry_points", side_effect=Exception("broken")):
            registry.ensure_loaded()

        assert len(registry.connector_names()) >= 6
