"""Tests fuer Phase 11.2 — Entry-Point-Integration mit WebCrawler-Plugin.

Testet die End-to-End-Integration der Entry-Point-Discovery:
ensure_loaded() laedt Built-in-Konnektoren UND externe Plugins via Entry-Points.
"""

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
    """Registry UND Module-Cache vor jedem Test zuruecksetzen."""
    saved_registry = dict(registry._registry)
    saved_loaded = registry._loaded
    saved_modules = {}

    registry._registry.clear()
    registry._loaded = False
    for key in list(sys.modules):
        if key.startswith(_CONNECTOR_MODULE_PREFIXES):
            saved_modules[key] = sys.modules.pop(key)

    yield

    registry._registry.clear()
    registry._registry.update(saved_registry)
    registry._loaded = saved_loaded
    sys.modules.update(saved_modules)


def _make_fake_connector(name, label="Test", badge_class="default"):
    """Erzeugt eine minimale ConnectorBase-Unterklasse."""
    return type(
        f"Fake_{name}",
        (ConnectorBase,),
        {
            "name": name,
            "label": label,
            "badge_class": badge_class,
            "config_schema": staticmethod(lambda: {}),
            "source_display": staticmethod(lambda config, row: ""),
            "effective_path": staticmethod(lambda config, row: ""),
            "test_connection": staticmethod(lambda config, row: {"success": True}),
            "prepare_and_scan": staticmethod(lambda config, row, ds_config: None),
        },
    )


class TestEntryPointLoadsWebCrawler:
    """1. Entry-Point laedt WebCrawler-Plugin."""

    def test_webcrawler_registered_via_entry_point(self):
        """Wenn webcrawler installiert ist, wird er via EP registriert."""
        FakeCrawler = _make_fake_connector("webcrawler", "Web-Crawler", "primary")
        mock_ep = MagicMock()
        mock_ep.name = "webcrawler"
        mock_ep.load.side_effect = lambda: registry.register(FakeCrawler)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        assert registry.get("webcrawler") is FakeCrawler
        assert "webcrawler" in registry.connector_names()

    def test_webcrawler_has_correct_attributes(self):
        """WebCrawlerConnector hat name, label, badge_class."""
        FakeCrawler = _make_fake_connector("webcrawler", "Web-Crawler", "primary")
        mock_ep = MagicMock()
        mock_ep.name = "webcrawler"
        mock_ep.load.side_effect = lambda: registry.register(FakeCrawler)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        cls = registry.get("webcrawler")
        assert cls.name == "webcrawler"
        assert cls.label == "Web-Crawler"
        assert cls.badge_class == "primary"


class TestBuiltinAndEntryPointCoexistence:
    """2. Entry-Point und Builtin-Konnektoren koexistieren."""

    def test_at_least_6_connectors_after_ensure_loaded(self):
        """Nach ensure_loaded(): mindestens 7 Konnektoren (6 Built-in + webcrawler)."""
        FakeCrawler = _make_fake_connector("webcrawler", "Web-Crawler", "primary")
        mock_ep = MagicMock()
        mock_ep.name = "webcrawler"
        mock_ep.load.side_effect = lambda: registry.register(FakeCrawler)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        assert len(registry.connector_names()) >= 7

    def test_all_connectors_have_metadata(self):
        """Alle Konnektoren haben name, label, badge_class."""
        FakeCrawler = _make_fake_connector("webcrawler", "Web-Crawler", "primary")
        mock_ep = MagicMock()
        mock_ep.name = "webcrawler"
        mock_ep.load.side_effect = lambda: registry.register(FakeCrawler)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        for name, cls in registry.all_connectors().items():
            assert hasattr(cls, "name"), f"{name} hat kein 'name'"
            assert hasattr(cls, "label"), f"{name} hat kein 'label'"
            assert hasattr(cls, "badge_class"), f"{name} hat kein 'badge_class'"
            assert isinstance(cls.name, str)
            assert isinstance(cls.label, str)
            assert isinstance(cls.badge_class, str)

    def test_builtins_present_with_external_plugin(self):
        """Built-in-Konnektoren bleiben registriert neben externem Plugin."""
        FakeCrawler = _make_fake_connector("webcrawler", "Web-Crawler", "primary")
        mock_ep = MagicMock()
        mock_ep.name = "webcrawler"
        mock_ep.load.side_effect = lambda: registry.register(FakeCrawler)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        names = registry.connector_names()
        for builtin in ("local", "smb", "webdav", "imap", "exchange_online", "sql"):
            assert builtin in names, f"Built-in '{builtin}' fehlt"
        assert "webcrawler" in names


class TestEntryPointSkipsDuplicates:
    """3. Entry-Point ueberspringt bereits registrierte Konnektoren."""

    def test_builtin_has_priority_over_entry_point(self):
        """Wenn ein EP denselben Namen wie ein Built-in hat, wird EP uebersprungen."""
        mock_ep = MagicMock()
        mock_ep.name = "local"  # Kollidiert mit Built-in

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        # EP.load() darf NICHT aufgerufen werden
        mock_ep.load.assert_not_called()
        # Built-in ist weiterhin registriert
        assert "local" in registry.connector_names()

    def test_duplicate_ep_does_not_overwrite(self):
        """Der Built-in-Konnektor bleibt nach EP-Discovery unveraendert."""
        mock_ep = MagicMock()
        mock_ep.name = "smb"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.ensure_loaded()

        cls = registry.get("smb")
        # Sicherstellen, dass es der Original-Built-in ist (hat config_schema)
        schema = cls.config_schema()
        assert "server" in schema or "share" in schema


class TestMultipleExternalPlugins:
    """4. Mehrere externe Plugins gleichzeitig."""

    def test_two_plugins_registered(self):
        """Zwei externe Plugins werden beide registriert."""
        FakeA = _make_fake_connector("plugin_a", "Plugin A")
        FakeB = _make_fake_connector("plugin_b", "Plugin B")

        mock_ep_a = MagicMock()
        mock_ep_a.name = "plugin_a"
        mock_ep_a.load.side_effect = lambda: registry.register(FakeA)

        mock_ep_b = MagicMock()
        mock_ep_b.name = "plugin_b"
        mock_ep_b.load.side_effect = lambda: registry.register(FakeB)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep_a, mock_ep_b]):
            registry.ensure_loaded()

        names = registry.connector_names()
        assert "plugin_a" in names
        assert "plugin_b" in names

    def test_broken_plugin_does_not_block_others(self):
        """Ein fehlerhaftes Plugin blockiert andere Plugins nicht."""
        FakeGood = _make_fake_connector("good_plugin", "Good Plugin")

        mock_broken = MagicMock()
        mock_broken.name = "broken_plugin"
        mock_broken.load.side_effect = ImportError("Modul nicht gefunden")

        mock_good = MagicMock()
        mock_good.name = "good_plugin"
        mock_good.load.side_effect = lambda: registry.register(FakeGood)

        with patch("importlib.metadata.entry_points", return_value=[mock_broken, mock_good]):
            registry.ensure_loaded()

        assert "good_plugin" in registry.connector_names()
        assert "broken_plugin" not in registry.connector_names()

    def test_three_plugins_mixed_failure(self):
        """3 Plugins: 2 OK, 1 fehlerhaft -> 2 registriert."""
        FakeA = _make_fake_connector("ext_a", "Ext A")
        FakeC = _make_fake_connector("ext_c", "Ext C")

        ep_a = MagicMock(name="ext_a")
        ep_a.name = "ext_a"
        ep_a.load.side_effect = lambda: registry.register(FakeA)

        ep_b = MagicMock(name="ext_b_broken")
        ep_b.name = "ext_b_broken"
        ep_b.load.side_effect = RuntimeError("kaputt")

        ep_c = MagicMock(name="ext_c")
        ep_c.name = "ext_c"
        ep_c.load.side_effect = lambda: registry.register(FakeC)

        with patch("importlib.metadata.entry_points", return_value=[ep_a, ep_b, ep_c]):
            registry.ensure_loaded()

        names = registry.connector_names()
        assert "ext_a" in names
        assert "ext_c" in names
        assert "ext_b_broken" not in names


class TestLoadOrder:
    """5. _load_entry_points() vs _load_builtin_modules() Reihenfolge."""

    def test_builtins_loaded_before_entry_points(self):
        """Built-ins werden VOR Entry-Points geladen."""
        load_order = []

        original_load_builtins = registry._load_builtin_modules
        original_load_eps = registry._load_entry_points

        def tracked_builtins():
            load_order.append("builtins")
            original_load_builtins()

        def tracked_eps():
            load_order.append("entry_points")
            # Keine echten EPs laden

        with patch.object(registry, "_load_builtin_modules", tracked_builtins):
            with patch.object(registry, "_load_entry_points", tracked_eps):
                registry.ensure_loaded()

        assert load_order == ["builtins", "entry_points"]

    def test_builtin_names_available_during_ep_loading(self):
        """Waehrend EP-Loading sind Built-in-Namen bereits in der Registry."""
        names_during_ep = []

        def capture_ep_names():
            names_during_ep.extend(registry.connector_names())

        with patch.object(registry, "_load_entry_points", capture_ep_names):
            registry.ensure_loaded()

        # Built-ins muessen bereits registriert sein
        for name in ("local", "smb", "webdav", "imap", "exchange_online", "sql"):
            assert name in names_during_ep, (
                f"Built-in '{name}' war waehrend EP-Loading noch nicht registriert"
            )


class TestLoadedFlag:
    """6. ensure_loaded() setzt _loaded korrekt (K1)."""

    def test_loaded_false_before_ensure_loaded(self):
        """_loaded ist False vor ensure_loaded()."""
        assert registry._loaded is False

    def test_loaded_true_after_ensure_loaded(self):
        """_loaded ist True nach ensure_loaded()."""
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.ensure_loaded()
        assert registry._loaded is True

    def test_loaded_true_even_after_ep_exception(self):
        """Bei Exception in _load_entry_points() wird _loaded trotzdem True."""
        with patch("importlib.metadata.entry_points", side_effect=Exception("EP kaputt")):
            registry.ensure_loaded()

        # _load_entry_points faengt die Exception intern ab
        assert registry._loaded is True
        # Built-ins sind trotzdem geladen
        assert len(registry.connector_names()) >= 5

    def test_second_ensure_loaded_is_noop(self):
        """Zweiter ensure_loaded()-Aufruf aendert nichts."""
        with patch("importlib.metadata.entry_points", return_value=[]):
            registry.ensure_loaded()

        first_count = len(registry.connector_names())
        first_names = set(registry.connector_names())

        # Zweiter Aufruf
        registry.ensure_loaded()

        assert len(registry.connector_names()) == first_count
        assert set(registry.connector_names()) == first_names

    def test_loaded_remains_false_during_loading(self):
        """_loaded ist False waehrend des Ladevorgangs."""
        loaded_during_builtins = []

        original_builtins = registry._load_builtin_modules

        def check_loaded_builtins():
            loaded_during_builtins.append(registry._loaded)
            original_builtins()

        with patch.object(registry, "_load_builtin_modules", check_loaded_builtins):
            with patch("importlib.metadata.entry_points", return_value=[]):
                registry.ensure_loaded()

        assert loaded_during_builtins[0] is False
        assert registry._loaded is True
