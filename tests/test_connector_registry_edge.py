"""Edge-Case-Tests fuer ConnectorRegistry (Phase 11.1).

Testet Idempotenz, partielle Import-Fehler, get() mit Sonderwerten
und auto_mount_point Grenzwerte.
"""

import logging
import sys
from unittest.mock import patch

import pytest

from kiara.connectors import registry as reg


@pytest.fixture(autouse=True)
def _reset_registry():
    """Setzt die Registry vor jedem Test zurueck.

    Sichert den vollstaendig geladenen Zustand, damit nachfolgende
    Test-Dateien eine gefuellte Registry vorfinden.
    """
    # Sicherstellen, dass die Registry geladen ist bevor wir sichern
    if not reg._loaded:
        reg.ensure_loaded()
    old_registry = dict(reg._registry)
    old_loaded = reg._loaded
    yield
    reg._registry.clear()
    reg._registry.update(old_registry)
    reg._loaded = old_loaded


# ---------------------------------------------------------------------------
# 1.1 ensure_loaded() — Idempotenz und Fehlertoleranz
# ---------------------------------------------------------------------------


class TestEnsureLoadedEdge:
    def test_ensure_loaded_idempotent(self):
        """Doppelter Aufruf liefert identische Ergebnisse."""
        # Nicht _registry leeren — das wuerde einen Reimport erfordern,
        # aber __import__ fuehrt Modul-Code (und @register) nur beim ersten Mal aus.
        reg.ensure_loaded()
        names_first = reg.connector_names()

        reg.ensure_loaded()
        names_second = reg.connector_names()

        assert names_first == names_second
        assert len(names_first) >= 5

    def test_ensure_loaded_partial_import_failure(self):
        """Einzelner fehlender Import blockiert nicht die anderen."""
        reg._loaded = False
        reg._registry.clear()

        # Module aus sys.modules entfernen, damit __import__ sie neu laedt
        connector_modules = [
            "kiara.connectors.local_connector",
            "kiara.connectors.smb_connector",
            "kiara.connectors.webdav_connector",
            "kiara.connectors.imap_connector",
            "kiara.connectors.exchange_connector",
            "kiara.connectors.sql_ds_connector",
        ]
        saved = {m: sys.modules.pop(m) for m in connector_modules if m in sys.modules}
        try:
            # smb_connector auf None setzen → ImportError bei __import__
            sys.modules["kiara.connectors.smb_connector"] = None
            reg.ensure_loaded()

            names = reg.connector_names()
            assert "smb" not in names
            for expected in ("local", "webdav", "imap", "exchange_online", "sql"):
                assert expected in names
        finally:
            # Module wiederherstellen
            sys.modules.pop("kiara.connectors.smb_connector", None)
            sys.modules.update(saved)

    def test_ensure_loaded_all_imports_fail(self, caplog):
        """Alle 5 Imports fehlerhaft: leere Registry, kein Crash."""
        reg._loaded = False
        reg._registry.clear()

        bad_modules = {
            "kiara.connectors.local_connector": None,
            "kiara.connectors.smb_connector": None,
            "kiara.connectors.webdav_connector": None,
            "kiara.connectors.imap_connector": None,
            "kiara.connectors.exchange_connector": None,
            "kiara.connectors.sql_ds_connector": None,
        }
        with patch.dict(sys.modules, bad_modules):
            with caplog.at_level(logging.ERROR, logger="kiara.connectors.registry"):
                reg.ensure_loaded()

        assert reg.connector_names() == []
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) >= 6


# ---------------------------------------------------------------------------
# 1.2 get() — Unbekannte Typen
# ---------------------------------------------------------------------------


class TestGetEdge:
    def setup_method(self):
        reg.ensure_loaded()

    def test_get_none_type(self):
        with pytest.raises(KeyError):
            reg.get(None)

    def test_get_empty_string(self):
        with pytest.raises(KeyError):
            reg.get("")

    def test_get_case_sensitive_upper(self):
        with pytest.raises(KeyError):
            reg.get("IMAP")

    def test_get_case_sensitive_title(self):
        with pytest.raises(KeyError):
            reg.get("Imap")


# ---------------------------------------------------------------------------
# 1.3 auto_mount_point() — Grenzwerte
# ---------------------------------------------------------------------------


class TestAutoMountPointEdge:
    def test_auto_mount_point_empty_name(self):
        row = {"name": "", "id": 1}
        assert reg.auto_mount_point(row) == "/mnt/kiara-ds-1"

    def test_auto_mount_point_unicode_name(self):
        row = {"name": "Büro-Daten (ÄÖÜ)", "id": 5}
        assert reg.auto_mount_point(row) == "/mnt/kiara-ds-5"

    def test_auto_mount_point_id_zero(self):
        row = {"name": "Test", "id": 0}
        assert reg.auto_mount_point(row) == "/mnt/kiara-ds-0"

    def test_auto_mount_point_negative_id(self):
        row = {"name": "Test", "id": -1}
        assert reg.auto_mount_point(row) == "/mnt/kiara-ds--1"

    def test_auto_mount_point_name_only_special_chars(self):
        row = {"name": "!!!???", "id": 1}
        assert reg.auto_mount_point(row) == "/mnt/kiara-ds-1"
