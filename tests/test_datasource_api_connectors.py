"""Tests fuer Datasource-API Hilfsfunktionen (Phase 11.1 Refactoring).

Testet _source_display(), _effective_path(), _mask_row() mit dem neuen
connector_config-JSON-Format statt Einzelspalten.
"""

import json

import pytest

from kiara.connectors.registry import ensure_loaded
from kiara.web.api_datasource import _effective_path, _mask_row, _source_display


@pytest.fixture(autouse=True)
def _load_connectors():
    ensure_loaded()


# ---------------------------------------------------------------------------
# _source_display() — IMAP
# ---------------------------------------------------------------------------


class TestSourceDisplayIMAP:
    def test_source_display_imap(self):
        """IMAP mit Server und User -> 'user@server'."""
        row = {
            "type": "imap",
            "connector_config": json.dumps(
                {
                    "server": "mail.firma.de",
                    "username": "user@firma.de",
                }
            ),
        }
        result = _source_display(row)
        assert result == "user@firma.de@mail.firma.de"

    def test_source_display_imap_no_user(self):
        """IMAP ohne Username -> nur Server."""
        row = {
            "type": "imap",
            "connector_config": json.dumps(
                {
                    "server": "mail.firma.de",
                    "username": "",
                }
            ),
        }
        result = _source_display(row)
        assert result == "mail.firma.de"


# ---------------------------------------------------------------------------
# _source_display() — SQL
# ---------------------------------------------------------------------------


class TestSourceDisplaySQL:
    def test_source_display_sql(self):
        """SQL mit Driver und Query -> '[driver] query'."""
        row = {
            "type": "sql",
            "connector_config": json.dumps(
                {
                    "dsn": "DRIVER={ODBC};SERVER=srv",
                    "driver": "mssql",
                    "query": "SELECT * FROM auftraege",
                }
            ),
        }
        result = _source_display(row)
        assert result == "[mssql] SELECT * FROM auftraege"

    def test_source_display_sql_long_query(self):
        """Lange SQL-Query (>40 Zeichen) wird abgeschnitten."""
        long_query = "SELECT a, b, c, d, e, f, g, h FROM very_long_table_name WHERE x = 1"
        row = {
            "type": "sql",
            "connector_config": json.dumps(
                {
                    "dsn": "some_dsn",
                    "driver": "mssql",
                    "query": long_query,
                }
            ),
        }
        result = _source_display(row)
        assert result.endswith("...")
        assert "[mssql]" in result
        query_part = result.split("] ")[1]
        assert len(query_part) == 43  # 40 + "..."


# ---------------------------------------------------------------------------
# _effective_path()
# ---------------------------------------------------------------------------


class TestEffectivePathIMAP:
    def test_effective_path_imap(self):
        """IMAP mit id=5 -> /var/lib/kiara/mail-cache/ds5."""
        row = {
            "type": "imap",
            "id": 5,
            "connector_config": json.dumps({}),
        }
        result = _effective_path(row)
        assert result == "/var/lib/kiara/mail-cache/ds5"


class TestEffectivePathSQL:
    def test_effective_path_sql(self):
        """SQL mit id=7 -> /var/lib/kiara/sql-cache/ds7."""
        row = {
            "type": "sql",
            "id": 7,
            "connector_config": json.dumps({}),
        }
        result = _effective_path(row)
        assert result == "/var/lib/kiara/sql-cache/ds7"


# ---------------------------------------------------------------------------
# _mask_row()
# ---------------------------------------------------------------------------


class TestMaskRowHidesPasswords:
    """_mask_row() maskiert sensible Felder innerhalb connector_config."""

    def test_mask_row_hides_imap_password(self):
        """IMAP-Passwort in connector_config wird zu '********' maskiert."""
        row = {
            "type": "imap",
            "connector_config": json.dumps(
                {
                    "server": "mail.firma.de",
                    "password": "geheim123",
                }
            ),
        }
        masked = _mask_row(row)
        cc = masked["connector_config"]
        assert cc["password"] == "********"
        assert cc["server"] == "mail.firma.de"

    def test_mask_row_hides_sql_dsn(self):
        """SQL-DSN in connector_config wird zu '********' maskiert."""
        row = {
            "type": "sql",
            "connector_config": json.dumps(
                {
                    "dsn": "DRIVER={ODBC};SERVER=srv;UID=user;PWD=secret",
                }
            ),
        }
        masked = _mask_row(row)
        cc = masked["connector_config"]
        assert cc["dsn"] == "********"

    def test_mask_row_hides_smb_password(self):
        """SMB-Passwort in connector_config wird maskiert."""
        row = {
            "type": "smb",
            "connector_config": json.dumps(
                {
                    "server": "192.168.1.10",
                    "share": "Daten",
                    "password": "secret",
                }
            ),
        }
        masked = _mask_row(row)
        cc = masked["connector_config"]
        assert cc["password"] == "********"
        assert cc["server"] == "192.168.1.10"

    def test_mask_row_empty_password_not_masked(self):
        """Leeres Passwort bleibt leer (nicht '********')."""
        row = {
            "type": "imap",
            "connector_config": json.dumps(
                {
                    "server": "mail.firma.de",
                    "password": "",
                }
            ),
        }
        masked = _mask_row(row)
        cc = masked["connector_config"]
        assert cc["password"] == ""

    def test_mask_row_strips_old_columns(self):
        """Alte Konnektor-Spalten werden aus dem Response entfernt."""
        row = {
            "type": "imap",
            "connector_config": json.dumps({"server": "mail.firma.de"}),
            "imap_server": "mail.firma.de",
            "imap_password": "geheim",
            "smb_server": "",
        }
        masked = _mask_row(row)
        assert "imap_server" not in masked
        assert "imap_password" not in masked
        assert "smb_server" not in masked
