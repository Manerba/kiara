"""Tests fuer Konnektor-Adapter (Phase 11.1).

Testet die reine Logik jedes Konnektors: Schema, source_display,
effective_path, secret_fields — ohne I/O oder Netzwerk.
"""

import inspect
from unittest.mock import patch

import pytest

from kiara.connectors.browse import ConnectorBrowseError
from kiara.connectors.registry import ensure_loaded
from kiara.connectors.registry import get as get_connector


@pytest.fixture(autouse=True)
def _load_connectors():
    ensure_loaded()


# ---------------------------------------------------------------------------
# LocalConnector
# ---------------------------------------------------------------------------


class TestLocalConnector:
    def setup_method(self):
        self.cls = get_connector("local")

    def test_metadata(self):
        assert self.cls.name == "local"
        assert self.cls.label == "Lokal"
        assert self.cls.supports_browse is True
        descriptor = self.cls.__dict__["list_directory"]
        assert isinstance(descriptor, staticmethod)
        assert inspect.isfunction(descriptor.__func__)

    def test_config_schema_has_path(self):
        schema = self.cls.config_schema()
        assert "path" in schema
        assert schema["path"]["type"] == "str"

    def test_secret_fields_empty(self):
        assert self.cls.secret_fields() == []

    def test_source_display_with_path(self):
        config = {"path": "/mnt/data"}
        assert self.cls.source_display(config, {}) == "/mnt/data"

    def test_source_display_fallback_to_row_path(self):
        config = {}
        row = {"path": "/fallback"}
        assert self.cls.source_display(config, row) == "/fallback"

    def test_source_display_empty(self):
        assert self.cls.source_display({}, {}) == ""

    def test_effective_path_from_config(self):
        config = {"path": "/mnt/custom"}
        assert self.cls.effective_path(config, {}) == "/mnt/custom"

    @patch("kiara.config.load_config", return_value={"datasource": {"path": "/default/path"}})
    def test_effective_path_fallback_to_global(self, mock_cfg):
        assert self.cls.effective_path({}, {}) == "/default/path"

    def test_prepare_and_scan_returns_path(self, tmp_path):
        config = {"path": str(tmp_path)}
        assert self.cls.prepare_and_scan(config, {}, {}) == str(tmp_path)

    def test_prepare_and_scan_missing_path_returns_none(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        config = {"path": str(missing)}
        assert self.cls.prepare_and_scan(config, {}, {}) is None

    def test_prepare_and_scan_empty_path_returns_none(self):
        with patch("kiara.config.load_config", return_value={"datasource": {"path": ""}}):
            assert self.cls.prepare_and_scan({}, {}, {}) is None


# ---------------------------------------------------------------------------
# IMAPConnector
# ---------------------------------------------------------------------------


class TestIMAPConnector:
    def setup_method(self):
        self.cls = get_connector("imap")

    def test_metadata(self):
        assert self.cls.name == "imap"
        assert self.cls.label == "E-Mail (IMAP)"
        assert getattr(self.cls, "supports_browse", False) is False
        with pytest.raises(ConnectorBrowseError) as exc_info:
            self.cls.list_directory({})
        assert exc_info.value.reason == "unsupported"

    def test_config_schema_fields(self):
        schema = self.cls.config_schema()
        for field in (
            "server",
            "port",
            "ssl",
            "username",
            "password",
            "folders",
            "since_days",
            "download_attachments",
        ):
            assert field in schema, f"Feld '{field}' fehlt"

    def test_secret_fields_has_password(self):
        assert "password" in self.cls.secret_fields()
        assert len(self.cls.secret_fields()) == 1

    def test_source_display_with_user(self):
        config = {"server": "mail.firma.de", "username": "user@firma.de"}
        assert self.cls.source_display(config, {}) == "user@firma.de@mail.firma.de"

    def test_source_display_no_user(self):
        config = {"server": "mail.firma.de", "username": ""}
        assert self.cls.source_display(config, {}) == "mail.firma.de"

    def test_source_display_empty(self):
        assert self.cls.source_display({}, {}) == ""

    def test_effective_path(self):
        row = {"id": 5}
        assert self.cls.effective_path({}, row) == "/var/lib/kiara/mail-cache/ds5"


# ---------------------------------------------------------------------------
# SQLDatasourceConnector
# ---------------------------------------------------------------------------


class TestSQLDatasourceConnector:
    def setup_method(self):
        self.cls = get_connector("sql")

    def test_metadata(self):
        assert self.cls.name == "sql"
        assert self.cls.label == "SQL-Datenbank"
        assert getattr(self.cls, "supports_browse", False) is False
        with pytest.raises(ConnectorBrowseError) as exc_info:
            self.cls.list_directory({})
        assert exc_info.value.reason == "unsupported"

    def test_config_schema_fields(self):
        schema = self.cls.config_schema()
        for field in (
            "dsn",
            "driver",
            "query",
            "text_columns",
            "id_column",
            "timestamp_column",
            "text_template",
        ):
            assert field in schema, f"Feld '{field}' fehlt"

    def test_secret_fields_has_dsn(self):
        assert "dsn" in self.cls.secret_fields()

    def test_source_display_full(self):
        config = {"dsn": "pg://host/db", "driver": "mssql", "query": "SELECT * FROM auftraege"}
        assert self.cls.source_display(config, {}) == "[mssql] SELECT * FROM auftraege"

    def test_source_display_long_query_truncated(self):
        long_query = "SELECT a, b, c, d, e, f, g, h FROM very_long_table_name WHERE x = 1"
        config = {"dsn": "x", "driver": "mssql", "query": long_query}
        result = self.cls.source_display(config, {})
        assert result.endswith("...")
        assert "[mssql]" in result
        query_part = result.split("] ")[1]
        assert len(query_part) == 43  # 40 + "..."

    def test_source_display_no_dsn(self):
        assert self.cls.source_display({}, {}) == ""

    def test_source_display_no_query(self):
        config = {"dsn": "x", "driver": "auto"}
        assert self.cls.source_display(config, {}) == "[auto]"

    def test_effective_path(self):
        row = {"id": 7}
        assert self.cls.effective_path({}, row) == "/var/lib/kiara/sql-cache/ds7"


# ---------------------------------------------------------------------------
# SMBConnector
# ---------------------------------------------------------------------------


class TestSMBConnector:
    def setup_method(self):
        self.cls = get_connector("smb")

    def test_metadata(self):
        assert self.cls.name == "smb"
        assert self.cls.label == "SMB-Freigabe"
        assert self.cls.supports_browse is True
        descriptor = self.cls.__dict__["list_directory"]
        assert isinstance(descriptor, staticmethod)
        assert inspect.isfunction(descriptor.__func__)

    def test_config_schema_fields(self):
        schema = self.cls.config_schema()
        for field in ("server", "share", "username", "password", "domain", "sub_path"):
            assert field in schema, f"Feld '{field}' fehlt"

    def test_secret_fields_has_password(self):
        assert "password" in self.cls.secret_fields()

    def test_source_display_with_subpath(self):
        config = {"server": "192.168.1.10", "share": "Daten", "sub_path": "Technik"}
        assert self.cls.source_display(config, {}) == "192.168.1.10:/Daten/Technik"

    def test_source_display_without_subpath(self):
        config = {"server": "192.168.1.10", "share": "Daten"}
        assert self.cls.source_display(config, {}) == "192.168.1.10:/Daten"

    def test_source_display_no_server(self):
        assert self.cls.source_display({}, {}) == ""

    def test_effective_path_with_subpath(self):
        config = {"sub_path": "Abteilung"}
        row = {"name": "Firma", "id": 3}
        path = self.cls.effective_path(config, row)
        assert path.endswith("/Abteilung")
        assert "/mnt/kiara-" in path

    def test_effective_path_blocks_escape_subpath(self):
        config = {"sub_path": "../etc"}
        row = {"name": "Firma", "id": 3}
        with pytest.raises(ValueError):
            self.cls.effective_path(config, row)

    def test_effective_path_normalizes_nested_subpath(self):
        config = {"sub_path": "/Abteilung/./Technik/"}
        row = {"name": "Firma", "id": 3}
        path = self.cls.effective_path(config, row)
        assert path == "/mnt/kiara-ds-3/Abteilung/Technik"

    def test_effective_path_without_subpath(self):
        config = {}
        row = {"name": "Firma", "id": 3}
        path = self.cls.effective_path(config, row)
        assert path == "/mnt/kiara-ds-3"


# ---------------------------------------------------------------------------
# WebDAVConnector
# ---------------------------------------------------------------------------


class TestWebDAVConnector:
    def setup_method(self):
        self.cls = get_connector("webdav")

    def test_metadata(self):
        assert self.cls.name == "webdav"
        assert self.cls.label == "WebDAV"
        assert self.cls.supports_browse is True
        descriptor = self.cls.__dict__["list_directory"]
        assert isinstance(descriptor, staticmethod)
        assert inspect.isfunction(descriptor.__func__)

    def test_config_schema_fields(self):
        schema = self.cls.config_schema()
        for field in ("url", "username", "password", "base_path"):
            assert field in schema, f"Feld '{field}' fehlt"

    def test_secret_fields_has_password(self):
        assert "password" in self.cls.secret_fields()

    def test_source_display(self):
        config = {"url": "https://cloud.firma.de/remote.php/dav/", "base_path": "/Docs"}
        assert self.cls.source_display(config, {}) == "cloud.firma.de:/Docs"

    def test_source_display_no_url(self):
        assert self.cls.source_display({}, {}) == ""

    def test_effective_path(self):
        row = {"name": "Cloud", "id": 2}
        path = self.cls.effective_path({}, row)
        assert path == "/mnt/kiara-ds-2"


# ---------------------------------------------------------------------------
# ExchangeOnlineConnector
# ---------------------------------------------------------------------------


class TestExchangeOnlineConnector:
    def setup_method(self):
        self.cls = get_connector("exchange_online")

    def test_metadata_and_no_directory_browse(self):
        assert self.cls.name == "exchange_online"
        assert self.cls.label == "Exchange Online (OAuth2)"
        assert getattr(self.cls, "supports_browse", False) is False
        with pytest.raises(ConnectorBrowseError) as exc_info:
            self.cls.list_directory({})
        assert exc_info.value.reason == "unsupported"
