"""Edge-Case-Tests fuer Konnektor-Adapter (Phase 11.1).

Testet leere Configs, Schema-Konsistenz, SMB mount/unmount,
WebDAV test_connection, IMAP/SQL Key-Translation.
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.connectors.registry import all_connectors, ensure_loaded
from kiara.connectors.registry import get as get_connector


@pytest.fixture(autouse=True)
def _load_connectors():
    from kiara.connectors import registry as _reg

    if _reg._loaded and not _reg._registry:
        _reg._loaded = False
    ensure_loaded()


# ---------------------------------------------------------------------------
# 4.1 Alle Konnektoren — Leere Config
# ---------------------------------------------------------------------------


class TestAllConnectorsEmptyConfig:
    def test_all_connectors_source_display_empty_config(self):
        """source_display({}, {}) darf nie crashen."""
        for name, cls in all_connectors().items():
            result = cls.source_display({}, {})
            assert isinstance(result, str), f"{name}.source_display crashed"

    def test_all_connectors_effective_path_empty_config(self):
        """effective_path({}, row) darf nie crashen."""
        for name, cls in all_connectors().items():
            row = {"id": 1, "name": "Test"}
            if name == "local":
                # LocalConnector greift auf load_config() zu
                with patch(
                    "kiara.config.load_config", return_value={"datasource": {"path": "/tmp"}}
                ):
                    result = cls.effective_path({}, row)
            else:
                result = cls.effective_path({}, row)
            assert isinstance(result, str), f"{name}.effective_path crashed"

    def test_all_connectors_default_config_matches_schema(self):
        """Alle Keys in default_config() muessen in config_schema() enthalten sein."""
        for name, cls in all_connectors().items():
            defaults = cls.default_config()
            schema = cls.config_schema()
            for key in defaults:
                assert key in schema, f"{name}: default_config Key '{key}' nicht in config_schema"

    def test_all_connectors_secret_fields_subset_of_schema(self):
        """Jedes Feld in secret_fields() muss in config_schema() existieren."""
        for name, cls in all_connectors().items():
            secrets = cls.secret_fields()
            schema = cls.config_schema()
            for field in secrets:
                assert field in schema, f"{name}: secret_field '{field}' nicht in config_schema"


# ---------------------------------------------------------------------------
# 4.2 SMB-Connector — prepare_and_scan()
# ---------------------------------------------------------------------------


class TestSMBPrepareAndScan:
    def setup_method(self):
        self.cls = get_connector("smb")

    @patch("kiara.shared.mount.is_mounted", return_value=False)
    @patch("kiara.shared.mount.mount_smb", return_value=(True, "ok"))
    def test_smb_prepare_and_scan_mount_success(self, mock_mount, mock_ismount):
        config = {"server": "192.168.1.10", "share": "Daten"}
        row = {"id": 1, "name": "SMB-Test"}
        result = self.cls.prepare_and_scan(config, row, {})
        assert result is not None
        assert "/mnt/kiara-" in result
        mock_mount.assert_called_once()

    @patch("kiara.shared.mount.is_mounted", return_value=False)
    @patch("kiara.shared.mount.mount_smb", return_value=(False, "Permission denied"))
    def test_smb_prepare_and_scan_mount_failure(self, mock_mount, mock_ismount):
        config = {"server": "192.168.1.10", "share": "Daten"}
        row = {"id": 1, "name": "SMB-Test"}
        result = self.cls.prepare_and_scan(config, row, {})
        assert result is None

    @patch("kiara.shared.mount.mount_source", return_value="//192.168.1.10/Daten")
    @patch("kiara.shared.mount.is_mounted", return_value=True)
    @patch("kiara.shared.mount.mount_smb")
    def test_smb_prepare_and_scan_already_mounted(
        self,
        mock_mount,
        mock_ismount,
        mock_source,
    ):
        config = {"server": "192.168.1.10", "share": "Daten"}
        row = {"id": 1, "name": "SMB-Test"}
        result = self.cls.prepare_and_scan(config, row, {})
        assert result is not None
        mock_mount.assert_not_called()

    @patch("kiara.shared.mount.unmount", return_value=(True, "ok"))
    @patch("kiara.shared.mount.mount_source", return_value="//192.0.2.99/Alt")
    @patch("kiara.shared.mount.is_mounted", side_effect=[True, False])
    @patch("kiara.shared.mount.mount_smb", return_value=(True, "ok"))
    def test_smb_prepare_and_scan_remounts_source_mismatch(
        self,
        mock_mount,
        mock_ismount,
        mock_source,
        mock_unmount,
    ):
        config = {"server": "192.168.1.10", "share": "Daten"}
        row = {"id": 1, "name": "SMB-Test"}

        result = self.cls.prepare_and_scan(config, row, {})

        assert result is not None
        mock_unmount.assert_called_once()
        mock_mount.assert_called_once()

    @patch("kiara.shared.mount.unmount", return_value=(False, "busy"))
    @patch("kiara.shared.mount.mount_source", return_value="//192.0.2.99/Alt")
    @patch("kiara.shared.mount.is_mounted", return_value=True)
    @patch("kiara.shared.mount.mount_smb")
    def test_smb_prepare_and_scan_source_mismatch_unmount_failure(
        self,
        mock_mount,
        mock_ismount,
        mock_source,
        mock_unmount,
    ):
        config = {"server": "192.168.1.10", "share": "Daten"}
        row = {"id": 1, "name": "SMB-Test"}

        result = self.cls.prepare_and_scan(config, row, {})

        assert result is None
        mock_mount.assert_not_called()


# ---------------------------------------------------------------------------
# 4.3 WebDAV-Connector — test_connection()
# ---------------------------------------------------------------------------


class TestWebDAVTestConnection:
    def setup_method(self):
        self.cls = get_connector("webdav")

    @patch("httpx.Client")
    def test_webdav_test_connection_success(self, mock_client_cls):
        mock_resp = MagicMock(status_code=207)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.request.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        config = {"url": "https://cloud.de/dav/", "username": "u", "password": "p"}
        result = self.cls.test_connection(config, {})
        assert result["success"] is True
        assert mock_client_cls.call_args.kwargs.get("verify") is not False

    @patch("httpx.Client")
    def test_webdav_test_connection_auth_failure(self, mock_client_cls):
        mock_resp = MagicMock(status_code=401)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.request.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        config = {"url": "https://cloud.de/dav/", "username": "u", "password": "p"}
        result = self.cls.test_connection(config, {})
        assert result["success"] is False
        assert "401" in result["message"]

    @patch("httpx.Client")
    def test_webdav_test_connection_timeout(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.request.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        config = {"url": "https://cloud.de/dav/", "username": "u", "password": "p"}
        result = self.cls.test_connection(config, {})
        assert result["success"] is False

    def test_webdav_test_connection_no_url(self):
        """Leere URL: kein Crash."""
        config = {"url": ""}
        result = self.cls.test_connection(config, {})
        assert result["success"] is False


# ---------------------------------------------------------------------------
# 4.4 IMAP-Connector — Config-Key-Translation
# ---------------------------------------------------------------------------


class TestIMAPPrepareAndScan:
    def setup_method(self):
        self.cls = get_connector("imap")

    @patch("kiara.ingestion.imap_scanner.IMAPScanner")
    @patch("os.makedirs")
    def test_imap_prepare_and_scan_translates_keys(self, mock_makedirs, mock_scanner_cls):
        """Config-Keys werden korrekt in imap_*-Keys uebersetzt."""
        mock_scanner = MagicMock()
        mock_scanner_cls.return_value = mock_scanner

        config = {
            "server": "mail.de",
            "port": 993,
            "username": "u",
            "password": "p",
            "ssl": True,
            "folders": "INBOX",
            "since_days": 30,
            "download_attachments": True,
            "max_attachment_mb": 10,
        }
        row = {"id": 5, "name": "Mail"}
        result = self.cls.prepare_and_scan(config, row, {})

        # Scanner wurde mit uebersetzten Keys erzeugt
        call_args = mock_scanner_cls.call_args[0][0]
        assert call_args["imap_server"] == "mail.de"
        assert call_args["imap_port"] == 993
        assert call_args["imap_username"] == "u"
        assert call_args["imap_password"] == "p"
        assert call_args["imap_ssl"] is True
        assert call_args["imap_folders"] == "INBOX"
        assert call_args["imap_since_days"] == 30
        assert call_args["imap_download_attachments"] is True
        assert call_args["imap_max_attachment_mb"] == 10
        assert result == "/var/lib/kiara/mail-cache/ds5"

    @patch("kiara.ingestion.imap_scanner.IMAPScanner")
    @patch("os.makedirs")
    def test_imap_prepare_and_scan_scanner_exception(self, mock_makedirs, mock_scanner_cls):
        """Scanner-Exception: None zurueck, logger.error() aufgerufen."""
        mock_scanner = MagicMock()
        mock_scanner.scan.side_effect = Exception("IMAP connection failed")
        mock_scanner_cls.return_value = mock_scanner

        config = {"server": "mail.de"}
        row = {"id": 1, "name": "Mail"}
        result = self.cls.prepare_and_scan(config, row, {})
        assert result is None


# ---------------------------------------------------------------------------
# 4.5 SQL-Connector — Config-Key-Translation
# ---------------------------------------------------------------------------


class TestSQLPrepareAndScan:
    def setup_method(self):
        self.cls = get_connector("sql")

    @patch("kiara.ingestion.sql_scanner.SQLScanner")
    @patch("os.makedirs")
    def test_sql_prepare_and_scan_translates_keys(self, mock_makedirs, mock_scanner_cls):
        """Config-Keys werden korrekt in sql_*-Keys uebersetzt."""
        mock_scanner = MagicMock()
        mock_scanner_cls.return_value = mock_scanner

        config = {
            "dsn": "postgresql://localhost:5432/test",
            "driver": "postgresql",
            "query": "SELECT 1",
            "text_columns": "",
            "id_column": "id",
            "timestamp_column": "",
            "text_template": "",
        }
        row = {"id": 3, "name": "DB"}
        result = self.cls.prepare_and_scan(config, row, {})

        call_args = mock_scanner_cls.call_args[0][0]
        assert call_args["sql_dsn"] == "postgresql://localhost:5432/test"
        assert call_args["sql_driver"] == "postgresql"
        assert call_args["sql_query"] == "SELECT 1"
        assert call_args["sql_text_columns"] == ""
        assert call_args["sql_id_column"] == "id"
        assert call_args["sql_timestamp_column"] == ""
        assert call_args["sql_text_template"] == ""
        assert result == "/var/lib/kiara/sql-cache/ds3"

    @patch("kiara.ingestion.sql_scanner.SQLScanner")
    @patch("os.makedirs")
    def test_sql_prepare_and_scan_scanner_exception(self, mock_makedirs, mock_scanner_cls):
        """Scanner-Exception: None zurueck, logger.error() aufgerufen."""
        mock_scanner = MagicMock()
        mock_scanner.scan.side_effect = Exception("Connection refused")
        mock_scanner_cls.return_value = mock_scanner

        config = {"dsn": "pg://host/db"}
        row = {"id": 1, "name": "DB"}
        result = self.cls.prepare_and_scan(config, row, {})
        assert result is None
