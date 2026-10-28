"""Tests fuer Exchange Online Connector + OAuth2IMAPScanner."""

import imaplib
import os
import tempfile
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import pytest

from kiara.connectors.exchange_connector import ExchangeOnlineConnector
from kiara.ingestion.oauth2_imap_scanner import (
    _EXCHANGE_IMAP_PORT,
    _EXCHANGE_IMAP_SCOPE,
    _EXCHANGE_IMAP_SERVER,
    OAuth2IMAPScanner,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _setup_uid_conn(mock_conn, mail_msg):
    """Konfiguriert mock_conn fuer UID-basiertes IMAP-Scan-Protokoll.

    Setzt select, response(UIDVALIDITY), uid("search"), uid("fetch") korrekt.
    """
    mock_conn.select.return_value = ("OK", [b"1"])
    mock_conn.response.return_value = ("UIDVALIDITY", [b"12345"])

    mail_bytes = mail_msg.as_bytes() if mail_msg else b""

    def uid_dispatch(cmd, *args):
        if cmd == "search":
            return ("OK", [b"1"])
        elif cmd == "fetch":
            uid_set = args[0] if args else b"1"
            uids_requested = uid_set.split(b",") if isinstance(uid_set, bytes) else [b"1"]
            results = []
            for u in uids_requested:
                header = u + b" (UID " + u + b" RFC822 {" + str(len(mail_bytes)).encode() + b"}"
                results.append((header, mail_bytes))
                results.append(b")")
            return ("OK", results)
        return ("OK", [b""])

    mock_conn.uid.side_effect = uid_dispatch


def _base_config():
    return {
        "imap_username": "user@firma.de",
        "imap_folders": "INBOX",
        "imap_since_days": 30,
        "tenant_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "client_id": "11111111-2222-3333-4444-555555555555",
        "client_secret": "super-secret-value",
    }


def _mock_msal_app(token="mock-access-token-xyz"):
    """Erzeugt ein Mock-ConfidentialClientApplication."""
    mock_app = MagicMock()
    mock_app.acquire_token_for_client.return_value = {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    return mock_app


def _mock_msal_app_error():
    """Erzeugt ein Mock-ConfidentialClientApplication das fehlschlaegt."""
    mock_app = MagicMock()
    mock_app.acquire_token_for_client.return_value = {
        "error": "invalid_client",
        "error_description": "AADSTS7000215: Invalid client secret provided.",
    }
    return mock_app


# ---------------------------------------------------------------------------
# OAuth2IMAPScanner — Token-Abruf
# ---------------------------------------------------------------------------


class TestTokenAcquisition:
    """MSAL Client-Credentials-Flow Token-Abruf."""

    @patch("kiara.ingestion.oauth2_imap_scanner.msal", create=True)
    def test_acquire_token_success(self, mock_msal_module):
        mock_app = _mock_msal_app()
        mock_msal_module.ConfidentialClientApplication.return_value = mock_app

        with patch.dict("sys.modules", {"msal": mock_msal_module}):
            scanner = OAuth2IMAPScanner(_base_config())
            token = scanner._acquire_token()

        assert token == "mock-access-token-xyz"
        mock_msal_module.ConfidentialClientApplication.assert_called_once()
        call_kwargs = mock_msal_module.ConfidentialClientApplication.call_args[1]
        assert call_kwargs["client_id"] == "11111111-2222-3333-4444-555555555555"
        assert call_kwargs["client_credential"] == "super-secret-value"
        assert (
            call_kwargs["authority"]
            == "https://login.microsoftonline.com/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )
        # http_client mit Timeout-Session wird fuer MSAL-Timeout-Schutz uebergeben
        assert call_kwargs["http_client"] is not None
        mock_app.acquire_token_for_client.assert_called_once_with(
            scopes=[_EXCHANGE_IMAP_SCOPE],
        )

    @patch("kiara.ingestion.oauth2_imap_scanner.msal", create=True)
    def test_acquire_token_tracks_expiry(self, mock_msal_module):
        """expires_in wird fuer proaktiven Token-Refresh gespeichert."""
        mock_app = _mock_msal_app("short-token")
        mock_app.acquire_token_for_client.return_value["expires_in"] = 120
        mock_msal_module.ConfidentialClientApplication.return_value = mock_app

        with (
            patch.dict("sys.modules", {"msal": mock_msal_module}),
            patch("kiara.ingestion.oauth2_imap_scanner.time.time", return_value=1000),
        ):
            scanner = OAuth2IMAPScanner(_base_config())
            token = scanner._acquire_token()

        assert token == "short-token"
        assert scanner._token_expires_at == 1120

    @patch("kiara.ingestion.oauth2_imap_scanner.msal", create=True)
    def test_acquire_token_failure_raises(self, mock_msal_module):
        mock_app = _mock_msal_app_error()
        mock_msal_module.ConfidentialClientApplication.return_value = mock_app

        with patch.dict("sys.modules", {"msal": mock_msal_module}):
            scanner = OAuth2IMAPScanner(_base_config())
            with pytest.raises(RuntimeError, match="invalid_client"):
                scanner._acquire_token()

    def test_missing_msal_raises(self):
        """Ohne installiertes msal -> klarer Fehlerhinweis."""
        scanner = OAuth2IMAPScanner(_base_config())
        with patch.dict("sys.modules", {"msal": None}):
            with pytest.raises(RuntimeError, match="msal"):
                scanner._acquire_token()


# ---------------------------------------------------------------------------
# OAuth2IMAPScanner — connect()
# ---------------------------------------------------------------------------


class TestOAuth2Connect:
    """XOAUTH2-SASL-Authentifizierung."""

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_connect_xoauth2(self, mock_imaplib):
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn

        scanner = OAuth2IMAPScanner(_base_config())
        with patch.object(scanner, "_acquire_token", return_value="test-token"):
            conn = scanner.connect()

        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        mock_imaplib.IMAP4_SSL.assert_called_once_with(
            _EXCHANGE_IMAP_SERVER,
            _EXCHANGE_IMAP_PORT,
            timeout=CONNECTOR_CONNECT_TIMEOUT,
        )
        mock_conn.authenticate.assert_called_once()
        call_args = mock_conn.authenticate.call_args
        assert call_args[0][0] == "XOAUTH2"

        # Auth-Callback pruefen
        auth_callback = call_args[0][1]
        auth_bytes = auth_callback(None)
        auth_str = auth_bytes.decode()
        assert "user=user@firma.de" in auth_str
        assert "auth=Bearer test-token" in auth_str

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_connect_uses_exchange_server(self, mock_imaplib):
        """Server/Port werden automatisch auf outlook.office365.com:993 gesetzt."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn

        config = _base_config()
        scanner = OAuth2IMAPScanner(config)

        assert scanner._server == _EXCHANGE_IMAP_SERVER
        assert scanner._port == _EXCHANGE_IMAP_PORT

    def test_auth_token_errors_are_reconnectable(self):
        """Exchange-OAuth-Ablehnungen triggern Reconnect statt stillem Fetch-Fehler."""
        scanner = OAuth2IMAPScanner(_base_config())

        assert scanner._is_connection_error(
            imaplib.IMAP4.error(b"NO [AUTHENTICATIONFAILED] Invalid credentials")
        )
        assert scanner._is_connection_error(
            imaplib.IMAP4.error(b"BAD User is authenticated but not connected")
        )
        assert scanner._is_connection_error(imaplib.IMAP4.error(b"401 invalid token"))

    def test_ensure_connection_ready_refreshes_expiring_token(self):
        """Vor IMAP-Operationen wird bei nahendem Ablauf reconnectet."""
        scanner = OAuth2IMAPScanner(
            {
                **_base_config(),
                "oauth2_token_refresh_skew_seconds": 300,
            }
        )
        conn = MagicMock(name="old_conn")
        new_conn = MagicMock(name="new_conn")
        scanner._token_expires_at = time.time() + 30

        with patch.object(scanner, "_reconnect", return_value=new_conn) as mock_reconnect:
            result = scanner._ensure_connection_ready(conn, folder="INBOX")

        assert result is new_conn
        mock_reconnect.assert_called_once_with(conn, folder="INBOX")

    def test_fetch_auth_error_reconnects_once_and_retries(self):
        """Auth-Fehler mid-scan fuehrt zu frischem Token/Reconnnect und einem Retry."""
        scanner = OAuth2IMAPScanner(_base_config())
        old_conn = MagicMock(name="old_conn")
        new_conn = MagicMock(name="new_conn")

        with (
            patch.object(scanner, "_reconnect", return_value=new_conn) as mock_reconnect,
            patch.object(scanner, "_fetch_batch") as mock_fetch,
        ):
            mock_fetch.side_effect = [
                imaplib.IMAP4.error(b"BAD token expired"),
                [(b"1", b"mail")],
            ]

            results, conn = scanner._fetch_batch_with_reconnect(
                old_conn,
                [b"1"],
                "INBOX",
            )

        assert results == [(b"1", b"mail")]
        assert conn is new_conn
        mock_reconnect.assert_called_once_with(old_conn, folder="INBOX")
        assert mock_fetch.call_args_list[0].args[0] is old_conn
        assert mock_fetch.call_args_list[1].args[0] is new_conn


# ---------------------------------------------------------------------------
# OAuth2IMAPScanner — erbt Scan-Logik
# ---------------------------------------------------------------------------


class TestInheritedScanLogic:
    """Stellt sicher, dass scan/test_connection vom IMAPScanner geerbt wird."""

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_scan_uses_inherited_logic(self, mock_imaplib):
        """scan() nutzt die geerbte Logik von IMAPScanner."""
        mail = MIMEMultipart()
        mail["Subject"] = "OAuth Test Mail"
        mail["From"] = "sender@firma.de"
        mail["To"] = "user@firma.de"
        mail["Date"] = "Mon, 01 Jan 2024 10:00:00 +0100"
        mail.attach(MIMEText("OAuth Body", "plain", "utf-8"))

        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn
        _setup_uid_conn(mock_conn, mail)

        scanner = OAuth2IMAPScanner(_base_config())
        with patch.object(scanner, "_acquire_token", return_value="tok"):
            with tempfile.TemporaryDirectory() as tmpdir:
                files = scanner.scan(tmpdir)
                assert len(files) == 1
                content = open(files[0], encoding="utf-8").read()
                assert "OAuth Test Mail" in content
                assert "OAuth Body" in content

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_test_connection_returns_folders(self, mock_imaplib):
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn
        mock_conn.list.return_value = (
            "OK",
            [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Sent Items"'],
        )

        scanner = OAuth2IMAPScanner(_base_config())
        with patch.object(scanner, "_acquire_token", return_value="tok"):
            result = scanner.test_connection()
            assert result["success"] is True
            assert len(result["folders"]) == 2


# ---------------------------------------------------------------------------
# ExchangeOnlineConnector — Config-Schema
# ---------------------------------------------------------------------------


class TestExchangeConnectorSchema:
    """Connector-Metadaten und Schema."""

    def test_name_and_label(self):
        assert ExchangeOnlineConnector.name == "exchange_online"
        assert "Exchange" in ExchangeOnlineConnector.label
        assert "OAuth2" in ExchangeOnlineConnector.label

    def test_schema_has_oauth_fields(self):
        schema = ExchangeOnlineConnector.config_schema()
        assert "tenant_id" in schema
        assert "client_id" in schema
        assert "client_secret" in schema
        assert schema["client_secret"].get("secret") is True

    def test_schema_has_mail_fields(self):
        schema = ExchangeOnlineConnector.config_schema()
        assert "username" in schema
        assert "folders" in schema
        assert "since_days" in schema
        assert "download_attachments" in schema

    def test_schema_no_server_port(self):
        """Server/Port nicht im Schema — fest auf outlook.office365.com."""
        schema = ExchangeOnlineConnector.config_schema()
        assert "server" not in schema
        assert "port" not in schema

    def test_secret_fields(self):
        secrets = ExchangeOnlineConnector.secret_fields()
        assert "client_secret" in secrets

    def test_required_fields(self):
        schema = ExchangeOnlineConnector.config_schema()
        required = [k for k, v in schema.items() if v.get("required")]
        assert "tenant_id" in required
        assert "client_id" in required
        assert "client_secret" in required
        assert "username" in required

    def test_source_display(self):
        config = {"username": "user@firma.de"}
        assert ExchangeOnlineConnector.source_display(config, {}) == "user@firma.de"

    def test_source_display_empty(self):
        assert ExchangeOnlineConnector.source_display({}, {}) == "Exchange Online"


# ---------------------------------------------------------------------------
# ExchangeOnlineConnector — _build_scanner_config
# ---------------------------------------------------------------------------


class TestBuildScannerConfig:
    """Scanner-Config-Mapping."""

    def test_build_maps_fields_correctly(self):
        config = {
            "username": "user@firma.de",
            "folders": "INBOX, Sent Items",
            "since_days": 60,
            "tenant_id": "t-id",
            "client_id": "c-id",
            "client_secret": "c-secret",
            "download_attachments": False,
            "max_attachment_mb": 10,
        }
        result = ExchangeOnlineConnector._build_scanner_config(config)
        assert result["imap_username"] == "user@firma.de"
        assert result["imap_folders"] == "INBOX, Sent Items"
        assert result["imap_since_days"] == 60
        assert result["tenant_id"] == "t-id"
        assert result["client_id"] == "c-id"
        assert result["client_secret"] == "c-secret"
        assert result["imap_download_attachments"] is False
        assert result["imap_max_attachment_mb"] == 10


# ---------------------------------------------------------------------------
# ExchangeOnlineConnector — test_connection / prepare_and_scan
# ---------------------------------------------------------------------------


class TestExchangeConnectorIntegration:
    """Connector-Methoden delegieren korrekt an OAuth2IMAPScanner."""

    @patch("kiara.ingestion.oauth2_imap_scanner.OAuth2IMAPScanner")
    def test_test_connection_delegates(self, MockScanner):
        mock_scanner = MagicMock()
        mock_scanner.test_connection.return_value = {
            "success": True,
            "folders": ["INBOX", "Sent Items", "Drafts"],
        }
        MockScanner.return_value = mock_scanner

        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        result = ExchangeOnlineConnector.test_connection(config, {"id": 1})
        assert result["success"] is True
        assert "3 Ordner" in result["message"]

    @patch("kiara.ingestion.oauth2_imap_scanner.OAuth2IMAPScanner")
    def test_test_connection_failure(self, MockScanner):
        mock_scanner = MagicMock()
        mock_scanner.test_connection.return_value = {
            "success": False,
            "message": "AADSTS7000215: Invalid client secret",
        }
        MockScanner.return_value = mock_scanner

        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        result = ExchangeOnlineConnector.test_connection(config, {"id": 1})
        assert result["success"] is False
        assert "Invalid client secret" in result["message"]

    @patch("kiara.ingestion.oauth2_imap_scanner.OAuth2IMAPScanner")
    def test_prepare_and_scan_creates_path(self, MockScanner):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = ["/tmp/mail.txt"]
        MockScanner.return_value = mock_scanner

        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": tmpdir}):
                path = ExchangeOnlineConnector.prepare_and_scan(
                    config,
                    {"id": 99},
                    {},
                )
                assert path is not None
                assert os.path.isdir(path)

    @patch("kiara.ingestion.oauth2_imap_scanner.OAuth2IMAPScanner")
    def test_prepare_and_scan_error_fails_closed(self, MockScanner):
        """OAuth-/Scannerfehler muessen bis zur Pipeline propagieren."""
        mock_scanner = MagicMock()
        mock_scanner.scan.side_effect = RuntimeError("Token expired")
        MockScanner.return_value = mock_scanner

        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": tmpdir}):
                with pytest.raises(RuntimeError, match="Token expired"):
                    ExchangeOnlineConnector.prepare_and_scan(
                        config,
                        {"id": 99},
                        {},
                    )


# ---------------------------------------------------------------------------
# Scan-Cache — Marker-Datei steuert Scan-Skip
# ---------------------------------------------------------------------------


class TestScanCache:
    """prepare_and_scan ueberspringt Scan wenn Marker aktuell."""

    @patch("kiara.ingestion.oauth2_imap_scanner.OAuth2IMAPScanner")
    def test_fresh_marker_skips_scan(self, MockScanner):
        """Marker < 24h alt -> Scan wird uebersprungen, Pfad trotzdem zurueckgegeben."""
        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": tmpdir}):
                ds_path = os.path.join(tmpdir, "ds42")
                os.makedirs(ds_path, exist_ok=True)
                # Frischen Marker anlegen
                with open(os.path.join(ds_path, ".scan_done"), "w") as f:
                    f.write("2026-03-02T12:00:00")

                path = ExchangeOnlineConnector.prepare_and_scan(
                    config,
                    {"id": 42},
                    {},
                )
                assert path == ds_path
                # Scanner wurde NICHT instanziiert
                MockScanner.assert_not_called()

    @patch("kiara.ingestion.oauth2_imap_scanner.OAuth2IMAPScanner")
    def test_stale_marker_triggers_scan(self, MockScanner):
        """Marker > 24h alt -> Scan wird ausgefuehrt."""
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = []
        MockScanner.return_value = mock_scanner

        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": tmpdir}):
                ds_path = os.path.join(tmpdir, "ds42")
                os.makedirs(ds_path, exist_ok=True)
                marker = os.path.join(ds_path, ".scan_done")
                with open(marker, "w") as f:
                    f.write("old")
                # Marker 25h in die Vergangenheit setzen
                old_time = time.time() - 90000
                os.utime(marker, (old_time, old_time))

                path = ExchangeOnlineConnector.prepare_and_scan(
                    config,
                    {"id": 42},
                    {},
                )
                assert path == ds_path
                mock_scanner.scan.assert_called_once_with(
                    ds_path,
                    progress_callback=None,
                    control_callback=None,
                )

    @patch("kiara.ingestion.oauth2_imap_scanner.OAuth2IMAPScanner")
    def test_no_marker_triggers_scan(self, MockScanner):
        """Kein Marker vorhanden -> Scan wird ausgefuehrt."""
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = []
        MockScanner.return_value = mock_scanner

        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": tmpdir}):
                path = ExchangeOnlineConnector.prepare_and_scan(
                    config,
                    {"id": 42},
                    {},
                )
                assert path is not None
                mock_scanner.scan.assert_called_once()


# ---------------------------------------------------------------------------
# XOAUTH2 Auth-Reject — Server lehnt Token ab
# ---------------------------------------------------------------------------


class TestAuthReject:
    """IMAP-Server lehnt XOAUTH2-Token ab."""

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_authenticate_reject_raises(self, mock_imaplib):
        """Server antwortet mit NO auf AUTHENTICATE -> Exception."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn
        mock_conn.authenticate.side_effect = imaplib.IMAP4.error(
            b"AUTHENTICATE failed: NO [AUTHENTICATIONFAILED]"
        )

        scanner = OAuth2IMAPScanner(_base_config())
        with patch.object(scanner, "_acquire_token", return_value="expired-token"):
            with pytest.raises(imaplib.IMAP4.error, match="AUTHENTICATIONFAILED"):
                scanner.connect()

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_auth_reject_in_test_connection(self, mock_imaplib):
        """test_connection faengt Auth-Fehler ab und gibt success=False zurueck."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn
        mock_conn.authenticate.side_effect = imaplib.IMAP4.error(b"AUTHENTICATE failed")

        scanner = OAuth2IMAPScanner(_base_config())
        with patch.object(scanner, "_acquire_token", return_value="bad-token"):
            result = scanner.test_connection()
            assert result["success"] is False
            assert "AUTHENTICATE" in result["message"]


# ---------------------------------------------------------------------------
# Leere / fehlende Config-Felder
# ---------------------------------------------------------------------------


class TestEmptyConfig:
    """Verhalten bei fehlenden oder leeren Config-Feldern."""

    def test_empty_tenant_id_uses_default(self):
        """Leere tenant_id -> leerer String, kein Crash bei Instanziierung."""
        config = _base_config()
        config["tenant_id"] = ""
        scanner = OAuth2IMAPScanner(config)
        assert scanner._tenant_id == ""

    def test_missing_client_secret(self):
        """Fehlender client_secret -> leerer String."""
        config = _base_config()
        del config["client_secret"]
        scanner = OAuth2IMAPScanner(config)
        assert scanner._client_secret == ""

    def test_missing_oauth_fields_token_error(self):
        """Leere OAuth-Felder -> MSAL gibt Fehler beim Token-Abruf."""
        config = _base_config()
        config["tenant_id"] = ""
        config["client_id"] = ""
        config["client_secret"] = ""

        scanner = OAuth2IMAPScanner(config)

        mock_msal = MagicMock()
        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            "error": "invalid_request",
            "error_description": "AADSTS900023: Tenant ID is empty.",
        }
        mock_msal.ConfidentialClientApplication.return_value = mock_app

        with patch.dict("sys.modules", {"msal": mock_msal}):
            with pytest.raises(RuntimeError, match="invalid_request"):
                scanner._acquire_token()

    def test_build_scanner_config_defaults(self):
        """_build_scanner_config mit leerem Dict -> sinnvolle Defaults."""
        result = ExchangeOnlineConnector._build_scanner_config({})
        assert result["imap_username"] == ""
        assert result["imap_folders"] == "INBOX"
        assert result["imap_since_days"] == 90
        assert result["imap_download_attachments"] is True
        assert result["imap_max_attachment_mb"] == 25
        assert result["tenant_id"] == ""
        assert result["client_id"] == ""
        assert result["client_secret"] == ""

    def test_default_config_from_schema(self):
        """default_config() liefert alle Schema-Felder mit Defaults."""
        defaults = ExchangeOnlineConnector.default_config()
        assert defaults["tenant_id"] == ""
        assert defaults["folders"] == "INBOX"
        assert defaults["since_days"] == 90
        assert defaults["download_attachments"] is True


# ---------------------------------------------------------------------------
# Registry-Integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    """ExchangeOnlineConnector ist korrekt in der Registry registriert."""

    def test_registered_in_registry(self):
        from kiara.connectors.registry import ensure_loaded, get

        ensure_loaded()
        cls = get("exchange_online")
        assert cls is ExchangeOnlineConnector

    def test_appears_in_connector_names(self):
        from kiara.connectors.registry import connector_names, ensure_loaded

        ensure_loaded()
        assert "exchange_online" in connector_names()

    def test_all_connectors_includes_exchange(self):
        from kiara.connectors.registry import all_connectors, ensure_loaded

        ensure_loaded()
        connectors = all_connectors()
        assert "exchange_online" in connectors
        assert connectors["exchange_online"] is ExchangeOnlineConnector


# ---------------------------------------------------------------------------
# safe_test_connection — Exception-Wrapping aus ConnectorBase
# ---------------------------------------------------------------------------


class TestSafeTestConnection:
    """BaseClass safe_test_connection faengt Exceptions ab."""

    def test_safe_test_connection_wraps_exception(self):
        """Exception in test_connection -> success=False mit message."""
        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        row = {"id": 1}

        with patch.object(
            ExchangeOnlineConnector,
            "test_connection",
            side_effect=ConnectionError("Network unreachable"),
        ):
            result = ExchangeOnlineConnector.safe_test_connection(config, row)
            assert result["success"] is False
            assert "Network unreachable" in result["message"]

    def test_safe_test_connection_passes_success(self):
        """Erfolgreicher test_connection -> wird durchgereicht."""
        config = {"username": "u", "tenant_id": "t", "client_id": "c", "client_secret": "s"}
        row = {"id": 1}

        with patch.object(
            ExchangeOnlineConnector,
            "test_connection",
            return_value={"success": True, "message": "OK"},
        ):
            result = ExchangeOnlineConnector.safe_test_connection(config, row)
            assert result["success"] is True


# ---------------------------------------------------------------------------
# OAuth2IMAPScanner — Server/Port Overrides
# ---------------------------------------------------------------------------


class TestServerDefaults:
    """Server und Port werden korrekt auf Exchange-Defaults gesetzt."""

    def test_custom_server_not_overridden(self):
        """Explizit gesetzter Server wird NICHT ueberschrieben (setdefault)."""
        config = _base_config()
        config["imap_server"] = "custom.mail.local"
        config["imap_port"] = 143
        scanner = OAuth2IMAPScanner(config)
        assert scanner._server == "custom.mail.local"
        assert scanner._port == 143

    def test_ssl_always_true(self):
        """SSL ist Default True fuer Exchange Online."""
        scanner = OAuth2IMAPScanner(_base_config())
        assert scanner._ssl is True

    def test_folders_parsed(self):
        """Kommaseparierte Ordner werden korrekt in Liste aufgeteilt."""
        config = _base_config()
        config["imap_folders"] = "INBOX, Sent Items, Drafts"
        scanner = OAuth2IMAPScanner(config)
        assert scanner._folders == ["INBOX", "Sent Items", "Drafts"]


# ---------------------------------------------------------------------------
# XOAUTH2 Auth-String Format (RFC 7628)
# ---------------------------------------------------------------------------


class TestAuthStringFormat:
    """XOAUTH2-Auth-String ist korrekt formatiert."""

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_auth_string_rfc_format(self, mock_imaplib):
        """Auth-String muss exakt user=...\\x01auth=Bearer ...\\x01\\x01 sein."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn

        scanner = OAuth2IMAPScanner(_base_config())
        with patch.object(scanner, "_acquire_token", return_value="my-token-123"):
            scanner.connect()

        auth_callback = mock_conn.authenticate.call_args[0][1]
        auth_bytes = auth_callback(None)
        auth_str = auth_bytes.decode("utf-8")

        # RFC 7628: user=<user>\x01auth=Bearer <token>\x01\x01
        assert auth_str == "user=user@firma.de\x01auth=Bearer my-token-123\x01\x01"

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_auth_string_with_special_email(self, mock_imaplib):
        """Sonderzeichen in E-Mail-Adresse werden korrekt durchgereicht."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn

        config = _base_config()
        config["imap_username"] = "müller+test@firma.de"
        scanner = OAuth2IMAPScanner(config)
        with patch.object(scanner, "_acquire_token", return_value="tok"):
            scanner.connect()

        auth_callback = mock_conn.authenticate.call_args[0][1]
        auth_str = auth_callback(None).decode("utf-8")
        assert "user=müller+test@firma.de" in auth_str


# ---------------------------------------------------------------------------
# Scan mit Anhaengen (geerbt von IMAPScanner)
# ---------------------------------------------------------------------------


class TestInheritedAttachmentScan:
    """Anhang-Download funktioniert ueber die geerbte Scan-Logik."""

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_scan_with_attachment_inherited(self, mock_imaplib):
        """Mail mit PDF-Anhang -> 2 Dateien (Mail-Text + Anhang)."""
        msg = MIMEMultipart()
        msg["Subject"] = "OAuth Mail mit Anhang"
        msg["From"] = "sender@firma.de"
        msg["To"] = "user@firma.de"
        msg["Date"] = "Mon, 15 Jan 2024 12:00:00 +0100"
        msg.attach(MIMEText("Siehe Anhang.", "plain", "utf-8"))

        attachment = MIMEBase("application", "pdf")
        attachment.set_payload(b"%PDF-1.4 fake content")
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename="bericht.pdf")
        msg.attach(attachment)

        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn
        _setup_uid_conn(mock_conn, msg)

        scanner = OAuth2IMAPScanner(_base_config())
        with patch.object(scanner, "_acquire_token", return_value="tok"):
            with tempfile.TemporaryDirectory() as tmpdir:
                files = scanner.scan(tmpdir)
                assert len(files) == 2
                att_files = [f for f in files if "attachments" in f]
                assert len(att_files) == 1
                assert "bericht" in att_files[0].lower()

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_scan_attachments_disabled(self, mock_imaplib):
        """download_attachments=False -> nur Mail-Text, keine Anhaenge."""
        msg = MIMEMultipart()
        msg["Subject"] = "Mail mit Anhang"
        msg["From"] = "sender@firma.de"
        msg["To"] = "user@firma.de"
        msg["Date"] = "Mon, 15 Jan 2024 12:00:00 +0100"
        msg.attach(MIMEText("Body.", "plain", "utf-8"))

        attachment = MIMEBase("application", "pdf")
        attachment.set_payload(b"%PDF-1.4 data")
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename="report.pdf")
        msg.attach(attachment)

        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn
        _setup_uid_conn(mock_conn, msg)

        config = _base_config()
        config["imap_download_attachments"] = False
        scanner = OAuth2IMAPScanner(config)
        with patch.object(scanner, "_acquire_token", return_value="tok"):
            with tempfile.TemporaryDirectory() as tmpdir:
                files = scanner.scan(tmpdir)
                assert len(files) == 1  # Nur Mail-Text
                assert "attachments" not in files[0]
