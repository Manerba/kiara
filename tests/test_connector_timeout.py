"""Tests fuer Connector Async-Blocking + Connect-Timeouts (V9).

Testet die fehlenden kritischen Pfade:
- IMAP connect() TimeoutError-Abfangen + Fehlermeldung
- IMAP/OAuth2 settimeout(None)-Reset nach Login
- OAuth2 _TimeoutSession Default-Timeout
- ERP-Endpoints sind sync (def, nicht async def)
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.ingestion.imap_scanner import IMAPScanner
from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner
from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _imap_config(**overrides):
    """Minimale IMAP-Config."""
    cfg = {
        "imap_server": "mail.firma.de",
        "imap_port": 993,
        "imap_username": "user@firma.de",
        "imap_password": "geheim",
    }
    cfg.update(overrides)
    return cfg


def _oauth2_config(**overrides):
    """Minimale OAuth2/Exchange-Config."""
    cfg = {
        "imap_username": "user@firma.de",
        "imap_folders": "INBOX",
        "imap_since_days": 30,
        "tenant_id": "aaaa-bbbb-cccc",
        "client_id": "1111-2222-3333",
        "client_secret": "secret",
    }
    cfg.update(overrides)
    return cfg


# ===========================================================================
# 1. IMAP Connect-Timeout
# ===========================================================================


class TestIMAPConnectTimeout:
    """TimeoutError bei nicht erreichbarem IMAP-Server."""

    @patch.object(IMAPScanner, "connect", side_effect=TimeoutError)
    def test_imap_connect_timeout_returns_failure(self, mock_connect):
        """connect() wirft TimeoutError -> test_connection gibt success=False."""
        scanner = IMAPScanner(_imap_config())
        result = scanner.test_connection()

        assert result["success"] is False
        assert "Timeout" in result["message"]

    @patch.object(IMAPScanner, "connect", side_effect=TimeoutError)
    def test_imap_timeout_error_message_contains_server_and_port(self, mock_connect):
        """Timeout-Fehlermeldung enthaelt Server-Adresse und Port."""
        scanner = IMAPScanner(
            _imap_config(
                imap_server="mail.test.local",
                imap_port=993,
            )
        )
        result = scanner.test_connection()

        assert result["success"] is False
        assert "mail.test.local" in result["message"]
        assert "993" in result["message"]
        assert str(CONNECTOR_CONNECT_TIMEOUT) in result["message"]

    @patch.object(IMAPScanner, "connect", side_effect=TimeoutError)
    def test_imap_timeout_no_logout_attempted(self, mock_connect):
        """Bei TimeoutError wird kein logout() versucht (conn existiert nicht)."""
        scanner = IMAPScanner(_imap_config())
        # Kein AttributeError -> conn.logout() wird nicht aufgerufen
        result = scanner.test_connection()
        assert result["success"] is False

    @patch.object(IMAPScanner, "connect", side_effect=TimeoutError)
    def test_imap_timeout_custom_port(self, mock_connect):
        """Timeout-Meldung zeigt auch nicht-standard Ports korrekt an."""
        scanner = IMAPScanner(
            _imap_config(
                imap_server="10.0.5.100",
                imap_port=143,
            )
        )
        result = scanner.test_connection()

        assert "10.0.5.100" in result["message"]
        assert "143" in result["message"]


# ===========================================================================
# 2. IMAP settimeout(None)-Reset nach connect()
# ===========================================================================


class TestIMAPConnectResetsTimeout:
    """Nach connect() wird Socket-Timeout auf None (Blocking) zurueckgesetzt."""

    @patch("kiara.ingestion.imap_scanner.imaplib")
    def test_imap_connect_resets_timeout_ssl(self, mock_imaplib):
        """SSL-connect() ruft sock.settimeout(None) nach Login auf."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn

        scanner = IMAPScanner(_imap_config(imap_ssl=True))
        conn = scanner.connect()

        # settimeout(None) muss nach login aufgerufen worden sein
        mock_conn.sock.settimeout.assert_called_with(None)
        # Reihenfolge: login -> settimeout(None)
        mock_conn.login.assert_called_once()

    @patch("kiara.ingestion.imap_scanner.imaplib")
    def test_imap_connect_resets_timeout_starttls(self, mock_imaplib):
        """STARTTLS-connect() ruft sock.settimeout(None) nach Login auf."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4.return_value = mock_conn

        scanner = IMAPScanner(_imap_config(imap_ssl=False))
        conn = scanner.connect()

        mock_conn.starttls.assert_called_once()
        assert "ssl_context" in mock_conn.starttls.call_args.kwargs
        mock_conn.login.assert_called_once()
        mock_conn.sock.settimeout.assert_called_with(None)

    @patch("kiara.ingestion.imap_scanner.imaplib")
    def test_imap_connect_passes_timeout_to_imap4_ssl(self, mock_imaplib):
        """IMAP4_SSL wird mit timeout=CONNECTOR_CONNECT_TIMEOUT aufgerufen."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn

        scanner = IMAPScanner(_imap_config())
        scanner.connect()

        mock_imaplib.IMAP4_SSL.assert_called_once()
        args, kwargs = mock_imaplib.IMAP4_SSL.call_args
        assert args == ("mail.firma.de", 993)
        assert kwargs["timeout"] == CONNECTOR_CONNECT_TIMEOUT
        assert "ssl_context" in kwargs


# ===========================================================================
# 3. OAuth2 settimeout(None)-Reset nach connect()
# ===========================================================================


class TestOAuth2ConnectResetsTimeout:
    """Nach OAuth2-connect() wird Socket-Timeout auf None zurueckgesetzt."""

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_oauth2_connect_resets_timeout(self, mock_imaplib):
        """OAuth2 connect() ruft sock.settimeout(None) nach authenticate auf."""
        mock_conn = MagicMock()
        mock_imaplib.IMAP4_SSL.return_value = mock_conn

        scanner = OAuth2IMAPScanner(_oauth2_config())
        with patch.object(scanner, "_acquire_token", return_value="test-token"):
            conn = scanner.connect()

        mock_conn.authenticate.assert_called_once()
        mock_conn.sock.settimeout.assert_called_with(None)

    @patch("kiara.ingestion.oauth2_imap_scanner.imaplib")
    def test_oauth2_connect_timeout_raises(self, mock_imaplib):
        """OAuth2 connect() mit TimeoutError -> propagiert zum Caller."""
        mock_imaplib.IMAP4_SSL.side_effect = TimeoutError("timed out")

        scanner = OAuth2IMAPScanner(_oauth2_config())
        with patch.object(scanner, "_acquire_token", return_value="test-token"):
            with pytest.raises(TimeoutError):
                scanner.connect()


# ===========================================================================
# 4. OAuth2 _TimeoutSession — Default-Timeout auf MSAL-HTTP-Calls
# ===========================================================================


class TestTimeoutSession:
    """_TimeoutSession erzwingt Default-Timeout auf allen HTTP-Calls."""

    @patch("kiara.ingestion.oauth2_imap_scanner.msal", create=True)
    def test_timeout_session_sets_default_timeout(self, mock_msal_module):
        """_TimeoutSession.request() setzt timeout als Default."""
        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            "access_token": "tok",
        }
        mock_msal_module.ConfidentialClientApplication.return_value = mock_app

        captured_session = {}

        original_init = mock_msal_module.ConfidentialClientApplication

        def capture_init(*args, **kwargs):
            captured_session["http_client"] = kwargs.get("http_client")
            return mock_app

        mock_msal_module.ConfidentialClientApplication.side_effect = capture_init

        with patch.dict("sys.modules", {"msal": mock_msal_module}):
            scanner = OAuth2IMAPScanner(_oauth2_config())
            scanner._acquire_token()

        session = captured_session["http_client"]
        assert session is not None

        # Session ist eine _TimeoutSession (requests.Session-Subclass)
        import requests

        assert isinstance(session, requests.Session)

        # request() setzt timeout als Default
        with patch.object(requests.Session, "request", return_value=MagicMock()) as mock_req:
            session.request("GET", "https://example.com")
            call_kwargs = mock_req.call_args[1]
            assert call_kwargs["timeout"] == CONNECTOR_CONNECT_TIMEOUT

    @patch("kiara.ingestion.oauth2_imap_scanner.msal", create=True)
    def test_timeout_session_does_not_override_explicit_timeout(self, mock_msal_module):
        """Explizit gesetzter Timeout wird nicht ueberschrieben (setdefault)."""
        mock_app = MagicMock()
        mock_app.acquire_token_for_client.return_value = {
            "access_token": "tok",
        }
        mock_msal_module.ConfidentialClientApplication.return_value = mock_app

        captured_session = {}

        def capture_init(*args, **kwargs):
            captured_session["http_client"] = kwargs.get("http_client")
            return mock_app

        mock_msal_module.ConfidentialClientApplication.side_effect = capture_init

        with patch.dict("sys.modules", {"msal": mock_msal_module}):
            scanner = OAuth2IMAPScanner(_oauth2_config())
            scanner._acquire_token()

        session = captured_session["http_client"]

        import requests

        with patch.object(requests.Session, "request", return_value=MagicMock()) as mock_req:
            session.request("GET", "https://example.com", timeout=5)
            call_kwargs = mock_req.call_args[1]
            # Expliziter Timeout 5 bleibt, wird nicht auf 30 ueberschrieben
            assert call_kwargs["timeout"] == 5


# ===========================================================================
# 5. ERP-Endpoints sind sync (def, nicht async def)
# ===========================================================================
