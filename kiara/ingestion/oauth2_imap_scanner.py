"""OAuth2-IMAP-Scanner fuer Exchange Online / Microsoft 365.

Erbt die gesamte Mail-Scan-Logik (Ordner, Parsing, Anhaenge) vom
IMAPScanner und ueberschreibt nur connect() fuer XOAUTH2-SASL-Auth.

Token-Abruf ueber MSAL (Microsoft Authentication Library) mit dem
Client-Credentials-Flow (App-only, kein User-Login noetig).
"""

import imaplib
import logging
import time

from kiara.ingestion.imap_scanner import IMAPScanner
from kiara.shared.constants import IMAP_PORT_SSL

logger = logging.getLogger(__name__)

# Microsoft IMAP-Scope fuer Client-Credentials-Flow
_EXCHANGE_IMAP_SCOPE = "https://outlook.office365.com/.default"
_EXCHANGE_IMAP_SERVER = "outlook.office365.com"
_EXCHANGE_IMAP_PORT = IMAP_PORT_SSL
_TOKEN_REFRESH_SKEW_SECONDS = 300


class OAuth2IMAPScanner(IMAPScanner):
    """IMAP-Scanner mit OAuth2-Authentifizierung fuer Exchange Online.

    Verwendet den MSAL Client-Credentials-Flow um ein Access-Token
    von Entra ID (Azure AD) zu holen und authentifiziert sich dann
    per XOAUTH2-SASL-Mechanismus am IMAP-Endpoint von Microsoft 365.

    Voraussetzungen in Entra ID:
        - App-Registrierung mit Permission IMAP.AccessAsApp
        - Admin-Consent erteilt
        - Service-Principal der App zum Postfach berechtigt
          (via Exchange Online PowerShell: New-ServicePrincipal / Add-MailboxPermission)
    """

    def __init__(self, config: dict):
        # Server/Port fest auf Exchange Online setzen
        config.setdefault("imap_server", _EXCHANGE_IMAP_SERVER)
        config.setdefault("imap_port", _EXCHANGE_IMAP_PORT)
        config.setdefault("imap_ssl", True)
        super().__init__(config)

        self._tenant_id = config.get("tenant_id", "")
        self._client_id = config.get("client_id", "")
        self._client_secret = config.get("client_secret", "")
        self._token_expires_at = 0.0
        self._cached_token: str | None = None
        self._token_refresh_skew_seconds = int(
            config.get("oauth2_token_refresh_skew_seconds", _TOKEN_REFRESH_SKEW_SECONDS)
        )

    def _acquire_token(self) -> str:
        """Holt ein Access-Token ueber den MSAL Client-Credentials-Flow.

        Returns:
            Access-Token-String fuer XOAUTH2.

        Raises:
            RuntimeError: Token-Abruf fehlgeschlagen.
        """
        # Cache-Hit: gueltiges Token wiederverwenden statt API-Quota zu belasten
        # (jeder connect() ruft _acquire_token; bei Reconnects waeren das viele Calls).
        if self._cached_token and not self._token_expires_soon():
            return self._cached_token

        try:
            import msal
        except ImportError:
            raise RuntimeError(
                "Exchange Online erfordert die 'msal'-Bibliothek. Installation: pip install msal"
            )

        import requests

        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        authority = f"https://login.microsoftonline.com/{self._tenant_id}"

        # requests.Session hat keinen Default-Timeout. Subclass erzwingt ihn
        # auf allen HTTP-Calls, die MSAL intern ausfuehrt (Token-Endpoint,
        # OpenID-Discovery, JWKS-Abruf).
        class _TimeoutSession(requests.Session):
            def request(self, *args, **kwargs):
                kwargs.setdefault("timeout", CONNECTOR_CONNECT_TIMEOUT)
                return super().request(*args, **kwargs)

        app = msal.ConfidentialClientApplication(
            client_id=self._client_id,
            client_credential=self._client_secret,
            authority=authority,
            http_client=_TimeoutSession(),
        )

        result = app.acquire_token_for_client(scopes=[_EXCHANGE_IMAP_SCOPE])

        if "access_token" not in result:
            from kiara.shared.secret_redact import redact_secrets_in_text

            error = result.get("error", "unknown")
            description = redact_secrets_in_text(result.get("error_description", "") or "")
            raise RuntimeError(f"OAuth2-Token-Abruf fehlgeschlagen: {error} — {description}")

        try:
            expires_in = int(result.get("expires_in", 3600))
        except (TypeError, ValueError):
            expires_in = 3600
        self._token_expires_at = time.time() + max(0, expires_in)
        self._cached_token = result["access_token"]

        logger.debug("OAuth2-Token erfolgreich abgerufen fuer Tenant %s", self._tenant_id)
        return self._cached_token

    @staticmethod
    def _is_auth_token_error(exc: Exception) -> bool:
        """Erkennt IMAP-Antworten, die auf abgelaufenes/ungueltiges OAuth hindeuten."""
        msg = str(exc).lower()
        markers = (
            "401",
            "authenticationfailed",
            "authenticate failed",
            "auth failed",
            "invalid credentials",
            "invalid token",
            "expired token",
            "token expired",
            "oauth",
            "xoauth",
            "bad",
            " no ",
        )
        return isinstance(exc, imaplib.IMAP4.error) and any(m in msg for m in markers)

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """OAuth2-Auth-Ablehnungen muessen einen Reconnect mit frischem Token triggern."""
        return IMAPScanner._is_connection_error(exc) or OAuth2IMAPScanner._is_auth_token_error(exc)

    def _token_expires_soon(self) -> bool:
        if self._token_expires_at <= 0:
            return False
        return time.time() >= (self._token_expires_at - self._token_refresh_skew_seconds)

    def _ensure_connection_ready(self, conn, folder: str | None = None):
        """Reconnectet proaktiv, bevor ein OAuth2-Token waehrend des Scans ablaeuft."""
        if not self._token_expires_soon():
            return conn
        logger.info("OAuth2-Token laeuft bald ab, erneuere IMAP-Verbindung.")
        return self._reconnect(conn, folder=folder)

    def _reconnect(self, conn, folder: str | None = None):
        # Server-seitig abgelehnter Token darf nicht erneut benutzt werden.
        self._cached_token = None
        self._token_expires_at = 0.0
        return super()._reconnect(conn, folder=folder)

    def connect(self) -> imaplib.IMAP4_SSL:
        """Verbindung zu Exchange Online per XOAUTH2-SASL herstellen."""
        from kiara.shared.constants import CONNECTOR_CONNECT_TIMEOUT

        token = self._acquire_token()

        conn = imaplib.IMAP4_SSL(self._server, self._port, timeout=CONNECTOR_CONNECT_TIMEOUT)

        try:
            # XOAUTH2-Auth-String gemaess RFC 7628
            auth_string = f"user={self._username}\x01auth=Bearer {token}\x01\x01"
            conn.authenticate("XOAUTH2", lambda _: auth_string.encode())

            logger.info("XOAUTH2-Login erfolgreich: %s", self._username)
            # Timeout nach Verbindungsaufbau zuruecksetzen (siehe IMAPScanner.connect())
            conn.sock.settimeout(None)
            return conn
        except Exception:
            # Socket-Leak vermeiden, falls authenticate/Setup fehlschlaegt
            try:
                conn.logout()
            except Exception:
                try:
                    conn.shutdown()
                except Exception:
                    pass
            raise
