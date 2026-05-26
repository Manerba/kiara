"""ExchangeOnlineConnector — E-Mail via Exchange Online (OAuth2)."""

import logging

from kiara.connectors.base import (
    ConnectorBase,
    build_mail_scanner_base,
    mail_cache_path,
    mail_list_folders,
    mail_prepare_and_scan,
)
from kiara.connectors.registry import register
from kiara.shared.constants import (
    IMAP_DEFAULT_MAX_ATTACHMENT_MB,
    IMAP_DEFAULT_SINCE_DAYS,
    SCAN_CACHE_MAX_AGE_SECS,
)
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)


@register
class ExchangeOnlineConnector(ConnectorBase):
    """Konnektor fuer Microsoft 365 / Exchange Online Postfaecher.

    Verwendet OAuth2 Client-Credentials-Flow ueber Entra ID (Azure AD).
    Authentifiziert sich per XOAUTH2-SASL am IMAP-Endpoint von Microsoft 365
    (outlook.office365.com:993).

    Voraussetzungen:
        1. App-Registrierung in Entra ID (Azure Portal)
        2. API-Permission: Office 365 Exchange Online → IMAP.AccessAsApp
        3. Admin-Consent erteilt
        4. Service-Principal zum Postfach berechtigt
           (Exchange Online PowerShell)
    """

    name = ConnectorType.EXCHANGE_ONLINE
    label = "Exchange Online (OAuth2)"
    badge_class = "info"
    HASH_MODE = "content"  # Mails werden bei Resync neu geschrieben
    SUPPORTS_PARALLEL_SCAN = True

    @staticmethod
    def config_schema() -> dict:
        return {
            "tenant_id": {
                "type": "str",
                "default": "",
                "label": "Tenant-ID",
                "placeholder": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "required": True,
                "hint": "Entra ID (Azure AD) Tenant-ID / Verzeichnis-ID",
            },
            "client_id": {
                "type": "str",
                "default": "",
                "label": "Client-ID (App)",
                "placeholder": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "required": True,
                "hint": "Anwendungs-ID der App-Registrierung",
            },
            "client_secret": {
                "type": "str",
                "default": "",
                "label": "Client-Secret",
                "secret": True,
                "required": True,
            },
            "username": {
                "type": "str",
                "default": "",
                "label": "Postfach (E-Mail)",
                "placeholder": "benutzer@firma.de",
                "required": True,
                "hint": "E-Mail-Adresse des Postfachs, auf das zugegriffen wird",
            },
            "folders": {
                "type": "str",
                "default": "INBOX",
                "label": "Ordner",
                "placeholder": "INBOX, Sent Items",
                "hint": "Kommasepariert, oder ueber 'Ordner laden' auswaehlen",
                "folder_tree": True,
            },
            "since_days": {
                "type": "int",
                "default": IMAP_DEFAULT_SINCE_DAYS,
                "label": "Zeitraum (Tage)",
            },
            "download_attachments": {
                "type": "bool",
                "default": True,
                "label": "Anhaenge herunterladen",
            },
            "max_attachment_mb": {
                "type": "int",
                "default": IMAP_DEFAULT_MAX_ATTACHMENT_MB,
                "label": "Max. Anhanggroesse (MB)",
            },
        }

    @staticmethod
    def source_display(config: dict, row: dict) -> str:
        user = config.get("username", "")
        return user if user else "Exchange Online"

    @staticmethod
    def effective_path(config: dict, row: dict) -> str:
        return mail_cache_path(row["id"])

    @staticmethod
    def _build_scanner_config(config: dict) -> dict:
        """Baut das Scanner-Config-Dict fuer OAuth2IMAPScanner."""
        cfg = build_mail_scanner_base(config)
        cfg.update(
            {
                "tenant_id": config.get("tenant_id", ""),
                "client_id": config.get("client_id", ""),
                "client_secret": config.get("client_secret", ""),
            }
        )
        return cfg

    @staticmethod
    def test_connection(config: dict, row: dict) -> dict:
        from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner

        scanner = OAuth2IMAPScanner(ExchangeOnlineConnector._build_scanner_config(config))
        result = scanner.test_connection()
        if result["success"]:
            folders = result.get("folders", [])
            return {
                "success": True,
                "message": f"OAuth2-Verbindung erfolgreich — {len(folders)} Ordner gefunden",
            }
        return result

    @staticmethod
    def list_folders(config: dict) -> list[dict]:
        """Gibt hierarchische Ordnerliste vom Exchange-Server zurueck."""
        from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner

        return mail_list_folders(
            ExchangeOnlineConnector,
            OAuth2IMAPScanner,
            "Exchange-Server",
            config,
        )

    SCAN_CACHE_MAX_AGE = SCAN_CACHE_MAX_AGE_SECS

    @staticmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        from kiara.ingestion.oauth2_imap_scanner import OAuth2IMAPScanner

        return mail_prepare_and_scan(
            ExchangeOnlineConnector,
            OAuth2IMAPScanner,
            "Exchange",
            config,
            row,
            progress_callback=progress_callback,
            control_callback=control_callback,
        )
