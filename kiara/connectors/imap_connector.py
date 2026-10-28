"""IMAPConnector — E-Mail via IMAP."""

import logging

from kiara.connectors.base import (
    ConnectorBase,
    build_mail_scanner_base,
    mail_cache_path,
    mail_list_folders,
    mail_prepare_and_scan,
    mail_reprocess_cached,
)
from kiara.connectors.registry import register
from kiara.shared.constants import (
    IMAP_DEFAULT_MAX_ATTACHMENT_MB,
    IMAP_DEFAULT_SINCE_DAYS,
    IMAP_PORT_SSL,
    SCAN_CACHE_MAX_AGE_SECS,
)
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)


@register
class IMAPConnector(ConnectorBase):
    """Konnektor fuer E-Mail-Postfaecher via IMAP.

    Verbindet sich mit einem IMAP-Server, liest E-Mails aus konfigurierten
    Ordnern und laedt optional Anhaenge herunter. Die Mails werden als
    Dateien im lokalen Mail-Cache abgelegt fuer die Ingestion-Pipeline.

    Config-Felder:
        server:                 IMAP-Server-Hostname
        port:                   Port (default 993)
        ssl:                    SSL verwenden (default True)
        username/password:      Zugangsdaten
        folders:                Kommaseparierte IMAP-Ordner (default "INBOX")
        since_days:             Zeitraum in Tagen (default 90)
        download_attachments:   Anhaenge herunterladen (default True)
        max_attachment_mb:      Max. Anhanggroesse in MB (default 25)
    """

    name = ConnectorType.IMAP
    label = "E-Mail (IMAP)"
    badge_class = "warning"
    HASH_MODE = "content"  # Mails werden bei Resync neu geschrieben
    SUPPORTS_PARALLEL_SCAN = True

    @staticmethod
    def config_schema() -> dict:
        return {
            "server": {
                "type": "str",
                "default": "",
                "label": "IMAP-Server",
                "placeholder": "mail.firma.de",
                "required": True,
            },
            "port": {
                "type": "int",
                "default": IMAP_PORT_SSL,
                "label": "Port",
            },
            "ssl": {
                "type": "bool",
                "default": True,
                "label": "SSL verwenden",
            },
            "username": {
                "type": "str",
                "default": "",
                "label": "Benutzername",
                "placeholder": "benutzer@firma.de",
            },
            "password": {
                "type": "str",
                "default": "",
                "label": "Passwort",
                "secret": True,
            },
            "folders": {
                "type": "str",
                "default": "INBOX",
                "label": "Ordner",
                "placeholder": "INBOX, Sent",
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
        server = config.get("server", "")
        user = config.get("username", "")
        if server:
            return f"{user}@{server}" if user else server
        return ""

    @staticmethod
    def effective_path(config: dict, row: dict) -> str:
        return mail_cache_path(row["id"])

    @staticmethod
    def _build_scanner_config(config: dict) -> dict:
        """Baut das Scanner-Config-Dict aus dem Connector-Config."""
        cfg = build_mail_scanner_base(config)
        cfg.update(
            {
                "imap_server": config.get("server", ""),
                "imap_port": config.get("port", IMAP_PORT_SSL),
                "imap_ssl": config.get("ssl", True),
                "imap_password": config.get("password", ""),
            }
        )
        return cfg

    @staticmethod
    def test_connection(config: dict, row: dict) -> dict:
        from kiara.ingestion.imap_scanner import IMAPScanner

        scanner = IMAPScanner(IMAPConnector._build_scanner_config(config))
        result = scanner.test_connection()
        if result["success"]:
            folders = result.get("folders", [])
            return {
                "success": True,
                "message": f"Verbindung erfolgreich — {len(folders)} Ordner gefunden",
            }
        return result

    @staticmethod
    def list_folders(config: dict) -> list[dict]:
        """Gibt hierarchische Ordnerliste vom IMAP-Server zurueck."""
        from kiara.ingestion.imap_scanner import IMAPScanner

        return mail_list_folders(IMAPConnector, IMAPScanner, "IMAP-Server", config)

    SCAN_CACHE_MAX_AGE = SCAN_CACHE_MAX_AGE_SECS

    @staticmethod
    def reprocess_cached(
        config: dict,
        row: dict,
        ds_config: dict,
        *,
        cache_paths: list[str] | None = None,
        progress_callback=None,
        control_callback=None,
        html_options: dict | None = None,
    ) -> dict:
        from kiara.ingestion.imap_scanner import IMAPScanner

        return mail_reprocess_cached(
            IMAPConnector,
            IMAPScanner,
            config,
            row,
            ds_config,
            cache_paths=cache_paths,
            progress_callback=progress_callback,
            control_callback=control_callback,
            html_options=html_options,
        )

    @staticmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        from kiara.ingestion.imap_scanner import IMAPScanner

        return mail_prepare_and_scan(
            IMAPConnector,
            IMAPScanner,
            "IMAP",
            config,
            row,
            progress_callback=progress_callback,
            control_callback=control_callback,
            runtime_config=ds_config,
        )
