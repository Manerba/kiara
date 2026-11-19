"""PSTConnector — E-Mail-Archiv aus einer Outlook-Datendatei (PST)."""

import logging
import os

from kiara.config import DEFAULT_PATHS
from kiara.connectors.base import ConnectorBase, mail_cache_path
from kiara.connectors.registry import register
from kiara.shared.constants import IMAP_DEFAULT_MAX_ATTACHMENT_MB
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)


def pst_dropzone() -> str:
    """Ablageverzeichnis fuer PST-Dateien (Samba-Freigabe auf dem Kiara-Server)."""
    return DEFAULT_PATHS["pst_dropzone"]


def list_pst_files() -> list[dict]:
    """PST-Dateien im Ablageverzeichnis (nur oberste Ebene), nach Name sortiert."""
    try:
        entries = list(os.scandir(pst_dropzone()))
    except OSError:
        return []
    files = []
    for entry in entries:
        if entry.name.startswith(".") or not entry.name.lower().endswith(".pst"):
            continue
        try:
            if entry.is_file():
                files.append({"name": entry.name, "size": entry.stat().st_size})
        except OSError:
            continue
    return sorted(files, key=lambda f: f["name"].lower())


def resolve_pst_path(config: dict) -> str | None:
    """Absoluter Pfad der gewaehlten PST, oder None bei leerem/unzulaessigem Namen."""
    name = str(config.get("pst_file") or "").strip()
    if not name or "/" in name or "\\" in name or name.startswith(".") or "\x00" in name:
        return None
    if not name.lower().endswith(".pst"):
        return None
    return os.path.join(pst_dropzone(), name)


def _format_size(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.1f} GB".replace(".", ",")
    return f"{size / 1024**2:.1f} MB".replace(".", ",")


@register
class PSTConnector(ConnectorBase):
    """Konnektor fuer Outlook-Datendateien (PST) aus der lokalen PST-Ablage.

    Die PST wird per Samba in die Ablage gelegt und beim Anlegen der Datenquelle
    ausgewaehlt. Mails, Termine und Anhaenge landen wie bei IMAP im Mail-Cache.
    Fehlt die Datei spaeter, bleibt die Datenquelle samt Index erhalten; neue
    Ingestion-Jobs werden abgelehnt (source_unavailable_reason).

    Config-Felder:
        pst_file:               Dateiname in der PST-Ablage
        folders:                "*" oder kommaseparierte Ordnerpfade
        include_calendar:       Termine indexieren (default True)
        download_attachments:   Anhaenge exportieren (default True)
        max_attachment_mb:      Max. Anhanggroesse in MB (default 25)
    """

    name = ConnectorType.PST
    label = "E-Mail-Archiv (PST)"
    badge_label = "PST"
    badge_class = "warning"
    HASH_MODE = "content"  # Cache-Dateien werden bei jedem Export neu geschrieben

    @staticmethod
    def config_schema() -> dict:
        files = list_pst_files()
        options = [{"value": "", "label": "— PST-Datei wählen —"}] + [
            {"value": f["name"], "label": f"{f['name']} ({_format_size(f['size'])})"} for f in files
        ]
        return {
            "pst_file": {
                "type": "str",
                "default": "",
                "label": "PST-Datei",
                "required": True,
                "options": options,
                "hint": f"Ablage: {pst_dropzone()} (Samba-Freigabe „pst“ auf diesem Server)",
            },
            "folders": {
                "type": "str",
                "default": "*",
                "label": "Ordner",
                "placeholder": "Posteingang, Kalender",
                "hint": "Kommasepariert, oder über 'Ordner laden' auswählen",
                "folder_tree": True,
            },
            "include_calendar": {
                "type": "bool",
                "default": True,
                "label": "Kalendereinträge indexieren",
            },
            "download_attachments": {
                "type": "bool",
                "default": True,
                "label": "Anhänge exportieren",
            },
            "max_attachment_mb": {
                "type": "int",
                "default": IMAP_DEFAULT_MAX_ATTACHMENT_MB,
                "label": "Max. Anhanggröße (MB)",
            },
        }

    @staticmethod
    def source_display(config: dict, row: dict) -> str:
        return str(config.get("pst_file") or "")

    @staticmethod
    def effective_path(config: dict, row: dict) -> str:
        return mail_cache_path(row["id"])

    @staticmethod
    def source_unavailable_reason(config: dict, row: dict) -> str | None:
        path = resolve_pst_path(config)
        if path is None:
            return "Keine PST-Datei ausgewählt"
        if not os.path.isfile(path):
            return f"PST-Datei {os.path.basename(path)} ist nicht mehr in der Ablage"
        if not os.access(path, os.R_OK):
            return f"PST-Datei {os.path.basename(path)} ist nicht lesbar"
        return None

    @staticmethod
    def _build_scanner_config(config: dict) -> dict:
        return {
            "pst_path": resolve_pst_path(config),
            "folders": config.get("folders", "*"),
            "include_calendar": config.get("include_calendar", True),
            "download_attachments": config.get("download_attachments", True),
            "max_attachment_mb": config.get("max_attachment_mb", IMAP_DEFAULT_MAX_ATTACHMENT_MB),
        }

    @staticmethod
    def test_connection(config: dict, row: dict) -> dict:
        from kiara.ingestion.pst_reader import PstFile

        reason = PSTConnector.source_unavailable_reason(config, row)
        if reason:
            return {"success": False, "message": reason}
        with PstFile(resolve_pst_path(config)) as pst:
            folders = list(pst.iter_folders())
            count = pst.count_messages(folders)
        return {
            "success": True,
            "message": f"PST lesbar — {len(folders)} Ordner, {count} Elemente",
        }

    @staticmethod
    def list_folders(config: dict) -> list[dict]:
        """Ordnerbaum der PST fuer den Folder-Picker."""
        from kiara.ingestion.pst_reader import PstFile

        reason = PSTConnector.source_unavailable_reason(config, {})
        if reason:
            raise RuntimeError(reason)
        with PstFile(resolve_pst_path(config)) as pst:
            return pst.folder_tree()

    @staticmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        from kiara.config import get_mail_archive_config
        from kiara.ingestion.pst_scanner import PSTScanner

        reason = PSTConnector.source_unavailable_reason(config, row)
        if reason:
            logger.error("PST-Datenquelle ds%s: %s", row["id"], reason)
            return None
        path = mail_cache_path(row["id"])
        os.makedirs(path, exist_ok=True)
        scanner_config = PSTConnector._build_scanner_config(config)
        scanner_config["mail_archive"] = get_mail_archive_config(ds_config)
        try:
            PSTScanner(scanner_config).scan(
                path, progress_callback=progress_callback, control_callback=control_callback
            )
        except InterruptedError:
            raise
        except Exception as exc:
            logger.error("PST scan failed for ds%s: %s", row["id"], exc)
            raise
        return path
