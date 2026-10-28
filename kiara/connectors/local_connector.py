"""LocalConnector — Lokales Dateisystem."""

import logging

from kiara.connectors.base import ConnectorBase
from kiara.connectors.registry import register
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)


@register
class LocalConnector(ConnectorBase):
    """Konnektor fuer lokale Dateisystem-Pfade.

    Einfachster Konnektor: Zeigt direkt auf ein lokales Verzeichnis.
    Wenn kein Pfad konfiguriert ist, wird der globale Pfad aus
    kiara.yaml (datasource.path) verwendet.

    Config-Felder:
        path:   Absoluter Verzeichnispfad (leer = globaler Pfad aus kiara.yaml)
    """

    name = ConnectorType.LOCAL
    label = "Lokal"
    badge_class = "default"
    supports_browse = True

    @staticmethod
    def config_schema() -> dict:
        return {
            "path": {
                "type": "str",
                "default": "",
                "label": "Verzeichnispfad",
                "placeholder": "/mnt/knowledgebase",
                "required": False,
                "hint": "Leer = globaler Pfad aus kiara.yaml",
            },
        }

    @staticmethod
    def source_display(config: dict, row: dict) -> str:
        return config.get("path") or row.get("path", "") or ""

    @staticmethod
    def effective_path(config: dict, row: dict) -> str:
        path = config.get("path") or ""
        if path:
            return path
        from kiara.config import load_config

        return load_config()["datasource"]["path"]

    @staticmethod
    def test_connection(config: dict, row: dict) -> dict:
        import os

        path = LocalConnector.effective_path(config, row)
        accessible = os.path.isdir(path)
        if accessible:
            return {"success": True, "message": f"Pfad erreichbar ({path})"}
        return {"success": False, "message": f"Pfad nicht erreichbar: {path}"}

    @staticmethod
    def list_directory(
        config: dict,
        path: str = "",
        *,
        directories_only: bool = False,
    ) -> dict:
        from kiara.config import load_config
        from kiara.connectors.browse import ConnectorBrowseError, list_local_directory

        try:
            base_path = config.get("path") or load_config()["datasource"]["path"]
        except Exception as exc:
            raise ConnectorBrowseError(
                "Lokaler Basispfad ist nicht konfiguriert",
                500,
                "config_error",
            ) from exc
        return list_local_directory(
            base_path,
            path,
            directories_only=directories_only,
            allow_absolute=False,
        )

    @staticmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        import os

        path = LocalConnector.effective_path(config, row)
        ds_id = row.get("id", "?")
        if not path:
            logger.error(
                "Lokaler Pfad fuer ds%s ist leer (weder connector_config.path "
                "noch datasource.path konfiguriert)",
                ds_id,
            )
            return None
        if not os.path.isdir(path):
            logger.error(
                "Lokaler Pfad fuer ds%s nicht erreichbar oder kein Verzeichnis: %r",
                ds_id,
                path,
            )
            return None
        if not os.access(path, os.R_OK):
            logger.error(
                "Lokaler Pfad fuer ds%s nicht lesbar: %r",
                ds_id,
                path,
            )
            return None
        return path
