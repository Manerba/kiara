"""WebDAVConnector — WebDAV-Freigaben."""

import logging

from kiara.config import DEFAULT_TIMEOUTS
from kiara.connectors.base import ConnectorBase, mount_and_return
from kiara.connectors.registry import auto_mount_point, register
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)


@register
class WebDAVConnector(ConnectorBase):
    """Konnektor fuer WebDAV-Freigaben (z.B. Nextcloud, ownCloud).

    Mountet eine WebDAV-URL via davfs2 unter einem generierten Mount-Punkt.
    Verbindungstests erfolgen via HTTP PROPFIND-Request.

    Config-Felder:
        url:        WebDAV-Basis-URL (z.B. https://cloud.firma.de/remote.php/dav/files/user/)
        username:   Benutzername (optional)
        password:   Passwort (verschluesselt gespeichert)
        base_path:  Unterverzeichnis innerhalb der WebDAV-Freigabe (default "/")
    """

    name = ConnectorType.WEBDAV
    label = "WebDAV"
    badge_class = "info"
    supports_browse = True

    @staticmethod
    def config_schema() -> dict:
        return {
            "url": {
                "type": "str",
                "default": "",
                "label": "WebDAV-URL",
                "placeholder": "https://cloud.firma.de/remote.php/dav/files/user/",
                "required": True,
            },
            "username": {
                "type": "str",
                "default": "",
                "label": "Benutzername",
            },
            "password": {
                "type": "str",
                "default": "",
                "label": "Passwort",
                "secret": True,
            },
            "base_path": {
                "type": "str",
                "default": "/",
                "label": "Basispfad",
                "placeholder": "/",
                "browse": {
                    "endpoint": "/api/admin/datasources/browse-webdav",
                    "depends_on": ["url", "username", "password"],
                },
            },
        }

    @staticmethod
    def source_display(config: dict, row: dict) -> str:
        from urllib.parse import urlparse

        url = config.get("url", "")
        base = config.get("base_path", "/")
        if url:
            parsed = urlparse(url)
            return f"{parsed.hostname}:{base}"
        return ""

    @staticmethod
    def effective_path(config: dict, row: dict) -> str:
        return auto_mount_point(row)

    @staticmethod
    def test_connection(config: dict, row: dict) -> dict:
        try:
            import httpx

            url = config.get("url", "")
            username = config.get("username", "")
            password = config.get("password", "")
            auth = (username, password) if username else None
            with httpx.Client(timeout=DEFAULT_TIMEOUTS["http_api"]) as client:
                resp = client.request("PROPFIND", url, auth=auth, headers={"Depth": "0"})
                if resp.status_code in (207, 200, 204):
                    return {"success": True, "message": "WebDAV-Verbindung erfolgreich"}
                if resp.status_code in (401, 403):
                    return {"success": False, "message": "WebDAV-Authentifizierung fehlgeschlagen"}
                return {"success": False, "message": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    @staticmethod
    def list_directory(
        config: dict,
        path: str = "",
        *,
        directories_only: bool = False,
    ) -> dict:
        from kiara.connectors.browse import browse_webdav_propfind

        return browse_webdav_propfind(
            dict(config),
            path,
            directories_only=directories_only,
        )

    @staticmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        from kiara.shared.mount import mount_webdav

        path = auto_mount_point(row)
        url = (
            (config.get("url") or "").rstrip("/")
            + "/"
            + (config.get("base_path") or "/").lstrip("/")
        )
        return mount_and_return(
            mount_webdav,
            {
                "url": url,
                "username": config.get("username", ""),
                "password": config.get("password", ""),
            },
            path,
            row,
            "WebDAV",
            expected_source=url,
        )
