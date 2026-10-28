"""SMBConnector — SMB/CIFS-Freigaben."""

import logging
import os
import posixpath

from kiara.config import DEFAULT_TIMEOUTS
from kiara.connectors.base import ConnectorBase, mount_and_return
from kiara.connectors.registry import auto_mount_point, register
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)


def _normalize_sub_path(sub_path: str | None) -> str:
    raw = str(sub_path or "").replace("\\", "/").strip()
    if not raw:
        return ""
    norm = posixpath.normpath(raw.strip("/"))
    if norm in ("", "."):
        return ""
    if norm == ".." or norm.startswith("../"):
        raise ValueError("SMB sub_path verlaesst den Mountpoint")
    return norm


def _effective_mount_sub_path(mount_point: str, sub_path: str | None) -> str:
    safe_sub_path = _normalize_sub_path(sub_path)
    if not safe_sub_path:
        return mount_point
    real_mount = os.path.realpath(mount_point)
    target = os.path.realpath(os.path.join(mount_point, safe_sub_path))
    if not (target == real_mount or target.startswith(real_mount + os.sep)):
        raise ValueError("SMB sub_path verlaesst den Mountpoint")
    return target


@register
class SMBConnector(ConnectorBase):
    """Konnektor fuer SMB/CIFS-Netzwerkfreigaben.

    Mountet eine SMB-Freigabe automatisch unter einem generierten
    Mount-Punkt und stellt den Pfad fuer die Ingestion-Pipeline bereit.
    Verbindungstests erfolgen via smbclient, Mounts via mount_smb().

    Config-Felder:
        server:     SMB-Server (IP oder Hostname)
        share:      Name der Freigabe
        username:   Benutzername (optional fuer Gaeste)
        password:   Passwort (verschluesselt gespeichert)
        domain:     Windows-Domaene (default: leer)
        sub_path:   Unterverzeichnis innerhalb der Freigabe (optional,
                    mit Browse-Button fuer Ordnerauswahl)
    """

    name = ConnectorType.SMB
    label = "SMB-Freigabe"
    badge_class = "info"
    supports_browse = True

    @staticmethod
    def config_schema() -> dict:
        return {
            "server": {
                "type": "str",
                "default": "",
                "label": "Server",
                "placeholder": "fileserver.example.com",
                "required": True,
            },
            "share": {
                "type": "str",
                "default": "",
                "label": "Freigabe",
                "placeholder": "Documents",
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
            "domain": {
                "type": "str",
                "default": "",
                "label": "Domaene",
                "placeholder": "WORKGROUP",
            },
            "sub_path": {
                "type": "str",
                "default": "",
                "label": "Unterverzeichnis",
                "placeholder": "Abteilung/Technik",
                "browse": {
                    "endpoint": "/api/admin/datasources/browse-share",
                    "depends_on": ["server", "share", "username", "password", "domain"],
                },
            },
        }

    @staticmethod
    def source_display(config: dict, row: dict) -> str:
        server = config.get("server", "")
        share = config.get("share", "")
        try:
            sub_path = _normalize_sub_path(config.get("sub_path"))
        except ValueError:
            sub_path = ""
        if server:
            display = f"{server}:/{share}"
            if sub_path:
                display += f"/{sub_path}"
            return display
        return ""

    @staticmethod
    def effective_path(config: dict, row: dict) -> str:
        mount_point = auto_mount_point(row)
        return _effective_mount_sub_path(mount_point, config.get("sub_path"))

    @staticmethod
    def test_connection(config: dict, row: dict) -> dict:
        import subprocess
        import tempfile

        server = config.get("server", "")
        share = config.get("share", "")
        username = config.get("username", "")
        password = config.get("password", "")
        domain = config.get("domain", "")

        # CR/LF in Credentials wuerden zusaetzliche Direktiven in die Auth-Datei
        # einschleusen — analog zu mount_smb() ablehnen.
        for _label, _val in (("username", username), ("password", password), ("domain", domain)):
            if "\n" in _val or "\r" in _val:
                return {"success": False, "message": f"Ungueltiges Zeichen (CR/LF) in {_label}"}

        # Credentials-File statt Kommandozeile (Passwort nicht in ps aux sichtbar)
        cred_lines = f"username={username}\npassword={password}\n"
        if domain:
            cred_lines += f"domain={domain}\n"
        fd, cred_path = tempfile.mkstemp(prefix="kiara-smb-test-", suffix=".cred")
        try:
            os.write(fd, cred_lines.encode())
            os.close(fd)
            fd = None
            os.chmod(cred_path, 0o600)
            result = subprocess.run(
                [
                    "smbclient",
                    f"//{server}/{share}",
                    f"--authentication-file={cred_path}",
                    "-c",
                    "ls",
                ],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUTS["subprocess_umount"],
            )
            if result.returncode == 0:
                return {"success": True, "message": "SMB-Verbindung erfolgreich"}
            return {
                "success": False,
                "message": result.stderr.strip() or "Verbindung fehlgeschlagen",
            }
        except FileNotFoundError:
            return {"success": False, "message": "smbclient nicht installiert"}
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Timeout"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(cred_path)
            except OSError:
                pass

    @staticmethod
    def list_directory(
        config: dict,
        path: str = "",
        *,
        directories_only: bool = False,
    ) -> dict:
        from kiara.connectors.browse import browse_smbclient

        return browse_smbclient(
            dict(config),
            path,
            directories_only=directories_only,
        )

    @staticmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        from kiara.shared.mount import mount_smb

        mount_point = auto_mount_point(row)
        result = mount_and_return(
            mount_smb,
            {
                "server": config.get("server", ""),
                "share": config.get("share", ""),
                "username": config.get("username", ""),
                "password": config.get("password", ""),
                "domain": config.get("domain", ""),
            },
            mount_point,
            row,
            "SMB",
            expected_source=f"//{config.get('server', '')}/{config.get('share', '')}",
        )
        if result is None:
            return None
        try:
            return _effective_mount_sub_path(mount_point, config.get("sub_path"))
        except ValueError as exc:
            logger.error("SMB scan blocked for ds%s: %s", row.get("id"), exc)
            return None
