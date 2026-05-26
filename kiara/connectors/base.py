"""ConnectorBase — Abstrakte Basisklasse fuer alle KIara-Konnektoren."""

import logging
import os
import socket
import time
from abc import ABC, abstractmethod
from typing import Any

from kiara.config import DEFAULT_PATHS
from kiara.shared.constants import (
    CONNECTOR_CONNECT_TIMEOUT,
    IMAP_DEFAULT_MAX_ATTACHMENT_MB,
    IMAP_DEFAULT_SINCE_DAYS,
)

logger = logging.getLogger(__name__)


def mail_cache_path(datasource_id: int) -> str:
    """Berechnet den Mail-Cache-Pfad fuer eine Datenquelle.

    Wird von IMAP- und Exchange-Connectoren fuer effective_path()
    und prepare_and_scan() verwendet.

    Args:
        datasource_id: ID der Datenquelle.

    Returns:
        Absoluter Pfad zum Mail-Cache-Verzeichnis (z.B. /var/lib/kiara/mail_cache/ds5).
    """
    return os.path.join(DEFAULT_PATHS["mail_cache"], f"ds{datasource_id}")


def raise_connection_timeout(server_type: str) -> None:
    """Wirft einen RuntimeError fuer fehlgeschlagene Verbindungsaufbauten.

    Standardisierte Fehlermeldung fuer Timeout bei Connector-Verbindungstests.

    Args:
        server_type: Typ des Servers (z.B. "IMAP-Server", "Exchange-Server").

    Raises:
        RuntimeError: Immer.
    """
    raise RuntimeError(
        f"Verbindungsaufbau fehlgeschlagen — {server_type} nicht erreichbar "
        f"(Timeout nach {CONNECTOR_CONNECT_TIMEOUT}s)"
    )


def build_mail_scanner_base(config: dict) -> dict:
    """Baut die gemeinsamen Scanner-Config-Felder fuer Mail-Konnektoren.

    Wird von IMAP- und Exchange-Connector verwendet. Jeder Konnektor
    ergaenzt das Ergebnis um seine protokollspezifischen Felder
    (z.B. server/port/ssl fuer IMAP, tenant_id/client_id/client_secret
    fuer Exchange).

    Args:
        config: Der connector_config-Dict (entschluesselt).

    Returns:
        Dict mit den 5 gemeinsamen imap_*-Feldern.
    """
    return {
        "imap_username": config.get("username", ""),
        "imap_folders": config.get("folders", "INBOX"),
        "imap_since_days": config.get("since_days", IMAP_DEFAULT_SINCE_DAYS),
        "imap_download_attachments": config.get("download_attachments", True),
        "imap_max_attachment_mb": config.get("max_attachment_mb", IMAP_DEFAULT_MAX_ATTACHMENT_MB),
    }


def mail_prepare_and_scan(
    connector_cls: type["ConnectorBase"],
    scanner_cls: type,
    label: str,
    config: dict,
    row: dict,
    progress_callback=None,
    control_callback=None,
) -> str | None:
    """Gemeinsame prepare_and_scan-Logik fuer Mail-Konnektoren (IMAP + Exchange).

    Erstellt den Cache-Pfad, prueft den Scan-Cache, baut die Scanner-Config,
    fuehrt den Scan durch und gibt den Pfad zurueck.

    Args:
        connector_cls: Die Konnektor-Klasse (fuer _check_scan_cache und
            _build_scanner_config).
        scanner_cls: Die Scanner-Klasse (z.B. IMAPScanner, OAuth2IMAPScanner).
        label: Label fuer Log-/Cache-Meldungen (z.B. "IMAP", "Exchange").
        config: Der connector_config-Dict (entschluesselt).
        row: Die vollstaendige DB-Row.
        progress_callback: Optionaler Fortschritts-Callback.
        control_callback: Optionaler Steuerungs-Callback.

    Returns:
        Absoluter Pfad zum Mail-Cache-Verzeichnis, oder None bei Fehler.
    """
    path = mail_cache_path(row["id"])
    os.makedirs(path, exist_ok=True)

    if connector_cls._check_scan_cache(path, label):
        return path

    scanner = scanner_cls(connector_cls._build_scanner_config(config))
    try:
        scanner.scan(path, progress_callback=progress_callback, control_callback=control_callback)
        return path
    except InterruptedError:
        # Cancel-Signal — kein Fehler-Log, an Worker durchreichen.
        raise
    except Exception as exc:
        logger.error("%s scan failed for ds%s: %s", label, row["id"], exc)
        raise


def mail_list_folders(
    connector_cls: type["ConnectorBase"],
    scanner_cls: type,
    server_label: str,
    config: dict,
) -> list[dict]:
    """Gemeinsame list_folders()-Logik fuer Mail-Konnektoren (IMAP + Exchange).

    Baut Scanner-Config, verbindet zum Server, listet Ordner auf und gibt
    eine hierarchische Ordnerliste zurueck.

    Args:
        connector_cls: Die Konnektor-Klasse (fuer _build_scanner_config).
        scanner_cls: Die Scanner-Klasse (z.B. IMAPScanner, OAuth2IMAPScanner).
        server_label: Label fuer Fehlermeldung (z.B. "IMAP-Server", "Exchange-Server").
        config: Der connector_config-Dict (entschluesselt).

    Returns:
        Hierarchische Ordnerliste als list[dict].

    Raises:
        RuntimeError: Wenn der Verbindungsaufbau fehlschlaegt (Timeout).
    """
    from kiara.ingestion.imap_scanner import IMAPScanner

    scanner = scanner_cls(connector_cls._build_scanner_config(config))
    try:
        conn = scanner.connect()
    except (TimeoutError, socket.timeout, socket.gaierror):
        raise_connection_timeout(server_label)
    try:
        conn.sock.settimeout(CONNECTOR_CONNECT_TIMEOUT)
        _status, raw_folders = conn.list()
        parsed = IMAPScanner._parse_imap_list(raw_folders)
        return IMAPScanner._build_folder_tree(parsed)
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def mount_and_return(
    mount_func,
    mount_kwargs: dict,
    mount_point: str,
    row: dict,
    label: str,
    expected_source: str | None = None,
) -> str | None:
    """Gemeinsame Mount-Logik fuer mount-basierte Konnektoren (SMB + WebDAV).

    Prueft ob der Mount-Point bereits gemountet ist, fuehrt bei Bedarf
    den Mount aus und gibt den Mount-Point zurueck (oder None bei Fehler).

    Args:
        mount_func: Die Mount-Funktion (z.B. mount_smb, mount_webdav).
        mount_kwargs: Keyword-Argumente fuer die Mount-Funktion.
        mount_point: Ziel-Mount-Point.
        row: Die vollstaendige DB-Row (fuer Logging).
        label: Label fuer Log-Meldungen (z.B. "SMB", "WebDAV").
        expected_source: Erwartete Remote-Source; vorhandene Mounts mit
            abweichender Source werden aus- und neu eingehangen.

    Returns:
        mount_point bei Erfolg, None bei Mount-Fehler.
    """
    from kiara.shared.mount import (
        is_mounted,
        mount_source,
        mount_source_matches,
        unmount,
    )

    if is_mounted(mount_point) and expected_source:
        actual_source = mount_source(mount_point)
        if not mount_source_matches(actual_source, expected_source):
            logger.warning(
                "%s mount source mismatch for ds%s at %s: actual=%r expected=%r; remounting",
                label,
                row["id"],
                mount_point,
                actual_source,
                expected_source,
            )
            ok, msg = unmount(mount_point)
            if not ok:
                logger.error("%s unmount failed for ds%s: %s", label, row["id"], msg)
                return None

    if not is_mounted(mount_point):
        ok, msg = mount_func(mount_point=mount_point, **mount_kwargs)
        if not ok:
            logger.error("%s mount failed for ds%s: %s", label, row["id"], msg)
            return None
    return mount_point


class ConnectorBase(ABC):
    """Basisklasse fuer alle KIara-Konnektoren.

    Jeder Konnektor beschreibt sich vollstaendig selbst:
    - Typ-Name und Anzeige-Label
    - Config-Schema (welche Felder, welche Defaults)
    - Passwort-Felder (fuer Verschluesselung und Maskierung)
    - Display-Logik (Quell-Anzeige in der Tabelle)
    - Pfad-Berechnung (wo liegen die Dateien fuer die Pipeline?)
    - Test-Logik (Verbindungstest)
    - Scan-Logik (Dateien scannen und in output_dir schreiben)
    """

    # --- Typ-Metadaten (Klassenattribute) ---

    name: str  # Eindeutiger Typ-Name, z.B. "imap", "smb", "local"
    label: str  # Anzeige-Label fuer die UI, z.B. "E-Mail (IMAP)"
    badge_class: str  # CSS-Klasse fuer das Badge, z.B. "info", "warning"

    # Hash-Strategie fuer die Ingestion-Pipeline.
    # "mtime" = schneller stat()-basierter Hash (Default).
    # "content" = liest Datei-Header, immun gegen mtime-Aenderungen.
    HASH_MODE: str = "mtime"
    SUPPORTS_PARALLEL_SCAN: bool = False
    supports_browse: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.supports_browse and "list_directory" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}: supports_browse=True erfordert eine eigene "
                f"list_directory-Implementierung (Default wirft ConnectorBrowseError)."
            )

    # --- Config-Schema ---

    @staticmethod
    @abstractmethod
    def config_schema() -> dict[str, dict[str, Any]]:
        """Gibt das Config-Schema zurueck.

        Jeder Key ist ein Feldname, der Wert beschreibt das Feld:
        {
            "server": {
                "type": "str",           # "str", "int" oder "bool"
                "default": "",           # Default-Wert
                "label": "IMAP-Server",  # Anzeige-Label
                "placeholder": "...",    # Placeholder-Text (optional)
                "required": True,        # Pflichtfeld (optional, default: False)
                "secret": True,          # Passwort-Feld (optional, default: False)
                "options": [...],        # Dropdown-Optionen (optional)
                "multiline": True,       # Textarea statt Input (optional)
                "hint": "...",           # Hilfetext unter dem Feld (optional)
                "browse": {              # Folder-Picker (optional)
                    "endpoint": "/api/...",
                    "depends_on": ["server", "share"],
                },
            },
        }

        Unterstuetzte Typen: "str", "int", "bool"

        Optionale Keys:
        - "label": Anzeige-Label fuer die UI
        - "placeholder": Platzhalter-Text
        - "required": Pflichtfeld (default False)
        - "secret": Passwort-Feld, wird verschluesselt und maskiert (default False)
        - "options": list[dict] mit {"value": str, "label": str} fuer Dropdown-Felder
        - "multiline": bool, True fuer Textarea statt Input-Feld
        - "hint": str, Hilfetext unterhalb des Felds
        - "browse": dict mit "endpoint" (str) und "depends_on" (list[str]),
          erzeugt einen "Durchsuchen..."-Button neben dem Feld
        - "folder_tree": bool, True fuer Ordner-Picker mit hierarchischer
          Baumansicht (Lade-Button + Checkboxen + "Alle"-Toggle)
        """
        ...

    @classmethod
    def secret_fields(cls) -> list[str]:
        """Gibt die Namen der Passwort-/Secret-Felder zurueck.

        Default-Implementierung: alle Felder mit "secret": True im Schema.
        """
        return [name for name, spec in cls.config_schema().items() if spec.get("secret", False)]

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """Gibt die Default-Konfiguration zurueck (alle Felder mit Defaults)."""
        return {
            name: spec.get("default", "" if spec.get("type") == "str" else None)
            for name, spec in cls.config_schema().items()
        }

    @staticmethod
    def list_directory(
        config: dict,
        path: str = "",
        *,
        directories_only: bool = False,
    ) -> dict:
        """Optionale Browse-Capability fuer dateibasierte Konnektoren."""
        from kiara.connectors.browse import ConnectorBrowseError

        raise ConnectorBrowseError("Browse nicht unterstuetzt", 400, "unsupported")

    # --- Display ---

    @staticmethod
    @abstractmethod
    def source_display(config: dict, row: dict) -> str:
        """Formatiert die Quell-Anzeige fuer die Tabelle.

        Args:
            config: Der connector_config-Dict (entschluesselt).
            row: Die vollstaendige DB-Row (fuer id, name etc.).

        Returns:
            z.B. "user@mail.firma.de" oder "fileserver:/Share/Docs"
        """
        ...

    @staticmethod
    @abstractmethod
    def effective_path(config: dict, row: dict) -> str:
        """Berechnet den effektiven Dateisystem-Pfad fuer die Pipeline.

        Args:
            config: Der connector_config-Dict (entschluesselt).
            row: Die vollstaendige DB-Row.

        Returns:
            Absoluter Pfad, z.B. "/mnt/kiara-ds-3"
        """
        ...

    # --- Verbindung ---

    @staticmethod
    @abstractmethod
    def test_connection(config: dict, row: dict) -> dict:
        """Testet die Verbindung zur Datenquelle.

        Args:
            config: Der connector_config-Dict (entschluesselt).
            row: Die vollstaendige DB-Row.

        Returns:
            {"success": True/False, "message": str, ...}
        """
        ...

    @classmethod
    def safe_test_connection(cls, config: dict, row: dict) -> dict:
        """Wraps test_connection() mit einheitlicher Fehlerbehandlung.

        Faengt alle Exceptions und gibt immer ein dict mit
        ``success`` und ``message`` zurueck.
        """
        try:
            return cls.test_connection(config, row)
        except Exception as exc:
            logger.error(
                "Verbindungstest %s fehlgeschlagen fuer ds%s: %s", cls.name, row.get("id", "?"), exc
            )
            return {"success": False, "message": str(exc)}

    @classmethod
    def safe_prepare_and_scan(
        cls, config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        """Wraps prepare_and_scan() mit einheitlichem Logging bei Fehlern."""
        try:
            return cls.prepare_and_scan(
                config,
                row,
                ds_config,
                progress_callback=progress_callback,
                control_callback=control_callback,
            )
        except Exception as exc:
            logger.error(
                "Prepare/scan %s fehlgeschlagen fuer ds%s: %s", cls.name, row.get("id", "?"), exc
            )
            return None

    # --- Scan-Cache ---

    SCAN_CACHE_MAX_AGE: int = 0  # Kein Cache per Default; Subklassen setzen z.B. 86400

    @classmethod
    def _check_scan_cache(cls, cache_dir: str, label: str = "") -> bool:
        """Prueft ob ein Scan-Cache noch aktuell ist.

        Args:
            cache_dir: Pfad zum Cache-Verzeichnis (mit .scan_done Marker).
            label: Anzeige-Label fuer Log-Meldungen (z.B. "IMAP", "Exchange").

        Returns:
            True wenn der Cache aktuell ist und der Scan uebersprungen werden kann.
        """
        if cls.SCAN_CACHE_MAX_AGE <= 0:
            return False
        marker = os.path.join(cache_dir, ".scan_done")
        if os.path.exists(marker):
            marker_age = time.time() - os.path.getmtime(marker)
            if marker_age < cls.SCAN_CACHE_MAX_AGE:
                logger.info(
                    "%s-Cache aktuell (Alter: %d Min) — Scan uebersprungen",
                    label or cls.name,
                    int(marker_age / 60),
                )
                return True
        return False

    # --- Scan ---

    @staticmethod
    @abstractmethod
    def prepare_and_scan(
        config: dict, row: dict, ds_config: dict, progress_callback=None, control_callback=None
    ) -> str | None:
        """Bereitet die Datenquelle vor (mount, scan) und gibt den Pfad zurueck.

        WICHTIG: Bei Fehlern den Fehler per logger.error() loggen, bevor None
        zurueckgegeben wird.

        Args:
            config: Der connector_config-Dict (entschluesselt).
            row: Die vollstaendige DB-Row.
            ds_config: Die globale KIara-Config (deepcopy, darf mutiert werden).
            progress_callback: Optionaler Callback(processed, total) fuer
                Fortschrittsanzeige waehrend der Synchronisierung.
            control_callback: Optionaler Callback() -> str|None.
                Gibt "cancel" zurueck wenn der Job abgebrochen werden soll.

        Returns:
            Absoluter Pfad zum Verzeichnis mit den zu indexierenden Dateien,
            oder None wenn die Vorbereitung fehlschlaegt.
        """
        ...
