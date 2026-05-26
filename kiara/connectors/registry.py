"""ConnectorRegistry — Zentrale Verwaltung aller Konnektoren.

Verwaltet die Zuordnung Datenquellen-Typ (z.B. "local", "imap") zu
Konnektor-Klasse. Konnektoren registrieren sich ueber:
  1. @register Decorator (fuer built-in Konnektoren)
  2. Entry-Points 'kiara.connectors' (fuer Plugins)

Laden ist lazy: ensure_loaded() wird beim ersten Zugriff aufgerufen.
"""

import importlib.metadata
import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiara.connectors.base import ConnectorBase

logger = logging.getLogger(__name__)

_registry: dict[str, type["ConnectorBase"]] = {}
_loaded: bool = False
_load_lock = threading.Lock()


def register(connector_cls: type["ConnectorBase"]) -> type["ConnectorBase"]:
    """Registriert einen Konnektor. Kann als Decorator verwendet werden.

    @register
    class IMAPConnector(ConnectorBase):
        name = "imap"
        ...
    """
    name = connector_cls.name
    if name in _registry:
        logger.warning("Connector '%s' already registered, overwriting", name)
    _registry[name] = connector_cls
    logger.debug("Connector registered: %s (%s)", name, connector_cls.__name__)
    return connector_cls


def get(name: str) -> type["ConnectorBase"]:
    """Gibt die Konnektor-Klasse fuer einen Typ-Namen zurueck."""
    if not isinstance(name, str) or not name:
        raise KeyError("Connector-Typ muss ein nicht-leerer String sein")
    try:
        return _registry[name]
    except KeyError as exc:
        raise KeyError(f"Unbekannter Connector-Typ: {name}") from exc


def all_connectors() -> dict[str, type["ConnectorBase"]]:
    """Gibt alle registrierten Konnektoren zurueck."""
    return dict(_registry)


def connector_names() -> list[str]:
    """Gibt alle registrierten Konnektor-Namen zurueck."""
    return list(_registry.keys())


def auto_mount_point(row) -> str:
    """Zentrale Mount-Point-Berechnung fuer mount-basierte Konnektoren.

    Erzeugt einen deterministischen Pfad aus der stabilen Datasource-ID.
    Ersetzt die bisherigen duplizierten Kopien in api_datasource.py
    und pipeline.py.
    """
    ds_id = row["id"]
    return f"/mnt/kiara-ds-{ds_id}"


def ensure_loaded() -> None:
    """Laedt alle verfuegbaren Konnektoren.

    1. Built-in-Konnektoren werden immer per direktem Import geladen
    2. Zusaetzlich werden Entry-Points gescannt fuer externe Plugins
       (z.B. ``pip install kiara-connector-webcrawler``)

    Built-in-Module werden immer direkt geladen, damit KIara auch ohne
    ``pip install -e .`` funktioniert (rsync-Deployment, Git-Checkout).
    Entry-Points ermoeglichen externe Plugins ohne Aenderung am Kerncode.
    """
    global _loaded
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return

        # 1. Built-in Module immer laden
        _load_builtin_modules()

        # 2. Externe Plugins via Entry-Points laden (zusaetzlich)
        _load_entry_points()

        # Erst NACH erfolgreichem Laden als geladen markieren
        _loaded = True

        loaded = connector_names()
        logger.info("Konnektoren geladen: %s", ", ".join(loaded) if loaded else "(keine)")


def _load_entry_points() -> None:
    """Laedt externe Konnektor-Plugins ueber importlib.metadata Entry-Points.

    Bereits registrierte Konnektoren (z.B. Built-ins) werden uebersprungen.
    """
    try:
        eps = importlib.metadata.entry_points(group="kiara.connectors")
    except Exception as exc:
        logger.debug("Entry-Point-Discovery fehlgeschlagen: %s", exc)
        return

    for ep in eps:
        if ep.name in _registry:
            logger.debug("Entry-Point '%s' uebersprungen (bereits registriert)", ep.name)
            continue
        try:
            ep.load()  # Import des Moduls triggert @register
            logger.debug("Konnektor via Entry-Point geladen: %s", ep.name)
        except Exception as exc:
            logger.error("Entry-Point '%s' konnte nicht geladen werden: %s", ep.name, exc)


def _load_builtin_modules() -> None:
    """Fallback: Laedt die 5 Built-in-Konnektoren per direktem Import.

    Wird verwendet wenn keine Entry-Points verfuegbar sind, z.B. bei
    rsync-Deployment oder direktem Start aus dem Git-Checkout.
    """
    _modules = [
        "kiara.connectors.local_connector",
        "kiara.connectors.smb_connector",
        "kiara.connectors.webdav_connector",
        "kiara.connectors.imap_connector",
        "kiara.connectors.exchange_connector",
        "kiara.connectors.sql_ds_connector",
    ]
    for mod in _modules:
        try:
            __import__(mod)
        except Exception as exc:
            logger.error("Konnektor-Modul %s konnte nicht geladen werden: %s", mod, exc)
