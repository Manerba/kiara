"""KIara Connector-Plugin-System.

Built-in-Konnektoren werden beim Import automatisch registriert.
Fuer neue Konnektoren: Eigene Datei in diesem Verzeichnis erstellen
und mit @register dekorieren.

Entry-Point-Discovery: Externe Plugins koennen sich ueber den
Entry-Point ``kiara.connectors`` registrieren (pip install).

Module:
    base      — ConnectorBase ABC (prepare_and_scan, test_connection, ...)
    registry  — Zentrale Registry (register, get, all_connectors, ensure_loaded)
    local_connector, smb_connector, webdav_connector,
    imap_connector, sql_ds_connector — Built-in-Konnektoren
"""

from kiara.connectors.base import ConnectorBase
from kiara.connectors.registry import all_connectors, ensure_loaded, get, register
