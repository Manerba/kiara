"""Connection-Registry fuer ERP-Konnektoren.

Verwaltet benannte ERP-Verbindungen, deren Konnektor-Instanzen
und die Zuordnung von Tools zu Verbindungen. Laedt/speichert
den Zustand aus/in die kiara.yaml.
"""

import logging

from kiara.shared.crypto import decrypt, is_encrypted
from kiara.tools.erp_connectors.base import (
    ConnectionInfo,
    ERPConnectorBase,
)

logger = logging.getLogger(__name__)


class ERPAssignmentBrokenError(RuntimeError):
    """Tool war einer Connection zugewiesen, Connection/Konnektor ist aber defekt.

    Unterscheidet sich bewusst von "kein Assignment" (None). Verhindert,
    dass bei zerbrochenen Zuordnungen still auf Legacy-Backend (andere
    Datenquelle!) zurueckgefallen wird.
    """


class ConnectionRegistry:
    """Verwaltet benannte ERP-Verbindungen und deren Konnektor-Instanzen."""

    # Registrierte Konnektor-Typen (class-level)
    _connector_types: dict[str, type[ERPConnectorBase]] = {}

    def __init__(self) -> None:
        self._connections: dict[str, ConnectionInfo] = {}
        self._instances: dict[str, ERPConnectorBase] = {}
        self._tool_assignments: dict[str, str] = {}

    @classmethod
    def register_type(cls, connector_cls: type[ERPConnectorBase]) -> None:
        """Konnektor-Typ registrieren (beim Import)."""
        defn = connector_cls.connector_definition()
        cls._connector_types[defn.type_id] = connector_cls
        logger.debug("ERP-Konnektor-Typ registriert: %s", defn.type_id)

    def add_connection(self, info: ConnectionInfo) -> None:
        """Verbindung hinzufuegen/aktualisieren. Invalidiert gecachte Instanz."""
        self._connections[info.id] = info
        self._instances.pop(info.id, None)

    def remove_connection(self, connection_id: str) -> None:
        """Verbindung entfernen. Entfernt auch gecachte Instanz + Tool-Zuordnungen."""
        self._connections.pop(connection_id, None)
        self._instances.pop(connection_id, None)
        # Dangling Assignments entfernen
        self._tool_assignments = {
            tool: conn_id
            for tool, conn_id in self._tool_assignments.items()
            if conn_id != connection_id
        }

    def get_connector(self, connection_id: str) -> ERPConnectorBase:
        """Konnektor-Instanz fuer eine Verbindung (lazy init).

        Raises:
            KeyError: Verbindung nicht gefunden.
            ValueError: Konnektor-Typ nicht registriert oder Verbindung deaktiviert.
        """
        conn = self._connections.get(connection_id)
        if conn is None:
            raise KeyError(f"Verbindung '{connection_id}' nicht gefunden.")

        if not conn.enabled:
            self._instances.pop(connection_id, None)
            raise ValueError(f"Verbindung '{conn.name}' ist deaktiviert.")

        if connection_id in self._instances:
            return self._instances[connection_id]

        connector_cls = self._connector_types.get(conn.connector_type)
        if connector_cls is None:
            raise ValueError(f"Konnektor-Typ '{conn.connector_type}' nicht registriert.")

        # Secret-Felder entschluesseln vor Uebergabe an Konnektor
        decrypted_config = dict(conn.config)
        connector_def = connector_cls.connector_definition()
        for field in connector_def.config_fields:
            if field.get("type") == "password":
                name = field["name"]
                raw = decrypted_config.get(name, "")
                decrypted_config[name] = decrypt(raw) if is_encrypted(raw) else raw

        decrypted_conn = ConnectionInfo(
            id=conn.id,
            name=conn.name,
            connector_type=conn.connector_type,
            config=decrypted_config,
            enabled=conn.enabled,
        )
        instance = connector_cls(decrypted_conn)
        self._instances[connection_id] = instance
        return instance

    def get_connector_for_tool(self, tool_name: str) -> ERPConnectorBase | None:
        """Konnektor fuer ein bestimmtes Tool (ueber Zuordnung).

        Returns:
            None wenn kein Assignment existiert (Legacy-Fallback erwuenscht).

        Raises:
            ERPAssignmentBrokenError: Assignment existiert, aber Connection
                ist geloescht, deaktiviert oder Konnektor-Typ nicht registriert.
                Fail-closed: kein stiller Fallback auf Legacy (andere Datenquelle!).
        """
        connection_id = self._tool_assignments.get(tool_name)
        if not connection_id:
            return None
        try:
            return self.get_connector(connection_id)
        except (KeyError, ValueError) as exc:
            logger.error(
                "Tool '%s' ist Connection '%s' zugewiesen, aber nicht verfuegbar: %s",
                tool_name,
                connection_id,
                exc,
            )
            raise ERPAssignmentBrokenError(
                f"Tool '{tool_name}' ist der Connection '{connection_id}' "
                f"zugewiesen, aber diese ist nicht verfuegbar: {exc}"
            ) from exc

    def assign_tool(self, tool_name: str, connection_id: str) -> None:
        """Tool einer Verbindung zuweisen."""
        self._tool_assignments[tool_name] = connection_id

    def unassign_tool(self, tool_name: str) -> None:
        """Tool-Zuordnung entfernen."""
        self._tool_assignments.pop(tool_name, None)

    def list_connections(self) -> list[ConnectionInfo]:
        """Alle registrierten Verbindungen."""
        return list(self._connections.values())

    def list_connector_types(self) -> list[dict]:
        """Alle verfuegbaren Konnektor-Typen als Dicts (fuer API-Response)."""
        result = []
        for connector_cls in self._connector_types.values():
            defn = connector_cls.connector_definition()
            result.append(
                {
                    "type_id": defn.type_id,
                    "display_name": defn.display_name,
                    "config_fields": defn.config_fields,
                }
            )
        return result

    def get_tool_assignments(self) -> dict[str, str]:
        """Aktuelle Tool→Verbindung Zuordnungen."""
        return dict(self._tool_assignments)

    def load_from_config(self, config: dict) -> None:
        """Verbindungen + Zuordnungen aus kiara.yaml laden.

        Einzelne fehlerhafte Connections werden uebersprungen (log.warning),
        damit ein Konfigurationsfehler nicht alle ERP-Tools blockiert.
        """
        erp_cfg = config.get("tools", {}).get("erp", {})
        connections = erp_cfg.get("connections", [])

        for conn_dict in connections:
            try:
                info = ConnectionInfo(
                    id=conn_dict["id"],
                    name=conn_dict["name"],
                    connector_type=conn_dict["type"],
                    config=conn_dict.get("config", {}),
                    enabled=conn_dict.get("enabled", True),
                )
                self.add_connection(info)
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Fehlerhafte ERP-Connection in Config uebersprungen: %s — %s",
                    conn_dict,
                    exc,
                )

        # Tool-Zuordnungen laden
        assignments = erp_cfg.get("tool_assignments", {})
        for tool_name, conn_id in assignments.items():
            self._tool_assignments[tool_name] = conn_id

    def save_to_config(self) -> dict:
        """Verbindungen + Zuordnungen als Config-Dict zurueckgeben."""
        connections = []
        for conn in self._connections.values():
            connections.append(
                {
                    "id": conn.id,
                    "name": conn.name,
                    "type": conn.connector_type,
                    "config": conn.config,
                    "enabled": conn.enabled,
                }
            )
        return {
            "connections": connections,
            "tool_assignments": dict(self._tool_assignments),
        }
