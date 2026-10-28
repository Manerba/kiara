"""Tool-Registry fuer KIara.

Verwaltet registrierte Tools, filtert nach Berechtigungen und generiert
den Prompt-Abschnitt, der dem LLM die verfuegbaren Werkzeuge beschreibt.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from kiara.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)

# Mapping von JSON-Schema-Typ auf lesbaren Kurznamen fuer den Prompt
_TYPE_LABELS: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


class ToolRegistry:
    """Zentrale Registry, in der alle verfuegbaren Tools registriert werden.

    Beispiel::

        registry = ToolRegistry()
        registry.register(MyTool())
        tools = registry.list_tools(user_groups=["admin"])
    """

    def __init__(
        self,
        *,
        tool_permissions: Mapping[str, Sequence[str]] | None = None,
        permission_provider: Callable[[], Mapping[str, Sequence[str]] | None] | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._tool_permissions = tool_permissions
        self._permission_provider = permission_provider

    def register(self, tool: BaseTool, *, allow_overwrite: bool = False) -> None:
        """Registriert ein Tool in der Registry.

        Args:
            tool: Eine Instanz einer BaseTool-Unterklasse.
            allow_overwrite: Explizite Ausnahme fuer bewusstes Ersetzen.
        """
        defn = tool.definition()
        if defn.name in self._tools and not allow_overwrite:
            raise ValueError(f"Tool '{defn.name}' ist bereits registriert")
        if defn.name in self._tools:
            logger.info("Tool '%s' wird explizit ersetzt.", defn.name)
        self._tools[defn.name] = tool
        logger.info("Tool '%s' registriert.", defn.name)

    def get_tool(self, name: str) -> BaseTool | None:
        """Gibt ein registriertes Tool anhand seines Namens zurueck.

        Args:
            name: Der technische Name des Tools.

        Returns:
            Die BaseTool-Instanz oder None, falls nicht vorhanden.
        """
        return self._tools.get(name)

    def list_tools(self, user_groups: list[str] | None = None) -> list[ToolDefinition]:
        """Listet alle Tool-Definitionen auf, gefiltert nach Berechtigungen.

        Regeln:
        - Builtin-Tools sind fuer alle sichtbar.
        - Wenn eine Permission-Matrix gesetzt ist, ist sie fuer alle nicht
          eingebauten Tools autoritativ: Nur Gruppen, deren Tool-Liste den
          Tool-Namen enthaelt, duerfen es sehen.
        - Ohne Permission-Matrix gilt die Tool-eigene ``required_permissions``.
        - Wenn ``user_groups`` den Wert ``"*"`` enthaelt, werden alle Tools angezeigt.
        - Andernfalls wird ein Tool nur angezeigt, wenn mindestens eine Gruppe des
          Users in den effektiven ``required_permissions`` enthalten ist.

        Args:
            user_groups: Gruppen/Rollen des anfragenden Users.

        Returns:
            Liste von ToolDefinition-Objekten, die dem User zur Verfuegung stehen.
        """
        groups = user_groups or []
        has_wildcard = "*" in groups
        permission_matrix = self._current_permission_matrix()
        result: list[ToolDefinition] = []

        for tool in self._tools.values():
            defn = tool.definition()
            effective = self._effective_definition(defn, permission_matrix)

            if effective.category == "builtin":
                result.append(effective)
                continue

            if has_wildcard:
                result.append(effective)
                continue

            if not effective.required_permissions:
                if permission_matrix is None:
                    result.append(effective)
                continue

            if set(groups) & set(effective.required_permissions):
                result.append(effective)

        return result

    def _current_permission_matrix(self) -> dict[str, list[str]] | None:
        if self._permission_provider is not None:
            try:
                raw = self._permission_provider()
            except Exception as exc:
                logger.error(
                    "Tool-Permission-Matrix konnte nicht geladen werden: %s",
                    exc,
                )
                return self._static_permission_matrix()
            if raw is None:
                return None
            try:
                return _normalize_permission_matrix(raw)
            except ValueError as exc:
                logger.error("Tool-Permission-Matrix ungueltig: %s", exc)
                return self._static_permission_matrix()

        return self._static_permission_matrix()

    def _static_permission_matrix(self) -> dict[str, list[str]] | None:
        if self._tool_permissions is None:
            return None
        try:
            return _normalize_permission_matrix(self._tool_permissions)
        except ValueError as exc:
            logger.error("Tool-Permission-Matrix ungueltig: %s", exc)
            return {}

    @staticmethod
    def _effective_definition(
        defn: ToolDefinition,
        permission_matrix: dict[str, list[str]] | None,
    ) -> ToolDefinition:
        if permission_matrix is None or defn.category == "builtin":
            return defn

        groups = [
            group
            for group, tools in permission_matrix.items()
            if "*" in tools or defn.name in tools
        ]
        return replace(defn, required_permissions=groups)

    def get_tools_prompt(self, user_groups: list[str] | None = None) -> str:
        """Generiert einen Prompt-Abschnitt, der dem LLM die Tools beschreibt.

        Args:
            user_groups: Gruppen/Rollen des anfragenden Users (fuer Filterung).

        Returns:
            Mehrzeiliger String mit Werkzeug-Beschreibungen und Aufruf-Anleitung.
            Leerer String, wenn keine Tools verfuegbar sind.
        """
        definitions = self.list_tools(user_groups=user_groups)
        if not definitions:
            return ""

        lines: list[str] = ["Verfuegbare Werkzeuge:"]

        for idx, defn in enumerate(definitions, start=1):
            try:
                param_parts = self._format_params(defn.parameters)
                lines.append(f"{idx}. {defn.name}({param_parts}) \u2013 {defn.description}")
            except Exception as exc:
                logger.error("Tool '%s' uebersprungen (Schema-Fehler): %s", defn.name, exc)

        lines.append("")
        lines.append("Wenn du ein Werkzeug nutzen moechtest, antworte mit:")
        lines.append('TOOL_CALL: tool_name(param1="wert1", param2="wert2")')

        return "\n".join(lines)

    @staticmethod
    def _format_params(schema: dict) -> str:
        """Formatiert die Parameter eines JSON-Schemas als Signatur-String.

        Args:
            schema: Das JSON-Schema-Dict aus der ToolDefinition.

        Returns:
            z.B. ``'order_id: str, limit: int'``
        """
        properties: dict = schema.get("properties", {})
        required: list[str] = schema.get("required", [])
        parts: list[str] = []

        for name, prop in properties.items():
            raw_type = prop.get("type")
            if isinstance(raw_type, list):
                type_str = "|".join(_TYPE_LABELS.get(t, str(t)) for t in raw_type)
            elif isinstance(raw_type, str) and raw_type:
                type_str = _TYPE_LABELS.get(raw_type, raw_type)
            else:
                type_str = "any"
            if name in required:
                parts.append(f"{name}: {type_str}")
            else:
                parts.append(f"{name}: {type_str} = optional")

        return ", ".join(parts)


def _normalize_permission_matrix(
    raw: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    if not isinstance(raw, Mapping):
        raise ValueError("Permission-Matrix muss ein Objekt sein.")

    normalized: dict[str, list[str]] = {}
    for group, tools in raw.items():
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Permission-Gruppe muss ein nicht-leerer String sein.")
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise ValueError(f"Tool-Liste fuer Gruppe {group!r} muss eine Liste sein.")
        cleaned: list[str] = []
        for tool in tools:
            if not isinstance(tool, str) or not tool.strip():
                raise ValueError(
                    f"Tool-Name fuer Gruppe {group!r} muss ein nicht-leerer String sein."
                )
            cleaned.append(tool)
        normalized[group] = cleaned
    return normalized


def load_config_tool_permissions() -> dict[str, list[str]]:
    """Laedt die aktuelle Tool-Permission-Matrix aus der KIara-Config."""
    from kiara.config import load_config

    config = load_config()
    permissions = (config.get("tools") or {}).get("permissions", {})
    return _normalize_permission_matrix(permissions)
