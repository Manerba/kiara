import logging
from typing import TYPE_CHECKING

from kiara.tools.base import ToolDefinition

if TYPE_CHECKING:
    from kiara.admin.function import AdminFunction

logger = logging.getLogger(__name__)


class AdminFunctionRegistry:
    """Singleton-Registry aller Admin-Funktionen."""

    def __init__(self) -> None:
        self._functions: dict[str, "AdminFunction"] = {}

    def register(self, func: "AdminFunction") -> None:
        """Registriert eine AdminFunction. Bei Duplikat: Warning + Überschreiben."""
        if func.name in self._functions:
            logger.warning("AdminFunction '%s' wird ueberschrieben.", func.name)
        self._functions[func.name] = func
        logger.info("AdminFunction '%s' registriert.", func.name)

    def get(self, name: str) -> "AdminFunction | None":
        return self._functions.get(name)

    def list_all(self) -> list["AdminFunction"]:
        return list(self._functions.values())

    def list_by_category(self, category: str) -> list["AdminFunction"]:
        return [f for f in self._functions.values() if f.category == category]

    def categories(self) -> list[str]:
        return sorted(set(f.category for f in self._functions.values()))

    # --- Tool-Integration ---

    def as_tool_definitions(self) -> list[ToolDefinition]:
        """Konvertiert alle AdminFunctions in ToolDefinitions.

        Mapping:
        - name -> name
        - description -> description
        - parameters -> parameters (bereits JSON-Schema)
        - requires_confirmation -> requires_confirmation
        - required_permissions -> [] (Admin-only, Berechtigung über get_admin_user)
        - category -> category
        """
        return [
            ToolDefinition(
                name=f.name,
                description=f.description,
                parameters=f.parameters,
                requires_confirmation=f.requires_confirmation,
                category=f.category,
            )
            for f in self._functions.values()
        ]

    def get_tools_prompt(self) -> str:
        """Generiert LLM-Prompt-Abschnitt analog zu ToolRegistry.get_tools_prompt()."""
        from kiara.tools.registry import ToolRegistry

        definitions = self.as_tool_definitions()
        if not definitions:
            return ""
        lines = ["Verfuegbare Admin-Funktionen:"]
        for idx, defn in enumerate(definitions, start=1):
            param_parts = ToolRegistry._format_params(defn.parameters)
            lines.append(f"{idx}. {defn.name}({param_parts}) – {defn.description}")
        lines.append("")
        lines.append("Wenn du eine Funktion nutzen moechtest, antworte mit:")
        lines.append('TOOL_CALL: function_name(param1="wert1", param2="wert2")')
        return "\n".join(lines)

    # --- Help-Integration (delegiert an help.py) ---

    def get_help(self, name: str) -> dict | None:
        """Delegiert an help.get_help()."""
        from kiara.admin.help import get_help

        func = self.get(name)
        if func is None:
            return None
        return get_help(func)

    def search(self, query: str) -> list["AdminFunction"]:
        """Substring-Suche über name, description und examples."""
        from kiara.admin.help import search_help

        return search_help(self, query)

    # --- Wiki / API-Katalog ---

    def _serialize_function(self, func: "AdminFunction") -> dict:
        """Serialisiert eine einzelne Funktion fuer Wiki/API-Katalog."""
        return {
            "name": func.name,
            "category": func.category,
            "description": func.description,
            "help": func.help,
            "examples": func.examples,
            "parameters": func.parameters,
            "returns": func.returns,
            "requires_confirmation": func.requires_confirmation,
            "confirm_template": func.confirm_template,
            "pk_param": func.pk_param,
            "changelog_area": func.changelog_area,
            "changelog_action": func.changelog_action,
            "http_method": func.http_method,
            "http_path": func.http_path,
            "raw_response": func.raw_response,
        }

    def serialize_all(self) -> list[dict]:
        """Serialisiert alle Funktionen fuer Wiki/API-Katalog.

        Gibt fuer jede Funktion ein Dict mit allen Metadaten zurueck,
        inklusive Self-Check-Ergebnis.
        """
        results = []
        for func in self._functions.values():
            data = self._serialize_function(func)
            data["checks"] = self._check_function(func)
            results.append(data)
        return results

    def get_serialized(self, name: str) -> dict | None:
        """Serialisiert eine einzelne Funktion inkl. Self-Check. Oeffentliche API.

        Gibt None zurueck wenn die Funktion nicht existiert.
        Kombiniert get() + _serialize_function() + _check_function().
        """
        func = self._functions.get(name)
        if not func:
            return None
        data = self._serialize_function(func)
        data["checks"] = self._check_function(func)
        return data

    def _check_function(self, func: "AdminFunction") -> list[dict]:
        """Self-Check fuer eine einzelne Funktion."""
        issues = []
        if not func.help:
            issues.append({"level": "warning", "message": "Kein Help-Text"})
        if not func.examples:
            issues.append({"level": "info", "message": "Keine Beispiele"})
        if func.parameters and "properties" not in func.parameters:
            issues.append({"level": "warning", "message": "Parameter-Schema ohne properties-Key"})
        if func.requires_confirmation and not func.confirm_template:
            issues.append({"level": "error", "message": "Confirmation ohne Template"})
        if bool(func.changelog_area) != bool(func.changelog_action):
            issues.append({"level": "warning", "message": "Changelog inkonsistent"})
        if bool(func.http_method) != bool(func.http_path):
            issues.append(
                {
                    "level": "warning",
                    "message": "HTTP-Mapping unvollstaendig (http_method und http_path muessen beide gesetzt sein)",
                }
            )
        elif not func.http_method and not func.http_path:
            issues.append({"level": "info", "message": "Chat-only (kein HTTP-Mapping)"})
        return issues

    def get_self_check_summary(self) -> dict:
        """Aggregierter Self-Check ueber alle Funktionen."""
        total = len(self._functions)
        categories = len(set(f.category for f in self._functions.values()))
        errors = []
        warnings = []
        infos = []
        for func in self._functions.values():
            for issue in self._check_function(func):
                entry = {"function": func.name, "message": issue["message"]}
                if issue["level"] == "error":
                    errors.append(entry)
                elif issue["level"] == "warning":
                    warnings.append(entry)
                else:
                    infos.append(entry)
        return {
            "total_functions": total,
            "total_categories": categories,
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
        }


# Globale Singleton-Instanz
admin_registry = AdminFunctionRegistry()


def load_admin_functions() -> None:
    """Importiert alle Admin-Funktionsmodule, damit deren Decorator die Registry befüllen.

    Analog zum Pattern in kiara/connectors/registry.py (_load_builtin_modules).
    Muss einmalig beim Service-Start aufgerufen werden.
    """
    _modules = [
        "kiara.admin.functions.system",  # system.ping (Sprint 01) + health/changelog (Sprint 04)
        "kiara.admin.functions.dummy",  # test.confirm, test.wizard, test.wizard_confirm (Sprint 03)
        "kiara.admin.functions.network",  # Sprint 07: Netzwerk-Diagnose
        "kiara.admin.functions.services",  # Sprint 07: Service-Management
        "kiara.admin.functions.dashboard",  # dashboard.status (Sprint 04)
        "kiara.admin.functions.datasources",  # datasource.* (Sprint 04)
        "kiara.admin.functions.jobs",  # jobs.* (Sprint 04)
        "kiara.admin.functions.ki",  # ki.* (Sprint 04)
        "kiara.admin.functions.features",  # features.* (Sprint 04)
        "kiara.admin.functions.help",  # help.search (Sprint 04)
        "kiara.admin.functions.users",  # users.* (Sprint 06)
        "kiara.admin.functions.groups",  # groups.* (Sprint 06)
        # Sprint 08: System & Infrastruktur
        "kiara.admin.functions.backup",  # backup.* (Sprint 08)
        "kiara.admin.functions.elasticsearch",  # elasticsearch.* (Sprint 08)
        "kiara.admin.functions.cache",  # cache.* (Sprint 08)
        "kiara.admin.functions.mcp",  # mcp.* (Sprint 08)
        "kiara.admin.functions.ldap",  # ldap.* (Sprint 08)
        "kiara.admin.functions.mail",  # mail.* (Sprint 08)
        "kiara.admin.functions.vectordb",  # vectordb.* (Sprint 08)
        # Sprint 09: Compliance, Tools & Abschluss
        "kiara.admin.functions.compliance",  # compliance.* (Sprint 09, 8 Funktionen, read-only)
        "kiara.admin.functions.tools",  # tools.* (Sprint 09, 17 Funktionen, 5 mit Confirmation)
        "kiara.admin.functions.cloud",  # cloud.* (Sprint 09, 5 Funktionen, 1 mit Confirmation)
        "kiara.admin.functions.workflows",  # workflows.* (Sprint 09, 7 Funktionen, 1 mit Confirmation)
        "kiara.admin.functions.workflows_admin",  # workflow.* (Sprint 11b, Guided Workflows)
        "kiara.admin.functions.testsuite",  # testsuite.* (Sprint 09, 2 Funktionen)
        "kiara.admin.functions.integrity",  # integrity.* (Integritaetspruefung)
        "kiara.admin.functions.monitoring",  # system.metrics (System-Monitor)
        "kiara.admin.functions.tracing",  # tracing.* (P17: Trace-Explorer)
        "kiara.admin.functions.system_events",  # system.events.* (Admin-System-Log)
    ]
    _failed = []
    for mod in _modules:
        try:
            __import__(mod)
            logger.info("Admin-Funktionsmodul '%s' geladen.", mod)
        except Exception as exc:
            logger.error("Admin-Funktionsmodul '%s' konnte nicht geladen werden: %s", mod, exc)
            _failed.append(mod)

    if _failed:
        logger.critical(
            "%d Admin-Funktionsmodul(e) NICHT geladen: %s",
            len(_failed),
            ", ".join(_failed),
        )
