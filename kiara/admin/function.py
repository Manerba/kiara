from dataclasses import dataclass
from typing import Callable


@dataclass
class AdminFunction:
    name: str  # "datasource.delete"
    category: str  # "datasources"
    description: str  # Kurzbeschreibung (1 Zeile)
    help: str  # Ausführliche Beschreibung (Absatz)
    examples: list[str]  # Natürlichsprachige Beispiele
    parameters: dict  # JSON-Schema (Standard-Format)
    returns: str  # Beschreibung des Rückgabewerts
    requires_confirmation: bool  # Server-seitige Bestätigung nötig?
    confirm_template: str | None  # z.B. "ds {ds_id} löschen"
    pk_param: str | None  # Name des PK-Parameters für Changelog target_id
    changelog_area: str | None  # Bereich für config_changelog
    changelog_action: str | None  # "create" | "update" | "delete"
    http_method: str | None  # "GET" | "POST" | "PUT" | "DELETE"
    http_path: str | None  # "/api/admin/datasources/{ds_id}"
    fn: Callable  # Die eigentliche async-Funktion
    raw_response: bool = False  # True → Auto-Router gibt data direkt zurück


def admin_function(
    name: str,
    category: str,
    description: str,
    help: str,
    examples: list[str],
    parameters: dict,
    returns: str,
    requires_confirmation: bool = False,
    confirm_template: str | None = None,
    pk_param: str | None = None,
    changelog_area: str | None = None,
    changelog_action: str | None = None,
    raw_response: bool = False,
    http: tuple[str, str] | None = None,
):
    """Decorator der eine Funktion als AdminFunction registriert.

    Args:
        http: Tuple (method, path), z.B. ("DELETE", "/api/admin/datasources/{ds_id}").
              Wird intern in http_method und http_path aufgesplittet.
    """

    def decorator(fn):
        http_method, http_path = http if http else (None, None)
        func = AdminFunction(
            name=name,
            category=category,
            description=description,
            help=help,
            examples=examples,
            parameters=parameters,
            returns=returns,
            requires_confirmation=requires_confirmation,
            confirm_template=confirm_template,
            pk_param=pk_param,
            changelog_area=changelog_area,
            changelog_action=changelog_action,
            http_method=http_method,
            http_path=http_path,
            fn=fn,
            raw_response=raw_response,
        )
        # Auto-Registrierung in der globalen Registry
        from kiara.admin.registry import admin_registry

        admin_registry.register(func)
        return fn  # Originalfunktion bleibt unverändert aufrufbar

    return decorator
