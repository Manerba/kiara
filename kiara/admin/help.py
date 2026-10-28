"""Hilfe-Funktionen fuer Admin-Funktionen — Aufbereitung, Auflistung und Suche.

Stellt get_help(), list_help() und search_help() bereit, um AdminFunction-Metadaten
(Name, Kategorie, Parameter-Schema, Beispiele) strukturiert abzufragen.
Wird von kiara.admin.functions.help (help.search AdminFunction) genutzt.
Importiert AdminFunction und AdminFunctionRegistry (TYPE_CHECKING).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiara.admin.function import AdminFunction
    from kiara.admin.registry import AdminFunctionRegistry


def get_help(func: AdminFunction) -> dict:
    return {
        "name": func.name,  # "datasource.delete"
        "category": func.category,  # "datasources"
        "short": func.description,  # 1 Zeile
        "long": func.help,  # Absatz
        "parameters": func.parameters,  # JSON-Schema
        "examples": func.examples,  # ["Lösche DS 12", ...]
        "returns": func.returns,  # "Bestätigung mit ..."
        "requires_confirmation": func.requires_confirmation,
    }


def list_help(registry: AdminFunctionRegistry, category: str | None = None) -> list[dict]:
    """Übersicht aller Funktionen, optional nach Kategorie gefiltert."""
    funcs = registry.list_by_category(category) if category else registry.list_all()
    return [get_help(f) for f in funcs]


def search_help(registry: AdminFunctionRegistry, query: str) -> list[AdminFunction]:
    """Substring-Suche (case-insensitive) über name, description und examples."""
    q = query.lower()
    results = []
    for f in registry.list_all():
        if (
            q in f.name.lower()
            or q in f.description.lower()
            or any(q in ex.lower() for ex in f.examples)
        ):
            results.append(f)
    return results
