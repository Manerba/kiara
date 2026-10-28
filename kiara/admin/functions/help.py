from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult


@admin_function(
    name="help.search",
    category="help",
    description="Hilfe zu Admin-Funktionen anzeigen",
    help="Durchsucht das Help-System nach Funktionen. Kann nach Name, "
    "Kategorie oder Freitext suchen.",
    examples=[
        "Hilfe zu Datenquellen",
        "Was kann ich mit Jobs machen?",
        "Wie erstelle ich eine neue Datenquelle?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Suchbegriff (Name, Kategorie oder Freitext)",
            },
            "category": {
                "type": "string",
                "description": "Kategorie filtern (z.B. 'datasources', 'jobs')",
            },
            "function_name": {
                "type": "string",
                "description": "Exakter Funktionsname (z.B. 'datasource.list')",
            },
        },
        "required": [],
    },
    returns="Liste der passenden Funktionen mit Beschreibung und Parametern",
)
async def help_search(
    user: dict, query: str = "", category: str = "", function_name: str = ""
) -> AdminResult:
    from kiara.admin.registry import admin_registry

    # Exakter Name -> Detail-Anzeige
    if function_name:
        info = admin_registry.get_help(function_name)
        if info:
            return AdminResult.ok(f"Hilfe zu {function_name}", data=info)
        return AdminResult.fail(f"Funktion '{function_name}' nicht gefunden")

    # Kategorie-Filter
    if category:
        funcs = admin_registry.list_by_category(category)
        if not funcs:
            return AdminResult.fail(f"Keine Funktionen in Kategorie '{category}'")
        data = [{"name": f.name, "description": f.description} for f in funcs]
        return AdminResult.ok(f"{len(funcs)} Funktionen in '{category}'", data={"functions": data})

    # Freitext-Suche
    if query:
        funcs = admin_registry.search(query)
        if not funcs:
            return AdminResult.fail(f"Keine Treffer für '{query}'")
        data = [{"name": f.name, "description": f.description} for f in funcs]
        return AdminResult.ok(f"{len(funcs)} Treffer", data={"functions": data})

    # Keine Parameter -> alle Funktionen auflisten
    funcs = admin_registry.list_all()
    data = [{"name": f.name, "description": f.description, "category": f.category} for f in funcs]
    return AdminResult.ok(f"{len(funcs)} Admin-Funktionen verfügbar", data={"functions": data})
