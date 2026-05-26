"""Dummy-Funktionen fuer Sprint-03-Tests (Confirmation + Wizard)."""

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult


@admin_function(
    name="test.confirm",
    category="test",
    description="Dummy-Funktion die Bestaetigung erfordert (nur fuer Tests)",
    help="Simuliert eine destruktive Aktion die bestaetigt werden muss.",
    examples=["Loesche Test-Eintrag 42"],
    parameters={
        "type": "object",
        "properties": {"item_id": {"type": "integer", "description": "ID des Test-Eintrags"}},
        "required": ["item_id"],
    },
    returns="Bestaetigungsmeldung",
    requires_confirmation=True,
    confirm_template="test {item_id} löschen",
    changelog_area="test",
    changelog_action="delete",
    pk_param="item_id",
)
async def test_confirm(user: dict, item_id: int) -> AdminResult:
    return AdminResult.ok(f"Test-Eintrag {item_id} geloescht (Dummy).")


@admin_function(
    name="test.wizard",
    category="test",
    description="Dummy-Funktion mit mehreren Pflichtfeldern (nur fuer Tests)",
    help="Simuliert eine Erstellung die mehrere Parameter erfordert.",
    examples=["Erstelle einen Test-Eintrag"],
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name des Eintrags"},
            "url": {"type": "string", "description": "URL (z.B. https://...)"},
            "username": {"type": "string", "description": "Benutzername"},
            "password": {"type": "string", "description": "Passwort"},
            "enabled": {"type": "boolean", "description": "Aktiviert?"},
        },
        "required": ["name", "url", "username", "password"],
    },
    returns="Erstellungsmeldung",
)
async def test_wizard(
    user: dict, name: str, url: str, username: str, password: str, enabled: bool = True
) -> AdminResult:
    return AdminResult.ok(f"Test-Eintrag '{name}' erstellt (Dummy): {url}, User={username}")


@admin_function(
    name="test.wizard_confirm",
    category="test",
    description="Dummy-Funktion mit Wizard UND Bestaetigung (nur fuer Tests)",
    help="Simuliert eine destruktive Aktion die erst Parameter sammelt, dann bestaetigt.",
    examples=["Loesche alle Eintraege fuer Projekt X"],
    parameters={
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Projektname"},
            "scope": {"type": "string", "description": "Bereich (all, docs, data)"},
        },
        "required": ["project", "scope"],
    },
    returns="Loeschmeldung",
    requires_confirmation=True,
    confirm_template="{project} {scope} löschen",
    changelog_area="test",
    changelog_action="delete",
    pk_param="project",
)
async def test_wizard_confirm(user: dict, project: str, scope: str) -> AdminResult:
    return AdminResult.ok(f"Projekt '{project}' ({scope}) geloescht (Dummy).")
