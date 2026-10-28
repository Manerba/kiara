"""Registry-Konsistenz-Tests fuer alle ~155 Admin-Funktionen.

Stellt sicher, dass jede AdminFunction korrekt definiert und konsistent
konfiguriert ist. Faengt Regressionen sofort auf (z.B. fehlender Help-Text).
"""

import asyncio
import inspect
import re

import pytest

from kiara.admin.registry import admin_registry, load_admin_functions


@pytest.fixture(scope="module", autouse=True)
def _load_registry():
    """Laedt alle Admin-Funktionen einmalig fuer das gesamte Modul."""
    load_admin_functions()


# ---------------------------------------------------------------------------
# 1. Help-Text
# ---------------------------------------------------------------------------


def test_all_functions_have_help():
    for func in admin_registry.list_all():
        assert func.help and func.help.strip(), f"{func.name}: help-Text fehlt oder leer"


# ---------------------------------------------------------------------------
# 2. Description
# ---------------------------------------------------------------------------


def test_all_functions_have_description():
    for func in admin_registry.list_all():
        assert func.description and func.description.strip(), (
            f"{func.name}: description fehlt oder leer"
        )


# ---------------------------------------------------------------------------
# 3. Examples
# ---------------------------------------------------------------------------


def test_all_functions_have_examples():
    for func in admin_registry.list_all():
        assert func.examples and len(func.examples) > 0, f"{func.name}: examples fehlt oder leer"


# ---------------------------------------------------------------------------
# 4. Parameter-Schema Validierung
# ---------------------------------------------------------------------------


def test_all_functions_have_valid_parameters_schema():
    for func in admin_registry.list_all():
        p = func.parameters
        assert isinstance(p, dict), f"{func.name}: parameters ist kein dict"
        assert p.get("type") == "object", (
            f"{func.name}: parameters.type muss 'object' sein, ist {p.get('type')}"
        )
        assert "properties" in p, f"{func.name}: parameters.properties fehlt"
        assert isinstance(p["properties"], dict), (
            f"{func.name}: parameters.properties ist kein dict"
        )


# ---------------------------------------------------------------------------
# 5. Parameter-Beschreibungen
# ---------------------------------------------------------------------------


def test_all_parameters_have_description():
    for func in admin_registry.list_all():
        props = func.parameters.get("properties", {})
        for param_name, param_def in props.items():
            assert param_def.get("description"), (
                f"{func.name}.{param_name}: Parameter-Beschreibung fehlt"
            )


# ---------------------------------------------------------------------------
# 6. Confirmation → Template
# ---------------------------------------------------------------------------


def test_confirmation_has_template():
    for func in admin_registry.list_all():
        if func.requires_confirmation:
            assert func.confirm_template and func.confirm_template.strip(), (
                f"{func.name}: requires_confirmation=True aber confirm_template fehlt"
            )


# ---------------------------------------------------------------------------
# 7. Confirmation-Template Platzhalter
# ---------------------------------------------------------------------------


def test_confirmation_template_uses_valid_params():
    for func in admin_registry.list_all():
        if not func.confirm_template:
            continue
        placeholders = re.findall(r"\{(\w+)\}", func.confirm_template)
        schema_params = set(func.parameters.get("properties", {}).keys())
        for ph in placeholders:
            assert ph in schema_params, (
                f"{func.name}: confirm_template referenziert '{{{ph}}}' "
                f"aber Parameter existiert nicht im Schema. Vorhanden: {schema_params}"
            )


# ---------------------------------------------------------------------------
# 8. Changelog-Felder gepaart
# ---------------------------------------------------------------------------


def test_changelog_fields_are_paired():
    for func in admin_registry.list_all():
        has_area = bool(func.changelog_area)
        has_action = bool(func.changelog_action)
        assert has_area == has_action, (
            f"{func.name}: changelog_area={func.changelog_area!r} aber "
            f"changelog_action={func.changelog_action!r} — muessen beide gesetzt oder beide leer sein"
        )


# ---------------------------------------------------------------------------
# 9. Keine doppelten Namen
# ---------------------------------------------------------------------------


def test_no_duplicate_names():
    names = [f.name for f in admin_registry.list_all()]
    duplicates = [n for n in names if names.count(n) > 1]
    assert not duplicates, f"Doppelte Funktionsnamen: {set(duplicates)}"


# ---------------------------------------------------------------------------
# 10. Keine doppelten HTTP-Pfade
# ---------------------------------------------------------------------------


def test_no_duplicate_http_paths():
    seen = {}
    for func in admin_registry.list_all():
        if func.http_method and func.http_path:
            key = f"{func.http_method} {func.http_path}"
            assert key not in seen, f"Doppelte Route: {key} → {func.name} und {seen[key]}"
            seen[key] = func.name


# ---------------------------------------------------------------------------
# 11. Namenskonvention
# ---------------------------------------------------------------------------


def test_naming_convention():
    pattern = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*){1,3}$")
    for func in admin_registry.list_all():
        assert pattern.match(func.name), (
            f"{func.name}: Name entspricht nicht dem Schema 'bereich.aktion'"
        )


# ---------------------------------------------------------------------------
# 12. HTTP-Funktionen → raw_response
# ---------------------------------------------------------------------------


def test_http_functions_have_raw_response():
    # Diese Funktionen verwenden AdminResult-Konvertierung statt raw_response.
    # Sie geben AdminResult zurueck, das automatisch zu JSON konvertiert wird.
    _ALLOWED_NON_RAW = {
        "services.list",
        "services.start",
        "services.stop",
        "services.restart",
        "system.metrics",
        "system.journal",
        "health.cleanup_history",
    }
    for func in admin_registry.list_all():
        if func.http_method and func.http_path and func.name not in _ALLOWED_NON_RAW:
            assert func.raw_response, (
                f"{func.name}: hat HTTP-Mapping aber raw_response=False "
                f"— Frontend erwartet Rohdaten"
            )


# ---------------------------------------------------------------------------
# 13. Alle Funktionen async
# ---------------------------------------------------------------------------


def test_functions_are_async():
    for func in admin_registry.list_all():
        assert asyncio.iscoroutinefunction(func.fn), (
            f"{func.name}: fn ist nicht async (muss async def sein)"
        )


# ---------------------------------------------------------------------------
# 14. Funktions-Signatur: user als erster Parameter
# ---------------------------------------------------------------------------


def test_function_signature_has_user_param():
    for func in admin_registry.list_all():
        sig = inspect.signature(func.fn)
        params = list(sig.parameters.keys())
        assert len(params) >= 1 and params[0] == "user", (
            f"{func.name}: Erster Parameter muss 'user' sein, ist {params}"
        )


# ---------------------------------------------------------------------------
# 15. serialize_all() Vollstaendigkeit
# ---------------------------------------------------------------------------


def test_serialize_all_returns_valid_data():
    serialized = admin_registry.serialize_all()
    assert len(serialized) > 100, f"Nur {len(serialized)} Funktionen — erwartet >100"
    required_keys = {
        "name",
        "category",
        "description",
        "help",
        "parameters",
        "requires_confirmation",
    }
    for entry in serialized:
        missing = required_keys - set(entry.keys())
        assert not missing, f"{entry.get('name', '?')}: Keys fehlen: {missing}"


# ---------------------------------------------------------------------------
# 16. Self-Check ohne Fehler
# ---------------------------------------------------------------------------


def test_self_check_no_errors():
    summary = admin_registry.get_self_check_summary()
    errors = summary.get("errors", [])
    assert not errors, f"Self-Check Fehler: {errors}"
