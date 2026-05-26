"""Tests fuer das Admin-Funktions-Framework (Sprint 01).

Testet AdminResult, AdminFunction, @admin_function Decorator,
AdminFunctionRegistry, Help-System und _coerce_params.
Reine Unit-Tests ohne DB- oder Netzwerk-Abhaengigkeiten.
"""

import asyncio
import logging

from kiara.admin.function import AdminFunction, admin_function
from kiara.admin.help import get_help, list_help, search_help
from kiara.admin.registry import AdminFunctionRegistry
from kiara.admin.result import AdminResult
from kiara.admin.router import _coerce_params
from kiara.tools.base import ToolDefinition

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus (fuer async-Funktionen in Tests)."""
    return asyncio.run(coro)


def _make_registry_with_functions():
    """Erstellt eine frische Registry mit 2 Test-Funktionen."""
    reg = AdminFunctionRegistry()

    async def _ping(user: dict) -> AdminResult:
        return AdminResult.ok("pong")

    async def _delete_ds(user: dict, ds_id: int) -> AdminResult:
        return AdminResult.ok(f"DS {ds_id} geloescht", data={"name": f"DS-{ds_id}"})

    ping = AdminFunction(
        name="system.ping",
        category="system",
        description="Health-Check",
        help="Gibt pong zurueck.",
        examples=["Ping", "Bist du da?"],
        parameters={"type": "object", "properties": {}, "required": []},
        returns="pong",
        requires_confirmation=False,
        confirm_template=None,
        pk_param=None,
        changelog_area=None,
        changelog_action=None,
        http_method="GET",
        http_path="/api/admin/chat/ping",
        fn=_ping,
    )

    delete = AdminFunction(
        name="datasource.delete",
        category="datasources",
        description="Datenquelle loeschen",
        help="Loescht DS vollstaendig inkl. Embeddings.",
        examples=["Loesche Datenquelle 12", "Entferne DS 5"],
        parameters={
            "type": "object",
            "properties": {"ds_id": {"type": "integer", "description": "ID der Datenquelle"}},
            "required": ["ds_id"],
        },
        returns="Bestaetigung mit Anzahl geloeschter Chunks",
        requires_confirmation=True,
        confirm_template="ds {ds_id} loeschen",
        pk_param="ds_id",
        changelog_area="datasources",
        changelog_action="delete",
        http_method="DELETE",
        http_path="/api/admin/datasources/{ds_id}",
        fn=_delete_ds,
    )

    reg.register(ping)
    reg.register(delete)
    return reg, ping, delete


# ---------------------------------------------------------------------------
# AdminResult
# ---------------------------------------------------------------------------


class TestAdminResult:
    def test_ok_minimal(self):
        r = AdminResult.ok("Erfolg")
        assert r.success is True
        assert r.message == "Erfolg"
        assert r.data is None
        assert r.error is None

    def test_ok_with_data(self):
        r = AdminResult.ok("Gespeichert", data={"id": 42})
        assert r.success is True
        assert r.data == {"id": 42}
        assert r.error is None

    def test_fail_minimal(self):
        r = AdminResult.fail("Nicht gefunden")
        assert r.success is False
        assert r.message == "Nicht gefunden"
        assert r.error == "Nicht gefunden"
        assert r.data is None

    def test_fail_with_data(self):
        r = AdminResult.fail("Fehler", data={"details": "Timeout"})
        assert r.success is False
        assert r.data == {"details": "Timeout"}
        assert r.error == "Fehler"


# ---------------------------------------------------------------------------
# AdminFunctionRegistry
# ---------------------------------------------------------------------------


class TestAdminFunctionRegistry:
    def test_register_and_get(self):
        reg, ping, _ = _make_registry_with_functions()
        assert reg.get("system.ping") is ping
        assert reg.get("nicht.vorhanden") is None

    def test_list_all(self):
        reg, _, _ = _make_registry_with_functions()
        assert len(reg.list_all()) == 2

    def test_list_by_category(self):
        reg, _, _ = _make_registry_with_functions()
        system_funcs = reg.list_by_category("system")
        assert len(system_funcs) == 1
        assert system_funcs[0].name == "system.ping"

        ds_funcs = reg.list_by_category("datasources")
        assert len(ds_funcs) == 1
        assert ds_funcs[0].name == "datasource.delete"

        assert reg.list_by_category("gibts_nicht") == []

    def test_categories(self):
        reg, _, _ = _make_registry_with_functions()
        cats = reg.categories()
        assert cats == ["datasources", "system"]  # alphabetisch sortiert

    def test_duplicate_warning(self, caplog):
        reg, ping, _ = _make_registry_with_functions()
        with caplog.at_level(logging.WARNING):
            reg.register(ping)
        assert "ueberschrieben" in caplog.text

    def test_empty_registry(self):
        reg = AdminFunctionRegistry()
        assert reg.list_all() == []
        assert reg.categories() == []
        assert reg.get("foo") is None


# ---------------------------------------------------------------------------
# ToolDefinition-Export
# ---------------------------------------------------------------------------


class TestToolDefinitionExport:
    def test_as_tool_definitions_count(self):
        reg, _, _ = _make_registry_with_functions()
        defs = reg.as_tool_definitions()
        assert len(defs) == 2

    def test_as_tool_definitions_type(self):
        reg, _, _ = _make_registry_with_functions()
        defs = reg.as_tool_definitions()
        assert all(isinstance(d, ToolDefinition) for d in defs)

    def test_as_tool_definitions_mapping(self):
        reg, _, _ = _make_registry_with_functions()
        defs = reg.as_tool_definitions()
        ping_def = next(d for d in defs if d.name == "system.ping")
        assert ping_def.description == "Health-Check"
        assert ping_def.requires_confirmation is False
        assert ping_def.category == "system"
        assert ping_def.parameters == {"type": "object", "properties": {}, "required": []}

    def test_as_tool_definitions_confirmation_flag(self):
        reg, _, _ = _make_registry_with_functions()
        defs = reg.as_tool_definitions()
        delete_def = next(d for d in defs if d.name == "datasource.delete")
        assert delete_def.requires_confirmation is True

    def test_as_tool_definitions_empty(self):
        reg = AdminFunctionRegistry()
        assert reg.as_tool_definitions() == []

    def test_get_tools_prompt_contains_functions(self):
        reg, _, _ = _make_registry_with_functions()
        prompt = reg.get_tools_prompt()
        assert "system.ping" in prompt
        assert "datasource.delete" in prompt
        assert "TOOL_CALL" in prompt

    def test_get_tools_prompt_empty(self):
        reg = AdminFunctionRegistry()
        assert reg.get_tools_prompt() == ""


# ---------------------------------------------------------------------------
# Help-System
# ---------------------------------------------------------------------------


class TestHelpSystem:
    def test_get_help_all_fields(self):
        _, ping, _ = _make_registry_with_functions()
        h = get_help(ping)
        assert h["name"] == "system.ping"
        assert h["category"] == "system"
        assert h["short"] == "Health-Check"
        assert h["long"] == "Gibt pong zurueck."
        assert h["parameters"] == {"type": "object", "properties": {}, "required": []}
        assert h["examples"] == ["Ping", "Bist du da?"]
        assert h["returns"] == "pong"
        assert h["requires_confirmation"] is False

    def test_get_help_via_registry(self):
        reg, _, _ = _make_registry_with_functions()
        h = reg.get_help("system.ping")
        assert h is not None
        assert h["name"] == "system.ping"

    def test_get_help_not_found(self):
        reg, _, _ = _make_registry_with_functions()
        assert reg.get_help("gibts.nicht") is None

    def test_list_help_all(self):
        reg, _, _ = _make_registry_with_functions()
        items = list_help(reg)
        assert len(items) == 2

    def test_list_help_by_category(self):
        reg, _, _ = _make_registry_with_functions()
        items = list_help(reg, category="system")
        assert len(items) == 1
        assert items[0]["name"] == "system.ping"

    def test_search_help_by_name(self):
        reg, _, _ = _make_registry_with_functions()
        results = search_help(reg, "ping")
        assert len(results) == 1
        assert results[0].name == "system.ping"

    def test_search_help_by_description(self):
        reg, _, _ = _make_registry_with_functions()
        results = search_help(reg, "loeschen")
        assert len(results) == 1
        assert results[0].name == "datasource.delete"

    def test_search_help_by_example(self):
        reg, _, _ = _make_registry_with_functions()
        results = search_help(reg, "Bist du da")
        assert len(results) == 1
        assert results[0].name == "system.ping"

    def test_search_help_case_insensitive(self):
        reg, _, _ = _make_registry_with_functions()
        results = search_help(reg, "PING")
        assert len(results) == 1

    def test_search_help_no_match(self):
        reg, _, _ = _make_registry_with_functions()
        assert search_help(reg, "xyznonexistent") == []

    def test_search_via_registry(self):
        reg, _, _ = _make_registry_with_functions()
        results = reg.search("Datenquelle")
        assert len(results) == 1
        assert results[0].name == "datasource.delete"


# ---------------------------------------------------------------------------
# @admin_function Decorator
# ---------------------------------------------------------------------------


class TestAdminFunctionDecorator:
    def test_decorator_registers_function(self):
        # system.ping wird beim Import von system.py registriert
        import kiara.admin.functions.system  # noqa: F401
        from kiara.admin.registry import admin_registry

        assert admin_registry.get("system.ping") is not None

    def test_decorator_preserves_original(self):
        """Dekorierte Funktion bleibt direkt aufrufbar."""
        from kiara.admin.functions.system import system_ping

        result = _run(system_ping(user={"id": 1, "username": "test"}))
        assert isinstance(result, AdminResult)
        assert result.success is True
        assert result.message == "pong"

    def test_decorator_http_splitting(self):
        """http-Tuple wird in http_method und http_path aufgesplittet."""
        import kiara.admin.functions.system  # noqa: F401
        from kiara.admin.registry import admin_registry

        f = admin_registry.get("system.ping")
        assert f.http_method == "GET"
        assert f.http_path == "/api/admin/chat/ping"

    def test_decorator_without_http(self):
        """Funktion ohne http-Tuple hat None fuer method/path."""
        reg = AdminFunctionRegistry()

        @admin_function(
            name="test.nohttp",
            category="test",
            description="Ohne HTTP",
            help="Nur Chat.",
            examples=["Test"],
            parameters={"type": "object", "properties": {}, "required": []},
            returns="ok",
        )
        async def _nohttp(user: dict) -> AdminResult:
            return AdminResult.ok("ok")

        # Der Decorator registriert in der globalen Registry,
        # daher pruefen wir direkt die Funktion
        from kiara.admin.registry import admin_registry

        f = admin_registry.get("test.nohttp")
        assert f is not None
        assert f.http_method is None
        assert f.http_path is None


# ---------------------------------------------------------------------------
# _coerce_params
# ---------------------------------------------------------------------------


class TestCoerceParams:
    def _schema(self, props: dict) -> dict:
        return {"type": "object", "properties": props, "required": []}

    def test_string_to_integer(self):
        schema = self._schema({"ds_id": {"type": "integer"}})
        result = _coerce_params({"ds_id": "42"}, schema)
        assert result["ds_id"] == 42
        assert isinstance(result["ds_id"], int)

    def test_string_to_number(self):
        schema = self._schema({"threshold": {"type": "number"}})
        result = _coerce_params({"threshold": "0.75"}, schema)
        assert result["threshold"] == 0.75
        assert isinstance(result["threshold"], float)

    def test_string_to_boolean_true(self):
        schema = self._schema({"active": {"type": "boolean"}})
        for truthy in ("true", "True", "TRUE", "1", "yes"):
            result = _coerce_params({"active": truthy}, schema)
            assert result["active"] is True, f"'{truthy}' sollte True ergeben"

    def test_string_to_boolean_false(self):
        schema = self._schema({"active": {"type": "boolean"}})
        for falsy in ("false", "0", "no"):
            result = _coerce_params({"active": falsy}, schema)
            assert result["active"] is False, f"'{falsy}' sollte False ergeben"

    def test_non_string_passthrough(self):
        """Bereits korrekt typisierte Werte (z.B. aus JSON-Body) bleiben unveraendert."""
        schema = self._schema({"ds_id": {"type": "integer"}})
        result = _coerce_params({"ds_id": 42}, schema)
        assert result["ds_id"] == 42

    def test_invalid_integer_stays_string(self):
        """Bei fehlgeschlagener Konvertierung bleibt der Originalwert erhalten."""
        schema = self._schema({"ds_id": {"type": "integer"}})
        result = _coerce_params({"ds_id": "abc"}, schema)
        assert result["ds_id"] == "abc"

    def test_invalid_number_stays_string(self):
        schema = self._schema({"val": {"type": "number"}})
        result = _coerce_params({"val": "not_a_number"}, schema)
        assert result["val"] == "not_a_number"

    def test_unknown_param_passthrough(self):
        """Parameter die nicht im Schema sind werden durchgereicht."""
        schema = self._schema({"ds_id": {"type": "integer"}})
        result = _coerce_params({"ds_id": "5", "extra": "foo"}, schema)
        assert result["ds_id"] == 5
        assert result["extra"] == "foo"

    def test_empty_params(self):
        schema = self._schema({})
        assert _coerce_params({}, schema) == {}

    def test_string_type_unchanged(self):
        schema = self._schema({"name": {"type": "string"}})
        result = _coerce_params({"name": "test"}, schema)
        assert result["name"] == "test"


# ---------------------------------------------------------------------------
# system.ping End-to-End
# ---------------------------------------------------------------------------


class TestSystemPing:
    def test_ping_returns_pong(self):
        from kiara.admin.functions.system import system_ping

        result = _run(system_ping(user={"id": 1, "username": "admin"}))
        assert result.success is True
        assert result.message == "pong"
        assert result.data is None
        assert result.error is None
