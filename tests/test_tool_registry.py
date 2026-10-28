"""Tests fuer die ToolRegistry (kiara.tools.registry)."""

import pytest

from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Dummy-Tools fuer Registry-Tests
# ---------------------------------------------------------------------------


class DummyPublicTool(BaseTool):
    """Tool ohne Berechtigungsanforderungen (oeffentlich)."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="public_tool",
            description="Ein oeffentliches Tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Eingabe"},
                },
                "required": ["input"],
            },
            requires_confirmation=False,
            required_permissions=[],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        return ToolResult(success=True, data=None, error=None, display_text="ok")


class DummyRestrictedTool(BaseTool):
    """Tool mit Berechtigungsanforderung."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="restricted_tool",
            description="Ein eingeschraenktes Tool",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "integer", "description": "Wert"},
                },
                "required": ["value"],
            },
            requires_confirmation=False,
            required_permissions=["ERP-Leser"],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        return ToolResult(success=True, data=None, error=None, display_text="ok")


class DummyAdminTool(BaseTool):
    """Tool nur fuer Admins."""

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="admin_tool",
            description="Ein Admin-Tool",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            requires_confirmation=True,
            required_permissions=["IT-Admin"],
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        return ToolResult(success=True, data=None, error=None, display_text="ok")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    """Leere Registry."""
    return ToolRegistry()


@pytest.fixture
def populated_registry():
    """Registry mit 3 registrierten Tools."""
    reg = ToolRegistry()
    reg.register(DummyPublicTool())
    reg.register(DummyRestrictedTool())
    reg.register(DummyAdminTool())
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_tool(self, registry):
        """Tool registrieren und ueber get_tool abrufen."""
        tool = DummyPublicTool()
        registry.register(tool)

        retrieved = registry.get_tool("public_tool")
        assert retrieved is not None
        assert retrieved.definition().name == "public_tool"

    def test_get_tool_not_found(self, registry):
        """None fuer ein nicht registriertes Tool."""
        assert registry.get_tool("nichtexistent") is None

    def test_list_tools_no_filter(self, populated_registry):
        """Alle Tools ohne Gruppenfilter — nur oeffentliche Tools."""
        tools = populated_registry.list_tools()
        # Ohne user_groups: nur Tools ohne required_permissions
        names = [t.name for t in tools]
        assert "public_tool" in names
        # Restricted und admin haben required_permissions → nicht sichtbar
        assert "restricted_tool" not in names
        assert "admin_tool" not in names

    def test_list_tools_with_permissions(self, populated_registry):
        """Nur berechtigte Tools werden zurueckgegeben."""
        tools = populated_registry.list_tools(user_groups=["ERP-Leser"])
        names = [t.name for t in tools]
        # Public ist immer sichtbar
        assert "public_tool" in names
        # ERP-Leser hat Zugriff auf restricted_tool
        assert "restricted_tool" in names
        # Admin-Tool braucht IT-Admin
        assert "admin_tool" not in names

    def test_list_tools_admin_wildcard(self, populated_registry):
        """Wildcard '*' Gruppe gibt alle Tools zurueck."""
        tools = populated_registry.list_tools(user_groups=["*"])
        names = [t.name for t in tools]
        assert "public_tool" in names
        assert "restricted_tool" in names
        assert "admin_tool" in names
        assert len(names) == 3

    def test_get_tools_prompt_format(self, populated_registry):
        """Prompt enthaelt TOOL_CALL-Anleitung und Tool-Beschreibungen."""
        prompt = populated_registry.get_tools_prompt(user_groups=["*"])
        assert "TOOL_CALL" in prompt
        assert "Verfuegbare Werkzeuge:" in prompt
        assert "public_tool" in prompt
        assert "restricted_tool" in prompt
        assert "admin_tool" in prompt

    def test_get_tools_prompt_empty_when_no_tools(self, registry):
        """Leerer Prompt wenn keine Tools registriert sind."""
        prompt = registry.get_tools_prompt(user_groups=["*"])
        assert prompt == ""

    def test_register_rejects_duplicate_by_default(self, registry):
        """Erneutes Registrieren mit gleichem Namen wird fail-closed abgelehnt."""
        tool1 = DummyPublicTool()
        tool2 = DummyPublicTool()
        registry.register(tool1)

        with pytest.raises(ValueError, match="public_tool"):
            registry.register(tool2)

        retrieved = registry.get_tool("public_tool")
        assert retrieved is tool1

    def test_register_allows_explicit_overwrite(self, registry):
        """Bewusstes Ersetzen bleibt moeglich, muss aber explizit sein."""
        tool1 = DummyPublicTool()
        tool2 = DummyPublicTool()
        registry.register(tool1)
        registry.register(tool2, allow_overwrite=True)

        retrieved = registry.get_tool("public_tool")
        assert retrieved is tool2

    def test_persisted_permissions_are_authoritative_for_non_builtin_tools(self):
        """Gespeicherte Matrix steuert Sichtbarkeit statt Tool-Defaults."""
        reg = ToolRegistry(tool_permissions={"cn=Sales,dc=test": ["restricted_tool"]})
        reg.register(DummyPublicTool())
        reg.register(DummyRestrictedTool())
        reg.register(DummyAdminTool())

        sales_tools = reg.list_tools(user_groups=["cn=Sales,dc=test"])
        sales_names = [t.name for t in sales_tools]
        assert "restricted_tool" in sales_names
        assert "admin_tool" not in sales_names
        assert next(t for t in sales_tools if t.name == "restricted_tool").required_permissions == [
            "cn=Sales,dc=test"
        ]

        assert [t.name for t in reg.list_tools(user_groups=["ERP-Leser"])] == []

    def test_persisted_group_wildcard_allows_all_non_builtin_tools(self):
        reg = ToolRegistry(tool_permissions={"IT-Admin": ["*"]})
        reg.register(DummyRestrictedTool())
        reg.register(DummyAdminTool())

        names = [t.name for t in reg.list_tools(user_groups=["IT-Admin"])]

        assert names == ["restricted_tool", "admin_tool"]

    def test_persisted_empty_permissions_fail_closed_for_non_builtin_tools(self):
        """Aktive, leere Matrix gibt nur Builtin-Tools frei."""

        class DummyBuiltinTool(DummyPublicTool):
            def definition(self) -> ToolDefinition:
                defn = super().definition()
                defn.category = "builtin"
                return defn

        reg = ToolRegistry(tool_permissions={})
        reg.register(DummyBuiltinTool())
        reg.register(DummyRestrictedTool())

        names = [t.name for t in reg.list_tools(user_groups=["ERP-Leser"])]
        assert names == ["public_tool"]

    def test_permission_provider_is_reloaded_on_each_list(self):
        calls = []

        def provider():
            calls.append(1)
            return {"Team": ["restricted_tool"]} if len(calls) == 1 else {}

        reg = ToolRegistry(permission_provider=provider)
        reg.register(DummyRestrictedTool())

        assert [t.name for t in reg.list_tools(user_groups=["Team"])] == ["restricted_tool"]
        assert reg.list_tools(user_groups=["Team"]) == []

    def test_permission_provider_error_falls_back_to_static_matrix(self):
        """Provider-Ausfall darf statische Tool-Permissions nicht ueberschreiben."""

        def provider():
            raise RuntimeError("db offline")

        reg = ToolRegistry(
            tool_permissions={"Team": ["restricted_tool"]},
            permission_provider=provider,
        )
        reg.register(DummyRestrictedTool())

        assert [t.name for t in reg.list_tools(user_groups=["Team"])] == ["restricted_tool"]

    def test_permission_provider_error_without_static_uses_tool_defaults(self):
        """Ohne statische Matrix gelten bei Provider-Ausfall die Tool-Defaults."""

        def provider():
            raise RuntimeError("db offline")

        reg = ToolRegistry(permission_provider=provider)
        reg.register(DummyRestrictedTool())

        assert [t.name for t in reg.list_tools(user_groups=["ERP-Leser"])] == ["restricted_tool"]
