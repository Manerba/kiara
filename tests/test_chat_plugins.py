"""Hauspatch chat-werkzeug-plugins: zusaetzliche Agent-Tools aus tools.plugins."""

import logging
import sys
import types

import pytest

from kiara.tools.base import BaseTool, ToolDefinition, ToolResult


class _Tool(BaseTool):
    def __init__(self, name, allowed):
        self.name = name
        self.allowed = allowed

    def definition(self):
        return ToolDefinition(name=self.name, description="Test", parameters={"type": "object", "properties": {}},
                              category="dms")

    async def execute(self, params, user_id=None):
        return ToolResult(True, {"allowed": self.allowed}, None, "ok")


@pytest.fixture
def plugin_modul(monkeypatch):
    modul = types.ModuleType("kiara_testtools")
    modul.aufrufe = []

    def chat_tools(*, config, allowed_datasource_ids):
        modul.aufrufe.append(allowed_datasource_ids)
        return [_Tool("plugin_tool", allowed_datasource_ids)]

    def doppelt(*, config, allowed_datasource_ids):
        return [_Tool("plugin_tool", allowed_datasource_ids)]

    def kaputt(*, config, allowed_datasource_ids):
        raise RuntimeError("Plugin defekt")

    modul.chat_tools = chat_tools
    modul.doppelt = doppelt
    modul.kaputt = kaputt
    monkeypatch.setitem(sys.modules, "kiara_testtools", modul)
    return modul


def test_ohne_konfiguration_keine_plugins():
    from kiara.tools.registry import ToolRegistry, register_tool_plugins

    registry = ToolRegistry()
    assert register_tool_plugins(registry, {}, None) == []
    assert register_tool_plugins(registry, {"tools": {"plugins": []}}, None) == []


def test_fehlerhafte_plugins_blockieren_die_uebrigen_nicht(plugin_modul, caplog):
    from kiara.tools.registry import ToolRegistry, register_tool_plugins

    registry = ToolRegistry()
    config = {"tools": {"plugins": ["kiara_gibt_es_nicht", "kiara_testtools:kaputt", "kiara_testtools",
                                    "kiara_testtools:doppelt"]}}
    with caplog.at_level(logging.ERROR, logger="kiara.tools.registry"):
        namen = register_tool_plugins(registry, config, [3, 4])
    assert namen == ["plugin_tool"]
    assert plugin_modul.aufrufe == [[3, 4]]
    fehler = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "kiara_gibt_es_nicht" in fehler and "Plugin defekt" in fehler and "bereits registriert" in fehler
    assert registry.get_tool("plugin_tool").allowed == [3, 4]


def test_plugin_tools_folgen_der_berechtigungsmatrix(plugin_modul):
    from kiara.tools.registry import ToolRegistry, register_tool_plugins

    registry = ToolRegistry(tool_permissions={"Gruppe-A": ["plugin_tool"]})
    register_tool_plugins(registry, {"tools": {"plugins": "kiara_testtools:chat_tools"}}, None)
    assert [d.name for d in registry.list_tools(user_groups=["Gruppe-A"])] == ["plugin_tool"]
    assert registry.list_tools(user_groups=["Gruppe-B"]) == []
    assert [d.name for d in registry.list_tools(user_groups=["*"])] == ["plugin_tool"]


def test_agent_registry_registriert_plugins(plugin_modul, monkeypatch):
    from kiara.web import api_chat

    class _Data:
        embedding_model = "bge-m3"
        k = 5
        filter_type = None
        filter_dir = None

    class _Stub:
        def __init__(self, *args, **kwargs):
            self._store = types.SimpleNamespace(get_collection_name=lambda: "kb")

    monkeypatch.setattr("kiara.retrieval.retriever.HybridRetriever", _Stub)
    monkeypatch.setattr("kiara.retrieval.reranker.create_reranker", lambda config: None)
    config = {"tools": {"enabled": False, "plugins": ["kiara_testtools"]}}
    registry = api_chat._build_agent_tool_registry(config, _Data(), [7])
    assert registry.get_tool("plugin_tool").allowed == [7]
    assert plugin_modul.aufrufe == [[7]]
