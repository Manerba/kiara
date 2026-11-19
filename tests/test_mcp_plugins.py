"""Hauspatch mcp-werkzeug-plugins: zusaetzliche MCP-Werkzeuge aus web.mcp_plugins."""

import asyncio
import logging
import sys
import types

import pytest


@pytest.fixture
def plugin_modul(monkeypatch):
    modul = types.ModuleType("kiara_testplugin")
    modul.aufrufe = []

    def register(mcp, *, config, current_user):
        modul.aufrufe.append((mcp, config, current_user))

        @mcp.tool
        async def testwerkzeug(x: int = 1) -> dict:
            """Nur fuer Tests."""
            return {"x": x, "user": current_user()}

        return ["testwerkzeug"]

    def kaputt(mcp, *, config, current_user):
        raise RuntimeError("Plugin defekt")

    modul.register = register
    modul.kaputt = kaputt
    monkeypatch.setitem(sys.modules, "kiara_testplugin", modul)
    return modul


def test_ohne_konfiguration_keine_plugins():
    from kiara.mcp_server import _register_mcp_plugins

    assert _register_mcp_plugins(object(), {}, lambda: None) == []
    assert _register_mcp_plugins(object(), {"web": {"mcp_plugins": []}}, lambda: None) == []


def test_fehlerhafte_plugins_blockieren_die_uebrigen_nicht(plugin_modul, caplog):
    from fastmcp import FastMCP

    from kiara.mcp_server import _register_mcp_plugins

    mcp = FastMCP("test")
    config = {
        "web": {
            "mcp_plugins": [
                "kiara_gibt_es_nicht:register",
                "kiara_testplugin:kaputt",
                "kiara_testplugin",
            ]
        }
    }
    with caplog.at_level(logging.ERROR, logger="kiara.mcp_server"):
        namen = _register_mcp_plugins(mcp, config, lambda: {"id": 1})
    assert namen == ["testwerkzeug"]
    fehler = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("kiara_gibt_es_nicht" in f and "ModuleNotFoundError" in f for f in fehler)
    assert any("kiara_testplugin:kaputt" in f and "Plugin defekt" in f for f in fehler)
    assert "testwerkzeug" in asyncio.run(mcp.get_tools())


def test_create_app_registriert_plugin_mit_aktuellem_benutzer(plugin_modul):
    from kiara.mcp_server import _create_app, _current_user

    app = _create_app(
        {"web": {"mcp_enabled": True, "mcp_retrieval_enabled": False, "mcp_plugins": "kiara_testplugin:register"}}
    )
    assert app is not None
    [(mcp, config, current_user)] = plugin_modul.aufrufe
    assert config["web"]["mcp_plugins"] == "kiara_testplugin:register"
    werkzeuge = asyncio.run(mcp.get_tools())
    assert {"query", "testwerkzeug"} <= set(werkzeuge)
    token = _current_user.set({"id": 42})
    try:
        assert current_user() == {"id": 42}
    finally:
        _current_user.reset(token)


def test_create_app_ohne_mcp_endpunkt_laedt_keine_plugins(plugin_modul):
    from kiara.mcp_server import _create_app

    _create_app({"web": {"mcp_enabled": False, "mcp_retrieval_enabled": False, "mcp_plugins": ["kiara_testplugin"]}})
    assert plugin_modul.aufrufe == []
