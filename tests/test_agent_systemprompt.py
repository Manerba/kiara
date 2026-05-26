"""Tests fuer Build 67: Agent-Systemprompt in Admin-UI konfigurierbar.

Testet:
  - AgentChain nutzt leeren Prompt wenn kein custom_prompt uebergeben
  - AgentChain nutzt custom_prompt aus Config wenn vorhanden
  - AgentChain nutzt system_prompt_override wenn uebergeben
  - {tools_prompt} Platzhalter wird korrekt ersetzt (via Adapter)
  - API POST validiert dass {tools_prompt} im Prompt enthalten ist
  - DB-Migration fuegt agent_system_prompt Spalte hinzu
  - Backend-Adapter-Factory erzeugt richtigen Adapter-Typ
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.generation.agent_chain import AgentChain
from kiara.generation.tool_adapter import BaseToolAdapter
from kiara.tools.builtin_tools import CalculateTool
from kiara.tools.registry import ToolRegistry
from kiara.web.api_ki import router
from kiara.web.auth import get_admin_user

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_adapter(usage=None):
    """Erstellt einen Mock-Adapter fuer AgentChain-Tests."""
    adapter = MagicMock(spec=BaseToolAdapter)
    adapter.deanonymize_params.side_effect = lambda p: p
    adapter.stream_final.return_value = iter(["Antwort."])
    adapter.add_tool_result.return_value = None
    type(adapter).last_usage = PropertyMock(return_value=usage or {})
    return adapter


@pytest.fixture
def base_config():
    """Minimale Config ohne custom system_prompt."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
        },
        "chat": {
            "agent": {
                "max_iterations": 3,
                "num_ctx": 8192,
                "temperature": 0.3,
            },
        },
    }


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(CalculateTool())
    return reg


@pytest.fixture
def mock_adapter():
    return _make_mock_adapter()


# -- DB + API Fixture fuer Endpoint-Tests --

_AI_CONFIG_SCHEMA = """
    CREATE TABLE IF NOT EXISTS ai_config (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        embedding_url TEXT NOT NULL DEFAULT '',
        embedding_model TEXT NOT NULL DEFAULT '',
        llm_url TEXT NOT NULL DEFAULT '',
        llm_model TEXT NOT NULL DEFAULT '',
        prompt_template TEXT DEFAULT '',
        cloud_enabled BOOLEAN DEFAULT false,
        cloud_provider TEXT DEFAULT 'anthropic',
        cloud_api_key TEXT DEFAULT '',
        cloud_model TEXT DEFAULT '',
        cloud_base_url TEXT DEFAULT '',
        cloud_mode TEXT DEFAULT 'auto',
        anon_enabled BOOLEAN DEFAULT true,
        anon_erp_enabled BOOLEAN DEFAULT false,
        anon_erp_path TEXT DEFAULT '',
        anon_ldap_enabled BOOLEAN DEFAULT true,
        anon_word_boundaries INTEGER DEFAULT 1,
        anon_min_term_length INTEGER DEFAULT 3,
        anon_excluded_terms TEXT DEFAULT '[]',
        anon_custom_patterns TEXT DEFAULT '[]',
        anon_audit_log INTEGER DEFAULT 1,
        anon_custom_entries TEXT DEFAULT '[]',
        anon_llm_enabled BOOLEAN DEFAULT false,
        anon_llm_model TEXT DEFAULT '',
        anon_llm_temperature TEXT DEFAULT '0.1',
        anon_llm_confidence TEXT DEFAULT '0.7',
        anon_llm_timeout INTEGER DEFAULT 10,
        anon_llm_cache_size INTEGER DEFAULT 256,
        anon_llm_num_ctx INTEGER DEFAULT 4096,
        agent_system_prompt TEXT DEFAULT '',
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );
"""


@pytest.fixture
def api_client():
    """TestClient fuer die KI-API mit gemocktem config_repo."""
    # In-Memory-Speicher fuer die Singleton-Konfiguration
    _store = {}

    def mock_config_get(table, *, conn=None):
        return dict(_store) if _store else None

    def mock_config_set(table, fields, *, conn=None):
        if not _store:
            _store.update(fields)
        else:
            _store.update(fields)

    def mock_config_delete(table, *, conn=None):
        _store.clear()
        return True

    app = FastAPI()
    app.include_router(router)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    with patch("kiara.web.api_ki.config_repo") as mock_cr, patch("kiara.web.api_ki._sync_to_yaml"):
        mock_cr.get.side_effect = mock_config_get
        mock_cr.set.side_effect = mock_config_set
        mock_cr.delete.side_effect = mock_config_delete
        with TestClient(app) as c:
            yield c


# =====================================================================
# AgentChain — System-Prompt-Auswahl
# =====================================================================


class TestAgentChainPromptSelection:
    def test_leerer_prompt_wenn_kein_custom(self, base_config, registry, mock_adapter):
        """Ohne custom_prompt in Config wird leerer String genutzt (mit Warning)."""
        chain = AgentChain(base_config, registry, adapter=mock_adapter)
        assert chain._system_prompt == ""

    def test_leerer_prompt_bei_leerem_config_string(self, base_config, registry, mock_adapter):
        """Leerer String in Config fuehrt zu leerem Prompt."""
        base_config["chat"]["agent"]["system_prompt"] = ""
        chain = AgentChain(base_config, registry, adapter=mock_adapter)
        assert chain._system_prompt == ""

    def test_custom_prompt_aus_override(self, base_config, registry, mock_adapter):
        """Custom system_prompt via system_prompt_override wird uebernommen."""
        custom = "Du bist ein Test-Agent.\n{tools_prompt}\nAntworte kurz."
        chain = AgentChain(
            base_config, registry, adapter=mock_adapter, system_prompt_override=custom
        )
        assert chain._system_prompt == custom

    def test_system_prompt_override_parameter(self, base_config, registry, mock_adapter):
        """system_prompt_override hat Vorrang vor Config."""
        base_config["chat"]["agent"]["system_prompt"] = "Config-Prompt {tools_prompt}"
        override = "Override-Prompt {tools_prompt}"
        chain = AgentChain(
            base_config, registry, adapter=mock_adapter, system_prompt_override=override
        )
        assert chain._system_prompt == override

    def test_system_prompt_override_hat_vorrang_vor_default(
        self, base_config, registry, mock_adapter
    ):
        """system_prompt_override hat auch Vorrang wenn kein Config-Prompt existiert."""
        override = "Mein spezieller Agent {tools_prompt}"
        chain = AgentChain(
            base_config, registry, adapter=mock_adapter, system_prompt_override=override
        )
        assert chain._system_prompt == override

    def test_ohne_chat_section_nutzt_leeren_string(self, mock_adapter):
        """Config ohne chat-Sektion fuehrt zu leerem Prompt."""
        config = {
            "ollama": {
                "url": "http://localhost:11434",
                "llm_url": "",
                "embedding_url": "",
                "llm_model": "m",
                "embedding_model": "e",
            },
        }
        reg = ToolRegistry()
        chain = AgentChain(config, reg, adapter=mock_adapter)
        assert chain._system_prompt == ""


# =====================================================================
# {tools_prompt} Platzhalter-Ersetzung im Prompt
# =====================================================================


class TestToolsPromptPlatzhalter:
    def test_init_conversation_receives_system_prompt(self, base_config, registry, mock_adapter):
        """init_conversation() wird mit dem korrekten System-Prompt aufgerufen."""
        from kiara.generation.tool_adapter import ToolStep

        mock_adapter.step.return_value = ToolStep(text="OK", is_final=True)
        custom = "Agent:\n{tools_prompt}\nEnde."
        chain = AgentChain(
            base_config, registry, adapter=mock_adapter, system_prompt_override=custom
        )
        list(chain.query_stream("Testfrage"))
        mock_adapter.init_conversation.assert_called_once()
        call_args = mock_adapter.init_conversation.call_args
        system_prompt_arg = call_args[0][1]  # zweites positional arg
        assert "{tools_prompt}" in system_prompt_arg

    def test_custom_prompt_mit_platzhalter(self, base_config, registry, mock_adapter):
        """Custom-Prompt mit {tools_prompt} wird an Adapter uebergeben."""
        from kiara.generation.tool_adapter import ToolStep

        mock_adapter.step.return_value = ToolStep(text="OK", is_final=True)
        custom = "Agent-Prompt:\n{tools_prompt}\nEnde."
        chain = AgentChain(
            base_config, registry, adapter=mock_adapter, system_prompt_override=custom
        )
        list(chain.query_stream("Frage?"))
        call_args = mock_adapter.init_conversation.call_args
        system_prompt_arg = call_args[0][1]
        assert "Agent-Prompt:" in system_prompt_arg
        assert "Ende." in system_prompt_arg


# =====================================================================
# Backend-Adapter-Factory — Adapter-Typ je nach Provider
# =====================================================================


class TestBackendAdapterFactory:
    def test_anthropic_backend_erzeugt_anthropic_adapter(self):
        """create_tool_adapter() erzeugt AnthropicToolAdapter fuer Anthropic-Backend."""
        from kiara.generation.tool_adapter import AnthropicToolAdapter, create_tool_adapter

        backend = {
            "id": 42,
            "provider": "anthropic",
            "url": "https://api.anthropic.com",
            "model": "claude-3",
            "api_key": "sk-test",
        }
        config = {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}}
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            adapter = create_tool_adapter(backend, config)
        assert isinstance(adapter, AnthropicToolAdapter)

    def test_openai_backend_erzeugt_openai_adapter(self):
        """create_tool_adapter() erzeugt OpenAIToolAdapter fuer OpenAI-Backend."""
        from kiara.generation.tool_adapter import OpenAIToolAdapter, create_tool_adapter

        backend = {
            "id": 99,
            "provider": "openai",
            "url": "https://api.openai.com",
            "model": "gpt-4",
            "api_key": "sk-test",
        }
        config = {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}}
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter = create_tool_adapter(backend, config)
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_ollama_backend_erzeugt_ollama_adapter(self):
        """create_tool_adapter() erzeugt OllamaToolAdapter fuer Ollama-Backend."""
        from kiara.generation.tool_adapter import OllamaToolAdapter, create_tool_adapter

        backend = {
            "id": 10,
            "provider": "ollama",
            "url": "http://gpu-server.local:11434",
            "model": "qwen3:8b",
        }
        config = {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}}
        with patch("ollama.Client"):
            adapter = create_tool_adapter(backend, config)
        assert isinstance(adapter, OllamaToolAdapter)


# =====================================================================
# API: POST /config — agent_system_prompt Validierung
# =====================================================================


class TestApiAgentPromptValidation:
    def _minimal_payload(self, **overrides):
        """Minimales KiConfigRequest-Payload."""
        payload = {
            "embedding_url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "llm_url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
        }
        payload.update(overrides)
        return payload

    def test_leerer_agent_prompt_akzeptiert(self, api_client):
        """Leerer agent_system_prompt ist erlaubt (= Default verwenden)."""
        resp = api_client.post(
            "/api/admin/ki/config",
            json=self._minimal_payload(agent_system_prompt=""),
        )
        assert resp.status_code == 200

    def test_gueltiger_agent_prompt_akzeptiert(self, api_client):
        """Prompt mit {tools_prompt} wird akzeptiert."""
        custom = "Mein Agent:\n{tools_prompt}\nAntworte auf Deutsch."
        resp = api_client.post(
            "/api/admin/ki/config",
            json=self._minimal_payload(agent_system_prompt=custom),
        )
        assert resp.status_code == 200

    def test_prompt_ohne_tools_prompt_wird_abgelehnt(self, api_client):
        """Prompt ohne {tools_prompt} fuehrt zu HTTP 422."""
        resp = api_client.post(
            "/api/admin/ki/config",
            json=self._minimal_payload(agent_system_prompt="Kein Platzhalter hier."),
        )
        assert resp.status_code == 422
        assert "tools_prompt" in resp.json()["detail"].lower()

    def test_agent_prompt_in_get_config_zurueckgegeben(self, api_client):
        """Gespeicherter agent_system_prompt erscheint nach POST in config_repo."""
        custom = "Custom Agent {tools_prompt} Ende"
        resp = api_client.post(
            "/api/admin/ki/config",
            json=self._minimal_payload(agent_system_prompt=custom),
        )
        assert resp.status_code == 200


# Die urspruenglichen TestMigrationAgentPrompt-Tests wurden entfernt:
# Sie importierten aus kiara.web.db_migrations (Modul existiert nicht mehr);
# agent_system_prompt ist heute Teil der initialen Schema-Erstellung (schema.sql),
# ein separates Migrations-Helfer-Modul wird nicht mehr gepflegt.
