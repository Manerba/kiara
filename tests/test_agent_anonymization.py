"""Tests fuer Agent-Anonymisierung: _create_agent_anonymizer() und Adapter-Integration."""

import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_anthropic_mock():
    """Erstellt ein minimales anthropic-Modul-Mock fuer Tests ohne installiertes Paket."""
    mock_module = types.ModuleType("anthropic")
    mock_module.Anthropic = MagicMock()
    mock_module.RateLimitError = type("RateLimitError", (Exception,), {})
    mock_module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    mock_module.InternalServerError = type("InternalServerError", (Exception,), {})
    mock_module.APITimeoutError = type("APITimeoutError", (Exception,), {})
    mock_module.OverloadedError = type("OverloadedError", (Exception,), {})
    return mock_module


# ===========================================================================
# _create_agent_anonymizer()
# ===========================================================================


class TestCreateAgentAnonymizer:
    def test_ollama_returns_none(self):
        """Ollama-Backend → kein Anonymizer (lokales LLM)."""
        from kiara.web.api_chat import _create_agent_anonymizer

        result = _create_agent_anonymizer(
            config={"anonymization": {"enabled": True}},
            backend={"provider": "ollama", "url": "http://localhost"},
        )
        assert result is None

    def test_ollama_default_provider_returns_none(self):
        """Backend ohne provider-Key → Default 'ollama' → None."""
        from kiara.web.api_chat import _create_agent_anonymizer

        result = _create_agent_anonymizer(
            config={"anonymization": {"enabled": True}},
            backend={"url": "http://localhost"},
        )
        assert result is None

    def test_cloud_disabled_returns_none(self, caplog):
        """Cloud-Backend mit disabled Anonymisierung → None + Log-Info."""
        from kiara.web.api_chat import _create_agent_anonymizer

        with caplog.at_level(logging.INFO):
            result = _create_agent_anonymizer(
                config={"anonymization": {"enabled": False}},
                backend={"provider": "anthropic", "api_key": "sk-test"},
            )
        assert result is None
        assert "ohne Anonymisierung" in caplog.text

    def test_cloud_enabled_returns_pseudonymizer(self):
        """Cloud-Backend mit aktivierter Anonymisierung → Pseudonymizer."""
        from kiara.web.api_chat import _create_agent_anonymizer

        mock_automaton = MagicMock()
        mock_pseudonymizer = MagicMock()

        with (
            patch(
                "kiara.anonymization.stammdaten.build_automaton_from_sources",
                return_value=mock_automaton,
            ),
            patch("kiara.anonymization.anonymizer.Anonymizer", return_value=mock_pseudonymizer),
        ):
            result = _create_agent_anonymizer(
                config={"anonymization": {"enabled": True}},
                backend={"provider": "anthropic", "api_key": "sk-test"},
            )

        assert result is mock_pseudonymizer

    def test_cloud_enabled_exception_raises_runtimeerror(self):
        """Fehler beim Erstellen → RuntimeError (fail-closed analog wrap_with_anonymization)."""
        from kiara.web.api_chat import _create_agent_anonymizer

        with patch(
            "kiara.anonymization.stammdaten.build_automaton_from_sources",
            side_effect=ImportError("Module not found"),
        ):
            with pytest.raises(RuntimeError, match="Anonymisierung ist aktiviert"):
                _create_agent_anonymizer(
                    config={"anonymization": {"enabled": True}},
                    backend={"provider": "openai", "api_key": "sk-test"},
                )

    def test_cloud_default_enabled_returns_pseudonymizer(self):
        """Cloud-Backend mit Default-Config (kein anonymization-Key) → Anonymizer (Default True)."""
        from kiara.web.api_chat import _create_agent_anonymizer

        mock_automaton = MagicMock()
        mock_pseudonymizer = MagicMock()

        with (
            patch(
                "kiara.anonymization.stammdaten.build_automaton_from_sources",
                return_value=mock_automaton,
            ),
            patch("kiara.anonymization.anonymizer.Anonymizer", return_value=mock_pseudonymizer),
        ):
            result = _create_agent_anonymizer(
                config={},
                backend={"provider": "anthropic", "api_key": "sk-test"},
            )
        assert result is mock_pseudonymizer


# ===========================================================================
# Adapter-Integration: Anonymisierung im Lebenszyklus
# ===========================================================================


class TestAdapterAnonymizationIntegration:
    @pytest.fixture
    def mock_anonymizer(self):
        anon = MagicMock()
        anon.anonymize.side_effect = lambda x: x.replace("Hetzner", "[Unternehmen_001]")
        anon.deanonymize.side_effect = lambda x: x.replace("[Unternehmen_001]", "Hetzner")
        return anon

    def test_anthropic_anonymizes_question(self, mock_anonymizer):
        """AnthropicToolAdapter anonymisiert die Frage bei init_conversation."""
        from kiara.generation.tool_adapter import AnthropicToolAdapter
        from kiara.tools.base import ToolDefinition

        mock_anthropic = _make_anthropic_mock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
                anonymizer=mock_anonymizer,
            )

        tools = [
            ToolDefinition(
                name="s", description="S", parameters={"type": "object", "properties": {}}
            )
        ]
        adapter.init_conversation("Hetzner Rechnung", "System", tools)

        assert "[Unternehmen_001]" in adapter._messages[0]["content"]
        assert "Hetzner" not in adapter._messages[0]["content"]

    def test_anthropic_anonymizes_tool_result(self, mock_anonymizer):
        """add_tool_result() anonymisiert den Ergebnistext."""
        from kiara.generation.tool_adapter import AnthropicToolAdapter
        from kiara.tools.base import ToolDefinition

        mock_anthropic = _make_anthropic_mock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
                anonymizer=mock_anonymizer,
            )

        tools = [
            ToolDefinition(
                name="s", description="S", parameters={"type": "object", "properties": {}}
            )
        ]
        adapter.init_conversation("Frage", "System", tools)
        adapter.add_tool_result("search", "Ergebnis von Hetzner", tool_use_id="t1")

        msg = adapter._messages[-1]
        content_text = msg["content"][0]["content"]
        assert "[Unternehmen_001]" in content_text
        assert "Hetzner" not in content_text

    def test_anthropic_deanonymizes_params(self, mock_anonymizer):
        """deanonymize_params() ersetzt Pseudonyme in String-Werten."""
        from kiara.generation.tool_adapter import AnthropicToolAdapter

        mock_anthropic = _make_anthropic_mock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
                anonymizer=mock_anonymizer,
            )

        result = adapter.deanonymize_params(
            {
                "query": "[Unternehmen_001] Rechnung",
                "k": 5,
            }
        )
        assert result["query"] == "Hetzner Rechnung"
        assert result["k"] == 5

    def test_ollama_no_anonymization(self):
        """OllamaToolAdapter hat keinen Anonymizer."""
        from kiara.generation.tool_adapter import OllamaToolAdapter

        with patch("ollama.Client"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        # deanonymize_params gibt neues Dict zurueck (Thread-Safety), aber Inhalt identisch
        params = {"query": "Hetzner Rechnung"}
        result = adapter.deanonymize_params(params)
        assert result == params
        assert result is not params  # Muss neues Dict sein

    def test_consistent_mapping_across_roundtrips(self, mock_anonymizer):
        """Derselbe Pseudonymizer wird fuer alle Roundtrips verwendet."""
        from kiara.generation.tool_adapter import AnthropicToolAdapter
        from kiara.tools.base import ToolDefinition

        mock_anthropic = _make_anthropic_mock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
                anonymizer=mock_anonymizer,
            )

        tools = [
            ToolDefinition(
                name="s", description="S", parameters={"type": "object", "properties": {}}
            )
        ]
        adapter.init_conversation("Hetzner Rechnung", "System", tools)

        # Gleicher Anonymizer fuer Tool-Ergebnis
        adapter.add_tool_result("search", "Weitere Hetzner Daten", tool_use_id="t1")
        msg = adapter._messages[-1]
        assert "[Unternehmen_001]" in msg["content"][0]["content"]

        # De-anonymisierung verwendet gleichen Anonymizer
        result = adapter.deanonymize_params({"query": "[Unternehmen_001]"})
        assert result["query"] == "Hetzner"
