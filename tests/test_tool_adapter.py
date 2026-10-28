"""Tests fuer kiara.generation.tool_adapter."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kiara.generation.tool_adapter import (
    AnthropicToolAdapter,
    OllamaToolAdapter,
    OpenAIToolAdapter,
    ToolAdapterError,
    ToolStep,
    create_tool_adapter,
)
from kiara.tools.base import ToolDefinition

# ===========================================================================
# ToolStep
# ===========================================================================


class TestToolStep:
    def test_default_values(self):
        step = ToolStep()
        assert step.tool_call is None
        assert step.text == ""
        assert step.tool_use_id is None
        assert step.is_final is False

    def test_tool_call_step(self):
        step = ToolStep(tool_call=("search", {"query": "test"}), text="Reasoning")
        assert step.tool_call == ("search", {"query": "test"})
        assert step.text == "Reasoning"
        assert step.is_final is False

    def test_final_step(self):
        step = ToolStep(text="Die Antwort ist 42.", is_final=True)
        assert step.tool_call is None
        assert step.is_final is True

    def test_with_tool_use_id(self):
        step = ToolStep(tool_call=("calc", {}), tool_use_id="toolu_123")
        assert step.tool_use_id == "toolu_123"


# ===========================================================================
# ToolAdapterError
# ===========================================================================


class TestToolAdapterError:
    def test_retryable_default_false(self):
        err = ToolAdapterError("fail")
        assert err.retryable is False
        assert str(err) == "fail"

    def test_retryable_true(self):
        err = ToolAdapterError("rate limit", retryable=True)
        assert err.retryable is True

    def test_is_exception(self):
        assert issubclass(ToolAdapterError, Exception)


# ===========================================================================
# OllamaToolAdapter — Parsing
# ===========================================================================


class TestOllamaToolAdapterParsing:
    def test_parse_double_quoted(self):
        result = OllamaToolAdapter._parse_tool_call('TOOL_CALL: search(query="Python")')
        assert result is not None
        assert result[0] == "search"
        assert result[1]["query"] == "Python"

    def test_parse_single_quoted(self):
        result = OllamaToolAdapter._parse_tool_call("TOOL_CALL: calc(expression='2+3')")
        assert result is not None
        assert result[1]["expression"] == "2+3"

    def test_parse_unquoted_int(self):
        result = OllamaToolAdapter._parse_tool_call('TOOL_CALL: search(query="test", k=5)')
        assert result is not None
        assert result[1]["k"] == 5
        assert isinstance(result[1]["k"], int)

    def test_parse_no_tool_call(self):
        result = OllamaToolAdapter._parse_tool_call("Die Antwort ist 42.")
        assert result is None

    def test_parse_case_insensitive(self):
        result = OllamaToolAdapter._parse_tool_call('tool_call: calc(expression="1")')
        assert result is not None

    def test_parse_empty_params(self):
        result = OllamaToolAdapter._parse_tool_call("TOOL_CALL: list_datasources()")
        assert result is not None
        assert result[0] == "list_datasources"
        assert result[1] == {}

    def test_parse_with_surrounding_text(self):
        text = 'Ich suche jetzt.\nTOOL_CALL: search(query="Hetzner")\nDas war es.'
        result = OllamaToolAdapter._parse_tool_call(text)
        assert result is not None
        assert result[0] == "search"

    def test_double_quotes_preferred_over_single(self):
        result = OllamaToolAdapter._parse_tool_call("""TOOL_CALL: tool(a="double", b='single')""")
        assert result is not None
        assert result[1]["a"] == "double"
        assert result[1]["b"] == "single"

    def test_parse_dotted_admin_function_name(self):
        """Dotted AdminFunction-Namen wie system.events.list muessen geparst werden.

        Regression fuer TODO_FEATURE_ADMIN_SYSTEM_LOG_V18.md Abschnitt "Ollama
        Tool-Name-Parser": Der alte Regex `\\w+` hat dotted Namen verschluckt.
        """
        result = OllamaToolAdapter._parse_tool_call(
            'TOOL_CALL: system.events.list(event_types="LOGIN_FAILED")'
        )
        assert result is not None
        assert result[0] == "system.events.list"
        assert result[1]["event_types"] == "LOGIN_FAILED"

    def test_undotted_names_still_lowercased(self):
        """Undotted Tool-Namen bleiben lowercase-normalisiert (Bestandsverhalten)."""
        result = OllamaToolAdapter._parse_tool_call('TOOL_CALL: SEARCH(query="x")')
        assert result is not None
        assert result[0] == "search"

    def test_parse_empty_string(self):
        assert OllamaToolAdapter._parse_tool_call("") is None

    # -- Klammerbalancierung --

    def test_parse_nested_parens_in_quoted_value(self):
        """Klammern in gequoteten Werten werden korrekt uebersprungen."""
        result = OllamaToolAdapter._parse_tool_call('TOOL_CALL: search(query="Server (Rack 5)")')
        assert result is not None
        assert result[0] == "search"
        assert result[1]["query"] == "Server (Rack 5)"

    def test_parse_math_expression_quoted(self):
        """Mathematische Ausdruecke mit Klammern in Quotes."""
        result = OllamaToolAdapter._parse_tool_call('TOOL_CALL: calculate(expression="sqrt(144)")')
        assert result is not None
        assert result[1]["expression"] == "sqrt(144)"

    def test_parse_complex_math_quoted(self):
        """Mehrfach verschachtelte Klammern in Quotes."""
        result = OllamaToolAdapter._parse_tool_call(
            'TOOL_CALL: calculate(expression="(2+3)*(4+5)")'
        )
        assert result is not None
        assert result[1]["expression"] == "(2+3)*(4+5)"

    def test_parse_unquoted_math_expression(self):
        """Unquotete Ausdruecke mit Klammern (LLM quotiert nicht immer)."""
        result = OllamaToolAdapter._parse_tool_call("TOOL_CALL: calculate(expression=sqrt(144))")
        assert result is not None
        assert result[0] == "calculate"
        # params_str ist "expression=sqrt(144)" — kein Quoted-Match,
        # aber der Klammer-Inhalt wird korrekt extrahiert
        assert result[1] == {}  # kein Regex-Match fuer unquotete Strings

    def test_parse_multiple_nested_parens_unquoted(self):
        """Mehrere Klammerebenen ohne Quotes."""
        result = OllamaToolAdapter._parse_tool_call("TOOL_CALL: calculate(expression=(2+3)*(4+5))")
        assert result is not None
        assert result[0] == "calculate"

    def test_parse_unbalanced_parens_returns_none(self):
        """Unbalancierte Klammern geben None zurueck."""
        result = OllamaToolAdapter._parse_tool_call('TOOL_CALL: search(query="test"')
        assert result is None

    def test_parse_parens_in_surrounding_text(self):
        """Klammern im umgebenden Text stoeren nicht."""
        text = (
            "Ich suche (jetzt).\n"
            'TOOL_CALL: search(query="Vertrag (2024)")\n'
            "Das war es (glaube ich)."
        )
        result = OllamaToolAdapter._parse_tool_call(text)
        assert result is not None
        assert result[1]["query"] == "Vertrag (2024)"

    def test_parse_single_quoted_with_parens(self):
        """Klammern in Single-Quoted Werten."""
        result = OllamaToolAdapter._parse_tool_call(
            "TOOL_CALL: search(query='Firewall (Port 443)')"
        )
        assert result is not None
        assert result[1]["query"] == "Firewall (Port 443)"

    def test_parse_mixed_params_with_parens(self):
        """Gequoteter Wert mit Klammern + unquoted Integer."""
        result = OllamaToolAdapter._parse_tool_call(
            'TOOL_CALL: search(query="Server (Rack 5)", k=10)'
        )
        assert result is not None
        assert result[1]["query"] == "Server (Rack 5)"
        assert result[1]["k"] == 10

    def test_parse_quoted_value_does_not_create_extra_params(self):
        """name=value innerhalb eines gequoteten Strings bleibt Teil des Werts."""
        result = OllamaToolAdapter._parse_tool_call('TOOL_CALL: search(query="find limit=10", k=3)')
        assert result is not None
        assert result[1] == {"query": "find limit=10", "k": 3}


# ===========================================================================
# _extract_balanced_params (Modul-Level-Hilfsfunktion)
# ===========================================================================


class TestExtractBalancedParams:
    def test_simple(self):
        from kiara.generation.adapters.base import _extract_balanced_params

        assert _extract_balanced_params('query="test")', 0) == 'query="test"'

    def test_nested(self):
        from kiara.generation.adapters.base import _extract_balanced_params

        assert _extract_balanced_params("expr=sqrt(144))", 0) == "expr=sqrt(144)"

    def test_deeply_nested(self):
        from kiara.generation.adapters.base import _extract_balanced_params

        assert _extract_balanced_params("e=(a+(b*c)))", 0) == "e=(a+(b*c))"

    def test_quoted_parens_ignored(self):
        from kiara.generation.adapters.base import _extract_balanced_params

        assert _extract_balanced_params('q="a(b)c")', 0) == 'q="a(b)c"'

    def test_unbalanced_returns_none(self):
        from kiara.generation.adapters.base import _extract_balanced_params

        assert _extract_balanced_params('query="test"', 0) is None

    def test_empty(self):
        from kiara.generation.adapters.base import _extract_balanced_params

        assert _extract_balanced_params(")", 0) == ""


# ===========================================================================
# OllamaToolAdapter — Vollstaendiger Lebenszyklus
# ===========================================================================


class TestOllamaToolAdapterLifecycle:
    @pytest.fixture
    def config(self):
        return {
            "chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}},
        }

    @pytest.fixture
    def backend(self):
        return {"url": "http://localhost:11434", "model": "qwen3:8b", "provider": "ollama"}

    @pytest.fixture
    def tools(self):
        return [
            ToolDefinition(
                name="search",
                description="Suche",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="calc",
                description="Rechne",
                parameters={
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
            ),
        ]

    @pytest.fixture
    def adapter(self, backend, config):
        with patch("ollama.Client") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            adpt = OllamaToolAdapter(backend, config)
            adpt._client = mock_client
            yield adpt

    def test_init_conversation_stores_state(self, adapter, tools):
        adapter.init_conversation("Was ist Python?", "System Prompt", tools, "History")
        assert adapter._question == "Was ist Python?"
        assert adapter._history_block == "History"
        assert "search" in adapter._tools_prompt
        assert "calc" in adapter._tools_prompt

    def test_step_with_tool_call(self, adapter, tools):
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {"response": 'TOOL_CALL: search(query="Python")'}
        step = adapter.step()
        assert step.tool_call is not None
        assert step.tool_call[0] == "search"
        assert step.tool_call[1]["query"] == "Python"
        assert step.is_final is False

    def test_step_disables_thinking(self, adapter, tools):
        """step() setzt think=False fuer schnelleres Tool-Calling."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {"response": "Antwort"}
        adapter.step()
        call_kwargs = adapter._client.generate.call_args
        assert call_kwargs.kwargs.get("think") is False

    def test_stream_final_default_thinking_disabled(self, adapter, tools):
        """stream_final() setzt think=False per Default (schnelle Antwort)."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = iter(
            [
                {
                    "response": "Eine ausfuehrliche Antwort.",
                    "done": True,
                    "prompt_eval_count": 50,
                    "eval_count": 20,
                    "total_duration": 500_000_000,
                },
            ]
        )
        list(adapter.stream_final())
        call_kwargs = adapter._client.generate.call_args
        assert call_kwargs.kwargs.get("think") is False

    def test_stream_final_thinking_enabled_when_requested(self, adapter, tools):
        """stream_final(think=True) aktiviert Thinking fuer Reasoning."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = iter(
            [
                {"response": "<think>\nIch denke nach.\n</think>\n"},
                {
                    "response": "Eine ausfuehrliche Antwort.",
                    "done": True,
                    "prompt_eval_count": 50,
                    "eval_count": 20,
                    "total_duration": 500_000_000,
                },
            ]
        )
        list(adapter.stream_final(think=True))
        call_kwargs = adapter._client.generate.call_args
        assert call_kwargs.kwargs.get("think") is True

    def test_step_without_tool_call(self, adapter, tools):
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {"response": "Die Antwort ist 42."}
        step = adapter.step()
        assert step.tool_call is None
        assert step.text == "Die Antwort ist 42."
        assert step.is_final is False  # Ollama: immer False

    def test_step_text_contains_reasoning(self, adapter, tools):
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {
            "response": 'Ich suche jetzt.\nTOOL_CALL: search(query="test")'
        }
        step = adapter.step()
        assert "Ich suche jetzt" in step.text

    def test_step_strips_think_tags_with_tool_call(self, adapter, tools):
        """<think>-Bloecke werden entfernt, TOOL_CALL ausserhalb wird erkannt."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {
            "response": (
                "<think>\nIch ueberlege was zu tun ist...\n</think>\n"
                'TOOL_CALL: search(query="Hetzner Rechnung")'
            )
        }
        step = adapter.step()
        assert step.tool_call is not None
        assert step.tool_call[0] == "search"
        assert step.tool_call[1]["query"] == "Hetzner Rechnung"
        assert "<think>" not in step.text

    def test_step_strips_think_tags_no_tool_call(self, adapter, tools):
        """<think>-Bloecke werden entfernt, sauberer Text bleibt uebrig."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {
            "response": (
                "<think>\nDie Frage kann nicht beantwortet werden.\n</think>\n"
                "Ich habe keine Informationen dazu."
            )
        }
        step = adapter.step()
        assert step.tool_call is None
        assert "<think>" not in step.text
        assert "Ich habe keine Informationen" in step.text

    def test_step_strips_unclosed_think_tag(self, adapter, tools):
        """Nicht geschlossene <think>-Tags werden ebenfalls entfernt."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {
            "response": "<think>\nEndloses Denken ohne Abschluss..."
        }
        step = adapter.step()
        assert "<think>" not in step.text
        assert step.tool_call is None

    def test_step_no_think_tags_unchanged(self, adapter, tools):
        """Ohne <think>-Tags bleibt der Text unveraendert."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = {"response": 'TOOL_CALL: search(query="normal")'}
        step = adapter.step()
        assert step.tool_call is not None
        assert step.tool_call[1]["query"] == "normal"

    def test_add_tool_result_truncates(self, adapter, tools):
        from kiara.shared.constants import AGENT_RESULT_TRUNCATE_OLLAMA

        adapter.init_conversation("Frage", "System", tools)
        long_text = "x" * 3000
        adapter.add_tool_result("search", long_text)
        assert len(adapter._previous_results[-1]["result"]) <= AGENT_RESULT_TRUNCATE_OLLAMA + 3

    def test_deanonymize_params_returns_new_dict(self, adapter):
        """deanonymize_params() gibt IMMER ein neues Dict zurueck (Thread-Safety)."""
        params = {"query": "[Unternehmen_001] Rechnung"}
        result = adapter.deanonymize_params(params)
        assert result == params
        assert result is not params  # Muss neues Dict sein

    def test_stream_final_yields_chunks(self, adapter, tools):
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = iter(
            [
                {"response": "Die "},
                {"response": "Antwort."},
                {
                    "response": "",
                    "done": True,
                    "prompt_eval_count": 100,
                    "eval_count": 50,
                    "total_duration": 500_000_000,
                },
            ]
        )
        chunks = list(adapter.stream_final())
        assert "".join(chunks) == "Die Antwort."

    def test_stream_final_sets_usage(self, adapter, tools):
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = iter(
            [
                {
                    "response": "OK",
                    "done": True,
                    "prompt_eval_count": 100,
                    "eval_count": 50,
                    "total_duration": 1_000_000_000,
                },
            ]
        )
        list(adapter.stream_final())
        usage = adapter.last_usage
        assert usage["prompt_tokens"] == 100
        assert usage["response_tokens"] == 50
        assert usage["model"] == "qwen3:8b"

    def test_stream_final_think_true_passes_tags_through(self, adapter, tools):
        """stream_final(think=True) streamt <think>-Bloecke durch (Frontend rendert sie)."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = iter(
            [
                {"response": "<think>\nIch denke"},
                {"response": " nach...\n</think>\n"},
                {"response": "Die Antwort"},
                {"response": " ist 42."},
                {
                    "response": "",
                    "done": True,
                    "prompt_eval_count": 50,
                    "eval_count": 30,
                    "total_duration": 500_000_000,
                },
            ]
        )
        chunks = list(adapter.stream_final(think=True))
        result = "".join(chunks)
        assert "<think>" in result
        assert "Ich denke" in result
        assert "Die Antwort" in result
        assert "42" in result

    def test_stream_final_no_think_tags_unchanged(self, adapter, tools):
        """Ohne <think>-Tags wird der Stream nicht veraendert."""
        adapter.init_conversation("Frage", "System", tools)
        # Genug Text (>20 chars) damit der Puffer sofort yielded wird
        adapter._client.generate.return_value = iter(
            [
                {"response": "Eine ganz normale Antwort "},
                {"response": "ohne Denken."},
                {
                    "response": "",
                    "done": True,
                    "prompt_eval_count": 50,
                    "eval_count": 20,
                    "total_duration": 300_000_000,
                },
            ]
        )
        chunks = list(adapter.stream_final())
        result = "".join(chunks)
        assert result == "Eine ganz normale Antwort ohne Denken."

    def test_stream_final_think_true_unclosed_tag_passed_through(self, adapter, tools):
        """stream_final(think=True) streamt auch nicht geschlossene <think>-Bloecke durch."""
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.return_value = iter(
            [
                {"response": "<think>\nEndloses Denken..."},
                {
                    "response": "",
                    "done": True,
                    "prompt_eval_count": 30,
                    "eval_count": 10,
                    "total_duration": 200_000_000,
                },
            ]
        )
        chunks = list(adapter.stream_final(think=True))
        result = "".join(chunks)
        # Bei think=True werden Tags durchgestreamt (Frontend kuemmert sich ums Rendering)
        assert "<think>" in result

    def test_stream_final_error_before_first_token_raises(self, adapter, tools):
        """Fehler vor dem ersten Stream-Token werden als retryable Adapterfehler gemeldet."""
        adapter.init_conversation("Frage", "System", tools)

        def _boom():
            raise RuntimeError("stream start failed")
            yield {}

        adapter._client.generate.return_value = _boom()
        with pytest.raises(ToolAdapterError) as exc_info:
            list(adapter._generate_raw("prompt", think=False))
        assert exc_info.value.retryable is True

    def test_step_exception_raises_adapter_error(self, adapter, tools):
        adapter.init_conversation("Frage", "System", tools)
        adapter._client.generate.side_effect = ConnectionError("timeout")
        with pytest.raises(ToolAdapterError) as exc_info:
            adapter.step()
        assert exc_info.value.retryable is True

    def test_build_tools_prompt_format(self):
        tools = [
            ToolDefinition(
                name="search_documents",
                description="Suche Dokumente",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                    "required": ["query"],
                },
            ),
        ]
        prompt = OllamaToolAdapter._build_tools_prompt(tools)
        assert "search_documents" in prompt
        assert "query: str" in prompt
        assert "k: int = optional" in prompt
        assert "TOOL_CALL:" in prompt


# ===========================================================================
# OllamaToolAdapter — Prompt-Aufbau
# ===========================================================================


class TestOllamaToolAdapterPromptBuilding:
    @pytest.fixture
    def adapter(self):
        with patch("ollama.Client"):
            return OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

    def test_agent_prompt_contains_question(self, adapter):
        adapter.init_conversation("Was ist RAG?", "System {tools_prompt}", [])
        prompt = adapter._build_agent_prompt()
        assert "Was ist RAG?" in prompt

    def test_agent_prompt_contains_history(self, adapter):
        adapter.init_conversation(
            "Frage", "System {tools_prompt}", [], history_block="Benutzer: Hallo"
        )
        prompt = adapter._build_agent_prompt()
        assert "Benutzer: Hallo" in prompt

    def test_agent_prompt_with_results(self, adapter):
        adapter.init_conversation("Frage", "System {tools_prompt}", [])
        adapter._previous_results = [{"tool": "search", "result": "Ergebnis"}]
        prompt = adapter._build_agent_prompt()
        assert "Bisherige Werkzeug-Ergebnisse" in prompt
        assert "search" in prompt

    def test_agent_prompt_no_search_documents_reference(self, adapter):
        """Nach Sprint 11a: Agent-Prompt fordert nicht mehr zu search_documents() auf."""
        adapter.init_conversation("Status?", "Admin-System {tools_prompt}", [])
        adapter._previous_results = [{"tool": "dashboard.status", "result": "OK"}]
        prompt = adapter._build_agent_prompt()
        assert "search_documents()" not in prompt
        assert "weiteres passendes Werkzeug" in prompt

    def test_final_prompt_preserves_prepared_documents(self, adapter):
        """Tool-Loop und finale Antwort bekommen denselben Dokumentinhalt."""
        from kiara.generation.context_selection import prepare_context

        adapter.init_conversation("Kosten?", "System {tools_prompt}", [])
        prepared = prepare_context(
            [
                {
                    "text": "Rechnung\n2,99 €",
                    "metadata": {"source": "rechnung.pdf", "chunk_index": 0},
                },
                {
                    "text": "Bedingung\nNicht erstattbar.",
                    "metadata": {"source": "vertrag.pdf", "chunk_index": 0},
                },
            ],
            budget=4000,
        )
        adapter.add_tool_result("search_documents", prepared.text, raw_chunks=prepared.chunks)
        assert prepared.text in adapter._build_agent_prompt()
        assert prepared.text in adapter._build_final_prompt()

    def test_final_prompt_fallback_without_chunks(self, adapter):
        """Reine Werkzeug-Ergebnisse bleiben im finalen Prompt sichtbar."""
        adapter.init_conversation("Frage", "System {tools_prompt}", [])
        adapter._previous_results = [{"tool": "calc", "result": "42"}]
        prompt = adapter._build_final_prompt()
        assert "Gesammelte Informationen" in prompt
        assert "42" in prompt

    def test_final_prompt_non_search_results_alongside_chunks(self, adapter):
        """Nicht-Such-Ergebnisse werden separat dargestellt wenn Chunks vorliegen."""
        adapter.init_conversation("Frage", "System {tools_prompt}", [])
        adapter.add_tool_result(
            "search_documents",
            "[Quelle 1: doc.pdf]\nInhalt",
            raw_chunks=[
                {"text": "Inhalt", "metadata": {"source": "doc.pdf", "chunk_index": 0}},
            ],
        )
        adapter._previous_results.append({"tool": "calculate", "result": "42"})
        prompt = adapter._build_final_prompt()
        assert "Weitere Werkzeug-Ergebnisse" in prompt
        assert "42" in prompt
        assert "[Quelle 1:" in prompt

    def test_final_prompt_with_additional_context(self, adapter):
        adapter.init_conversation("Frage", "System {tools_prompt}", [])
        prompt = adapter._build_final_prompt("Bitte zusammenfassen.")
        assert "Bitte zusammenfassen." in prompt


# ===========================================================================
# Factory
# ===========================================================================


class TestCreateToolAdapter:
    @pytest.fixture
    def config(self):
        return {
            "cloud": {"max_tokens": 4096, "temperature": 0.3},
            "chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}},
        }

    def test_ollama_backend(self, config):
        with patch("ollama.Client"):
            adapter = create_tool_adapter(
                {"url": "http://localhost:11434", "model": "m", "provider": "ollama"},
                config,
            )
        assert isinstance(adapter, OllamaToolAdapter)

    def test_default_provider_is_ollama(self, config):
        with patch("ollama.Client"):
            adapter = create_tool_adapter(
                {"url": "http://localhost:11434", "model": "m"},
                config,
            )
        assert isinstance(adapter, OllamaToolAdapter)

    def test_anthropic_backend(self, config):
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = create_tool_adapter(
                {"api_key": "sk-test", "model": "claude", "provider": "anthropic"},
                config,
            )
        assert isinstance(adapter, AnthropicToolAdapter)

    def test_openai_backend(self, config):
        with patch.dict(sys.modules, {"openai": MagicMock()}):
            adapter = create_tool_adapter(
                {"api_key": "sk-test", "model": "gpt-4", "provider": "openai"},
                config,
            )
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_ollama_gets_no_anonymizer(self, config):
        """Ollama-Adapter soll keinen Anonymizer erhalten, auch wenn einer uebergeben wird."""
        mock_anon = MagicMock()
        with patch("ollama.Client"):
            adapter = create_tool_adapter(
                {"url": "http://localhost:11434", "model": "m", "provider": "ollama"},
                config,
                anonymizer=mock_anon,
            )
        # OllamaToolAdapter hat kein _anonymizer-Attribut
        assert not hasattr(adapter, "_anonymizer")

    def test_anthropic_gets_anonymizer(self, config):
        mock_anon = MagicMock()
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = create_tool_adapter(
                {"api_key": "sk-test", "model": "claude", "provider": "anthropic"},
                config,
                anonymizer=mock_anon,
            )
        assert adapter._anonymizer is mock_anon


# ===========================================================================
# AnthropicToolAdapter
# ===========================================================================


class TestAnthropicToolAdapter:
    @pytest.fixture
    def config(self):
        return {"cloud": {"max_tokens": 4096, "temperature": 0.3}}

    @pytest.fixture
    def backend(self):
        return {"api_key": "sk-test", "model": "claude-sonnet", "provider": "anthropic"}

    @pytest.fixture
    def tools(self):
        return [
            ToolDefinition(
                name="search",
                description="Suche",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ]

    @pytest.fixture
    def adapter(self, backend, config):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adpt = AnthropicToolAdapter(backend, config)
            adpt._client = mock_client
            yield adpt

    def test_init_conversation_anonymizes_question(self, backend, config, tools):
        mock_anon = MagicMock()
        mock_anon.anonymize.return_value = "[Anonymisiert] Frage"
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(backend, config, anonymizer=mock_anon)
        adapter.init_conversation("Original Frage", "System", tools)
        assert adapter._messages[0]["content"] == "[Anonymisiert] Frage"

    def test_init_conversation_embeds_history(self, backend, config, tools):
        mock_anon = MagicMock()
        mock_anon.anonymize.side_effect = lambda x: x
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(backend, config, anonymizer=mock_anon)
        adapter.init_conversation("Frage", "System", tools, history_block="History Block")
        # History wird VOR Anonymisierung eingebettet
        assert "History Block" in mock_anon.anonymize.call_args[0][0]
        assert "Frage" in mock_anon.anonymize.call_args[0][0]

    def test_step_tool_use_response(self, adapter, tools):
        # Mock Anthropic-Response mit tool_use Block
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Ich suche jetzt."

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search"
        mock_tool_block.input = {"query": "Python"}
        mock_tool_block.id = "toolu_abc"

        mock_resp = MagicMock()
        mock_resp.content = [mock_text_block, mock_tool_block]
        mock_resp.usage.input_tokens = 100
        mock_resp.usage.output_tokens = 50

        adapter._client.messages.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()

        assert step.tool_call == ("search", {"query": "Python"})
        assert step.text == "Ich suche jetzt."
        assert step.tool_use_id == "toolu_abc"
        assert step.is_final is False

    def test_step_tool_use_without_id_raises(self, adapter, tools):
        # Anthropic-Schema: id ist Pflicht. Schema-Drift -> ToolAdapterError statt None.
        # Validation muss VOR _messages-Append + Usage-Buchung erfolgen, damit kein
        # invalider Assistant-Turn die History/das Audit kontaminiert.
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search"
        mock_tool_block.input = {"query": "Python"}
        mock_tool_block.id = None

        mock_resp = MagicMock()
        mock_resp.content = [mock_tool_block]
        mock_resp.usage.input_tokens = 50
        mock_resp.usage.output_tokens = 20

        adapter._client.messages.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)
        # Baseline nach init: nur die User-Frage in _messages, Usage = 0
        assert len(adapter._messages) == 1
        assert adapter._messages[0]["role"] == "user"
        assert adapter._usage == {"input": 0, "output": 0}

        with pytest.raises(ToolAdapterError, match="ohne id"):
            adapter.step()

        # State-Invariante: KEIN Assistant-Turn appended, KEINE Usage gebucht
        assert len(adapter._messages) == 1
        assert adapter._messages[0]["role"] == "user"
        assert adapter._usage == {"input": 0, "output": 0}

    def test_build_api_params_old_model_uses_budgeted_thinking(self, adapter, tools):
        adapter.init_conversation("Frage", "System", tools)
        adapter._think = True

        params = adapter._build_api_params()

        assert params["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }
        assert params["max_tokens"] == 4096
        assert "temperature" not in params

    def test_step_final_response(self, adapter, tools):
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Die Antwort."

        mock_resp = MagicMock()
        mock_resp.content = [mock_text_block]
        mock_resp.usage.input_tokens = 100
        mock_resp.usage.output_tokens = 30

        adapter._client.messages.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()

        assert step.tool_call is None
        assert step.text == "Die Antwort."
        assert step.is_final is True

    def test_step_final_deanonymizes(self, backend, config, tools):
        mock_anon = MagicMock()
        mock_anon.anonymize.side_effect = lambda x: x
        mock_anon.deanonymize.return_value = "De-anonymisierte Antwort"

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(backend, config, anonymizer=mock_anon)

        mock_text = MagicMock(type="text", text="[Anon] Antwort")
        mock_resp = MagicMock()
        mock_resp.content = [mock_text]
        mock_resp.usage.input_tokens = 50
        mock_resp.usage.output_tokens = 20
        mock_client.messages.create.return_value = mock_resp

        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()
        assert step.text == "De-anonymisierte Antwort"
        assert step.is_final is True

    def test_step_tool_call_deanonymizes_reasoning(self, backend, config, tools):
        """Tool-Call-Pfad: step.text (inkl. <think>-Block) wird de-anonymisiert.

        Regression: ohne De-Anonymisierung hat agent_chain.py den anonymisierten
        <think>-Block an den User geyielded (ANON:Person_001 sichtbar im UI).
        """
        mock_anon = MagicMock()
        mock_anon.anonymize.side_effect = lambda x: x
        mock_anon.deanonymize.return_value = "Reasoning mit echtem Namen"

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(backend, config, anonymizer=mock_anon)

        mock_text = MagicMock(type="text", text="Reasoning mit ANON:Person_001")
        mock_tool = MagicMock(type="tool_use", input={"query": "x"}, id="toolu_x")
        mock_tool.name = "search"
        mock_resp = MagicMock()
        mock_resp.content = [mock_text, mock_tool]
        mock_resp.usage.input_tokens = 50
        mock_resp.usage.output_tokens = 20
        mock_client.messages.create.return_value = mock_resp

        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()
        assert step.tool_call is not None
        assert step.text == "Reasoning mit echtem Namen"
        assert step.is_final is False

    def test_parallel_tool_calls_returned(self, adapter, tools, caplog):
        """Parallele Tool-Calls werden in ToolStep.parallel_calls zurueckgegeben."""
        import logging

        block1 = MagicMock(type="tool_use", input={"query": "a"}, id="t1")
        block1.name = "search"
        block2 = MagicMock(type="tool_use", input={"expr": "1"}, id="t2")
        block2.name = "calc"
        mock_resp = MagicMock()
        mock_resp.content = [block1, block2]
        mock_resp.usage.input_tokens = 50
        mock_resp.usage.output_tokens = 20

        adapter._client.messages.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)

        with caplog.at_level(logging.INFO, logger="kiara.generation.adapters.anthropic"):
            step = adapter.step()
        assert "parallele Tool-Calls" in caplog.text
        # Erster Tool-Call
        assert step.tool_call[0] == "search"
        # Parallele Calls in parallel_calls
        assert step.parallel_calls is not None
        assert len(step.parallel_calls) == 1
        assert step.parallel_calls[0] == ("calc", {"expr": "1"}, "t2")
        # _skipped_tool_ids existiert nicht mehr
        assert not hasattr(adapter, "_skipped_tool_ids")

    def test_add_tool_results_batch(self, adapter, tools):
        """add_tool_results() sendet alle Results in einer user-Message (Anthropic-Pflicht)."""
        adapter.init_conversation("Frage", "System", tools)
        results = [
            ("search", "Ergebnis 1", "t1"),
            ("calc", "42", "t2"),
        ]
        adapter.add_tool_results(results)
        msg = adapter._messages[-1]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["tool_use_id"] == "t1"
        assert msg["content"][0]["content"] == "Ergebnis 1"
        assert msg["content"][1]["tool_use_id"] == "t2"
        assert msg["content"][1]["content"] == "42"

    def test_add_tool_result_truncates(self, adapter, tools):
        from kiara.shared.constants import AGENT_RESULT_TRUNCATE_CLOUD

        adapter.init_conversation("Frage", "System", tools)
        long_text = "y" * 10000
        adapter.add_tool_result("search", long_text, tool_use_id="toolu_x")
        msg = adapter._messages[-1]
        content_text = msg["content"][0]["content"]
        assert len(content_text) <= AGENT_RESULT_TRUNCATE_CLOUD + 3

    def test_add_tool_result_anonymizes(self, backend, config, tools):
        mock_anon = MagicMock()
        mock_anon.anonymize.return_value = "[ANON] Ergebnis"
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(backend, config, anonymizer=mock_anon)
        adapter.init_conversation("Frage", "System", tools)
        adapter.add_tool_result("search", "Original Ergebnis", tool_use_id="t1")
        msg = adapter._messages[-1]
        assert msg["content"][0]["content"] == "[ANON] Ergebnis"

    def test_deanonymize_params(self, backend, config):
        mock_anon = MagicMock()
        mock_anon.deanonymize.side_effect = lambda x: x.replace("[U1]", "Hetzner")
        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(backend, config, anonymizer=mock_anon)
        result = adapter.deanonymize_params({"query": "[U1] Rechnung", "k": 5})
        assert result["query"] == "Hetzner Rechnung"
        assert result["k"] == 5  # int bleibt unveraendert

    def test_deanonymize_params_no_anonymizer(self, adapter):
        params = {"query": "test"}
        result = adapter.deanonymize_params(params)
        assert result == params
        assert result is not params  # Muss neues Dict sein

    def test_last_usage_cumulative(self, adapter, tools):
        mock_resp1 = MagicMock()
        mock_resp1.content = [MagicMock(type="tool_use", name="s", input={}, id="t1")]
        mock_resp1.usage.input_tokens = 100
        mock_resp1.usage.output_tokens = 50

        mock_resp2 = MagicMock()
        mock_resp2.content = [MagicMock(type="text", text="OK")]
        mock_resp2.usage.input_tokens = 200
        mock_resp2.usage.output_tokens = 80

        adapter._client.messages.create.side_effect = [mock_resp1, mock_resp2]
        adapter.init_conversation("F", "S", tools)
        adapter.step()  # Tool-Call
        adapter.step()  # Final
        usage = adapter.last_usage
        assert usage["prompt_tokens"] == 300
        assert usage["response_tokens"] == 130


# ===========================================================================
# OpenAIToolAdapter
# ===========================================================================


class TestOpenAIToolAdapter:
    @pytest.fixture
    def config(self):
        return {"cloud": {"max_tokens": 4096, "temperature": 0.3}}

    @pytest.fixture
    def backend(self):
        return {"api_key": "sk-test", "model": "gpt-4o", "provider": "openai", "base_url": ""}

    @pytest.fixture
    def tools(self):
        return [
            ToolDefinition(
                name="search",
                description="Suche",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
        ]

    @pytest.fixture
    def adapter(self, backend, config):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict(sys.modules, {"openai": mock_openai}):
            adpt = OpenAIToolAdapter(backend, config)
            adpt._client = mock_client
            yield adpt

    @staticmethod
    def _responses_adapter(config):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        backend = {
            "api_key": "sk-test",
            "model": "gpt-5.5-pro",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
        }
        with patch.dict(sys.modules, {"openai": mock_openai}):
            adapter = OpenAIToolAdapter(backend, config)
        adapter._client = mock_client
        return adapter, mock_client

    @staticmethod
    def _response_item(**values):
        item = MagicMock()
        for name, value in values.items():
            setattr(item, name, value)
        item.model_dump.return_value = dict(values)
        return item

    def test_step_tool_call(self, adapter, tools):
        mock_tc = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = json.dumps({"query": "Python"})
        mock_tc.id = "call_abc"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_message.content = "Reasoning"
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Reasoning",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "function": {"name": "search", "arguments": '{"query":"Python"}'},
                }
            ],
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 100
        mock_resp.usage.completion_tokens = 40

        adapter._client.chat.completions.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()

        assert step.tool_call == ("search", {"query": "Python"})
        assert step.text == "Reasoning"
        assert step.tool_use_id == "call_abc"

    def test_step_o_series_uses_max_completion_tokens(self, backend, config, tools):
        backend = {**backend, "model": "o3-mini"}
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict(sys.modules, {"openai": mock_openai}):
            adapter = OpenAIToolAdapter(backend, config)
            adapter._client = mock_client

        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Antwort"
        mock_message.model_dump.return_value = {"role": "assistant", "content": "Antwort"}
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message = mock_message
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 20
        mock_resp.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_resp

        adapter.init_conversation("Frage", "System", tools)
        adapter.step()

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_completion_tokens"] == 4096
        assert "max_tokens" not in call_kwargs
        assert "temperature" not in call_kwargs

    def test_step_partial_tool_calls_with_length_finish_raises(self, adapter, tools):
        """finish_reason='length' mit teilweisem tool_calls darf nicht als finale
        Antwort durchrutschen (Token-Limit-Truncation der Tool-Args)."""
        mock_tc = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"query":"Pyth'  # abgeschnitten
        mock_tc.id = "call_partial"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_message.content = ""
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_partial"}],
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 100
        mock_resp.usage.completion_tokens = 4096

        adapter._client.chat.completions.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)
        original_messages = list(adapter._messages)

        with pytest.raises(ToolAdapterError) as exc_info:
            adapter.step()

        assert exc_info.value.retryable is False
        assert "finish_reason='length'" in str(exc_info.value)
        assert adapter._messages == original_messages

    def test_step_malformed_tool_arguments_raises_without_state_mutation(self, adapter, tools):
        mock_tc = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '{"query":'
        mock_tc.id = "call_bad"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_message.content = ""
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_bad"}],
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 20
        mock_resp.usage.completion_tokens = 5

        adapter._client.chat.completions.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)
        original_messages = list(adapter._messages)

        with pytest.raises(ToolAdapterError) as exc_info:
            adapter.step()

        # JSON-Decode-Fehler in Tool-Arguments sind NICHT retryable: ein Retry
        # ohne Aenderung des Modells/Prompts wuerde dieselbe kaputte JSON-Antwort
        # erzeugen. Anthropic-Adapter haert das gleich (Z. 252, 277). Ollama
        # behandelt es bewusst anders (kleinere lokale Modelle, Retry-Hint hilft).
        assert exc_info.value.retryable is False
        assert adapter._messages == original_messages

    def test_step_non_object_tool_arguments_raises(self, adapter, tools):
        mock_tc = MagicMock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = '["query", "Python"]'
        mock_tc.id = "call_list"

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tc]
        mock_message.content = ""
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_list"}],
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 20
        mock_resp.usage.completion_tokens = 5

        adapter._client.chat.completions.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)

        with pytest.raises(ToolAdapterError) as exc_info:
            adapter.step()

        assert exc_info.value.retryable is True
        assert "JSON-Objekt" in str(exc_info.value)

    def test_step_final_response(self, adapter, tools):
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Die Antwort."
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Die Antwort.",
            "tool_calls": None,
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 80
        mock_resp.usage.completion_tokens = 20

        adapter._client.chat.completions.create.return_value = mock_resp
        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()

        assert step.tool_call is None
        assert step.text == "Die Antwort."
        assert step.is_final is True

    def test_stream_final_appends_answer_to_last_conversation(self, adapter, tools):
        def _chunk(text):
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=text))]
            chunk.usage = None
            return chunk

        adapter.init_conversation("Frage", "System", tools)
        adapter._client.chat.completions.create.return_value = [
            _chunk("Hallo "),
            _chunk("Welt"),
        ]

        assert list(adapter.stream_final()) == ["Hallo Welt"]
        conversation = json.loads(adapter.last_conversation)
        assert conversation["messages"][-1] == {
            "role": "assistant",
            "content": "Hallo Welt",
        }

    def test_model_dump_cleans_null_tool_calls(self, adapter, tools):
        """tool_calls=None wird aus model_dump entfernt (Proxy-Kompatibilitaet)."""
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Antwort"
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Antwort",
            "tool_calls": None,
            "function_call": None,
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 10

        adapter._client.chat.completions.create.return_value = mock_resp
        adapter.init_conversation("F", "S", tools)
        adapter.step()

        # Letzte Message im Messages-Array pruefen
        last_msg = adapter._messages[-1]
        assert "tool_calls" not in last_msg
        assert "function_call" not in last_msg

    def test_parallel_tool_calls_warning(self, adapter, tools, caplog):
        import logging

        tc1 = MagicMock()
        tc1.function.name = "search"
        tc1.function.arguments = '{"query": "a"}'
        tc1.id = "c1"
        tc2 = MagicMock()
        tc2.function.name = "calc"
        tc2.function.arguments = '{"expr": "1"}'
        tc2.id = "c2"

        mock_message = MagicMock()
        mock_message.tool_calls = [tc1, tc2]
        mock_message.content = ""
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1"}, {"id": "c2"}],
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message = mock_message

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 50
        mock_resp.usage.completion_tokens = 20

        adapter._client.chat.completions.create.return_value = mock_resp
        adapter.init_conversation("F", "S", tools)

        with caplog.at_level(logging.INFO, logger="kiara.generation.adapters.openai"):
            step = adapter.step()
        assert "parallele Tool-Calls" in caplog.text
        assert step.tool_call[0] == "search"
        # Parallele Calls in parallel_calls
        assert step.parallel_calls is not None
        assert len(step.parallel_calls) == 1
        assert step.parallel_calls[0] == ("calc", {"expr": "1"}, "c2")
        assert not hasattr(adapter, "_skipped_tool_ids")

    def test_add_tool_results_batch(self, adapter, tools):
        """add_tool_results() sendet separate tool-Messages pro Result (OpenAI-Format)."""
        adapter.init_conversation("F", "S", tools)
        results = [
            ("search", "Ergebnis 1", "c1"),
            ("calc", "42", "c2"),
        ]
        adapter.add_tool_results(results)
        # Letzte 2 Messages sind tool-Messages
        assert adapter._messages[-2]["role"] == "tool"
        assert adapter._messages[-2]["tool_call_id"] == "c1"
        assert adapter._messages[-2]["content"] == "Ergebnis 1"
        assert adapter._messages[-1]["role"] == "tool"
        assert adapter._messages[-1]["tool_call_id"] == "c2"
        assert adapter._messages[-1]["content"] == "42"

    def test_add_tool_result_format(self, adapter, tools):
        adapter.init_conversation("F", "S", tools)
        adapter.add_tool_result("search", "Ergebnis Text", tool_use_id="call_x")
        msg = adapter._messages[-1]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_x"
        assert msg["content"] == "Ergebnis Text"

    def test_init_conversation_embeds_history(self, backend, config, tools):
        mock_anon = MagicMock()
        mock_anon.anonymize.side_effect = lambda x: x
        with patch.dict(sys.modules, {"openai": MagicMock()}):
            adapter = OpenAIToolAdapter(backend, config, anonymizer=mock_anon)
        adapter.init_conversation("Frage", "System", tools, history_block="Verlauf")
        assert "Verlauf" in mock_anon.anonymize.call_args[0][0]

    def test_gpt55_pro_init_uses_responses_item_format(self, config, tools):
        adapter, _ = self._responses_adapter(config)

        adapter.init_conversation("Frage", "System", tools)

        assert adapter._messages == [{"role": "user", "content": "Frage"}]
        assert adapter._instructions == "System"
        assert adapter._tool_defs == [
            {
                "type": "function",
                "name": "search",
                "description": "Suche",
                "parameters": tools[0].parameters,
                "strict": False,
            }
        ]

    def test_gpt55_pro_step_parses_and_preserves_function_calls(self, config, tools):
        adapter, mock_client = self._responses_adapter(config)
        reasoning = self._response_item(
            id="rs_1",
            type="reasoning",
            summary=[],
            status="completed",
        )
        first_call = self._response_item(
            id="fc_1",
            type="function_call",
            name="search",
            arguments='{"query":"Python"}',
            call_id="call_1",
            status="completed",
        )
        second_call = self._response_item(
            id="fc_2",
            type="function_call",
            name="calc",
            arguments='{"expr":"6*7"}',
            call_id="call_2",
            status="completed",
        )
        mock_client.responses.create.return_value = SimpleNamespace(
            status="completed",
            output=[reasoning, first_call, second_call],
            output_text="Werkzeuge werden aufgerufen.",
            usage=SimpleNamespace(input_tokens=100, output_tokens=40),
        )

        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()

        assert step.tool_call == ("search", {"query": "Python"})
        assert step.tool_use_id == "call_1"
        assert step.parallel_calls == [("calc", {"expr": "6*7"}, "call_2")]
        assert adapter._messages[-3:] == [
            reasoning.model_dump.return_value,
            first_call.model_dump.return_value,
            second_call.model_dump.return_value,
        ]
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["instructions"] == "System"
        assert call_kwargs["max_output_tokens"] == 4096
        assert call_kwargs["tools"][0]["name"] == "search"
        assert "temperature" not in call_kwargs
        assert adapter.last_usage["prompt_tokens"] == 100
        assert adapter.last_usage["response_tokens"] == 40

        adapter.add_tool_results(
            [
                ("search", "Treffer", "call_1"),
                ("calc", "42", "call_2"),
            ]
        )
        assert adapter._messages[-2:] == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Treffer",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "42",
            },
        ]

    def test_gpt55_pro_malformed_function_call_does_not_mutate_input(self, config, tools):
        adapter, mock_client = self._responses_adapter(config)
        malformed_call = self._response_item(
            id="fc_bad",
            type="function_call",
            name="search",
            arguments='{"query":',
            call_id="call_bad",
            status="completed",
        )
        mock_client.responses.create.return_value = SimpleNamespace(
            status="completed",
            output=[malformed_call],
            output_text="",
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        )
        adapter.init_conversation("Frage", "System", tools)
        original_input = list(adapter._messages)

        with pytest.raises(ToolAdapterError) as exc_info:
            adapter.step()

        assert exc_info.value.retryable is False
        assert adapter._messages == original_input

    def test_gpt55_pro_final_response(self, config, tools):
        adapter, mock_client = self._responses_adapter(config)
        message_item = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Die Antwort."}],
        }
        mock_client.responses.create.return_value = SimpleNamespace(
            status="completed",
            output=[message_item],
            output_text="Die Antwort.",
            usage=SimpleNamespace(input_tokens=80, output_tokens=20),
        )

        adapter.init_conversation("Frage", "System", tools)
        step = adapter.step()

        assert step.tool_call is None
        assert step.text == "Die Antwort."
        assert step.is_final is True
        assert adapter._messages[-1] == message_item

    def test_gpt55_pro_stream_final_uses_typed_events(self, config, tools):
        adapter, mock_client = self._responses_adapter(config)
        mock_client.responses.create.return_value = iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="Hallo "),
                SimpleNamespace(type="response.output_text.delta", delta="Welt"),
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        usage=SimpleNamespace(input_tokens=60, output_tokens=10)
                    ),
                ),
            ]
        )
        adapter.init_conversation("Frage", "System", tools)

        assert list(adapter.stream_final()) == ["Hallo Welt"]
        conversation = json.loads(adapter.last_conversation)
        assert conversation["instructions"] == "System"
        assert conversation["input"][-1] == {
            "role": "assistant",
            "content": "Hallo Welt",
        }
        assert adapter.last_usage["prompt_tokens"] == 60
        assert adapter.last_usage["response_tokens"] == 10
