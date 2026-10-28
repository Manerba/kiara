"""Tests fuer kiara.generation.agent_chain (Adapter-basiert)."""

from unittest.mock import MagicMock, PropertyMock

import pytest

from kiara.generation.agent_chain import (
    AgentChain,
    AgentEvent,
    _deduplicate_sources,
    _freeze_params,
)
from kiara.generation.tool_adapter import BaseToolAdapter, ToolAdapterError, ToolStep
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.builtin_tools import CalculateTool
from kiara.tools.registry import ToolRegistry


@pytest.fixture
def base_config():
    return {
        "chat": {
            "agent": {
                "max_iterations": 3,
                "temperature": 0.3,
            },
        },
    }


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(CalculateTool())
    return reg


class _FailingTool(BaseTool):
    """Test-Tool, das immer eine Exception wirft."""

    def definition(self):
        return ToolDefinition(
            name="failing_tool",
            description="Wirft immer einen Fehler.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, params, user_id=None):
        raise RuntimeError("Tool-Explosion")


class _SemanticFailTool(BaseTool):
    """Test-Tool mit fachlichem Fehler, der nicht retryable ist."""

    def __init__(self):
        self.calls = 0

    def definition(self):
        return ToolDefinition(
            name="semantic_fail",
            description="Liefert einen fachlichen Fehler.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, params, user_id=None):
        self.calls += 1
        return ToolResult(
            success=False,
            data=None,
            error="Eintrag nicht gefunden",
            display_text="",
            retryable=False,
        )


class _RetryableFailTool(BaseTool):
    """Test-Tool mit transientem Fehler, der retryable ist."""

    def __init__(self):
        self.calls = 0

    def definition(self):
        return ToolDefinition(
            name="retryable_fail",
            description="Liefert einen transienten Fehler.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(self, params, user_id=None):
        self.calls += 1
        return ToolResult(
            success=False,
            data=None,
            error="Backend kurz nicht erreichbar",
            display_text="",
            retryable=True,
        )


class _SourceTool(BaseTool):
    """Test-Tool, das Sources zurueckgibt (wie SearchDocuments)."""

    def definition(self):
        return ToolDefinition(
            name="source_tool",
            description="Liefert Quellen.",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )

    async def execute(self, params, user_id=None):
        return ToolResult(
            success=True,
            data={
                "results": [],
                "sources": [
                    {"source": "doc1.txt", "score": 0.9, "excerpt": "Auszug 1"},
                    {"source": "doc2.txt", "score": 0.7, "excerpt": "Auszug 2"},
                ],
            },
            error=None,
            display_text="2 Dokumente gefunden",
        )


def _make_mock_adapter(steps: list[ToolStep], usage: dict | None = None):
    """Erstellt einen Mock-Adapter der die gegebenen ToolSteps zurueckgibt."""
    adapter = MagicMock(spec=BaseToolAdapter)
    adapter.prepare_tool_text.side_effect = str
    adapter.step.side_effect = steps
    adapter.deanonymize_params.side_effect = lambda p: {**p}
    adapter.stream_final.return_value = iter(["Finale ", "Antwort."])
    adapter.add_tool_result.return_value = None
    adapter.add_tool_results.return_value = None
    type(adapter).last_usage = PropertyMock(
        return_value=usage
        or {
            "prompt_tokens": 100,
            "response_tokens": 50,
            "total_duration_ms": 500,
            "num_ctx": 8192,
            "model": "test",
        }
    )
    type(adapter).result_truncate_limit = PropertyMock(return_value=24000)
    return adapter


# =====================================================================
# AgentEvent
# =====================================================================


class TestAgentEvent:
    def test_create_event(self):
        event = AgentEvent(event_type="chunk", data={"text": "hello"})
        assert event.event_type == "chunk"
        assert event.data == {"text": "hello"}

    def test_event_types(self):
        for etype in ["tool_call", "tool_result", "sources", "chunk", "done", "error"]:
            event = AgentEvent(event_type=etype)
            assert event.event_type == etype

    def test_event_default_data(self):
        event = AgentEvent(event_type="done")
        assert event.data is None

    def test_event_with_list_data(self):
        sources = [{"source": "a.txt"}, {"source": "b.txt"}]
        event = AgentEvent(event_type="sources", data=sources)
        assert isinstance(event.data, list)
        assert len(event.data) == 2


# =====================================================================
# AgentChain — Direkte Antwort (kein Tool-Call)
# =====================================================================


class TestAgentChainDirectAnswer:
    def test_is_final_true_yields_text_directly(self, base_config, registry):
        """Cloud-Adapter mit is_final=True → text direkt als chunk yielded."""
        adapter = _make_mock_adapter(
            [
                ToolStep(text="Die Antwort ist 42.", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Frage"))

        event_types = [e.event_type for e in events]
        assert "chunk" in event_types
        assert "done" in event_types
        assert "tool_call" not in event_types

        chunk = next(e for e in events if e.event_type == "chunk")
        assert chunk.data["text"] == "Die Antwort ist 42."

    def test_is_final_false_calls_stream_final(self, base_config, registry):
        """Ollama-Adapter mit is_final=False → stream_final() aufgerufen."""
        adapter = _make_mock_adapter(
            [
                ToolStep(text="Draft", is_final=False),
            ]
        )
        adapter.stream_final.return_value = iter(["Gestreamte ", "Antwort."])

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Frage"))

        adapter.stream_final.assert_called_once()
        chunks = [e for e in events if e.event_type == "chunk"]
        assert len(chunks) == 2
        assert chunks[0].data["text"] == "Gestreamte "
        assert chunks[1].data["text"] == "Antwort."

    def test_stream_final_generic_exception_yields_error_and_done(self, base_config, registry):
        """Unerwartete stream_final-Fehler beenden den SSE-Strom kontrolliert."""
        adapter = _make_mock_adapter(
            [
                ToolStep(text="Draft", is_final=False),
            ]
        )
        adapter.stream_final.side_effect = RuntimeError("boom")

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Frage"))

        event_types = [e.event_type for e in events]
        assert event_types[-2:] == ["error", "done"]
        assert events[-2].data["message"] == "Finale Antwort konnte nicht erzeugt werden."


# =====================================================================
# AgentChain — Tool-Call-Workflow
# =====================================================================


class TestAgentChainToolWorkflow:
    def test_tool_call_then_final(self, base_config, registry):
        """LLM ruft Tool auf, dann finale Antwort."""
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("calculate", {"expression": "2**10"}), text="Berechne..."),
                ToolStep(text="2**10 = 1024", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Berechne 2 hoch 10"))

        event_types = [e.event_type for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "chunk" in event_types
        assert "done" in event_types

        tc = next(e for e in events if e.event_type == "tool_call")
        assert tc.data["tool"] == "calculate"
        assert tc.data["iteration"] == 1

        tr = next(e for e in events if e.event_type == "tool_result")
        assert tr.data["success"] is True

    def test_unknown_tool_error(self, base_config, registry):
        """Unbekanntes Tool → error im tool_result, kein Crash."""
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("nonexistent_tool", {"param": "x"})),
                ToolStep(text="Konnte nicht helfen.", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Test"))

        tr = next(e for e in events if e.event_type == "tool_result")
        assert tr.data["success"] is False

    def test_add_tool_results_called(self, base_config, registry):
        """adapter.add_tool_results() wird nach Tool-Ausfuehrung aufgerufen."""
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("calculate", {"expression": "1+1"}), tool_use_id="t1"),
                ToolStep(text="OK", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        list(chain.query_stream("Rechne"))

        adapter.add_tool_results.assert_called_once()
        call_args = adapter.add_tool_results.call_args
        results_list = call_args[0][0]
        assert len(results_list) == 1
        assert results_list[0][0] == "calculate"  # tool_name
        assert results_list[0][2] == "t1"  # tool_use_id

    def test_deanonymize_params_called(self, base_config, registry):
        """adapter.deanonymize_params() wird fuer jeden Tool-Call aufgerufen."""
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("calculate", {"expression": "1+1"})),
                ToolStep(text="OK", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        list(chain.query_stream("Rechne"))

        adapter.deanonymize_params.assert_called_once_with({"expression": "1+1"})


# =====================================================================
# AgentChain — Sources
# =====================================================================


class TestAgentChainSources:
    def test_sources_emitted(self, base_config):
        """Sources aus Tool-Ergebnissen werden als sources-Event emittiert."""
        reg = ToolRegistry()
        reg.register(_SourceTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("source_tool", {"q": "test"})),
                ToolStep(text="Zusammenfassung", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        events = list(chain.query_stream("Suche"))

        sources_events = [e for e in events if e.event_type == "sources"]
        assert len(sources_events) == 1
        assert len(sources_events[0].data) == 2

    def test_sources_accumulated_and_deduplicated(self, base_config):
        """Quellen aus mehreren Tool-Calls werden kumuliert und dedupliziert."""
        reg = ToolRegistry()
        reg.register(_SourceTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("source_tool", {"q": "a"})),
                ToolStep(tool_call=("source_tool", {"q": "b"})),
                ToolStep(text="Fertig", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=5)
        events = list(chain.query_stream("Suche"))

        sources_events = [e for e in events if e.event_type == "sources"]
        assert len(sources_events) == 1
        # _SourceTool liefert doc1.txt + doc2.txt — 2x gerufen aber dedupliziert
        assert len(sources_events[0].data) == 2
        source_names = {s["source"] for s in sources_events[0].data}
        assert source_names == {"doc1.txt", "doc2.txt"}

    def test_no_sources_when_none(self, base_config, registry):
        """Kein sources-Event wenn kein Tool Quellen liefert."""
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("calculate", {"expression": "1+1"})),
                ToolStep(text="2", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("1+1?"))
        assert not any(e.event_type == "sources" for e in events)


# =====================================================================
# AgentChain — Max-Iterations
# =====================================================================


class TestAgentChainMaxIterations:
    def test_max_iterations_calls_stream_final(self, base_config, registry):
        """Bei max_iterations → stream_final(additional_context=...) aufgerufen."""
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("calculate", {"expression": "1+1"})),
                ToolStep(tool_call=("calculate", {"expression": "2+2"})),
                ToolStep(tool_call=("calculate", {"expression": "3+3"})),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter, max_iterations=3)
        events = list(chain.query_stream("Endlos"))

        adapter.stream_final.assert_called_once()
        call_args = adapter.stream_final.call_args
        # additional_context kann als positional oder keyword arg kommen
        all_args_str = str(call_args).lower()
        assert "bisherigen ergebnissen" in all_args_str

    def test_max_iterations_emits_sources(self, base_config):
        """Bei max_iterations werden gesammelte Sources emittiert."""
        reg = ToolRegistry()
        reg.register(_SourceTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("source_tool", {"q": "x"})),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=1)
        events = list(chain.query_stream("Limit"))
        assert any(e.event_type == "sources" for e in events)


# =====================================================================
# AgentChain — Error-Handling
# =====================================================================


class TestAgentChainErrorHandling:
    def test_retryable_error_retries_once(self, base_config, registry):
        """ToolAdapterError(retryable=True) → einmaliger Retry."""
        adapter = _make_mock_adapter([])
        adapter.step.side_effect = [
            ToolAdapterError("Rate limit", retryable=True),
            ToolStep(text="Nach Retry OK", is_final=True),
        ]

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Retry-Test"))

        assert any(e.event_type == "chunk" for e in events)
        assert not any(e.event_type == "error" for e in events)

    def test_non_retryable_error_aborts(self, base_config, registry):
        """ToolAdapterError(retryable=False) → sofortiger Abbruch."""
        adapter = _make_mock_adapter([])
        adapter.step.side_effect = ToolAdapterError("Auth failed", retryable=False)

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Auth-Test"))

        assert any(e.event_type == "error" for e in events)
        err = next(e for e in events if e.event_type == "error")
        assert "Auth failed" in err.data["message"]

    def test_retryable_double_failure_aborts(self, base_config, registry):
        """Retryable-Fehler zweimal → Abbruch mit error-Event."""
        adapter = _make_mock_adapter([])
        adapter.step.side_effect = ToolAdapterError("Server down", retryable=True)

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Double-Fail"))

        assert any(e.event_type == "error" for e in events)

    def test_tool_exception_caught(self, base_config):
        """Exception in Tool → error im Result, Agent laeuft weiter."""
        reg = ToolRegistry()
        reg.register(_FailingTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("failing_tool", {})),
                ToolStep(text="Konnte nicht helfen.", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        events = list(chain.query_stream("Fail-Test"))

        tr = next(e for e in events if e.event_type == "tool_result")
        assert tr.data["success"] is False
        assert any(e.event_type == "done" for e in events)

    def test_semantic_tool_result_is_not_retried(self, base_config):
        """Fachliche ToolResult-Fehler werden nicht zu Backend-Retries gemacht."""
        tool = _SemanticFailTool()
        reg = ToolRegistry()
        reg.register(tool)

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("semantic_fail", {})),
                ToolStep(text="Nicht gefunden.", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        events = list(chain.query_stream("Fail-Test"))

        assert tool.calls == 1
        tr = next(e for e in events if e.event_type == "tool_result")
        assert tr.data["success"] is False
        assert "Eintrag nicht gefunden" in tr.data["summary"]
        assert "Nach 3 Versuchen" not in tr.data["summary"]

    def test_retryable_tool_result_is_retried(self, base_config):
        """Transiente ToolResult-Fehler werden weiterhin begrenzt retried."""
        tool = _RetryableFailTool()
        reg = ToolRegistry()
        reg.register(tool)

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("retryable_fail", {})),
                ToolStep(text="Nicht erreichbar.", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        events = list(chain.query_stream("Retry-Test"))

        assert tool.calls == 3
        tr = next(e for e in events if e.event_type == "tool_result")
        assert tr.data["success"] is False
        assert "Nach 3 Versuchen" in tr.data["summary"]


# =====================================================================
# AgentChain — user_id Propagation
# =====================================================================


class TestAgentChainUserPropagation:
    def test_user_id_passed_to_tool(self, base_config, registry):
        """user_id wird an tool.execute() durchgereicht."""
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.definition.return_value = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        mock_tool.validate_params.return_value = (True, "ok")
        mock_tool.execute.return_value = ToolResult(
            success=True,
            data={},
            error=None,
            display_text="OK",
        )

        reg = ToolRegistry()
        reg.register(mock_tool)

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("test_tool", {})),
                ToolStep(text="Done", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, user_id=42)
        list(chain.query_stream("Test"))

        # user_id=42 muss an execute() uebergeben werden
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args[1]
        assert call_kwargs["user_id"] == 42

    def test_list_tools_called_without_user_groups(self, base_config):
        """list_tools() wird mit user_groups=None aufgerufen (Default)."""
        reg = MagicMock(spec=ToolRegistry)
        reg.list_tools.return_value = []

        adapter = _make_mock_adapter(
            [
                ToolStep(text="OK", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        list(chain.query_stream("Test"))

        reg.list_tools.assert_called_once_with(user_groups=None)


# =====================================================================
# AgentChain — Event-Ordering
# =====================================================================


class TestAgentChainEventOrdering:
    def test_ordering_tool_then_answer(self, base_config, registry):
        """Event-Reihenfolge: tool_call → tool_result → chunk → done."""
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("calculate", {"expression": "3+4"})),
                ToolStep(text="Sieben", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("3+4?"))
        types = [e.event_type for e in events]

        tc_idx = types.index("tool_call")
        tr_idx = types.index("tool_result")
        chunk_idx = types.index("chunk")
        done_idx = types.index("done")

        assert tc_idx < tr_idx < chunk_idx < done_idx

    def test_done_always_last(self, base_config, registry):
        adapter = _make_mock_adapter([ToolStep(text="OK", is_final=True)])
        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Test"))
        assert events[-1].event_type == "done"

    def test_iteration_numbers(self, base_config, registry):
        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("calculate", {"expression": "1+1"})),
                ToolStep(tool_call=("calculate", {"expression": "2+2"})),
                ToolStep(text="Fertig", is_final=True),
            ]
        )

        chain = AgentChain(base_config, registry, adapter=adapter, max_iterations=5)
        events = list(chain.query_stream("Multi"))
        tc_events = [e for e in events if e.event_type == "tool_call"]

        assert len(tc_events) == 2
        assert tc_events[0].data["iteration"] == 1
        assert tc_events[1].data["iteration"] == 2


# =====================================================================
# AgentChain — Config
# =====================================================================


class TestAgentChainConfig:
    def test_custom_system_prompt(self, base_config, registry):
        adapter = _make_mock_adapter([ToolStep(text="OK", is_final=True)])
        chain = AgentChain(
            base_config,
            registry,
            adapter=adapter,
            system_prompt_override="Custom Prompt {tools_prompt}",
        )
        assert chain._system_prompt == "Custom Prompt {tools_prompt}"

    def test_default_system_prompt_is_empty(self, base_config, registry):
        """Ohne Override und ohne Config-Prompt ist der System-Prompt leer."""
        adapter = _make_mock_adapter([ToolStep(text="OK", is_final=True)])
        chain = AgentChain(base_config, registry, adapter=adapter)
        assert chain._system_prompt == ""

    def test_max_iterations_from_param(self, base_config, registry):
        adapter = _make_mock_adapter([ToolStep(text="OK", is_final=True)])
        chain = AgentChain(base_config, registry, adapter=adapter, max_iterations=7)
        assert chain._max_iterations == 7


# =====================================================================
# AgentChain — Forced Initial Search
# =====================================================================


class _SearchDocumentsTool(BaseTool):
    """Test-Tool mit Name 'search_documents' fuer Forced-Search-Tests."""

    def definition(self):
        return ToolDefinition(
            name="search_documents",
            description="Durchsucht die Wissensbasis.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    async def execute(self, params, user_id=None):
        return ToolResult(
            success=True,
            data={
                "results": [],
                "sources": [
                    {"source": "rechnung.pdf", "score": 0.9, "excerpt": "Hetzner 49.00"},
                ],
            },
            error=None,
            display_text="1 Dokument gefunden: rechnung.pdf",
        )


class TestAgentChainForcedSearch:
    def test_forced_search_on_no_tool_call(self, base_config):
        """Wenn Iteration 0 keinen Tool-Call ergibt (Ollama), wird search_documents erzwungen."""
        reg = ToolRegistry()
        reg.register(_SearchDocumentsTool())

        adapter = _make_mock_adapter(
            [
                # Iteration 0: Kein Tool-Call (Ollama: is_final=False)
                ToolStep(text="Keine Informationen.", is_final=False),
                # Iteration 1: Nach forced search mit Ergebnissen
                ToolStep(text="Basierend auf den Ergebnissen...", is_final=False),
            ]
        )
        adapter.stream_final.return_value = iter(["Hetzner: 49 EUR."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=5)
        events = list(chain.query_stream("Hetzner Rechnungen"))

        event_types = [e.event_type for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types

        tc = next(e for e in events if e.event_type == "tool_call")
        assert tc.data["tool"] == "search_documents"
        assert tc.data["params"]["query"] == "Hetzner Rechnungen"

    def test_forced_search_emits_sources(self, base_config):
        """Forced search sammelt Sources korrekt."""
        reg = ToolRegistry()
        reg.register(_SearchDocumentsTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(text="", is_final=False),  # Iteration 0: kein Tool-Call
                ToolStep(text="Antwort", is_final=False),  # Iteration 1: nach forced search
            ]
        )
        adapter.stream_final.return_value = iter(["Finale Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=5)
        events = list(chain.query_stream("Suche"))

        sources = [e for e in events if e.event_type == "sources"]
        assert len(sources) == 1
        assert len(sources[0].data) == 1
        assert sources[0].data[0]["source"] == "rechnung.pdf"

    def test_no_forced_search_for_cloud_adapter(self, base_config):
        """Cloud-Adapter (is_final=True) loest keine forced search aus."""
        reg = ToolRegistry()
        reg.register(_SearchDocumentsTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(text="Cloud-Antwort.", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        events = list(chain.query_stream("Test"))

        event_types = [e.event_type for e in events]
        assert "tool_call" not in event_types
        assert "chunk" in event_types

    def test_no_forced_search_without_search_tool(self, base_config, registry):
        """Ohne search_documents im Registry keine forced search."""
        adapter = _make_mock_adapter(
            [
                ToolStep(text="Draft", is_final=False),
            ]
        )
        adapter.stream_final.return_value = iter(["Fallback."])

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("Test"))

        event_types = [e.event_type for e in events]
        assert "tool_call" not in event_types
        assert "chunk" in event_types

    def test_no_forced_search_on_iteration_1(self, base_config):
        """Forced search nur bei Iteration 0, nicht bei spaeteren Iterationen."""
        reg = ToolRegistry()
        reg.register(_SearchDocumentsTool())
        reg.register(CalculateTool())

        adapter = _make_mock_adapter(
            [
                # Iteration 0: Normaler Tool-Call
                ToolStep(tool_call=("calculate", {"expression": "1+1"})),
                # Iteration 1: Kein Tool-Call — KEINE forced search
                ToolStep(text="Fertig.", is_final=False),
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=5)
        events = list(chain.query_stream("Rechne"))

        tc_events = [e for e in events if e.event_type == "tool_call"]
        # Nur der calculate-Call, kein forced search_documents
        assert len(tc_events) == 1
        assert tc_events[0].data["tool"] == "calculate"

    def test_forced_search_feeds_results_to_adapter(self, base_config):
        """Forced search ruft adapter.add_tool_result() korrekt auf."""
        reg = ToolRegistry()
        reg.register(_SearchDocumentsTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(text="Hmm...", is_final=False),
                ToolStep(text="OK", is_final=False),
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=5)
        list(chain.query_stream("Frage"))

        adapter.add_tool_result.assert_called_once()
        call_args = adapter.add_tool_result.call_args
        assert call_args[0][0] == "search_documents"
        # result.data wird als JSON serialisiert — Inhalt pruefen
        assert "rechnung.pdf" in call_args[0][1]


# =====================================================================
# Hilfsfunktionen: _freeze_params, _deduplicate_sources
# =====================================================================


class TestFreezeParams:
    def test_same_dict_same_frozen(self):
        assert _freeze_params({"a": "1", "b": "2"}) == _freeze_params({"b": "2", "a": "1"})

    def test_different_dicts_different_frozen(self):
        assert _freeze_params({"a": "1"}) != _freeze_params({"a": "2"})

    def test_empty_dict(self):
        assert _freeze_params({}) == frozenset()

    def test_non_string_values_converted(self):
        """Nicht-String-Werte bleiben typsicher unterscheidbar."""
        f = _freeze_params({"k": 5, "x": True})
        assert ("k", ("int", 5)) in f
        assert ("x", ("bool", True)) in f

    def test_type_stable_values_do_not_collide(self):
        assert _freeze_params({"a": 1}) != _freeze_params({"a": "1"})

    def test_nested_values_are_order_stable(self):
        assert _freeze_params({"a": {"b": 1, "c": [2, "x"]}}) == _freeze_params(
            {
                "a": {"c": [2, "x"], "b": 1},
            }
        )


class TestDeduplicateSources:
    def test_no_duplicates(self):
        sources = [
            {"source": "a.txt", "score": 0.9},
            {"source": "b.txt", "score": 0.8},
        ]
        result = _deduplicate_sources(sources)
        assert len(result) == 2

    def test_removes_duplicates(self):
        sources = [
            {"source": "a.txt", "score": 0.9, "excerpt": "first"},
            {"source": "b.txt", "score": 0.8},
            {"source": "a.txt", "score": 0.7, "excerpt": "duplicate"},
        ]
        result = _deduplicate_sources(sources)
        assert len(result) == 2
        assert result[0]["excerpt"] == "first"  # Erstes Vorkommen behalten

    def test_four_identical_sources(self):
        """Vier identische Quellen → nur eine bleibt."""
        sources = [{"source": "same.pdf", "score": s} for s in [0.9, 0.85, 0.8, 0.7]]
        result = _deduplicate_sources(sources)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_empty_source_key_kept(self):
        """Quellen ohne source-Feld werden nicht dedupliziert."""
        sources = [
            {"excerpt": "no source 1"},
            {"excerpt": "no source 2"},
        ]
        result = _deduplicate_sources(sources)
        assert len(result) == 2

    def test_empty_list(self):
        assert _deduplicate_sources([]) == []


# =====================================================================
# AgentChain — Loop-Detection
# =====================================================================


class _DummySearchTool(BaseTool):
    """Test-Tool fuer Loop-Detection-Tests (liefert immer gleiche Sources)."""

    def definition(self):
        return ToolDefinition(
            name="search_documents",
            description="Suche.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    async def execute(self, params, user_id=None):
        return ToolResult(
            success=True,
            data={
                "results": [],
                "sources": [
                    {"source": "doc.txt", "score": 0.9, "excerpt": "Inhalt"},
                ],
            },
            error=None,
            display_text="1 Dokument gefunden",
        )


class TestAgentChainLoopDetection:
    def test_identical_consecutive_call_aborts(self, base_config):
        """Identischer Aufruf direkt hintereinander → Loop-Abbruch."""
        reg = ToolRegistry()
        reg.register(_DummySearchTool())

        adapter = _make_mock_adapter(
            [
                # Iteration 0: Erster search_documents-Call — wird ausgefuehrt
                ToolStep(tool_call=("search_documents", {"query": "VPN"})),
                # Iteration 1: Identischer Call — Loop-Detection greift
                ToolStep(tool_call=("search_documents", {"query": "VPN"})),
                # Wird nie erreicht:
                ToolStep(text="Antwort", is_final=True),
            ]
        )
        adapter.stream_final.return_value = iter(["Loop-Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("VPN einrichten"))

        # 1x normaler tool_call + 1x skipped tool_call (Loop-Break Events)
        tc_events = [e for e in events if e.event_type == "tool_call"]
        assert len(tc_events) == 2
        assert tc_events[1].data.get("skipped") is True

        # Finale Antwort trotzdem emittiert
        assert any(e.event_type == "chunk" for e in events)
        assert any(e.event_type == "done" for e in events)

        # add_tool_results: 1x echtes Ergebnis + 1x Loop-Warnung (Break)
        assert adapter.add_tool_results.call_count == 2
        loop_call = adapter.add_tool_results.call_args_list[1]
        loop_results = loop_call[0][0]
        assert "bereits" in loop_results[0][1]

    def test_same_tool_hard_limit_aborts(self, base_config):
        """Gleicher Tool-Name >= max_same_tool_calls → Abbruch."""
        reg = ToolRegistry()
        reg.register(_DummySearchTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "VPN"})),
                ToolStep(tool_call=("search_documents", {"query": "VPN Anleitung"})),
                ToolStep(tool_call=("search_documents", {"query": "VPN Setup"})),
                # Iteration 3: Vierter Call — Hard-Limit bei max_same_tool_calls=3
                ToolStep(tool_call=("search_documents", {"query": "VPN Config"})),
                ToolStep(text="Nie erreicht", is_final=True),
            ]
        )
        adapter.stream_final.return_value = iter(["Freq-Antwort."])

        # Explizit max_same_tool_calls=3 setzen (Default ist jetzt 5)
        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=3
        )
        events = list(chain.query_stream("VPN"))

        # 3 normale + 1 skipped tool_call
        tc_events = [e for e in events if e.event_type == "tool_call"]
        normal_calls = [e for e in tc_events if not e.data.get("skipped")]
        assert len(normal_calls) == 3

        assert any(e.event_type == "done" for e in events)

    def test_no_loop_for_different_tools(self, base_config):
        """Verschiedene Tools loesen keine Loop-Detection aus."""
        reg = ToolRegistry()
        reg.register(_DummySearchTool())
        reg.register(CalculateTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "test"})),
                ToolStep(tool_call=("calculate", {"expression": "1+1"})),
                ToolStep(tool_call=("search_documents", {"query": "anderer"})),
                ToolStep(text="Alles gut", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Multi-Tool"))

        tc_events = [e for e in events if e.event_type == "tool_call"]
        # Alle 3 Calls werden ausgefuehrt (kein Loop)
        assert len(tc_events) == 3
        assert any(e.event_type == "done" for e in events)

    def test_loop_emits_sources(self, base_config):
        """Bei Loop-Abbruch werden gesammelte Sources trotzdem emittiert."""
        reg = ToolRegistry()
        reg.register(_DummySearchTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "test"})),
                ToolStep(tool_call=("search_documents", {"query": "test"})),  # Loop
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        sources = [e for e in events if e.event_type == "sources"]
        assert len(sources) == 1
        assert len(sources[0].data) == 1  # doc.txt

    def test_loop_deduplicates_sources(self, base_config):
        """Loop-Abbruch mit duplizierten Quellen → dedupliziert."""
        reg = ToolRegistry()
        reg.register(_DummySearchTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "a"})),
                ToolStep(tool_call=("search_documents", {"query": "b"})),
                ToolStep(tool_call=("search_documents", {"query": "c"})),
                # 4. Aufruf → Frequency-Detection greift (3 vorherige)
                ToolStep(tool_call=("search_documents", {"query": "d"})),
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=3
        )
        events = list(chain.query_stream("Dupe-Test"))

        sources = [e for e in events if e.event_type == "sources"]
        assert len(sources) == 1
        # _DummySearchTool liefert jedes Mal doc.txt → 3 Calls, aber nur 1 Quelle
        assert len(sources[0].data) == 1
        assert sources[0].data[0]["source"] == "doc.txt"


# =====================================================================
# AgentChain — Quellen-Deduplizierung (Integration)
# =====================================================================


class _DuplicateSourceTool(BaseTool):
    """Test-Tool das bei jedem Call dieselbe Quelle zurueckgibt."""

    def definition(self):
        return ToolDefinition(
            name="dup_source_tool",
            description="Liefert immer doc.txt.",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )

    async def execute(self, params, user_id=None):
        return ToolResult(
            success=True,
            data={
                "results": [],
                "sources": [
                    {"source": "doc.txt", "score": 0.9, "excerpt": "Gleicher Inhalt"},
                ],
            },
            error=None,
            display_text="1 Dokument gefunden",
        )


class TestAgentChainSourceDedup:
    def test_duplicate_sources_deduplicated(self, base_config):
        """Gleiche Quelle aus mehreren Tool-Calls → nur einmal im Event."""
        reg = ToolRegistry()
        reg.register(_DuplicateSourceTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("dup_source_tool", {"q": "a"})),
                ToolStep(tool_call=("dup_source_tool", {"q": "b"})),
                ToolStep(text="Fertig", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=5)
        events = list(chain.query_stream("Dupe-Quellen"))

        sources = [e for e in events if e.event_type == "sources"]
        assert len(sources) == 1
        # doc.txt kommt 2x rein, wird auf 1x dedupliziert
        assert len(sources[0].data) == 1
        assert sources[0].data[0]["source"] == "doc.txt"

    def test_different_sources_kept(self, base_config):
        """Verschiedene Quellen bleiben alle erhalten."""
        reg = ToolRegistry()
        reg.register(_SourceTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("source_tool", {"q": "test"})),
                ToolStep(text="OK", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=5)
        events = list(chain.query_stream("Verschiedene"))

        sources = [e for e in events if e.event_type == "sources"]
        assert len(sources) == 1
        # _SourceTool liefert doc1.txt und doc2.txt — beide bleiben
        assert len(sources[0].data) == 2


# =====================================================================
# Issue #1055 — Tool-Call-Streaming muss Secrets in params redigieren
# =====================================================================


class _SecretEchoTool(BaseTool):
    """Test-Tool, das ein password als Parameter akzeptiert."""

    def definition(self):
        return ToolDefinition(
            name="secret_echo",
            description="Echo mit Geheim-Param.",
            parameters={
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "password": {"type": "string"},
                    "api_key": {"type": "string"},
                },
                "required": [],
            },
        )

    async def execute(self, params, user_id=None):
        return ToolResult(success=True, data={"ok": True}, error=None, display_text="OK")


class TestAgentChainSecretRedaction:
    """Issue #1055: tool_call-Events und _tool_events duerfen keine Klartext-Secrets enthalten."""

    def test_tool_call_event_redacts_secrets_in_params(self, base_config):
        """Im SSE-tool_call-Event sind Werte fuer password/api_key maskiert; harmlose Felder bleiben."""
        reg = ToolRegistry()
        reg.register(_SecretEchoTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=(
                        "secret_echo",
                        {
                            "username": "alice",
                            "password": "geheim123",
                            "api_key": "sk-abcdef",
                        },
                    )
                ),
                ToolStep(text="OK", is_final=True),
            ]
        )
        chain = AgentChain(base_config, reg, adapter=adapter)

        events = list(chain.query_stream("Test"))
        tc = next(e for e in events if e.event_type == "tool_call")

        from kiara.shared.constants import PASSWORD_MASK

        assert tc.data["params"]["username"] == "alice"
        assert tc.data["params"]["password"] == PASSWORD_MASK
        assert tc.data["params"]["api_key"] == PASSWORD_MASK

    def test_persisted_tool_events_redact_secrets(self, base_config):
        """Auch der persistierbare _tool_events-Buffer enthaelt keine Klartext-Secrets."""
        reg = ToolRegistry()
        reg.register(_SecretEchoTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("secret_echo", {"username": "bob", "password": "topsecret"})),
                ToolStep(text="OK", is_final=True),
            ]
        )
        chain = AgentChain(base_config, reg, adapter=adapter)
        list(chain.query_stream("Test"))

        from kiara.shared.constants import PASSWORD_MASK

        assert chain._tool_events, "tool_events-Buffer leer"
        persisted = chain._tool_events[0]
        assert persisted["params"]["username"] == "bob"
        assert persisted["params"]["password"] == PASSWORD_MASK
        assert "topsecret" not in str(persisted)
