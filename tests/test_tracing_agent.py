"""Tests fuer Tracing-Instrumentierung der Agent-Chain (Sprint 17.3)."""

import threading
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kiara.generation.agent_chain import AgentChain, AgentEvent
from kiara.generation.tool_adapter import BaseToolAdapter, ToolAdapterError, ToolStep
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.builtin_tools import CalculateTool
from kiara.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# RecordingTracer / RecordingSpan
# ---------------------------------------------------------------------------


class RecordingSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        pass

    def add_event(self, name, attributes=None):
        pass

    def get_span_context(self):
        ctx = MagicMock()
        ctx.trace_id = 0xABCDEF1234567890ABCDEF1234567890
        ctx.span_id = 0x1234567890ABCDEF
        return ctx

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class RecordingTracer:
    def __init__(self):
        self.spans = []
        self._lock = threading.Lock()

    def start_as_current_span(self, name, **kwargs):
        span = RecordingSpan(name)
        with self._lock:
            self.spans.append(span)
        return span

    def start_span(self, name, **kwargs):
        span = RecordingSpan(name)
        with self._lock:
            self.spans.append(span)
        return span

    def get_span_names(self):
        with self._lock:
            return [s.name for s in self.spans]

    def get_span(self, name):
        with self._lock:
            for s in self.spans:
                if s.name == name:
                    return s
        return None

    def get_spans(self, name):
        with self._lock:
            return [s for s in self.spans if s.name == name]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    return {
        "chat": {"agent": {"max_iterations": 3, "temperature": 0.3}},
    }


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(CalculateTool())
    return reg


def _make_mock_adapter(steps, usage=None):
    adapter = MagicMock(spec=BaseToolAdapter)
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


class _SearchDocumentsTool(BaseTool):
    def definition(self):
        return ToolDefinition(
            name="search_documents",
            description="Sucht Dokumente.",
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
                "sources": [{"source": "doc.txt", "score": 0.9, "excerpt": "Test"}],
            },
            error=None,
            display_text="1 Dokument gefunden",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentChainTracing:
    """Agent-Chain Tracing-Tests."""

    def test_agent_query_stream_persists_last_trace_id(self, base_config, registry):
        """query_stream() setzt last_trace_id auf die Root-Trace-ID."""
        recorder = RecordingTracer()
        adapter = _make_mock_adapter(
            [
                ToolStep(text="Finale Antwort", tool_call=None, is_final=True),
            ]
        )
        chain = AgentChain(
            config=base_config,
            adapter=adapter,
            tool_registry=registry,
            system_prompt_override="Du bist ein Test-Agent.",
        )

        with patch("kiara.generation.agent_chain._tracer", recorder):
            events = list(chain.query_stream("Was ist 2+2?"))

        assert chain.last_trace_id is not None
        assert len(chain.last_trace_id) == 32
        assert "agent_query" in recorder.get_span_names()
        assert any(e.event_type == "done" for e in events)

    def test_llm_step_span_marks_retry_true_on_second_attempt(self, base_config, registry):
        """Bei retryable Fehler wird retried=True im llm_step-Span gesetzt."""
        recorder = RecordingTracer()
        adapter = _make_mock_adapter([])
        # Erster step: retryable Fehler, zweiter step: Erfolg
        adapter.step.side_effect = [
            ToolAdapterError("timeout", retryable=True),
            ToolStep(text="OK", tool_call=None, is_final=True),
        ]
        chain = AgentChain(
            config=base_config,
            adapter=adapter,
            tool_registry=registry,
            system_prompt_override="Test",
        )

        with patch("kiara.generation.agent_chain._tracer", recorder):
            with patch("kiara.generation.agent_chain.time.sleep"):
                events = list(chain.query_stream("Test"))

        step_spans = recorder.get_spans("llm_step")
        assert len(step_spans) >= 1
        assert step_spans[0].attributes.get("retried") is True

    def test_forced_search_emits_traced_tool_call(self, base_config):
        """Forced Search erzeugt forced_search und tool_call Spans."""
        recorder = RecordingTracer()
        reg = ToolRegistry()
        reg.register(_SearchDocumentsTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(text="Hmm...", tool_call=None, is_final=False),
                ToolStep(text="Antwort", tool_call=None, is_final=True),
            ]
        )
        chain = AgentChain(
            config=base_config,
            adapter=adapter,
            tool_registry=reg,
            system_prompt_override="Test",
            force_initial_search=True,
        )

        with patch("kiara.generation.agent_chain._tracer", recorder):
            events = list(chain.query_stream("Suche nach KIara"))

        span_names = recorder.get_span_names()
        assert "forced_search" in span_names
        assert "tool_call: search_documents" in span_names

    def test_loop_abort_still_finishes_with_final_answer_span(self, base_config, registry):
        """Bei Loop-Abbruch existiert ein final_answer-Span."""
        recorder = RecordingTracer()
        # Gleicher Tool-Call in jeder Iteration → Loop-Detection
        steps = [
            ToolStep(text="", tool_call=("calculate", {"expression": "1+1"}), is_final=False),
            ToolStep(text="", tool_call=("calculate", {"expression": "1+1"}), is_final=False),
            ToolStep(text="", tool_call=("calculate", {"expression": "1+1"}), is_final=False),
        ]
        adapter = _make_mock_adapter(steps)
        chain = AgentChain(
            config=base_config,
            adapter=adapter,
            tool_registry=registry,
            system_prompt_override="Test",
        )

        with patch("kiara.generation.agent_chain._tracer", recorder):
            events = list(chain.query_stream("Berechne 1+1"))

        assert "final_answer" in recorder.get_span_names()

    def test_parallel_tool_calls_keep_parent_context(self, base_config):
        """Parallele Tool-Calls erzeugen tool_call-Spans unter agent_query."""
        recorder = RecordingTracer()
        reg = ToolRegistry()
        reg.register(CalculateTool())
        reg.register(_SearchDocumentsTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("calculate", {"expression": "1+1"}),
                    tool_use_id="tu_1",
                    parallel_calls=[("search_documents", {"query": "Test"}, "tu_2")],
                ),
                ToolStep(text="Ergebnis", tool_call=None, is_final=True),
            ]
        )
        chain = AgentChain(
            config=base_config,
            adapter=adapter,
            tool_registry=reg,
            system_prompt_override="Test",
        )

        with patch("kiara.generation.agent_chain._tracer", recorder):
            events = list(chain.query_stream("Berechne und suche"))

        span_names = recorder.get_span_names()
        # Beide tool_call-Spans muessen existieren
        tool_spans = [s for s in span_names if s.startswith("tool_call:")]
        assert len(tool_spans) == 2
        assert "tool_call: calculate" in span_names
        assert "tool_call: search_documents" in span_names
        # Alle Spans tragen dieselbe trace_id (via RecordingSpan.get_span_context)
        assert "agent_query" in span_names

    def test_stream_agent_logs_trace_id_from_chain_property(self, base_config):
        """_stream_agent reicht chain.last_trace_id an _log_query_audit weiter."""

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="chunk", data={"text": "Antwort"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "trace-agent-789"
        mock_chain.last_conversation = ""

        backend = {"url": "http://localhost:11434", "model": "test", "provider": "ollama"}

        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            import asyncio

            from kiara.web.api_chat import _stream_agent

            config = {
                "cloud": {"enabled": False},
                "chat": {"agent": {"max_iterations": 3}},
            }
            data = MagicMock()
            data.question = "Test"
            data.k = 5
            data.think = False
            data.datasource_ids = None
            user = {"id": 1, "username": "test"}

            async def _consume():
                collected = []
                async for chunk in _stream_agent(config, data, 1, user):
                    collected.append(chunk)
                return collected

            asyncio.run(_consume())
            mock_audit.assert_called_once()
            assert mock_audit.call_args.kwargs["trace_id"] == "trace-agent-789"
