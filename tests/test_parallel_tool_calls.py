"""Tests fuer parallele Tool-Calls (Fix 1): Adapter + AgentChain.

Verifiziert, dass ALLE LLM-angeforderten Tool-Calls parallel
ausgefuehrt werden (nicht nur der erste) und Adapter-Methoden
korrekt arbeiten.
"""

import sys
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kiara.generation.adapters.base import BaseToolAdapter, ToolStep
from kiara.generation.agent_chain import AgentChain
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.registry import ToolRegistry

# =====================================================================
# Hilfsfunktionen
# =====================================================================


class _SlowTool(BaseTool):
    """Test-Tool mit konfigurierbarer Verzoegerung."""

    def __init__(self, name="slow_tool", delay=0.1):
        self._name = name
        self._delay = delay

    def definition(self):
        return ToolDefinition(
            name=self._name,
            description="Langsames Test-Tool",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )

    async def execute(self, params, user_id=None):
        time.sleep(self._delay)
        return ToolResult(
            success=True,
            data={
                "results": [
                    {
                        "text": f"Result for {params['query']}",
                        "metadata": {"source": f"{params['query']}.txt", "chunk_index": 0},
                    }
                ]
            },
            error=None,
            display_text=f"Found: {params['query']}",
        )


def _make_mock_adapter(steps):
    """Erzeugt einen Mock-Adapter mit vordefinierten Steps."""
    adapter = MagicMock(spec=BaseToolAdapter)
    adapter.step.side_effect = iter(steps)
    adapter.init_conversation.return_value = None
    adapter.add_tool_result.return_value = None
    adapter.add_tool_results.return_value = None
    adapter.stream_final.return_value = iter(["Antwort."])
    adapter.deanonymize_params.side_effect = lambda p: {**p}
    adapter.last_usage = {"input": 0, "output": 0}
    type(adapter).result_truncate_limit = PropertyMock(return_value=24000)
    return adapter


# =====================================================================
# Anthropic Adapter: parallel_calls
# =====================================================================


class TestAnthropicParallelCalls:
    def test_step_two_tool_use_blocks(self):
        """step() mit 2 tool_use-Bloecken → parallel_calls befuellt."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

            # Mock-Response mit 2 tool_use-Bloecken
            block1 = MagicMock()
            block1.type = "tool_use"
            block1.name = "search_documents"
            block1.input = {"query": "A"}
            block1.id = "tu_1"

            block2 = MagicMock()
            block2.type = "tool_use"
            block2.name = "search_documents"
            block2.input = {"query": "B"}
            block2.id = "tu_2"

            resp = MagicMock()
            resp.content = [block1, block2]
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50

            mock_client.messages.create.return_value = resp
            adapter._client = mock_client
            adapter.init_conversation(
                "Test",
                "System",
                [
                    ToolDefinition(
                        name="search_documents",
                        description="S",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
            )

            step = adapter.step()

        assert step.tool_call == ("search_documents", {"query": "A"})
        assert step.tool_use_id == "tu_1"
        assert step.parallel_calls is not None
        assert len(step.parallel_calls) == 1
        assert step.parallel_calls[0] == ("search_documents", {"query": "B"}, "tu_2")

    def test_step_single_tool_use(self):
        """step() mit 1 tool_use-Block → parallel_calls ist None."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

            block1 = MagicMock()
            block1.type = "tool_use"
            block1.name = "search_documents"
            block1.input = {"query": "A"}
            block1.id = "tu_1"

            resp = MagicMock()
            resp.content = [block1]
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50

            mock_client.messages.create.return_value = resp
            adapter._client = mock_client
            adapter.init_conversation("Test", "System", [])

            step = adapter.step()

        assert step.tool_call == ("search_documents", {"query": "A"})
        assert step.parallel_calls is None

    def test_add_tool_results_one_user_message(self):
        """add_tool_results() sendet alle Ergebnisse in EINER user-Message."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        adapter._messages = []
        results = [
            ("search_documents", "Result A", "tu_1"),
            ("search_documents", "Result B", "tu_2"),
        ]
        adapter.add_tool_results(results)

        assert len(adapter._messages) == 1
        msg = adapter._messages[0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["tool_use_id"] == "tu_1"
        assert msg["content"][1]["tool_use_id"] == "tu_2"

    def test_skipped_tool_ids_removed(self):
        """_skipped_tool_ids existiert nicht mehr als Attribut."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        assert not hasattr(adapter, "_skipped_tool_ids")

    def test_step_max_tokens_during_tool_use_fails_closed(self):
        """stop_reason=max_tokens bei tool_use → ToolAdapterError, kein History-Append."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter
        from kiara.generation.adapters.base import ToolAdapterError

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

            block = MagicMock()
            block.type = "tool_use"
            block.name = "search_documents"
            block.input = {"query": "trun"}  # Truncated payload
            block.id = "tu_1"

            resp = MagicMock()
            resp.content = [block]
            resp.stop_reason = "max_tokens"
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50

            mock_client.messages.create.return_value = resp
            adapter._client = mock_client
            adapter.init_conversation(
                "Test",
                "System",
                [
                    ToolDefinition(
                        name="search_documents",
                        description="S",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
            )

            messages_before = list(adapter._messages)
            with pytest.raises(ToolAdapterError) as excinfo:
                adapter.step()

        assert "max_tokens" in str(excinfo.value)
        assert excinfo.value.retryable is False
        # History darf nicht um die unvollstaendige assistant-Message erweitert sein
        assert adapter._messages == messages_before

    def test_step_max_tokens_text_only_passes(self):
        """stop_reason=max_tokens ohne tool_use → finale Antwort, kein Fehler."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Eine abgeschnittene"

            resp = MagicMock()
            resp.content = [text_block]
            resp.stop_reason = "max_tokens"
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50

            mock_client.messages.create.return_value = resp
            adapter._client = mock_client
            adapter.init_conversation("Test", "System", [])

            step = adapter.step()

        assert step.is_final is True
        assert step.tool_call is None


# =====================================================================
# OpenAI Adapter: parallel_calls
# =====================================================================


class TestOpenAIParallelCalls:
    def test_step_two_tool_calls(self):
        """step() mit 2 tool_calls → parallel_calls befuellt."""
        from kiara.generation.adapters.openai import OpenAIToolAdapter

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict(sys.modules, {"openai": mock_openai}):
            adapter = OpenAIToolAdapter(
                {"api_key": "sk", "model": "gpt-4", "provider": "openai"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

            tc1 = MagicMock()
            tc1.function.name = "search_documents"
            tc1.function.arguments = '{"query": "A"}'
            tc1.id = "call_1"

            tc2 = MagicMock()
            tc2.function.name = "search_documents"
            tc2.function.arguments = '{"query": "B"}'
            tc2.id = "call_2"

            choice = MagicMock()
            choice.finish_reason = "tool_calls"
            choice.message.tool_calls = [tc1, tc2]
            choice.message.content = ""
            choice.message.model_dump.return_value = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search_documents", "arguments": '{"query":"A"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "search_documents", "arguments": '{"query":"B"}'},
                    },
                ],
            }

            resp = MagicMock()
            resp.choices = [choice]
            resp.usage.prompt_tokens = 100
            resp.usage.completion_tokens = 50

            mock_client.chat.completions.create.return_value = resp
            adapter._client = mock_client
            adapter.init_conversation("Test", "System", [])

            step = adapter.step()

        assert step.tool_call == ("search_documents", {"query": "A"})
        assert step.parallel_calls is not None
        assert len(step.parallel_calls) == 1
        assert step.parallel_calls[0][2] == "call_2"

    def test_add_tool_results_separate_messages(self):
        """add_tool_results() sendet separate tool-Messages pro Result."""
        from kiara.generation.adapters.openai import OpenAIToolAdapter

        with patch.dict(sys.modules, {"openai": MagicMock()}):
            adapter = OpenAIToolAdapter(
                {"api_key": "sk", "model": "gpt-4", "provider": "openai"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        adapter._messages = []
        results = [
            ("search_documents", "Result A", "call_1"),
            ("search_documents", "Result B", "call_2"),
        ]
        adapter.add_tool_results(results)

        assert len(adapter._messages) == 2
        assert adapter._messages[0]["role"] == "tool"
        assert adapter._messages[0]["tool_call_id"] == "call_1"
        assert adapter._messages[1]["tool_call_id"] == "call_2"

    def test_skipped_tool_ids_removed(self):
        """_skipped_tool_ids existiert nicht mehr als Attribut."""
        from kiara.generation.adapters.openai import OpenAIToolAdapter

        with patch.dict(sys.modules, {"openai": MagicMock()}):
            adapter = OpenAIToolAdapter(
                {"api_key": "sk", "model": "gpt-4", "provider": "openai"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        assert not hasattr(adapter, "_skipped_tool_ids")


# =====================================================================
# Ollama Adapter: Backward-Kompatibilitaet
# =====================================================================


class TestOllamaBackwardCompat:
    def test_default_add_tool_results_delegates(self):
        """Default add_tool_results() delegiert an add_tool_result() (mit raw_chunks)."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        with patch("ollama.Client"), patch("httpx.Timeout"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        adapter.init_conversation("Test", "System", [])

        # Spy auf add_tool_result
        original_add = adapter.add_tool_result
        call_log = []

        def spy_add(tool_name, result_text, tool_use_id=None, raw_chunks=None):
            call_log.append((tool_name, result_text, tool_use_id, raw_chunks))
            original_add(tool_name, result_text, tool_use_id, raw_chunks=raw_chunks)

        adapter.add_tool_result = spy_add

        chunks = [{"text": "c1", "metadata": {"source": "s1", "chunk_index": 0}}]
        adapter.add_tool_results(
            [("search_documents", "Result", None)],
            raw_chunks_map={None: chunks},
        )

        assert len(call_log) == 1
        assert call_log[0][0] == "search_documents"
        assert call_log[0][3] == chunks  # raw_chunks korrekt uebergeben

    def test_add_tool_result_preserves_prepared_content(self):
        """Der Adapter kuerzt bereits aufbereitete Tool-Inhalte nicht erneut."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter
        from kiara.shared.constants import AGENT_RESULT_TRUNCATE_OLLAMA

        with patch("ollama.Client"), patch("httpx.Timeout"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        adapter.init_conversation("Test", "System", [])
        assert adapter.result_truncate_limit == AGENT_RESULT_TRUNCATE_OLLAMA

        # Zentral budgetierter Text bleibt auch oberhalb des alten Adapterlimits erhalten.
        long_text = "x" * (AGENT_RESULT_TRUNCATE_OLLAMA + 100)
        adapter.add_tool_result("search_documents", long_text)

        stored = adapter._previous_results[-1]["result"]
        assert stored == long_text

    def test_final_prompt_uses_prepared_results(self):
        """Der finale Prompt verwendet die bereits aufbereiteten Ergebnisinhalte."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        with patch("ollama.Client"), patch("httpx.Timeout"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        adapter.init_conversation("Test", "System", [])

        chunks1 = [{"text": "c1", "metadata": {"source": "s1", "chunk_index": 0}}]
        chunks2 = [{"text": "c2", "metadata": {"source": "s2", "chunk_index": 0}}]

        adapter.add_tool_result("search_documents", "R1", raw_chunks=chunks1)
        adapter.add_tool_result("search_documents", "R2", raw_chunks=chunks2)

        assert "R1" in adapter._build_final_prompt()
        assert "R2" in adapter._build_final_prompt()

    def test_raw_chunks_map_none_key(self):
        """raw_chunks_map mit None-Key wird korrekt aufgeloest."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        with patch("ollama.Client"), patch("httpx.Timeout"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        adapter.init_conversation("Test", "System", [])

        chunks = [{"text": "c1", "metadata": {"source": "s1", "chunk_index": 0}}]
        adapter.add_tool_results(
            [("search_documents", "Result", None)],
            raw_chunks_map={None: chunks},
        )

        assert adapter._previous_results[-1]["result"] == "Result"
        assert "Result" in adapter._build_final_prompt()


# =====================================================================
# AgentChain: Parallele Ausfuehrung
# =====================================================================


class TestAgentChainParallel:
    @pytest.fixture
    def base_config(self):
        return {"chat": {"agent": {"max_iterations": 10, "temperature": 0.3}}}

    def test_run_tool_is_method(self):
        """_run_tool() ist Methode auf AgentChain."""
        assert hasattr(AgentChain, "_run_tool")
        assert callable(AgentChain._run_tool)

    def test_parallel_calls_both_executed(self, base_config):
        """Parallele Calls werden beide ausgefuehrt und Events ge-yieldet."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}),
                    tool_use_id="tu_1",
                    parallel_calls=[("search_documents", {"query": "B"}, "tu_2")],
                ),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        tool_calls = [
            e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")
        ]
        assert len(tool_calls) == 2
        assert tool_calls[0].data["params"]["query"] == "A"
        assert tool_calls[1].data["params"]["query"] == "B"

    def test_parallel_execution_timing(self, base_config):
        """ThreadPoolExecutor fuehrt Calls parallel aus (schneller als sequentiell)."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0.2))

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}),
                    tool_use_id="tu_1",
                    parallel_calls=[
                        ("search_documents", {"query": "B"}, "tu_2"),
                        ("search_documents", {"query": "C"}, "tu_3"),
                    ],
                ),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)

        start = time.monotonic()
        events = list(chain.query_stream("Test"))
        elapsed = time.monotonic() - start

        tool_calls = [e for e in events if e.event_type == "tool_call"]
        assert len(tool_calls) == 3

        # Parallel: ~0.2s statt ~0.6s sequentiell
        assert elapsed < 0.5, f"Parallel execution took {elapsed:.2f}s (expected <0.5s)"

    def test_max_workers_capped(self):
        """max_workers nie groesser als _MAX_PARALLEL_TOOL_WORKERS (4)."""
        from kiara.generation.agent_chain import _MAX_PARALLEL_TOOL_WORKERS

        assert _MAX_PARALLEL_TOOL_WORKERS == 4

    def test_all_parallel_calls_in_recent_calls(self, base_config):
        """Alle parallelen Calls erscheinen in recent_calls (Loop-Detection).

        Beweis: Nach Batch [A, B] landet B als letzter Eintrag in recent_calls.
        Ein erneuter Call B → identical_repeat → gefiltert.
        """
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}),
                    tool_use_id="tu_1",
                    parallel_calls=[("search_documents", {"query": "B"}, "tu_2")],
                ),
                # B nochmal → identical_repeat (B war letzter Eintrag in recent_calls)
                ToolStep(
                    tool_call=("search_documents", {"query": "B"}),
                    tool_use_id="tu_3",
                ),
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # B wurde als identical_repeat gefiltert → beweist dass B in recent_calls war
        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1
        assert skipped[0].data["tool"] == "search_documents"

    def test_batch_duplicates_deduplicated(self, base_config):
        """Duplikate im selben Batch werden dedupliziert."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}),
                    tool_use_id="tu_1",
                    parallel_calls=[("search_documents", {"query": "A"}, "tu_2")],
                ),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # Nur 1 echter Tool-Call (der Duplikat wird gefiltert)
        tool_calls = [
            e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")
        ]
        assert len(tool_calls) == 1

    def test_single_call_uses_add_tool_results(self, base_config):
        """Einzel-Call geht auch ueber add_tool_results() (kein Sonderweg)."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}),
                    tool_use_id="tu_1",
                ),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # add_tool_results wurde aufgerufen (nicht add_tool_result)
        assert adapter.add_tool_results.call_count == 1
        assert adapter.add_tool_result.call_count == 0

    def test_budget_uses_adapter_property(self, base_config):
        """Budget-Tracking nutzt adapter.result_truncate_limit (Property)."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}),
                    tool_use_id="tu_1",
                ),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )
        # Custom-Limit setzen
        type(adapter).result_truncate_limit = PropertyMock(return_value=1000)

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        assert any(e.event_type == "done" for e in events)

    def test_looped_calls_yield_skipped_events(self, base_config):
        """Geloopte Calls yielden tool_call Event mit skipped=True."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_1"),
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_2"
                ),  # identical_repeat → skipped
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1

    def test_complete_loop_break_yields_skipped(self, base_config):
        """Kompletter Loop-Break yieldet skipped-Events fuer ALLE gefilterten Calls."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_1"),
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}),
                    tool_use_id="tu_2",
                    parallel_calls=[("search_documents", {"query": "A"}, "tu_3")],
                ),
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        # Batch 2: 1 unique Call (A) → identical_repeat → gefiltert,
        # plus 1 batch-dup (A) → auch uebersprungen.
        # Nur der unique Call bekommt skipped-Event.
        assert len(skipped) >= 1

    def test_ollama_no_parallel_still_works(self, base_config):
        """Ollama-Adapter (kein parallel) funktioniert weiterhin."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "A"})),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )
        type(adapter).result_truncate_limit = PropertyMock(return_value=4000)

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        tool_calls = [e for e in events if e.event_type == "tool_call"]
        assert len(tool_calls) == 1
        assert any(e.event_type == "done" for e in events)

    def test_exception_in_run_tool_returns_error(self, base_config):
        """Exception in _run_tool → Error-Dict statt Crash."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_1"),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )
        # deanonymize_params wirft Exception
        adapter.deanonymize_params.side_effect = RuntimeError("Anonymizer-Fehler")

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # Kein Crash — Agent faengt Exception ab und sendet Error-Result
        results = [e for e in events if e.event_type == "tool_result"]
        assert len(results) >= 1

    def test_exception_in_execute_tool_uses_deanon_params(self, base_config):
        """Exception in _execute_tool → params bleibt de-anonymisiert (nicht cp)."""
        reg = ToolRegistry()
        reg.register(_SlowTool("search_documents", delay=0))

        # deanonymize_params liefert transformierte Params
        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "[Firma_001]"}), tool_use_id="tu_1"
                ),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )
        adapter.deanonymize_params.side_effect = lambda p: {
            k: v.replace("[Firma_001]", "Hetzner") if isinstance(v, str) else v
            for k, v in p.items()
        }

        # Tool-Execute wirft Exception
        class _FailTool(BaseTool):
            def definition(self):
                return ToolDefinition(
                    name="search_documents",
                    description="Fail-Tool",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                )

            async def execute(self, params, user_id=None):
                raise RuntimeError("DB-Fehler")

        reg._tools = {"search_documents": _FailTool()}

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # tool_call Event zeigt de-anonymisierte Params
        tc_events = [e for e in events if e.event_type == "tool_call"]
        assert len(tc_events) >= 1
        assert tc_events[0].data["params"]["query"] == "Hetzner"

    def test_raw_chunks_ollama_tid_none(self, base_config):
        """raw_chunks werden fuer Ollama (tid=None) korrekt uebergeben."""
        reg = ToolRegistry()

        # Tool das raw_chunks zurueckgibt
        class _ChunkTool(BaseTool):
            def definition(self):
                return ToolDefinition(
                    name="search_documents",
                    description="Chunk-Tool",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                )

            async def execute(self, params, user_id=None):
                return ToolResult(
                    success=True,
                    data={
                        "results": [
                            {"text": "chunk1", "metadata": {"source": "doc.txt", "chunk_index": 0}}
                        ]
                    },
                    error=None,
                    display_text="Found chunk1",
                )

        reg.register(_ChunkTool())

        # Ollama-Stil: kein tool_use_id (None)
        adapter = _make_mock_adapter(
            [
                ToolStep(
                    tool_call=("search_documents", {"query": "A"}), tool_use_id=None
                ),  # Ollama hat keine tool_use_ids
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ]
        )
        type(adapter).result_truncate_limit = PropertyMock(return_value=4000)

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # add_tool_results wurde mit raw_chunks_map aufgerufen
        assert adapter.add_tool_results.call_count == 1
        call_args = adapter.add_tool_results.call_args
        # raw_chunks_map (zweites Argument)
        raw_map = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("raw_chunks_map")
        # Key ist None (Ollama tid) und Chunks sind vorhanden
        assert raw_map is not None
        assert None in raw_map
        assert len(raw_map[None]) >= 1
