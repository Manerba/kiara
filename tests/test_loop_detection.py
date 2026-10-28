"""Tests fuer Loop-Detection (Fix 2): Per-Call-Filterung und konfigurierbare Limits.

Verifiziert die Loop-Erkennung im AgentChain: identical_repeat,
same_params_count, same_tool_count Hard-Limit, max_same_tool_calls=0,
und gemischte Batch-Szenarien.
"""

import logging
from unittest.mock import MagicMock, PropertyMock

import pytest

from kiara.generation.adapters.base import BaseToolAdapter, ToolStep
from kiara.generation.agent_chain import AgentChain, _freeze_params
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.registry import ToolRegistry

# =====================================================================
# Hilfs-Tools
# =====================================================================


class _DummyTool(BaseTool):
    """Minimal-Tool fuer Loop-Tests."""

    def __init__(self, name="search_documents"):
        self._name = name

    def definition(self):
        return ToolDefinition(
            name=self._name,
            description=f"Test-Tool {self._name}",
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
                "results": [{"text": "Result", "metadata": {"source": "doc.txt", "chunk_index": 0}}]
            },
            error=None,
            display_text="Found something",
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


@pytest.fixture
def base_config():
    return {"chat": {"agent": {"max_iterations": 20, "temperature": 0.3}}}


# =====================================================================
# _freeze_params
# =====================================================================


class TestFreezeParams:
    def test_basic_dict(self):
        """_freeze_params macht Dict hashbar."""
        fp = _freeze_params({"query": "test", "k": 5})
        assert isinstance(fp, frozenset)
        assert ("query", ("str", "test")) in fp
        assert ("k", ("int", 5)) in fp

    def test_identical_dicts_equal(self):
        """Gleiche Dicts → gleiche frozensets."""
        assert _freeze_params({"a": 1, "b": 2}) == _freeze_params({"b": 2, "a": 1})

    def test_different_dicts_different(self):
        """Verschiedene Dicts → verschiedene frozensets."""
        assert _freeze_params({"a": 1}) != _freeze_params({"a": 2})
        assert _freeze_params({"a": 1}) != _freeze_params({"a": "1"})


# =====================================================================
# Identical Repeat
# =====================================================================


class TestIdenticalRepeat:
    def test_identical_params_twice_second_filtered(self, base_config):
        """Identische Parameter 2x hintereinander → Zweiter Call gefiltert."""
        reg = ToolRegistry()
        reg.register(_DummyTool())

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_1"),
                ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_2"),
            ]
        )
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1

        normal = [e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")]
        assert len(normal) == 1


# =====================================================================
# Same Params Count (Position 1, 4, 7)
# =====================================================================


class TestSameParamsCount:
    def test_same_params_position_1_4_7(self, base_config):
        """Identische Parameter an Position 1, 4, 7 → Call bei Position 7 gefiltert."""
        reg = ToolRegistry()
        reg.register(_DummyTool())

        # Position 1: search(query=A) — ok
        # Position 2: search(query=B) — ok
        # Position 3: search(query=C) — ok
        # Position 4: search(query=A) — same_params=1, < 2 → ok
        # Position 5: search(query=D) — ok
        # Position 6: search(query=E) — ok
        # Position 7: search(query=A) — same_params=2, >= 2 → gefiltert!
        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=("search_documents", {"query": "B"}), tool_use_id="t2"),
            ToolStep(tool_call=("search_documents", {"query": "C"}), tool_use_id="t3"),
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t4"),
            ToolStep(tool_call=("search_documents", {"query": "D"}), tool_use_id="t5"),
            ToolStep(tool_call=("search_documents", {"query": "E"}), tool_use_id="t6"),
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t7"),
        ]
        adapter = _make_mock_adapter(steps)
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=0
        )
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1

        normal = [e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")]
        assert len(normal) == 6


# =====================================================================
# Hard-Limit: max_same_tool_calls
# =====================================================================


class TestHardLimit:
    def test_different_params_5x_no_filter(self, base_config):
        """Verschiedene Parameter 5x → kein Call gefiltert (max_same_tool_calls=5)."""
        reg = ToolRegistry()
        reg.register(_DummyTool())

        steps = [
            ToolStep(tool_call=("search_documents", {"query": f"Q{i}"}), tool_use_id=f"t{i}")
            for i in range(5)
        ]
        steps.append(ToolStep(tool_call=None, text="Fertig", is_final=True))

        adapter = _make_mock_adapter(steps)
        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=5
        )
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 0

        normal = [e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")]
        assert len(normal) == 5

    def test_different_params_6x_sixth_filtered(self, base_config):
        """Verschiedene Parameter 6x → 6. Call gefiltert (same_tool_count=5 >= 5)."""
        reg = ToolRegistry()
        reg.register(_DummyTool())

        steps = [
            ToolStep(tool_call=("search_documents", {"query": f"Q{i}"}), tool_use_id=f"t{i}")
            for i in range(6)
        ]
        adapter = _make_mock_adapter(steps)
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=5
        )
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1

        normal = [e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")]
        assert len(normal) == 5


# =====================================================================
# max_same_tool_calls=0: Kein Hard-Limit
# =====================================================================


class TestNoHardLimit:
    def test_no_hard_limit_but_identical_repeat(self, base_config):
        """max_same_tool_calls=0 → kein Hard-Limit, aber identical_repeat greift."""
        reg = ToolRegistry()
        reg.register(_DummyTool())

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(
                tool_call=("search_documents", {"query": "A"}), tool_use_id="t2"
            ),  # identical_repeat → gefiltert
        ]
        adapter = _make_mock_adapter(steps)
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=0
        )
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1

    def test_no_hard_limit_but_same_params_ge2(self, base_config):
        """max_same_tool_calls=0 → kein Hard-Limit, aber same_params >= 2 greift."""
        reg = ToolRegistry()
        reg.register(_DummyTool())

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=("search_documents", {"query": "B"}), tool_use_id="t2"),
            ToolStep(
                tool_call=("search_documents", {"query": "A"}), tool_use_id="t3"
            ),  # same_params=1 → ok
            ToolStep(
                tool_call=("search_documents", {"query": "A"}), tool_use_id="t4"
            ),  # same_params=2 → gefiltert!
        ]
        adapter = _make_mock_adapter(steps)
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=0
        )
        events = list(chain.query_stream("Test"))

        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1

        normal = [e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")]
        assert len(normal) == 3


# =====================================================================
# Log-Meldung
# =====================================================================


class TestLoopLogMessage:
    def test_log_shows_correct_counts(self, base_config, caplog):
        """Log-Meldung zeigt korrekte Zaehler-Semantik ('2x in History, Aufruf #3')."""
        reg = ToolRegistry()
        reg.register(_DummyTool())

        # A,B,A,B,A → A tritt 3x auf, aber nie direkt hintereinander.
        # Beim 3. A: same_params=2 → "2x in History, dies waere Aufruf #3"
        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=("search_documents", {"query": "B"}), tool_use_id="t2"),
            ToolStep(
                tool_call=("search_documents", {"query": "A"}), tool_use_id="t3"
            ),  # same_params=1 → ok
            ToolStep(
                tool_call=("search_documents", {"query": "B"}), tool_use_id="t4"
            ),  # same_params for B=1 → ok
            ToolStep(
                tool_call=("search_documents", {"query": "A"}), tool_use_id="t5"
            ),  # same_params=2 → gefiltert!
        ]
        adapter = _make_mock_adapter(steps)
        adapter.stream_final.return_value = iter(["Antwort."])

        chain = AgentChain(
            base_config, reg, adapter=adapter, max_iterations=10, max_same_tool_calls=0
        )

        with caplog.at_level(logging.INFO):
            events = list(chain.query_stream("Test"))

        # Log muss "2x in History" und "Aufruf #3" enthalten
        assert any("2x in History" in r.message for r in caplog.records)
        assert any("#3" in r.message for r in caplog.records)


# =====================================================================
# Gemischte Batch-Szenarien
# =====================================================================


class TestMixedBatch:
    def test_partial_loop_in_batch(self, base_config):
        """Batch mit 1 Loop + 1 Nicht-Loop → Nicht-Loop ausgefuehrt, Loop gefiltert."""
        reg = ToolRegistry()
        reg.register(_DummyTool("search_documents"))
        reg.register(_DummyTool("calculate"))

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            # Zweiter Step: search(A) ist Loop, calculate(1+1) ist neu
            ToolStep(
                tool_call=("search_documents", {"query": "A"}),
                tool_use_id="t2",
                parallel_calls=[("calculate", {"query": "1+1"}, "t3")],
            ),
            ToolStep(tool_call=None, text="Fertig", is_final=True),
        ]
        adapter = _make_mock_adapter(steps)

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # search(A) wurde gefiltert (identical_repeat)
        skipped = [e for e in events if e.event_type == "tool_call" and e.data.get("skipped")]
        assert len(skipped) == 1
        assert skipped[0].data["tool"] == "search_documents"

        # calculate(1+1) wurde ausgefuehrt
        normal = [e for e in events if e.event_type == "tool_call" and not e.data.get("skipped")]
        assert any(e.data["tool"] == "calculate" for e in normal)

    def test_all_loops_break(self, base_config):
        """Batch mit NUR Loops → ALLE gefiltert, break (finale Antwort)."""
        reg = ToolRegistry()
        reg.register(_DummyTool("search_documents"))

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            # Zweiter Step: A (repeat) und A-dup (dedup) → alle gefiltert → break
            ToolStep(
                tool_call=("search_documents", {"query": "A"}),
                tool_use_id="t2",
                parallel_calls=[("search_documents", {"query": "A"}, "t3")],
            ),
        ]
        adapter = _make_mock_adapter(steps)
        adapter.stream_final.return_value = iter(["Finale Antwort."])

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # Nach Break wird stream_final aufgerufen
        assert any(e.event_type == "done" for e in events)
        chunks = [e for e in events if e.event_type == "chunk"]
        assert any("Finale Antwort" in str(e.data) for e in chunks)
