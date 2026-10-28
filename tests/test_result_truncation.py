"""Tests fuer Result-Truncation (Fix 3): Budget-System und Limits.

Verifiziert: Per-Result-Truncation (Cloud/Ollama), Gesamt-Budget,
500-Zeichen-Floor, Budget-erschoepft-Warnung, und keine doppelte
Truncation bei add_tool_results().
"""

import sys
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kiara.generation.adapters.base import BaseToolAdapter, ToolStep
from kiara.generation.agent_chain import AgentChain
from kiara.shared.constants import (
    AGENT_RESULT_TRUNCATE_CLOUD,
    AGENT_RESULT_TRUNCATE_OLLAMA,
    AGENT_TOTAL_RESULT_BUDGET,
)
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.registry import ToolRegistry

# =====================================================================
# Hilfsfunktionen
# =====================================================================


class _ResultTool(BaseTool):
    """Test-Tool das konfigurierbare Ergebnis-Laengen zurueckgibt."""

    def __init__(self, result_length=100, name="search_documents"):
        self._result_length = result_length
        self._name = name

    def definition(self):
        return ToolDefinition(
            name=self._name,
            description="Test-Tool",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )

    async def execute(self, params, user_id=None):
        text = "R" * self._result_length
        return ToolResult(
            success=True,
            data={"value": text},
            error=None,
            display_text=text,
        )


def _make_mock_adapter(steps, truncate_limit=AGENT_RESULT_TRUNCATE_CLOUD):
    """Erzeugt einen Mock-Adapter mit konfigurierbarem Truncation-Limit."""
    adapter = MagicMock(spec=BaseToolAdapter)
    adapter.prepare_tool_text.side_effect = lambda text: BaseToolAdapter.prepare_tool_text(
        adapter, text
    )
    adapter.step.side_effect = iter(steps)
    adapter.init_conversation.return_value = None
    adapter.add_tool_result.return_value = None
    adapter.add_tool_results.return_value = None
    adapter.stream_final.return_value = iter(["Antwort."])
    adapter.deanonymize_params.side_effect = lambda p: {**p}
    adapter.last_usage = {"input": 0, "output": 0}
    type(adapter).result_truncate_limit = PropertyMock(return_value=truncate_limit)
    return adapter


@pytest.fixture
def base_config():
    return {"chat": {"agent": {"max_iterations": 20, "temperature": 0.3}}}


# =====================================================================
# Konstanten
# =====================================================================


class TestConstants:
    def test_cloud_limit(self):
        """Cloud-Limit ist 24000."""
        assert AGENT_RESULT_TRUNCATE_CLOUD == 24000

    def test_ollama_limit(self):
        """Ollama-Limit ist 4000."""
        assert AGENT_RESULT_TRUNCATE_OLLAMA == 4000

    def test_total_budget(self):
        """Gesamt-Budget ist 96000."""
        assert AGENT_TOTAL_RESULT_BUDGET == 96000


# =====================================================================
# Per-Result-Truncation
# =====================================================================


class TestPerResultTruncation:
    def test_cloud_truncation_at_24000(self, base_config):
        """Per-Result-Truncation bei 24000 Zeichen (Cloud-Adapter)."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=30000))

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_1"),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ],
            truncate_limit=AGENT_RESULT_TRUNCATE_CLOUD,
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # add_tool_results wurde mit getruncatem Text aufgerufen
        call_args = adapter.add_tool_results.call_args[0][0]
        result_text = call_args[0][1]  # (tool_name, result_text, tid)
        assert len(result_text) <= AGENT_RESULT_TRUNCATE_CLOUD + 5  # +3 fuer "..." + Toleranz
        assert result_text.endswith("...")

    def test_ollama_truncation_at_4000(self, base_config):
        """Per-Result-Truncation bei 4000 Zeichen (Ollama via result_truncate_limit)."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=5000))

        adapter = _make_mock_adapter(
            [
                ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="tu_1"),
                ToolStep(tool_call=None, text="Fertig", is_final=True),
            ],
            truncate_limit=AGENT_RESULT_TRUNCATE_OLLAMA,
        )

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        call_args = adapter.add_tool_results.call_args[0][0]
        result_text = call_args[0][1]
        assert len(result_text) <= AGENT_RESULT_TRUNCATE_OLLAMA + 5
        assert result_text.endswith("...")


# =====================================================================
# Gesamt-Budget
# =====================================================================


class TestBudget:
    def test_later_results_truncated_harder(self, base_config):
        """5. Result wird staerker gekuerzt wenn Budget fast erschoepft."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=20000))

        # 5 Calls, jeder liefert 20000 Zeichen — Budget = 96000
        # Bei Cloud-Limit 24000: jeder wird auf 20000 (< 24000) belassen,
        # bis remaining_budget < 20000 → Truncation
        steps = []
        for i in range(5):
            steps.append(
                ToolStep(
                    tool_call=("search_documents", {"query": f"Q{i}"}),
                    tool_use_id=f"t{i}",
                )
            )
        steps.append(ToolStep(tool_call=None, text="Fertig", is_final=True))

        adapter = _make_mock_adapter(steps, truncate_limit=24000)
        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # Alle add_tool_results-Aufrufe sammeln
        all_results = []
        for call in adapter.add_tool_results.call_args_list:
            for _name, text, _tid in call[0][0]:
                all_results.append(text)

        # Die letzten Results sollten kuerzer sein als die ersten
        assert len(all_results) == 5
        # Erstes Result: voll (JSON-serialisiert, < 24000 per-result-limit)
        # data wird als JSON serialisiert → ~80 Zeichen Overhead durch JSON-Struktur
        assert len(all_results[0]) <= 24000
        assert len(all_results[0]) >= 20000  # Nicht truncated
        # Letztes Result: gekuerzt (Budget-Rest)
        assert len(all_results[-1]) < len(all_results[0])

    def test_minimum_500_chars(self, base_config):
        """Gesamt-Budget: Minimum 500 Zeichen pro Result."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=1000))

        # Budget fast aufbrauchen: 4 Calls mit je 24000
        steps = []
        for i in range(4):
            steps.append(
                ToolStep(
                    tool_call=("search_documents", {"query": f"Q{i}"}),
                    tool_use_id=f"t{i}",
                )
            )

        # 5. Call wenn Budget fast erschoepft
        steps.append(
            ToolStep(
                tool_call=("search_documents", {"query": "final"}),
                tool_use_id="t_final",
            )
        )
        steps.append(ToolStep(tool_call=None, text="Fertig", is_final=True))

        adapter = _make_mock_adapter(steps, truncate_limit=24000)

        # Force: Erster Result hat volle 24000 Zeichen
        tool = _ResultTool(result_length=24000)
        reg._tools = {"search_documents": tool}

        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        # Alle Results pruefen: keiner darf < 500 Zeichen sein
        for call in adapter.add_tool_results.call_args_list:
            for _name, text, _tid in call[0][0]:
                # Kurze Strings (z.B. Loop-Messages) sind ok,
                # nur echte Results muessen >= 500 sein
                if len(text) > 100:  # Nur echte Results pruefen
                    assert len(text) >= 500 or len(text) == 24000, (
                        f"Result zu kurz: {len(text)} Zeichen"
                    )


# =====================================================================
# Budget-Truncation-Szenarien (V9)
# =====================================================================


class TestBudgetTruncationScenarios:
    def test_short_result_no_truncation_warning(self, base_config):
        """Budget erschoepft + kurzes Result (< 500 Zeichen) → keine Truncation."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=300))  # Kurz genug

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=None, text="Fertig", is_final=True),
        ]

        # effective_limit > 300 → keine Truncation
        adapter = _make_mock_adapter(steps, truncate_limit=24000)
        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        call_args = adapter.add_tool_results.call_args[0][0]
        result_text = call_args[0][1]
        assert "Budget erschoepft" not in result_text
        assert not result_text.endswith("...")

    def test_effective_limit_300_display_400_floor_keeps(self, base_config):
        """effective_limit=300, display=480 (JSON) → Floor greift, display bleibt."""
        # result.data wird als JSON serialisiert → 400 Zeichen Raw + ~80 JSON-Overhead = ~480
        # effective_limit=300, display=480:
        #   len(display)=480 > effective_limit=300 → target = max(300, 500) = 500
        #   len(display)=480 <= 500 → KEINE Truncation!
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=400))

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=None, text="Fertig", is_final=True),
        ]
        adapter = _make_mock_adapter(steps, truncate_limit=300)
        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        call_args = adapter.add_tool_results.call_args[0][0]
        result_text = call_args[0][1]
        # Floor 500 > 300, display ~480 < 500 → bleibt unveraendert (keine Truncation)
        assert not result_text.endswith("...")
        assert "Budget erschoepft" not in result_text

    def test_effective_limit_300_display_800_truncated_to_500(self, base_config):
        """effective_limit=300, display=800 → auf 500 truncated + '[Budget erschoepft]'."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=800))

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=None, text="Fertig", is_final=True),
        ]
        adapter = _make_mock_adapter(steps, truncate_limit=300)
        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        call_args = adapter.add_tool_results.call_args[0][0]
        result_text = call_args[0][1]
        # target = max(300, 500) = 500, display=800 > 500 → truncated
        assert len(result_text) < 800
        assert "Budget erschoepft" in result_text

    def test_effective_limit_2000_display_5000_truncated_with_dots(self, base_config):
        """effective_limit=2000, display=5000 → auf 2000 truncated + '...'."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=5000))

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=None, text="Fertig", is_final=True),
        ]
        adapter = _make_mock_adapter(steps, truncate_limit=2000)
        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        call_args = adapter.add_tool_results.call_args[0][0]
        result_text = call_args[0][1]
        # effective_limit=2000 >= 500, target=2000, display=5000 > 2000
        assert len(result_text) <= 2005  # 2000 + "..."
        assert result_text.endswith("...")

    def test_effective_limit_5000_display_3000_no_truncation(self, base_config):
        """effective_limit=5000, display=~3080 (JSON) → keine Truncation."""
        reg = ToolRegistry()
        reg.register(_ResultTool(result_length=3000))

        steps = [
            ToolStep(tool_call=("search_documents", {"query": "A"}), tool_use_id="t1"),
            ToolStep(tool_call=None, text="Fertig", is_final=True),
        ]
        adapter = _make_mock_adapter(steps, truncate_limit=5000)
        chain = AgentChain(base_config, reg, adapter=adapter, max_iterations=10)
        events = list(chain.query_stream("Test"))

        call_args = adapter.add_tool_results.call_args[0][0]
        result_text = call_args[0][1]
        # result.data wird als JSON serialisiert → ~80 Zeichen Overhead
        assert len(result_text) >= 3000
        assert len(result_text) <= 5000  # Unter dem Limit, keine Truncation
        assert not result_text.endswith("...")


# =====================================================================
# add_tool_results() Cloud: Keine doppelte Truncation
# =====================================================================


class TestNoDoubleTruncation:
    def test_cloud_add_tool_results_no_truncation(self):
        """Cloud add_tool_results() trunciert NICHT nochmals (nur Anonymisierung)."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        adapter._messages = []
        # 30000 Zeichen — add_tool_results() soll NICHT truncieren
        long_text = "X" * 30000
        adapter.add_tool_results([("search", long_text, "tu_1")])

        msg = adapter._messages[0]
        content = msg["content"][0]["content"]
        assert len(content) == 30000  # Kein Truncating

    def test_cloud_single_result_keeps_prepared_content(self):
        """Auch der Single-Call-Pfad uebertraegt den zentral budgetierten Inhalt."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        adapter._messages = []
        long_text = "X" * 30000
        adapter.add_tool_result("search", long_text, "tu_1")

        msg = adapter._messages[0]
        content = msg["content"][0]["content"]
        assert content == long_text


# =====================================================================
# Ollama: result_truncate_limit Property
# =====================================================================


class TestOllamaProperty:
    def test_ollama_keeps_prepared_content(self):
        """Die AgentChain nutzt das Limit; der Adapter kuerzt nicht erneut."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        with patch("ollama.Client"), patch("httpx.Timeout"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        assert adapter.result_truncate_limit == AGENT_RESULT_TRUNCATE_OLLAMA

        adapter.init_conversation("Test", "System", [])
        long_text = "Y" * 5000
        adapter.add_tool_result("search", long_text)

        stored = adapter._previous_results[-1]["result"]
        assert stored == long_text

    def test_single_call_via_add_tool_results_no_double_truncation(self):
        """Einzel-Call via add_tool_results(): keine doppelte Truncation."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        with patch("ollama.Client"), patch("httpx.Timeout"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        adapter.init_conversation("Test", "System", [])

        # Text genau am Limit — soll NICHT nochmals gekuerzt werden
        text = "Z" * AGENT_RESULT_TRUNCATE_OLLAMA
        adapter.add_tool_results([("search", text, None)])

        stored = adapter._previous_results[-1]["result"]
        assert len(stored) == AGENT_RESULT_TRUNCATE_OLLAMA
        assert not stored.endswith("...")
