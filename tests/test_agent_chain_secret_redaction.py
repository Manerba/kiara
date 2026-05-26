"""Sicherheits-Regressionstests fuer Agent-/Stream-Exception-Pfade.

Plan V15 verlangt:
- Toolparameter/Secrets duerfen NICHT neu in Logs/SSE/Audit/Trace landen.
- ToolAdapterError-Text und Tool-Exception-`str(exc)` muessen vor
  Ausgabe ueber `redact_secrets_in_text` laufen.
- `exc_info=True` darf den Original-Text in Tool-Pfaden nicht via
  Traceback erneut leaken.

Diese Tests decken die Pfade aus dem V15-Review (Befund 1 und 2) ab.
"""

import logging
from unittest.mock import MagicMock, PropertyMock

import pytest

from kiara.generation.agent_chain import AgentChain
from kiara.generation.tool_adapter import BaseToolAdapter, ToolAdapterError, ToolStep
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult
from kiara.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


SECRET_FRAGMENTS = (
    "sk-live-123",
    "supersecret",
    "hunter2",
    "Bearer-abc.def.ghi",
    "topsecret",
)


def _assert_no_secret(haystack: str) -> None:
    for fragment in SECRET_FRAGMENTS:
        assert fragment not in haystack, f"Secret '{fragment}' leaked in: {haystack!r}"


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


def _make_adapter(*, step_side_effect=None, steps=None):
    adapter = MagicMock(spec=BaseToolAdapter)
    if step_side_effect is not None:
        adapter.step.side_effect = step_side_effect
    elif steps is not None:
        adapter.step.side_effect = steps
    adapter.deanonymize_params.side_effect = lambda p: {**p}
    adapter.stream_final.return_value = iter(["fallback"])
    adapter.add_tool_result.return_value = None
    adapter.add_tool_results.return_value = None
    type(adapter).last_usage = PropertyMock(
        return_value={
            "prompt_tokens": 1,
            "response_tokens": 1,
            "total_duration_ms": 1,
            "num_ctx": 1,
            "model": "test",
        }
    )
    type(adapter).result_truncate_limit = PropertyMock(return_value=24000)
    return adapter


# ---------------------------------------------------------------------------
# Befund 1: ToolAdapterError redakten
# ---------------------------------------------------------------------------


class TestStepWithRetryRedactsToolAdapterError:
    """ToolAdapterError-Text darf nicht ungesehen in SSE/Audit landen."""

    def test_non_retryable_tool_adapter_error_redacted_in_event(self, base_config, caplog):
        registry = ToolRegistry()
        leak_msg = "api_key=sk-live-123 password=supersecret"
        adapter = _make_adapter(step_side_effect=ToolAdapterError(leak_msg, retryable=False))

        chain = AgentChain(base_config, registry, adapter=adapter)
        caplog.set_level(logging.ERROR, logger="kiara.generation.agent_chain")
        events = list(chain.query_stream("auth-test"))

        err = next(e for e in events if e.event_type == "error")
        _assert_no_secret(err.data["message"])
        assert "***" in err.data["message"]

    def test_retryable_double_failure_redacted(self, base_config):
        registry = ToolRegistry()
        adapter = _make_adapter(
            step_side_effect=ToolAdapterError("params={'password': 'supersecret'}", retryable=True)
        )

        chain = AgentChain(base_config, registry, adapter=adapter)
        events = list(chain.query_stream("transient-test"))

        err = next(e for e in events if e.event_type == "error")
        _assert_no_secret(err.data["message"])


# ---------------------------------------------------------------------------
# Befund 2: Tool-Exception (Future) darf nicht via exc_info leaken
# ---------------------------------------------------------------------------


class _LeakySecretTool(BaseTool):
    """Tool, dessen Exception-Text Tool-Parameter zurueckspiegelt."""

    def __init__(self, marker: str = "supersecret"):
        self._marker = marker

    def definition(self):
        return ToolDefinition(
            name="leaky_secret",
            description="Wirft eine Exception mit echo-artigem Secret.",
            parameters={
                "type": "object",
                "properties": {"password": {"type": "string"}},
                "required": ["password"],
            },
        )

    async def execute(self, params, user_id=None):
        raise RuntimeError(f"backend rejected request password={self._marker}")


class TestRunToolRedactsSecrets:
    """Parallel- und Single-Thread Tool-Ausfuehrung muss redakten."""

    def test_single_tool_exception_redacted_in_event_and_log(self, base_config, caplog):
        reg = ToolRegistry()
        reg.register(_LeakySecretTool())
        adapter = _make_adapter(
            steps=[
                ToolStep(tool_call=("leaky_secret", {"password": "supersecret"})),
                ToolStep(text="ok", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        caplog.set_level(logging.ERROR, logger="kiara.generation.agent_chain")
        events = list(chain.query_stream("test"))

        tr = next(e for e in events if e.event_type == "tool_result")
        # Tool-Result darf das Secret nicht enthalten — der Plan verlangt
        # Redaction auch beim _execute_tool-Pfad ueber redact_secrets_in_text.
        assert "supersecret" not in str(tr.data)

        joined_logs = "\n".join(record.getMessage() for record in caplog.records)
        _assert_no_secret(joined_logs)
        # Tracebacks im LogRecord wuerden den Originalfehler erneut zeigen —
        # gezielt verifizieren, dass keiner haengen blieb.
        for record in caplog.records:
            if record.exc_info:
                # Wir akzeptieren keinen Tracebacks in Tool-Pfaden (Befund 2).
                rendered = logging.Formatter().formatException(record.exc_info)
                _assert_no_secret(rendered)

    def test_parallel_tool_future_exception_redacted(self, base_config, caplog):
        """Zwei Tools werden parallel ausgefuehrt; das fehlerhafte leaked nicht."""

        class _GoodTool(BaseTool):
            def definition(self):
                return ToolDefinition(
                    name="good",
                    description="OK-Tool",
                    parameters={"type": "object", "properties": {}, "required": []},
                )

            async def execute(self, params, user_id=None):
                return ToolResult(
                    success=True,
                    data={"ok": True},
                    error=None,
                    display_text="ok",
                )

        reg = ToolRegistry()
        reg.register(_LeakySecretTool())
        reg.register(_GoodTool())
        # Zwei parallele Tool-Calls in einem Step
        adapter = _make_adapter(
            steps=[
                ToolStep(
                    tool_call=("leaky_secret", {"password": "supersecret"}),
                    tool_use_id="id-1",
                    parallel_calls=[("good", {}, "id-2")],
                ),
                ToolStep(text="ok", is_final=True),
            ]
        )

        chain = AgentChain(base_config, reg, adapter=adapter)
        caplog.set_level(logging.ERROR, logger="kiara.generation.agent_chain")
        events = list(chain.query_stream("Parallel-Test"))

        # Tool-Result-Events fuer beide Calls sammeln und auf Leak pruefen
        tool_results = [e for e in events if e.event_type == "tool_result"]
        for tr in tool_results:
            assert "supersecret" not in str(tr.data)

        # Auch der LogRecord (inkl. evtl. exc_info-Traceback) darf nicht leaken
        for record in caplog.records:
            assert "supersecret" not in record.getMessage()
            if record.exc_info:
                rendered = logging.Formatter().formatException(record.exc_info)
                _assert_no_secret(rendered)


# ---------------------------------------------------------------------------
# Befund 2: stream_final-Pfad darf nicht via exc_info leaken
# ---------------------------------------------------------------------------


class TestStreamFinalRedaction:
    def test_stream_final_tool_adapter_error_redacted(self, base_config, caplog):
        registry = ToolRegistry()
        adapter = _make_adapter(steps=[ToolStep(text="Draft", is_final=False)])
        adapter.stream_final.side_effect = ToolAdapterError("backend says: api_key=sk-live-123")

        chain = AgentChain(base_config, registry, adapter=adapter)
        caplog.set_level(logging.ERROR, logger="kiara.generation.agent_chain")
        events = list(chain.query_stream("test"))

        err = next(e for e in events if e.event_type == "error")
        _assert_no_secret(err.data["message"])
        for record in caplog.records:
            _assert_no_secret(record.getMessage())
            if record.exc_info:
                rendered = logging.Formatter().formatException(record.exc_info)
                _assert_no_secret(rendered)

    def test_stream_final_unexpected_exception_generic_message_no_secret(self, base_config, caplog):
        """Generischer Exception-Pfad liefert sichere User-Message und keine Log-Leaks."""
        registry = ToolRegistry()
        adapter = _make_adapter(steps=[ToolStep(text="Draft", is_final=False)])
        adapter.stream_final.side_effect = RuntimeError("params={'password': 'hunter2'}")

        chain = AgentChain(base_config, registry, adapter=adapter)
        caplog.set_level(logging.ERROR, logger="kiara.generation.agent_chain")
        events = list(chain.query_stream("test"))

        err = next(e for e in events if e.event_type == "error")
        # User-Message ist generisch — kein Secret im SSE.
        assert "hunter2" not in err.data["message"]
        for record in caplog.records:
            _assert_no_secret(record.getMessage())
            if record.exc_info:
                rendered = logging.Formatter().formatException(record.exc_info)
                _assert_no_secret(rendered)
