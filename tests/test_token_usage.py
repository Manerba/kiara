"""Testet die Usage-Pipeline der verbleibenden Chain-/Backend-Integrationen.

Deckt ab: UnchainedChain, AgentChain, RAGChain → last_usage,
sowie die SSE-Events in api_chat.py.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def base_config():
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
        },
        "chat": {
            "unchained": {"temperature": 0.8, "system_prompt": ""},
            "agent": {"max_iterations": 3, "num_ctx": 8192, "temperature": 0.3},
        },
    }


@pytest.fixture
def mock_ollama_client():
    with patch("ollama.Client") as MockClient:
        instance = MagicMock()
        MockClient.return_value = instance
        yield instance


# ---- 1. UnchainedChain.last_usage (delegiert an Backend) ----


class TestUnchainedChainUsage:
    def test_last_usage_delegates_to_backend(self, base_config):
        from kiara.generation.unchained_chain import UnchainedChain

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"
        mock_backend.last_usage = {
            "prompt_tokens": 20,
            "response_tokens": 10,
            "total_duration_ms": 2000,
            "num_ctx": 4096,
            "model": "m",
        }
        mock_backend.generate_stream.return_value = iter(["A", "B"])

        chain = UnchainedChain(base_config, backend=mock_backend, system_prompt="Test")

        chunks = list(chain.query_stream("Test", num_ctx=4096))
        assert chunks == ["A", "B"]
        assert chain.last_usage["prompt_tokens"] == 20
        assert chain.last_usage["response_tokens"] == 10
        assert chain.last_usage["num_ctx"] == 4096
        assert chain.last_usage["model"] == "m"


# ---- 3. AgentChain.last_usage ----


class TestAgentChainUsage:
    def test_last_usage_after_streaming(self, base_config):
        from unittest.mock import PropertyMock

        from kiara.generation.agent_chain import AgentChain
        from kiara.generation.tool_adapter import BaseToolAdapter, ToolStep
        from kiara.tools.registry import ToolRegistry

        mock_adapter = MagicMock(spec=BaseToolAdapter)
        mock_adapter.step.return_value = ToolStep(text="Direkte Antwort", is_final=True)
        mock_adapter.deanonymize_params.side_effect = lambda p: p
        usage_data = {
            "prompt_tokens": 100,
            "response_tokens": 50,
            "total_duration_ms": 5000,
            "num_ctx": 8192,
            "model": "m",
        }
        type(mock_adapter).last_usage = PropertyMock(return_value=usage_data)

        registry = ToolRegistry()
        chain = AgentChain(base_config, registry, adapter=mock_adapter)

        events = list(chain.query_stream("Test"))
        assert chain.last_usage["prompt_tokens"] == 100
        assert chain.last_usage["response_tokens"] == 50
        assert chain.last_usage["num_ctx"] == 8192
        assert chain.last_usage["model"] == "m"


# ---- 4. RAGChain.last_usage delegiert an Backend ----


class TestRAGChainUsage:
    def test_last_usage_delegates_to_backend(self, base_config):
        mock_backend = MagicMock()
        mock_backend.provider = "ollama"
        mock_backend.last_usage = {
            "prompt_tokens": 77,
            "response_tokens": 33,
            "total_duration_ms": 1000,
            "num_ctx": 8192,
        }

        with (
            patch("kiara.generation.chain.HybridRetriever"),
            patch("kiara.generation.chain.CrossEncoderReranker"),
        ):
            from kiara.generation.chain import RAGChain

            chain = RAGChain(
                base_config,
                backend=mock_backend,
                prompt_template_override="Test: {context}\n{question}",
            )

        assert chain.last_usage == mock_backend.last_usage


# ---- 5-8. SSE usage-Event ----


def _parse_sse_events(output: str) -> list[tuple[str, dict]]:
    """Parst SSE-Output in (event_name, data) Paare."""
    events = []
    current_event = ""
    for line in output.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append((current_event, data))
            except json.JSONDecodeError:
                pass
            current_event = ""
    return events


class TestSSEUsageEvent:
    def _make_sse_output(self, generator):
        """Sammelt SSE-Output eines async Generators."""
        import asyncio

        async def collect():
            parts = []
            async for chunk in generator:
                parts.append(chunk)
            return "".join(parts)

        return asyncio.run(collect())

    def test_usage_event_in_stream_standard(self, base_config):
        chain_instance = MagicMock()
        chain_instance.query_stream.return_value = (iter(["Antwort"]), [])
        chain_instance.context_warning = None
        chain_instance.route_used = "local"
        chain_instance.last_usage = {
            "prompt_tokens": 100,
            "response_tokens": 50,
            "total_duration_ms": 2000,
            "num_ctx": 8192,
        }
        chain_instance.last_entities_anonymized = 0
        chain_instance.last_prompt = ""
        chain_instance.last_anon_question = ""
        chain_instance.last_anon_answer = ""
        chain_instance.last_expansions = []

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(chain_instance, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            from kiara.web.api_chat import _stream_standard

            data = MagicMock()
            data.question = "Test"
            data.k = 5
            data.num_ctx = 8192
            data.filter_dir = None
            data.filter_type = None
            data.filter_datasource_ids = None
            data.filter_datasource_exclude = False
            data.embedding_model = None
            data.think = None
            data.llm_backend_id = None
            user = {"id": 1, "is_admin": True, "datasource_ids": []}

            output = self._make_sse_output(_stream_standard(base_config, data, 1, user))

        events = _parse_sse_events(output)
        usage_events = [(e, d) for e, d in events if e == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0][1]["prompt_tokens"] == 100
        assert usage_events[0][1]["response_tokens"] == 50

        # usage kommt vor done
        event_names = [e for e, _ in events]
        assert event_names.index("usage") < event_names.index("done")

    def test_usage_event_in_stream_unchained(self, base_config):
        mock_backend = MagicMock()
        mock_backend.provider = "ollama"

        chain_instance = MagicMock()
        chain_instance.query_stream.return_value = iter(["Antwort"])
        chain_instance.route_used = "local"
        chain_instance.last_entities_anonymized = 0
        chain_instance.last_anon_question = ""
        chain_instance.last_anon_answer = ""
        chain_instance.last_prompt = ""
        chain_instance.last_usage = {
            "prompt_tokens": 30,
            "response_tokens": 15,
            "total_duration_ms": 1000,
            "num_ctx": 4096,
        }

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=chain_instance),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            from kiara.web.api_chat import _stream_unchained

            data = MagicMock()
            data.question = "Test"
            data.num_ctx = 4096
            data.llm_backend_id = None
            data.think = None
            data.temperature = None
            user = {"id": 1, "is_admin": True}

            output = self._make_sse_output(_stream_unchained(base_config, data, 1, user))

        events = _parse_sse_events(output)
        usage_events = [(e, d) for e, d in events if e == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0][1]["num_ctx"] == 4096
        event_names = [e for e, _ in events]
        assert event_names.index("usage") < event_names.index("done")

    def test_usage_event_in_stream_agent(self, base_config):
        from kiara.generation.agent_chain import AgentEvent

        chain_instance = MagicMock()
        chain_instance.query_stream.return_value = iter(
            [
                AgentEvent(event_type="chunk", data={"text": "Agent-Antwort"}),
                AgentEvent(event_type="done"),
            ]
        )
        chain_instance.last_usage = {
            "prompt_tokens": 200,
            "response_tokens": 80,
            "total_duration_ms": 4000,
            "num_ctx": 8192,
        }
        chain_instance._tool_events = []

        mock_backend = {"url": "http://x", "model": "m", "provider": "ollama"}

        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(chain_instance, mock_backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            from kiara.web.api_chat import _stream_agent

            data = MagicMock()
            data.question = "Test"
            data.k = 5
            data.num_ctx = 8192
            data.embedding_model = None
            data.llm_backend_id = None
            data.filter_datasource_ids = None
            data.filter_datasource_exclude = False
            data.think = None

            user = {"id": 1, "is_admin": True, "datasource_ids": []}
            output = self._make_sse_output(_stream_agent(base_config, data, 1, user))

        events = _parse_sse_events(output)
        usage_events = [(e, d) for e, d in events if e == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0][1]["prompt_tokens"] == 200
        event_names = [e for e, _ in events]
        assert event_names.index("usage") < event_names.index("done")

    def test_usage_event_in_stream_rewriter(self, base_config):
        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Antwort"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = ""
        mock_chain.rewritten_query = "Test"
        mock_chain.last_usage = {
            "prompt_tokens": 120,
            "response_tokens": 40,
            "total_duration_ms": 1500,
            "num_ctx": 8192,
        }

        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=None),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch(
                "kiara.generation.cloud_routing._create_chain_backend",
                return_value=MagicMock(provider="ollama", model="test-model"),
            ),
            patch("kiara.generation.rewriter_chain.RewriterChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            from kiara.web.api_chat import _stream_rewriter

            data = MagicMock()
            data.question = "Test"
            data.k = 5
            data.num_ctx = 8192
            data.filter_dir = None
            data.filter_type = None
            data.filter_datasource_ids = None
            data.filter_datasource_exclude = False
            data.embedding_model = None
            data.think = None
            data.llm_backend_id = None
            user = {"id": 1, "is_admin": True, "datasource_ids": []}

            output = self._make_sse_output(_stream_rewriter(base_config, data, 1, user))

        events = _parse_sse_events(output)
        event_names = [e for e, _ in events]
        assert event_names.index("usage") < event_names.index("done")

    def test_no_usage_event_when_empty(self, base_config):
        mock_backend = MagicMock()
        mock_backend.provider = "ollama"

        chain_instance = MagicMock()
        chain_instance.query_stream.return_value = iter(["ok"])
        chain_instance.route_used = "local"
        chain_instance.last_entities_anonymized = 0
        chain_instance.last_anon_question = ""
        chain_instance.last_anon_answer = ""
        chain_instance.last_prompt = ""
        chain_instance.last_usage = {}

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=chain_instance),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            from kiara.web.api_chat import _stream_unchained

            data = MagicMock()
            data.question = "Test"
            data.num_ctx = 4096
            data.llm_backend_id = None
            data.think = None
            data.temperature = None
            user = {"id": 1, "is_admin": True}

            output = self._make_sse_output(_stream_unchained(base_config, data, 1, user))

        events = _parse_sse_events(output)
        usage_events = [e for e, _ in events if e == "usage"]
        assert len(usage_events) == 0
