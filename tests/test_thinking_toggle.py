"""Tests fuer Thinking-Toggle V4 (Build 86).

Testet:
- QueryRequest.think Default (False)
- _strip_think_blocks: geschlossen, offen, mehrfach, kein Match
- Standard-Stream reicht think an RAGChain durch
- Standard-Stream stripped <think>-Bloecke vor DB-Speicherung
- Unchained reicht think an LLMBackend.generate_stream durch
- Unchained Cloud-Pfad: think wird an Backend durchgereicht
- Agent reicht data.think an AgentChain weiter
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.generation.backends.base import LLMBackend


def _make_mock_backend(provider="ollama", model="test-model"):
    """Erstellt einen Mock-LLMBackend mit Standard-Konfiguration."""
    mock = MagicMock(spec=LLMBackend)
    mock.provider = provider
    mock.model = model
    mock.last_usage = {}
    return mock


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def standard_config():
    """Minimale Config fuer Standard-/Unchained-Tests."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
            "prompt_template": "",
        },
        "vectordb": {"backend": "qdrant", "path": "/tmp/test", "collection": "test"},
        "retrieval": {"reranking": False},
        "datasource": {"path": "/tmp/test"},
        "cloud": {"enabled": False},
        "chat": {
            "unchained": {"temperature": 0.8, "system_prompt": ""},
        },
    }


# ===========================================================================
# 1. QueryRequest.think Default
# ===========================================================================


class TestQueryRequestThink:
    def test_query_request_think_default(self):
        """QueryRequest.think ist standardmaessig False."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test?")
        assert req.think is False

    def test_query_request_think_true(self):
        """QueryRequest.think kann auf True gesetzt werden."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test?", think=True)
        assert req.think is True


# ===========================================================================
# 2-5. _strip_think_blocks
# ===========================================================================


class TestStripThinkBlocks:
    def test_strip_think_blocks_closed(self):
        """Geschlossener <think>-Block wird entfernt."""
        from kiara.web.api_chat import _strip_think_blocks

        text = "<think>Das ist Reasoning.</think>Die eigentliche Antwort."
        result = _strip_think_blocks(text)
        assert result == "Die eigentliche Antwort."

    def test_strip_think_blocks_unclosed(self):
        """Offener <think>-Block (ohne </think>) wird entfernt."""
        from kiara.web.api_chat import _strip_think_blocks

        text = "Anfang <think>reasoning ohne Ende"
        result = _strip_think_blocks(text)
        assert result == "Anfang"

    def test_strip_think_blocks_multiple(self):
        """Mehrere <think>-Bloecke werden alle entfernt."""
        from kiara.web.api_chat import _strip_think_blocks

        text = "<think>Block 1</think>Mitte<think>Block 2</think>Ende"
        result = _strip_think_blocks(text)
        assert result == "MitteEnde"

    def test_strip_think_blocks_no_match(self):
        """Sauberer Text bleibt unveraendert."""
        from kiara.web.api_chat import _strip_think_blocks

        text = "Ganz normaler Text ohne Think-Bloecke."
        result = _strip_think_blocks(text)
        assert result == text

    def test_strip_think_blocks_empty(self):
        """Leerer String bleibt leer."""
        from kiara.web.api_chat import _strip_think_blocks

        assert _strip_think_blocks("") == ""


# ===========================================================================
# 6. Standard-Stream reicht think an RAGChain durch
# ===========================================================================


class TestStandardStreamPassesThink:
    @pytest.mark.asyncio
    async def test_standard_stream_passes_think_true(self, standard_config):
        """_stream_standard reicht think=True an chain.query_stream weiter."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Antwort"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = ""

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard", think=True)
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        # query_stream muss mit think=True aufgerufen worden sein
        _, kwargs = mock_chain.query_stream.call_args
        assert kwargs.get("think") is True

    @pytest.mark.asyncio
    async def test_standard_stream_passes_think_false(self, standard_config):
        """_stream_standard reicht think=False an chain.query_stream weiter."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Antwort"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = ""

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard", think=False)
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        _, kwargs = mock_chain.query_stream.call_args
        assert kwargs.get("think") is False


# ===========================================================================
# 7. Standard-Stream stripped <think> vor DB-Speicherung
# ===========================================================================


class TestStandardStreamStripsBeforeSave:
    @pytest.mark.asyncio
    async def test_standard_stream_strips_before_save(self, standard_config):
        """_stream_standard entfernt <think>-Bloecke vor _save_assistant_message."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        def _thinking_generator():
            yield "<think>Reasoning...</think>"
            yield "Saubere Antwort."

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (_thinking_generator(), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = ""

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        # _save_assistant_message muss mit gestripptem Text aufgerufen werden
        mock_save.assert_called_once()
        saved_text = mock_save.call_args[0][1]
        assert "<think>" not in saved_text
        assert "Saubere Antwort." in saved_text


# ===========================================================================
# 8. Unchained reicht think an LLMBackend.generate_stream durch
# ===========================================================================


class TestUnchainedPassesThink:
    def test_unchained_passes_think_to_backend(self, standard_config):
        """UnchainedChain.query_stream reicht think=False an backend.generate_stream weiter."""
        from kiara.generation.unchained_chain import UnchainedChain

        mock_backend = _make_mock_backend(provider="ollama")
        mock_backend.generate_stream.return_value = iter(["Antwort"])

        chain = UnchainedChain(standard_config, backend=mock_backend, system_prompt="Test")
        list(chain.query_stream("Test?", think=False))

        # generate_stream muss mit think=False aufgerufen worden sein
        _, kwargs = mock_backend.generate_stream.call_args
        assert kwargs.get("think") is False

    def test_unchained_passes_think_true(self, standard_config):
        """UnchainedChain.query_stream reicht think=True an backend.generate_stream weiter."""
        from kiara.generation.unchained_chain import UnchainedChain

        mock_backend = _make_mock_backend(provider="ollama")
        mock_backend.generate_stream.return_value = iter(["Antwort"])

        chain = UnchainedChain(standard_config, backend=mock_backend, system_prompt="Test")
        list(chain.query_stream("Test?", think=True))

        _, kwargs = mock_backend.generate_stream.call_args
        assert kwargs.get("think") is True

    def test_unchained_none_think_passes_none(self, standard_config):
        """UnchainedChain.query_stream mit think=None reicht None an Backend weiter."""
        from kiara.generation.unchained_chain import UnchainedChain

        mock_backend = _make_mock_backend(provider="ollama")
        mock_backend.generate_stream.return_value = iter(["Antwort"])

        chain = UnchainedChain(standard_config, backend=mock_backend, system_prompt="Test")
        list(chain.query_stream("Test?", think=None))

        # think=None wird durchgereicht (Backend entscheidet wie es damit umgeht)
        _, kwargs = mock_backend.generate_stream.call_args
        assert kwargs.get("think") is None


# ===========================================================================
# 9. Unchained Cloud reicht think an Backend durch
# ===========================================================================


class TestUnchainedCloudPassesThink:
    def test_unchained_cloud_passes_think_to_backend(self, standard_config):
        """Im Cloud-Pfad wird think an das Backend durchgereicht (Backend entscheidet)."""
        from kiara.generation.unchained_chain import UnchainedChain

        mock_backend = _make_mock_backend(provider="anthropic", model="claude-sonnet-4-6")
        mock_backend.generate_stream.return_value = iter(["Cloud-Antwort"])

        chain = UnchainedChain(standard_config, backend=mock_backend, system_prompt="Test")
        result = list(chain.query_stream("Test?", think=True))

        # Backend.generate_stream wird mit think=True aufgerufen
        mock_backend.generate_stream.assert_called_once()
        _, kwargs = mock_backend.generate_stream.call_args
        assert kwargs.get("think") is True

    def test_unchained_cloud_route_is_cloud(self, standard_config):
        """Cloud-Backend ergibt route_used='cloud'."""
        from kiara.generation.unchained_chain import UnchainedChain

        mock_backend = _make_mock_backend(provider="anthropic", model="claude-sonnet-4-6")
        chain = UnchainedChain(standard_config, backend=mock_backend, system_prompt="Test")
        assert chain.route_used == "cloud"

    def test_unchained_ollama_route_is_local(self, standard_config):
        """Ollama-Backend ergibt route_used='local'."""
        from kiara.generation.unchained_chain import UnchainedChain

        mock_backend = _make_mock_backend(provider="ollama")
        chain = UnchainedChain(standard_config, backend=mock_backend, system_prompt="Test")
        assert chain.route_used == "local"


# ===========================================================================
# 10. Agent reicht data.think weiter
# ===========================================================================


class TestAgentPassesThink:
    @pytest.mark.asyncio
    async def test_agent_passes_think_true(self, standard_config):
        """_stream_agent reicht data.think=True an AgentChain.query_stream weiter."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        done_event = AgentEvent(event_type="done", data={})
        mock_chain.query_stream.return_value = iter([done_event])
        mock_chain.last_usage = {}

        with (
            patch("kiara.backend_selector.get_backend_by_id", return_value=None),
            patch(
                "kiara.generation.cloud_routing.get_next_backend",
                return_value={
                    "url": "http://localhost:11434",
                    "model": "test",
                    "provider": "ollama",
                },
            ),
            patch(
                "kiara.config.get_llm_backend",
                return_value={
                    "url": "http://localhost:11434",
                    "model": "test",
                    "provider": "ollama",
                },
            ),
            patch("kiara.generation.agent_chain.AgentChain", return_value=mock_chain),
            patch("kiara.generation.tool_adapter.create_tool_adapter", return_value=MagicMock()),
            patch("kiara.retrieval.retriever.HybridRetriever", return_value=MagicMock()),
            patch("kiara.retrieval.reranker.CrossEncoderReranker", return_value=MagicMock()),
            patch("kiara.tools.builtin_tools.SearchDocumentsTool", return_value=MagicMock()),
            patch("kiara.tools.builtin_tools.ListDatasourcesTool", return_value=MagicMock()),
            patch("kiara.tools.builtin_tools.CalculateTool", return_value=MagicMock()),
            patch("kiara.tools.registry.ToolRegistry") as MockRegistry,
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="agent", think=True)
            events = []
            async for event in _stream_agent(standard_config, data, 1, user):
                events.append(event)

        # AgentChain.query_stream muss mit think=True aufgerufen werden
        _, kwargs = mock_chain.query_stream.call_args
        assert kwargs.get("think") is True

    @pytest.mark.asyncio
    async def test_agent_passes_think_false(self, standard_config):
        """_stream_agent reicht data.think=False an AgentChain.query_stream weiter."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        done_event = AgentEvent(event_type="done", data={})
        mock_chain.query_stream.return_value = iter([done_event])
        mock_chain.last_usage = {}

        with (
            patch("kiara.backend_selector.get_backend_by_id", return_value=None),
            patch(
                "kiara.generation.cloud_routing.get_next_backend",
                return_value={
                    "url": "http://localhost:11434",
                    "model": "test",
                    "provider": "ollama",
                },
            ),
            patch(
                "kiara.config.get_llm_backend",
                return_value={
                    "url": "http://localhost:11434",
                    "model": "test",
                    "provider": "ollama",
                },
            ),
            patch("kiara.generation.agent_chain.AgentChain", return_value=mock_chain),
            patch("kiara.generation.tool_adapter.create_tool_adapter", return_value=MagicMock()),
            patch("kiara.retrieval.retriever.HybridRetriever", return_value=MagicMock()),
            patch("kiara.retrieval.reranker.CrossEncoderReranker", return_value=MagicMock()),
            patch("kiara.tools.builtin_tools.SearchDocumentsTool", return_value=MagicMock()),
            patch("kiara.tools.builtin_tools.ListDatasourcesTool", return_value=MagicMock()),
            patch("kiara.tools.builtin_tools.CalculateTool", return_value=MagicMock()),
            patch("kiara.tools.registry.ToolRegistry") as MockRegistry,
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="agent", think=False)
            events = []
            async for event in _stream_agent(standard_config, data, 1, user):
                events.append(event)

        # AgentChain.query_stream muss mit think=False aufgerufen werden
        _, kwargs = mock_chain.query_stream.call_args
        assert kwargs.get("think") is False
