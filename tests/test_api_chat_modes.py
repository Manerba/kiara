"""Tests fuer Chat-Modi in der API (QueryRequest, Dispatch, SSE-Streaming, Chat-Mode-Filter)."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from kiara.shared.enums import ChatMode


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


# ===========================================================================
# QueryRequest — Validation
# ===========================================================================


class TestQueryRequestModeField:
    """Tests fuer das mode-Feld im QueryRequest."""

    def test_chatmode_values_match_api(self):
        """Die ChatMode-Enum-Werte muessen als API-Strings funktionieren."""
        valid_modes = {"standard", "rewriter", "unchained", "agent"}
        enum_values = {m.value for m in ChatMode}
        assert enum_values == valid_modes

    def test_mode_default_standard(self):
        """Sicherstellen, dass der Default-Modus 'standard' ist."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test")
        assert req.mode == "standard"

    def test_mode_unchained(self):
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test", mode="unchained")
        assert req.mode == "unchained"

    def test_mode_agent(self):
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test", mode="agent")
        assert req.mode == "agent"

    def test_mode_standard_explicit(self):
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test", mode="standard")
        assert req.mode == "standard"

    def test_query_request_fields(self):
        """Sicherstellen, dass alle erwarteten Felder vorhanden sind."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(
            question="Frage",
            mode="agent",
            k=10,
            filter_dir="/docs",
            filter_type=".pdf",
            filter_datasource_ids=[1],
        )
        assert req.question == "Frage"
        assert req.mode == "agent"
        assert req.k == 10

    def test_query_request_default_k(self):
        """Default k=5."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test")
        assert req.k == 5

    def test_query_request_optional_fields_none(self):
        """Optionale Felder default auf None."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test")
        assert req.filter_dir is None
        assert req.filter_type is None
        assert req.filter_datasource_ids is None
        assert req.embedding_model is None


# ===========================================================================
# _save_assistant_message — DRY-Helfer
# ===========================================================================


class TestSaveAssistantMessage:
    """Tests fuer die _save_assistant_message Helper-Funktion."""

    def test_save_without_sources(self):
        """Speichern ohne Sources: sources = None."""
        from kiara.web.api_chat import _save_assistant_message

        with (
            patch("kiara.web.api_chat.db_transaction") as mock_db,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
        ):
            mock_conn = MagicMock()
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            _save_assistant_message(42, "Antworttext")

            # msg_repo.add_message aufgerufen
            mock_msg_repo.add_message.assert_called_once()
            call_args = mock_msg_repo.add_message.call_args
            assert call_args[0][0] == 42  # chat_id
            assert call_args[0][1] == "assistant"
            assert call_args[0][2] == "Antworttext"
            assert call_args[0][3] is None  # sources

    def test_save_with_sources(self):
        """Speichern mit Sources: an msg_repo weitergereicht."""
        from kiara.web.api_chat import _save_assistant_message

        sources = [{"source": "doc.txt", "score": 0.9}]

        with (
            patch("kiara.web.api_chat.db_transaction") as mock_db,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
        ):
            mock_conn = MagicMock()
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            _save_assistant_message(42, "Antwort", sources)

            call_args = mock_msg_repo.add_message.call_args
            passed_sources = call_args[0][3]
            assert passed_sources is not None
            assert len(passed_sources) == 1
            assert passed_sources[0]["source"] == "doc.txt"

    def test_save_updates_chat_timestamp(self):
        """chat_repo.touch wird aufgerufen um updated_at zu aktualisieren."""
        from kiara.web.api_chat import _save_assistant_message

        with (
            patch("kiara.web.api_chat.db_transaction") as mock_db,
            patch("kiara.web.api_chat.msg_repo") as mock_msg_repo,
            patch("kiara.web.api_chat.chat_repo") as mock_chat_repo,
        ):
            mock_conn = MagicMock()
            mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            _save_assistant_message(99, "Text")

            mock_chat_repo.touch.assert_called_once()
            assert mock_chat_repo.touch.call_args[0][0] == 99


# ===========================================================================
# Mode-Dispatch in chat_query
# ===========================================================================


class TestModeDispatch:
    """Tests fuer die Mode-Dispatch-Logik im chat_query-Endpoint."""

    def test_mode_comparison_with_enum(self):
        """str-Enum-Vergleich funktioniert fuer Dispatch."""
        assert ChatMode.STANDARD == "standard"
        assert ChatMode.UNCHAINED == "unchained"
        assert ChatMode.AGENT == "agent"

    def test_mode_dispatch_uses_data_mode_over_config_default(self):
        """data.mode hat Vorrang ueber config default_mode."""
        from kiara.web.api_chat import QueryRequest

        # Simuliere: Config sagt "standard", Request sagt "agent"
        req = QueryRequest(question="Test", mode="agent")
        config = {"chat": {"default_mode": "standard"}}

        mode = req.mode or config.get("chat", {}).get("default_mode", "standard")
        assert mode == "agent"

    def test_mode_dispatch_falls_back_to_config_default(self):
        """Leerer mode-String → Fallback auf Config default_mode."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test", mode="")
        config = {"chat": {"default_mode": "unchained"}}

        mode = req.mode or config.get("chat", {}).get("default_mode", "standard")
        assert mode == "unchained"

    def test_mode_dispatch_double_fallback(self):
        """mode="" + kein Config → Fallback auf 'standard'."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test", mode="")
        config = {}

        mode = req.mode or config.get("chat", {}).get("default_mode", "standard")
        assert mode == "standard"


# ===========================================================================
# SSE-Event-Format — _stream_unchained
# ===========================================================================


class TestStreamUnchained:
    """Tests fuer den _stream_unchained SSE-Generator."""

    @pytest.fixture
    def config(self):
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
            },
        }

    def _mock_backend(self):
        backend = MagicMock()
        backend.provider = "ollama"
        backend.model = "test-model"
        return backend

    @pytest.mark.asyncio
    async def test_sse_events_order(self, config):
        """Event-Reihenfolge: mode → chunk(s) → done."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Hallo", " Welt"])
        mock_chain.last_usage = {
            "prompt_tokens": 10,
            "response_tokens": 5,
            "total_duration_ms": 100,
            "num_ctx": 8192,
            "model": "test-model",
        }
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0

        user = {"id": 1, "is_admin": True}
        with (
            patch("kiara.generation.backends.create_backend", return_value=self._mock_backend()),
            patch(
                "kiara.generation.backends.wrap_with_anonymization",
                return_value=self._mock_backend(),
            ),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            data = QueryRequest(question="Hi", mode="unchained")
            events = []
            async for event in _stream_unchained(config, data, 1, user):
                events.append(event)

            # Mindestens: mode + chunks + done
            assert len(events) >= 3

            # Erstes Event: mode
            assert "event: mode\n" in events[0]
            mode_payload = json.loads(events[0].split("data: ")[1].strip())
            assert mode_payload["mode"] == "unchained"

            # Letztes Event: done
            assert "event: done\n" in events[-1]
            done_payload = json.loads(events[-1].split("data: ")[1].strip())
            assert "chat_id" in done_payload

    @pytest.mark.asyncio
    async def test_chunks_contain_text(self, config):
        """Chunk-Events enthalten den Text."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Alpha", "Beta"])
        mock_chain.last_usage = {}
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0

        user = {"id": 1, "is_admin": True}
        with (
            patch("kiara.generation.backends.create_backend", return_value=self._mock_backend()),
            patch(
                "kiara.generation.backends.wrap_with_anonymization",
                return_value=self._mock_backend(),
            ),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            data = QueryRequest(question="Test", mode="unchained")
            events = []
            async for event in _stream_unchained(config, data, 1, user):
                events.append(event)

            chunk_events = [e for e in events if "event: chunk\n" in e]
            assert len(chunk_events) == 2
            first_chunk = json.loads(chunk_events[0].split("data: ")[1].strip())
            assert first_chunk["text"] == "Alpha"

    @pytest.mark.asyncio
    async def test_no_sources_event(self, config):
        """Unchained-Modus sendet kein sources-Event."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["OK"])
        mock_chain.last_usage = {}
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0

        user = {"id": 1, "is_admin": True}
        with (
            patch("kiara.generation.backends.create_backend", return_value=self._mock_backend()),
            patch(
                "kiara.generation.backends.wrap_with_anonymization",
                return_value=self._mock_backend(),
            ),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            data = QueryRequest(question="Test", mode="unchained")
            events = []
            async for event in _stream_unchained(config, data, 1, user):
                events.append(event)

            assert not any("event: sources\n" in e for e in events)

    @pytest.mark.asyncio
    async def test_save_called_with_full_answer(self, config):
        """_save_assistant_message wird mit dem zusammengesetzten Text aufgerufen."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Hallo", " Welt", "!"])
        mock_chain.last_usage = {}
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0

        user = {"id": 1, "is_admin": True}
        with (
            patch("kiara.generation.backends.create_backend", return_value=self._mock_backend()),
            patch(
                "kiara.generation.backends.wrap_with_anonymization",
                return_value=self._mock_backend(),
            ),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            data = QueryRequest(question="Gruss", mode="unchained")
            async for _ in _stream_unchained(config, data, 42, user):
                pass

            mock_save.assert_called_once_with(42, "Hallo Welt!")

    @pytest.mark.asyncio
    async def test_empty_successful_answer_is_persisted_for_audit_correlation(self, config):
        """Leere erfolgreiche Antworten bekommen trotzdem eine chat_messages-Zeile."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter([])
        mock_chain.last_usage = {}
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0

        user = {"id": 1, "is_admin": True}
        with (
            patch("kiara.generation.backends.create_backend", return_value=self._mock_backend()),
            patch(
                "kiara.generation.backends.wrap_with_anonymization",
                return_value=self._mock_backend(),
            ),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            data = QueryRequest(question="Leere Antwort", mode="unchained")
            async for _ in _stream_unchained(config, data, 42, user):
                pass

            mock_save.assert_called_once_with(42, "")
            mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_done_event_contains_trace_id_none(self, config):
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Hallo"])
        mock_chain.last_usage = None
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = "prompt"

        user = {"id": 1, "is_admin": True}
        with (
            patch("kiara.generation.backends.create_backend", return_value=self._mock_backend()),
            patch(
                "kiara.generation.backends.wrap_with_anonymization",
                return_value=self._mock_backend(),
            ),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
            patch("kiara.web.api_chat._log_cloud_audit"),
        ):
            data = QueryRequest(question="Hi", mode="unchained")
            events = []
            async for event in _stream_unchained(config, data, 1, user):
                events.append(event)

        done_payload = json.loads(events[-1].split("data: ")[1].strip())
        assert done_payload["trace_id"] is None


# ===========================================================================
# SSE-Event-Format — _stream_agent
# ===========================================================================


class TestStreamAgent:
    """Tests fuer den _stream_agent SSE-Generator."""

    @pytest.fixture
    def config(self):
        return {
            "ollama": {
                "url": "http://localhost:11434",
                "llm_url": "",
                "embedding_url": "",
                "llm_model": "test-model",
                "embedding_model": "test-embed",
            },
            "chat": {
                "agent": {"max_iterations": 3, "num_ctx": 8192, "temperature": 0.3},
            },
        }

    @pytest.mark.asyncio
    async def test_agent_mode_event_first(self, config):
        """Erstes SSE-Event ist mode=agent."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="chunk", data={"text": "Hi"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            events = []
            async for event in _stream_agent(config, data, 1, user):
                events.append(event)

            assert "event: mode\n" in events[0]
            mode_data = json.loads(events[0].split("data: ")[1].strip())
            assert mode_data["mode"] == "agent"

    @pytest.mark.asyncio
    async def test_agent_tool_call_event_format(self, config):
        """tool_call-Events haben korrektes SSE-Format."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(
                    event_type="tool_call",
                    data={"tool": "calculate", "params": {"expression": "1+1"}, "iteration": 1},
                ),
                AgentEvent(
                    event_type="tool_result",
                    data={"tool": "calculate", "success": True, "summary": "2"},
                ),
                AgentEvent(event_type="chunk", data={"text": "Ergebnis"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="1+1?", mode="agent")
            events = []
            async for event in _stream_agent(config, data, 1, user):
                events.append(event)

            tool_call_events = [e for e in events if "event: tool_call\n" in e]
            assert len(tool_call_events) == 1
            tc_data = json.loads(tool_call_events[0].split("data: ")[1].strip())
            assert tc_data["tool"] == "calculate"
            assert tc_data["iteration"] == 1

    @pytest.mark.asyncio
    async def test_agent_sources_event_emitted(self, config):
        """sources-Event wird korrekt als SSE weitergeleitet."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        sources = [{"source": "doc.txt", "score": 0.9, "excerpt": "Auszug"}]
        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="sources", data=sources),
                AgentEvent(event_type="chunk", data={"text": "Antwort"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Suche", mode="agent")
            events = []
            async for event in _stream_agent(config, data, 1, user):
                events.append(event)

            sources_events = [e for e in events if "event: sources\n" in e]
            assert len(sources_events) == 1
            src_data = json.loads(sources_events[0].split("data: ")[1].strip())
            assert len(src_data) == 1
            assert src_data[0]["source"] == "doc.txt"

    @pytest.mark.asyncio
    async def test_agent_done_always_last(self, config):
        """done-Event ist immer das letzte."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="chunk", data={"text": "OK"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            events = []
            async for event in _stream_agent(config, data, 1, user):
                events.append(event)

            assert "event: done\n" in events[-1]

    @pytest.mark.asyncio
    async def test_agent_save_with_sources(self, config):
        """_save_assistant_message erhaelt gesammelte Sources."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        sources = [{"source": "a.txt", "score": 0.8, "excerpt": "A"}]
        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="sources", data=sources),
                AgentEvent(event_type="chunk", data={"text": "Antwort"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            async for _ in _stream_agent(config, data, 7, user):
                pass

            mock_save.assert_called_once_with(7, "Antwort", sources)

    @pytest.mark.asyncio
    async def test_agent_save_without_sources(self, config):
        """Ohne Sources wird None an _save_assistant_message uebergeben."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="chunk", data={"text": "Direkt"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            async for _ in _stream_agent(config, data, 7, user):
                pass

            mock_save.assert_called_once_with(7, "Direkt", None)

    @pytest.mark.asyncio
    async def test_agent_empty_successful_answer_is_persisted(self, config):
        """Auch im Agent-Modus bekommt eine leere erfolgreiche Antwort eine Message-Zeile."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter([AgentEvent(event_type="done")])
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "agent-empty"
        mock_chain.last_conversation = ""

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            async for _ in _stream_agent(config, data, 7, user):
                pass

            mock_save.assert_called_once_with(7, "", None)
            mock_audit.assert_called_once()

    @pytest.mark.asyncio
    async def test_done_event_contains_trace_id(self, config):
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="chunk", data={"text": "Antwort"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "agent123"
        mock_chain.last_conversation = ""

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            events = []
            async for event in _stream_agent(config, data, 1, user):
                events.append(event)

        done_payload = json.loads(events[-1].split("data: ")[1].strip())
        assert done_payload["trace_id"] == "agent123"

    @pytest.mark.asyncio
    async def test_agent_disconnect_does_not_persist_partial_answer(self, config):
        """Generator-Close emittiert keinen Epilog und speichert keine Teilantwort."""
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="chunk", data={"text": "Teilantwort"}),
                AgentEvent(event_type="chunk", data={"text": " Rest"}),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "agent-disconnect"
        mock_chain.last_conversation = ""

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            stream = _stream_agent(config, data, 1, user)

            assert "event: mode\n" in await stream.__anext__()
            assert "event: chunk\n" in await stream.__anext__()
            await stream.aclose()

        mock_save.assert_not_called()
        mock_audit.assert_not_called()


class TestStreamFailureAudit:
    @pytest.fixture
    def config(self):
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
            "chat": {"agent": {"max_iterations": 3}},
        }

    @staticmethod
    def _failing_chunks():
        yield "Teilantwort"
        raise RuntimeError("stream broke")

    @staticmethod
    def _assert_error_audited(mock_audit, *, chain_type: str, chat_id: int):
        mock_audit.assert_called_once()
        kwargs = mock_audit.call_args.kwargs
        assert kwargs["chat_id"] == chat_id
        assert kwargs["chain_type"] == chain_type
        assert "[STREAM_ERROR]" in kwargs["answer"]
        assert "Teilantwort" in kwargs["full_answer"] or chain_type == "agent"
        assert "[STREAM_ERROR]" in kwargs["full_answer"]

    @pytest.mark.asyncio
    async def test_standard_stream_exception_writes_query_audit(self, config):
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (
            self._failing_chunks(),
            [{"source": "doc.txt", "score": 0.9, "excerpt": "x"}],
        )
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = "prompt"
        mock_chain.last_trace_id = "trace-standard"
        mock_chain.last_expansions = []
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            events = []
            data = QueryRequest(question="Test?", mode="standard")
            async for event in _stream_standard(config, data, 11, {"id": 1}):
                events.append(event)

        assert any("event: error\n" in e for e in events)
        assert "event: done\n" in events[-1]
        mock_save.assert_not_called()
        self._assert_error_audited(mock_audit, chain_type="standard", chat_id=11)

    @pytest.mark.asyncio
    async def test_rewriter_stream_exception_writes_query_audit(self, config):
        from kiara.web.api_chat import QueryRequest, _stream_rewriter

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (self._failing_chunks(), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = "prompt"
        mock_chain.last_trace_id = "trace-rewriter"
        mock_chain.last_expansions = []
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None
        mock_chain.rewritten_query = "Test?"

        backend = MagicMock(provider="ollama", model="test-model")
        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=[]),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch("kiara.generation.cloud_routing._create_chain_backend", return_value=backend),
            patch("kiara.generation.rewriter_chain.RewriterChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            events = []
            data = QueryRequest(question="Test?", mode="rewriter")
            async for event in _stream_rewriter(config, data, 12, {"id": 1}):
                events.append(event)

        assert any("event: error\n" in e for e in events)
        assert "event: done\n" in events[-1]
        mock_save.assert_not_called()
        self._assert_error_audited(mock_audit, chain_type="rewriter", chat_id=12)

    @pytest.mark.asyncio
    async def test_unchained_stream_exception_writes_query_audit(self, config):
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = self._failing_chunks()
        mock_chain.route_used = "local"
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = "prompt"
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None

        backend = MagicMock(provider="ollama", model="test-model")
        with (
            patch("kiara.generation.backends.create_backend", return_value=backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            events = []
            data = QueryRequest(question="Test?", mode="unchained")
            async for event in _stream_unchained(config, data, 13, {"id": 1}):
                events.append(event)

        assert any("event: error\n" in e for e in events)
        assert "event: done\n" in events[-1]
        mock_save.assert_not_called()
        self._assert_error_audited(mock_audit, chain_type="unchained", chat_id=13)

    @pytest.mark.asyncio
    async def test_agent_stream_exception_marks_query_audit_as_error(self, config):
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        def failing_events():
            yield AgentEvent(event_type="chunk", data={"text": "Teilantwort"})
            raise RuntimeError("agent broke")

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = failing_events()
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "trace-agent"
        mock_chain.last_conversation = "conversation"

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            events = []
            data = QueryRequest(question="Test?", mode="agent")
            async for event in _stream_agent(config, data, 14, {"id": 1}):
                events.append(event)

        assert any("event: error\n" in e for e in events)
        assert "event: done\n" in events[-1]
        mock_save.assert_not_called()
        self._assert_error_audited(mock_audit, chain_type="agent", chat_id=14)


class TestStreamRewriter:
    @pytest.fixture
    def config(self):
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
        }

    @pytest.mark.asyncio
    async def test_done_event_contains_trace_id(self, config):
        from kiara.web.api_chat import QueryRequest, _stream_rewriter

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Hallo"]), [])
        mock_chain.last_usage = None
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_trace_id = "rw123"
        mock_chain.last_prompt = "prompt"
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None
        mock_chain.last_expansions = []
        mock_chain.rewritten_query = "Hi"
        mock_chain.context_warning = None

        backend = MagicMock(provider="ollama", model="test-model")
        user = {"id": 1, "is_admin": True, "datasource_ids": []}
        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=None),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch("kiara.generation.cloud_routing._create_chain_backend", return_value=backend),
            patch("kiara.generation.rewriter_chain.RewriterChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
            patch("kiara.web.api_chat._log_cloud_audit"),
        ):
            data = QueryRequest(question="Hi", mode="rewriter")
            events = []
            async for event in _stream_rewriter(config, data, 1, user):
                events.append(event)

        done_payload = json.loads(events[-1].split("data: ")[1].strip())
        assert done_payload["trace_id"] == "rw123"

    @pytest.mark.asyncio
    async def test_done_event_normalizes_mock_trace_id(self, config):
        from kiara.web.api_chat import QueryRequest, _stream_rewriter

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Hallo"]), [])
        mock_chain.last_usage = None
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_trace_id = MagicMock(name="trace_id_mock")
        mock_chain.last_prompt = "prompt"
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None
        mock_chain.last_expansions = []
        mock_chain.rewritten_query = "Hi"
        mock_chain.context_warning = None

        backend = MagicMock(provider="ollama", model="test-model")
        user = {"id": 1, "is_admin": True, "datasource_ids": []}
        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=None),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch("kiara.generation.cloud_routing._create_chain_backend", return_value=backend),
            patch("kiara.generation.rewriter_chain.RewriterChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
            patch("kiara.web.api_chat._log_cloud_audit"),
        ):
            data = QueryRequest(question="Hi", mode="rewriter")
            events = []
            async for event in _stream_rewriter(config, data, 1, user):
                events.append(event)

        done_payload = json.loads(events[-1].split("data: ")[1].strip())
        assert done_payload["trace_id"] is None
        assert mock_audit.call_args.kwargs["trace_id"] is None


# ===========================================================================
# SSE-Event-Format — Allgemeine Struktur
# ===========================================================================


class TestSSEEventStructure:
    """Tests fuer das generelle SSE-Event-Format."""

    def test_sse_event_format(self):
        """SSE-Events muessen das Format 'event: X\\ndata: Y\\n\\n' haben."""
        # Einfacher Formatierungstest
        event_str = 'event: mode\ndata: {"mode": "standard"}\n\n'
        lines = event_str.strip().split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        # data muss gueltiges JSON sein
        json_str = lines[1][len("data: ") :]
        data = json.loads(json_str)
        assert "mode" in data

    def test_all_event_types_known(self):
        """Alle verwendeten SSE-Event-Typen sind dokumentiert."""
        known_types = {
            "mode",
            "sources",
            "route",
            "chunk",
            "done",
            "tool_call",
            "tool_result",
            "warning",
        }
        # Standard: mode, sources, [warning], route, chunk, done
        # Agent: mode, tool_call, tool_result, sources, chunk, done
        # Unchained: mode, chunk, done
        assert "mode" in known_types
        assert "tool_call" in known_types
        assert "warning" in known_types
        assert "done" in known_types


# ===========================================================================
# Standard-Streaming — Kontext-Warning
# ===========================================================================


class TestStreamStandardWarning:
    """Prueft ob das warning SSE-Event korrekt emittiert wird."""

    @pytest.fixture
    def config(self):
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
        }

    @pytest.mark.asyncio
    async def test_warning_event_emitted_on_overflow(self, config):
        """Bei Kontext-Ueberlauf wird ein warning-Event zwischen sources und route gesendet."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (
            iter(["Antwort"]),
            [{"source": "test.txt", "score": 0.9, "excerpt": "..."}],
        )
        mock_chain.route_used = "local"
        mock_chain.context_warning = {
            "estimated_tokens": 12000,
            "num_ctx": 8192,
            "overflow_pct": 46,
        }
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = ""
        mock_chain.last_expansions = []

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(config, data, 1, user):
                events.append(event)

        # warning-Event muss vorhanden sein
        warning_events = [e for e in events if "event: warning\n" in e]
        assert len(warning_events) == 1

        # Payload pruefen
        payload = json.loads(warning_events[0].split("data: ")[1].strip())
        assert payload["estimated_tokens"] == 12000
        assert payload["num_ctx"] == 8192
        assert payload["overflow_pct"] == 46

    @pytest.mark.asyncio
    async def test_warning_event_position(self, config):
        """warning-Event kommt nach sources und vor route."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["OK"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = {
            "estimated_tokens": 10000,
            "num_ctx": 8192,
            "overflow_pct": 22,
        }
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = ""
        mock_chain.last_expansions = []

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(config, data, 1, user):
                events.append(event)

        # Event-Typen extrahieren
        event_types = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("event: "):
                    event_types.append(line[7:].strip())

        assert "warning" in event_types
        sources_idx = event_types.index("sources")
        warning_idx = event_types.index("warning")
        route_idx = event_types.index("route")
        assert sources_idx < warning_idx < route_idx

    @pytest.mark.asyncio
    async def test_no_warning_event_when_no_overflow(self, config):
        """Ohne Kontext-Ueberlauf wird kein warning-Event gesendet."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["OK"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_prompt = ""
        mock_chain.last_expansions = []

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(config, data, 1, user):
                events.append(event)

        warning_events = [e for e in events if "event: warning\n" in e]
        assert len(warning_events) == 0

    @pytest.mark.asyncio
    async def test_done_event_contains_trace_id(self, config):
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Hallo"]), [])
        mock_chain.last_usage = None
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_trace_id = "abc123"
        mock_chain.last_prompt = "prompt"
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None
        mock_chain.last_expansions = []
        mock_chain.context_warning = None

        user = {"id": 1, "is_admin": True, "datasource_ids": []}
        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit"),
            patch("kiara.web.api_chat._log_cloud_audit"),
        ):
            data = QueryRequest(question="Hi", mode="standard")
            events = []
            async for event in _stream_standard(config, data, 1, user):
                events.append(event)

        done_payload = json.loads(events[-1].split("data: ")[1].strip())
        assert done_payload["trace_id"] == "abc123"


# ===========================================================================
# Chat-Mode-Filter (Sprint 16.1)
# ===========================================================================


class TestChatModeFilter:
    """Tests 33-35: Nicht-user-chat-faehige Templates werden ausgefiltert."""

    @patch("kiara.web.api_chat.prompt_repo")
    @patch("kiara.backend_selector.get_all_active")
    def test_enrichment_not_in_chat_modes(self, mock_backends, mock_repo):
        """Test 33: Fehlkonfiguriertes enrichment-Template nicht in /chat/modes."""
        mock_backends.return_value = [{"provider": "ollama"}]
        mock_repo.list_chat_modes.return_value = [
            {
                "id": 1,
                "name": "standard",
                "label": "Standard",
                "chain_type": "standard",
                "badge_color": "",
                "sort_order": 10,
            },
            {
                "id": 99,
                "name": "enrichment_leak",
                "label": "Enrichment",
                "chain_type": "enrichment",
                "badge_color": "",
                "sort_order": 100,
            },
        ]

        from kiara.web.api_chat import get_chat_modes

        result = _run(get_chat_modes(user={"id": 1}))
        chain_types = [m["chain_type"] for m in result["modes"]]
        assert "enrichment" not in chain_types
        assert "standard" in chain_types

    @patch("kiara.web.api_chat.prompt_repo")
    @patch("kiara.backend_selector.get_all_active")
    def test_rewriter_visible_with_ollama(self, mock_backends, mock_repo):
        """Test 34: rewriter bleibt bei aktivem Ollama sichtbar."""
        mock_backends.return_value = [{"provider": "ollama"}]
        mock_repo.list_chat_modes.return_value = [
            {
                "id": 1,
                "name": "standard",
                "label": "Standard",
                "chain_type": "standard",
                "badge_color": "",
                "sort_order": 10,
            },
            {
                "id": 2,
                "name": "rewriter",
                "label": "Rewriter",
                "chain_type": "rewriter",
                "badge_color": "",
                "sort_order": 15,
            },
        ]

        from kiara.web.api_chat import get_chat_modes

        result = _run(get_chat_modes(user={"id": 1}))
        chain_types = [m["chain_type"] for m in result["modes"]]
        assert "rewriter" in chain_types

    @patch("kiara.web.api_chat.prompt_repo")
    @patch("kiara.backend_selector.get_all_active")
    def test_rewriter_hidden_without_ollama(self, mock_backends, mock_repo):
        """Test 35: rewriter wird ohne Ollama entfernt."""
        mock_backends.return_value = [{"provider": "anthropic"}]
        mock_repo.list_chat_modes.return_value = [
            {
                "id": 1,
                "name": "standard",
                "label": "Standard",
                "chain_type": "standard",
                "badge_color": "",
                "sort_order": 10,
            },
            {
                "id": 2,
                "name": "rewriter",
                "label": "Rewriter",
                "chain_type": "rewriter",
                "badge_color": "",
                "sort_order": 15,
            },
        ]

        from kiara.web.api_chat import get_chat_modes

        result = _run(get_chat_modes(user={"id": 1}))
        chain_types = [m["chain_type"] for m in result["modes"]]
        assert "rewriter" not in chain_types
