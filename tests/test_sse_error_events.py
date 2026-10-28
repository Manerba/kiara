"""Tests fuer SSE Error-Events (Sprint 2/3 Backend-Abstraction).

Testet:
- SSE error-Event Emission bei LLM-Fehlern (_stream_standard, _stream_unchained)
- Backend-Fehler werden als SSE error-Event propagiert (nicht silent fail)
- error-Event hat korrektes Format (event: error, data: {"message": ...})
- BackendUnavailableError wird als SSE error-Event propagiert
- Agent-Adapter-Factory erzeugt richtigen Adapter-Typ je nach Provider
- Agent-Prepare-Fehler (RuntimeError) werden als SSE error+done propagiert
"""

import json
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_backend(provider="ollama", model="test-model"):
    """Erzeugt ein Mock-LLMBackend mit provider/model."""
    backend = MagicMock()
    backend.provider = provider
    backend.model = model
    backend.last_usage = {}
    backend.last_anon_question = ""
    backend.last_anon_answer = ""
    backend.last_entities_anonymized = 0
    backend.anonymizer_stats = {"total_entities": 0, "entity_types": {}}
    return backend


def _make_mock_chain(
    *,
    query_stream_return=None,
    query_stream_side_effect=None,
    route_used="local",
    provider="ollama",
    model="test-model",
):
    """Erzeugt ein Mock-Chain-Objekt mit allen erwarteten Attributen.

    Setzt sowohl query_stream return/side_effect als auch die Attribut-Properties,
    die _stream_standard/_stream_unchained nach dem Streaming lesen.
    """
    chain = MagicMock()
    if query_stream_return is not None:
        chain.query_stream.return_value = query_stream_return
    if query_stream_side_effect is not None:
        chain.query_stream.side_effect = query_stream_side_effect

    chain.route_used = route_used
    chain.context_warning = None
    chain.last_usage = {}
    chain.last_prompt = ""
    chain.last_expansions = []
    chain.last_entities_anonymized = 0
    chain.last_anon_question = ""
    chain.last_anon_answer = ""

    mock_backend = _make_mock_backend(provider, model)
    chain.backend = mock_backend
    return chain


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
# _stream_standard — SSE error-Event bei LLM-Fehler
# ===========================================================================


class TestStreamStandardErrorEvent:
    """Tests fuer error-Event Emission in _stream_standard."""

    @pytest.mark.asyncio
    async def test_llm_error_emits_sse_error_event(self, standard_config):
        """Wenn der LLM-Generator eine Exception wirft, wird ein SSE error-Event gesendet."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        def _failing_generator():
            yield "Anfang"
            raise RuntimeError("Ollama connection refused")

        mock_chain = _make_mock_chain(
            query_stream_return=(
                _failing_generator(),
                [{"source": "doc.txt", "score": 0.9, "excerpt": "..."}],
            ),
        )

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

        # error-Event muss vorhanden sein
        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        # _save_assistant_message darf bei Fehler NICHT aufgerufen werden
        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_event_has_correct_format(self, standard_config):
        """error-Event hat das Format: event: error\\ndata: {\"message\": ...}\\n\\n"""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        def _failing_generator():
            raise ConnectionError("Netzwerkfehler")
            yield  # Generator-Syntax erfordert yield.

        mock_chain = _make_mock_chain(
            query_stream_return=(_failing_generator(), []),
        )

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        # Format pruefen: event: error\ndata: JSON\n\n
        error_str = error_events[0]
        lines = error_str.strip().split("\n")
        assert lines[0] == "event: error"
        assert lines[1].startswith("data: ")

        # data muss gueltiges JSON mit "message"-Key sein
        payload = json.loads(lines[1][len("data: ") :])
        assert "message" in payload
        # Generische Fehlermeldung (kein Exception-Leak an Client, F-58)
        assert "Fehler" in payload["message"]

    @pytest.mark.asyncio
    async def test_error_event_contains_error_message(self, standard_config):
        """Die Fehlermeldung wird korrekt im error-Event transportiert."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        error_msg = "GPU out of memory: CUDA allocation failed"

        def _failing_generator():
            raise RuntimeError(error_msg)
            yield

        mock_chain = _make_mock_chain(
            query_stream_return=(_failing_generator(), []),
        )

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        payload = json.loads(error_events[0].split("data: ")[1].strip())
        # Generische Fehlermeldung statt Exception-Details (F-58)
        assert "Fehler" in payload["message"]

    @pytest.mark.asyncio
    async def test_done_event_follows_error(self, standard_config):
        """Nach dem error-Event kommt noch ein done-Event (damit Frontend aufraeumt)."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        def _failing_generator():
            raise ValueError("Unbekannter Fehler")
            yield

        mock_chain = _make_mock_chain(
            query_stream_return=(_failing_generator(), []),
        )

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        # Event-Typen extrahieren
        event_types = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("event: "):
                    event_types.append(line[7:].strip())

        # error und done muessen vorhanden sein
        assert "error" in event_types
        assert "done" in event_types

        # done kommt direkt nach error
        error_idx = event_types.index("error")
        done_idx = event_types.index("done")
        assert done_idx == error_idx + 1

    @pytest.mark.asyncio
    async def test_partial_chunks_before_error(self, standard_config):
        """Chunks vor dem Fehler werden gesendet, dann error + done."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        def _partial_then_fail():
            yield "Teil1"
            yield "Teil2"
            raise RuntimeError("Mid-stream failure")

        mock_chain = _make_mock_chain(
            query_stream_return=(_partial_then_fail(), []),
        )

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

        chunk_events = [e for e in events if "event: chunk\n" in e]
        error_events = [e for e in events if "event: error\n" in e]

        # 2 Chunks wurden gesendet bevor der Fehler kam
        assert len(chunk_events) == 2
        assert len(error_events) == 1

        # Bei Fehler wird keine Nachricht gespeichert
        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_done_payload_after_error_has_no_trace_id(self, standard_config):
        from kiara.web.api_chat import QueryRequest, _stream_standard

        def _failing_generator():
            raise RuntimeError("boom")
            yield

        mock_chain = _make_mock_chain(
            query_stream_return=(_failing_generator(), []),
        )

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        done_payload = json.loads(
            [e for e in events if "event: done\n" in e][-1].split("data: ")[1].strip()
        )
        assert done_payload == {"chat_id": 1}


class TestStreamRewriterErrorEvent:
    @pytest.mark.asyncio
    async def test_done_payload_after_backend_init_error_has_no_trace_id(self, standard_config):
        from kiara.web.api_chat import QueryRequest, _stream_rewriter

        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=None),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch(
                "kiara.generation.cloud_routing._create_chain_backend",
                side_effect=RuntimeError("backend kaputt"),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True, "datasource_ids": []}
            data = QueryRequest(question="Test?", mode="rewriter")
            events = []
            async for event in _stream_rewriter(standard_config, data, 1, user):
                events.append(event)

        done_payload = json.loads(
            [e for e in events if "event: done\n" in e][-1].split("data: ")[1].strip()
        )
        assert done_payload == {"chat_id": 1}
        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_done_payload_after_stream_generator_error_has_no_trace_id(self, standard_config):
        from kiara.web.api_chat import QueryRequest, _stream_rewriter

        def _failing_generator():
            raise RuntimeError("boom")
            yield

        mock_chain = _make_mock_chain(
            query_stream_return=(_failing_generator(), []),
        )
        mock_chain.rewritten_query = "Test?"

        backend = MagicMock(provider="ollama", model="test-model")
        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=None),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch("kiara.generation.cloud_routing._create_chain_backend", return_value=backend),
            patch("kiara.generation.rewriter_chain.RewriterChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True, "datasource_ids": []}
            data = QueryRequest(question="Test?", mode="rewriter")
            events = []
            async for event in _stream_rewriter(standard_config, data, 1, user):
                events.append(event)

        done_payload = json.loads(
            [e for e in events if "event: done\n" in e][-1].split("data: ")[1].strip()
        )
        assert done_payload == {"chat_id": 1}
        mock_save.assert_not_called()


class TestStreamUnchainedErrorDonePayload:
    @pytest.mark.asyncio
    async def test_done_payload_after_error_has_no_trace_id(self, standard_config):
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = _make_mock_chain(
            query_stream_side_effect=RuntimeError("kaputt"),
        )
        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="unchained")
            events = []
            async for event in _stream_unchained(standard_config, data, 1, user):
                events.append(event)

        done_payload = json.loads(
            [e for e in events if "event: done\n" in e][-1].split("data: ")[1].strip()
        )
        assert done_payload == {"chat_id": 1}


class TestStreamAgentErrorDonePayload:
    @pytest.mark.asyncio
    async def test_agent_prepare_error_done_payload_has_no_trace_id(self, standard_config):
        from kiara.web.api_chat import QueryRequest, _stream_agent

        exc = RuntimeError("Backend nicht verfuegbar")
        with (
            patch("kiara.web.api_chat._prepare_agent_context", side_effect=exc),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="agent")
            events = []
            async for event in _stream_agent(standard_config, data, 1, user):
                events.append(event)

        done_payload = json.loads(
            [e for e in events if "event: done\n" in e][-1].split("data: ")[1].strip()
        )
        assert done_payload == {"chat_id": 1}

    @pytest.mark.asyncio
    async def test_agent_runtime_error_done_payload_has_no_trace_id(self, standard_config):
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="error", data={"message": "boom"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "should-not-leak"
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
            data = QueryRequest(question="Test?", mode="agent")
            events = []
            async for event in _stream_agent(standard_config, data, 1, user):
                events.append(event)

        done_payload = json.loads(
            [e for e in events if "event: done\n" in e][-1].split("data: ")[1].strip()
        )
        assert done_payload == {"chat_id": 1}

    @pytest.mark.asyncio
    async def test_agent_runtime_error_logs_no_trace_id_to_audit(self, standard_config):
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(
            [
                AgentEvent(event_type="error", data={"message": "boom"}),
                AgentEvent(event_type="done"),
            ]
        )
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "should-not-leak"
        mock_chain.last_conversation = ""

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="agent")
            async for _ in _stream_agent(standard_config, data, 1, user):
                pass

        assert mock_audit.call_args.kwargs["trace_id"] is None

    @pytest.mark.asyncio
    async def test_agent_stream_exception_done_payload_has_no_trace_id(self, standard_config):
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        def _failing_events():
            yield AgentEvent(event_type="chunk", data={"text": "Teilantwort"})
            raise RuntimeError("kaputt")

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = _failing_events()
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "should-not-leak"
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
            data = QueryRequest(question="Test?", mode="agent")
            events = []
            async for event in _stream_agent(standard_config, data, 1, user):
                events.append(event)

        done_payload = json.loads(
            [e for e in events if "event: done\n" in e][-1].split("data: ")[1].strip()
        )
        assert done_payload == {"chat_id": 1}

    @pytest.mark.asyncio
    async def test_agent_stream_exception_logs_no_trace_id_to_audit(self, standard_config):
        from kiara.generation.agent_chain import AgentEvent
        from kiara.web.api_chat import QueryRequest, _stream_agent

        def _failing_events():
            yield AgentEvent(event_type="chunk", data={"text": "Teilantwort"})
            raise RuntimeError("kaputt")

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = _failing_events()
        mock_chain.last_usage = {}
        mock_chain._tool_events = []
        mock_chain.last_trace_id = "should-not-leak"
        mock_chain.last_conversation = ""

        backend = {"url": "http://localhost:11434", "model": "test-model", "provider": "ollama"}
        with (
            patch(
                "kiara.web.api_chat._prepare_agent_context",
                return_value=(mock_chain, backend, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="agent")
            async for _ in _stream_agent(standard_config, data, 1, user):
                pass

        assert mock_audit.call_args.kwargs["trace_id"] is None


# ===========================================================================
# _stream_standard — BackendUnavailableError bei Context-Aufbau
# ===========================================================================


class TestStreamStandardBackendUnavailable:
    """Tests fuer BackendUnavailableError in _stream_standard (query_stream)."""

    @pytest.mark.asyncio
    async def test_backend_unavailable_in_query_stream_emits_sse_error(self, standard_config):
        """BackendUnavailableError in query_stream wird als SSE error-Event propagiert."""
        from kiara.backend_selector import BackendUnavailableError
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_chain = _make_mock_chain()
        mock_chain.query_stream.side_effect = BackendUnavailableError(
            "Embedding-Backend nicht erreichbar"
        )

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

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        payload = json.loads(error_events[0].split("data: ")[1].strip())
        assert "Embedding-Backend" in payload["message"]

        mock_save.assert_not_called()


# ===========================================================================
# _stream_unchained — SSE error-Event bei LLM-Fehler
# ===========================================================================


class TestStreamUnchainedErrorEvent:
    """Tests fuer error-Event Emission in _stream_unchained."""

    @pytest.mark.asyncio
    async def test_unchained_llm_error_emits_error_event(self, standard_config):
        """Fehler im Unchained-Modus wird als SSE error-Event gesendet."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = _make_mock_chain(
            query_stream_side_effect=RuntimeError("Model not found: test-model"),
        )
        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Hi", mode="unchained")
            events = []
            async for event in _stream_unchained(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        payload = json.loads(error_events[0].split("data: ")[1].strip())
        assert "message" in payload
        # Generische Fehlermeldung (kein Exception-Leak, F-58)
        assert "Fehler" in payload["message"]

        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_unchained_error_event_format(self, standard_config):
        """Unchained error-Event hat identisches Format wie Standard."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = _make_mock_chain(
            query_stream_side_effect=ConnectionError("Timeout"),
        )
        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="unchained")
            events = []
            async for event in _stream_unchained(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        error_str = error_events[0]
        assert error_str.endswith("\n\n"), "SSE-Event muss mit doppeltem Newline enden"

        lines = error_str.strip().split("\n")
        assert lines[0] == "event: error"
        assert lines[1].startswith("data: ")
        payload = json.loads(lines[1][len("data: ") :])
        assert "message" in payload

    @pytest.mark.asyncio
    async def test_unchained_done_after_error(self, standard_config):
        """Auch bei Unchained-Fehler kommt done-Event nach error."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = _make_mock_chain(
            query_stream_side_effect=Exception("Generischer Fehler"),
        )
        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="unchained")
            events = []
            async for event in _stream_unchained(standard_config, data, 1, user):
                events.append(event)

        event_types = []
        for e in events:
            for line in e.split("\n"):
                if line.startswith("event: "):
                    event_types.append(line[7:].strip())

        assert "error" in event_types
        assert "done" in event_types
        assert event_types[-1] == "done"

    @pytest.mark.asyncio
    async def test_unchained_partial_stream_then_error(self, standard_config):
        """Unchained: Chunks vor dem Fehler + error-Event danach."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        def _partial_fail():
            yield "Start"
            raise RuntimeError("Stream abgebrochen")

        mock_chain = _make_mock_chain(
            query_stream_return=_partial_fail(),
        )
        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="unchained")
            events = []
            async for event in _stream_unchained(standard_config, data, 1, user):
                events.append(event)

        chunk_events = [e for e in events if "event: chunk\n" in e]
        error_events = [e for e in events if "event: error\n" in e]
        assert len(chunk_events) == 1
        assert len(error_events) == 1

        mock_save.assert_not_called()


# ===========================================================================
# _stream_unchained — BackendUnavailableError bei Backend-Erstellung
# ===========================================================================


class TestStreamUnchainedBackendUnavailable:
    """Tests fuer BackendUnavailableError in _stream_unchained."""

    @pytest.mark.asyncio
    async def test_backend_unavailable_emits_sse_error(self, standard_config):
        """BackendUnavailableError bei create_backend wird als SSE error-Event propagiert."""
        from kiara.backend_selector import BackendUnavailableError
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        with (
            patch(
                "kiara.generation.backends.create_backend",
                side_effect=BackendUnavailableError("Backend ID 99 nicht gefunden"),
            ),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="unchained", llm_backend_id=99)
            events = []
            async for event in _stream_unchained(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        payload = json.loads(error_events[0].split("data: ")[1].strip())
        assert "Backend" in payload["message"] or "verfuegbar" in payload["message"]

        mock_save.assert_not_called()


# ===========================================================================
# Backend-Fehler propagiert als SSE error-Event (kein stiller Fallback)
# ===========================================================================


class TestBackendErrorPropagation:
    """Tests dass Backend-Fehler als SSE error-Event propagiert werden.

    In Sprint 2 gibt es keinen Cloud-Fallback mehr — Fehler werden direkt
    als SSE error-Event an den Client gesendet.
    """

    @pytest.mark.asyncio
    async def test_backend_error_becomes_sse_error_event(self, standard_config):
        """Backend-Fehler beim Streaming wird als SSE error-Event propagiert."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        def _backend_failing_generator():
            raise ConnectionError("Anthropic API error: 429 Too Many Requests")
            yield

        mock_chain = _make_mock_chain(
            query_stream_return=(
                _backend_failing_generator(),
                [{"source": "doc.txt", "score": 0.8, "excerpt": "..."}],
            ),
            route_used="cloud",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Analyse?", mode="standard", llm_backend_id=5)
            events = []
            async for event in _stream_standard(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        payload = json.loads(error_events[0].split("data: ")[1].strip())
        # Generische Fehlermeldung (kein Exception-Leak, F-58)
        assert "Fehler" in payload["message"]

    @pytest.mark.asyncio
    async def test_unchained_backend_error_becomes_sse_error(self, standard_config):
        """Backend-Fehler im Unchained-Modus wird als SSE error-Event propagiert."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_chain = _make_mock_chain(
            query_stream_side_effect=ConnectionError("OpenAI API rate limit exceeded"),
            route_used="cloud",
            provider="openai",
            model="gpt-4o",
        )
        mock_backend = _make_mock_backend(provider="openai", model="gpt-4o")

        with (
            patch("kiara.generation.backends.create_backend", return_value=mock_backend),
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Analyse?", mode="unchained", llm_backend_id=7)
            events = []
            async for event in _stream_unchained(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        payload = json.loads(error_events[0].split("data: ")[1].strip())
        assert "Fehler" in payload["message"]


# ===========================================================================
# Agent-Adapter-Factory — richtiger Adapter-Typ je nach Provider
# ===========================================================================


class TestAgentAdapterFactory:
    """create_tool_adapter() erzeugt den passenden Adapter-Typ."""

    def test_cloud_backend_erzeugt_anthropic_adapter(self):
        """Anthropic-Backend erzeugt AnthropicToolAdapter (Cloud wird jetzt unterstuetzt)."""
        from kiara.generation.tool_adapter import AnthropicToolAdapter, create_tool_adapter

        cloud_backend = {
            "id": 5,
            "name": "Claude",
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "url": "https://api.anthropic.com",
        }
        config = {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}}
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            adapter = create_tool_adapter(cloud_backend, config)
        assert isinstance(adapter, AnthropicToolAdapter)

    def test_ollama_backend_erzeugt_ollama_adapter(self):
        """Ollama-Backend erzeugt OllamaToolAdapter."""
        from kiara.generation.tool_adapter import OllamaToolAdapter, create_tool_adapter

        ollama_backend = {
            "id": 2,
            "name": "Lokaler Ollama",
            "provider": "ollama",
            "model": "llama3:8b",
            "url": "http://gpu-server.local:11434",
        }
        config = {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}}
        with patch("ollama.Client"):
            adapter = create_tool_adapter(ollama_backend, config)
        assert isinstance(adapter, OllamaToolAdapter)

    def test_openai_backend_erzeugt_openai_adapter(self):
        """OpenAI-Backend erzeugt OpenAIToolAdapter."""
        from kiara.generation.tool_adapter import OpenAIToolAdapter, create_tool_adapter

        openai_backend = {
            "id": 7,
            "name": "GPT-4",
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "url": "https://api.openai.com",
        }
        config = {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}}
        mock_openai = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            adapter = create_tool_adapter(openai_backend, config)
        assert isinstance(adapter, OpenAIToolAdapter)

    def test_default_backend_erzeugt_ollama_adapter(self):
        """Backend ohne expliziten Provider erzeugt OllamaToolAdapter."""
        from kiara.generation.tool_adapter import OllamaToolAdapter, create_tool_adapter

        default_backend = {
            "url": "http://gpu-server.local:11434",
            "model": "qwen3:8b",
        }
        config = {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}}
        with patch("ollama.Client"):
            adapter = create_tool_adapter(default_backend, config)
        assert isinstance(adapter, OllamaToolAdapter)


# ===========================================================================
# SSE-Event-Struktur — error-Event Typ
# ===========================================================================


class TestSSEErrorEventStructure:
    """Strukturelle Tests fuer das error-Event-Format."""

    def test_error_event_in_known_types(self):
        """error ist ein bekannter SSE-Event-Typ."""
        known_types = {
            "mode",
            "sources",
            "route",
            "chunk",
            "done",
            "tool_call",
            "tool_result",
            "warning",
            "error",
            "usage",
        }
        assert "error" in known_types

    def test_error_event_json_structure(self):
        """error-Event data hat einen 'message'-Key mit String-Wert."""
        error_str = 'event: error\ndata: {"message": "Testfehler"}\n\n'
        lines = error_str.strip().split("\n")
        assert lines[0] == "event: error"
        payload = json.loads(lines[1][len("data: ") :])
        assert isinstance(payload["message"], str)
        assert payload["message"] == "Testfehler"

    def test_error_event_double_newline_termination(self):
        """SSE-Events muessen mit doppeltem Newline terminiert sein."""
        import json as json_mod

        msg = "Test Error"
        err_data = json_mod.dumps({"message": msg}, ensure_ascii=False)
        event = f"event: error\ndata: {err_data}\n\n"
        assert event.endswith("\n\n")


# ===========================================================================
# _stream_agent — RuntimeError bei Prepare
# ===========================================================================


class TestColBERTInitializationStreamErrors:
    def _assert_controlled_error_done(self, events, mock_audit, *, chain_type: str, chat_id: int):
        error_events = [event for event in events if "event: error\n" in event]
        assert len(error_events) == 1
        payload = json.loads(error_events[0].split("data: ")[1].strip())
        assert "ColBERT konnte nicht initialisiert werden" in payload["message"]
        assert "event: done\n" in events[-1]

        mock_audit.assert_called_once()
        kwargs = mock_audit.call_args.kwargs
        assert kwargs["chat_id"] == chat_id
        assert kwargs["chain_type"] == chain_type
        assert "[STREAM_ERROR]" in kwargs["answer"]
        assert "ColBERT konnte nicht initialisiert werden" in kwargs["answer"]

    @pytest.mark.asyncio
    async def test_standard_prepare_colbert_init_error_emits_sse_error_done_and_audit(
        self, standard_config
    ):
        from kiara.exceptions import ColBERTInitializationError
        from kiara.web.api_chat import QueryRequest, _stream_standard

        exc = ColBERTInitializationError(
            "ColBERT konnte nicht initialisiert werden: HTTP 409 Conflict"
        )
        with (
            patch("kiara.web.api_chat._prepare_standard_context", side_effect=exc),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            events = []
            data = QueryRequest(question="Test?", mode="standard")
            async for event in _stream_standard(standard_config, data, 41, {"id": 1}):
                events.append(event)

        mock_save.assert_not_called()
        self._assert_controlled_error_done(
            events,
            mock_audit,
            chain_type="standard",
            chat_id=41,
        )

    @pytest.mark.asyncio
    async def test_rewriter_prepare_colbert_init_error_emits_sse_error_done_and_audit(
        self, standard_config
    ):
        from kiara.exceptions import ColBERTInitializationError
        from kiara.web.api_chat import QueryRequest, _stream_rewriter

        backend = MagicMock(provider="ollama", model="test-model")
        exc = ColBERTInitializationError(
            "ColBERT konnte nicht initialisiert werden: HTTP 409 Conflict"
        )
        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=[]),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch("kiara.generation.cloud_routing._create_chain_backend", return_value=backend),
            patch("kiara.generation.rewriter_chain.RewriterChain", side_effect=exc),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            events = []
            data = QueryRequest(question="Test?", mode="rewriter")
            user = {"id": 1, "is_admin": True, "datasource_ids": []}
            async for event in _stream_rewriter(standard_config, data, 42, user):
                events.append(event)

        mock_save.assert_not_called()
        self._assert_controlled_error_done(
            events,
            mock_audit,
            chain_type="rewriter",
            chat_id=42,
        )

    @pytest.mark.asyncio
    async def test_agent_prepare_colbert_init_error_emits_sse_error_done_and_audit(
        self, standard_config
    ):
        from kiara.exceptions import ColBERTInitializationError
        from kiara.web.api_chat import QueryRequest, _stream_agent

        exc = ColBERTInitializationError(
            "ColBERT konnte nicht initialisiert werden: HTTP 409 Conflict"
        )
        with (
            patch("kiara.web.api_chat._prepare_agent_context", side_effect=exc),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            events = []
            data = QueryRequest(question="Test?", mode="agent")
            async for event in _stream_agent(standard_config, data, 43, {"id": 1}):
                events.append(event)

        mock_save.assert_not_called()
        self._assert_controlled_error_done(
            events,
            mock_audit,
            chain_type="agent",
            chat_id=43,
        )


class TestStreamAgentPrepareErrors:
    @pytest.mark.asyncio
    async def test_prepare_context_runtime_error_emits_sse_error_done(self, standard_config):
        from kiara.web.api_chat import QueryRequest, _stream_agent

        exc = RuntimeError(
            "Cloud-Backend durch Restriction-Level gesperrt und kein lokales LLM-Backend verfuegbar"
        )

        with (
            patch("kiara.web.api_chat._prepare_agent_context", side_effect=exc),
            patch("kiara.web.api_chat._save_assistant_message") as mock_save,
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="agent")
            events = []
            async for event in _stream_agent(standard_config, data, 1, user):
                events.append(event)

        error_events = [e for e in events if "event: error\n" in e]
        assert len(error_events) == 1

        payload = json.loads(error_events[0].split("data: ")[1].strip())
        assert "Backend nicht verfuegbar" in payload["message"]
        assert "event: done\n" in events[-1]
        mock_save.assert_not_called()
