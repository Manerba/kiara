"""Tests fuer LLM-Backend-Auswahl im Chat.

Testet:
    - QueryRequest akzeptiert llm_backend_id (int und None)
    - /api/chat/filters liefert llm_backends-Feld
    - llm_backend_id wird an _create_chain_backend (Standard/Rewriter)
      bzw. create_backend (Unchained) bzw. _resolve_backend_dict (Agent) durchgereicht
"""

from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# QueryRequest — llm_backend_id Feld
# ===========================================================================


class TestQueryRequestBackendId:
    """Tests fuer das llm_backend_id-Feld im QueryRequest."""

    def test_llm_backend_id_default_none(self):
        """llm_backend_id ist per Default None."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test")
        assert req.llm_backend_id is None

    def test_llm_backend_id_accepts_int(self):
        """llm_backend_id akzeptiert einen Integer-Wert."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test", llm_backend_id=42)
        assert req.llm_backend_id == 42

    def test_llm_backend_id_accepts_none_explicit(self):
        """llm_backend_id akzeptiert explizites None."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(question="Test", llm_backend_id=None)
        assert req.llm_backend_id is None

    def test_llm_backend_id_coexists_with_other_fields(self):
        """llm_backend_id funktioniert zusammen mit allen anderen Feldern."""
        from kiara.web.api_chat import QueryRequest

        req = QueryRequest(
            question="Frage",
            mode="unchained",
            k=10,
            llm_backend_id=7,
            filter_dir="/docs",
            embedding_model="nomic-embed-text",
        )
        assert req.llm_backend_id == 7
        assert req.mode == "unchained"
        assert req.k == 10

    def test_llm_backend_id_type_annotation(self):
        """llm_backend_id hat den Typ int | None."""
        from kiara.web.api_chat import QueryRequest

        field_info = QueryRequest.model_fields["llm_backend_id"]
        # Pydantic setzt default auf None
        assert field_info.default is None


# ===========================================================================
# /api/chat/filters — llm_backends in Response
# ===========================================================================


class TestFiltersEndpointBackends:
    """Tests fuer das llm_backends-Feld im /api/chat/filters-Endpoint."""

    @pytest.fixture
    def config(self):
        return {
            "datasource": {"file_types": [".txt", ".pdf"]},
            "ollama": {"embedding_model": "nomic-embed-text"},
        }

    @pytest.mark.asyncio
    async def test_filters_response_contains_llm_backends(self, config):
        """Response enthaelt das Feld llm_backends."""
        from kiara.web.api_chat import get_chat_filters

        backends_data = [
            {
                "id": 1,
                "name": "Local Qwen",
                "model": "qwen3:8b",
                "provider": "ollama",
                "extra_config": "{}",
            },
            {
                "id": 2,
                "name": "Claude",
                "model": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "extra_config": "{}",
            },
        ]

        with (
            patch("kiara.web.api_chat.ds_repo") as mock_ds_repo,
            patch("kiara.web.api_chat.embedding_repo") as mock_emb_repo,
            patch("kiara.web.api_chat.load_config", return_value=config),
            patch(
                "kiara.backend_selector.get_all_active",
                side_effect=lambda cat: backends_data if cat == "llm" else [],
            ),
            patch("kiara.config.get_embedding_model", return_value="nomic-embed-text"),
        ):
            mock_ds_repo.list_filtered.return_value = ([], 0)
            mock_emb_repo.get_active_models.return_value = []

            user = {"id": 1, "is_admin": True}
            result = await get_chat_filters(user=user)

        assert "llm_backends" in result
        assert isinstance(result["llm_backends"], list)

    @pytest.mark.asyncio
    async def test_filters_llm_backends_structure(self, config):
        """Jedes Backend-Dict hat id, name, model, provider, num_ctx."""
        from kiara.web.api_chat import get_chat_filters

        backends_data = [
            {
                "id": 5,
                "name": "Qwen Local",
                "model": "qwen3:8b",
                "provider": "ollama",
                "extra_config": '{"num_ctx": 32768}',
            },
        ]

        with (
            patch("kiara.web.api_chat.ds_repo") as mock_ds_repo,
            patch("kiara.web.api_chat.embedding_repo") as mock_emb_repo,
            patch("kiara.web.api_chat.load_config", return_value=config),
            patch(
                "kiara.backend_selector.get_all_active",
                side_effect=lambda cat: backends_data if cat == "llm" else [],
            ),
            patch("kiara.config.get_embedding_model", return_value="nomic-embed-text"),
        ):
            mock_ds_repo.list_filtered.return_value = ([], 0)
            mock_emb_repo.get_active_models.return_value = []

            user = {"id": 1, "is_admin": True}
            result = await get_chat_filters(user=user)

        backends = result["llm_backends"]
        assert len(backends) == 1
        b = backends[0]
        assert b["id"] == 5
        assert b["name"] == "Qwen Local"
        assert b["model"] == "qwen3:8b"
        assert b["provider"] == "ollama"
        assert b["num_ctx"] == 32768

    @pytest.mark.asyncio
    async def test_filters_llm_backends_num_ctx_none_when_missing(self, config):
        """num_ctx ist None wenn nicht in extra_config gesetzt."""
        from kiara.web.api_chat import get_chat_filters

        backends_data = [
            {
                "id": 3,
                "name": "Cloud",
                "model": "gpt-4o",
                "provider": "openai",
                "extra_config": "{}",
            },
        ]

        with (
            patch("kiara.web.api_chat.ds_repo") as mock_ds_repo,
            patch("kiara.web.api_chat.embedding_repo") as mock_emb_repo,
            patch("kiara.web.api_chat.load_config", return_value=config),
            patch(
                "kiara.backend_selector.get_all_active",
                side_effect=lambda cat: [] if cat == "llm" else backends_data,
            ),
            patch("kiara.config.get_embedding_model", return_value="nomic-embed-text"),
        ):
            mock_ds_repo.list_filtered.return_value = ([], 0)
            mock_emb_repo.get_active_models.return_value = []

            user = {"id": 1, "is_admin": True}
            result = await get_chat_filters(user=user)

        backends = result["llm_backends"]
        assert len(backends) == 1
        assert backends[0]["num_ctx"] is None

    @pytest.mark.asyncio
    async def test_filters_llm_backends_includes_cloud(self, config):
        """llm_backends enthaelt sowohl LLM- als auch Cloud-Backends."""
        from kiara.web.api_chat import get_chat_filters

        llm_backends = [
            {
                "id": 1,
                "name": "Ollama Local",
                "model": "qwen3:8b",
                "provider": "ollama",
                "extra_config": "{}",
            },
        ]
        cloud_backends = [
            {
                "id": 2,
                "name": "Anthropic",
                "model": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "extra_config": "{}",
            },
        ]

        def mock_get_all_active(cat):
            if cat == "llm":
                return llm_backends
            if cat == "cloud":
                return cloud_backends
            return []

        with (
            patch("kiara.web.api_chat.ds_repo") as mock_ds_repo,
            patch("kiara.web.api_chat.embedding_repo") as mock_emb_repo,
            patch("kiara.web.api_chat.load_config", return_value=config),
            patch("kiara.backend_selector.get_all_active", side_effect=mock_get_all_active),
            patch("kiara.config.get_embedding_model", return_value="nomic-embed-text"),
        ):
            mock_ds_repo.list_filtered.return_value = ([], 0)
            mock_emb_repo.get_active_models.return_value = []

            user = {"id": 1, "is_admin": True}
            result = await get_chat_filters(user=user)

        backends = result["llm_backends"]
        assert len(backends) == 2
        providers = {b["provider"] for b in backends}
        assert "ollama" in providers
        assert "anthropic" in providers

    @pytest.mark.asyncio
    async def test_filters_empty_on_exception(self):
        """Bei Exception: leere llm_backends-Liste im Fallback."""
        from kiara.web.api_chat import get_chat_filters

        with patch("kiara.web.api_chat.ds_repo") as mock_ds_repo:
            mock_ds_repo.list_filtered.side_effect = Exception("DB kaputt")
            user = {"id": 1, "is_admin": True}
            result = await get_chat_filters(user=user)

        assert result["llm_backends"] == []


# ===========================================================================
# Chain-Durchreichung: llm_backend_id an _create_chain_backend (Standard-Modus)
# ===========================================================================


class TestBackendIdPassthroughStandard:
    """Prueft, dass llm_backend_id an _create_chain_backend weitergereicht wird."""

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
    async def test_standard_passes_backend_id(self, config):
        """_stream_standard reicht llm_backend_id an _create_chain_backend weiter."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["OK"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_expansions = []

        with (
            patch(
                "kiara.generation.cloud_routing._create_chain_backend", return_value=mock_backend
            ) as mock_create,
            patch("kiara.generation.chain.RAGChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=[]),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard", llm_backend_id=99)
            async for _ in _stream_standard(config, data, 1, user):
                pass

            mock_create.assert_called_once()
            assert mock_create.call_args.kwargs.get("llm_backend_id") == 99

    @pytest.mark.asyncio
    async def test_standard_passes_none_backend_id(self, config):
        """_stream_standard reicht None als llm_backend_id an _create_chain_backend weiter."""
        from kiara.web.api_chat import QueryRequest, _stream_standard

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["OK"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_expansions = []

        with (
            patch(
                "kiara.generation.cloud_routing._create_chain_backend", return_value=mock_backend
            ) as mock_create,
            patch("kiara.generation.chain.RAGChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=[]),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test?", mode="standard")
            async for _ in _stream_standard(config, data, 1, user):
                pass

            mock_create.assert_called_once()
            assert mock_create.call_args.kwargs.get("llm_backend_id") is None


# ===========================================================================
# Chain-Durchreichung: llm_backend_id an create_backend (Unchained-Modus)
# ===========================================================================


class TestBackendIdPassthroughUnchained:
    """Prueft, dass llm_backend_id an create_backend weitergereicht wird."""

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

    @pytest.mark.asyncio
    async def test_unchained_passes_backend_id(self, config):
        """_stream_unchained reicht llm_backend_id an create_backend weiter."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Antwort"])
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_usage = {}
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None

        with (
            patch(
                "kiara.generation.backends.create_backend", return_value=mock_backend
            ) as mock_create,
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Hallo", mode="unchained", llm_backend_id=5)
            async for _ in _stream_unchained(config, data, 1, user):
                pass

            mock_create.assert_called_once_with(config, 5)

    @pytest.mark.asyncio
    async def test_unchained_passes_none_backend_id(self, config):
        """_stream_unchained reicht None als backend_id an create_backend weiter."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_backend = MagicMock()
        mock_backend.provider = "ollama"

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Antwort"])
        mock_chain.route_used = "local"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_usage = {}
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None

        with (
            patch(
                "kiara.generation.backends.create_backend", return_value=mock_backend
            ) as mock_create,
            patch("kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend),
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Hallo", mode="unchained")
            async for _ in _stream_unchained(config, data, 1, user):
                pass

            mock_create.assert_called_once_with(config, None)


# ===========================================================================
# Chain-Durchreichung: llm_backend_id an AgentChain
# ===========================================================================


class TestBackendIdPassthroughAgent:
    """Prueft, dass _stream_agent den Shared-Resolver _resolve_backend_dict nutzt."""

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
    async def test_agent_passes_adapter_from_resolved_backend(self, config):
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

        resolved_backend = {
            "id": 12,
            "provider": "ollama",
            "url": "http://gpu-server.local:11434",
            "model": "qwen3:8b",
        }
        mock_adapter = MagicMock()

        with (
            patch("kiara.generation.agent_chain.AgentChain", return_value=mock_chain) as MockChain,
            patch(
                "kiara.generation.tool_adapter.create_tool_adapter", return_value=mock_adapter
            ) as MockFactory,
            patch(
                "kiara.generation.cloud_routing._resolve_backend_dict",
                return_value=resolved_backend,
            ),
            patch("kiara.web.api_chat._build_agent_tool_registry", return_value=MagicMock()),
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=[]),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._create_agent_anonymizer", return_value=None),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent", llm_backend_id=12)
            async for _ in _stream_agent(config, data, 1, user):
                pass

        MockFactory.assert_called_once()
        assert MockFactory.call_args[0][0]["id"] == 12
        assert MockChain.call_args.kwargs.get("adapter") is mock_adapter

    @pytest.mark.asyncio
    async def test_agent_uses_shared_resolver_for_default_backend(self, config):
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

        default_backend = {
            "id": 3,
            "provider": "ollama",
            "url": "http://localhost:11434",
            "model": "test-model",
        }
        mock_adapter = MagicMock()

        with (
            patch("kiara.generation.agent_chain.AgentChain", return_value=mock_chain) as MockChain,
            patch("kiara.generation.tool_adapter.create_tool_adapter", return_value=mock_adapter),
            patch(
                "kiara.generation.cloud_routing._resolve_backend_dict", return_value=default_backend
            ),
            patch("kiara.web.api_chat._build_agent_tool_registry", return_value=MagicMock()),
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=[]),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._create_agent_anonymizer", return_value=None),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Test", mode="agent")
            async for _ in _stream_agent(config, data, 1, user):
                pass

        MockChain.assert_called_once()

    def test_prepare_agent_context_passes_allowed_ds_to_resolver(self, config):
        from kiara.web.api_chat import QueryRequest, _prepare_agent_context

        resolved_backend = {
            "id": 12,
            "provider": "ollama",
            "url": "http://localhost:11434",
            "model": "qwen3:8b",
        }

        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=[4, 8]),
            patch(
                "kiara.generation.cloud_routing._resolve_backend_dict",
                return_value=resolved_backend,
            ) as mock_resolve,
            patch(
                "kiara.web.api_chat._build_agent_tool_registry", return_value=MagicMock()
            ) as mock_registry,
            patch("kiara.generation.tool_adapter.create_tool_adapter", return_value=MagicMock()),
            patch("kiara.generation.agent_chain.AgentChain", return_value=MagicMock()),
        ):
            user = {"id": 1, "is_admin": False, "datasource_ids": [4, 8]}
            data = QueryRequest(question="Test", mode="agent")
            _prepare_agent_context(config, data, user, None, None)

        assert mock_resolve.call_args.kwargs["filter_datasource_ids"] == [4, 8]
        assert mock_registry.call_args.args[2] == [4, 8]


# ===========================================================================
# _stream_unchained — trusted-Flag Logik
# ===========================================================================


class TestUnchainedTrustedLogic:
    """Prueft trusted-Flag Logik in _stream_unchained."""

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

    @pytest.mark.asyncio
    async def test_unchained_trusted_cloud_no_wrapping(self, config):
        """trusted=True Cloud-Backend wird NICHT mit Anonymisierung gewrappt."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_backend = MagicMock()
        mock_backend.provider = "anthropic"

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Antwort"])
        mock_chain.route_used = "cloud"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_usage = {}
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None

        with (
            patch(
                "kiara.generation.backends.create_backend", return_value=mock_backend
            ) as mock_create,
            patch(
                "kiara.backend_selector.get_backend_by_id",
                return_value={"id": 5, "provider": "anthropic", "trusted": True},
            ),
            patch(
                "kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend
            ) as mock_wrap,
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._emit_cloud_audit"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Hallo", mode="unchained", llm_backend_id=5)
            async for _ in _stream_unchained(config, data, 1, user):
                pass

            mock_wrap.assert_not_called()

    @pytest.mark.asyncio
    async def test_unchained_untrusted_cloud_wraps(self, config):
        """trusted=False Cloud-Backend wird mit Anonymisierung gewrappt."""
        from kiara.web.api_chat import QueryRequest, _stream_unchained

        mock_backend = MagicMock()
        mock_backend.provider = "anthropic"

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = iter(["Antwort"])
        mock_chain.route_used = "cloud"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_usage = {}
        mock_chain.last_anon_question = None
        mock_chain.last_anon_answer = None

        with (
            patch(
                "kiara.generation.backends.create_backend", return_value=mock_backend
            ) as mock_create,
            patch(
                "kiara.backend_selector.get_backend_by_id",
                return_value={"id": 5, "provider": "anthropic", "trusted": False},
            ),
            patch(
                "kiara.generation.backends.wrap_with_anonymization", return_value=mock_backend
            ) as mock_wrap,
            patch("kiara.generation.unchained_chain.UnchainedChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._emit_cloud_audit"),
        ):
            user = {"id": 1, "is_admin": True}
            data = QueryRequest(question="Hallo", mode="unchained", llm_backend_id=5)
            async for _ in _stream_unchained(config, data, 1, user):
                pass

            mock_wrap.assert_called_once()


# ===========================================================================
# _create_agent_anonymizer — Restriction-Level Logik
# ===========================================================================


class TestCreateAgentAnonymizer:
    """Prueft _create_agent_anonymizer() mit restriction_levels."""

    def test_trusted_returns_none(self):
        """trusted=True Backend gibt None zurueck (keine Anonymisierung)."""
        from kiara.web.api_chat import _create_agent_anonymizer

        result = _create_agent_anonymizer(
            {},
            {"provider": "anthropic", "trusted": True},
            restriction_levels=["anonymize"],
        )
        assert result is None

    def test_public_returns_none(self):
        """Nur public restriction_levels gibt None zurueck."""
        from kiara.web.api_chat import _create_agent_anonymizer

        result = _create_agent_anonymizer(
            {},
            {"provider": "anthropic", "trusted": False},
            restriction_levels=["public", "public"],
        )
        assert result is None

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources")
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_anonymize_returns_pseudonymizer(self, MockPseudo, mock_build):
        """anonymize restriction_levels erstellt Pseudonymizer."""
        from kiara.web.api_chat import _create_agent_anonymizer

        mock_pseudo_instance = MagicMock()
        MockPseudo.return_value = mock_pseudo_instance
        mock_build.return_value = MagicMock()

        result = _create_agent_anonymizer(
            {},
            {"provider": "anthropic", "trusted": False},
            restriction_levels=["anonymize"],
        )
        assert result is mock_pseudo_instance

    def test_anonymize_init_error_raises_runtime(self):
        """Init-Fehler bei anonymize -> RuntimeError (nicht fail-open)."""
        from kiara.web.api_chat import _create_agent_anonymizer

        with patch(
            "kiara.anonymization.stammdaten.build_automaton_from_sources",
            side_effect=Exception("LDAP down"),
        ):
            with pytest.raises(RuntimeError, match="Agent-Anonymisierung"):
                _create_agent_anonymizer(
                    {},
                    {"provider": "anthropic", "trusted": False},
                    restriction_levels=["anonymize"],
                )

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources")
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_empty_list_falls_back_to_global_default(self, MockPseudo, mock_build):
        """Leere restriction_levels-Liste wird wie None behandelt (Issue #1960).

        Edge-Case: _build_restriction_filter kann eine leere Liste produzieren
        (z.B. Admin-Exclude-Filter). _load_restriction_levels gibt dann []
        zurueck — vor dem Fix wurde der Fallback auf anonymization.enabled
        uebersprungen und Cloud-PII liefen ungeschuetzt durch.
        """
        from kiara.web.api_chat import _create_agent_anonymizer

        mock_pseudo_instance = MagicMock()
        MockPseudo.return_value = mock_pseudo_instance
        mock_build.return_value = MagicMock()

        result = _create_agent_anonymizer(
            {"anonymization": {"enabled": True}},
            {"provider": "anthropic", "trusted": False},
            restriction_levels=[],
        )
        assert result is mock_pseudo_instance
