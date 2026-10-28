"""Tests fuer die Backend-Injection in RAGChain, UnchainedChain und RewriterChain.

Sprint 2: Chains bekommen LLMBackend per Konstruktor injiziert.
"""

from unittest.mock import patch

import pytest

from kiara.generation.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# MockBackend — wiederverwendbarer Test-Mock
# ---------------------------------------------------------------------------


class MockBackend(LLMBackend):
    """Test-Mock fuer LLMBackend."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "test-model",
        generate_response: str = "Mock-Antwort",
        stream_chunks: list[str] | None = None,
    ):
        self._provider = provider
        self._model = model
        self._generate_response = generate_response
        self._stream_chunks = stream_chunks or ["Mock", "-", "Antwort"]
        self._last_usage = {}
        self._call_count = 0
        self._last_system_prompt = ""
        self._last_user_message = ""

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_usage(self) -> dict:
        return self._last_usage

    def generate(self, system_prompt, user_message, **kwargs):
        self._call_count += 1
        self._last_system_prompt = system_prompt
        self._last_user_message = user_message
        self._last_usage = {
            "prompt_tokens": 10,
            "response_tokens": 5,
            "model": self._model,
            "num_ctx": 0,
            "total_duration_ms": 100,
        }
        return self._generate_response

    def generate_stream(self, system_prompt, user_message, **kwargs):
        self._call_count += 1
        self._last_system_prompt = system_prompt
        self._last_user_message = user_message
        self._last_usage = {
            "prompt_tokens": 10,
            "response_tokens": 5,
            "model": self._model,
            "num_ctx": 0,
            "total_duration_ms": 100,
        }
        for chunk in self._stream_chunks:
            yield chunk


class MockAnonymizingBackend(LLMBackend):
    """Mock fuer AnonymizingBackend mit Audit-Properties."""

    def __init__(self, inner_provider="anthropic"):
        self._provider = inner_provider
        self._model = "anon-model"
        self._last_usage = {}
        self.last_anon_question = ""
        self.last_anon_answer = ""
        self.last_entities_anonymized = 3

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_usage(self) -> dict:
        return self._last_usage

    @property
    def anonymizer_stats(self) -> dict:
        return {"total_entities": 3, "entity_types": {"Person": 2, "Firma": 1}}

    def generate(self, system_prompt, user_message, **kwargs):
        self.last_anon_question = f"[anon]{user_message}"
        self._last_usage = {"prompt_tokens": 20, "response_tokens": 10, "model": self._model}
        answer = "Anonymisierte Antwort"
        self.last_anon_answer = answer
        return answer

    def generate_stream(self, system_prompt, user_message, **kwargs):
        self.last_anon_question = f"[anon]{user_message}"
        self._last_usage = {"prompt_tokens": 20, "response_tokens": 10, "model": self._model}
        chunks = ["Anon", "-", "Antwort"]
        self.last_anon_answer = "".join(chunks)
        for c in chunks:
            yield c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_config(**overrides):
    cfg = {
        "datasource": {"path": "/tmp/test"},
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
            "prompt_template": "Kontext:\n{context}\n\nFrage: {question}",
        },
        "vectordb": {"backend": "qdrant", "path": "/tmp/vectordb", "collection": "test"},
        "retrieval": {"reranking": False},
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _make_results(n):
    return [
        {
            "text": f"Dokument {i}",
            "metadata": {"source": f"doc{i}.txt", "chunk_index": i},
            "score": round(1.0 - i * 0.05, 2),
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Suite 1: RAGChain mit Backend-Injection
# ---------------------------------------------------------------------------


class TestRAGChainBackendInjection:
    """RAGChain akzeptiert LLMBackend per Konstruktor."""

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_ollama_backend_query_stream(self, MockReranker, MockRetriever):
        """RAGChain mit OllamaBackend-Mock: generate_stream wird aufgerufen."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(3)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(3)

        backend = MockBackend(provider="ollama", stream_chunks=["Hallo", " Welt"])
        config = _base_config()
        chain = RAGChain(
            config,
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        gen, sources = chain.query_stream("Test-Frage", k=3)
        result = "".join(gen)

        assert result == "Hallo Welt"
        assert len(sources) == 3
        assert backend._call_count == 1

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_cloud_backend_query_stream(self, MockReranker, MockRetriever):
        """RAGChain mit Cloud-Backend-Mock: generate_stream wird aufgerufen."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(3)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(3)

        backend = MockBackend(provider="anthropic", stream_chunks=["Cloud", "-", "Antwort"])
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        gen, sources = chain.query_stream("Test-Frage", k=3)
        result = "".join(gen)

        assert result == "Cloud-Antwort"

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_anonymizing_backend(self, MockReranker, MockRetriever):
        """RAGChain mit AnonymizingBackend-Mock: Audit-Properties korrekt."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(2)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(2)

        backend = MockAnonymizingBackend()
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        gen, sources = chain.query_stream("Frage mit PII", k=2)
        "".join(gen)

        assert chain.last_anon_question != ""
        assert chain.last_entities_anonymized == 3

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_route_used_local(self, MockReranker, MockRetriever):
        """route_used == 'local' bei OllamaBackend."""
        from kiara.generation.chain import RAGChain

        backend = MockBackend(provider="ollama")
        chain = RAGChain(
            _base_config(), backend=backend, prompt_template_override="{context}{question}"
        )
        assert chain.route_used == "local"

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_route_used_cloud(self, MockReranker, MockRetriever):
        """route_used == 'cloud' bei AnthropicBackend."""
        from kiara.generation.chain import RAGChain

        backend = MockBackend(provider="anthropic")
        chain = RAGChain(
            _base_config(), backend=backend, prompt_template_override="{context}{question}"
        )
        assert chain.route_used == "cloud"

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_route_used_anonymizing(self, MockReranker, MockRetriever):
        """route_used == 'cloud' bei AnonymizingBackend (anthropic inner)."""
        from kiara.generation.chain import RAGChain

        backend = MockAnonymizingBackend(inner_provider="anthropic")
        chain = RAGChain(
            _base_config(), backend=backend, prompt_template_override="{context}{question}"
        )
        assert chain.route_used == "cloud"

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_last_anon_question(self, MockReranker, MockRetriever):
        """last_anon_question delegiert an AnonymizingBackend."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(1)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(1)

        backend = MockAnonymizingBackend()
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )
        gen, _ = chain.query_stream("Test", k=1)
        "".join(gen)
        assert chain.last_anon_question != ""

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_last_usage_delegation(self, MockReranker, MockRetriever):
        """last_usage delegiert an Backend."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(1)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(1)

        backend = MockBackend()
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )
        gen, _ = chain.query_stream("Test", k=1)
        "".join(gen)
        assert chain.last_usage.get("prompt_tokens") == 10

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_build_prompt(self, MockReranker, MockRetriever):
        """_build_prompt() baut date_prefix + history + template + Prompt-Defense."""
        from kiara.generation.chain import RAGChain

        backend = MockBackend()
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
            system_prompt="Du bist ein Assistent.",
        )
        with patch("kiara.shared.prompt_defense_audit.log_defense_event"):
            sys_p, user_msg = chain._build_prompt("ctx-text", "meine Frage", "Verlauf hier")
        # System-Prompt wird jetzt gehaertet (Praeambel + Original)
        assert "SICHERHEITSHINWEIS" in sys_p
        assert "Du bist ein Assistent." in sys_p
        assert "ctx-text" in user_msg
        assert "meine Frage" in user_msg
        # History wird in ¤-Tags gewickelt
        assert "Verlauf hier" in user_msg

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_query_non_streaming(self, MockReranker, MockRetriever):
        """query() nutzt backend.generate()."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(2)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(2)

        backend = MockBackend(generate_response="Sync-Antwort")
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )
        result = chain.query("Frage?", k=2)
        assert result["answer"] == "Sync-Antwort"
        assert len(result["sources"]) == 2

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_context_override(self, MockReranker, MockRetriever):
        """query_stream mit context_override ueberspringt Retrieval."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        backend = MockBackend(stream_chunks=["Override-Antwort"])
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        override = _make_results(2)
        mock_retriever.load_context_chunks.return_value = override
        gen, sources = chain.query_stream("Test", context_override=override)
        result = "".join(gen)

        assert result == "Override-Antwort"
        assert len(sources) == 2
        mock_retriever.retrieve.assert_not_called()
        mock_retriever.load_context_chunks.assert_called_once_with(override, None)

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_context_override_empty_list(self, MockReranker, MockRetriever):
        """context_override=[] wird als leerer Override behandelt (keine Dokumente)."""
        from kiara.generation.chain import RAGChain

        backend = MockBackend()
        MockRetriever.return_value.load_context_chunks.return_value = []
        chain = RAGChain(
            _base_config(), backend=backend, prompt_template_override="{context}{question}"
        )
        gen, sources = chain.query_stream("Test", context_override=[])
        result = "".join(gen)

        assert "Keine relevanten Dokumente" in result
        assert sources == []

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_ragchain_empty_results(self, MockReranker, MockRetriever):
        """Leere Retrieval-Ergebnisse -> 'Keine relevanten Dokumente'."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = []
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = []

        backend = MockBackend()
        chain = RAGChain(
            _base_config(), backend=backend, prompt_template_override="{context}{question}"
        )
        gen, sources = chain.query_stream("Nichts da?", k=5)
        result = "".join(gen)

        assert "Keine relevanten Dokumente" in result
        assert sources == []


# ---------------------------------------------------------------------------
# Suite 2: UnchainedChain mit Backend-Injection
# ---------------------------------------------------------------------------


class TestUnchainedChainBackendInjection:
    """UnchainedChain akzeptiert LLMBackend per Konstruktor."""

    def test_unchained_ollama_stream(self):
        """UnchainedChain mit OllamaBackend-Mock: Streaming korrekt."""
        from kiara.generation.unchained_chain import UnchainedChain

        backend = MockBackend(provider="ollama", stream_chunks=["Hallo", " Welt"])
        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="Test-Prompt")

        result = "".join(chain.query_stream("Frage?"))
        assert result == "Hallo Welt"

    def test_unchained_cloud_stream(self):
        """UnchainedChain mit Cloud-Backend-Mock: Streaming korrekt."""
        from kiara.generation.unchained_chain import UnchainedChain

        backend = MockBackend(provider="anthropic", stream_chunks=["Cloud", " ", "OK"])
        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="System")

        result = "".join(chain.query_stream("Cloud-Frage"))
        assert result == "Cloud OK"

    def test_unchained_anonymizing_stream(self):
        """Anonymisierung transparent ueber Backend."""
        from kiara.generation.unchained_chain import UnchainedChain

        backend = MockAnonymizingBackend()
        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="")

        result = "".join(chain.query_stream("PII-Frage"))
        assert result == "Anon-Antwort"
        assert chain.last_entities_anonymized == 3

    def test_unchained_route_used(self):
        """route_used korrekt abgeleitet."""
        from kiara.generation.unchained_chain import UnchainedChain

        local = UnchainedChain(_base_config(), backend=MockBackend(provider="ollama"))
        assert local.route_used == "local"

        cloud = UnchainedChain(_base_config(), backend=MockBackend(provider="openai"))
        assert cloud.route_used == "cloud"

    def test_unchained_last_prompt(self):
        """last_prompt enthaelt system_prompt + user_message."""
        from kiara.generation.unchained_chain import UnchainedChain

        backend = MockBackend(stream_chunks=["ok"])
        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="System-Text")
        "".join(chain.query_stream("Meine Frage"))

        assert "System-Text" in chain.last_prompt
        assert "Meine Frage" in chain.last_prompt

    def test_unchained_temperature_override(self):
        """temperature-kwarg wird durchgereicht."""
        from kiara.generation.unchained_chain import UnchainedChain

        backend = MockBackend(stream_chunks=["ok"])
        chain = UnchainedChain(_base_config(), backend=backend)
        "".join(chain.query_stream("Test", temperature=0.42))
        # Backend wurde aufgerufen (kein Crash = temperature wird akzeptiert)
        assert backend._call_count == 1


# ---------------------------------------------------------------------------
# Suite 3: RewriterChain mit Backend-Injection
# ---------------------------------------------------------------------------


class TestRewriterChainBackendInjection:
    """RewriterChain akzeptiert rag_backend per Konstruktor."""

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    @patch("kiara.backend_selector.get_all_active")
    @patch("kiara.generation.backends.ollama.OllamaBackend")
    def test_rewriter_ollama_rewrite(self, MockOB, mock_get_all, MockReranker, MockRetriever):
        """Rewrite-Schritt nutzt OllamaBackend.generate()."""
        from kiara.generation.rewriter_chain import RewriterChain

        mock_get_all.return_value = [
            {"provider": "ollama", "url": "http://localhost:11434", "model": "test"}
        ]
        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(3)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(3)

        rag_backend = MockBackend(provider="ollama", stream_chunks=["RAG-Antwort"])
        mock_retriever.load_context_chunks.side_effect = lambda hits, allowed: hits
        rewrite_be = MockBackend(generate_response="optimierter suchbegriff")
        MockOB.return_value = rewrite_be

        chain = RewriterChain(
            _base_config(),
            rag_backend=rag_backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )
        gen, sources = chain.query_stream("Was ist Docker?", k=3)
        result = "".join(gen)

        assert result == "RAG-Antwort"
        assert rewrite_be._call_count == 1  # Rewrite wurde aufgerufen

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    @patch("kiara.backend_selector.get_all_active")
    @patch("kiara.generation.backends.ollama.OllamaBackend")
    def test_rewriter_rag_with_cloud_backend(
        self, MockOB, mock_get_all, MockReranker, MockRetriever
    ):
        """RAG-Antwort nutzt injiziertes Cloud-Backend."""
        from kiara.generation.rewriter_chain import RewriterChain

        mock_get_all.return_value = [
            {"provider": "ollama", "url": "http://localhost:11434", "model": "test"}
        ]
        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(2)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(2)

        rag_backend = MockBackend(provider="anthropic", stream_chunks=["Cloud-RAG"])
        mock_retriever.load_context_chunks.side_effect = lambda hits, allowed: hits
        MockOB.return_value = MockBackend(generate_response="rewrite")

        chain = RewriterChain(
            _base_config(),
            rag_backend=rag_backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )
        gen, _ = chain.query_stream("Frage?", k=2)
        result = "".join(gen)
        assert result == "Cloud-RAG"
        assert chain.route_used == "cloud"

    @patch("kiara.generation.backends.ollama.OllamaBackend")
    @patch("kiara.backend_selector.get_all_active")
    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_rewriter_no_ollama_backend_error(
        self, MockReranker, MockRetriever, mock_get_all, MockOB
    ):
        """BackendUnavailableError wenn kein Ollama verfuegbar."""
        from kiara.backend_selector import BackendUnavailableError
        from kiara.generation.rewriter_chain import RewriterChain

        mock_get_all.return_value = []  # Kein Ollama

        with pytest.raises(BackendUnavailableError):
            RewriterChain(_base_config(), rag_backend=MockBackend())

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    @patch("kiara.backend_selector.get_all_active")
    @patch("kiara.generation.backends.ollama.OllamaBackend")
    def test_rewriter_delegated_properties(self, MockOB, mock_get_all, MockReranker, MockRetriever):
        """route_used, last_anon_* delegieren an RAGChain."""
        from kiara.generation.rewriter_chain import RewriterChain

        mock_get_all.return_value = [
            {"provider": "ollama", "url": "http://localhost:11434", "model": "test"}
        ]

        anon_backend = MockAnonymizingBackend()
        MockOB.return_value = MockBackend(generate_response="rewrite")

        chain = RewriterChain(
            _base_config(),
            rag_backend=anon_backend,
            prompt_template_override="{context}{question}",
        )
        assert chain.route_used == "cloud"
        assert chain.backend is anon_backend


# ---------------------------------------------------------------------------
# Suite 5: Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-End-Tests mit Mock-Backends."""

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_standard_chain_end_to_end(self, MockReranker, MockRetriever):
        """RAGChain(OllamaBackend-Mock) -> query_stream -> Chunks + Sources."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(3)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(3)

        backend = MockBackend(stream_chunks=["Die ", "Antwort ", "lautet."])
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        gen, sources = chain.query_stream("Integration?", k=3)
        answer = "".join(gen)

        assert answer == "Die Antwort lautet."
        assert len(sources) == 3
        assert chain.last_prompt != ""

    def test_unchained_chain_end_to_end(self):
        """UnchainedChain(OllamaBackend-Mock) -> query_stream -> Chunks."""
        from kiara.generation.unchained_chain import UnchainedChain

        backend = MockBackend(stream_chunks=["Unchained ", "Antwort"])
        chain = UnchainedChain(_base_config(), backend=backend, system_prompt="System")

        answer = "".join(chain.query_stream("Frei?"))
        assert answer == "Unchained Antwort"
        assert chain.route_used == "local"

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_cloud_anonymization_end_to_end(self, MockReranker, MockRetriever):
        """RAGChain(AnonymizingBackend(CloudMock)) -> anonymize/deanonymize korrekt."""
        from kiara.generation.chain import RAGChain

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(2)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(2)

        backend = MockAnonymizingBackend()
        chain = RAGChain(
            _base_config(),
            backend=backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        gen, _ = chain.query_stream("PII-Frage", k=2)
        answer = "".join(gen)

        assert answer == "Anon-Antwort"
        assert chain.last_anon_question != ""
        assert chain.last_entities_anonymized == 3
        assert chain.route_used == "cloud"
