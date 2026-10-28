"""Tests fuer die Reranker-Integration und Kontext-Overflow in der RAG-Chain."""

from unittest.mock import MagicMock, patch

from kiara.generation.backends.base import LLMBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "Kontext:\n{context}\n\nFrage: {question}"


def _make_results(n):
    """Erzeugt n Dummy-Retrieval-Ergebnisse."""
    return [
        {
            "text": f"Dokument {i}",
            "metadata": {"source": f"doc{i}.txt"},
            "score": round(1.0 - i * 0.05, 2),
        }
        for i in range(n)
    ]


def _base_config(**overrides):
    """Minimale Config fuer RAGChain mit optionalen Overrides."""
    cfg = {
        "datasource": {"path": "/tmp/test"},
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
            "prompt_template": PROMPT_TEMPLATE,
        },
        "vectordb": {
            "backend": "qdrant",
            "path": "/tmp/vectordb",
            "collection": "test",
        },
        "retrieval": {
            "reranking": False,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_k": 5,
            "rerank_fetch_multiplier": 3,
        },
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _make_mock_backend(**overrides):
    """Erzeugt ein Mock-LLMBackend mit sinnvollen Defaults."""
    backend = MagicMock(spec=LLMBackend)
    backend.provider = "ollama"
    backend.model = "test-model"
    backend.last_usage = {}
    backend.generate.return_value = "Antwort"
    backend.generate_stream.return_value = iter(["Antwort"])
    for key, val in overrides.items():
        setattr(backend, key, val)
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChainRerankerIntegration:
    """Prueft die Integration von CrossEncoderReranker in RAGChain."""

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_reranker_called_when_enabled(self, MockReranker, MockRetriever):
        """Bei aktivem Reranking wird rerank() auf den Ergebnissen aufgerufen."""
        from kiara.generation.chain import RAGChain

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = True
        mock_reranker.rerank.return_value = _make_results(5)

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(15)

        mock_backend = _make_mock_backend()
        mock_backend.generate.return_value = "Test-Antwort"

        config = _base_config(retrieval={"reranking": True, "rerank_fetch_multiplier": 3})
        chain = RAGChain(config, backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE)

        result = chain.query("Was ist Kubernetes?", k=5)

        # Reranker muss mit query, results und top_k aufgerufen worden sein
        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        assert call_args[0][0] == "Was ist Kubernetes?"
        assert call_args[1]["top_k"] == 5 or call_args[0][2] == 5

        assert result["answer"] == "Test-Antwort"

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_reranker_not_called_when_disabled(self, MockReranker, MockRetriever):
        """Bei deaktiviertem Reranking wird rerank() trotzdem aufgerufen,
        gibt aber results[:k] unveraendert zurueck (CrossEncoderReranker-Logik)."""
        from kiara.generation.chain import RAGChain

        results_5 = _make_results(5)

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = results_5

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = results_5

        mock_backend = _make_mock_backend()

        config = _base_config()  # reranking=False per default
        chain = RAGChain(config, backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE)

        chain.query("Test?", k=5)

        # Bei deaktiviertem Reranking: Retriever bekommt k (nicht k*multiplier)
        retrieve_call = mock_retriever.retrieve.call_args
        assert retrieve_call[1]["k"] == 5

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_fetch_multiplier(self, MockReranker, MockRetriever):
        """Bei aktivem Reranking wird k * multiplier an den Retriever uebergeben."""
        from kiara.generation.chain import RAGChain

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = True
        mock_reranker.rerank.return_value = _make_results(5)

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(15)

        mock_backend = _make_mock_backend()

        config = _base_config(
            retrieval={
                "reranking": True,
                "rerank_fetch_multiplier": 4,
            }
        )
        chain = RAGChain(config, backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE)

        chain.query("Test?", k=5)

        # Retriever muss k * multiplier = 5 * 4 = 20 bekommen
        retrieve_call = mock_retriever.retrieve.call_args
        assert retrieve_call[1]["k"] == 20

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_results_pass_through_to_context(self, MockReranker, MockRetriever):
        """Die reranked Results werden korrekt an _format_context weitergegeben."""
        from kiara.generation.chain import RAGChain

        # Reranker gibt genau 2 spezifische Ergebnisse zurueck
        reranked = [
            {"text": "Bestes Ergebnis", "metadata": {"source": "best.txt"}, "score": 0.99},
            {"text": "Zweitbestes", "metadata": {"source": "second.txt"}, "score": 0.95},
        ]

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = True
        mock_reranker.rerank.return_value = reranked

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = _make_results(15)

        mock_backend = _make_mock_backend()

        config = _base_config(retrieval={"reranking": True})
        chain = RAGChain(config, backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE)

        result = chain.query("Test?", k=2)

        # Der LLM-Prompt muss die reranked Ergebnisse enthalten
        # generate() wird mit (system_prompt, user_message) aufgerufen
        user_message = mock_backend.generate.call_args[0][1]
        assert "Bestes Ergebnis" in user_message
        assert "Zweitbestes" in user_message

        # Sources muessen aus den reranked Results stammen
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source"] == "best.txt"
        assert result["sources"][1]["source"] == "second.txt"

        # context_chunks muessen die reranked Results sein; die Chain annotiert
        # zusaetzlich final_context_rank (1-basiert) auf Kopien der Chunks.
        assert [
            {k: v for k, v in chunk.items() if k != "final_context_rank"}
            for chunk in result["context_chunks"]
        ] == reranked
        assert [c["final_context_rank"] for c in result["context_chunks"]] == [1, 2]


# ---------------------------------------------------------------------------
# Kontext-Overflow-Pruefung
# ---------------------------------------------------------------------------


class TestContextOverflow:
    """Prueft _check_context_overflow() und die context_warning Property."""

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_no_warning_when_within_limit(self, MockReranker, MockRetriever):
        """Kein Warning wenn der Prompt ins Kontextfenster passt."""
        from kiara.generation.chain import RAGChain

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = _make_results(3)
        MockRetriever.return_value.retrieve.return_value = _make_results(3)

        mock_backend = _make_mock_backend()
        mock_backend.generate_stream.return_value = iter(["OK"])

        chain = RAGChain(
            _base_config(), backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE
        )
        chain.query_stream("Kurze Frage?", k=3)

        assert chain.context_warning is None

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_warning_when_exceeding_limit(self, MockReranker, MockRetriever):
        """Warning wird gesetzt wenn geschaetzte Tokens > num_ctx."""
        from kiara.generation.chain import RAGChain

        # Grosse Chunks erzeugen: 50 * 2000 Zeichen = 100.000 Zeichen
        big_results = [
            {"text": "x" * 2000, "metadata": {"source": f"big{i}.txt"}, "score": 0.9}
            for i in range(50)
        ]

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = big_results
        MockRetriever.return_value.retrieve.return_value = big_results

        mock_backend = _make_mock_backend()
        mock_backend.generate_stream.return_value = iter(["OK"])

        chain = RAGChain(
            _base_config(), backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE
        )
        chain.query_stream("Frage?", k=50)

        warning = chain.context_warning
        assert warning is not None
        assert warning["estimated_tokens"] > warning["num_ctx"]
        assert warning["overflow_pct"] > 0
        assert warning["num_ctx"] == 8192

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.retrieval.reranker.CrossEncoderReranker")
    def test_warning_cleared_on_small_followup(self, MockReranker, MockRetriever):
        """Warning wird zurueckgesetzt bei nachfolgender Anfrage die passt."""
        from kiara.generation.chain import RAGChain

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False

        mock_backend = _make_mock_backend()
        mock_backend.generate_stream.return_value = iter(["OK"])

        chain = RAGChain(
            _base_config(), backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE
        )

        # Erste Anfrage: gross -> Warning
        big_results = [
            {"text": "x" * 2000, "metadata": {"source": f"big{i}.txt"}, "score": 0.9}
            for i in range(50)
        ]
        mock_reranker.rerank.return_value = big_results
        MockRetriever.return_value.retrieve.return_value = big_results
        chain.query_stream("Frage 1?", k=50)
        assert chain.context_warning is not None

        # Zweite Anfrage: klein -> Warning weg
        small_results = _make_results(3)
        mock_reranker.rerank.return_value = small_results
        MockRetriever.return_value.retrieve.return_value = small_results
        mock_backend.generate_stream.return_value = iter(["OK"])
        chain.query_stream("Frage 2?", k=3)
        assert chain.context_warning is None

    def test_check_context_overflow_direct(self):
        """Direkte Pruefung von _check_context_overflow() mit kontrolliertem Input."""
        from kiara.generation.chain import RAGChain
        from kiara.shared.constants import CHARS_PER_TOKEN

        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.chain.HybridRetriever"),
            patch("kiara.retrieval.reranker.CrossEncoderReranker"),
        ):
            chain = RAGChain(
                _base_config(), backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE
            )

        # Exakt an der Grenze: 8192 Tokens * 3.5 chars = 28672 Zeichen
        num_ctx = 8192
        prompt_at_limit = "a" * int(num_ctx * CHARS_PER_TOKEN)
        chain._check_context_overflow(prompt_at_limit, num_ctx)
        assert chain.context_warning is None

        # Knapp darueber
        prompt_over = "a" * int(num_ctx * CHARS_PER_TOKEN + 100)
        chain._check_context_overflow(prompt_over, num_ctx)
        assert chain.context_warning is not None
        assert chain.context_warning["overflow_pct"] >= 0

    def test_overflow_pct_calculation(self):
        """overflow_pct wird korrekt berechnet."""
        from kiara.generation.chain import RAGChain
        from kiara.shared.constants import CHARS_PER_TOKEN

        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.chain.HybridRetriever"),
            patch("kiara.retrieval.reranker.CrossEncoderReranker"),
        ):
            chain = RAGChain(
                _base_config(), backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE
            )

        # 2x num_ctx -> 100% overflow
        num_ctx = 1000
        prompt_2x = "a" * int(num_ctx * CHARS_PER_TOKEN * 2)
        chain._check_context_overflow(prompt_2x, num_ctx)
        assert chain.context_warning is not None
        assert chain.context_warning["overflow_pct"] == 100

    def test_no_results_no_warning(self):
        """Bei leeren Ergebnissen wird kein Warning gesetzt."""
        from kiara.generation.chain import RAGChain

        mock_backend = _make_mock_backend()

        with (
            patch("kiara.generation.chain.HybridRetriever") as MockRetriever,
            patch("kiara.retrieval.reranker.CrossEncoderReranker") as MockReranker,
        ):
            mock_reranker = MockReranker.return_value
            mock_reranker.enabled = False
            mock_reranker.rerank.return_value = []
            MockRetriever.return_value.retrieve.return_value = []

            chain = RAGChain(
                _base_config(), backend=mock_backend, prompt_template_override=PROMPT_TEMPLATE
            )
            gen, sources = chain.query_stream("Frage?", k=5)
            # Generator konsumieren
            list(gen)

        assert chain.context_warning is None
