"""Tests fuer Tracing-Instrumentierung der Retrieval-Pipeline (Sprint 17.2)."""

import threading
from unittest.mock import MagicMock, PropertyMock, patch

# ---------------------------------------------------------------------------
# RecordingTracer / RecordingSpan — leichtgewichtiges Test-Double
# ---------------------------------------------------------------------------


class RecordingSpan:
    """Span der Attribute und Events aufzeichnet (ohne OTel-Abhaengigkeit)."""

    def __init__(self, name, parent=None):
        self.name = name
        self.attributes = {}
        self.events = []
        self.exceptions = []
        self.status = None
        self.parent = parent
        self._children = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        self.status = status

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes))

    def get_span_context(self):
        ctx = MagicMock()
        ctx.trace_id = 0xABCDEF1234567890ABCDEF1234567890
        ctx.span_id = 0x1234567890ABCDEF
        return ctx

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class RecordingTracer:
    """Tracer der Spans aufzeichnet (thread-safe)."""

    def __init__(self):
        self.spans = []
        self._lock = threading.Lock()

    def start_as_current_span(self, name, **kwargs):
        span = RecordingSpan(name)
        with self._lock:
            self.spans.append(span)
        return span

    def start_span(self, name, **kwargs):
        span = RecordingSpan(name)
        with self._lock:
            self.spans.append(span)
        return span

    def get_span_names(self):
        with self._lock:
            return [s.name for s in self.spans]

    def get_span(self, name):
        with self._lock:
            for s in self.spans:
                if s.name == name:
                    return s
        return None


# ---------------------------------------------------------------------------
# RAGChain Tests
# ---------------------------------------------------------------------------


class TestRAGChainTracing:
    """Tests fuer Tracing in kiara.generation.chain.RAGChain."""

    def _make_chain(self, config=None):
        """Erstellt eine RAGChain mit Mock-Backend."""
        from kiara.generation.chain import RAGChain

        cfg = config or {
            "retrieval": {
                "hybrid_search": "none",
                "reranking": False,
                "query_expansion": False,
                "max_context_chunks": 10,
            },
            "ingestion": {"chunk_size": 1000, "chunk_overlap": 200},
        }
        backend = MagicMock()
        backend.provider = "ollama"
        backend.model = "test-model"
        backend.generate_stream.return_value = iter(["Test ", "Antwort."])
        type(backend).last_usage = PropertyMock(return_value={})

        with patch("kiara.generation.chain.HybridRetriever") as MockRetriever:
            retriever_instance = MagicMock()
            retriever_instance.retrieve.return_value = [
                {"text": "Chunk 1", "metadata": {"source": "test.txt", "chunk_index": 0}},
            ]
            MockRetriever.return_value = retriever_instance

            with patch("kiara.retrieval.reranker.CrossEncoderReranker") as MockReranker:
                reranker = MagicMock()
                reranker.enabled = False
                reranker.rerank.side_effect = lambda q, r, top_k=None: r
                MockReranker.return_value = reranker

                chain = RAGChain(
                    cfg,
                    backend=backend,
                    prompt_template_override="Beantworte: {question}\nKontext: {context}",
                )
        return chain

    def test_rag_chain_query_stream_persists_last_trace_id(self):
        """query_stream() setzt last_trace_id auf die trace_id des rag_query-Spans."""
        recorder = RecordingTracer()
        chain = self._make_chain()

        with patch("kiara.generation.chain._tracer", recorder):
            gen, sources = chain.query_stream("Was ist KIara?")
            # Generator komplett konsumieren
            answer = "".join(gen)

        assert chain.last_trace_id is not None
        assert len(chain.last_trace_id) == 32  # 032x hex
        assert "rag_query" in recorder.get_span_names()
        assert len(sources) > 0

    def test_rag_chain_marks_empty_datasource_list_as_filter_present(self):
        """filter_datasource_ids=[] wird als aktiver Filter erkannt."""
        recorder = RecordingTracer()
        chain = self._make_chain()

        with patch("kiara.generation.chain._tracer", recorder):
            gen, sources = chain.query_stream("Test", filter_datasource_ids=[])

        rag_span = recorder.get_span("rag_query")
        assert rag_span is not None
        assert rag_span.attributes["has_filter"] is True
        assert rag_span.attributes["has_datasource_filter"] is True
        assert rag_span.attributes["datasource_filter_count"] == 0


# ---------------------------------------------------------------------------
# _stream_standard / _stream_rewriter trace_id Call-Sites
# ---------------------------------------------------------------------------


class TestApiChatTraceidCallSites:
    """Tests fuer trace_id-Weiterreichung in api_chat.py."""

    def test_stream_standard_logs_trace_id_from_chain_property(self):
        """_stream_standard reicht chain.last_trace_id an _log_query_audit weiter."""

        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Antwort"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_prompt = ""
        mock_chain.last_expansions = []
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_anon_question = ""
        mock_chain.last_anon_answer = ""
        mock_chain.last_trace_id = "trace-standard-123"
        mock_chain.backend = MagicMock(provider="ollama", model="test")

        with (
            patch(
                "kiara.web.api_chat._prepare_standard_context",
                return_value=(mock_chain, None, None),
            ),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            import asyncio

            from kiara.web.api_chat import _stream_standard

            config = {"cloud": {"enabled": False}}
            data = MagicMock()
            data.question = "Test"
            data.k = 5
            data.num_ctx = None
            data.filter_dir = None
            data.filter_type = None
            data.think = False
            data.datasource_ids = None
            data.search_results = None
            user = {"id": 1, "username": "test"}

            async def _consume():
                collected = []
                async for chunk in _stream_standard(
                    config,
                    data,
                    1,
                    user,
                    template_content="Kontext:\n{context}\n\nFrage: {question}",
                ):
                    collected.append(chunk)
                return collected

            asyncio.run(_consume())
            mock_audit.assert_called_once()
            assert mock_audit.call_args.kwargs["trace_id"] == "trace-standard-123"

    def test_stream_rewriter_logs_trace_id_from_chain_property(self):
        """_stream_rewriter reicht chain.last_trace_id an _log_query_audit weiter."""
        mock_chain = MagicMock()
        mock_chain.query_stream.return_value = (iter(["Antwort"]), [])
        mock_chain.route_used = "local"
        mock_chain.context_warning = None
        mock_chain.last_usage = {}
        mock_chain.last_prompt = ""
        mock_chain.rewritten_query = "optimiert"
        mock_chain.last_entities_anonymized = 0
        mock_chain.last_anon_question = ""
        mock_chain.last_anon_answer = ""
        mock_chain.last_trace_id = "trace-rewriter-456"
        mock_chain.backend = MagicMock(provider="ollama", model="test")

        with (
            patch("kiara.web.api_chat._build_datasource_filter", return_value=None),
            patch("kiara.web.api_chat._build_restriction_filter", return_value=None),
            patch("kiara.web.api_chat._build_context_override", return_value=None),
            patch(
                "kiara.generation.cloud_routing._create_chain_backend",
                return_value=MagicMock(provider="ollama", model="test"),
            ),
            patch("kiara.generation.rewriter_chain.RewriterChain", return_value=mock_chain),
            patch("kiara.web.api_chat._save_assistant_message"),
            patch("kiara.web.api_chat._log_query_audit") as mock_audit,
        ):
            import asyncio

            from kiara.web.api_chat import _stream_rewriter

            config = {
                "cloud": {"enabled": False},
                "chat": {"rewriter": {}},
                "ollama": {"url": "http://localhost:11434", "llm_model": "test"},
            }
            data = MagicMock()
            data.question = "Test"
            data.k = 5
            data.num_ctx = None
            data.filter_dir = None
            data.filter_type = None
            data.think = False
            data.datasource_ids = None
            data.search_results = None
            data.filter_datasource_ids = None
            data.filter_datasource_exclude = False
            data.context_chunks = None
            data.embedding_model = None
            data.llm_backend_id = None
            user = {"id": 1, "username": "test"}

            async def _consume():
                collected = []
                async for chunk in _stream_rewriter(
                    config,
                    data,
                    1,
                    user,
                    template_content="Kontext:\n{context}\n\nFrage: {question}",
                ):
                    collected.append(chunk)
                return collected

            asyncio.run(_consume())
            mock_audit.assert_called_once()
            assert mock_audit.call_args.kwargs["trace_id"] == "trace-rewriter-456"


# ---------------------------------------------------------------------------
# RewriterChain Delegation
# ---------------------------------------------------------------------------


class TestRewriterChainTracing:
    """Tests fuer RewriterChain.last_trace_id Delegation."""

    def test_rewriter_chain_delegates_last_trace_id(self):
        """RewriterChain.last_trace_id delegiert an RAGChain._last_trace_id."""
        from kiara.generation.rewriter_chain import RewriterChain

        mock_rag = MagicMock()
        mock_rag.last_trace_id = "abc123"

        chain = object.__new__(RewriterChain)
        chain._rag_chain = mock_rag

        assert chain.last_trace_id == "abc123"


# ---------------------------------------------------------------------------
# Reranker Tracing
# ---------------------------------------------------------------------------


class TestRerankerTracing:
    """Tests fuer Tracing im CrossEncoderReranker."""

    def _make_reranker(self, enabled=True):
        from kiara.retrieval.reranker import CrossEncoderReranker

        config = {
            "retrieval": {
                "reranking": enabled,
                "rerank_model": "test-model",
                "rerank_top_k": 3,
            },
        }
        return CrossEncoderReranker(config)

    def test_reranker_sets_output_count_on_disabled_guard(self):
        """Deaktivierter Reranker setzt output_count im Span."""
        recorder = RecordingTracer()
        reranker = self._make_reranker(enabled=False)
        results = [{"text": "a"}, {"text": "b"}]

        with patch("kiara.retrieval.reranker._tracer", recorder):
            out = reranker.rerank("query", results)

        span = recorder.get_span("rerank")
        assert span is not None
        assert span.attributes["enabled"] is False
        assert span.attributes["output_count"] == len(out)

    def test_reranker_sets_output_count_on_predict_failure(self):
        """Cross-Encoder predict()-Fehler: Fallback + output_count gesetzt."""
        recorder = RecordingTracer()
        reranker = self._make_reranker(enabled=True)
        reranker._cross_encoder = MagicMock()
        reranker._cross_encoder.predict.side_effect = RuntimeError("GPU OOM")
        results = [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}]

        with patch("kiara.retrieval.reranker._tracer", recorder):
            out = reranker.rerank("query", results)

        span = recorder.get_span("rerank")
        assert span is not None
        assert span.attributes["output_count"] == len(out)
        assert len(out) == 3  # top_k=3

    def test_reranker_sets_output_count_on_length_mismatch(self):
        """Score-Liste falscher Laenge: Fallback + output_count gesetzt."""
        recorder = RecordingTracer()
        reranker = self._make_reranker(enabled=True)
        reranker._cross_encoder = MagicMock()
        reranker._cross_encoder.predict.return_value = [0.5]  # Falsche Laenge
        results = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

        with patch("kiara.retrieval.reranker._tracer", recorder):
            out = reranker.rerank("query", results)

        span = recorder.get_span("rerank")
        assert span.attributes["output_count"] == len(out)

    def test_reranker_sets_output_count_after_load_model_fallback(self):
        """_load_model() setzt enabled=False: Fallback + output_count gesetzt."""
        recorder = RecordingTracer()
        reranker = self._make_reranker(enabled=True)
        reranker._cross_encoder = None

        def fake_load():
            reranker._enabled = False

        reranker._load_model = fake_load
        results = [{"text": "a"}, {"text": "b"}]

        with patch("kiara.retrieval.reranker._tracer", recorder):
            out = reranker.rerank("query", results)

        span = recorder.get_span("rerank")
        assert span.attributes["output_count"] == len(out)
