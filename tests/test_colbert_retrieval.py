"""Tests fuer ColBERT-Retrieval-Integration in HybridRetriever."""

import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")


class TestSetupColbert:
    """_setup_colbert Initialisierung."""

    def _make_retriever(self, config, embedding_model="nomic-embed-text"):
        """Erstellt einen HybridRetriever mit gemocktem Backend."""
        with (
            patch("kiara.retrieval.retriever.HybridRetriever._setup_hybrid_mode"),
            patch("kiara.retrieval.retriever.HybridRetriever._setup_graph"),
            patch("kiara.config.get_embedding_model", return_value=embedding_model),
            patch("kiara.retrieval.vectorstore_qdrant.QdrantVectorStore") as mock_vs,
        ):
            mock_store = MagicMock()
            mock_store.get_collection_name.return_value = "kb_nomic-embed-text"
            mock_vs.return_value = mock_store

            from kiara.retrieval.retriever import HybridRetriever

            r = HybridRetriever(config, embedding_model=embedding_model)
            return r

    def test_colbert_disabled_by_default(self):
        config = {"retrieval": {}}
        r = self._make_retriever(config)
        assert r._colbert_enabled is False
        assert r._colbert_model == ""
        assert r._colbert_store is None

    @patch("kiara.retrieval.colbert.ColBERTStore")
    def test_colbert_enabled(self, mock_store_cls):
        config = {
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
            "vectordb": {},
        }
        mock_store_cls.return_value = MagicMock()
        r = self._make_retriever(config)
        assert r._colbert_enabled is True
        assert r._colbert_model == "colbert-xm"

    def test_colbert_init_error_fail_closed(self):
        from kiara.exceptions import ColBERTInitializationError

        config = {"retrieval": {"colbert": {"enabled": True, "model": "bad-model"}}, "vectordb": {}}
        with (
            patch("kiara.retrieval.colbert.ColBERTStore", side_effect=Exception("fail")),
            pytest.raises(
                ColBERTInitializationError, match="ColBERT konnte nicht initialisiert werden"
            ),
        ):
            self._make_retriever(config)

    def test_colbert_empty_model_name_fail_closed(self):
        from kiara.exceptions import ColBERTInitializationError

        config = {"retrieval": {"colbert": {"enabled": True, "model": "  "}}, "vectordb": {}}
        with pytest.raises(ColBERTInitializationError, match="kein gueltiger Modellname"):
            self._make_retriever(config)


class TestColbertRetrieve:
    """_colbert_retrieve Methode."""

    def test_disabled_returns_empty(self):
        from kiara.retrieval.retriever import HybridRetriever

        r = HybridRetriever.__new__(HybridRetriever)
        r._colbert_enabled = False
        r._colbert_store = None
        result = r._colbert_retrieve("test query", 5, None)
        assert result == []

    def test_enabled_with_mock_store(self):
        from kiara.retrieval.retriever import HybridRetriever

        r = HybridRetriever.__new__(HybridRetriever)
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._config = {"retrieval": {"colbert": {"url": "http://test:11434"}}, "vectordb": {}}

        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "text": "result1",
                "metadata": {"source": "/test.txt"},
                "score": 0.95,
                "_doc_id": "abc123",
            },
        ]
        r._colbert_store = mock_store
        r._store = MagicMock()
        r._store.get_documents_by_ids.return_value = {
            "ids": ["abc123"],
            "documents": ["Aktueller gespeicherter Text"],
            "metadatas": [{"source": "/test.txt", "processing_version": "fertig"}],
        }

        with patch(
            "kiara.retrieval.colbert.embed_colbert_query",
            return_value=[[0.1, 0.2]],
        ) as embed_query:
            results = r._colbert_retrieve("query", 5, None)
            assert len(results) == 1
            assert results[0]["_doc_id"] == "abc123"
            assert results[0]["text"] == "Aktueller gespeicherter Text"
            assert results[0]["metadata"]["processing_version"] == "fertig"
            assert results[0]["score"] == 0.95
            embed_query.assert_called_once_with(r._config, "colbert-xm", "query")
            mock_store.search.assert_called_once()

    def test_where_passed_to_store(self):
        from kiara.retrieval.retriever import HybridRetriever

        r = HybridRetriever.__new__(HybridRetriever)
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._config = {"retrieval": {"colbert": {"url": "http://test:11434"}}, "vectordb": {}}

        mock_store = MagicMock()
        mock_store.search.return_value = []
        r._colbert_store = mock_store

        where = {"ds_ids": {"$in": [1, 2]}}
        with patch("kiara.retrieval.colbert.embed_colbert_query", return_value=[[0.1]]):
            r._colbert_retrieve("query", 5, where)
            call_args = mock_store.search.call_args
            assert call_args.kwargs.get("where") == where or call_args[1].get("where") == where

    def test_backend_failure_is_diagnostic_not_empty_success(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.retriever import HybridRetriever

        r = HybridRetriever.__new__(HybridRetriever)
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._config = {"retrieval": {"colbert": {"url": "http://test:11434"}}, "vectordb": {}}
        r._colbert_store = MagicMock()

        with patch(
            "kiara.retrieval.colbert.embed_colbert_query",
            side_effect=BackendUnavailableError("HTTP 409 Conflict"),
        ):
            result = r._colbert_retrieve_result("query", 5, None)

        assert result.attempted is True
        assert result.failed is True
        assert result.results == []
        assert result.error_type == "BackendUnavailableError"
        diagnostics = r.get_last_colbert_diagnostics()
        assert diagnostics["failed"] is True
        assert diagnostics["result_count"] == 0

    def test_empty_query_tokens_are_colbert_failure(self):
        from kiara.retrieval.retriever import HybridRetriever

        r = HybridRetriever.__new__(HybridRetriever)
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._config = {"retrieval": {"colbert": {"url": "http://test:11434"}}, "vectordb": {}}
        r._colbert_store = MagicMock()

        with patch("kiara.retrieval.colbert.embed_colbert_query", return_value=[]):
            result = r._colbert_retrieve_result("query", 5, None)

        assert result.attempted is True
        assert result.failed is True
        assert result.error_type == "RuntimeError"
        assert "keine Token-Vektoren" in result.error
        r._colbert_store.search.assert_not_called()

    def test_store_failure_is_colbert_failure(self):
        from kiara.exceptions import VectorDBConnectionError
        from kiara.retrieval.retriever import HybridRetriever

        r = HybridRetriever.__new__(HybridRetriever)
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._config = {"retrieval": {"colbert": {"url": "http://test:11434"}}, "vectordb": {}}
        r._colbert_store = MagicMock()
        r._colbert_store.search.side_effect = VectorDBConnectionError("qdrant timeout")

        with patch("kiara.retrieval.colbert.embed_colbert_query", return_value=[[0.1]]):
            result = r._colbert_retrieve_result("query", 5, None)

        assert result.attempted is True
        assert result.failed is True
        assert result.error_type == "VectorDBConnectionError"
        assert "qdrant timeout" in result.error

    def test_colbert_diagnostics_are_thread_local(self):
        from kiara.retrieval.retriever import HybridRetriever, _ColbertRetrieveResult

        r = HybridRetriever.__new__(HybridRetriever)
        r._colbert_diagnostics_local = threading.local()

        r._set_last_colbert_diagnostics(
            _ColbertRetrieveResult(
                attempted=True,
                failed=False,
                results=[{"_doc_id": "main"}],
                model="colbert-xm",
            )
        )

        worker_diag = {}

        def worker():
            r._set_last_colbert_diagnostics(
                _ColbertRetrieveResult(
                    attempted=True,
                    failed=True,
                    results=[],
                    model="colbert-xm",
                    error_type="BackendUnavailableError",
                )
            )
            worker_diag.update(r.get_last_colbert_diagnostics())

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert worker_diag["failed"] is True
        assert r.get_last_colbert_diagnostics()["failed"] is False
        assert r.get_last_colbert_diagnostics()["result_count"] == 1

    def test_empty_datasource_filter_resets_colbert_diagnostics(self):
        from kiara.retrieval.retriever import HybridRetriever, _ColbertRetrieveResult

        r = HybridRetriever.__new__(HybridRetriever)
        r._hybrid_mode = "none"
        r._graph_enabled = False
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._colbert_diagnostics_local = threading.local()
        r._set_last_colbert_diagnostics(
            _ColbertRetrieveResult(
                attempted=True,
                failed=True,
                results=[],
                model="colbert-xm",
                error_type="BackendUnavailableError",
            )
        )

        assert r.retrieve("query", filter_datasource_ids=[]) == []
        diagnostics = r.get_last_colbert_diagnostics()
        assert diagnostics["attempted"] is False
        assert diagnostics["failed"] is False

    def test_retrieve_resets_stale_colbert_diagnostics_before_other_failure(self):
        from kiara.retrieval.retriever import HybridRetriever, _ColbertRetrieveResult

        r = HybridRetriever.__new__(HybridRetriever)
        r._store = MagicMock()
        r._store.get_collection_name.return_value = "kb_test"
        r._config = {}
        r._bm25_enabled = False
        r._es_enabled = True
        r._hybrid_mode = "elasticsearch"
        r._graph_enabled = False
        r._graph_retriever = None
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._colbert_store = MagicMock()
        r._chunk_ds = MagicMock()
        r._embed_model_name = "nomic-embed-text"
        r._rrf_k = 60
        r._bm25_ttl = 3600
        r._colbert_diagnostics_local = threading.local()
        r._set_last_colbert_diagnostics(
            _ColbertRetrieveResult(
                attempted=True,
                failed=True,
                results=[],
                model="colbert-xm",
                error_type="BackendUnavailableError",
            )
        )
        r._check_bm25_ttl = MagicMock()
        r._semantic_retrieve = MagicMock(return_value=[])
        r._keyword_retrieve = MagicMock(side_effect=RuntimeError("keyword failed"))

        with pytest.raises(RuntimeError, match="keyword failed"):
            r.retrieve("query", k=5)

        diagnostics = r.get_last_colbert_diagnostics()
        assert diagnostics["attempted"] is False
        assert diagnostics["failed"] is False
        retrieval_diagnostics = r.get_last_retrieval_diagnostics()
        assert retrieval_diagnostics["paths"]["keyword"]["status"] == "error"
        assert retrieval_diagnostics["rrf"]["status"] == "skipped"
        assert retrieval_diagnostics["skipped_reason"] == "keyword_failed"


class TestEmbedColbert:
    """embed_colbert HTTP-Fehler, Retry und Fehlerdetails."""

    endpoint = "http://colbert.local/api/embed_colbert"

    def _response(self, status_code, *, json_payload=None, text=""):
        request = httpx.Request("POST", self.endpoint)
        if json_payload is not None:
            return httpx.Response(status_code, json=json_payload, request=request)
        return httpx.Response(status_code, text=text, request=request)

    def test_transient_http_error_retries_and_returns_embeddings(self):
        from kiara.retrieval.colbert import embed_colbert
        from kiara.shared.constants import WARMUP_RETRY_DELAY

        busy = self._response(501, text="transient proxy failure")
        ok = self._response(200, json_payload={"embeddings": [[[0.1, 0.2]]]})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", side_effect=[busy, ok]) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
        ):
            result = embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        assert result == [[[0.1, 0.2]]]
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(WARMUP_RETRY_DELAY)

    def test_429_retries_and_returns_embeddings(self):
        from kiara.retrieval.colbert import embed_colbert

        busy = self._response(429, text="rate limited")
        ok = self._response(200, json_payload={"embeddings": [[[0.1, 0.2]]]})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", side_effect=[busy, ok]) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
        ):
            result = embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        assert result == [[[0.1, 0.2]]]
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    def test_connect_error_retries_and_returns_embeddings(self):
        from kiara.retrieval.colbert import embed_colbert

        request = httpx.Request("POST", self.endpoint)
        connect_error = httpx.ConnectError("connection refused", request=request)
        ok = self._response(200, json_payload={"embeddings": [[[0.1, 0.2]]]})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch(
                "kiara.retrieval.colbert.httpx.post", side_effect=[connect_error, ok]
            ) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
        ):
            result = embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        assert result == [[[0.1, 0.2]]]
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    def test_read_error_retries_then_reports_concrete_last_error(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import _COLBERT_INDEX_TRANSIENT_RETRIES, embed_colbert

        request = httpx.Request("POST", self.endpoint)
        read_error = httpx.ReadError("server disconnected", request=request)

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", side_effect=read_error) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "ReadError" in message
        assert "server disconnected" in message
        assert "nach 3 Versuchen" in message
        assert mock_post.call_count == _COLBERT_INDEX_TRANSIENT_RETRIES + 1
        assert mock_sleep.call_count == _COLBERT_INDEX_TRANSIENT_RETRIES

    def test_404_does_not_retry_and_includes_endpoint_status_and_body(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        not_found = self._response(404, text="404 page not found")

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=not_found) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "ColBERT-Endpoint oder Modell nicht gefunden" in message
        assert "colbert-xm" in message
        assert self.endpoint in message
        assert "nach 1 Versuch" in message
        assert "HTTP 404 Not Found" in message
        assert "404 page not found" in message
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_409_retries_then_reports_concrete_last_error(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import _COLBERT_INDEX_TRANSIENT_RETRIES, embed_colbert

        conflict = self._response(409, text="GPU-Operation blockiert; vram_lease=active")

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=conflict) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "ColBERT-Backend-Fehler" in message
        assert self.endpoint in message
        assert "nach 3 Versuchen" in message
        assert "HTTP 409 Conflict" in message
        assert "vram_lease=active" in message
        assert mock_post.call_count == _COLBERT_INDEX_TRANSIENT_RETRIES + 1
        assert mock_sleep.call_count == _COLBERT_INDEX_TRANSIENT_RETRIES

    def test_query_uses_hardcoded_retrieval_budget(self):
        from kiara.retrieval.colbert import _COLBERT_QUERY_TIMEOUT, embed_colbert_query

        response = self._response(200, json_payload={"embeddings": [[[0.1, 0.2]]]})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response) as mock_post,
        ):
            result = embed_colbert_query({}, "colbert-xm", "query")

        assert result == [[0.1, 0.2]]
        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == _COLBERT_QUERY_TIMEOUT
        assert kwargs["json"] == {
            "model": "colbert-xm",
            "input": ["query"],
            "input_type": "search_query",
        }

    def test_query_uses_single_retry_budget(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import _COLBERT_QUERY_TRANSIENT_RETRIES, embed_colbert_query

        conflict = self._response(409, text="GPU-Operation blockiert; vram_lease=active")

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=conflict) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert_query({}, "colbert-xm", "query")

        message = str(exc_info.value)
        assert "nach 2 Versuchen" in message
        assert "HTTP 409 Conflict" in message
        assert mock_post.call_count == _COLBERT_QUERY_TRANSIENT_RETRIES + 1
        assert mock_sleep.call_count == _COLBERT_QUERY_TRANSIENT_RETRIES

    def test_permanent_error_body_does_not_retry(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        response = self._response(500, text="out of memory loading colbert-xm")

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "HTTP 500 Internal Server Error" in message
        assert "out of memory" in message
        assert "nach 1 Versuch" in message
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_permanent_error_code_variant_does_not_retry(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        response = self._response(500, json_payload={"code": "model_not_found"})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response) as mock_post,
            patch("kiara.retrieval.colbert.time.sleep") as mock_sleep,
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "HTTP 500 Internal Server Error" in message
        assert "model_not_found" in message
        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_invalid_json_reports_response_body(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        response = self._response(200, text="<html>bad gateway</html>")

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response),
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "ungueltige JSON-Antwort" in message
        assert self.endpoint in message
        assert "<html>bad gateway</html>" in message

    def test_missing_embeddings_field_reports_payload(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        response = self._response(200, json_payload={"error": "no embeddings"})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response),
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "Feld 'embeddings' fehlt" in message
        assert "no embeddings" in message

    def test_embeddings_not_list_reports_payload(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        response = self._response(200, json_payload={"embeddings": {"bad": "shape"}})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response),
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "ungueltiges embeddings-Payload" in message
        assert "erwartet list, erhalten dict" in message
        assert self.endpoint in message

    def test_cardinality_mismatch_reports_payload(self):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        response = self._response(200, json_payload={"embeddings": [[[0.1]], [[0.2]]]})

        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url", return_value="http://colbert.local"
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response),
            pytest.raises(BackendUnavailableError) as exc_info,
        ):
            embed_colbert({}, "colbert-xm", ["query"], input_type="search_query")

        message = str(exc_info.value)
        assert "2 Embeddings fuer 1 Texte" in message
        assert self.endpoint in message


class TestRetrieveWithColbert:
    """retrieve() mit 4 Quellen."""

    def test_use_rrf_includes_colbert(self, chunk_memberships):
        """ColBERT aktiv → use_rrf = True, auch wenn BM25/ES aus."""
        from kiara.retrieval.retriever import HybridRetriever, _ColbertRetrieveResult

        r = HybridRetriever.__new__(HybridRetriever)
        r._store = MagicMock()
        r._store.get_collection_name.return_value = "kb_test"
        r._config = {}
        r._bm25_enabled = False
        r._es_enabled = False
        r._hybrid_mode = "none"
        r._graph_enabled = False
        r._graph_retriever = None
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._colbert_store = MagicMock()
        r._chunk_ds = MagicMock()
        r._embed_model_name = "nomic-embed-text"
        r._rrf_k = 60
        r._bm25_ttl = 3600

        # Mock alle Retrieval-Methoden
        r._semantic_retrieve = MagicMock(
            return_value=[
                {"text": "a", "metadata": {"source": "a.txt"}, "score": 0.9, "_doc_id": "1"},
            ]
        )
        r._keyword_retrieve = MagicMock(return_value=[])
        r._graph_retrieve = MagicMock(return_value=[])
        r._colbert_retrieve_result = MagicMock(
            return_value=_ColbertRetrieveResult(
                attempted=True,
                failed=False,
                results=[
                    {"text": "b", "metadata": {"source": "b.txt"}, "score": 0.8, "_doc_id": "2"},
                ],
                model="colbert-xm",
            )
        )
        r._check_bm25_ttl = MagicMock()

        chunk_memberships(r, "1", "2")
        results = r.retrieve("test", k=5)
        assert len(results) == 2
        r._colbert_retrieve_result.assert_called_once()

    def test_colbert_failure_aborts_retrieve(self):
        """ColBERT aktiv + Fehler → keine stille RRF-Fusion ohne ColBERT."""
        from kiara.retrieval.retriever import HybridRetriever, _ColbertRetrieveResult

        r = HybridRetriever.__new__(HybridRetriever)
        r._store = MagicMock()
        r._store.get_collection_name.return_value = "kb_test"
        r._config = {}
        r._bm25_enabled = False
        r._es_enabled = False
        r._hybrid_mode = "none"
        r._graph_enabled = False
        r._graph_retriever = None
        r._colbert_enabled = True
        r._colbert_model = "colbert-xm"
        r._colbert_store = MagicMock()
        r._chunk_ds = MagicMock()
        r._embed_model_name = "nomic-embed-text"
        r._rrf_k = 60
        r._bm25_ttl = 3600
        r._semantic_retrieve = MagicMock(return_value=[])
        r._keyword_retrieve = MagicMock(return_value=[])
        r._graph_retrieve = MagicMock(return_value=[])
        r._colbert_retrieve_result = MagicMock(
            return_value=_ColbertRetrieveResult(
                attempted=True,
                failed=True,
                results=[],
                model="colbert-xm",
                error_type="BackendUnavailableError",
                error="HTTP 409 Conflict",
            )
        )
        r._check_bm25_ttl = MagicMock()

        with pytest.raises(RuntimeError, match="ColBERT-Retrieval fehlgeschlagen"):
            r.retrieve("test", k=5)

        diagnostics = r.get_last_retrieval_diagnostics()
        assert diagnostics["paths"]["colbert"]["status"] == "error"
        assert diagnostics["paths"]["colbert"]["error_type"] == "BackendUnavailableError"
        assert diagnostics["rrf"]["status"] == "skipped"
        assert diagnostics["skipped_reason"] == "colbert_failed"
        assert diagnostics["final_results"] == []


class TestVectorstoreFactory:
    """vectorstore_factory Erweiterungen."""

    def test_create_retrieval_store_kb(self):
        from kiara.retrieval.vectorstore_factory import create_retrieval_store

        with patch("kiara.retrieval.vectorstore_factory.create_vectorstore") as mock_vs:
            mock_vs.return_value = MagicMock()
            store = create_retrieval_store({}, "kb", "nomic-embed-text")
            mock_vs.assert_called_once()

    def test_create_retrieval_store_colbert(self):
        from kiara.retrieval.vectorstore_factory import create_retrieval_store

        with patch("kiara.retrieval.colbert.ColBERTStore") as mock_cs:
            mock_cs.return_value = MagicMock()
            store = create_retrieval_store({"vectordb": {}}, "colbert", "colbert-xm")
            mock_cs.assert_called_once()

    def test_create_retrieval_store_unknown_raises(self):
        from kiara.retrieval.vectorstore_factory import create_retrieval_store

        with pytest.raises(ValueError, match="Unbekannter"):
            create_retrieval_store({}, "unknown", "model")

    @patch("kiara.retrieval.vectorstore_factory.list_collections_for_backend")
    @patch("kiara.retrieval.colbert.ColBERTStore.list_colbert_collections")
    def test_list_all_retrieval_collections(self, mock_colbert, mock_dense):
        from kiara.retrieval.vectorstore_factory import list_all_retrieval_collections

        mock_dense.return_value = [
            {"name": "kb_nomic-embed-text", "model": "nomic-embed-text", "chunks": 100}
        ]
        mock_colbert.return_value = [
            {"kind": "colbert", "name": "colbert_colbert-xm", "model": "colbert-xm", "chunks": 50}
        ]
        result = list_all_retrieval_collections({})
        assert len(result) == 2
        assert result[0]["kind"] == "kb"
        assert result[1]["kind"] == "colbert"
