"""Tests fuer HybridRetriever._es_search() (kiara.retrieval.retriever).

Deckt ab: Query-Bau, fuzziness, Filter-Propagation, DS-Filter, k-Limit.
BugScan-Referenz: F04_HOCH_es_search_untested
"""

import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from kiara.retrieval.retriever import HybridRetriever, _FilterParams


@pytest.fixture(autouse=True)
def search_targets(monkeypatch):
    # These legacy tests isolate query construction/PG point filtering. Full
    # inventory, permissions and model guards are exercised in the S3 suite.
    from kiara.retrieval.elasticsearch_indices import DatasourceIndexTarget

    @contextmanager
    def scope(config, ids, model=None):
        yield () if ids == [] else (DatasourceIndexTarget(1, "kiara_test_ds_1", 1),)

    monkeypatch.setattr("kiara.retrieval.elasticsearch_service.search_scope", scope)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _es_config(fuzziness="none"):
    """Config mit Elasticsearch aktiviert."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "test",
            "embedding_model": "test-embed",
        },
        "vectordb": {
            "path": "/tmp/test_vectordb",
            "collection": "test",
        },
        "datasource": {"path": "/tmp/test"},
        "retrieval": {
            "hybrid_search": "elasticsearch",
            "rrf_k": 60,
            "elasticsearch_url": "http://localhost:9200",
            "elasticsearch_index": "kiara_test",
            "elasticsearch_fuzziness": fuzziness,
        },
    }


def _make_es_hit(doc_id, text, source, score=1.0, file_type=".txt", directory="/data"):
    """Erzeugt einen ES-Hit im Response-Format."""
    return {
        "_id": doc_id,
        "_index": "kiara_test_ds_1",
        "_score": score,
        "_source": {
            "text": text,
            "metadata": {"source": source, "file_type": file_type, "directory": directory},
        },
    }


def _make_es_response(hits):
    """Wraps hits in ES response format."""
    return {
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "failed": 0},
        "hits": {"hits": hits, "total": {"value": len(hits)}},
    }


def _make_retriever(config):
    """Erzeugt einen HybridRetriever mit gemocktem VectorStore + ES-Client."""
    with (
        patch("kiara.retrieval.retriever.VectorStore") as mock_vs_cls,
        patch("kiara.config.get_embedding_model", return_value="test-embed"),
        patch("kiara.backend_selector.normalize_model_name", return_value="test-embed"),
    ):
        mock_vs_cls.return_value = MagicMock()
        retriever = HybridRetriever(config, embedding_model="test-embed")
    mock_es = MagicMock()
    mock_es.indices.get_settings.return_value = {
        "kiara_test_ds_1": {"defaults": {"index.max_result_window": "10000"}}
    }
    retriever._es_client = mock_es
    return retriever, mock_es


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEsSearch:
    """Tests fuer _es_search() Methode."""

    def test_rejects_window_overflow_before_search(self):
        retriever, mock_es = _make_retriever(_es_config())
        with pytest.raises(ValueError, match="32400.*10000"):
            retriever._es_search("Rechnungen 2025", k=2025 * 4)
        mock_es.search.assert_not_called()

    def test_configured_multiplier_respects_actual_index_limit(self):
        config = _es_config()
        config["retrieval"]["keyword_fetch_multiplier"] = 3
        retriever, mock_es = _make_retriever(config)
        mock_es.indices.get_settings.return_value["kiara_test_ds_1"]["settings"] = {
            "index.max_result_window": "600"
        }
        mock_es.search.return_value = _make_es_response([])
        retriever._es_search("Test", k=200)
        assert mock_es.search.call_args.kwargs["body"]["size"] == 600
        mock_es.search.reset_mock()
        with pytest.raises(ValueError, match="603.*600"):
            retriever._es_search("Test", k=201)
        mock_es.search.assert_not_called()

    def test_basic_query_no_fuzziness(self):
        """Basis-Query ohne Fuzziness: multi_match ohne fuzziness-Key."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response(
            [
                _make_es_hit("c1", "Hello", "/data/a.txt", score=2.0),
            ]
        )

        results = retriever._es_search("test query", k=5)

        assert len(results) == 1
        assert results[0]["_doc_id"] == "c1"
        assert results[0]["text"] == "Hello"
        assert results[0]["score"] == 2.0

        call_body = mock_es.search.call_args[1]["body"]
        multi_match = call_body["query"]["bool"]["must"][0]["multi_match"]
        assert multi_match["query"] == "test query"
        assert "fuzziness" not in multi_match

    def test_fuzziness_auto(self):
        """Fuzziness 'AUTO': multi_match enthaelt fuzziness-Key."""
        retriever, mock_es = _make_retriever(_es_config(fuzziness="AUTO"))
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search("test", k=5)

        call_body = mock_es.search.call_args[1]["body"]
        multi_match = call_body["query"]["bool"]["must"][0]["multi_match"]
        assert multi_match["fuzziness"] == "AUTO"

    def test_fuzziness_none_string(self):
        """Fuzziness 'none' (String): kein fuzziness-Key in Query."""
        retriever, mock_es = _make_retriever(_es_config(fuzziness="none"))
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search("test", k=5)

        call_body = mock_es.search.call_args[1]["body"]
        multi_match = call_body["query"]["bool"]["must"][0]["multi_match"]
        assert "fuzziness" not in multi_match

    def test_filter_source(self):
        """filter_source erzeugt wildcard auf metadata.source.raw."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search("test", k=5, filters=_FilterParams(source="invoice"))

        call_body = mock_es.search.call_args[1]["body"]
        es_filter = call_body["query"]["bool"]["filter"]
        assert any("wildcard" in f and "metadata.source.raw" in f["wildcard"] for f in es_filter)

    def test_filter_type_with_dot(self):
        """filter_type mit Punkt: wird direkt als term verwendet."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search("test", k=5, filters=_FilterParams(file_type=".pdf"))

        call_body = mock_es.search.call_args[1]["body"]
        es_filter = call_body["query"]["bool"]["filter"]
        term_filters = [f for f in es_filter if "term" in f]
        assert any(f["term"]["metadata.file_type"] == ".pdf" for f in term_filters)

    def test_filter_type_without_dot(self):
        """filter_type ohne Punkt: Punkt wird vorangestellt."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search("test", k=5, filters=_FilterParams(file_type="pdf"))

        call_body = mock_es.search.call_args[1]["body"]
        es_filter = call_body["query"]["bool"]["filter"]
        term_filters = [f for f in es_filter if "term" in f]
        assert any(f["term"]["metadata.file_type"] == ".pdf" for f in term_filters)

    def test_filter_dir(self):
        """filter_dir erzeugt wildcard auf metadata.directory."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search("test", k=5, filters=_FilterParams(directory="projects"))

        call_body = mock_es.search.call_args[1]["body"]
        es_filter = call_body["query"]["bool"]["filter"]
        assert any("wildcard" in f and "metadata.directory" in f["wildcard"] for f in es_filter)

    def test_k_limit_applied(self):
        """Ergebnisse werden auf k begrenzt nach DS-Filter."""
        retriever, mock_es = _make_retriever(_es_config())
        hits = [
            _make_es_hit(f"c{i}", f"text{i}", f"/data/{i}.txt", score=10 - i) for i in range(20)
        ]
        mock_es.search.return_value = _make_es_response(hits)

        results = retriever._es_search("test", k=3)

        assert len(results) == 3

    def test_ds_filter_applied(self):
        """DS-Filter entfernt nicht-berechtigte Chunks."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response(
            [
                _make_es_hit("c1", "allowed", "/a.txt", score=2.0),
                _make_es_hit("c2", "denied", "/b.txt", score=1.5),
                _make_es_hit("c3", "allowed2", "/c.txt", score=1.0),
            ]
        )
        retriever._chunk_ds = MagicMock()
        retriever._chunk_ds.filter_allowed.return_value = {"c1", "c3"}

        results = retriever._es_search("test", k=10, filters=_FilterParams(datasource_ids=[1, 2]))

        assert len(results) == 2
        assert results[0]["_doc_id"] == "c1"
        assert results[1]["_doc_id"] == "c3"

    def test_ds_filter_none_no_filtering(self):
        """DS-Filter=None: alle Ergebnisse werden durchgelassen."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response(
            [
                _make_es_hit("c1", "text1", "/a.txt"),
                _make_es_hit("c2", "text2", "/b.txt"),
            ]
        )

        results = retriever._es_search("test", k=10, filters=_FilterParams())

        assert len(results) == 2

    def test_es_unavailable_raises_for_path_diagnostics(self):
        """ES nicht verbunden: Exception statt stiller leerer Keyword-Pfad."""
        retriever, _ = _make_retriever(_es_config())
        retriever._es_client = None
        HybridRetriever._es_fail_until = time.time() + 60

        try:
            with pytest.raises(RuntimeError, match="Elasticsearch-Client nicht verfuegbar"):
                retriever._es_search("test", k=5)
        finally:
            HybridRetriever._es_fail_until = 0.0

    def test_es_search_exception_raises_for_path_diagnostics(self):
        """ES-Search-Exception: Exception statt stiller leerer Keyword-Pfad."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.side_effect = ConnectionError("connection refused")

        with pytest.raises(ConnectionError, match="connection refused"):
            retriever._es_search("test", k=5)

    def test_overfetch_factor(self):
        """ES-Query holt k * KEYWORD_OVERFETCH_FACTOR Ergebnisse."""
        from kiara.shared.constants import KEYWORD_OVERFETCH_FACTOR

        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search("test", k=5)

        call_body = mock_es.search.call_args[1]["body"]
        assert call_body["size"] == 5 * KEYWORD_OVERFETCH_FACTOR

    def test_combined_filters(self):
        """Mehrere Filter gleichzeitig: alle in ES bool.filter."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response([])

        retriever._es_search(
            "test",
            k=5,
            filters=_FilterParams(source="invoice", file_type="pdf", directory="projects"),
        )

        call_body = mock_es.search.call_args[1]["body"]
        es_filter = call_body["query"]["bool"]["filter"]
        assert len(es_filter) == 3

    def test_empty_results(self):
        """Leere ES-Antwort: leere Liste."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.return_value = _make_es_response([])

        results = retriever._es_search("test", k=5)

        assert results == []
