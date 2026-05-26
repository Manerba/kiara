"""Tests fuer HybridRetriever._es_search() (kiara.retrieval.retriever).

Deckt ab: Query-Bau, fuzziness, Filter-Propagation, DS-Filter, k-Limit.
BugScan-Referenz: F04_HOCH_es_search_untested
"""

import time
from unittest.mock import MagicMock, patch

from kiara.retrieval.retriever import HybridRetriever, _FilterParams

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
        "_score": score,
        "_source": {
            "text": text,
            "metadata": {"source": source, "file_type": file_type, "directory": directory},
        },
    }


def _make_es_response(hits):
    """Wraps hits in ES response format."""
    return {"hits": {"hits": hits, "total": {"value": len(hits)}}}


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
    retriever._es_client = mock_es
    return retriever, mock_es


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEsSearch:
    """Tests fuer _es_search() Methode."""

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

    def test_es_unavailable_returns_empty(self):
        """ES nicht verbunden: leere Liste statt Exception."""
        retriever, _ = _make_retriever(_es_config())
        retriever._es_client = None
        HybridRetriever._es_fail_until = time.time() + 60

        results = retriever._es_search("test", k=5)

        assert results == []
        HybridRetriever._es_fail_until = 0.0

    def test_es_search_exception_returns_empty(self):
        """ES-Search-Exception: leere Liste statt Crash."""
        retriever, mock_es = _make_retriever(_es_config())
        mock_es.search.side_effect = ConnectionError("connection refused")

        results = retriever._es_search("test", k=5)

        assert results == []

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
