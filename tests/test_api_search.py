"""Tests fuer api_search.py — Textsuche und Semantische Suche.

Testet:
- Route-Level Empty-Set-Guard (allowed_ds_ids=[])
- Total-Count Korrektheit (kein min(total, page_size))
- ES-terms-Filter fuer DS-Zugriffskontrolle
- Admin-Suche (ungefiltert und gefiltert)
- Semantische Suche exakter Total + Fallback
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# elasticsearch ist im Test-Env ein Namespace-Stub — Elasticsearch-Klasse ergaenzen
import elasticsearch as _es_pkg
import pytest

if not hasattr(_es_pkg, "Elasticsearch"):
    _es_pkg.Elasticsearch = MagicMock


@pytest.fixture(autouse=True)
def text_search_boundaries(monkeypatch):
    # Query/format tests; real boundary checks have their own S3 regressions.
    from kiara.retrieval.elasticsearch_indices import DatasourceIndexTarget

    @contextmanager
    def scope(config, ids, model=None):
        yield () if ids == [] else (DatasourceIndexTarget(1, "kiara_test_ds_1", 1),)

    monkeypatch.setattr("kiara.retrieval.elasticsearch_service.search_scope", scope)
    monkeypatch.setattr("kiara.web.api_search._assert_text_hits_available", lambda *a: None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config():
    return {
        "retrieval": {
            "hybrid_search": "elasticsearch",
            "elasticsearch_url": "http://localhost:9200",
            "elasticsearch_index": "kiara_test",
            "elasticsearch_fuzziness": "AUTO",
        }
    }


def _make_es_response(total=200, hits=None, max_score=10.0):
    """Erzeugt eine ES-Search-Response."""
    if hits is None:
        hits = [
            {
                "_id": f"chunk_{i}",
                "_score": 10.0 - i * 0.1,
                "_source": {
                    "text": f"Text {i}",
                    "metadata": {
                        "source": f"/data/file{i}.txt",
                        "directory": "/data",
                        "file_type": "txt",
                        "modified": "2025-01-01",
                        "ds_ids": [1, 2],
                    },
                },
            }
            for i in range(10)
        ]
    for hit in hits:
        hit["_index"] = "kiara_test_ds_1"
    return {
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "failed": 0},
        "hits": {
            "total": {"value": total},
            "max_score": max_score,
            "hits": hits,
        },
    }


def _make_search_request(**kwargs):
    """Erzeugt ein SearchRequest-Objekt."""
    from kiara.web.api_search import SearchRequest

    defaults = {"query": "test", "mode": "text", "page": 1, "page_size": 10}
    defaults.update(kwargs)
    return SearchRequest(**defaults)


def _make_es_mock(response):
    """Erzeugt einen gemockten ES-Client."""
    es = MagicMock()
    es.indices.get_settings.return_value = {
        "kiara_test_ds_1": {"defaults": {"index.max_result_window": "10000"}}
    }
    es.info.return_value = {}
    es.search.return_value = response
    return es


# ---------------------------------------------------------------------------
# Empty-Set-Guard (Spec Punkt 3)
# ---------------------------------------------------------------------------


class TestEmptySetGuard:
    """allowed_ds_ids=[] muss ohne Backend-Call 0 Treffer liefern."""

    @patch("kiara.web.api_search._build_datasource_filter", return_value=[])
    @patch("kiara.web.api_search.load_config", return_value=_config())
    @patch("kiara.web.api_search._search_text")
    @patch("kiara.web.api_search._search_semantic")
    @pytest.mark.anyio
    async def test_empty_ds_ids_text(self, mock_sem, mock_text, mock_config, mock_ds):
        """Text-Modus mit leerer DS-Liste -> 0 Treffer, kein ES-Call."""
        from kiara.web.api_search import SearchRequest, search

        data = SearchRequest(query="test", mode="text")
        result = await search(data, user={"role": "user"})

        assert result["total"] == 0
        assert result["results"] == []
        mock_text.assert_not_called()
        mock_sem.assert_not_called()

    @patch("kiara.config.get_embedding_model", return_value="dummy-model")
    @patch("kiara.web.api_search._build_datasource_filter", return_value=[])
    @patch("kiara.web.api_search.load_config", return_value=_config())
    @patch("kiara.web.api_search._search_text")
    @patch("kiara.web.api_search._search_semantic")
    @pytest.mark.anyio
    async def test_empty_ds_ids_semantic(self, mock_sem, mock_text, mock_config, mock_ds, mock_emb):
        """Semantic-Modus mit leerer DS-Liste -> 0 Treffer."""
        from kiara.web.api_search import SearchRequest, search

        data = SearchRequest(query="test", mode="semantic")
        result = await search(data, user={"role": "user"})

        assert result["total"] == 0
        assert result["results"] == []
        mock_text.assert_not_called()
        mock_sem.assert_not_called()


# ---------------------------------------------------------------------------
# Total-Count (Spec Punkt 2 — kein min(total, page_size))
# ---------------------------------------------------------------------------


class TestTotalCount:
    """Total in Response muss der echte Total sein, nicht page_size."""

    @patch("kiara.web.api_search._build_datasource_filter", return_value=None)
    @patch("kiara.web.api_search.load_config", return_value=_config())
    @patch("kiara.web.api_search._search_text", return_value=([], 500))
    @pytest.mark.anyio
    async def test_total_not_capped(self, mock_text, mock_config, mock_ds):
        """Total=500 darf NICHT auf page_size=10 gekappt werden."""
        from kiara.web.api_search import SearchRequest, search

        data = SearchRequest(query="test", mode="text", page_size=10)
        result = await search(data, user={"role": "admin"})

        assert result["total"] == 500


# ---------------------------------------------------------------------------
# ES-terms-Filter (Spec Punkt 1)
# ---------------------------------------------------------------------------


class TestEsTermsFilter:
    """DS-Zugriffskontrolle muss direkt ueber ES terms-Filter laufen."""

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_non_admin_sets_terms_filter(self, mock_configured):
        """Non-Admin mit allowed_ds_ids setzt terms-Filter in ES-Query."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=50))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request()
            results, total = _search_text(_config(), data, allowed_ds_ids=[1, 2])

        call_body = es_mock.search.call_args[1]["body"]
        filters = call_body["query"]["bool"]["filter"]
        ds_filter = [f for f in filters if "terms" in f and "metadata.ds_ids" in f["terms"]]
        assert len(ds_filter) == 1
        assert ds_filter[0]["terms"]["metadata.ds_ids"] == [1, 2]

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_admin_no_terms_filter(self, mock_configured):
        """Admin ohne DS-Auswahl (allowed_ds_ids=None) -> kein terms-Filter."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=200))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request()
            results, total = _search_text(_config(), data, allowed_ds_ids=None)

        # Keine Filter oder kein ds_ids-Filter
        call_body = es_mock.search.call_args[1]["body"]
        query = call_body["query"]["bool"]
        if "filter" in query:
            for f in query["filter"]:
                assert "metadata.ds_ids" not in f.get("terms", {})

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_admin_with_ds_selection_sets_filter(self, mock_configured):
        """Admin mit expliziter DS-Auswahl setzt terms-Filter."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=30))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request()
            results, total = _search_text(_config(), data, allowed_ds_ids=[5])

        call_body = es_mock.search.call_args[1]["body"]
        filters = call_body["query"]["bool"]["filter"]
        ds_filter = [f for f in filters if "terms" in f and "metadata.ds_ids" in f["terms"]]
        assert len(ds_filter) == 1
        assert ds_filter[0]["terms"]["metadata.ds_ids"] == [5]

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_total_from_es_response(self, mock_configured):
        """Total kommt direkt aus der ES-Response, nicht aus len(hits)."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=500))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(page_size=10)
            results, total = _search_text(_config(), data, allowed_ds_ids=[1])

        assert total == 500

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_no_overfetch(self, mock_configured):
        """fetch_size == page_size (kein 3x Overfetch mehr)."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=100))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(page_size=10)
            _search_text(_config(), data, allowed_ds_ids=[1, 2])

        call_body = es_mock.search.call_args[1]["body"]
        assert call_body["size"] == 10  # Nicht 30


# ---------------------------------------------------------------------------
# Datumsfilter
# ---------------------------------------------------------------------------


class TestDateFilters:
    """Date-only filter_date_to muss den ganzen ausgewaehlten Tag einschliessen."""

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_text_search_date_to_uses_end_of_day(self, mock_configured):
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=10))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(filter_date_to="2025-01-15")
            _search_text(_config(), data, allowed_ds_ids=None)

        filters = es_mock.search.call_args[1]["body"]["query"]["bool"]["filter"]
        range_filter = next(f for f in filters if "range" in f)
        assert range_filter["range"]["metadata.modified"]["lte"] == "2025-01-15T23:59:59.999999"

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x)
    @patch("kiara.config.get_embedding_model", return_value="model1")
    def test_semantic_search_date_to_uses_end_of_day(self, mock_model, mock_norm):
        from kiara.web.api_search import _search_semantic

        store_mock = MagicMock()
        store_mock.search.return_value = []
        store_mock.count_filtered.return_value = 0

        with (
            patch("kiara.retrieval.vectorstore_qdrant.QdrantVectorStore", return_value=store_mock),
            patch(
                "kiara.retrieval.vectorstore.VectorStore.list_collections",
                return_value=[{"model": "model1", "chunks": 1}],
            ),
        ):
            data = _make_search_request(
                mode="semantic",
                embedding_model="model1",
                filter_date_to="2025-01-15",
            )
            _search_semantic(_config(), data, allowed_ds_ids=None)

        where = store_mock.search.call_args.kwargs["where"]
        assert where["$and"][0]["modified"]["$lte"] == "2025-01-15T23:59:59.999999"


# ---------------------------------------------------------------------------
# Semantische Suche — exakter Total, kein spekulativer Fallback
# ---------------------------------------------------------------------------


class TestSemanticTotal:
    """Semantische Suche nutzt count_filtered() fail-closed."""

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x)
    @patch("kiara.config.get_embedding_model", return_value="model1")
    def test_exact_count(self, mock_model, mock_norm):
        """count_filtered() liefert exakten Total."""
        from kiara.web.api_search import _search_semantic

        store_mock = MagicMock()
        store_mock.search.return_value = [
            {"_doc_id": f"c{i}", "text": f"T{i}", "score": 0.9, "metadata": {}} for i in range(10)
        ]
        store_mock.count_filtered.return_value = 150

        with (
            patch("kiara.retrieval.vectorstore_qdrant.QdrantVectorStore", return_value=store_mock),
            patch(
                "kiara.retrieval.vectorstore.VectorStore.list_collections",
                return_value=[{"model": "model1", "chunks": 1}],
            ),
        ):
            data = _make_search_request(mode="semantic", page_size=10)
            results, total = _search_semantic(_config(), data, allowed_ds_ids=[1])

        assert total == 150
        store_mock.count_filtered.assert_called_once()

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x)
    @patch("kiara.config.get_embedding_model", return_value="model1")
    def test_short_page_skips_count(self, mock_model, mock_norm):
        """Letzte Page (len(raw_results) < page_size) -> Total ohne count_filtered ableiten."""
        from kiara.web.api_search import _search_semantic

        store_mock = MagicMock()
        store_mock.search.return_value = [
            {"_doc_id": f"c{i}", "text": f"T{i}", "score": 0.9, "metadata": {}} for i in range(3)
        ]
        store_mock.count_filtered.return_value = 999

        with (
            patch("kiara.retrieval.vectorstore_qdrant.QdrantVectorStore", return_value=store_mock),
            patch(
                "kiara.retrieval.vectorstore.VectorStore.list_collections",
                return_value=[{"model": "model1", "chunks": 1}],
            ),
        ):
            data = _make_search_request(mode="semantic", page=2, page_size=10)
            results, total = _search_semantic(_config(), data, allowed_ds_ids=[1])

        assert total == 13
        store_mock.count_filtered.assert_not_called()

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x)
    @patch("kiara.config.get_embedding_model", return_value="model1")
    def test_count_error_fails_closed(self, mock_model, mock_norm):
        """count_filtered() Exception -> HTTP 503 statt spekulativer Total-Count."""
        from fastapi import HTTPException

        from kiara.web.api_search import _search_semantic

        store_mock = MagicMock()
        store_mock.search.return_value = [
            {"_doc_id": f"c{i}", "text": f"T{i}", "score": 0.9, "metadata": {}} for i in range(10)
        ]
        store_mock.count_filtered.side_effect = RuntimeError("Qdrant error")

        with (
            patch("kiara.retrieval.vectorstore_qdrant.QdrantVectorStore", return_value=store_mock),
            patch(
                "kiara.retrieval.vectorstore.VectorStore.list_collections",
                return_value=[{"model": "model1", "chunks": 1}],
            ),
        ):
            data = _make_search_request(mode="semantic", page=2, page_size=10)
            with pytest.raises(HTTPException) as exc_info:
                _search_semantic(_config(), data, allowed_ds_ids=[1])

        assert exc_info.value.status_code == 503

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x)
    @patch("kiara.config.get_embedding_model", return_value="model1")
    def test_missing_doc_id_fails_closed(self, mock_model, mock_norm):
        """Fehlendes _doc_id im Suchergebnis -> KeyError statt leerem ID-String (Frontend-Magnet)."""
        from kiara.web.api_search import _search_semantic

        store_mock = MagicMock()
        store_mock.search.return_value = [
            {"text": "T1", "score": 0.9, "metadata": {}},  # _doc_id fehlt
        ]
        store_mock.count_filtered.return_value = 1

        with (
            patch("kiara.retrieval.vectorstore_qdrant.QdrantVectorStore", return_value=store_mock),
            patch(
                "kiara.retrieval.vectorstore.VectorStore.list_collections",
                return_value=[{"model": "model1", "chunks": 1}],
            ),
        ):
            data = _make_search_request(mode="semantic", page_size=10)
            with pytest.raises(KeyError):
                _search_semantic(_config(), data, allowed_ds_ids=[1])

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x)
    @patch("kiara.config.get_embedding_model", return_value="model1")
    @patch("kiara.web.api_search._build_datasource_filter", return_value=None)
    @patch("kiara.web.api_search.load_config", return_value=_config())
    @pytest.mark.anyio
    async def test_directory_filter_rejects_wildcards(
        self, mock_config, mock_ds, mock_model, mock_norm
    ):
        """Directory-Filter ist literal und lehnt Wildcard-Syntax einheitlich ab."""
        from fastapi import HTTPException

        from kiara.web.api_search import SearchRequest, search

        data = SearchRequest(query="test", mode="text", filter_dir="/data/*")

        with pytest.raises(HTTPException) as exc_info:
            await search(data, user={"id": 1, "role": "admin"})

        assert exc_info.value.status_code == 400
        assert "Wildcards" in exc_info.value.detail

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_text_search_directory_filter_uses_path_segment_anchor(self, mock_configured):
        """Relative Filter matchen Pfadsegmente, aber keine beliebigen Substrings."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=10))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(filter_dir="etc")
            _search_text(_config(), data, allowed_ds_ids=None)

        filters = es_mock.search.call_args[1]["body"]["query"]["bool"]["filter"]
        dir_filter = next(f for f in filters if "bool" in f and "should" in f["bool"])
        clauses = dir_filter["bool"]["should"]
        wildcards = {c["wildcard"]["metadata.directory"] for c in clauses if "wildcard" in c}

        assert "*etc*" not in wildcards
        assert "*/etc" in wildcards
        assert "*/etc/*" in wildcards
        assert "etc/*" in wildcards

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_text_search_absolute_directory_filter_uses_prefix(self, mock_configured):
        """Absolute Filter bleiben auf diesen Pfad und dessen Kinder beschraenkt."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=10))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(filter_dir="/data")
            _search_text(_config(), data, allowed_ds_ids=None)

        filters = es_mock.search.call_args[1]["body"]["query"]["bool"]["filter"]
        dir_filter = next(f for f in filters if "bool" in f and "should" in f["bool"])
        clauses = dir_filter["bool"]["should"]

        assert {"term": {"metadata.directory": "/data"}} in clauses
        assert {"wildcard": {"metadata.directory": "/data/*"}} in clauses
        assert len(clauses) == 2


# ---------------------------------------------------------------------------
# Pagination — korrekter from-Offset (Spec Punkt 3)
# ---------------------------------------------------------------------------


class TestPagination:
    """Seite 2+ muss den korrekten from-Offset an ES senden."""

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_page2_sends_correct_offset(self, mock_configured):
        """page=2, page_size=10 -> from=10."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=50))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(page=2, page_size=10)
            _search_text(_config(), data, allowed_ds_ids=[1])

        call_body = es_mock.search.call_args[1]["body"]
        assert call_body["from"] == 10
        assert call_body["size"] == 10

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_page3_sends_correct_offset(self, mock_configured):
        """page=3, page_size=5 -> from=10."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=100))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(page=3, page_size=5)
            _search_text(_config(), data, allowed_ds_ids=[1, 2])

        call_body = es_mock.search.call_args[1]["body"]
        assert call_body["from"] == 10
        assert call_body["size"] == 5

    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_page1_from_is_zero(self, mock_configured):
        """page=1 -> from=0."""
        from kiara.web.api_search import _search_text

        es_mock = _make_es_mock(_make_es_response(total=100))
        with patch("kiara.retrieval.elasticsearch_service._get_client", return_value=es_mock):
            data = _make_search_request(page=1, page_size=10)
            _search_text(_config(), data, allowed_ds_ids=None)

        call_body = es_mock.search.call_args[1]["body"]
        assert call_body["from"] == 0
