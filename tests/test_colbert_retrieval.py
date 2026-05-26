"""Tests fuer ColBERT-Retrieval-Integration in HybridRetriever."""

from unittest.mock import MagicMock, patch

import pytest


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

    def test_colbert_init_error_graceful(self):
        config = {"retrieval": {"colbert": {"enabled": True, "model": "bad-model"}}, "vectordb": {}}
        with patch("kiara.retrieval.colbert.ColBERTStore", side_effect=Exception("fail")):
            r = self._make_retriever(config)
            assert r._colbert_enabled is False


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

        with patch("kiara.retrieval.colbert.embed_colbert_query", return_value=[[0.1, 0.2]]):
            results = r._colbert_retrieve("query", 5, None)
            assert len(results) == 1
            assert results[0]["_doc_id"] == "abc123"
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


class TestRetrieveWithColbert:
    """retrieve() mit 4 Quellen."""

    def test_use_rrf_includes_colbert(self):
        """ColBERT aktiv → use_rrf = True, auch wenn BM25/ES aus."""
        from kiara.retrieval.retriever import HybridRetriever

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
                {"text": "a", "metadata": {}, "score": 0.9, "_doc_id": "1"},
            ]
        )
        r._keyword_retrieve = MagicMock(return_value=[])
        r._graph_retrieve = MagicMock(return_value=[])
        r._colbert_retrieve = MagicMock(
            return_value=[
                {"text": "b", "metadata": {}, "score": 0.8, "_doc_id": "2"},
            ]
        )
        r._check_bm25_ttl = MagicMock()

        results = r.retrieve("test", k=5)
        assert len(results) == 2
        r._colbert_retrieve.assert_called_once()


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
