"""Tests fuer den HybridRetriever (kiara.retrieval.retriever)."""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kiara.retrieval.retriever import (
    KEYWORD_OVERFETCH_FACTOR,
    HybridRetriever,
    _FilterParams,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Minimale Config ohne Hybrid Search."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "test",
            "embedding_model": "test",
        },
        "vectordb": {
            "path": "/tmp/test_vectordb",
            "collection": "test",
        },
        "datasource": {"path": "/tmp/test"},
        "retrieval": {
            "hybrid_search": False,
            "rrf_k": 60,
        },
    }


@pytest.fixture
def hybrid_config():
    """Config mit aktiviertem Hybrid Search."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "test",
            "embedding_model": "test",
        },
        "vectordb": {
            "path": "/tmp/test_vectordb",
            "collection": "test",
        },
        "datasource": {"path": "/tmp/test"},
        "retrieval": {
            "hybrid_search": True,
            "rrf_k": 60,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    @patch("kiara.retrieval.retriever.VectorStore")
    def test_hybrid_disabled_fallback(self, mock_vs_cls, base_config):
        """Ohne hybrid_search=True: BM25 wird nicht aktiviert."""
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs

        retriever = HybridRetriever(base_config, embedding_model="test-model")
        assert retriever._bm25_enabled is False

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_hybrid_retriever_init(self, mock_vs_cls, hybrid_config):
        """Korrekte Initialisierung mit Hybrid-Config."""
        HybridRetriever._bm25_cache.clear()
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs

        retriever = HybridRetriever(hybrid_config, embedding_model="test-model")
        # BM25 nur aktiviert wenn rank_bm25 installiert ist
        assert retriever._rrf_k == 60
        assert retriever._bm25_cache == {}

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_rrf_fusion_basic(self, mock_vs_cls, hybrid_config):
        """RRF-Fusion mit bekannten Ranglisten liefert korrekte Reihenfolge."""
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs

        retriever = HybridRetriever(hybrid_config, embedding_model="test-model")
        retriever._rrf_k = 60

        # Semantic: DocA rank 1, DocB rank 2
        semantic = [
            {"text": "DocA text", "metadata": {"source": "a.txt"}, "score": 0.9, "_doc_id": "docA"},
            {"text": "DocB text", "metadata": {"source": "b.txt"}, "score": 0.7, "_doc_id": "docB"},
        ]
        # BM25: DocB rank 1, DocC rank 2
        bm25 = [
            {"text": "DocB text", "metadata": {"source": "b.txt"}, "score": 5.0, "_doc_id": "docB"},
            {"text": "DocC text", "metadata": {"source": "c.txt"}, "score": 3.0, "_doc_id": "docC"},
        ]

        fused = retriever._rrf_fusion_multi([semantic, bm25], k=3)

        # DocB sollte oben sein (Rang 2 in semantic + Rang 1 in BM25 → hoechster RRF-Score)
        assert len(fused) == 3
        assert fused[0]["metadata"]["source"] == "b.txt"
        # Alle haben einen rrf_score
        for doc in fused:
            assert "rrf_score" in doc
            assert doc["rrf_score"] > 0

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_rrf_fusion_one_empty(self, mock_vs_cls, hybrid_config):
        """Eine leere Rangliste — die andere Seite gewinnt."""
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs

        retriever = HybridRetriever(hybrid_config, embedding_model="test-model")
        retriever._rrf_k = 60

        semantic = [
            {"text": "Only doc", "metadata": {"source": "a.txt"}, "score": 0.9, "_doc_id": "docA"},
        ]
        bm25 = []

        fused = retriever._rrf_fusion_multi([semantic, bm25], k=5)

        assert len(fused) == 1
        assert fused[0]["metadata"]["source"] == "a.txt"
        assert "rrf_score" in fused[0]

    def test_tokenize(self):
        """Tokenisierung: lowercase, Stoppwoerter entfernt, kurze Token entfernt."""
        tokens = HybridRetriever._tokenize("Wie kann ich die Firewall-Regeln konfigurieren?")
        # 'wie', 'kann', 'ich', 'die' sind Stoppwoerter
        assert "wie" not in tokens
        assert "kann" not in tokens
        assert "ich" not in tokens
        assert "die" not in tokens
        # Fachbegriffe sollten erhalten bleiben
        assert "firewall" in tokens
        assert "regeln" in tokens
        assert "konfigurieren" in tokens

    def test_tokenize_removes_single_chars(self):
        """Einzelbuchstaben werden entfernt."""
        tokens = HybridRetriever._tokenize("a b c test")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "test" in tokens

    @patch("kiara.config.get_embedding_model", return_value="test-model")
    @patch("kiara.retrieval.retriever.VectorStore")
    def test_keyword_retrieve_no_double_overfetch(self, mock_vs_cls, _mock_emb, hybrid_config):
        """_keyword_retrieve reicht k unveraendert an Sub-Methoden weiter (kein Doppel-Overfetch)."""
        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        retriever = HybridRetriever(hybrid_config)
        filters = _FilterParams()

        # BM25-Pfad
        with patch.object(retriever, "_bm25_search", return_value=[]) as mock_bm25:
            retriever._keyword_retrieve(
                "test query", k=5, use_bm25=True, use_es=False, filters=filters
            )
            mock_bm25.assert_called_once_with("test query", k=5, filters=filters)

        # ES-Pfad
        with patch.object(retriever, "_es_search", return_value=[]) as mock_es:
            retriever._keyword_retrieve(
                "test query", k=5, use_bm25=False, use_es=True, filters=filters
            )
            mock_es.assert_called_once_with("test query", k=5, filters=filters)

    @patch("kiara.config.get_embedding_model", return_value="test-model")
    @patch("kiara.retrieval.retriever.VectorStore")
    def test_bm25_top_k_indices_applies_keyword_overfetch(
        self, mock_vs_cls, _mock_emb, hybrid_config
    ):
        """_bm25_top_k_indices liefert k*KEYWORD_OVERFETCH_FACTOR overfetchte Kandidaten."""
        mock_vs_cls.return_value = MagicMock()
        retriever = HybridRetriever(hybrid_config)
        # 30 positive Scores — mehr als k*KEYWORD_OVERFETCH_FACTOR=20
        scores = np.random.uniform(0.1, 5.0, size=30)
        result = retriever._bm25_top_k_indices(scores, k=5)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 5 * KEYWORD_OVERFETCH_FACTOR  # 20 Kandidaten
        # Indizes muessen absteigend nach Score sortiert sein
        result_scores = scores[result]
        assert all(result_scores[i] >= result_scores[i + 1] for i in range(len(result_scores) - 1))

    @patch("kiara.config.get_embedding_model", return_value="test-model")
    @patch("kiara.retrieval.retriever.VectorStore")
    def test_retrieve_rebuilds_bm25_cache_on_cache_miss(
        self,
        mock_vs_cls,
        _mock_emb,
        hybrid_config,
    ):
        """Hybrid-Retrieval baut BM25 bei Cache-Miss synchron nach."""
        HybridRetriever._bm25_cache.clear()
        HybridRetriever._bm25_building = False

        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        mock_vs_cls.return_value = mock_store

        retriever = HybridRetriever(hybrid_config)
        retriever._bm25_enabled = True
        retriever._semantic_retrieve = MagicMock(return_value=[])
        retriever._graph_retrieve = MagicMock(return_value=[])
        retriever._colbert_retrieve = MagicMock(return_value=[])
        retriever._rrf_fusion_multi = MagicMock(return_value=[])

        def populate_cache():
            HybridRetriever._bm25_cache["test"] = (
                MagicMock(),
                [],
                [],
                [],
                time.time(),
            )

        with (
            patch.object(retriever, "build_bm25_index", side_effect=populate_cache) as mock_build,
            patch.object(retriever, "_keyword_retrieve", return_value=[]) as mock_keyword,
        ):
            retriever.retrieve("test query", k=5)

        mock_build.assert_called_once()
        mock_keyword.assert_called_once()
        assert mock_keyword.call_args.args[2] is True

    def test_bm25_filters_before_top_k_window(self):
        """BM25 wendet Metadatenfilter vor dem globalen Top-K-Fenster an."""
        HybridRetriever._bm25_cache.clear()

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._store = MagicMock()
        retriever._store.get_collection_name.return_value = "test"
        retriever._chunk_ds = MagicMock()
        retriever._embed_model_name = "test-model"

        bm25 = MagicMock()
        bm25.get_scores.return_value = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        doc_ids = [f"doc{i}" for i in range(6)]
        doc_texts = [f"text {i}" for i in range(6)]
        doc_metadatas = [
            {"source": f"/docs/{i}.txt", "file_type": ".txt", "directory": "/docs"}
            for i in range(5)
        ] + [{"source": "/docs/invoice.pdf", "file_type": ".pdf", "directory": "/docs"}]
        HybridRetriever._bm25_cache["test"] = (
            bm25,
            doc_ids,
            doc_texts,
            doc_metadatas,
            time.time(),
        )

        result = retriever._bm25_search(
            "invoice",
            k=1,
            filters=_FilterParams(file_type=".pdf"),
        )

        assert [r["_doc_id"] for r in result] == ["doc5"]

    def test_bm25_datasource_filter_before_top_k_window(self):
        """BM25 wendet Datasource-Filter vor dem globalen Top-K-Fenster an."""
        HybridRetriever._bm25_cache.clear()

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever._store = MagicMock()
        retriever._store.get_collection_name.return_value = "test"
        retriever._embed_model_name = "test-model"
        retriever._chunk_ds = MagicMock()
        retriever._chunk_ds.filter_allowed.return_value = {"doc5"}

        bm25 = MagicMock()
        bm25.get_scores.return_value = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
        doc_ids = [f"doc{i}" for i in range(6)]
        doc_texts = [f"text {i}" for i in range(6)]
        doc_metadatas = [
            {"source": f"/docs/{i}.txt", "file_type": ".txt", "directory": "/docs"}
            for i in range(6)
        ]
        HybridRetriever._bm25_cache["test"] = (
            bm25,
            doc_ids,
            doc_texts,
            doc_metadatas,
            time.time(),
        )

        result = retriever._bm25_search(
            "invoice",
            k=1,
            filters=_FilterParams(datasource_ids=[2]),
        )

        assert [r["_doc_id"] for r in result] == ["doc5"]
        retriever._chunk_ds.filter_allowed.assert_called_once_with(
            doc_ids,
            [2],
            "test-model",
        )
