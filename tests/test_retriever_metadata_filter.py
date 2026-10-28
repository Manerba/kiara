"""Tests fuer Dateityp-/Verzeichnis-Filter im HybridRetriever.

Testet die in 0.0.1+186 eingefuehrte Filter-Propagation an alle
drei Retrieval-Pfade (Semantisch, BM25/ES, Graph).
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np

# Mock get_embedding_model BEFORE HybridRetriever import
_embed_model_patch = patch("kiara.config.get_embedding_model", return_value="nomic-embed-text")
_embed_model_patch.start()

from kiara.retrieval.retriever import (
    HybridRetriever,
    Retriever,
    _apply_metadata_filter,
    _FilterParams,
)

# ---------------------------------------------------------------------------
# Hilfs-Funktionen
# ---------------------------------------------------------------------------


def _make_doc(
    source, file_type=".pdf", directory="/data/docs", doc_id=None, score=0.5, chunk_index=0
):
    """Erstellt ein Test-Dokument mit Metadaten."""
    d = {
        "text": f"Text von {source}",
        "metadata": {
            "source": f"{directory}/{source}",
            "file_type": file_type,
            "directory": directory,
            "chunk_index": chunk_index,
        },
        "score": score,
    }
    if doc_id:
        d["_doc_id"] = doc_id
    return d


def _make_config(hybrid_search=True, graph_enabled=False):
    """Minimale Config fuer HybridRetriever."""
    return {
        "ollama": {"url": "http://localhost:11434", "llm_model": "test", "embedding_model": "test"},
        "vectordb": {"path": "/tmp/test_vectordb", "collection": "test", "backend": "qdrant"},
        "datasource": {"path": "/tmp", "file_types": [".txt"], "exclude_patterns": []},
        "retrieval": {"hybrid_search": hybrid_search, "rrf_k": 60},
        "graph": {"enabled": graph_enabled},
        "database": {"path": "/tmp/test.db"},
    }


# ===================================================================
# _apply_metadata_filter (Modul-Level-Funktion)
# ===================================================================


class TestApplyMetadataFilter:
    """Unit-Tests fuer die Post-hoc Metadaten-Filterfunktion."""

    def test_no_filter_returns_all(self):
        """Ohne Filter werden alle Ergebnisse unveraendert zurueckgegeben."""
        docs = [_make_doc("a.pdf"), _make_doc("b.txt", file_type=".txt")]
        result = _apply_metadata_filter(docs, _FilterParams())
        assert result == docs

    def test_filter_type_pdf(self):
        """filter_type='.pdf' entfernt nicht-PDF-Dokumente."""
        docs = [
            _make_doc("a.pdf", file_type=".pdf"),
            _make_doc("b.txt", file_type=".txt"),
            _make_doc("c.pdf", file_type=".pdf"),
        ]
        result = _apply_metadata_filter(docs, _FilterParams(file_type=".pdf"))
        assert len(result) == 2
        assert all(r["metadata"]["file_type"] == ".pdf" for r in result)

    def test_filter_type_without_dot(self):
        """Punkt-Normalisierung: 'pdf' wird zu '.pdf' via normalize()."""
        docs = [
            _make_doc("a.pdf", file_type=".pdf"),
            _make_doc("b.txt", file_type=".txt"),
        ]
        result = _apply_metadata_filter(docs, _FilterParams(file_type="pdf").normalize())
        assert len(result) == 1
        assert result[0]["metadata"]["file_type"] == ".pdf"

    def test_filter_type_case_insensitive(self):
        """Case-insensitiver Match: '.PDF' findet '.pdf'."""
        docs = [
            _make_doc("a.pdf", file_type=".pdf"),
            _make_doc("b.PDF", file_type=".PDF"),
        ]
        result = _apply_metadata_filter(docs, _FilterParams(file_type=".PDF"))
        assert len(result) == 2

    def test_filter_dir_substring(self):
        """Verzeichnis-Filter als Substring-Match."""
        docs = [
            _make_doc("a.pdf", directory="/data/invoices/2025"),
            _make_doc("b.pdf", directory="/data/contracts"),
            _make_doc("c.pdf", directory="/data/invoices/2024"),
        ]
        result = _apply_metadata_filter(docs, _FilterParams(directory="invoices"))
        assert len(result) == 2

    def test_filter_dir_case_insensitive(self):
        """Verzeichnis-Filter ist case-insensitiv."""
        docs = [_make_doc("a.pdf", directory="/Data/Invoices")]
        result = _apply_metadata_filter(docs, _FilterParams(directory="invoices"))
        assert len(result) == 1

    def test_combined_filter(self):
        """Kombination von Dateityp- und Verzeichnis-Filter."""
        docs = [
            _make_doc("a.pdf", file_type=".pdf", directory="/data/invoices"),
            _make_doc("b.txt", file_type=".txt", directory="/data/invoices"),
            _make_doc("c.pdf", file_type=".pdf", directory="/data/contracts"),
        ]
        result = _apply_metadata_filter(docs, _FilterParams(file_type=".pdf", directory="invoices"))
        assert len(result) == 1
        assert result[0]["metadata"]["source"] == "/data/invoices/a.pdf"

    def test_empty_input(self):
        """Leere Eingabeliste ergibt leere Ausgabe."""
        result = _apply_metadata_filter([], _FilterParams(file_type=".pdf"))
        assert result == []

    def test_no_matches(self):
        """Filter der nichts findet gibt leere Liste zurueck."""
        docs = [_make_doc("a.txt", file_type=".txt")]
        result = _apply_metadata_filter(docs, _FilterParams(file_type=".pdf"))
        assert result == []

    def test_missing_metadata_skipped(self):
        """Dokumente ohne file_type-Metadaten werden bei Typ-Filter uebersprungen."""
        docs = [
            {"text": "kein metadata", "metadata": {}, "score": 0.5},
            _make_doc("a.pdf", file_type=".pdf"),
        ]
        result = _apply_metadata_filter(docs, _FilterParams(file_type=".pdf"))
        assert len(result) == 1


class TestLoadChunksBySource:
    """Tests fuer das Nachladen kompletter Quellen fuer Sibling-Expansion."""

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_load_chunks_by_source_filters_datasource_ids_fail_closed(
        self, MockStore, chunk_memberships
    ):
        store = MockStore.return_value
        store.get_documents_by_source.return_value = {
            "ids": ["c1", "c2", "c3"],
            "documents": ["DS 7", "DS 8", "ohne DS"],
            "metadatas": [
                {"source": "doc.txt", "ds_ids": [7], "chunk_index": 0},
                {"source": "doc.txt", "ds_ids": [8], "chunk_index": 1},
                {"source": "doc.txt", "chunk_index": 2},
            ],
        }

        retriever = Retriever(_make_config(), embedding_model="test")
        ds = chunk_memberships(retriever, "c1")
        result = retriever.load_chunks_by_source(
            "doc.txt",
            filter_datasource_ids=[ds],
        )

        assert [r["text"] for r in result] == ["DS 7"]
        assert result[0]["_doc_id"] == "c1"


# ===================================================================
# _bm25_collect_candidates mit Metadaten-Filter
# ===================================================================


class TestBM25CollectCandidatesFilter:
    """Tests fuer filter_type/filter_dir in _bm25_collect_candidates."""

    def _make_scores_and_metadata(self):
        """Erzeugt BM25-Scores und zugehoerige Metadaten."""
        scores = np.array([5.0, 3.0, 4.0, 2.0, 0.0])
        metadatas = [
            {"source": "/docs/report.pdf", "file_type": ".pdf", "directory": "/docs"},
            {"source": "/mail/msg.txt", "file_type": ".txt", "directory": "/mail"},
            {"source": "/docs/invoice.pdf", "file_type": ".pdf", "directory": "/docs"},
            {"source": "/data/sheet.xlsx", "file_type": ".xlsx", "directory": "/data"},
            {"source": "/docs/empty.pdf", "file_type": ".pdf", "directory": "/docs"},
        ]
        # Sortierte Indizes (absteigend nach Score)
        top_indices = np.array([0, 2, 1, 3])
        return top_indices, scores, metadatas

    def test_no_filter(self):
        """Ohne Filter werden alle Kandidaten mit Score > 0 zurueckgegeben."""
        top_indices, scores, metadatas = self._make_scores_and_metadata()
        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores,
            metadatas,
            _FilterParams(),
        )
        assert len(candidates) == 4

    def test_filter_type_pdf(self):
        """filter_type='.pdf' entfernt nicht-PDF-Kandidaten."""
        top_indices, scores, metadatas = self._make_scores_and_metadata()
        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores,
            metadatas,
            _FilterParams(file_type=".pdf"),
        )
        assert len(candidates) == 2
        for _, _, meta in candidates:
            assert meta["file_type"] == ".pdf"

    def test_filter_type_without_dot(self):
        """Punkt-Normalisierung in BM25-Filterung."""
        top_indices, scores, metadatas = self._make_scores_and_metadata()
        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores,
            metadatas,
            _FilterParams(file_type="pdf").normalize(),
        )
        assert len(candidates) == 2

    def test_filter_dir(self):
        """filter_dir filtert nach Verzeichnis-Substring."""
        top_indices, scores, metadatas = self._make_scores_and_metadata()
        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores,
            metadatas,
            _FilterParams(directory="/docs"),
        )
        assert len(candidates) == 2
        for _, _, meta in candidates:
            assert "/docs" in meta["directory"]

    def test_combined_filter_type_and_dir(self):
        """Kombination beider Filter."""
        top_indices, scores, metadatas = self._make_scores_and_metadata()
        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores,
            metadatas,
            _FilterParams(file_type=".pdf", directory="/docs"),
        )
        assert len(candidates) == 2

    def test_combined_with_source_filter(self):
        """Alle drei Filter gleichzeitig."""
        top_indices, scores, metadatas = self._make_scores_and_metadata()
        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores,
            metadatas,
            _FilterParams(source="invoice", file_type=".pdf"),
        )
        assert len(candidates) == 1
        assert "invoice" in candidates[0][2]["source"]

    def test_filter_type_case_insensitive(self):
        """Case-insensitiver Dateityp-Match in BM25."""
        scores = np.array([5.0])
        metadatas = [{"source": "x.PDF", "file_type": ".PDF", "directory": "/"}]
        top_indices = np.array([0])
        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores,
            metadatas,
            _FilterParams(file_type=".pdf"),
        )
        assert len(candidates) == 1


# ===================================================================
# Integration: retrieve() propagiert Filter an alle Pfade
# ===================================================================


class TestRetrieveFilterPropagation:
    """Integrations-Tests: filter_type/filter_dir erreichen alle Retrieval-Pfade."""

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_semantic_only_uses_where_filter(self, mock_vs_cls, chunk_memberships):
        """Ohne BM25/ES: filter_type geht via where-Filter an VectorStore.query()."""
        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        mock_store.query.return_value = [_make_doc("a.pdf", doc_id="d1")]
        mock_vs_cls.return_value = mock_store

        config = _make_config(hybrid_search=False)
        ret = HybridRetriever(config)
        chunk_memberships(ret, "d1")

        ret.retrieve("test query", k=5, filter_type=".pdf")

        # VectorStore.query muss mit where-Filter aufgerufen worden sein
        call_kwargs = mock_store.query.call_args
        where = call_kwargs.kwargs.get("where") or call_kwargs[1].get("where")
        assert where is not None
        assert where.get("file_type") == ".pdf"

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_bm25_receives_filter_type(self, mock_vs_cls):
        """BM25-Pfad empfaengt filter_type und filter_dir."""
        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        mock_store.query.return_value = []
        mock_vs_cls.return_value = mock_store

        config = _make_config(hybrid_search=True)
        ret = HybridRetriever(config)
        ret._bm25_enabled = True
        ret._bm25_cache["test"] = (MagicMock(), [], [], [], time.time())

        with patch.object(ret, "_bm25_search", return_value=[]) as mock_bm25:
            ret.retrieve("test", k=5, filter_type=".pdf", filter_dir="/docs")

        mock_bm25.assert_called_once()
        call_kwargs = mock_bm25.call_args
        fp = call_kwargs.kwargs.get("filters")
        assert fp.file_type == ".pdf"
        assert fp.directory == "/docs"

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_graph_receives_filter(self, mock_vs_cls, chunk_memberships):
        """Graph-Pfad wendet Metadaten-Filter an."""
        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        mock_store.query.return_value = []
        mock_vs_cls.return_value = mock_store

        config = _make_config(hybrid_search=True, graph_enabled=True)

        with patch("kiara.graph.store.GraphStore"):
            with patch("kiara.graph.retriever.GraphRetriever") as MockGR:
                mock_graph = MagicMock()
                mock_graph._max_chunks = 20
                # Graph liefert pdf + txt, nur pdf soll durchkommen
                mock_graph.retrieve.return_value = [
                    _make_doc("a.pdf", file_type=".pdf", doc_id="d1"),
                    _make_doc("b.txt", file_type=".txt", doc_id="d2"),
                ]
                MockGR.return_value = mock_graph

                ret = HybridRetriever(config)
                ret._graph_enabled = True
                ret._graph_retriever = mock_graph
                ret._bm25_enabled = True
                ret._bm25_cache["test"] = (MagicMock(), [], [], [], time.time())

        chunk_memberships(ret, "d1", "d2")
        with patch.object(ret, "_bm25_search", return_value=[]):
            results = ret.retrieve("test", k=5, filter_type=".pdf")

        # Graph-Ergebnisse muessen nach file_type gefiltert sein
        graph_sources = [r for r in results if r["metadata"].get("file_type") == ".txt"]
        assert len(graph_sources) == 0

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_no_filter_passes_none(self, mock_vs_cls, chunk_memberships):
        """Ohne Filter werden None-Werte durchgereicht (kein Filter aktiv)."""
        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        mock_store.query.return_value = [_make_doc("a.pdf", doc_id="d1")]
        mock_vs_cls.return_value = mock_store

        config = _make_config(hybrid_search=False)
        ret = HybridRetriever(config)
        chunk_memberships(ret, "d1")

        ret.retrieve("test query", k=5)

        call_kwargs = mock_store.query.call_args
        where = call_kwargs.kwargs.get("where") or call_kwargs[1].get("where")
        assert where is None

    @patch("kiara.retrieval.retriever.VectorStore")
    def test_full_hybrid_filters_all_paths(self, mock_vs_cls, chunk_memberships):
        """End-to-End: Alle 3 Pfade respektieren filter_type='.pdf'."""
        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        # Semantic: liefert gemischte Ergebnisse (Qdrant filtert nativ)
        mock_store.query.return_value = [
            _make_doc("sem.pdf", file_type=".pdf", doc_id="d1"),
        ]
        mock_vs_cls.return_value = mock_store

        config = _make_config(hybrid_search=True, graph_enabled=True)

        with patch("kiara.graph.store.GraphStore"):
            with patch("kiara.graph.retriever.GraphRetriever") as MockGR:
                mock_graph = MagicMock()
                mock_graph._max_chunks = 20
                mock_graph.retrieve.return_value = [
                    _make_doc("graph.pdf", file_type=".pdf", doc_id="d3"),
                    _make_doc("graph.txt", file_type=".txt", doc_id="d4"),
                ]
                MockGR.return_value = mock_graph

                ret = HybridRetriever(config)
                ret._graph_enabled = True
                ret._graph_retriever = mock_graph
                ret._bm25_enabled = True
                ret._bm25_cache["test"] = (MagicMock(), [], [], [], time.time())

        # BM25 liefert nur PDFs (weil _bm25_collect_candidates filtert)
        bm25_results = [_make_doc("bm25.pdf", file_type=".pdf", doc_id="d2")]
        chunk_memberships(ret, "d1", "d2", "d3", "d4")
        with patch.object(ret, "_bm25_search", return_value=bm25_results):
            results = ret.retrieve("test", k=10, filter_type=".pdf")

        # Kein einziges Ergebnis darf .txt sein
        for r in results:
            assert r["metadata"]["file_type"] != ".txt", (
                f"Ungefilterte .txt-Datei in Ergebnis: {r['metadata']['source']}"
            )
        assert len(results) >= 2  # mindestens semantic + bm25
