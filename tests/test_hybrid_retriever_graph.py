"""Tests fuer HybridRetriever mit GraphRAG-Integration (3-Quellen-Retrieval)."""

import time
from unittest.mock import MagicMock, patch

import pytest

from kiara.retrieval.retriever import HybridRetriever


def _make_doc(source, chunk_index, score=0.5, doc_id=None):
    """Erstellt ein Test-Dokument im Retriever-Format."""
    d = {
        "text": f"Text von {source}:{chunk_index}",
        "metadata": {"source": source, "chunk_index": chunk_index},
        "score": score,
    }
    if doc_id:
        d["_doc_id"] = doc_id
    return d


def _make_config(graph_enabled=False, hybrid_search=True):
    """Erstellt eine minimale Config."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
        },
        "vectordb": {"path": "/tmp/test-vectordb", "collection": "test", "backend": "qdrant"},
        "datasource": {"path": "/tmp", "file_types": [".txt"], "exclude_patterns": []},
        "retrieval": {"hybrid_search": hybrid_search, "rrf_k": 60},
        "graph": {
            "enabled": graph_enabled,
            "max_traversal_depth": 2,
            "max_chunks": 20,
        },
        "database": {"path": "/tmp/test.db"},
    }


@pytest.fixture
def retriever_with_graph():
    """HybridRetriever mit aktiviertem GraphRAG (GraphRetriever gemockt)."""
    config = _make_config(graph_enabled=True, hybrid_search=True)

    with patch("kiara.retrieval.retriever.VectorStore") as mock_vs:
        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        mock_vs.return_value = mock_store

        with patch("kiara.graph.store.GraphStore"):
            with patch("kiara.graph.retriever.GraphRetriever") as MockGR:
                mock_graph_retriever = MagicMock()
                mock_graph_retriever._max_chunks = 20
                MockGR.return_value = mock_graph_retriever

                ret = HybridRetriever(config, embedding_model="nomic-embed-text")
                # Sicherstellen, dass Graph aktiviert ist
                ret._graph_enabled = True
                ret._graph_retriever = mock_graph_retriever

                yield ret, mock_graph_retriever


@pytest.fixture
def retriever_without_graph():
    """HybridRetriever ohne GraphRAG."""
    config = _make_config(graph_enabled=False, hybrid_search=True)

    with patch("kiara.retrieval.retriever.VectorStore") as mock_vs:
        mock_store = MagicMock()
        mock_store.get_collection_name.return_value = "test"
        mock_vs.return_value = mock_store

        ret = HybridRetriever(config, embedding_model="nomic-embed-text")
        yield ret


def test_hybrid_with_graph_enabled(retriever_with_graph):
    """Wenn graph.enabled=True, wird GraphRetriever.retrieve() aufgerufen
    und Ergebnisse in die RRF-Fusion einbezogen."""
    ret, mock_graph = retriever_with_graph

    semantic_results = [_make_doc("a.txt", 0, doc_id="doc1")]
    bm25_results = [_make_doc("b.txt", 0, doc_id="doc2")]
    graph_results = [_make_doc("c.txt", 0, doc_id="doc3")]

    mock_graph.retrieve.return_value = graph_results

    # Mock semantic (store.query) und BM25
    with patch.object(ret, "_bm25_search", return_value=bm25_results):
        with patch.object(ret._store, "query", return_value=semantic_results):
            # BM25-Index muss vorhanden sein
            ret._bm25_enabled = True
            ret._bm25_cache["test"] = (MagicMock(), [], [], [], time.time())

            results = ret.retrieve("Testabfrage", k=5)

    # GraphRetriever muss aufgerufen worden sein
    mock_graph.retrieve.assert_called_once()

    # Alle 3 Quellen muessen im Ergebnis vertreten sein
    result_sources = {r["metadata"]["source"] for r in results}
    assert "a.txt" in result_sources  # semantic
    assert "b.txt" in result_sources  # BM25
    assert "c.txt" in result_sources  # graph


def test_hybrid_without_graph(retriever_without_graph):
    """Wenn graph.enabled=False, wird nur semantic + BM25 fusioniert."""
    ret = retriever_without_graph

    assert ret._graph_enabled is False
    assert ret._graph_retriever is None

    semantic_results = [_make_doc("a.txt", 0, doc_id="doc1")]
    bm25_results = [_make_doc("b.txt", 0, doc_id="doc2")]

    with patch.object(ret, "_bm25_search", return_value=bm25_results):
        with patch.object(ret._store, "query", return_value=semantic_results):
            ret._bm25_enabled = True
            ret._bm25_cache["test"] = (MagicMock(), [], [], [], time.time())

            results = ret.retrieve("Testabfrage", k=5)

    result_sources = {r["metadata"]["source"] for r in results}
    assert "a.txt" in result_sources
    assert "b.txt" in result_sources
    assert len(results) == 2


def test_graph_failure_doesnt_break_retrieval(retriever_with_graph):
    """Wenn GraphRetriever eine Exception wirft, liefert der HybridRetriever
    trotzdem semantic + BM25 Ergebnisse."""
    ret, mock_graph = retriever_with_graph

    semantic_results = [_make_doc("a.txt", 0, doc_id="doc1")]
    bm25_results = [_make_doc("b.txt", 0, doc_id="doc2")]

    # GraphRetriever wirft Exception
    mock_graph.retrieve.side_effect = Exception("Graph-DB nicht erreichbar")

    with patch.object(ret, "_bm25_search", return_value=bm25_results):
        with patch.object(ret._store, "query", return_value=semantic_results):
            ret._bm25_enabled = True
            ret._bm25_cache["test"] = (MagicMock(), [], [], [], time.time())

            results = ret.retrieve("Testabfrage", k=5)

    # Ergebnis darf nicht leer sein
    assert len(results) >= 2
    result_sources = {r["metadata"]["source"] for r in results}
    assert "a.txt" in result_sources
    assert "b.txt" in result_sources


def test_doc_in_all_three_sources_gets_highest_score():
    """Ein Dokument das in semantic, BM25 UND Graph auftaucht
    hat den hoechsten RRF-Score."""
    config = _make_config(graph_enabled=False)
    with patch("kiara.retrieval.retriever.VectorStore"):
        ret = HybridRetriever(config, embedding_model="nomic-embed-text")

    # doc1 in allen 3 Listen, doc2 nur in einer
    list1 = [_make_doc("shared.txt", 0, doc_id="doc1")]
    list2 = [_make_doc("shared.txt", 0, doc_id="doc1")]
    list3 = [
        _make_doc("shared.txt", 0, doc_id="doc1"),
        _make_doc("only_graph.txt", 0, doc_id="doc2"),
    ]

    results = ret._rrf_fusion_multi([list1, list2, list3], k=5)

    # doc1 muss an erster Stelle stehen (3x vorhanden)
    assert results[0]["metadata"]["source"] == "shared.txt"
    if len(results) > 1:
        assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_graph_only_results_appear_in_output():
    """Dokumente die NUR vom Graph gefunden werden (nicht von semantic/BM25)
    erscheinen trotzdem im Ergebnis."""
    config = _make_config(graph_enabled=False)
    with patch("kiara.retrieval.retriever.VectorStore"):
        ret = HybridRetriever(config, embedding_model="nomic-embed-text")

    semantic = [_make_doc("a.txt", 0, doc_id="doc1")]
    bm25 = [_make_doc("b.txt", 0, doc_id="doc2")]
    graph = [_make_doc("graph_only.txt", 0, doc_id="doc3")]

    results = ret._rrf_fusion_multi([semantic, bm25, graph], k=5)

    result_sources = {r["metadata"]["source"] for r in results}
    assert "graph_only.txt" in result_sources
