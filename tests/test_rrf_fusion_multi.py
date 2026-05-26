"""Tests fuer _rrf_fusion_multi in HybridRetriever."""

from unittest.mock import patch

import pytest

# Mock get_embedding_model BEFORE HybridRetriever import
_embed_model_patch = patch("kiara.config.get_embedding_model", return_value="nomic-embed-text")
_embed_model_patch.start()

from kiara.retrieval.retriever import HybridRetriever


@pytest.fixture
def retriever():
    """Erstellt einen HybridRetriever mit minimaler Config."""
    config = {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
        },
        "vectordb": {"path": "/tmp/test-vectordb", "collection": "test"},
        "datasource": {"path": "/tmp", "file_types": [".txt"], "exclude_patterns": []},
        "retrieval": {"hybrid_search": False, "rrf_k": 60},
    }
    with patch("kiara.retrieval.retriever.VectorStore"):
        return HybridRetriever(config)


def _make_doc(source: str, chunk_index: int, score: float = 0.5, doc_id: str = None) -> dict:
    """Erstellt ein Test-Dokument."""
    d = {
        "text": f"Text von {source}:{chunk_index}",
        "metadata": {"source": source, "chunk_index": chunk_index},
        "score": score,
    }
    if doc_id:
        d["_doc_id"] = doc_id
    return d


def test_fusion_three_sources(retriever):
    """Drei Listen fusionieren, Top-K korrekt."""
    list1 = [_make_doc("a.txt", 0, doc_id="doc1"), _make_doc("b.txt", 0, doc_id="doc2")]
    list2 = [_make_doc("b.txt", 0, doc_id="doc2"), _make_doc("c.txt", 0, doc_id="doc3")]
    list3 = [_make_doc("c.txt", 0, doc_id="doc3"), _make_doc("a.txt", 0, doc_id="doc1")]

    results = retriever._rrf_fusion_multi([list1, list2, list3], k=3)

    assert len(results) == 3
    # All three docs should appear
    result_sources = {r["metadata"]["source"] for r in results}
    assert result_sources == {"a.txt", "b.txt", "c.txt"}
    # All should have rrf_score
    for r in results:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0


def test_fusion_empty_source(retriever):
    """Eine leere Liste → andere funktionieren."""
    list1 = [_make_doc("a.txt", 0, doc_id="doc1")]
    list2 = []
    list3 = [_make_doc("a.txt", 0, doc_id="doc1")]

    results = retriever._rrf_fusion_multi([list1, list2, list3], k=5)

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "a.txt"
    # Score should be from list1 (rank=0) + list3 (rank=0) = 2/(60+1) = 2/61
    expected_score = round(2.0 / 61.0, 6)
    assert results[0]["rrf_score"] == expected_score


def test_fusion_backward_compatible(retriever):
    """Zwei Listen = gleiches Ergebnis wie _rrf_fusion."""
    list1 = [
        _make_doc("a.txt", 0, doc_id="doc1"),
        _make_doc("b.txt", 0, doc_id="doc2"),
    ]
    list2 = [
        _make_doc("b.txt", 0, doc_id="doc2"),
        _make_doc("c.txt", 0, doc_id="doc3"),
    ]

    multi_results = retriever._rrf_fusion_multi([list1, list2], k=3)

    # doc2 appears in both lists at positions [1, 0] → highest RRF
    # doc1 appears only in list1 at position [0] → lower
    # doc3 appears only in list2 at position [1] → lower
    assert len(multi_results) == 3
    assert multi_results[0]["metadata"]["source"] == "b.txt"  # highest: 1/62 + 1/61


def test_fusion_respects_k_limit(retriever):
    """k-Limit wird eingehalten."""
    list1 = [_make_doc("a.txt", i, doc_id=f"doc{i}") for i in range(10)]

    results = retriever._rrf_fusion_multi([list1], k=3)
    assert len(results) == 3


def test_fusion_all_empty(retriever):
    """Alle Listen leer → leeres Ergebnis."""
    results = retriever._rrf_fusion_multi([[], [], []], k=5)
    assert results == []


def test_fusion_doc_id_removed(retriever):
    """_doc_id wird aus Ergebnis entfernt."""
    list1 = [_make_doc("a.txt", 0, doc_id="doc1")]

    results = retriever._rrf_fusion_multi([list1], k=5)
    assert "_doc_id" not in results[0]


def test_fusion_score_ordering(retriever):
    """Docs die in mehreren Listen vorkommen haben hoeheren Score."""
    # doc1 in allen 3 Listen, doc2 nur in 1
    list1 = [_make_doc("a.txt", 0, doc_id="doc1"), _make_doc("b.txt", 0, doc_id="doc2")]
    list2 = [_make_doc("a.txt", 0, doc_id="doc1")]
    list3 = [_make_doc("a.txt", 0, doc_id="doc1")]

    results = retriever._rrf_fusion_multi([list1, list2, list3], k=5)

    # doc1 should be first (appears in 3 lists)
    assert results[0]["metadata"]["source"] == "a.txt"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
