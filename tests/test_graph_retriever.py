"""Tests fuer GraphRetriever."""

from unittest.mock import MagicMock, patch

import pytest

from kiara.graph.retriever import GraphRetriever
from kiara.graph.store import GraphStore


@pytest.fixture
def graph_store():
    """Erstellt einen GraphStore (PostgreSQL, via conftest._init_pg_backend)."""
    store = GraphStore()
    store.clear()
    yield store
    store.clear()


@pytest.fixture
def config():
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
        },
        "vectordb": {"path": "/tmp/test-vectordb", "collection": "test"},
        "datasource": {"path": "/tmp", "file_types": [".txt"], "exclude_patterns": []},
        "graph": {"max_traversal_depth": 2, "max_chunks": 20},
    }


def _mock_vectordb_get(ids=None, include=None):
    """Simuliert VectorStore.get_documents_by_ids()."""
    docs = []
    metas = []
    valid_ids = []
    for cid in ids:
        if cid.startswith("chunk"):
            valid_ids.append(cid)
            docs.append(f"Inhalt von {cid}")
            metas.append({"source": f"/path/to/{cid}.txt", "chunk_index": 0, "datasource_id": 1})
    return {"ids": valid_ids, "documents": docs, "metadatas": metas}


def test_retrieve_finds_related_chunks(config, graph_store):
    """Query mit bekannter Entity -> Chunks via Graph."""
    # Setup graph
    graph_store.add_node("supplier:mueller-gmbh", "supplier", "Mueller GmbH")
    graph_store.add_node("article:art-10001", "article", "Aluminium Gehaeuse")
    graph_store.add_edge("supplier:mueller-gmbh", "article:art-10001", "liefert", chunk_id="chunk1")

    retriever = GraphRetriever(config, graph_store)

    with patch("kiara.retrieval.vectorstore.VectorStore") as MockVS:
        mock_store = MagicMock()
        mock_store.get_documents_by_ids = _mock_vectordb_get
        MockVS.return_value = mock_store

        results = retriever.retrieve("Mueller GmbH")

    assert len(results) > 0
    assert results[0]["text"] == "Inhalt von chunk1"


def test_retrieve_empty_on_unknown_entity(config, graph_store):
    """Unbekannte Entity -> leere Liste."""
    retriever = GraphRetriever(config, graph_store)
    results = retriever.retrieve("XYZ Unbekannte Firma 12345")
    assert results == []


def test_scoring_by_distance(config, graph_store):
    """0-Hop > 1-Hop > 2-Hop Score."""
    # Chain: A -> B -> C, each with a chunk
    graph_store.add_node("topic:netzwerk", "topic", "Netzwerk")
    graph_store.add_node("topic:router", "topic", "Router")
    graph_store.add_node("topic:firewall", "topic", "Firewall")
    graph_store.add_edge("topic:netzwerk", "topic:router", "betrifft", chunk_id="chunk_hop0")
    graph_store.add_edge("topic:router", "topic:firewall", "betrifft", chunk_id="chunk_hop1")

    retriever = GraphRetriever(config, graph_store)

    with patch("kiara.retrieval.vectorstore.VectorStore") as MockVS:
        mock_store = MagicMock()
        mock_store.get_documents_by_ids = _mock_vectordb_get
        MockVS.return_value = mock_store

        results = retriever.retrieve("Netzwerk")

    # Results should exist
    assert len(results) >= 1
    # First result should have higher score (closer to query entity)
    if len(results) >= 2:
        assert results[0]["score"] >= results[1]["score"]


def test_datasource_filter(config, graph_store):
    """GraphRetriever liefert alle Chunks — DS-Filterung passiert im HybridRetriever."""
    graph_store.add_node("topic:test", "topic", "Testthema")
    graph_store.add_edge("topic:test", "topic:test", "betrifft", chunk_id="chunk_ds1")

    retriever = GraphRetriever(config, graph_store)

    with patch("kiara.retrieval.vectorstore.VectorStore") as MockVS:
        mock_store = MagicMock()
        mock_store.get_documents_by_ids = _mock_vectordb_get
        MockVS.return_value = mock_store

        # GraphRetriever gibt alle Chunks zurueck (kein DS-Filter)
        results = retriever.retrieve("Testthema")

    # Chunks kommen ungefiltert zurueck — Filterung macht HybridRetriever
    assert len(results) >= 1


def test_max_chunks_limits_loaded_chunk_ids(config, graph_store):
    """graph.max_chunks begrenzt die VectorStore-Punkt-Lookups."""
    config["graph"]["max_chunks"] = 2
    graph_store.add_node("topic:test", "topic", "Testthema")
    for chunk_id in ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]:
        graph_store.link_node_to_chunk("topic:test", chunk_id)

    retriever = GraphRetriever(config, graph_store)

    with patch("kiara.retrieval.vectorstore.VectorStore") as MockVS:
        mock_store = MagicMock()
        mock_store.get_documents_by_ids = MagicMock(side_effect=_mock_vectordb_get)
        MockVS.return_value = mock_store

        results = retriever.retrieve("Testthema", k=5)

    loaded_ids = mock_store.get_documents_by_ids.call_args.kwargs["ids"]
    assert loaded_ids == ["chunk1", "chunk2"]
    assert [row["_doc_id"] for row in results] == ["chunk1", "chunk2"]


def test_collect_chunk_candidates_batches_graph_lookup(config):
    """Chunk-Kandidaten werden fuer alle Nodes in einem DB-Call gesammelt."""
    graph_store = MagicMock(spec=GraphStore)
    graph_store.get_connected_chunks_by_node.return_value = [
        ("node:a", "chunk-a"),
        ("node:b", "chunk-shared"),
        ("node:c", "chunk-shared"),
    ]
    retriever = GraphRetriever(config, graph_store)

    chunk_ids, chunk_depth = retriever._collect_chunk_candidates(
        {
            "node:a": 2,
            "node:b": 1,
            "node:c": 0,
        }
    )

    graph_store.get_connected_chunks_by_node.assert_called_once_with(
        [
            "node:c",
            "node:b",
            "node:a",
        ]
    )
    graph_store.get_connected_chunks.assert_not_called()
    assert chunk_ids == ["chunk-shared", "chunk-a"]
    assert chunk_depth == {"chunk-shared": 0, "chunk-a": 2}


def test_entity_search_terms_are_deduped_and_capped(config):
    """Entity-Search feuert keine unlimitierte Folge roher Query-Tokens ab."""
    graph_store = MagicMock(spec=GraphStore)
    graph_store.find_nodes.return_value = []
    retriever = GraphRetriever(config, graph_store)

    retriever._match_entities(
        ["alpha", "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota"],
        "complete phrase",
    )

    queried_terms = [call.args[0] for call in graph_store.find_nodes.call_args_list]
    assert queried_terms == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "complete phrase",
    ]


def test_retrieve_empty_query(config, graph_store):
    """Leere Query -> leere Liste."""
    retriever = GraphRetriever(config, graph_store)
    results = retriever.retrieve("")
    assert results == []
