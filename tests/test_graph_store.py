"""Tests fuer GraphStore."""

import pytest

from kiara.graph.store import GraphStore


@pytest.fixture
def graph_store():
    """Erstellt einen GraphStore (PostgreSQL, via conftest._init_pg_backend)."""
    store = GraphStore()
    store.clear()
    yield store
    store.clear()


def test_add_node_and_retrieve(graph_store):
    """Knoten hinzufuegen und per find_nodes finden."""
    graph_store.add_node(
        "customer:k-1001", "customer", "Mueller GmbH", {"city": "Stuttgart"}, source="erp_import"
    )
    results = graph_store.find_nodes("Mueller")
    assert len(results) == 1
    assert results[0]["id"] == "customer:k-1001"
    assert results[0]["entity_type"] == "customer"
    assert results[0]["label"] == "Mueller GmbH"
    assert results[0]["properties"]["city"] == "Stuttgart"
    assert results[0]["source"] == "erp_import"


def test_find_nodes_matches_id_and_properties(graph_store):
    """Entity-Suche findet ERP-IDs auch ohne ID im Label."""
    graph_store.add_node(
        "article:art-10001",
        "article",
        "Aluminium Gehaeuse",
        {"sku": "ART-10001", "external_id": "A_10001"},
    )

    by_id = graph_store.find_nodes("ART-10001")
    assert len(by_id) == 1
    assert by_id[0]["id"] == "article:art-10001"

    by_property = graph_store.find_nodes("A_10001")
    assert len(by_property) == 1
    assert by_property[0]["id"] == "article:art-10001"


def test_find_nodes_escapes_like_wildcards(graph_store):
    """LIKE-Sonderzeichen in Suchbegriffen werden literal behandelt."""
    graph_store.add_node("topic:literal", "topic", "A_B Literal")
    graph_store.add_node("topic:wildcard", "topic", "ACB Wildcard")

    results = graph_store.find_nodes("A_B")
    assert [row["id"] for row in results] == ["topic:literal"]


def test_add_edge_and_neighbors(graph_store):
    """Kante hinzufuegen, get_neighbors pruefen."""
    graph_store.add_node("supplier:alu-form", "supplier", "Alu-Form GmbH")
    graph_store.add_node("article:art-10001", "article", "Aluminium-Gehaeuse Typ A")
    graph_store.add_edge("supplier:alu-form", "article:art-10001", "liefert")

    subgraph = graph_store.get_neighbors("supplier:alu-form", max_depth=1)
    assert len(subgraph["nodes"]) == 2
    assert len(subgraph["edges"]) == 1
    assert subgraph["edges"][0]["relation"] == "liefert"


def test_add_edge_rejects_missing_nodes(graph_store):
    """Kanten duerfen nie ohne beide Endknoten persistiert werden."""
    graph_store.add_node("n:a", "topic", "A")

    with pytest.raises(ValueError, match="fehlende Nodes"):
        graph_store.add_edge("n:a", "n:missing", "betrifft")

    assert graph_store.get_stats()["total_edges"] == 0


def test_get_connected_chunks(graph_store):
    """Chunks ueber Kanten-chunk_id finden."""
    graph_store.add_node("topic:netzwerk", "topic", "Netzwerk")
    graph_store.add_node("chunk:abc123", "chunk", "Chunk ABC")
    graph_store.add_edge("topic:netzwerk", "chunk:abc123", "erwaehnt_in", chunk_id="abc123")

    chunks = graph_store.get_connected_chunks(["topic:netzwerk"])
    assert "abc123" in chunks


def test_get_connected_chunks_by_node_batches_pairs(graph_store):
    """Batch-Abfrage liefert Node/Chunk-Paare fuer GraphRetriever."""
    graph_store.add_node("topic:a", "topic", "A")
    graph_store.add_node("topic:b", "topic", "B")
    graph_store.add_edge("topic:a", "topic:b", "betrifft", chunk_id="chunk-edge")
    graph_store.link_node_to_chunk("topic:a", "chunk-node")

    pairs = set(graph_store.get_connected_chunks_by_node(["topic:a", "topic:b"]))

    assert ("topic:a", "chunk-edge") in pairs
    assert ("topic:b", "chunk-edge") in pairs
    assert ("topic:a", "chunk-node") in pairs


def test_delete_by_chunk_id(graph_store):
    """Kante loeschen, verwaisten Knoten pruefen."""
    graph_store.add_node("topic:test", "topic", "Testthema", source="extraction")
    graph_store.add_node("chunk:c1", "chunk", "Chunk 1", source="extraction")
    graph_store.add_edge("topic:test", "chunk:c1", "erwaehnt_in", chunk_id="c1")

    # Before deletion
    assert graph_store.get_stats()["total_edges"] == 1

    graph_store.delete_by_chunk_id("c1")

    stats = graph_store.get_stats()
    assert stats["total_edges"] == 0
    # Orphaned extraction nodes should be deleted
    assert stats["total_nodes"] == 0


def test_clear_by_source(graph_store):
    """Nur ERP-Knoten loeschen, Extractions bleiben."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH", source="erp_import")
    graph_store.add_node("topic:netzwerk", "topic", "Netzwerk", source="extraction")

    deleted = graph_store.clear(source="erp_import")
    assert deleted == 1

    stats = graph_store.get_stats()
    assert stats["total_nodes"] == 1
    results = graph_store.find_nodes("Netzwerk")
    assert len(results) == 1


def test_clear_by_source_removes_source_edges_even_when_nodes_are_extraction(graph_store):
    """ERP-Import-Kanten werden auch bei existing-node-Konflikten entfernt."""
    graph_store.add_node("customer:k-1", "customer", "Kunde", source="extraction")
    graph_store.add_node("order:a-1", "order", "Auftrag", source="extraction")
    graph_store.add_edge(
        "order:a-1",
        "customer:k-1",
        "bestellt_bei",
        source="erp_import",
    )

    deleted = graph_store.clear(source="erp_import")

    assert deleted == 0
    assert graph_store.get_stats()["total_edges"] == 0
    assert graph_store.get_stats()["total_nodes"] == 2


def test_delete_by_chunk_id_keeps_unchunked_manual_extraction_edge(graph_store):
    """Chunk-Cleanup loescht keine nicht betroffenen manuellen Extraction-Edges."""
    graph_store.add_node("topic:a", "topic", "A", source="extraction")
    graph_store.add_node("topic:b", "topic", "B", source="extraction")
    graph_store.add_node("topic:c", "topic", "C", source="extraction")
    graph_store.add_edge("topic:a", "topic:b", "betrifft", chunk_id="chunk-1")
    graph_store.add_edge("topic:b", "topic:c", "betrifft", source="extraction")

    graph_store.delete_by_chunk_id("chunk-1")

    assert graph_store.get_stats()["total_edges"] == 1
    assert graph_store.node_exists("topic:b")
    assert graph_store.node_exists("topic:c")


def test_stats(graph_store):
    """Statistiken pruefen."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH", source="erp_import")
    graph_store.add_node("supplier:alu-form", "supplier", "Alu-Form GmbH", source="erp_import")
    graph_store.add_node("topic:test", "topic", "Test", source="extraction")
    graph_store.add_edge("supplier:alu-form", "customer:k-1001", "liefert")

    stats = graph_store.get_stats()
    assert stats["total_nodes"] == 3
    assert stats["total_edges"] == 1
    assert stats["nodes_by_type"]["customer"] == 1
    assert stats["nodes_by_type"]["supplier"] == 1
    assert stats["nodes_by_type"]["topic"] == 1
    assert stats["edges_by_relation"]["liefert"] == 1
    assert stats["sources"]["erp_import"] == 2
    assert stats["sources"]["extraction"] == 1


def test_upsert_node(graph_store):
    """Gleiche ID → Update, nicht Duplikat."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH", {"city": "Stuttgart"})
    graph_store.add_node(
        "customer:k-1001", "customer", "Mueller Maschinenbau GmbH", {"city": "Boeblingen"}
    )

    results = graph_store.find_nodes("Mueller")
    assert len(results) == 1
    assert results[0]["label"] == "Mueller Maschinenbau GmbH"
    assert results[0]["properties"]["city"] == "Boeblingen"


def test_bfs_max_depth(graph_store):
    """Tiefenbegrenzung bei get_neighbors."""
    # Chain: A -> B -> C -> D
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    graph_store.add_node("n:c", "topic", "C")
    graph_store.add_node("n:d", "topic", "D")
    graph_store.add_edge("n:a", "n:b", "betrifft")
    graph_store.add_edge("n:b", "n:c", "betrifft")
    graph_store.add_edge("n:c", "n:d", "betrifft")

    # Depth 1: should find A and B only
    subgraph = graph_store.get_neighbors("n:a", max_depth=1)
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert "n:a" in node_ids
    assert "n:b" in node_ids
    assert "n:c" not in node_ids
    assert "n:d" not in node_ids

    # Depth 2: should find A, B, C
    subgraph = graph_store.get_neighbors("n:a", max_depth=2)
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert "n:a" in node_ids
    assert "n:b" in node_ids
    assert "n:c" in node_ids
    assert "n:d" not in node_ids


def test_bfs_max_nodes_applies_at_boundary_depth(graph_store):
    """max_nodes begrenzt auch Nodes, die erst an der Tiefengrenze liegen."""
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    graph_store.add_node("n:c", "topic", "C")
    graph_store.add_edge("n:a", "n:b", "betrifft")
    graph_store.add_edge("n:b", "n:c", "betrifft")

    subgraph = graph_store.get_neighbors("n:a", max_depth=2, max_nodes=2)
    node_ids = {n["id"] for n in subgraph["nodes"]}

    assert len(node_ids) == 2
    assert node_ids == {"n:a", "n:b"}


def test_bfs_max_nodes_returns_only_edges_between_returned_nodes(graph_store):
    """Subgraph enthaelt keine Kanten zu Nodes ausserhalb des max_nodes-Limits."""
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    graph_store.add_edge("n:a", "n:b", "betrifft")

    subgraph = graph_store.get_neighbors("n:a", max_depth=1, max_nodes=1)

    assert {n["id"] for n in subgraph["nodes"]} == {"n:a"}
    assert subgraph["edges"] == []


def test_duplicate_edge_ignored(graph_store):
    """Doppelte Kanten werden ignoriert."""
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    graph_store.add_edge("n:a", "n:b", "betrifft")
    graph_store.add_edge("n:a", "n:b", "betrifft")  # Duplikat

    stats = graph_store.get_stats()
    assert stats["total_edges"] == 1


def test_same_edge_with_different_source_keeps_provenance(graph_store):
    """ERP- und Extraction-Provenienz derselben Kante bleiben getrennt."""
    graph_store.add_node("supplier:alu-form", "supplier", "Alu-Form", source="erp_import")
    graph_store.add_node("article:art-1", "article", "Artikel 1", source="erp_import")
    graph_store.add_edge(
        "supplier:alu-form",
        "article:art-1",
        "liefert",
        chunk_id="chunk-extraction",
        source="extraction",
    )
    graph_store.add_edge(
        "supplier:alu-form",
        "article:art-1",
        "liefert",
        source="erp_import",
    )

    assert graph_store.get_stats()["total_edges"] == 2
    assert graph_store.get_connected_chunks(["supplier:alu-form"]) == ["chunk-extraction"]

    subgraph = graph_store.get_neighbors("supplier:alu-form", max_depth=1)
    assert len(subgraph["edges"]) == 1
    assert subgraph["edges"][0]["source"] == "extraction"

    graph_store.clear(source="erp_import")

    assert graph_store.get_stats()["total_edges"] == 1
    assert graph_store.get_connected_chunks(["supplier:alu-form"]) == ["chunk-extraction"]


def test_find_nodes_by_type(graph_store):
    """find_nodes mit entity_type Filter."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH")
    graph_store.add_node("supplier:mueller", "supplier", "Mueller Zulieferer")

    # Ohne Filter: beide
    results = graph_store.find_nodes("Mueller")
    assert len(results) == 2

    # Mit Filter: nur customer
    results = graph_store.find_nodes("Mueller", entity_type="customer")
    assert len(results) == 1
    assert results[0]["entity_type"] == "customer"


def test_clear_all(graph_store):
    """clear() ohne source loescht alles."""
    graph_store.add_node("n:a", "topic", "A", source="extraction")
    graph_store.add_node("n:b", "topic", "B", source="erp_import")
    graph_store.add_edge("n:a", "n:b", "betrifft")

    deleted = graph_store.clear()
    assert deleted == 2
    stats = graph_store.get_stats()
    assert stats["total_nodes"] == 0
    assert stats["total_edges"] == 0


def test_relation_filter(graph_store):
    """get_neighbors mit relation_filter."""
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    graph_store.add_node("n:c", "topic", "C")
    graph_store.add_edge("n:a", "n:b", "liefert")
    graph_store.add_edge("n:a", "n:c", "betrifft")

    # Only liefert edges
    subgraph = graph_store.get_neighbors("n:a", max_depth=1, relation_filter=["liefert"])
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert "n:b" in node_ids
    assert "n:c" not in node_ids
