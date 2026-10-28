"""Edge-Case-Tests fuer GraphStore."""

import time
import warnings

import pytest

from kiara.graph.store import GraphStore


@pytest.fixture
def graph_store():
    """Erstellt einen GraphStore (PostgreSQL, via conftest._init_pg_backend)."""
    store = GraphStore()
    store.clear()
    yield store
    store.clear()


def test_delete_by_chunk_id_preserves_erp_nodes(graph_store):
    """ERP-Knoten (source=erp_import) werden bei Edge-Deletion NICHT geloescht,
    auch wenn sie danach verwaist sind. Nur extraction-Knoten werden entfernt."""
    # ERP-Knoten (sollen erhalten bleiben)
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH", source="erp_import")
    # Extraction-Knoten (soll geloescht werden wenn verwaist)
    graph_store.add_node("topic:lieferung", "topic", "Lieferung", source="extraction")
    # Kante mit chunk_id
    graph_store.add_edge("topic:lieferung", "customer:k-1001", "betrifft", chunk_id="chunk-42")

    assert graph_store.get_stats()["total_nodes"] == 2
    assert graph_store.get_stats()["total_edges"] == 1

    # Chunk loeschen
    graph_store.delete_by_chunk_id("chunk-42")

    stats = graph_store.get_stats()
    assert stats["total_edges"] == 0

    # ERP-Knoten muss noch da sein
    erp_nodes = graph_store.find_nodes("Mueller")
    assert len(erp_nodes) == 1
    assert erp_nodes[0]["source"] == "erp_import"

    # Extraction-Knoten (verwaist) muss geloescht sein
    extraction_nodes = graph_store.find_nodes("Lieferung")
    assert len(extraction_nodes) == 0


def test_graph_cycle_traversal(graph_store):
    """BFS mit einem Zyklus (A→B→C→A) terminiert korrekt
    und liefert alle 3 Knoten genau einmal."""
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    graph_store.add_node("n:c", "topic", "C")
    graph_store.add_edge("n:a", "n:b", "betrifft")
    graph_store.add_edge("n:b", "n:c", "betrifft")
    graph_store.add_edge("n:c", "n:a", "betrifft")  # Zyklus!

    subgraph = graph_store.get_neighbors("n:a", max_depth=10)

    node_ids = [n["id"] for n in subgraph["nodes"]]
    assert len(node_ids) == 3  # Genau 3, nicht mehr (kein Endlos-Loop)
    assert set(node_ids) == {"n:a", "n:b", "n:c"}

    # Jeder Knoten darf nur einmal vorkommen
    assert len(node_ids) == len(set(node_ids))


def test_bidirectional_edge_traversal(graph_store):
    """get_neighbors findet Knoten unabhaengig von der Kantenrichtung.
    A→B: Traversal ab B findet A."""
    graph_store.add_node("n:a", "topic", "A")
    graph_store.add_node("n:b", "topic", "B")
    # Kante nur in eine Richtung: A → B
    graph_store.add_edge("n:a", "n:b", "betrifft")

    # Traversal ab B (gegen die Kantenrichtung)
    subgraph = graph_store.get_neighbors("n:b", max_depth=1)
    node_ids = {n["id"] for n in subgraph["nodes"]}

    assert "n:b" in node_ids
    assert "n:a" in node_ids  # A muss trotzdem gefunden werden


def test_large_graph_bfs_performance(graph_store):
    """BFS ueber 500 Knoten mit 1000 Kanten laeuft in < 2 Sekunden.
    Kein harter Fail bei Ueberschreitung, nur Warning."""
    import random

    from kiara.shared.db import db_transaction

    random.seed(42)

    # Batch: alle Knoten + Kanten in einer Transaction (statt 1500 einzelner Connections)
    with db_transaction() as conn:
        for i in range(500):
            graph_store.add_node(f"n:{i}", "topic", f"Node {i}", conn=conn)
        for _ in range(1000):
            a = random.randint(0, 499)
            b = random.randint(0, 499)
            if a != b:
                graph_store.add_edge(f"n:{a}", f"n:{b}", "betrifft", conn=conn)

    start = time.time()
    subgraph = graph_store.get_neighbors("n:0", max_depth=3)
    duration = time.time() - start

    # Ergebnis muss nicht-leer sein (n:0 plus Nachbarn)
    assert len(subgraph["nodes"]) >= 1

    if duration > 2.0:
        warnings.warn(
            f"BFS Performance: {duration:.2f}s fuer 500 Knoten/1000 Kanten "
            f"(Limit: 2.0s) – evtl. Optimierung noetig",
            UserWarning,
            stacklevel=2,
        )


def test_special_characters_in_labels(graph_store):
    """Knoten mit Umlauten, Sonderzeichen und Unicode werden korrekt gespeichert."""
    test_labels = [
        ("n:umlaut", "customer", "Müller & Söhne GmbH"),
        ("n:special", "supplier", "O'Brien <Corp.>"),
        ("n:unicode", "topic", "日本語テスト"),
        ("n:emoji", "topic", "Qualität ★★★"),
        ("n:quotes", "document", 'Datei "test.txt" (Kopie)'),
    ]

    for node_id, entity_type, label in test_labels:
        graph_store.add_node(node_id, entity_type, label)

    # Alle Knoten muessen korrekt gespeichert sein
    for node_id, _entity_type, label in test_labels:
        results = graph_store.find_nodes(label[:10])
        found = [r for r in results if r["id"] == node_id]
        assert len(found) == 1, f"Knoten '{label}' nicht gefunden"
        assert found[0]["label"] == label


def test_empty_properties_json(graph_store):
    """Knoten mit properties=None und properties={} funktionieren beide."""
    graph_store.add_node("n:none", "topic", "None Props", properties=None)
    graph_store.add_node("n:empty", "topic", "Empty Props", properties={})
    graph_store.add_node("n:data", "topic", "With Props", properties={"key": "value"})

    # Alle 3 muessen existieren
    stats = graph_store.get_stats()
    assert stats["total_nodes"] == 3

    # None -> {} (leeres Dict)
    results = graph_store.find_nodes("None Props")
    assert len(results) == 1
    assert results[0]["properties"] == {}

    # {} -> {}
    results = graph_store.find_nodes("Empty Props")
    assert len(results) == 1
    assert results[0]["properties"] == {}

    # {"key": "value"} -> {"key": "value"}
    results = graph_store.find_nodes("With Props")
    assert len(results) == 1
    assert results[0]["properties"] == {"key": "value"}


def test_find_nodes_case_insensitive(graph_store):
    """find_nodes('mueller') findet 'Mueller GmbH'.
    PostgreSQL ILIKE ist case-insensitive."""
    graph_store.add_node("customer:k-1001", "customer", "Mueller GmbH")
    graph_store.add_node("supplier:meier", "supplier", "MEIER AG")

    # Kleinbuchstaben-Suche findet Mueller
    results = graph_store.find_nodes("mueller")
    assert len(results) == 1
    assert results[0]["label"] == "Mueller GmbH"

    # Grossbuchstaben-Suche findet MEIER
    results = graph_store.find_nodes("meier")
    assert len(results) == 1
    assert results[0]["label"] == "MEIER AG"

    # Mixed Case
    results = graph_store.find_nodes("MUELLER")
    assert len(results) == 1
    assert results[0]["label"] == "Mueller GmbH"
