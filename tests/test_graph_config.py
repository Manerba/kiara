"""Tests fuer GraphRAG-Config-Integration."""

import os
import tempfile

from kiara.config import DEFAULT_CONFIG, _deep_merge, load_config


def test_default_config_has_graph_section():
    """DEFAULT_CONFIG enthaelt graph-Section mit allen erwarteten Feldern."""
    assert "graph" in DEFAULT_CONFIG

    graph = DEFAULT_CONFIG["graph"]
    assert "enabled" in graph
    assert "extract_on_ingestion" in graph
    assert "max_traversal_depth" in graph
    assert "max_chunks" in graph

    # Typen pruefen
    assert isinstance(graph["enabled"], bool)
    assert isinstance(graph["extract_on_ingestion"], bool)
    assert isinstance(graph["max_traversal_depth"], int)
    assert isinstance(graph["max_chunks"], int)


def test_graph_config_merge():
    """Benutzerdefinierte Config wird korrekt mit Defaults gemergt (deep_merge)."""
    user_config = {
        "graph": {
            "enabled": True,
            "max_traversal_depth": 3,
        },
    }

    merged = _deep_merge(DEFAULT_CONFIG, user_config)

    # Ueberschriebene Werte
    assert merged["graph"]["enabled"] is True
    assert merged["graph"]["max_traversal_depth"] == 3

    # Default-Werte bleiben erhalten
    assert (
        merged["graph"]["extract_on_ingestion"] == DEFAULT_CONFIG["graph"]["extract_on_ingestion"]
    )
    assert merged["graph"]["max_chunks"] == DEFAULT_CONFIG["graph"]["max_chunks"]

    # Andere Sections bleiben unberuehrt
    assert merged["ollama"] == DEFAULT_CONFIG["ollama"]
    assert merged["vectordb"] == DEFAULT_CONFIG["vectordb"]


def test_graph_disabled_by_default():
    """graph.enabled ist standardmaessig False."""
    assert DEFAULT_CONFIG["graph"]["enabled"] is False

    # Auch bei load_config mit nicht-existierender Datei
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    os.unlink(path)  # Datei loeschen -> load_config nutzt Defaults

    # Umgebungsvariable setzen, damit keine echte Config geladen wird
    old_env = os.environ.get("KIARA_CONFIG")
    try:
        os.environ["KIARA_CONFIG"] = path
        config = load_config()
        assert config["graph"]["enabled"] is False
    finally:
        if old_env:
            os.environ["KIARA_CONFIG"] = old_env
        elif "KIARA_CONFIG" in os.environ:
            del os.environ["KIARA_CONFIG"]
