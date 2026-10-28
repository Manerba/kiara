"""Tests fuer Graph-Entity-Extraction waehrend der Ingestion-Pipeline."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

# Mock get_llm_backend BEFORE imports that trigger EntityExtractor
_llm_backend_patch = patch(
    "kiara.config.get_llm_backend",
    return_value={"url": "http://localhost:11434", "model": "qwen3:8b"},
)
_llm_backend_patch.start()

from kiara.graph.store import GraphStore
from kiara.ingestion.pipeline import run_ingestion


@pytest.fixture
def graph_store():
    """Erstellt einen GraphStore (PostgreSQL, via conftest._init_pg_backend)."""
    store = GraphStore()
    store.clear()
    yield store
    store.clear()


@pytest.fixture
def base_config(tmp_path, graph_store):
    """Minimale Config fuer Ingestion mit graph-Unterstützung."""
    ds_dir = tmp_path / "docs"
    ds_dir.mkdir()
    (ds_dir / "test.txt").write_text("Mueller GmbH liefert Aluminium-Gehaeuse an Schmidt AG.")

    return {
        "datasource": {
            "path": str(ds_dir),
            "file_types": [".txt"],
            "exclude_patterns": [],
        },
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "qwen3:8b",
            "embedding_model": "nomic-embed-text",
        },
        "vectordb": {
            "path": str(tmp_path / "vectordb"),
            "collection": "test",
            "backend": "qdrant",
        },
        "ingestion": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 100,
        },
        "database": {
            "path": "ignored",
        },
        "graph": {
            "enabled": True,
            "extract_on_ingestion": True,
            "max_traversal_depth": 2,
            "max_chunks": 20,
        },
    }


def _mock_extract_result():
    """Beispiel-Ergebnis einer Entity-Extraction."""
    return {
        "entities": [
            {
                "id": "supplier:mueller-gmbh",
                "type": "supplier",
                "label": "Mueller GmbH",
                "properties": {},
            },
            {
                "id": "customer:schmidt-ag",
                "type": "customer",
                "label": "Schmidt AG",
                "properties": {},
            },
        ],
        "relations": [
            {
                "source": "supplier:mueller-gmbh",
                "target": "customer:schmidt-ag",
                "relation": "liefert",
                "properties": {},
            },
        ],
    }


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
@patch("kiara.ingestion.runtime.chunk_document")
def test_ingestion_calls_entity_extractor(mock_chunk, mock_load, mock_vs, base_config, graph_store):
    """Wenn graph.enabled=True und extract_on_ingestion=True, wird EntityExtractor aufgerufen."""
    mock_load.return_value = {
        "text": "Mueller GmbH liefert Aluminium-Gehaeuse.",
        "metadata": {"source": "/tmp/test.txt", "chunk_index": 0},
    }
    mock_chunk.return_value = [
        {
            "text": "Mueller GmbH liefert Aluminium-Gehaeuse.",
            "metadata": {"source": "/tmp/test.txt", "chunk_index": 0},
        },
    ]
    mock_vs_instance = mock_vs.return_value
    mock_vs_instance.add = MagicMock()
    mock_vs_instance.delete_by_sources = MagicMock()

    with patch(
        "kiara.graph.extractor.EntityExtractor._call_llm", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.return_value = '{"entities": [{"id": "supplier:mueller", "type": "supplier", "label": "Mueller GmbH"}], "relations": []}'

        stats = run_ingestion(base_config, embedding_model="nomic-embed-text")

        assert stats["processed_files"] == 1
        mock_llm.assert_called()


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
@patch("kiara.ingestion.runtime.chunk_document")
def test_ingestion_skips_extraction_when_disabled(
    mock_chunk, mock_load, mock_vs, base_config, graph_store
):
    """Wenn graph.enabled=False, wird EntityExtractor NICHT aufgerufen."""
    base_config["graph"]["enabled"] = False

    mock_load.return_value = {
        "text": "Test text.",
        "metadata": {"source": "/tmp/test.txt", "chunk_index": 0},
    }
    mock_chunk.return_value = [
        {"text": "Test text.", "metadata": {"source": "/tmp/test.txt", "chunk_index": 0}},
    ]
    mock_vs_instance = mock_vs.return_value
    mock_vs_instance.add = MagicMock()
    mock_vs_instance.delete_by_sources = MagicMock()

    with patch(
        "kiara.graph.extractor.EntityExtractor._call_llm", new_callable=AsyncMock
    ) as mock_llm:
        stats = run_ingestion(base_config, embedding_model="nomic-embed-text")

        assert stats["processed_files"] == 1
        mock_llm.assert_not_called()


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
@patch("kiara.ingestion.runtime.chunk_document")
def test_ingestion_stores_extracted_entities(
    mock_chunk, mock_load, mock_vs, base_config, graph_store
):
    """Extrahierte Entities und Relations landen im GraphStore."""
    mock_load.return_value = {
        "text": "Mueller GmbH liefert an Schmidt AG.",
        "metadata": {"source": "/tmp/test.txt", "chunk_index": 0},
    }
    mock_chunk.return_value = [
        {
            "text": "Mueller GmbH liefert an Schmidt AG.",
            "metadata": {"source": "/tmp/test.txt", "chunk_index": 0},
        },
    ]
    mock_vs_instance = mock_vs.return_value
    mock_vs_instance.add = MagicMock()
    mock_vs_instance.delete_by_sources = MagicMock()

    extract_result = _mock_extract_result()
    import json

    llm_response = json.dumps(extract_result)

    with patch(
        "kiara.graph.extractor.EntityExtractor._call_llm", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.return_value = llm_response

        stats = run_ingestion(base_config, embedding_model="nomic-embed-text")

    assert stats["processed_files"] == 1

    # Pruefen, dass Entities im GraphStore gelandet sind
    gs_stats = graph_store.get_stats()
    assert gs_stats["total_nodes"] >= 2
    assert gs_stats["total_edges"] >= 1

    nodes = graph_store.find_nodes("Mueller")
    assert len(nodes) >= 1
    assert nodes[0]["label"] == "Mueller GmbH"


@patch("kiara.ingestion.pipeline.VectorStore")
@patch("kiara.ingestion.runtime.load_file")
@patch("kiara.ingestion.runtime.chunk_document")
def test_ingestion_handles_extraction_error(
    mock_chunk, mock_load, mock_vs, base_config, graph_store
):
    """Wenn EntityExtractor eine Exception wirft, laeuft die Ingestion trotzdem weiter."""
    mock_load.return_value = {
        "text": "Test text.",
        "metadata": {"source": "/tmp/test.txt", "chunk_index": 0},
    }
    mock_chunk.return_value = [
        {"text": "Test text.", "metadata": {"source": "/tmp/test.txt", "chunk_index": 0}},
    ]
    mock_vs_instance = mock_vs.return_value
    mock_vs_instance.add = MagicMock()
    mock_vs_instance.delete_by_sources = MagicMock()

    with patch(
        "kiara.graph.extractor.EntityExtractor._call_llm", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.side_effect = Exception("LLM nicht erreichbar")

        stats = run_ingestion(base_config, embedding_model="nomic-embed-text")

        # Ingestion darf nicht abbrechen
        assert stats["processed_files"] == 1
        assert stats["errors"] == 0
        # Chunks muessen trotzdem geschrieben worden sein
        assert stats["total_chunks"] >= 1
