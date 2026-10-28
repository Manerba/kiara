"""Cross-Backend-Integrationstests fuer Interface-Konformitaet.

Prueft, dass das Qdrant-Backend das korrekte
Rueckgabeformat liefert (get_stats, query).
"""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance, VectorParams

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

# Mock get_embedding_model/get_embedding_dimension BEFORE QdrantVectorStore import
_embed_model_patch = patch(
    "kiara.retrieval.vectorstore_base.get_embedding_model", return_value="nomic-embed-text"
)
_embed_dim_patch = patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=768)
_embed_model_patch.start()
_embed_dim_patch.start()

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIG_BASE = {
    "ollama": {
        "url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "embedding_url": "",
    },
}

FAKE_EMBEDDING = [0.1] * 768


def _configure_mock_client(mock_client, *, points_count=42):
    """Modelliert eine vorhandene, schema-konforme Test-Collection."""
    collection = MagicMock()
    collection.name = "kb_nomic-embed-text"
    collections_response = MagicMock()
    collections_response.collections = [collection]
    mock_client.get_collections.return_value = collections_response

    mock_info = MagicMock()
    mock_info.points_count = points_count
    mock_info.status.value = "green"
    mock_info.config.params.vectors = VectorParams(size=768, distance=Distance.COSINE)
    mock_info.payload_schema = {}
    mock_client.get_collection.return_value = mock_info


def _qdrant_config():
    return {
        **CONFIG_BASE,
        "vectordb": {
            "backend": "qdrant",
            "path": "/tmp/test",
            "collection": "test",
            "vector_size": 768,
            "qdrant_url": "http://localhost:6333",
            "qdrant_api_key": "",
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def qdrant_store():
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch(
            "kiara.retrieval.vectorstore_qdrant.embed_texts", return_value=[FAKE_EMBEDDING]
        ) as mock_et,
        patch(
            "kiara.retrieval.vectorstore_qdrant.embed_query", return_value=FAKE_EMBEDDING
        ) as mock_eq,
    ):
        mock_client = MagicMock()
        # Explizite Collection- und VectorParams-Daten bilden den aktuellen
        # read-only Konstruktorvertrag ab, ohne eine echte Qdrant-Verbindung.
        _configure_mock_client(mock_client)

        MockClient.return_value = mock_client
        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(_qdrant_config())
        store._mock_client = mock_client
        store._mock_embed_query = mock_eq
        yield store


# ---------------------------------------------------------------------------
# 1. test_qdrant_returns_dict_with_backend_key
# ---------------------------------------------------------------------------


def test_qdrant_returns_dict_with_backend_key(qdrant_store):
    """get_stats() gibt ein Dict mit 'backend' Key zurueck."""
    stats = qdrant_store.get_stats()
    assert isinstance(stats, dict)
    assert "backend" in stats
    assert stats["backend"] == "qdrant"


# ---------------------------------------------------------------------------
# 2. test_qdrant_returns_total_chunks
# ---------------------------------------------------------------------------


def test_qdrant_returns_total_chunks(qdrant_store):
    """get_stats() gibt 'total_chunks' als Integer zurueck."""
    stats = qdrant_store.get_stats()
    assert "total_chunks" in stats
    assert isinstance(stats["total_chunks"], int)


# ---------------------------------------------------------------------------
# 3. test_qdrant_returns_collection_name
# ---------------------------------------------------------------------------


def test_qdrant_returns_collection_name(qdrant_store):
    """get_stats() gibt 'collection' zurueck."""
    stats = qdrant_store.get_stats()
    assert "collection" in stats
    assert isinstance(stats["collection"], str)
    assert stats["collection"].startswith("kb_")


# ---------------------------------------------------------------------------
# 4. test_query_result_format_qdrant
# ---------------------------------------------------------------------------


def test_query_result_format_qdrant(qdrant_store):
    """Qdrant query() gibt Liste von {text, metadata, score} zurueck."""
    mock_point = MagicMock()
    mock_point.payload = {"_text": "Hello World", "source": "test.txt"}
    mock_point.score = 0.95
    query_response = MagicMock()
    query_response.points = [mock_point]
    qdrant_store._mock_client.query_points.return_value = query_response

    with patch("kiara.retrieval.vectorstore_qdrant.embed_query", return_value=FAKE_EMBEDDING):
        results = qdrant_store.query("test", k=1)

    assert isinstance(results, list)
    assert len(results) == 1
    r = results[0]
    assert "text" in r and isinstance(r["text"], str)
    assert "metadata" in r and isinstance(r["metadata"], dict)
    assert "score" in r and isinstance(r["score"], float)
    assert r["text"] == "Hello World"
    assert "_text" not in r["metadata"]


# ---------------------------------------------------------------------------
# 5. test_backward_compat_vectorstore_is_qdrant
# ---------------------------------------------------------------------------


def test_vectorstore_facade_returns_qdrant():
    """Die aktuelle VectorStore-Fassade erzeugt das Qdrant-Backend."""
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts", return_value=[FAKE_EMBEDDING]),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query", return_value=FAKE_EMBEDDING),
    ):
        mock_client = MagicMock()
        _configure_mock_client(mock_client)
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore import VectorStore
        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = VectorStore(_qdrant_config())
        assert isinstance(store, QdrantVectorStore)


# ---------------------------------------------------------------------------
# 6. test_backward_compat_collection_name_reexport
# ---------------------------------------------------------------------------


def test_collection_name_comes_from_embedding_module():
    """Collection-Namen werden vom aktuellen Embedding-Modul bereitgestellt."""
    from kiara.retrieval.embedding import collection_name_for_model

    assert callable(collection_name_for_model)
    assert collection_name_for_model("nomic-embed-text") == "kb_nomic-embed-text"
