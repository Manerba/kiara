"""Tests fuer Fehlerbehandlung des Qdrant VectorDB-Backends.

Testet Edge-Cases: leere Eingaben, nicht erreichbare Server,
nicht-existierende Datasources, Factory-Fehler.
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
# Gemeinsame Configs
# ---------------------------------------------------------------------------

CONFIG_QDRANT = {
    "ollama": {
        "url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "embedding_url": "",
    },
    "vectordb": {
        "backend": "qdrant",
        "path": "/tmp/test",
        "collection": "test",
        "vector_size": 768,
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": "",
    },
}

FAKE_EMBEDDING = [0.1] * 768


# ---------------------------------------------------------------------------
# 1. test_qdrant_connection_refused
# ---------------------------------------------------------------------------


def test_qdrant_connection_refused():
    """QdrantVectorStore mit nicht erreichbarem Server -> Exception."""
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ConnectionError("Connection refused")
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        with pytest.raises(ConnectionError, match="Connection refused"):
            QdrantVectorStore(CONFIG_QDRANT)


# ---------------------------------------------------------------------------
# 2. test_qdrant_filter_with_empty_where
# ---------------------------------------------------------------------------


def test_qdrant_filter_with_empty_where():
    """_build_filter({}) -> Filter(must=[])."""
    from qdrant_client.models import Filter

    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = MagicMock()
        collections_response = MagicMock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response
        info = MagicMock()
        info.points_count = 0
        info.config.params.vectors = VectorParams(size=768, distance=Distance.COSINE)
        info.payload_schema = {}
        mock_client.get_collection.return_value = info
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(CONFIG_QDRANT, intent="write")

        result = store._build_filter({})
        assert isinstance(result, Filter)
        assert result.must == []


# ---------------------------------------------------------------------------
# 3. test_factory_with_missing_config_keys
# ---------------------------------------------------------------------------


def test_factory_with_missing_config_keys():
    """Factory mit leerer Config nutzt im expliziten Write-Pfad Qdrant."""
    from kiara.retrieval.vectorstore_factory import create_vectorstore
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    info = MagicMock()
    info.points_count = 0
    info.config.params.vectors = VectorParams(size=768, distance=Distance.COSINE)
    info.payload_schema = {}
    with patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as mock_client:
        raw = MagicMock()
        raw.get_collections.return_value.collections = []
        raw.get_collection.return_value = info
        mock_client.return_value = raw
        store = create_vectorstore({}, intent="write")
        assert isinstance(store, QdrantVectorStore)
