"""Tests fuer die VectorStore-Factory."""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import Distance, VectorParams

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

# Mock get_embedding_model/get_embedding_dimension BEFORE imports
_embed_model_patch = patch(
    "kiara.retrieval.vectorstore_base.get_embedding_model", return_value="nomic-embed-text"
)
_embed_dim_patch = patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=768)
_embed_model_patch.start()
_embed_dim_patch.start()

from kiara.retrieval.vectorstore_base import VectorStoreBase
from kiara.retrieval.vectorstore_factory import create_vectorstore
from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore


# ---------------------------------------------------------------------------
# Hilfs-Funktion: Test-Config erzeugen
# ---------------------------------------------------------------------------
def _base_config(backend="qdrant"):
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "embedding_url": "",
        },
        "vectordb": {
            "backend": backend,
            "path": "/tmp/test_vectordb",
            "collection": "test",
            "vector_size": 768,
            "qdrant_url": "http://localhost:6333",
            "qdrant_api_key": "",
        },
    }


def _qdrant_patches():
    mock_collections_response = MagicMock()
    mock_collections_response.collections = []

    mock_qdrant_client = MagicMock()
    mock_qdrant_client.get_collections.return_value = mock_collections_response
    info = MagicMock()
    info.points_count = 0
    info.config.params.vectors = VectorParams(size=768, distance=Distance.COSINE)
    info.payload_schema = {}
    mock_qdrant_client.get_collection.return_value = info

    return [
        patch(
            "kiara.retrieval.vectorstore_qdrant.QdrantClient",
            return_value=mock_qdrant_client,
        ),
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ]


def test_factory_qdrant_default():
    """Factory ohne expliziten Backend-Key liefert QdrantVectorStore."""
    config = _base_config("qdrant")
    del config["vectordb"]["backend"]

    patches = _qdrant_patches()
    for p in patches:
        p.start()
    try:
        store = create_vectorstore(config, intent="write")
        assert isinstance(store, QdrantVectorStore)
        assert isinstance(store, VectorStoreBase)
    finally:
        for p in patches:
            p.stop()


def test_factory_qdrant():
    config = _base_config("qdrant")

    mock_collections_response = MagicMock()
    mock_collections_response.collections = []

    mock_qdrant_client = MagicMock()
    mock_qdrant_client.get_collections.return_value = mock_collections_response
    info = MagicMock()
    info.points_count = 0
    info.config.params.vectors = VectorParams(size=768, distance=Distance.COSINE)
    info.payload_schema = {}
    mock_qdrant_client.get_collection.return_value = info

    with (
        patch(
            "kiara.retrieval.vectorstore_qdrant.QdrantClient",
            return_value=mock_qdrant_client,
        ),
        patch(
            "kiara.retrieval.vectorstore_qdrant.embed_texts",
        ),
        patch(
            "kiara.retrieval.vectorstore_qdrant.embed_query",
        ),
    ):
        store = create_vectorstore(config, intent="write")
        assert isinstance(store, QdrantVectorStore)
        assert isinstance(store, VectorStoreBase)


def test_backward_compat_import():
    from kiara.retrieval.vectorstore import VectorStore

    config = _base_config("qdrant")

    patches = _qdrant_patches()
    for p in patches:
        p.start()
    try:
        store = VectorStore(config, intent="write")
        assert isinstance(store, VectorStoreBase)
    finally:
        for p in patches:
            p.stop()
