"""Tests fuer list_embedding_models() in api_datasource_features.py.

Prueft, dass Collection-Counts exakt ueber den vollstaendigen normalisierten
Modellnamen gematcht werden — kein split(":")[0] Fallback mehr.
"""

from unittest.mock import patch

import pytest

from kiara.web.api_datasource_features import list_embedding_models


def _reset_cache():
    """Cache zuruecksetzen damit Tests isoliert laufen."""
    import kiara.web.api_datasource_features as mod

    mod._embed_cache_result = None
    mod._embed_cache_ts = 0.0


_FAKE_USER = {"username": "admin"}


@pytest.fixture(autouse=True)
def clear_cache():
    _reset_cache()
    yield
    _reset_cache()


@pytest.mark.asyncio
@patch("kiara.web.api_datasource_features.load_config", return_value={})
@patch("kiara.backend_selector.get_all_active")
@patch("kiara.retrieval.vectorstore.VectorStore.list_collections")
async def test_exact_model_match(mock_list, mock_backends, mock_config):
    """Collection-Count wird exakt ueber den vollstaendigen Modellnamen gematcht."""
    mock_backends.return_value = [
        {"model": "nomic-embed-text", "url": "http://localhost:11434"},
    ]
    mock_list.return_value = [
        {"name": "kb_nomic-embed-text", "model": "nomic-embed-text", "chunks": 500},
    ]

    result = await list_embedding_models(user=_FAKE_USER)

    assert len(result["models"]) == 1
    m = result["models"][0]
    assert m["name"] == "nomic-embed-text"
    assert m["has_collection"] is True
    assert m["chunk_count"] == 500


@pytest.mark.asyncio
@patch("kiara.web.api_datasource_features.load_config", return_value={})
@patch("kiara.backend_selector.get_all_active")
@patch("kiara.retrieval.vectorstore.VectorStore.list_collections")
async def test_tagged_model_not_matched_to_base(mock_list, mock_backends, mock_config):
    """Tagged-Modell (z.B. nomic:q4_0) darf NICHT dem Basismodell zugeschlagen werden."""
    mock_backends.return_value = [
        {"model": "nomic-embed-text:q4_0", "url": "http://localhost:11434"},
    ]
    # Collection existiert nur fuer das Basismodell, nicht fuer den Tag
    mock_list.return_value = [
        {"name": "kb_nomic-embed-text", "model": "nomic-embed-text", "chunks": 500},
    ]

    result = await list_embedding_models(user=_FAKE_USER)

    assert len(result["models"]) == 1
    m = result["models"][0]
    assert m["name"] == "nomic-embed-text:q4_0"
    assert m["has_collection"] is False
    assert m["chunk_count"] == 0


@pytest.mark.asyncio
@patch("kiara.web.api_datasource_features.load_config", return_value={})
@patch("kiara.backend_selector.get_all_active")
@patch("kiara.retrieval.vectorstore.VectorStore.list_collections")
async def test_no_backends_returns_error(mock_list, mock_backends, mock_config):
    """Keine Embedding-Backends -> Error-Response."""
    mock_backends.return_value = []

    result = await list_embedding_models(user=_FAKE_USER)

    assert result["models"] == []
    assert "error" in result
