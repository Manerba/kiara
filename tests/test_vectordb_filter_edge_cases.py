"""Tests fuer Filter-Edge-Cases: Verschachtelung, leere Listen, Escaping.

Testet die Konvertierung von where-Filtern nach Qdrant-Filter.
"""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText

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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def qdrant_store():
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = MagicMock()
        collections_response = MagicMock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response
        MockClient.return_value = mock_client
        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        yield QdrantVectorStore(CONFIG_QDRANT)


# ---------------------------------------------------------------------------
# 1. test_qdrant_nested_and
# ---------------------------------------------------------------------------


def test_qdrant_nested_and(qdrant_store):
    """Verschachteltes $and: {"$and": [{"$and": [{a:1},{b:2}]}, {c:3}]}."""
    where = {
        "$and": [
            {"$and": [{"a": 1}, {"b": 2}]},
            {"c": 3},
        ]
    }
    result = qdrant_store._build_filter(where)

    assert isinstance(result, Filter)
    # Alle 3 Conditions werden in die must-Liste geflattened
    assert len(result.must) == 3

    keys = [c.key for c in result.must]
    assert "a" in keys
    assert "b" in keys
    assert "c" in keys


# ---------------------------------------------------------------------------
# 2. test_qdrant_in_with_single_value
# ---------------------------------------------------------------------------


def test_qdrant_in_with_single_value(qdrant_store):
    """$in mit einem einzigen Element -> MatchAny mit einem Element."""
    where = {"key": {"$in": ["one"]}}
    result = qdrant_store._build_filter(where)

    assert isinstance(result, Filter)
    assert len(result.must) == 1
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == ["one"]


# ---------------------------------------------------------------------------
# 3. test_qdrant_mixed_operators
# ---------------------------------------------------------------------------


def test_qdrant_mixed_operators(qdrant_store):
    """Gemischte Operatoren: Equality + $in -> zwei Conditions in must."""
    where = {"a": 1, "b": {"$in": [2, 3]}}
    result = qdrant_store._build_filter(where)

    assert isinstance(result, Filter)
    assert len(result.must) == 2

    cond_types = {}
    for c in result.must:
        cond_types[c.key] = type(c.match).__name__
    assert cond_types["a"] == "MatchValue"
    assert cond_types["b"] == "MatchAny"


# ---------------------------------------------------------------------------
# 4. test_qdrant_empty_in_list
# ---------------------------------------------------------------------------


def test_qdrant_empty_in_list(qdrant_store):
    """$in mit leerer Liste -> MatchAny mit leerer any-Liste."""
    where = {"key": {"$in": []}}
    result = qdrant_store._build_filter(where)

    assert isinstance(result, Filter)
    assert len(result.must) == 1
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert isinstance(cond.match, MatchAny)
    assert cond.match.any == []


# ---------------------------------------------------------------------------
# 5. test_qdrant_contains_operator
# ---------------------------------------------------------------------------


def test_qdrant_contains_source(qdrant_store):
    """$contains auf source -> MatchText (Substring-Suche)."""
    where = {"source": {"$contains": "invoice"}}
    result = qdrant_store._build_filter(where)

    assert isinstance(result, Filter)
    assert len(result.must) == 1
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "source"
    assert isinstance(cond.match, MatchText)
    assert cond.match.text == "invoice"


def test_qdrant_contains_directory(qdrant_store):
    """$contains auf directory -> MatchText (Substring-Suche)."""
    where = {"directory": {"$contains": "projects/alpha"}}
    result = qdrant_store._build_filter(where)

    assert isinstance(result, Filter)
    assert len(result.must) == 1
    cond = result.must[0]
    assert isinstance(cond, FieldCondition)
    assert cond.key == "directory"
    assert isinstance(cond.match, MatchText)
    assert cond.match.text == "projects/alpha"


def test_qdrant_contains_combined_with_in(qdrant_store):
    """$contains + $in in $and -> beide Conditions korrekt."""
    where = {
        "$and": [
            {"source": {"$contains": "report"}},
            {"ds_ids": {"$in": [1, 2, 3]}},
        ]
    }
    result = qdrant_store._build_filter(where)

    assert isinstance(result, Filter)
    assert len(result.must) == 2
    types = {c.key: type(c.match).__name__ for c in result.must}
    assert types["source"] == "MatchText"
    assert types["ds_ids"] == "MatchAny"
