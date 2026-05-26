"""Tests for VectorStoreBase ABC."""

import pytest

from kiara.retrieval.vectorstore_base import VectorStoreBase

EXPECTED_ABSTRACT_METHODS = {
    "add",
    "add_with_embeddings",
    "query",
    "delete_by_ids",
    "delete_by_sources",
    "get_ids_by_sources",
    "count_total",
    "query_by_ids",
    "scroll_all",
    "update_ds_ids",
    "remove_payload_field",
    "get_stats",
    "get_directories",
    "get_file_types",
    "get_collection_name",
    "get_all_documents",
    "get_documents_by_ids",
    "list_collections",
    "list_sources_for_datasource",
    "search",
    "count_filtered",
}


def test_abc_cannot_instantiate():
    """VectorStoreBase cannot be directly instantiated."""
    cfg = {"ollama": {"embedding_model": "nomic-embed-text"}}
    with pytest.raises(TypeError):
        VectorStoreBase(cfg)


def test_interface_methods_exist():
    """All expected abstract methods exist on VectorStoreBase."""
    assert VectorStoreBase.__abstractmethods__ == EXPECTED_ABSTRACT_METHODS
