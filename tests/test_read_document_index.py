"""Die Index-Ergaenzung muss auch nicht ausgewaehlte Embedding-Modelle erreichen."""

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from kiara.retrieval.index_compatibility import GuardedQdrantClient, IndexCompatibilityError
from scripts.ensure_document_read_indexes import ensure_indexes


def test_index_update_covers_every_collection_without_embedding_calls():
    stores = [MagicMock(), MagicMock()]
    with patch("scripts.ensure_document_read_indexes.VectorStore") as factory:
        factory.list_collections.return_value = [
            {"name": "kb_empty_unverified", "model": "empty", "chunks": 0},
            {"name": "kb_a", "model": "a", "chunks": 7},
            {"name": "kb_b", "model": "b", "chunks": 11},
        ]
        factory.side_effect = stores
        stores[0].ensure_document_read_index.return_value = False
        stores[1].ensure_document_read_index.return_value = True
        config = {"vectordb": {}}
        results = ensure_indexes(config)
    assert factory.call_args_list == [
        call({"vectordb": {"timeout": 300.0}}, embedding_model="a", intent="maintenance"),
        call({"vectordb": {"timeout": 300.0}}, embedding_model="b", intent="maintenance"),
    ]
    assert config == {"vectordb": {}}
    assert results == [
        {"collection": "kb_empty_unverified", "created": False, "empty": True},
        {"collection": "kb_a", "created": False},
        {"collection": "kb_b", "created": True},
    ]
    for store in stores:
        store.ensure_document_read_index.assert_called_once_with()
        store._client.close.assert_called_once_with()


@pytest.mark.parametrize("reason", ["index_contract_unverified", "index_contract_mismatch"])
def test_unverified_collections_stay_blocked_and_other_contract_failures_propagate(reason):
    error = IndexCompatibilityError(
        reason, "read", SimpleNamespace(collection_name="kb_blocked"), "unverified"
    )
    with patch("scripts.ensure_document_read_indexes.VectorStore") as factory:
        factory.list_collections.return_value = [{"name": "kb_blocked", "model": "b", "chunks": 4}]
        factory.side_effect = error
        if reason == "index_contract_unverified":
            assert ensure_indexes({}) == [
                {
                    "collection": "kb_blocked",
                    "created": False,
                    "blocked": reason,
                }
            ]
        else:
            with pytest.raises(IndexCompatibilityError):
                ensure_indexes({})


@pytest.mark.parametrize("index_error", [False, True])
def test_maintenance_closes_transport_without_live_model_contract(index_error):
    transport = MagicMock()
    guarded = GuardedQdrantClient(transport, {}, SimpleNamespace(collection_name="kb_offline"))
    store = MagicMock(_client=guarded)
    store.ensure_document_read_index.return_value = False
    if index_error:
        store.ensure_document_read_index.side_effect = RuntimeError("Indexfehler")
    with (
        patch("scripts.ensure_document_read_indexes.VectorStore") as factory,
        patch(
            "kiara.retrieval.index_compatibility.guard_index_for_active_job",
            side_effect=AssertionError("Kein Live-Modellvertrag beim Schliessen"),
        ) as guard,
    ):
        factory.list_collections.return_value = [
            {"name": "kb_offline", "model": "offline", "chunks": 3}
        ]
        factory.return_value = store
        if index_error:
            with pytest.raises(RuntimeError, match="Indexfehler"):
                ensure_indexes({})
        else:
            assert ensure_indexes({}) == [{"collection": "kb_offline", "created": False}]
    guard.assert_not_called()
    transport.close.assert_called_once_with()
