"""P64 targeted contracts for stored inventory identity and profile dimensions.

Inventory tests use the real kiara_test PostgreSQL fixture and S4 persistence
helpers. The dimension fixture represents an already validated profile guard;
remote discovery and Qdrant I/O are deliberately absent from these tests.
"""

import copy
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_index_compatibility_s4 import _config, _persist_contract, _resolved

from kiara.retrieval import embedding
from kiara.retrieval import index_compatibility as contract
from kiara.retrieval.index_compatibility import IndexVectorKind
from kiara.retrieval.index_contract_backfill import ReadOnlyQdrantClient, inventory_targets


def _settings(late):
    config = copy.deepcopy(_config())
    config["enrichment"] = {"late_chunking": late}
    config["ollama"] = {"embedding_model": "model"}
    return config


def _inventory_client():
    raw = MagicMock(spec=["get_collections"])
    raw.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="kb_model")]
    )
    return ReadOnlyQdrantClient(raw)


@pytest.mark.parametrize(
    ("stored_kind", "configured_late"),
    [(IndexVectorKind.LATE_CHUNKING, False), (IndexVectorKind.DENSE, True)],
)
def test_inventory_preserves_stored_kind_when_current_mode_differs(
    test_db, stored_kind, configured_late
):
    assert test_db.execute("SELECT current_database() AS name").fetchone()["name"] == "kiara_test"
    config = _settings(configured_late)
    target = contract.make_index_target(
        config, collection_name="kb_model", vector_kind=stored_kind, requested_model="model"
    )
    late = stored_kind is IndexVectorKind.LATE_CHUNKING
    resolved = replace(
        _resolved(target),
        profile_id="profile-late-v1" if late else "profile-dense-v1",
        profile_endpoint="/api/embed_late" if late else "/api/embed",
    )
    _persist_contract(test_db, target, resolved)
    test_db.commit()

    targets = inventory_targets(config, _inventory_client())

    assert len(targets) == 1
    assert targets[0] == target
    row = test_db.execute(
        "SELECT vector_kind, logical_index_key FROM index_contracts WHERE collection_name = %s",
        (target.collection_name,),
    ).fetchone()
    assert row["vector_kind"] == stored_kind.value
    assert row["logical_index_key"] == target.logical_index_key


def test_unbound_historical_collection_remains_dense_candidate_under_late_config(test_db):
    assert test_db.execute("SELECT current_database() AS name").fetchone()["name"] == "kiara_test"
    assert test_db.execute("SELECT COUNT(*) AS n FROM index_contracts").fetchone()["n"] == 0
    config = _settings(True)

    targets = inventory_targets(config, _inventory_client())

    assert targets == [contract.dense_index_target(config, "model")]
    assert test_db.execute("SELECT COUNT(*) AS n FROM index_contracts").fetchone()["n"] == 0


@pytest.fixture
def validated_profile_guard():
    """Supply the selected immutable profile snapshot at the post-validation boundary."""

    @contextmanager
    def activate(kind):
        config = _settings(kind is IndexVectorKind.LATE_CHUNKING)
        target = contract.make_index_target(
            config, collection_name="kb_model", vector_kind=kind, requested_model="model"
        )
        profile_id = f"selected-{kind.value}-profile"
        snapshot = json.dumps(
            {
                "schema_version": 1,
                "profiles": [
                    {"profile_id": "other-profile", "dimensions": 1536},
                    {"profile_id": profile_id, "dimensions": 768},
                ],
            },
            sort_keys=True,
        )
        snapshot_hash = hashlib.sha256(snapshot.encode()).hexdigest()
        resolved = replace(
            _resolved(target),
            profile_id=profile_id,
            profile_endpoint="/api/embed_late"
            if kind is IndexVectorKind.LATE_CHUNKING
            else "/api/embed",
            backend={
                "id": 7,
                "provider": "ollama",
                "url": "http://dimension.test:11434",
                "model": "model",
                "extra_config": json.dumps({"embedding_dim": 4096}),
            },
            contract_snapshot_json_text=snapshot,
            contract_snapshot_canonical_text=snapshot,
            contract_snapshot_sha256=snapshot_hash,
        )
        token = contract.IndexGuardToken(
            target=target,
            operation="document_embedding",
            write=True,
            validation_scope=contract.IndexValidationScope.LIVE_PROFILE,
            require_configured_default=False,
            resolved=resolved,
            stored={},
            snapshot_drift=contract.ContractSnapshotDrift(
                raw_changed=False,
                canonical_changed=False,
                sha256_changed=False,
                stored_sha256=snapshot_hash,
                current_sha256=snapshot_hash,
            ),
            fence_version=1,
            connection=None,
        )
        previous = contract._active_tokens.get()
        reset = contract._active_tokens.set((*previous, token))
        try:
            yield config
        finally:
            contract._active_tokens.reset(reset)

    return activate


@pytest.mark.parametrize("kind", [IndexVectorKind.DENSE, IndexVectorKind.LATE_CHUNKING])
def test_validated_profile_dimension_precedes_backend_wide_cache(
    monkeypatch, validated_profile_guard, kind
):
    forbidden_calls = []
    for name in (
        "get_all_active",
        "get_embedding_backend_for_model",
        "_fetch_embedding_dim_sync",
        "_cache_embedding_dim",
    ):
        forbidden = MagicMock(side_effect=AssertionError(f"unexpected dimension fallback: {name}"))
        monkeypatch.setattr(embedding, name, forbidden)
        forbidden_calls.append(forbidden)

    with validated_profile_guard(kind) as config:
        assert embedding.get_embedding_dimension("model", config) == 768

    for forbidden in forbidden_calls:
        forbidden.assert_not_called()
