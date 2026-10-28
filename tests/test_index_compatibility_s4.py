"""S4 regression gates for persistent index compatibility and migration."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from kiara.retrieval.embedding_contract import ContractIssue, ContractState, canonical_json
from kiara.retrieval.embedding_discovery import ServerContractDiscovery
from kiara.retrieval.index_compatibility import (
    IndexCompatibilityError,
    IndexContractUnavailableError,
    IndexTarget,
    IndexValidationScope,
    IndexVectorKind,
    ResolvedIndexContract,
    _colbert_backend,
    _qdrant_unavailability_cause,
    _validate_stored_contract,
    guard_index,
    insert_contract_evidence,
    insert_index_contract,
    make_index_target,
    resolve_current_index_contract,
    validate_job_contract_snapshot,
)
from kiara.retrieval.index_contract_backfill import (
    WITNESS_ABS_TOLERANCE,
    WITNESS_COSINE_TOLERANCE,
    BackfillAssessment,
    ReadOnlyQdrantClient,
    compare_witness_vectors,
    rollback_backfill_metadata,
    run_index_contract_backfill,
    run_vector_witness,
)
from kiara.web.index_contract_migration import migrate_index_contract_schema


def _config() -> dict:
    return {
        "vectordb": {
            "qdrant_url": "http://secret:password@QDRANT.EXAMPLE:6333/?token=x",
        },
        "retrieval": {"colbert": {"enabled": False}},
    }


def _contract_example(name: str) -> dict:
    examples = json.loads(
        (
            Path(__file__).parent / "fixtures" / "kiron-embedding-capabilities-v1.examples.json"
        ).read_text(encoding="utf-8")
    )
    return copy.deepcopy(next(item["value"] for item in examples["valid"] if item["name"] == name))


def _role_sensitive_contract() -> dict:
    return _contract_example("role_sensitive_provider_phase")


def _strict_role_sensitive_contract() -> dict:
    return _contract_example("role_sensitive_strict_phase_same_ids")


def _server_discovery(contract: dict):
    from kiara.retrieval.embedding_discovery import discover_server_contracts

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": contract["canonical_model_id"],
                            "model": contract["canonical_model_id"],
                            "details": {},
                            "kiron_capabilities": contract,
                        }
                    ]
                },
            )
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "model": contract["canonical_model_id"],
                    "kiron_capabilities": contract,
                },
            )
        return httpx.Response(404)

    return discover_server_contracts(
        "http://s4-profile-selection.test",
        transport=httpx.MockTransport(handler),
        force_refresh=True,
    )


def _backend_connection(backend: dict) -> MagicMock:
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [backend]
    return conn


def _target(kind: IndexVectorKind = IndexVectorKind.DENSE) -> IndexTarget:
    return make_index_target(
        _config(),
        collection_name="kb_model",
        vector_kind=kind,
        requested_model="Model:latest",
    )


def _guard_error(
    operation: str = "vector_payload_read",
    kind: IndexVectorKind = IndexVectorKind.DENSE,
) -> IndexCompatibilityError:
    return IndexCompatibilityError(
        "index_contract_mismatch",
        operation,
        _target(kind),
        "injected low-level compatibility failure",
        current_state="mismatch",
    )


def _resolved(target: IndexTarget | None = None) -> ResolvedIndexContract:
    target = target or _target()
    snapshot_text = '{"schema_version":1}'
    return ResolvedIndexContract(
        state=ContractState.VALID,
        reason_code="index_contract_verified",
        target=target,
        backend={"id": 7, "provider": "ollama", "url": "http://kiron:11434"},
        backend_id=7,
        backend_provider="ollama",
        backend_url="http://kiron:11434",
        canonical_model_id="canonical:model",
        profile_id="profile-dense-v1",
        profile_kind=target.vector_kind.value,
        profile_endpoint="/api/embed",
        contract_schema_version=1,
        contract_schema_sha256="schema-hash",
        contract_snapshot_json_text=snapshot_text,
        contract_snapshot_canonical_text=snapshot_text,
        contract_snapshot_sha256=hashlib.sha256(snapshot_text.encode()).hexdigest(),
        index_compatibility_id="sha256:" + "1" * 64,
        query_compatibility_id="sha256:" + "2" * 64,
        backend_type="transformers",
        backend_implementation="test",
        backend_implementation_revision="impl-rev",
        artifact_repository="repo",
        artifact_revision="rev",
        artifact_manifest_digest="sha256:" + "3" * 64,
        artifact_json_text='{"repository":"repo"}',
    )


def _stored(resolved: ResolvedIndexContract | None = None) -> dict:
    resolved = resolved or _resolved()
    return {
        "requested_model": resolved.target.requested_model,
        "backend_id": resolved.backend_id,
        "backend_provider": resolved.backend_provider,
        "backend_url": resolved.backend_url,
        "canonical_model_id": resolved.canonical_model_id,
        "profile_id": resolved.profile_id,
        "profile_kind": resolved.profile_kind,
        "profile_endpoint": resolved.profile_endpoint,
        "contract_schema_version": resolved.contract_schema_version,
        "contract_schema_sha256": resolved.contract_schema_sha256,
        "contract_snapshot_json_text": resolved.contract_snapshot_json_text,
        "contract_snapshot_canonical_text": resolved.contract_snapshot_canonical_text,
        "contract_snapshot_sha256": resolved.contract_snapshot_sha256,
        "evidence_contract_snapshot_sha256": resolved.contract_snapshot_sha256,
        "index_compatibility_id": resolved.index_compatibility_id,
        "query_compatibility_id": resolved.query_compatibility_id,
        "backend_type": resolved.backend_type,
        "backend_implementation": resolved.backend_implementation,
        "backend_implementation_revision": resolved.backend_implementation_revision,
        "artifact_repository": resolved.artifact_repository,
        "artifact_revision": resolved.artifact_revision,
        "artifact_manifest_digest": resolved.artifact_manifest_digest,
        "artifact_json_text": resolved.artifact_json_text,
        "verification_state": "verified",
        "reason_code": "witness_verified",
    }


def _persist_contract(
    conn,
    target: IndexTarget | None = None,
    resolved: ResolvedIndexContract | None = None,
    *,
    verification_state: str = "verified",
    backfill_run_id: str | None = None,
) -> dict:
    target = target or _target()
    resolved = resolved or _resolved(target)
    references = "[]"
    diagnostics = '{"test":true}'
    evidence_type = "provenance" if verification_state == "verified" else verification_state
    evidence_id = insert_contract_evidence(
        conn,
        {
            "logical_index_key": target.logical_index_key,
            "evidence_type": evidence_type,
            "result_state": verification_state,
            "reason_code": f"test_{verification_state}",
            "sample_count": 0,
            "collection_points_count": 1,
            "references_json_text": references,
            "references_sha256": hashlib.sha256(references.encode()).hexdigest(),
            "diagnostics_json_text": diagnostics,
            "diagnostics_sha256": hashlib.sha256(diagnostics.encode()).hexdigest(),
            "candidate_index_compatibility_id": (
                resolved.index_compatibility_id if verification_state == "verified" else None
            ),
            "candidate_query_compatibility_id": (
                resolved.query_compatibility_id if verification_state == "verified" else None
            ),
            "contract_snapshot_sha256": resolved.contract_snapshot_sha256,
            "backfill_run_id": backfill_run_id,
        },
    )
    return insert_index_contract(
        conn,
        target,
        resolved,
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        evidence_sample_count=0,
        verification_state=verification_state,
        reason_code=f"test_{verification_state}",
        observed_points_count=1,
        backfill_run_id=backfill_run_id,
    )


@pytest.mark.parametrize(
    "extra_config",
    [
        None,
        {},
        "",
        "{",
        "[]",
        [],
        json.dumps({"profile_id": ""}),
        json.dumps({"embedding_profile_id": " "}),
        json.dumps({"profile_id": "does-not-exist"}),
        json.dumps(
            {
                "embedding_profile_id": "service-dense-v1",
                "profile_id": "other-dense-v1",
            }
        ),
        json.dumps({"embedding_profile_id": "service-dense-v1"}),
        json.dumps(
            {
                "embedding_profile_id": "service-dense-v1",
                "profile_id": "service-dense-v1",
            }
        ),
    ],
    ids=[
        "none",
        "empty-mapping",
        "empty-string",
        "invalid-json",
        "non-object-json",
        "wrong-type",
        "empty-profile-id",
        "whitespace-profile-id",
        "unknown-profile-id",
        "conflicting-profile-ids",
        "valid-explicit-profile-id",
        "valid-matching-profile-ids",
    ],
)
def test_s3_and_s4_share_end_to_end_profile_selection(extra_config):
    from kiara.retrieval.embedding_discovery import (
        discover_configured_embedding_backends,
    )

    contract = _role_sensitive_contract()
    server = _server_discovery(contract)
    backend = {
        "id": 701,
        "name": "KIron",
        "category": "embedding",
        "provider": "ollama",
        "url": "http://s4-profile-selection.test",
        "model": contract["canonical_model_id"],
        "priority": 10,
        "extra_config": extra_config,
    }
    target = make_index_target(
        _config(),
        collection_name="kb_profile_selection",
        vector_kind=IndexVectorKind.DENSE,
        requested_model=contract["canonical_model_id"],
    )

    with patch(
        "kiara.retrieval.embedding_discovery.discover_server_contracts",
        return_value=server,
    ):
        s3 = discover_configured_embedding_backends([backend]).backends[0]
    with patch(
        "kiara.retrieval.index_compatibility.discover_server_contracts",
        return_value=server,
    ) as s4_consumer:
        s4 = resolve_current_index_contract(
            _backend_connection(backend),
            _config(),
            target,
            force_refresh=True,
        )

    assert s4.state is s3.state
    if s3.issues:
        assert s4.reason_code == s3.issues[0].reason_code
        assert s4.issues == tuple(issue.to_dict() for issue in s3.issues)
        assert s4_consumer.call_count == 0
    elif s3.resolution and s3.resolution.issues:
        assert s4.reason_code == s3.resolution.issues[0].reason_code
        assert s4.issues == tuple(issue.to_dict() for issue in s3.resolution.issues)
        s4_consumer.assert_called_once()
    else:
        assert s3.resolution is not None
        assert s3.resolution.profile is not None
        assert s4.profile_id == s3.resolution.profile.profile_id
        assert s4.index_compatibility_id == s3.resolution.profile.index_compatibility_id
        assert s4.query_compatibility_id == s3.resolution.profile.query_compatibility_id
        s4_consumer.assert_called_once()

    if not (s3.resolution and s3.resolution.profile and s3.rag_verified):
        assert s4.index_compatibility_id is None
        assert s4.query_compatibility_id is None


def test_empty_backend_extra_config_blocks_full_s4_resolver_before_consumers():
    contract = _role_sensitive_contract()
    backend = {
        "id": 702,
        "name": "KIron",
        "category": "embedding",
        "provider": "ollama",
        "url": "http://s4-profile-selection.test",
        "model": contract["canonical_model_id"],
        "priority": 10,
        "extra_config": "",
    }
    target = make_index_target(
        _config(),
        collection_name="kb_empty_extra_config",
        vector_kind=IndexVectorKind.DENSE,
        requested_model=contract["canonical_model_id"],
    )

    with (
        patch("kiara.retrieval.index_compatibility.discover_server_contracts") as discover,
        patch("kiara.retrieval.index_compatibility.resolve_embedding_profile") as consume,
    ):
        resolved = resolve_current_index_contract(
            _backend_connection(backend),
            _config(),
            target,
        )

    assert resolved.state is ContractState.INCONSISTENT
    assert resolved.reason_code == "backend_extra_config_malformed"
    assert resolved.index_compatibility_id is None
    assert resolved.query_compatibility_id is None
    assert resolved.issues[0]["reason_code"] == "backend_extra_config_malformed"
    discover.assert_not_called()
    consume.assert_not_called()


def test_real_guard_accepts_provider_index_with_strict_enforcement_for_read_query_and_write(
    test_db,
    monkeypatch,
):
    """Exact #190 regression: enforcement drift is not vector-space drift."""
    from kiara.retrieval import index_compatibility as module

    provider_contract = _role_sensitive_contract()
    strict_contract = _strict_role_sensitive_contract()
    provider_profile = provider_contract["profiles"][0]
    strict_profile = strict_contract["profiles"][0]
    assert provider_profile["index_compatibility_id"] == strict_profile["index_compatibility_id"]
    assert provider_profile["query_compatibility_id"] == strict_profile["query_compatibility_id"]
    assert provider_profile["input_type"] == {
        **strict_profile["input_type"],
        "required": False,
        "default": "search_document",
        "missing_role_behavior": "use_default",
    }

    backend = {
        "id": 703,
        "name": "KIron strict-transition fixture",
        "category": "embedding",
        "provider": "ollama",
        "url": "http://s4-profile-selection.test",
        "model": provider_contract["canonical_model_id"],
        "priority": 1,
        "extra_config": json.dumps({"embedding_profile_id": provider_profile["profile_id"]}),
    }
    test_db.execute(
        "INSERT INTO ki_backends "
        "(id, category, provider, name, url, model, is_active, priority, extra_config) "
        "VALUES (%s, %s, %s, %s, %s, %s, true, %s, %s)",
        (
            backend["id"],
            backend["category"],
            backend["provider"],
            backend["name"],
            backend["url"],
            backend["model"],
            backend["priority"],
            backend["extra_config"],
        ),
    )
    target = make_index_target(
        _config(),
        collection_name="kb_strict_transition",
        vector_kind=IndexVectorKind.DENSE,
        requested_model=provider_contract["canonical_model_id"],
    )
    provider_server = _server_discovery(provider_contract)
    with patch(
        "kiara.retrieval.index_compatibility.discover_server_contracts",
        return_value=provider_server,
    ):
        provider_resolved = resolve_current_index_contract(
            test_db,
            _config(),
            target,
            force_refresh=True,
        )
    assert provider_resolved.verified is True
    persisted = _persist_contract(test_db, target, provider_resolved)
    test_db.commit()

    strict_server = _server_discovery(strict_contract)
    monkeypatch.setattr(module, "_collection_state", lambda *_args: (True, 1))
    raw_qdrant = MagicMock()
    downstream_provider = MagicMock()
    metadata_before = {
        table: test_db.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
        for table in ("index_contracts", "index_contract_evidence", "ingestion_jobs")
    }

    with patch(
        "kiara.retrieval.index_compatibility.discover_server_contracts",
        return_value=strict_server,
    ):
        for operation, write in (
            ("read", False),
            ("query_embedding", False),
            ("vector_upsert", True),
        ):
            with guard_index(
                _config(),
                target,
                operation=operation,
                write=write,
                qdrant_client=raw_qdrant,
                force_refresh=True,
            ) as token:
                assert token.resolved.verified is True
                assert token.resolved.profile_input_type_required is True
                assert token.resolved.profile_input_type_default is None
                assert token.resolved.profile_input_type_missing_role_behavior == "reject"
                assert token.resolved.index_compatibility_id == persisted["index_compatibility_id"]
                assert token.resolved.query_compatibility_id == persisted["query_compatibility_id"]
                assert token.snapshot_drift.detected is True
                assert token.snapshot_drift.stored_sha256 == persisted["contract_snapshot_sha256"]
                assert token.snapshot_drift.current_sha256 != persisted["contract_snapshot_sha256"]
                # Job provenance remains bound to the immutable historical
                # index snapshot instead of the operative Strict snapshot.
                assert (
                    token.job_fields()["contract_snapshot_sha256"]
                    == persisted["contract_snapshot_sha256"]
                )
                validate_job_contract_snapshot(
                    {
                        "embedding_model": target.requested_model,
                        **token.job_fields(),
                    },
                    token,
                    operation=operation,
                )
                downstream_provider(operation)

    assert downstream_provider.call_count == 3
    metadata_after = {
        table: test_db.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
        for table in ("index_contracts", "index_contract_evidence", "ingestion_jobs")
    }
    assert metadata_after == metadata_before
    for method in (
        "upsert",
        "delete",
        "create_collection",
        "delete_collection",
        "set_payload",
        "create_payload_index",
    ):
        getattr(raw_qdrant, method).assert_not_called()


@pytest.mark.parametrize("current_failure", ["malformed", "unverified", "stale_id"])
def test_real_resolver_keeps_invalid_current_contracts_fail_closed(current_failure):
    provider_contract = _role_sensitive_contract()
    current_contract = _strict_role_sensitive_contract()
    if current_failure == "malformed":
        current_contract["schema_version"] = 2
    elif current_failure == "unverified":
        profile = current_contract["profiles"][0]
        profile["verification"] = {
            "status": "unverified",
            "evidence": [],
            "blocking_reasons": ["fixture_not_verified"],
        }
        profile["index_compatibility_id"] = None
        profile["query_compatibility_id"] = None
    else:
        current_contract["profiles"][0]["index_compatibility_id"] = "sha256:" + "f" * 64

    backend = {
        "id": 704,
        "name": "KIron current-contract negative fixture",
        "category": "embedding",
        "provider": "ollama",
        "url": "http://s4-profile-selection.test",
        "model": provider_contract["canonical_model_id"],
        "priority": 1,
        "extra_config": json.dumps(
            {"embedding_profile_id": provider_contract["profiles"][0]["profile_id"]}
        ),
    }
    target = make_index_target(
        _config(),
        collection_name="kb_current_contract_negative",
        vector_kind=IndexVectorKind.DENSE,
        requested_model=provider_contract["canonical_model_id"],
    )
    with patch(
        "kiara.retrieval.index_compatibility.discover_server_contracts",
        return_value=_server_discovery(provider_contract),
    ):
        provider_resolved = resolve_current_index_contract(
            _backend_connection(backend),
            _config(),
            target,
            force_refresh=True,
        )
    with patch(
        "kiara.retrieval.index_compatibility.discover_server_contracts",
        return_value=_server_discovery(current_contract),
    ):
        current = resolve_current_index_contract(
            _backend_connection(backend),
            _config(),
            target,
            force_refresh=True,
        )

    assert provider_resolved.verified is True
    assert current.verified is False
    with pytest.raises(IndexCompatibilityError) as error:
        _validate_stored_contract("read", target, _stored(provider_resolved), current)
    assert error.value.reason_code == "index_current_profile_not_verified"


def test_logical_key_is_physical_kind_aware_and_secret_free():
    dense = _target(IndexVectorKind.DENSE)
    late = _target(IndexVectorKind.LATE_CHUNKING)

    assert dense.logical_index_key != late.logical_index_key
    assert dense.store_locator == "http://qdrant.example:6333"
    assert "secret" not in dense.logical_index_key
    assert "password" not in dense.logical_index_key
    assert dense.requested_model == "model"


@pytest.mark.parametrize(
    ("field", "current", "reason_code"),
    [
        ("backend_id", 8, "index_backend_mismatch"),
        ("backend_provider", "other", "index_backend_provider_mismatch"),
        ("backend_url", "http://other", "index_backend_url_mismatch"),
        ("canonical_model_id", "other", "index_canonical_model_mismatch"),
        ("profile_id", "other", "index_profile_mismatch"),
        ("profile_kind", "late_chunking", "index_profile_kind_mismatch"),
        ("profile_endpoint", "/api/embed_late", "index_profile_endpoint_mismatch"),
        ("contract_schema_version", 2, "index_contract_schema_version_mismatch"),
        ("contract_schema_sha256", "other", "index_contract_schema_hash_mismatch"),
        ("index_compatibility_id", "sha256:" + "3" * 64, "index_compatibility_id_mismatch"),
        ("query_compatibility_id", "sha256:" + "4" * 64, "index_query_compatibility_id_mismatch"),
        ("backend_type", "other", "index_backend_type_mismatch"),
        ("backend_implementation", "other", "index_backend_implementation_mismatch"),
        (
            "backend_implementation_revision",
            "other",
            "index_backend_implementation_revision_mismatch",
        ),
        ("artifact_repository", "other", "index_artifact_repository_mismatch"),
        ("artifact_revision", "other", "index_artifact_revision_mismatch"),
        ("artifact_manifest_digest", "other", "index_artifact_manifest_mismatch"),
        ("artifact_json_text", '{"repository":"other"}', "index_artifact_mismatch"),
    ],
)
def test_verified_guard_rejects_every_identity_mismatch(field, current, reason_code):
    resolved = replace(_resolved(), **{field: current})
    with pytest.raises(IndexCompatibilityError) as error:
        _validate_stored_contract("query", resolved.target, _stored(), resolved)
    assert error.value.reason_code == reason_code


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ({"contract_snapshot_json_text": "{"}, "index_stored_snapshot_malformed"),
        (
            {"contract_snapshot_canonical_text": '{"schema_version":2}'},
            "index_stored_snapshot_canonical_mismatch",
        ),
        ({"contract_snapshot_sha256": "0" * 64}, "index_stored_snapshot_hash_mismatch"),
        (
            {"evidence_contract_snapshot_sha256": None},
            "index_stored_snapshot_evidence_hash_missing",
        ),
    ],
)
def test_guard_revalidates_persisted_raw_snapshot_bundle(mutation, reason_code):
    stored = _stored()
    stored.update(mutation)
    with pytest.raises(IndexCompatibilityError) as error:
        _validate_stored_contract("read", _target(), stored, _resolved())
    assert error.value.reason_code == reason_code


def test_guard_rejects_consistently_rewritten_snapshot_bundle_against_evidence_anchor():
    stored = _stored()
    rewritten = {"schema_version": 1, "tampered": True}
    canonical = canonical_json(rewritten)
    stored.update(
        {
            "contract_snapshot_json_text": json.dumps(rewritten, separators=(",", ":")),
            "contract_snapshot_canonical_text": canonical,
            "contract_snapshot_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        }
    )

    with pytest.raises(IndexCompatibilityError) as error:
        _validate_stored_contract("read", _target(), stored, _resolved())

    assert error.value.reason_code == "index_stored_snapshot_evidence_hash_mismatch"


def test_verified_guard_reports_complete_snapshot_drift_without_treating_it_as_identity():
    current_raw = '{"enforcement":"strict","schema_version":1}'
    current = replace(
        _resolved(),
        contract_snapshot_json_text=current_raw,
        contract_snapshot_canonical_text=current_raw,
        contract_snapshot_sha256=hashlib.sha256(current_raw.encode()).hexdigest(),
    )

    drift = _validate_stored_contract("query", current.target, _stored(), current)

    assert drift.detected is True
    assert drift.raw_changed is True
    assert drift.canonical_changed is True
    assert drift.sha256_changed is True
    assert drift.stored_sha256 == _stored()["contract_snapshot_sha256"]
    assert drift.current_sha256 == current.contract_snapshot_sha256
    assert drift.to_dict()["reason_code"] == "index_contract_snapshot_drift"
    assert "enforcement" not in json.dumps(drift.to_dict())


def test_guard_rejects_requested_model_binding_mismatch():
    stored = _stored()
    stored["requested_model"] = "other-model"
    with pytest.raises(IndexCompatibilityError) as error:
        _validate_stored_contract("read", _target(), stored, _resolved())
    assert error.value.reason_code == "index_requested_model_mismatch"


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        ("unverified", "index_contract_unverified"),
        ("evidence_failed", "index_contract_evidence_failed"),
        ("missing", "index_contract_state_invalid"),
    ],
)
def test_guard_rejects_non_verified_persisted_states(state, reason):
    stored = _stored()
    stored["verification_state"] = state
    with pytest.raises(IndexCompatibilityError) as error:
        _validate_stored_contract("read", _target(), stored, _resolved())
    assert error.value.reason_code == reason


def test_guard_rejects_current_unverified_even_with_old_verified_row():
    current = replace(
        _resolved(),
        state=ContractState.UNVERIFIED,
        reason_code="profile_unverified",
        index_compatibility_id=None,
        query_compatibility_id=None,
    )
    with pytest.raises(IndexCompatibilityError) as error:
        _validate_stored_contract("read", current.target, _stored(), current)
    assert error.value.reason_code == "index_current_profile_not_verified"


def test_witness_accepts_abs_threshold_exactly_and_requires_both_limits():
    passed, reason, max_abs, max_cos, _ = compare_witness_vectors(
        [[1.0, 0.0]],
        [[1.0 + WITNESS_ABS_TOLERANCE, 0.0]],
    )
    assert passed is True
    assert reason == "witness_verified"
    assert max_abs == pytest.approx(WITNESS_ABS_TOLERANCE)
    assert max_cos == pytest.approx(0.0)

    passed, reason, *_ = compare_witness_vectors(
        [[1.0, 0.0]],
        [[1.0 + WITNESS_ABS_TOLERANCE + 1e-10, 0.0]],
    )
    assert passed is False
    assert reason == "witness_tolerance_exceeded"


def test_witness_cosine_threshold_edge_independent_of_abs_limit():
    # Twenty orthogonal 1e-3 components remain just below 1e-5 cosine
    # distance; the 21st crosses it while max-abs remains exactly 1e-3.
    stored_20 = [1.0, *([0.0] * 20)]
    generated_20 = [1.0, *([0.001] * 20)]
    passed, _, max_abs, max_cos, _ = compare_witness_vectors([stored_20], [generated_20])
    assert passed is True
    assert max_abs == pytest.approx(WITNESS_ABS_TOLERANCE)
    assert max_cos <= WITNESS_COSINE_TOLERANCE

    stored_21 = [1.0, *([0.0] * 21)]
    generated_21 = [1.0, *([0.001] * 21)]
    passed, reason, max_abs, max_cos, _ = compare_witness_vectors([stored_21], [generated_21])
    assert passed is False
    assert reason == "witness_tolerance_exceeded"
    assert max_abs == pytest.approx(WITNESS_ABS_TOLERANCE)
    assert max_cos > WITNESS_COSINE_TOLERANCE


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_witness_rejects_every_non_finite_value(bad):
    passed, reason, *_ = compare_witness_vectors([[1.0, 0.0]], [[1.0, bad]])
    assert passed is False
    assert reason == "witness_non_finite"


def test_witness_rejects_cardinality_shape_and_row_order_changes():
    assert compare_witness_vectors([[1.0]], [])[1] == "witness_cardinality_mismatch"
    assert compare_witness_vectors([[1.0, 0.0]], [[1.0]])[1] == "witness_shape_mismatch"
    passed, reason, *_ = compare_witness_vectors(
        [[[1.0, 0.0], [0.0, 1.0]]],
        [[[0.0, 1.0], [1.0, 0.0]]],
    )
    assert passed is False
    assert reason == "witness_tolerance_exceeded"


def test_readonly_qdrant_facade_forbids_every_mutating_entrypoint():
    raw = MagicMock()
    readonly = ReadOnlyQdrantClient(raw)
    readonly.get_collection("x")
    raw.get_collection.assert_called_once_with("x")
    for method in ("upsert", "delete", "create_collection", "delete_collection", "set_payload"):
        with pytest.raises(RuntimeError, match=f"index_backfill_qdrant_write_forbidden:{method}"):
            getattr(readonly, method)(collection_name="x")


def test_witness_insufficient_text_references_never_calls_backend(monkeypatch):
    point = MagicMock()
    point.id = "a" * 32
    point.payload = {"source": "without-text"}
    client = MagicMock()
    client.scroll.side_effect = [([point], None)]
    readonly = ReadOnlyQdrantClient(client)
    backend_call = MagicMock()
    monkeypatch.setattr(
        "kiara.retrieval.index_contract_backfill._request_witness_vectors",
        backend_call,
    )

    result = run_vector_witness(readonly, _target(), _resolved())

    assert result.state == "evidence_failed"
    assert result.reason_code == "witness_insufficient_references"
    assert result.sample_count == 0
    backend_call.assert_not_called()


def test_migration_is_idempotent_and_additive_in_test_schema(test_db):
    migrate_index_contract_schema(test_db)
    migrate_index_contract_schema(test_db)

    tables = test_db.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name LIKE 'index_contract%%' "
        "ORDER BY table_name"
    ).fetchall()
    assert [row["table_name"] for row in tables] == [
        "index_contract_evidence",
        "index_contracts",
    ]
    columns = test_db.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'ingestion_jobs' "
        "AND left(column_name, 9) = 'contract_' ORDER BY column_name"
    ).fetchall()
    assert len(columns) == 9


def test_migration_witness_minimum_constraint_and_transaction_rollback(test_db):
    migrate_index_contract_schema(test_db)
    test_db.execute("SAVEPOINT s4_witness_constraint")
    with pytest.raises(Exception):
        test_db.execute(
            "INSERT INTO index_contract_evidence "
            "(logical_index_key, evidence_type, result_state, reason_code, sample_count, "
            "references_sha256, diagnostics_sha256) "
            "VALUES ('key', 'witness', 'verified', 'bad', 15, 'r', 'd')"
        )
    test_db.execute("ROLLBACK TO SAVEPOINT s4_witness_constraint")
    count = test_db.execute(
        "SELECT COUNT(*) AS cnt FROM index_contract_evidence WHERE logical_index_key = 'key'"
    ).fetchone()["cnt"]
    assert count == 0


def test_cache_guard_runs_before_existing_hit():
    from kiara.cache.query_cache import QueryCache

    cache = QueryCache({"cache": {"enabled": True}})
    key = "legacy-key-that-must-never-be-read"
    cache._cache[key] = (0.0, [{"text": "must-not-leak"}])

    @contextlib.contextmanager
    def blocked(_operation):
        raise IndexCompatibilityError(
            "index_contract_mismatch",
            "cache_read",
            _target(),
            "test",
        )
        yield

    cache._contract_guard = blocked
    with pytest.raises(IndexCompatibilityError):
        cache.get("query")


def test_hybrid_path_never_converts_contract_error_to_empty_results():
    from kiara.retrieval.retriever import HybridRetriever

    retriever = object.__new__(HybridRetriever)
    error = IndexCompatibilityError(
        "index_contract_mismatch",
        "query",
        _target(),
        "test",
    )
    with pytest.raises(IndexCompatibilityError) as caught:
        retriever._run_retrieval_path(
            "semantic",
            enabled=True,
            callback=lambda: (_ for _ in ()).throw(error),
        )
    assert caught.value is error


def test_vectorstore_guard_precedes_every_constructor_qdrant_side_effect(monkeypatch):
    from kiara.retrieval import vectorstore_qdrant as module

    raw = MagicMock()
    monkeypatch.setattr(module, "QdrantClient", MagicMock(return_value=raw))

    @contextlib.contextmanager
    def blocked(*_args, **_kwargs):
        raise IndexCompatibilityError(
            "index_contract_missing",
            "read",
            _target(),
            "test",
        )
        yield

    monkeypatch.setattr("kiara.retrieval.index_compatibility.guard_index", blocked)
    with pytest.raises(IndexCompatibilityError):
        module.QdrantVectorStore(
            {
                "vectordb": {"qdrant_url": "http://qdrant.example:6333"},
                "ollama": {"embedding_model": "model"},
            },
            embedding_model="model",
        )
    raw.get_collections.assert_not_called()
    raw.create_collection.assert_not_called()
    raw.create_payload_index.assert_not_called()


@pytest.mark.parametrize("late", [False, True])
def test_embedding_nested_guard_reuses_client_but_keeps_validation(monkeypatch, late):
    from kiara.retrieval import embedding
    from kiara.retrieval import index_compatibility as module
    from kiara.retrieval.embedding_roles import SEARCH_DOCUMENT

    target = _target(IndexVectorKind.LATE_CHUNKING if late else IndexVectorKind.DENSE)
    token = SimpleNamespace(
        target=target,
        write=True,
        require_configured_default=False,
        validation_scope=module.IndexValidationScope.LIVE_PROFILE,
    )
    factory = MagicMock(side_effect=AssertionError("redundant client construction"))
    monkeypatch.setattr(module, "raw_qdrant_client", factory)
    profile = object()
    monkeypatch.setattr(embedding, "request_profile_from_guard", lambda *a, **k: profile)
    reset = module._active_tokens.set((token,))
    try:
        decorate = (
            embedding._guard_late_embedding("late_embedding")
            if late
            else embedding._guard_dense_embedding("document_embedding", SEARCH_DOCUMENT)
        )

        @decorate
        def embed(_config, _model, *, _request_profile):
            return _request_profile

        assert embed(_config(), "model") is profile
        token.validation_scope = module.IndexValidationScope.STORED_BINDING
        with pytest.raises(
            IndexCompatibilityError, match="index_guard_validation_scope_escalation"
        ):
            embed(_config(), "model")
    finally:
        module._active_tokens.reset(reset)
    factory.assert_not_called()


def test_owned_guard_client_is_closed_after_failure(monkeypatch):
    from kiara.retrieval import index_compatibility as module

    raw = MagicMock()
    factory = MagicMock(return_value=raw)
    monkeypatch.setattr(module, "raw_qdrant_client", factory)
    monkeypatch.setattr(
        module, "_collection_state", MagicMock(side_effect=RuntimeError("transport"))
    )
    with pytest.raises(Exception):
        with module.guard_index(_config(), _target(), operation="read"):
            pytest.fail("must fail before yield")
    factory.assert_called_once()
    raw.close.assert_called_once()


def test_colbert_and_late_cannot_reuse_dense_logical_proof():
    dense = _target(IndexVectorKind.DENSE)
    late = _target(IndexVectorKind.LATE_CHUNKING)
    multi = _target(IndexVectorKind.MULTI_VECTOR)
    assert len({dense.logical_index_key, late.logical_index_key, multi.logical_index_key}) == 3


def test_colbert_backend_resolution_never_uses_health_failover():
    rows = [
        {"id": 1, "model": "primary", "url": "http://primary"},
        {"id": 2, "model": "fallback", "url": "http://fallback"},
    ]
    backend, url = _colbert_backend(_config(), rows)
    assert backend["id"] == 1
    assert url == "http://primary"


def test_migration_parallel_starts_are_serialized_and_idempotent(test_db):
    from kiara.shared.db import get_connection

    barrier = threading.Barrier(4)

    def migrate_once():
        conn = get_connection()
        try:
            barrier.wait(timeout=5)
            migrate_index_contract_schema(conn)
            conn.commit()
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(migrate_once) for _ in range(4)]
        for future in futures:
            future.result(timeout=10)

    assert test_db.execute("SELECT COUNT(*) AS cnt FROM index_contracts").fetchone()["cnt"] == 0
    constraints = test_db.execute(
        "SELECT COUNT(*) AS cnt FROM pg_constraint "
        "WHERE conrelid = 'public.index_contracts'::regclass"
    ).fetchone()["cnt"]
    assert constraints >= 9


def test_current_migration_skips_ddl_under_concurrent_ingestion_job_reader(test_db):
    from kiara.shared.db import get_connection
    from kiara.web.index_contract_migration import _schema_is_current

    assert _schema_is_current(test_db) is True
    holder = get_connection()
    holder.execute("LOCK TABLE public.ingestion_jobs IN ACCESS SHARE MODE")

    def migrate_current_schema():
        conn = get_connection()
        try:
            migrate_index_contract_schema(conn)
            timeout = conn.execute("SHOW statement_timeout").fetchone()["statement_timeout"]
            conn.commit()
            return timeout
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(migrate_current_schema)
        try:
            timeout = future.result(timeout=2)
        except FutureTimeoutError:
            holder.rollback()
            future.result(timeout=5)
            pytest.fail("current-schema migration attempted blocking DDL")
        finally:
            holder.rollback()
            holder.close()

    assert timeout == "5min"


def test_migration_partial_failure_rolls_back_transactionally(test_db, monkeypatch):
    from kiara.shared.db import db_transaction
    from kiara.web import index_contract_migration as module

    def fail_after_partial_ddl(conn):
        conn.execute("CREATE TABLE public.s4_partial_marker (id integer)")
        raise RuntimeError("injected_migration_failure")

    monkeypatch.setattr(module, "_schema_is_current", lambda _conn: False)
    monkeypatch.setattr(module, "_create_contract_table", fail_after_partial_ddl)
    with pytest.raises(RuntimeError, match="injected_migration_failure"):
        with db_transaction() as conn:
            module.migrate_index_contract_schema(conn)

    marker = test_db.execute("SELECT to_regclass('public.s4_partial_marker') AS name").fetchone()
    assert marker["name"] is None


def test_contract_snapshot_repository_reads_are_detached(test_db):
    from kiara.repositories.index_contract import IndexContractRepository

    persisted = _persist_contract(test_db)
    test_db.commit()

    repository = IndexContractRepository()
    first = repository.get_by_logical_key(persisted["logical_index_key"])
    assert first["contract_snapshot_json_text"] == '{"schema_version":1}'
    first["contract_snapshot_json_text"] = "mutated"
    second = repository.get_by_logical_key(persisted["logical_index_key"])
    assert second["contract_snapshot_json_text"] == '{"schema_version":1}'


def test_physical_collection_binding_is_unique_across_vector_kinds(test_db):
    dense = _target(IndexVectorKind.DENSE)
    _persist_contract(test_db, dense, _resolved(dense))
    late = make_index_target(
        _config(),
        collection_name=dense.collection_name,
        vector_kind=IndexVectorKind.LATE_CHUNKING,
        requested_model=dense.requested_model,
    )
    test_db.execute("SAVEPOINT s4_physical_unique")
    with pytest.raises(Exception):
        _persist_contract(test_db, late, _resolved(late))
    test_db.execute("ROLLBACK TO SAVEPOINT s4_physical_unique")
    assert test_db.execute("SELECT COUNT(*) AS cnt FROM index_contracts").fetchone()["cnt"] == 1


def test_empty_first_write_initializes_metadata_before_any_vector_write(test_db, monkeypatch):
    from kiara.retrieval import index_compatibility as module

    target = _target()
    monkeypatch.setattr(module, "_collection_state", lambda *_args: (False, 0))
    monkeypatch.setattr(
        module,
        "resolve_current_index_contract",
        lambda _conn, _config, current_target, **_kwargs: _resolved(current_target),
    )
    raw = MagicMock()
    with guard_index(
        _config(),
        target,
        operation="first_write_test",
        write=True,
        allow_initialize=True,
        qdrant_client=raw,
    ) as token:
        assert token.stored["evidence_type"] == "empty_first_write"
        assert token.stored["observed_points_count"] == 0
        raw.upsert.assert_not_called()

    row = test_db.execute(
        "SELECT verification_state, evidence_type FROM index_contracts "
        "WHERE logical_index_key = %s",
        (target.logical_index_key,),
    ).fetchone()
    assert dict(row) == {
        "verification_state": "verified",
        "evidence_type": "empty_first_write",
    }


def test_nonempty_legacy_index_cannot_use_first_write_initialization(test_db, monkeypatch):
    from kiara.retrieval import index_compatibility as module

    target = _target()
    monkeypatch.setattr(module, "_collection_state", lambda *_args: (True, 17))
    monkeypatch.setattr(
        module,
        "resolve_current_index_contract",
        lambda _conn, _config, current_target, **_kwargs: _resolved(current_target),
    )
    with pytest.raises(IndexCompatibilityError) as error:
        with guard_index(
            _config(),
            target,
            operation="first_write_test",
            write=True,
            allow_initialize=True,
            qdrant_client=MagicMock(),
        ):
            pass
    assert error.value.reason_code == "index_collection_nonempty_without_contract"
    assert test_db.execute("SELECT COUNT(*) AS cnt FROM index_contracts").fetchone()["cnt"] == 0


def test_qdrant_transport_failure_is_retryable_index_dependency_outage(test_db):
    from qdrant_client.http.exceptions import ResponseHandlingException

    source = httpx.RemoteProtocolError("Server disconnected without sending a response.")
    wrapped = ResponseHandlingException(source)
    client = MagicMock()
    client.get_collection.side_effect = wrapped

    with pytest.raises(IndexContractUnavailableError) as caught:
        with guard_index(
            _config(),
            _target(),
            operation="job_claim",
            write=True,
            qdrant_client=client,
        ):
            pass

    assert caught.value.__cause__ is wrapped
    assert caught.value.reason_code == "qdrant_temporarily_unavailable"
    assert caught.value.current_state == "unavailable"
    diagnostic = caught.value.to_dict()
    assert diagnostic["retryable"] is True
    assert diagnostic["issues"] == [
        {
            "reason_code": "qdrant_request_failed",
            "field_path": "http://qdrant.example:6333",
            "detail": "RemoteProtocolError: Server disconnected without sending a response.",
        }
    ]


def test_qdrant_response_validation_failure_is_not_retryable():
    from qdrant_client.http.exceptions import ResponseHandlingException

    wrapped = ResponseHandlingException(ValueError("invalid collection response schema"))

    assert _qdrant_unavailability_cause(wrapped) is None


@pytest.mark.parametrize("status, retryable", [(401, False), (503, True)])
def test_qdrant_http_status_retryability_is_explicit(status, retryable):
    from qdrant_client.http.exceptions import UnexpectedResponse

    error = UnexpectedResponse(
        status_code=status,
        reason_phrase="test",
        content=b"{}",
        headers=httpx.Headers(),
    )

    assert (_qdrant_unavailability_cause(error) is error) is retryable


def test_exactly_sixteen_witness_references_pass_without_qdrant_writes(monkeypatch):
    points = []
    for index in range(16):
        point = MagicMock()
        point.id = f"{index + 1:032x}"
        point.payload = {
            "_text": f"Referenz {index}",
            "source": f"/docs/{index}.txt",
            "directory": "/docs",
            "file_type": "txt",
            "ds_ids": [index % 3 + 1],
        }
        point.vector = [1.0, 0.0]
        points.append(point)
    raw = MagicMock()
    raw.scroll.return_value = (points, None)
    raw.retrieve.return_value = points
    readonly = ReadOnlyQdrantClient(raw)
    backend = MagicMock(return_value=[[1.0, 0.0] for _ in range(16)])
    monkeypatch.setattr(
        "kiara.retrieval.index_contract_backfill._request_witness_vectors",
        backend,
    )

    result = run_vector_witness(readonly, _target(), _resolved(), reference_count=16)

    assert result.passed is True
    assert result.sample_count == 16
    assert result.max_abs_error == 0.0
    assert result.max_cosine_distance == 0.0
    backend.assert_called_once()
    for method in ("upsert", "delete", "create_collection", "delete_collection"):
        getattr(raw, method).assert_not_called()


@pytest.mark.parametrize("mode", ["partial", "backend_error"])
def test_witness_partial_or_backend_failure_never_partially_passes(monkeypatch, mode):
    references = [{"point_id": f"{index + 1:032x}", "text": f"Text {index}"} for index in range(16)]
    monkeypatch.setattr(
        "kiara.retrieval.index_contract_backfill.select_witness_references",
        lambda *_args, **_kwargs: references,
    )
    monkeypatch.setattr(
        "kiara.retrieval.index_contract_backfill._stored_vectors_for_references",
        lambda *_args, **_kwargs: [[1.0, 0.0] for _ in range(16)],
    )
    if mode == "partial":

        def replacement(*_args, **_kwargs):
            return [[1.0, 0.0] for _ in range(15)]
    else:

        def replacement(*_args, **_kwargs):
            raise httpx.ConnectError("backend unavailable")

    monkeypatch.setattr(
        "kiara.retrieval.index_contract_backfill._request_witness_vectors",
        replacement,
    )
    result = run_vector_witness(ReadOnlyQdrantClient(MagicMock()), _target(), _resolved())
    assert result.state == "evidence_failed"
    assert result.reason_code == (
        "witness_cardinality_mismatch" if mode == "partial" else "witness_backend_error"
    )


def test_backfill_dry_run_apply_and_rollback_never_write_qdrant(test_db, monkeypatch):
    from kiara.retrieval import index_contract_backfill as module

    target = _target()
    unresolved = replace(
        _resolved(target),
        state=ContractState.UNVERIFIED,
        reason_code="contract_unverified",
        index_compatibility_id=None,
        query_compatibility_id=None,
    )
    assessment = BackfillAssessment(
        target=target,
        points_count=17,
        collection_config_sha256="c" * 64,
        resolved=unresolved,
        verification_state="unverified",
        evidence_type="unverified",
        reason_code="contract_unverified",
        witness=None,
    )
    raw = MagicMock()
    monkeypatch.setattr(module, "raw_qdrant_client", lambda _config: raw)
    monkeypatch.setattr(module, "inventory_targets", lambda *_args: [target])
    monkeypatch.setattr(module, "_assess_target", lambda *_args, **_kwargs: assessment)

    dry_run = run_index_contract_backfill(_config(), apply=False)
    assert dry_run["transaction"] == "read_only_no_metadata_writes"
    assert dry_run["qdrant_write_operations"] == 0
    assert test_db.execute("SELECT COUNT(*) AS cnt FROM index_contracts").fetchone()["cnt"] == 0

    applied = run_index_contract_backfill(_config(), apply=True)
    assert applied["contracts_inserted"] == 1
    row = test_db.execute(
        "SELECT verification_state, index_compatibility_id, query_compatibility_id "
        "FROM index_contracts"
    ).fetchone()
    assert dict(row) == {
        "verification_state": "unverified",
        "index_compatibility_id": None,
        "query_compatibility_id": None,
    }

    rolled_back = rollback_backfill_metadata(_config(), applied["run_id"])
    assert rolled_back["contracts_removed"] == 1
    assert rolled_back["qdrant_write_operations"] == 0
    assert test_db.execute("SELECT COUNT(*) AS cnt FROM index_contracts").fetchone()["cnt"] == 0
    for method in (
        "upsert",
        "delete",
        "create_collection",
        "delete_collection",
        "set_payload",
        "create_payload_index",
    ):
        getattr(raw, method).assert_not_called()


@pytest.mark.parametrize(
    ("operation", "write"),
    [
        ("query", False),
        ("job_enqueue", True),
        ("job_claim", True),
        ("vector_upsert", True),
        ("full_reingest_cleanup", True),
    ],
)
def test_config_change_cannot_cross_any_guard_window(
    test_db,
    monkeypatch,
    operation,
    write,
):
    from kiara.retrieval import index_compatibility as module
    from kiara.shared.db import get_connection

    target = _target()
    _persist_contract(test_db, target, _resolved(target))
    test_db.execute(
        "INSERT INTO ki_backends "
        "(id, category, provider, name, url, model, is_active, priority, extra_config) "
        "VALUES (700, 'embedding', 'ollama', 'guard-race', 'http://kiron:11434', "
        "'model', true, 10, '{}')"
    )
    test_db.commit()
    monkeypatch.setattr(module, "_collection_state", lambda *_args: (True, 1))
    monkeypatch.setattr(
        module,
        "resolve_current_index_contract",
        lambda _conn, _config, current_target, **_kwargs: _resolved(current_target),
    )

    entered = threading.Event()
    release = threading.Event()
    update_started = threading.Event()
    update_done = threading.Event()

    def guarded_window():
        with guard_index(
            _config(),
            target,
            operation=operation,
            write=write,
            qdrant_client=MagicMock(),
        ):
            entered.set()
            assert release.wait(5)

    def config_change():
        conn = get_connection()
        try:
            update_started.set()
            conn.execute("UPDATE ki_backends SET priority = priority + 1 WHERE id = 700")
            conn.commit()
            update_done.set()
        finally:
            conn.close()

    guard_thread = threading.Thread(target=guarded_window)
    update_thread = threading.Thread(target=config_change)
    guard_thread.start()
    assert entered.wait(5)
    update_thread.start()
    assert update_started.wait(5)
    time.sleep(0.1)
    assert not update_done.is_set()
    release.set()
    guard_thread.join(timeout=5)
    update_thread.join(timeout=5)
    assert update_done.is_set()


def test_default_dense_guard_rejects_stale_model_before_qdrant_access(test_db, monkeypatch):
    """A model selected before the config lock cannot outlive a switch."""
    from kiara.retrieval import index_compatibility as module

    target = _target()
    _persist_contract(test_db, target, _resolved(target))
    test_db.execute(
        "INSERT INTO ki_backends "
        "(id, category, provider, name, url, model, is_active, priority, extra_config) "
        "VALUES (701, 'embedding', 'ollama', 'new-default', 'http://kiron:11434', "
        "'other-model', true, 1, '{}')"
    )
    test_db.commit()
    collection_probe = MagicMock()
    monkeypatch.setattr(module, "_collection_state", collection_probe)

    with pytest.raises(IndexCompatibilityError) as error:
        with guard_index(
            _config(),
            target,
            operation="query",
            write=False,
            require_configured_default=True,
            qdrant_client=MagicMock(),
        ):
            pass

    assert error.value.reason_code == "index_configured_default_model_mismatch"
    collection_probe.assert_not_called()


def test_elasticsearch_rebuild_guard_precedes_every_es_side_effect(monkeypatch):
    from kiara.retrieval import elasticsearch_service as module

    es_factory = MagicMock()
    monkeypatch.setattr(module, "_get_client", es_factory)
    monkeypatch.setattr("kiara.config.get_embedding_model", lambda _config: "model")

    @contextlib.contextmanager
    def blocked(*_args, **_kwargs):
        raise IndexCompatibilityError(
            "index_contract_mismatch",
            "elasticsearch_rebuild_from_vectors",
            _target(),
            "test",
        )
        yield

    monkeypatch.setattr("kiara.retrieval.index_compatibility.guard_index", blocked)
    with pytest.raises(IndexCompatibilityError):
        module.rebuild_from_vectorstore(_config())
    es_factory.assert_not_called()


def test_job_enqueue_guard_failure_leaves_queue_unchanged(test_db, monkeypatch):
    from kiara.retrieval import index_compatibility as module
    from kiara.web import ingestion_worker

    ds_id = test_db.execute(
        "INSERT INTO datasources (name, type, is_active) "
        "VALUES ('s4-enqueue-boundary', 'local', true) RETURNING id"
    ).fetchone()["id"]
    test_db.commit()
    monkeypatch.setattr("kiara.config.load_config", lambda: _config())
    monkeypatch.setattr(module, "raw_qdrant_client", lambda _config: MagicMock())

    @contextlib.contextmanager
    def blocked(*_args, **_kwargs):
        raise IndexCompatibilityError(
            "index_contract_missing",
            "job_enqueue",
            _target(),
            "test",
        )
        yield

    monkeypatch.setattr(module, "guard_index", blocked)
    with pytest.raises(IndexCompatibilityError):
        ingestion_worker.enqueue_job(ds_id, embedding_model="model")

    queued = test_db.execute(
        "SELECT COUNT(*) AS cnt FROM ingestion_jobs WHERE datasource_id = %s",
        (ds_id,),
    ).fetchone()["cnt"]
    assert queued == 0


def test_colbert_payload_batch_does_not_continue_after_guard_error():
    from kiara.retrieval.colbert import ColBERTStore

    error = _guard_error("colbert_payload_read", IndexVectorKind.MULTI_VECTOR)
    store = object.__new__(ColBERTStore)
    store._collection = "colbert_guard"
    store._client = MagicMock()
    store._client.retrieve.side_effect = error

    with pytest.raises(IndexCompatibilityError) as caught:
        ColBERTStore.ids_with_datasource.__wrapped__(
            store,
            ["a" * 32, "b" * 32],
            7,
        )

    assert caught.value is error
    store._client.retrieve.assert_called_once()


def test_colbert_count_does_not_degrade_guard_error_to_zero():
    from kiara.retrieval.colbert import ColBERTStore

    error = _guard_error("colbert_count", IndexVectorKind.MULTI_VECTOR)
    store = object.__new__(ColBERTStore)
    store._collection = "colbert_guard"
    store._client = MagicMock()
    store._client.get_collection.side_effect = error

    with pytest.raises(IndexCompatibilityError) as caught:
        ColBERTStore.get_point_count.__wrapped__(store)

    assert caught.value is error
    store._client.get_collection.assert_called_once_with("colbert_guard")


def test_qdrant_payload_read_does_not_retry_or_wrap_guard_error():
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    error = _guard_error("vector_payload_read")
    store = object.__new__(QdrantVectorStore)
    store._collection = "kb_guard"
    store._client = MagicMock()
    store._client.retrieve.side_effect = error

    with patch("time.sleep") as sleep:
        with pytest.raises(IndexCompatibilityError) as caught:
            QdrantVectorStore.get_documents_by_ids.__wrapped__(
                store,
                ["a" * 32],
            )

    assert caught.value is error
    store._client.retrieve.assert_called_once()
    sleep.assert_not_called()


def test_active_job_snapshot_is_revalidated_before_embedding_and_upsert(
    test_db,
    monkeypatch,
):
    from kiara.ingestion.runtime import set_current_job_id
    from kiara.retrieval import embedding
    from kiara.retrieval import index_compatibility as module
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    target = _target()
    resolved = _resolved(target)
    _persist_contract(test_db, target, resolved)
    ds_id = test_db.execute(
        "INSERT INTO datasources (name, type, is_active) "
        "VALUES ('s4-job-revalidation', 'local', true) RETURNING id"
    ).fetchone()["id"]
    job_id = test_db.execute(
        "INSERT INTO ingestion_jobs "
        "(datasource_id, embedding_model, contract_logical_index_key, "
        "contract_index_compatibility_id, contract_query_compatibility_id, "
        "contract_canonical_model_id, contract_profile_id, contract_profile_kind, "
        "contract_profile_endpoint, contract_snapshot_sha256, contract_fence_version) "
        "VALUES (%s, 'model', %s, %s, %s, %s, 'wrong-profile', %s, %s, %s, 1) "
        "RETURNING id",
        (
            ds_id,
            target.logical_index_key,
            resolved.index_compatibility_id,
            resolved.query_compatibility_id,
            resolved.canonical_model_id,
            resolved.profile_kind,
            resolved.profile_endpoint,
            resolved.contract_snapshot_sha256,
        ),
    ).fetchone()["id"]
    test_db.commit()

    monkeypatch.setattr(module, "_collection_state", lambda *_args: (True, 1))
    monkeypatch.setattr(
        module,
        "resolve_current_index_contract",
        lambda _conn, _config, current_target, **_kwargs: _resolved(current_target),
    )
    raw = MagicMock()
    monkeypatch.setattr(module, "raw_qdrant_client", lambda _config: raw)
    backend_call = MagicMock()
    monkeypatch.setattr(embedding, "_ollama_embed", backend_call)

    store = object.__new__(QdrantVectorStore)
    store._config = _config()
    store._index_target = target
    store._client = raw
    store._require_configured_default = False

    set_current_job_id(job_id)
    try:
        with pytest.raises(IndexCompatibilityError) as embed_error:
            embedding.embed_texts(_config(), "model", ["test"])
        with pytest.raises(IndexCompatibilityError) as upsert_error:
            store.add(["test"], [{}], ["a" * 32])
    finally:
        set_current_job_id(None)

    assert embed_error.value.reason_code == "index_job_contract_mismatch"
    assert upsert_error.value.reason_code == "index_job_contract_mismatch"
    backend_call.assert_not_called()
    raw.upsert.assert_not_called()


def test_full_reingest_guard_failure_precedes_every_cleanup_side_effect(
    test_db,
    monkeypatch,
):
    from kiara.retrieval import index_compatibility as module
    from kiara.web.ingestion_worker import IngestionWorker

    ds_id = test_db.execute(
        "INSERT INTO datasources (name, type, is_active) "
        "VALUES ('s4-cleanup-boundary', 'local', true) RETURNING id"
    ).fetchone()["id"]
    job_id = test_db.execute(
        "INSERT INTO ingestion_jobs (datasource_id, embedding_model, job_type) "
        "VALUES (%s, 'model', 'full') RETURNING id",
        (ds_id,),
    ).fetchone()["id"]
    test_db.commit()

    @contextlib.contextmanager
    def blocked(*_args, **_kwargs):
        raise IndexCompatibilityError(
            "index_job_contract_missing",
            "full_reingest_cleanup",
            _target(),
            "test",
        )
        yield

    monkeypatch.setattr(module, "guard_job_contract", blocked)
    vectorstore = MagicMock()
    chunk_datasources = MagicMock()
    hash_store = MagicMock()
    monkeypatch.setattr("kiara.retrieval.vectorstore.VectorStore", vectorstore)
    monkeypatch.setattr("kiara.ingestion._file_utils.ChunkDatasources", chunk_datasources)
    monkeypatch.setattr("kiara.ingestion._file_utils.HashStore", hash_store)

    worker = IngestionWorker.__new__(IngestionWorker)
    with pytest.raises(IndexCompatibilityError):
        worker._prepare_full_reingest(job_id, ds_id, _config(), "model")

    vectorstore.assert_not_called()
    chunk_datasources.assert_not_called()
    hash_store.assert_not_called()


@pytest.mark.asyncio
async def test_admin_diagnostic_failure_is_additive_and_isolated():
    from kiara.web.api_datasource_features import list_embedding_contracts

    inventory = MagicMock()
    inventory.to_dict.return_value = {
        "contract": {"schema_version": 1},
        "servers": [],
        "backends": [{"backend_name": "still-visible"}],
    }
    with (
        patch("kiara.backend_selector.get_all_active", return_value=[]),
        patch(
            "kiara.retrieval.embedding_discovery.discover_configured_embedding_backends",
            return_value=inventory,
        ),
        patch(
            "kiara.web.api_datasource_features._persisted_index_contract_diagnostics",
            side_effect=RuntimeError("diagnostic unavailable"),
        ),
    ):
        result = await list_embedding_contracts(user={"id": 1})
    assert result["backends"] == [{"backend_name": "still-visible"}]
    assert result["indexes"] == []
    assert result["index_diagnostics_error"] == "RuntimeError"


def test_admin_embedding_contract_route_requires_authentication():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kiara.web.api_datasource_features import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app, raise_server_exceptions=False).get("/api/admin/embedding-contracts")
    assert response.status_code in (401, 403)


def test_persisted_index_frontend_runtime_escapes_xss():
    source = (
        Path(__file__).parents[1] / "kiara" / "web" / "static" / "js" / "tab_ki_modelle.js"
    ).read_text(encoding="utf-8")
    harness = r"""
globalThis.SharedUi = { runtime: { escapeHtml: (value) => String(value)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;') } };
embeddingContractDiagnosticsError = '';
embeddingContractDiagnostics = {
  contract: {schema_version: 1}, backends: [], servers: [],
  indexes: [{
    collection_name: '<img src=x onerror=alert(1)>', vector_kind: 'dense',
    compatibility_state: 'mismatch<script>', reason_code: '<svg/onload=alert(2)>',
    operator_action: '<b>unsafe</b>', stored: {profile_id: '<iframe>', profile_endpoint: '/api/embed'},
    current: {index_compatibility_id: '<script>'}, evidence: {type: '<img>', sample_count: 16}
  }]
};
process.stdout.write(_embeddingContractDiagnosticsHtml());
"""
    completed = subprocess.run(
        ["node", "-"],
        input=source + "\n" + harness,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "<img" not in completed.stdout
    assert "<script" not in completed.stdout
    assert "&lt;img" in completed.stdout


def test_resolver_retries_complete_discovery_after_transient_request_failure():
    contract = _role_sensitive_contract()
    backend = {
        "id": 901,
        "name": "KIron retry fixture",
        "category": "embedding",
        "provider": "ollama",
        "url": "http://s4-profile-selection.test",
        "model": contract["canonical_model_id"],
        "priority": 1,
        "extra_config": "{}",
    }
    config = {
        **_config(),
        "index_contract": {
            "discovery_attempts": 3,
            "discovery_timeout_seconds": 1,
            "discovery_backoff_initial_seconds": 0.01,
            "discovery_backoff_max_seconds": 0.01,
        },
    }
    target = make_index_target(
        config,
        collection_name="kb_discovery_retry",
        vector_kind=IndexVectorKind.DENSE,
        requested_model=contract["canonical_model_id"],
    )
    unavailable = ServerContractDiscovery(
        backend_url=backend["url"],
        state=ContractState.DISCOVERY_ERROR,
        groups=(),
        plain_model_names=(),
        issues=(
            ContractIssue(
                "contract_discovery_request_failed",
                "/api/tags",
                "ReadTimeout",
            ),
        ),
    )

    with (
        patch(
            "kiara.retrieval.index_compatibility.discover_server_contracts",
            side_effect=[unavailable, _server_discovery(contract)],
        ) as discover,
        patch("kiara.retrieval.index_compatibility.time.sleep") as sleep,
    ):
        resolved = resolve_current_index_contract(
            _backend_connection(backend),
            config,
            target,
        )

    assert resolved.verified is True
    assert resolved.discovery_attempts == 2
    assert discover.call_count == 2
    assert discover.call_args_list[1].kwargs["force_refresh"] is True
    sleep.assert_called_once_with(0.01)


def test_exhausted_transient_discovery_uses_retryable_error_class():
    current = replace(
        _resolved(),
        state=ContractState.DISCOVERY_ERROR,
        reason_code="contract_discovery_request_failed",
        index_compatibility_id=None,
        query_compatibility_id=None,
        issues=(
            {
                "reason_code": "contract_discovery_request_failed",
                "field_path": "/api/tags",
                "detail": "ConnectError",
            },
        ),
        discovery_attempts=3,
    )

    with pytest.raises(IndexContractUnavailableError) as caught:
        _validate_stored_contract("vector_upsert", current.target, _stored(), current)

    diagnostic = caught.value.to_dict()
    assert diagnostic["retryable"] is True
    assert diagnostic["discovery_attempts"] == 3
    assert diagnostic["issues"][0]["detail"] == "ConnectError"


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_discovery_http_statuses_are_retryable(status):
    from kiara.retrieval.index_compatibility import _issues_are_transient_discovery

    assert _issues_are_transient_discovery(
        (
            {
                "reason_code": "contract_discovery_http_error",
                "field_path": "/api/show",
                "detail": f"HTTP {status}",
            },
            {
                "reason_code": "contract_server_invalid",
                "field_path": "/api/tags",
                "detail": "server discovery state is inconsistent",
            },
        )
    )


def test_permanent_discovery_http_status_is_not_retryable():
    from kiara.retrieval.index_compatibility import _issues_are_transient_discovery

    assert not _issues_are_transient_discovery(
        (
            {
                "reason_code": "contract_discovery_http_error",
                "field_path": "/api/show",
                "detail": "HTTP 404",
            },
        )
    )


def test_transient_issue_never_masks_parallel_contract_corruption():
    current = replace(
        _resolved(),
        state=ContractState.MALFORMED,
        reason_code="contract_discovery_request_failed",
        index_compatibility_id=None,
        query_compatibility_id=None,
        issues=(
            {
                "reason_code": "contract_discovery_request_failed",
                "field_path": "/api/show",
                "detail": "ReadTimeout",
            },
            {
                "reason_code": "contract_tags_model_malformed",
                "field_path": "/models/1",
                "detail": "model entry is not an object",
            },
        ),
        discovery_attempts=3,
    )

    with pytest.raises(IndexCompatibilityError) as caught:
        _validate_stored_contract("vector_upsert", current.target, _stored(), current)

    assert not isinstance(caught.value, IndexContractUnavailableError)


def test_stored_binding_guard_never_contacts_kiron(test_db, monkeypatch):
    from kiara.retrieval import index_compatibility as module

    target = _target()
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    monkeypatch.setattr(module, "_collection_state", lambda *_args: (True, 17))

    with patch(
        "kiara.retrieval.index_compatibility.resolve_current_index_contract",
        side_effect=AssertionError("stored binding contacted KIron"),
    ) as resolve:
        with guard_index(
            _config(),
            target,
            operation="vector_payload_update",
            write=True,
            qdrant_client=MagicMock(),
            validation_scope=IndexValidationScope.STORED_BINDING,
        ) as token:
            assert token.validation_scope is IndexValidationScope.STORED_BINDING
            assert token.resolved.reason_code == "index_stored_binding_verified"
            assert token.snapshot_drift.detected is False

    resolve.assert_not_called()


def test_stored_binding_guard_cannot_be_escalated_to_live(test_db, monkeypatch):
    from kiara.retrieval import index_compatibility as module

    target = _target()
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    monkeypatch.setattr(module, "_collection_state", lambda *_args: (True, 17))

    with guard_index(
        _config(),
        target,
        operation="vector_delete",
        write=True,
        qdrant_client=MagicMock(),
        validation_scope=IndexValidationScope.STORED_BINDING,
    ):
        with pytest.raises(IndexCompatibilityError) as caught:
            with guard_index(
                _config(),
                target,
                operation="vector_upsert",
                write=True,
                qdrant_client=MagicMock(),
            ):
                pass

    assert caught.value.reason_code == "index_guard_validation_scope_escalation"


def test_document_completion_uses_real_stored_guard_without_live_backend(test_db):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from kiara.retrieval.index_compatibility import GuardedQdrantClient
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    target = _target()
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    with contextlib.closing(QdrantClient(location=":memory:")) as raw:
        raw.create_collection(
            target.collection_name, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
        )
        cid = "d" * 32
        raw.upsert(
            target.collection_name,
            [
                PointStruct(
                    id=cid,
                    vector=[1.0, 0.0],
                    payload={
                        "_text": "Vollstaendig gespeicherter Text",
                        "source": "document.txt",
                        "source_key": "document.txt",
                        "parser": "standard",
                        "chunk_index": 0,
                        "chunk_total": 1,
                        "processing_pending": True,
                    },
                )
            ],
        )
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store._client = GuardedQdrantClient(raw, _config(), target)
        store._config, store._index_target = _config(), target
        store._collection, store._use_source_key = target.collection_name, True
        store._require_configured_default = False
        with patch(
            "kiara.retrieval.index_compatibility.resolve_current_index_contract",
            side_effect=AssertionError("Kein Live-Modell beim Dokumentabschluss"),
        ):
            store.finalize_document_versions([cid])
        payload = raw.retrieve(target.collection_name, [cid])[0].payload
        assert payload["processing_pending"] is False
        assert len(payload["processing_version"]) == 64
        assert payload["_text"] == "Vollstaendig gespeicherter Text"
