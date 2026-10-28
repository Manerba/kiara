"""P64 schema admission: real PostgreSQL fences/audit, captured KIron profiles.

Only discovery and Qdrant transport are doubled. No live index is contacted.
"""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client.models import Distance, VectorParams
from test_index_compatibility_s4 import _persist_contract, _server_discovery

from kiara.retrieval import index_compatibility as contracts
from kiara.retrieval.embedding_contract import canonical_json, parse_capabilities


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def transition_case(test_db, monkeypatch, request):
    from kiara.retrieval import index_schema_transition as transition

    assert test_db.execute("SELECT current_database() AS name").fetchone()["name"] == "kiara_test"
    kind = getattr(request, "param", contracts.IndexVectorKind.DENSE)
    historical_provider = kind == "historical_provider"
    if historical_provider:
        kind = contracts.IndexVectorKind.DENSE
    capture = json.loads(
        (Path(__file__).parent / "fixtures/p64-kiron-profiles-20260928.json").read_text()
    )["contracts"]["nomic-embed-text:latest"]
    config = {"vectordb": {"qdrant_url": "http://schema-transition.invalid:6333"}}
    target = contracts.make_index_target(
        config,
        collection_name="kb_nomic_schema_transition",
        vector_kind=kind,
        requested_model="nomic-embed-text",
    )
    profile = next(
        value
        for value in capture["profiles"]
        if value["kind"] == kind.value and value["verification"]["status"] == "verified"
    )
    backend = {
        "id": 7,
        "provider": "ollama",
        "url": "http://kiron-schema-transition.invalid:11434",
        "model": "nomic-embed-text",
        "extra_config": json.dumps({"embedding_profile_id": profile["profile_id"]}),
    }
    monkeypatch.setattr(contracts, "_backend_rows", lambda _conn: [backend])
    monkeypatch.setattr(
        contracts, "discover_server_contracts", lambda *a, **kw: _server_discovery(capture)
    )
    current = contracts.resolve_current_index_contract(test_db, config, target, force_refresh=True)
    assert current.verified
    assert current.contract_schema_sha256 == transition.NEW_SCHEMA_SHA256
    old = replace(current, contract_schema_sha256=transition.OLD_SCHEMA_SHA256)
    if historical_provider:
        snapshot = copy.deepcopy(capture)
        historical = next(p for p in snapshot["profiles"] if p["profile_id"] == current.profile_id)
        historical["input_type"].update(
            required=False, default="search_document", missing_role_behavior="use_default"
        )
        for item in historical["artifact"]["weights"] + historical["artifact"]["auxiliary"]:
            if item.get("size_bytes") is None:
                item.pop("size_bytes", None)
        parsed = parse_capabilities(snapshot)
        prior_profile = parsed.profile_by_id(current.profile_id)
        assert prior_profile.index_compatibility_id == current.index_compatibility_id
        assert prior_profile.query_compatibility_id == current.query_compatibility_id
        old = replace(
            old,
            contract_snapshot_json_text=parsed.snapshot.json_text,
            contract_snapshot_canonical_text=parsed.snapshot.canonical_json,
            contract_snapshot_sha256=parsed.snapshot.canonical_sha256,
            profile_input_type_required=False,
            profile_input_type_default="search_document",
            profile_input_type_missing_role_behavior="use_default",
        )
    stored = _persist_contract(test_db, target, old)
    test_db.commit()
    resolver = Mock(return_value=current)
    monkeypatch.setattr(contracts, "resolve_current_index_contract", resolver)
    client = Mock()
    client.collection_exists.return_value = True
    client.count.return_value = SimpleNamespace(count=1)
    client.get_collection.return_value = SimpleNamespace(
        points_count=1,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(size=profile["dimensions"], distance=Distance.COSINE),
            ),
        ),
    )

    def contract_row():
        return dict(
            test_db.execute(
                "SELECT * FROM index_contracts WHERE logical_index_key = %s",
                (target.logical_index_key,),
            ).fetchone()
        )

    def evidence_rows():
        return [
            dict(row)
            for row in test_db.execute(
                "SELECT * FROM index_contract_evidence WHERE logical_index_key = %s ORDER BY id",
                (target.logical_index_key,),
            ).fetchall()
        ]

    return SimpleNamespace(
        module=transition,
        config=config,
        target=target,
        current=current,
        old=old,
        stored=stored,
        client=client,
        resolver=resolver,
        db=test_db,
        contract=contract_row,
        evidence=evidence_rows,
        capture=capture,
    )


def _transition(case, **kwargs):
    return case.module.transition_schema(
        case.config, case.target, qdrant_client=case.client, **kwargs
    )


def _rollback(case, evidence_id, **kwargs):
    return case.module.rollback_schema_transition(
        case.config, case.target, evidence_id, qdrant_client=case.client, **kwargs
    )


def _live_guard(case):
    return contracts.guard_index(
        case.config, case.target, operation="p64_live_read", qdrant_client=case.client
    )


@pytest.mark.parametrize(
    "transition_case",
    [contracts.IndexVectorKind.DENSE, contracts.IndexVectorKind.LATE_CHUNKING],
    indirect=True,
)
def test_schema_admission_preserves_source_and_enables_normal_live_guard(transition_case):
    case = transition_case
    before = case.contract()
    evidence_before = case.evidence()
    with pytest.raises(contracts.IndexCompatibilityError) as blocked:
        with _live_guard(case):
            pytest.fail("old schema binding bypassed the normal live guard")
    assert blocked.value.reason_code == "index_contract_schema_hash_mismatch"

    assert _transition(case)["status"] == "planned"
    assert case.contract() == before
    assert case.evidence() == evidence_before
    result = _transition(case, apply=True)
    assert result["status"] == "applied"
    after = case.contract()
    assert after["contract_schema_sha256"] == case.module.NEW_SCHEMA_SHA256
    assert after["contract_snapshot_sha256"] == case.current.contract_snapshot_sha256
    assert after["fence_version"] == before["fence_version"] + 1
    assert after["evidence_id"] == result["evidence_id"] != before["evidence_id"]
    assert case.evidence()[:-1] == evidence_before
    audit = case.evidence()[-1]
    assert audit["evidence_type"] == "provenance"
    assert _sha(audit["diagnostics_json_text"]) == audit["diagnostics_sha256"]
    diagnostics = json.loads(audit["diagnostics_json_text"])
    assert diagnostics["procedure"] == "p64-schema-admission-v1"
    assert diagnostics["direction"] == "apply"
    # Real lock keys exceed IEEE-754's exact integer range; the persisted audit
    # must preserve the complete original row for repeat/rollback checks.
    assert abs(before["lock_key"]) > 2**53
    assert diagnostics["before_contract"] == json.loads(
        json.dumps(before, default=lambda value: value.isoformat())
    )
    assert diagnostics["expected_after"]["lock_key"] == before["lock_key"]
    assert diagnostics["before_contract"]["evidence_id"] == before["evidence_id"]
    assert (
        diagnostics["before_contract"]["contract_snapshot_json_text"]
        == before["contract_snapshot_json_text"]
    )
    assert diagnostics["before_evidence"]["id"] == evidence_before[0]["id"]
    assert "task:189" in json.dumps(diagnostics)
    with _live_guard(case) as token:
        assert token.fence_version == after["fence_version"]
    assert any(call.kwargs.get("force_refresh") is True for call in case.resolver.call_args_list)
    case.client.upsert.assert_not_called()
    case.client.delete.assert_not_called()
    case.client.set_payload.assert_not_called()


@pytest.mark.parametrize("transition_case", ["historical_provider"], indirect=True)
def test_historical_provider_phase_admits_current_strict_profile_without_identity_drift(
    transition_case,
):
    case = transition_case
    before, evidence_before = case.contract(), case.evidence()
    assert before["contract_snapshot_sha256"] != case.current.contract_snapshot_sha256
    assert case.current.profile_input_type_required is True
    assert case.current.profile_input_type_default is None
    assert case.current.profile_input_type_missing_role_behavior == "reject"
    # Explicit search_document/search_query requests retain the same formatting
    # and both compatibility IDs. Default-role enforcement and an absent versus
    # null optional weight size do not change their embedding identity.
    result = _transition(case, apply=True)
    assert result["status"] == "applied"
    after = case.contract()
    for field in ("index_compatibility_id", "query_compatibility_id", "artifact_json_text"):
        assert after[field] == before[field]
    assert after["contract_snapshot_sha256"] == case.current.contract_snapshot_sha256
    assert case.evidence()[:-1] == evidence_before
    audit = json.loads(case.evidence()[-1]["diagnostics_json_text"])
    assert (
        audit["before_contract"]["contract_snapshot_json_text"]
        == before["contract_snapshot_json_text"]
    )
    with _live_guard(case) as token:
        assert token.resolved.profile_input_type_required is True
        assert token.resolved.profile_input_type_missing_role_behavior == "reject"


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("index_compatibility_id", "sha256:" + "a" * 64),
        ("query_compatibility_id", "sha256:" + "b" * 64),
        ("backend_id", 99),
        ("backend_url", "http://different.invalid:11434"),
        ("artifact_revision", "different-revision"),
        ("artifact_manifest_digest", "sha256:" + "c" * 64),
        ("profile_id", "another-profile"),
        ("profile_kind", "late_chunking"),
        ("canonical_model_id", "another-canonical-model"),
    ],
)
def test_transition_rejects_identity_changes_without_mutation(transition_case, field, changed):
    case = transition_case
    before, evidence_before = case.contract(), case.evidence()
    case.resolver.return_value = replace(case.current, **{field: changed})
    with pytest.raises(contracts.IndexCompatibilityError):
        _transition(case, apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before


@pytest.mark.parametrize(
    "changed_field", ["dimensions", "output_normalized", "role_sensitive", "formatting"]
)
def test_transition_rejects_profile_data_drift_with_unchanged_claimed_ids(
    transition_case, changed_field
):
    case = transition_case
    snapshot = copy.deepcopy(case.capture)
    profile = next(
        value for value in snapshot["profiles"] if value["profile_id"] == case.current.profile_id
    )
    if changed_field == "role_sensitive":
        profile["input_type"]["role_sensitive"] = False
    elif changed_field == "formatting":
        profile["pipeline"]["formatting"]["search_document"]["template"] = "changed: {text}"
    else:
        profile[changed_field] = 17 if changed_field == "dimensions" else False
    raw = json.dumps(snapshot)
    canonical = canonical_json(snapshot)
    case.resolver.return_value = replace(
        case.current,
        contract_snapshot_json_text=raw,
        contract_snapshot_canonical_text=canonical,
        contract_snapshot_sha256=_sha(canonical),
    )
    before, evidence_before = case.contract(), case.evidence()
    with pytest.raises(contracts.IndexCompatibilityError):
        _transition(case, apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before


@pytest.mark.parametrize("field, changed", [("size", 17), ("distance", Distance.DOT)])
def test_transition_rejects_actual_qdrant_geometry_drift(transition_case, field, changed):
    case = transition_case
    vectors = case.client.get_collection.return_value.config.params.vectors
    setattr(vectors, field, changed)
    before, evidence_before = case.contract(), case.evidence()
    with pytest.raises(contracts.IndexCompatibilityError):
        _transition(case, apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before


def test_apply_failure_rolls_back_evidence_and_binding_then_can_retry(transition_case, monkeypatch):
    case = transition_case
    before, evidence_before = case.contract(), case.evidence()
    insert = contracts.insert_contract_evidence

    def fail_after_insert(*args, **kwargs):
        insert(*args, **kwargs)
        raise RuntimeError("injected failure after evidence insert")

    monkeypatch.setattr(contracts, "insert_contract_evidence", fail_after_insert)
    with pytest.raises(RuntimeError, match="injected failure"):
        _transition(case, apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before
    monkeypatch.setattr(contracts, "insert_contract_evidence", insert)
    assert _transition(case, apply=True)["status"] == "applied"


def test_apply_idempotency_and_audited_rollback_never_reuses_fence(transition_case):
    case = transition_case
    original = case.contract()
    original_evidence = case.evidence()
    applied = _transition(case, apply=True)
    applied_row, applied_evidence = case.contract(), case.evidence()
    again = _transition(case, apply=True)
    assert again["status"] == "already_applied"
    assert again["evidence_id"] == applied["evidence_id"]
    assert case.contract() == applied_row
    assert case.evidence() == applied_evidence
    assert _rollback(case, applied["evidence_id"])["status"] == "planned"
    assert case.contract() == applied_row
    reverted = _rollback(case, applied["evidence_id"], apply=True)
    assert reverted["status"] == "rolled_back"
    restored = case.contract()
    for field in (
        "contract_schema_sha256",
        "contract_snapshot_json_text",
        "contract_snapshot_canonical_text",
        "contract_snapshot_sha256",
        "index_compatibility_id",
        "query_compatibility_id",
    ):
        assert restored[field] == original[field]
    assert restored["fence_version"] == applied_row["fence_version"] + 1
    assert restored["evidence_id"] not in {original["evidence_id"], applied["evidence_id"]}
    assert restored["evidence_id"] == reverted["evidence_id"]
    assert case.evidence()[:-1] == applied_evidence
    assert case.evidence()[:1] == original_evidence
    audit = case.evidence()[-1]
    assert json.loads(audit["diagnostics_json_text"])["direction"] == "rollback"
    assert _sha(audit["diagnostics_json_text"]) == audit["diagnostics_sha256"]
    with pytest.raises(contracts.IndexCompatibilityError):
        with _live_guard(case):
            pytest.fail("rollback must not bypass the new consumer schema guard")
    assert _rollback(case, applied["evidence_id"], apply=True)["status"] == "already_rolled_back"
    assert case.contract() == restored
    assert len(case.evidence()) == 3


@pytest.mark.parametrize(
    "field, changed",
    [("index_compatibility_id", "sha256:" + "d" * 64), ("backend_id", 99)],
)
def test_rollback_rejects_changed_live_binding(transition_case, field, changed):
    case = transition_case
    applied = _transition(case, apply=True)
    case.resolver.return_value = replace(case.current, **{field: changed})
    before, evidence_before = case.contract(), case.evidence()
    with pytest.raises(contracts.IndexCompatibilityError):
        _rollback(case, applied["evidence_id"], apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before


@pytest.mark.parametrize("drift", ["qdrant_geometry", "strict_policy"])
def test_already_applied_rechecks_live_geometry_and_strict_policy(transition_case, drift):
    case = transition_case
    _transition(case, apply=True)
    if drift == "qdrant_geometry":
        case.client.get_collection.return_value.config.params.vectors.size = 17
    else:
        snapshot = copy.deepcopy(case.capture)
        profile = next(
            value
            for value in snapshot["profiles"]
            if value["profile_id"] == case.current.profile_id
        )
        profile["input_type"].update(
            required=False, default="search_document", missing_role_behavior="use_default"
        )
        parsed = parse_capabilities(snapshot)
        assert (
            parsed.profile_by_id(case.current.profile_id).index_compatibility_id
            == case.current.index_compatibility_id
        )
        case.resolver.return_value = replace(
            case.current,
            contract_snapshot_json_text=parsed.snapshot.json_text,
            contract_snapshot_canonical_text=parsed.snapshot.canonical_json,
            contract_snapshot_sha256=parsed.snapshot.canonical_sha256,
            profile_input_type_required=False,
            profile_input_type_default="search_document",
            profile_input_type_missing_role_behavior="use_default",
        )
    before, evidence_before = case.contract(), case.evidence()
    with pytest.raises(contracts.IndexCompatibilityError):
        _transition(case, apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before


def test_reapply_checks_original_evidence_through_rollback_ancestry(transition_case):
    case = transition_case
    original_evidence_id = case.contract()["evidence_id"]
    applied = _transition(case, apply=True)
    _rollback(case, applied["evidence_id"], apply=True)
    # The original row remains locally self-consistent. Its immutable reference
    # in the first admission must still detect this change two steps later.
    changed = '{"test":true,"tampered_after_rollback":true}'
    case.db.execute(
        "UPDATE index_contract_evidence SET diagnostics_json_text = %s, diagnostics_sha256 = %s "
        "WHERE id = %s",
        (changed, _sha(changed), original_evidence_id),
    )
    case.db.commit()
    before, evidence_before = case.contract(), case.evidence()
    with pytest.raises(contracts.IndexCompatibilityError):
        _transition(case, apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before


@pytest.mark.parametrize("status", ["pending", "running", "paused"])
@pytest.mark.parametrize("direction", ["apply", "rollback"])
def test_active_index_job_blocks_schema_mutation(transition_case, status, direction):
    case = transition_case
    applied = _transition(case, apply=True) if direction == "rollback" else None
    case.db.execute("INSERT INTO datasources(id,name,type) VALUES (1,'P64 schema fence','local')")
    case.db.execute(
        "INSERT INTO ingestion_jobs(id,datasource_id,status,embedding_model,contract_logical_index_key) "
        "VALUES (1,1,%s,%s,%s)",
        (status, case.target.requested_model, case.target.logical_index_key),
    )
    case.db.commit()
    before, evidence_before = case.contract(), case.evidence()
    with pytest.raises(contracts.IndexCompatibilityError):
        if applied:
            _rollback(case, applied["evidence_id"], apply=True)
        else:
            _transition(case, apply=True)
    assert case.contract() == before
    assert case.evidence() == evidence_before
