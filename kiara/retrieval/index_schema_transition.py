"""Explicit, audited P64 admission between two approved KIron schema versions.

The normal guard remains strict. This operation changes no vector, payload,
model identity or historical evidence. PostgreSQL commits the new admission
and its complete predecessor record together under the existing index fence.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import closing, nullcontext
from dataclasses import replace
from functools import lru_cache
from importlib.resources import files

from jsonschema import Draft202012Validator

from kiara.retrieval import embedding_contract
from kiara.retrieval import index_compatibility as contracts

OLD_SCHEMA_SHA256 = "2dafc62888439ef3be15d7ae4fac29d2718ee16ddcfb6f82f61bf87149b14264"
NEW_SCHEMA_SHA256 = "d4d2698831e16bafe54dc03ffcb5b927933b9439111242e0d5997f0815dea477"
_PROCEDURE = "p64-schema-admission-v1"
_MODELS = frozenset({"nomic-embed-text", "hellord/e5-mistral-7b-instruct:q4_0"})
_SNAPSHOT_FIELDS = (
    "contract_schema_version",
    "contract_schema_sha256",
    "contract_snapshot_json_text",
    "contract_snapshot_canonical_text",
    "contract_snapshot_sha256",
)
_AUDIT_FIELDS = frozenset(
    {
        "evidence_id",
        "evidence_at",
        "created_at",
        "updated_at",
        "evidence_contract_snapshot_sha256",
    }
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value) -> str:
    # Audit rows contain PostgreSQL BIGINT identities (notably advisory-lock
    # keys). JCS uses IEEE-754 numbers and would round these on persistence.
    # Contract snapshots retain their own JCS text and hash unchanged.
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _serializable(row) -> dict:
    return json.loads(json.dumps(dict(row), default=lambda value: value.isoformat()))


def _state(row) -> dict:
    return {key: value for key, value in _serializable(row).items() if key not in _AUDIT_FIELDS}


def _reject(target, reason, detail):
    raise contracts.IndexCompatibilityError(reason, "schema_transition", target, detail)


def _check_scope(target):
    if target.requested_model not in _MODELS or target.vector_kind not in (
        contracts.IndexVectorKind.DENSE,
        contracts.IndexVectorKind.LATE_CHUNKING,
    ):
        _reject(
            target, "schema_transition_scope_invalid", "Modell oder Vektorart nicht freigegeben"
        )
    if embedding_contract.SCHEMA_SHA256 != NEW_SCHEMA_SHA256:
        _reject(
            target, "schema_transition_consumer_invalid", "Nicht der freigegebene Consumerstand"
        )


@lru_cache(maxsize=1)
def _historical_schema():
    raw = (
        files("kiara.retrieval")
        .joinpath("schemas", "kiron-embedding-capabilities-v1.pre-reject.schema.json")
        .read_bytes()
    )
    if hashlib.sha256(raw).hexdigest() != OLD_SCHEMA_SHA256:
        raise RuntimeError("historical KIron schema hash mismatch")
    schema = embedding_contract.decode_json_document(raw)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _profile(target, row, *, historical=False):
    try:
        raw = embedding_contract.decode_json_document(row["contract_snapshot_json_text"].encode())
        if historical:
            _historical_schema().validate(raw)
        parsed = embedding_contract.parse_capabilities(raw)
        if parsed.snapshot.canonical_sha256 != row["contract_snapshot_sha256"]:
            raise ValueError("Snapshot-Hash passt nicht zum geprueften Profil")
        profile = next(p for p in raw["profiles"] if p["profile_id"] == row["profile_id"])
        if (
            parsed.canonical_model_id != row["canonical_model_id"]
            or profile["verification"]["status"] != "verified"
            or profile["kind"] != target.vector_kind.value
            or profile["kind"] != row["profile_kind"]
            or profile["endpoint"] != row["profile_endpoint"]
            or profile["index_compatibility_id"] != row["index_compatibility_id"]
            or profile["query_compatibility_id"] != row["query_compatibility_id"]
        ):
            raise ValueError("Ausgewaehltes Profil widerspricht der Indexbindung")
        identity = {key: value for key, value in profile.items() if key != "verification"}
        policy = identity["input_type"]
        if (
            not historical
            and policy["role_sensitive"]
            and (
                policy["required"] is not True
                or policy["default"] is not None
                or policy["missing_role_behavior"] != "reject"
            )
        ):
            raise ValueError("Aktuelles rollenabhaengiges Profil erzwingt keine explizite Rolle")
        # S4 already separates request enforcement from vector identity.
        # KIara sends explicit roles in both phases; the accepted change in
        # the behavior for omitted roles changes neither formatter nor ID.
        for field in ("required", "default", "missing_role_behavior"):
            policy.pop(field)
        # The typed ArtifactDigest has always represented missing optional
        # sizes as None. Preserve and compare actual non-null sizes too.
        for field in ("weights", "auxiliary"):
            for digest in identity["artifact"][field]:
                digest.setdefault("size_bytes", None)
        return identity
    except Exception as exc:
        _reject(target, "schema_transition_profile_invalid", str(exc))


def _geometry(target, client, profile):
    try:
        vectors = client.get_collection(target.collection_name).config.params.vectors
        actual = {
            "size": int(vectors.size),
            "distance": str(vectors.distance.value).lower(),
            "multivector": vectors.multivector_config is not None,
        }
        expected = {
            "size": profile["dimensions"],
            "distance": {"euclidean": "euclid"}.get(profile["similarity"], profile["similarity"]),
            "multivector": False,
        }
        if actual != expected:
            raise ValueError(f"Vektorgeometrie geaendert: {actual!r} != {expected!r}")
        return actual
    except Exception as exc:
        _reject(target, "schema_transition_vector_identity_mismatch", str(exc))


def _no_active_jobs(conn, target):
    row = conn.execute(
        "SELECT id FROM ingestion_jobs WHERE contract_logical_index_key = %s "
        "AND status IN ('pending', 'running', 'paused') ORDER BY id LIMIT 1",
        (target.logical_index_key,),
    ).fetchone()
    if row:
        _reject(target, "schema_transition_active_job", f"Index hat aktiven Job #{row['id']}")


def _evidence(conn, target, evidence_id):
    row = conn.execute(
        "SELECT * FROM index_contract_evidence WHERE id = %s FOR SHARE", (evidence_id,)
    ).fetchone()
    if not row or row["logical_index_key"] != target.logical_index_key:
        _reject(
            target,
            "schema_transition_evidence_missing",
            "Evidence fehlt oder gehoert zu anderem Index",
        )
    value = _serializable(row)
    for field in ("references", "diagnostics"):
        text = value.get(f"{field}_json_text")
        if not isinstance(text, str) or _sha(text) != value.get(f"{field}_sha256"):
            _reject(target, "schema_transition_evidence_invalid", f"{field}-Hash ist ungueltig")
        try:
            embedding_contract.decode_json_document(text.encode())
        except ValueError as exc:
            _reject(target, "schema_transition_evidence_invalid", str(exc))
    return value


def _source_evidence(conn, target, stored):
    evidence = _evidence(conn, target, stored["evidence_id"])
    for field, expected in (
        ("result_state", "verified"),
        ("evidence_type", stored["evidence_type"]),
        ("sample_count", stored["evidence_sample_count"]),
        ("contract_snapshot_sha256", stored["contract_snapshot_sha256"]),
        ("candidate_index_compatibility_id", stored["index_compatibility_id"]),
        ("candidate_query_compatibility_id", stored["query_compatibility_id"]),
    ):
        if evidence.get(field) != expected:
            _reject(
                target, "schema_transition_evidence_invalid", f"Evidence-Feld {field} weicht ab"
            )
    return evidence


def _audit(conn, target, evidence, *, seen=None):
    seen = set() if seen is None else seen
    if evidence["id"] in seen:
        _reject(target, "schema_transition_audit_invalid", "Zyklische Evidence-Verweiskette")
    seen.add(evidence["id"])
    diagnostic = embedding_contract.decode_json_document(evidence["diagnostics_json_text"].encode())
    try:
        if diagnostic["procedure"] != _PROCEDURE or diagnostic["approval"] != "task:189":
            raise ValueError("Kein Nachweis des genehmigten Uebergangs")
        before = diagnostic["before_contract"]
        if _sha(_json(before)) != diagnostic["before_contract_sha256"]:
            raise ValueError("Historischer Vertragsstand ist veraendert")
        reference = diagnostic["before_evidence"]
        if before["evidence_id"] != reference["id"]:
            raise ValueError("Historischer Evidence-Anker widerspricht Vertrag")
        old_evidence = _source_evidence(conn, target, before)
        if _sha(_json(old_evidence)) != reference["sha256"]:
            raise ValueError("Historische Evidenz ist veraendert")
        if diagnostic["direction"] not in ("apply", "rollback"):
            raise ValueError("Unbekannte Uebergangsrichtung")
        pair = (
            before["contract_schema_sha256"],
            diagnostic["expected_after"]["contract_schema_sha256"],
        )
        expected_pair = (OLD_SCHEMA_SHA256, NEW_SCHEMA_SHA256)
        if diagnostic["direction"] == "rollback":
            expected_pair = tuple(reversed(expected_pair))
        if pair != expected_pair:
            raise ValueError("Nicht genehmigtes Schema-Paar")
        if old_evidence["reason_code"] in ("index_schema_apply", "index_schema_rollback"):
            predecessor = _audit(conn, target, old_evidence, seen=seen)
            _verify_after(target, before, predecessor)
    except (KeyError, TypeError, ValueError) as exc:
        _reject(target, "schema_transition_audit_invalid", str(exc))
    return diagnostic


def _live_identity(conn, config, target, stored, client):
    current = contracts.resolve_current_index_contract(conn, config, target, force_refresh=True)
    if not current.verified or current.contract_schema_sha256 != NEW_SCHEMA_SHA256:
        _reject(
            target,
            "schema_transition_current_invalid",
            "Aktuelles Profil ist nicht im freigegebenen Schema verifiziert",
        )
    if stored["contract_schema_sha256"] not in (OLD_SCHEMA_SHA256, NEW_SCHEMA_SHA256):
        _reject(target, "schema_transition_source_invalid", "Nicht genehmigter Schemastand")
    # The schema substitution is local to this explicitly authorized procedure;
    # all other shared identity checks remain identical to the runtime guard.
    contracts._validate_stored_contract(
        "schema_transition",
        target,
        stored,
        replace(current, contract_schema_sha256=stored["contract_schema_sha256"]),
    )
    profile = _profile(
        target, stored, historical=stored["contract_schema_sha256"] == OLD_SCHEMA_SHA256
    )
    snapshot = {field: getattr(current, field) for field in _SNAPSHOT_FIELDS}
    if profile != _profile(target, {**stored, **snapshot}):
        _reject(
            target,
            "schema_transition_profile_changed",
            "Geschuetzte Profilidentitaet wurde geaendert",
        )
    return snapshot, _geometry(target, client, profile)


def _verify_after(target, stored, diagnostic):
    if _state(stored) != diagnostic["expected_after"]:
        _reject(
            target,
            "schema_transition_admission_changed",
            "Aktive Bindung weicht vom Annahmestand ab",
        )


def _admit(conn, target, stored, source_evidence, snapshot, geometry, *, direction, reverts=None):
    before = _serializable(stored)
    before.pop("evidence_contract_snapshot_sha256", None)
    changes = {field: snapshot[field] for field in _SNAPSHOT_FIELDS}
    changes.update(
        fence_version=int(stored["fence_version"]) + 1,
        reason_code=f"index_schema_{direction}",
        evidence_type="provenance",
        evidence_sample_count=0,
        backfill_run_id=None,
    )
    expected = {**_state(stored), **changes}
    diagnostics = {
        "procedure": _PROCEDURE,
        "approval": "task:189",
        "direction": direction,
        "before_contract": before,
        "before_contract_sha256": _sha(_json(before)),
        "before_evidence": {"id": source_evidence["id"], "sha256": _sha(_json(source_evidence))},
        "expected_after": expected,
        "vector_geometry": geometry,
        "reverts_evidence_id": reverts,
    }
    references = _json(
        [{"evidence_id": source_evidence["id"], "purpose": "unchanged_vector_identity"}]
    )
    text = _json(diagnostics)
    evidence_id = contracts.insert_contract_evidence(
        conn,
        {
            "logical_index_key": target.logical_index_key,
            "evidence_type": "provenance",
            "result_state": "verified",
            "reason_code": changes["reason_code"],
            "sample_count": 0,
            "collection_points_count": stored["observed_points_count"],
            "references_json_text": references,
            "references_sha256": _sha(references),
            "diagnostics_json_text": text,
            "diagnostics_sha256": _sha(text),
            "candidate_index_compatibility_id": stored["index_compatibility_id"],
            "candidate_query_compatibility_id": stored["query_compatibility_id"],
            "contract_snapshot_sha256": snapshot["contract_snapshot_sha256"],
        },
    )
    changes["evidence_id"] = evidence_id
    assignments = ", ".join(f"{field} = %s" for field in changes)
    row = conn.execute(
        f"UPDATE index_contracts SET {assignments}, evidence_at = now(), updated_at = now() "
        "WHERE logical_index_key = %s AND fence_version = %s AND evidence_id = %s RETURNING *",
        (
            *changes.values(),
            target.logical_index_key,
            stored["fence_version"],
            stored["evidence_id"],
        ),
    ).fetchone()
    if not row:
        _reject(
            target, "schema_transition_fence_changed", "Vertragsversion wurde parallel geaendert"
        )
    _verify_after(target, row, diagnostics)
    return {
        "status": "applied" if direction == "apply" else "rolled_back",
        "evidence_id": evidence_id,
        "fence_version": changes["fence_version"],
        "logical_index_key": target.logical_index_key,
    }


def transition_schema(config, target, *, apply=False, qdrant_client=None):
    """Plan or admit precisely the approved old-to-new schema transition."""
    _check_scope(target)
    client_context = (
        nullcontext(qdrant_client)
        if qdrant_client is not None
        else closing(contracts.raw_qdrant_client(config))
    )
    with (
        client_context as client,
        contracts.guard_index(
            config,
            target,
            operation="schema_transition",
            write=True,
            validation_scope="stored_binding",
            qdrant_client=client,
        ) as token,
    ):
        stored, conn = token.stored, token.connection
        _no_active_jobs(conn, target)
        evidence = _source_evidence(conn, target, stored)
        snapshot, geometry = _live_identity(conn, config, target, stored, client)
        if stored["contract_schema_sha256"] == NEW_SCHEMA_SHA256:
            audit = _audit(conn, target, evidence)
            if audit["direction"] != "apply":
                _reject(target, "schema_transition_audit_invalid", "Kein angewandter Uebergang")
            _verify_after(target, stored, audit)
            return {
                "status": "already_applied",
                "evidence_id": evidence["id"],
                "fence_version": token.fence_version,
            }
        if (
            stored["contract_schema_sha256"] != OLD_SCHEMA_SHA256
            or stored["contract_schema_version"] != 1
        ):
            _reject(
                target, "schema_transition_source_invalid", "Nicht der genehmigte alte Schemastand"
            )
        if evidence["reason_code"] == "index_schema_rollback":
            audit = _audit(conn, target, evidence)
            _verify_after(target, stored, audit)
        if not apply:
            return {
                "status": "planned",
                "logical_index_key": target.logical_index_key,
                "from_schema": OLD_SCHEMA_SHA256,
                "to_schema": NEW_SCHEMA_SHA256,
                "fence_version": token.fence_version,
                "source_evidence_id": evidence["id"],
            }
        return _admit(conn, target, stored, evidence, snapshot, geometry, direction="apply")


def rollback_schema_transition(config, target, evidence_id, *, apply=False, qdrant_client=None):
    """Restore the recorded schema/snapshot with a new fence and rollback proof.

    This deliberately does not make the old schema usable by the new runtime.
    The matching saved consumer release is needed for semantic operations.
    """
    _check_scope(target)
    client_context = (
        nullcontext(qdrant_client)
        if qdrant_client is not None
        else closing(contracts.raw_qdrant_client(config))
    )
    with (
        client_context as client,
        contracts.guard_index(
            config,
            target,
            operation="schema_transition_rollback",
            write=True,
            validation_scope="stored_binding",
            qdrant_client=client,
        ) as token,
    ):
        stored, conn = token.stored, token.connection
        _no_active_jobs(conn, target)
        current_evidence = _source_evidence(conn, target, stored)
        current_audit = _audit(conn, target, current_evidence)
        _verify_after(target, stored, current_audit)
        _live_identity(conn, config, target, stored, client)
        if (
            current_audit["direction"] == "rollback"
            and current_audit["reverts_evidence_id"] == evidence_id
        ):
            return {
                "status": "already_rolled_back",
                "evidence_id": current_evidence["id"],
                "fence_version": token.fence_version,
            }
        if current_evidence["id"] != evidence_id or current_audit["direction"] != "apply":
            _reject(
                target,
                "schema_transition_rollback_not_current",
                "Nur der aktuelle Uebergang kann zurueckgenommen werden",
            )
        before = current_audit["before_contract"]
        old_profile = _profile(target, before, historical=True)
        if _profile(target, stored) != old_profile:
            _reject(
                target,
                "schema_transition_profile_changed",
                "Aktives Profil weicht vom gesicherten Stand ab",
            )
        geometry = _geometry(target, client, old_profile)
        if not apply:
            return {
                "status": "planned",
                "reverts_evidence_id": evidence_id,
                "fence_version": token.fence_version,
            }
        return _admit(
            conn,
            target,
            stored,
            current_evidence,
            before,
            geometry,
            direction="rollback",
            reverts=evidence_id,
        )
