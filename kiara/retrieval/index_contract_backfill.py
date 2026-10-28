"""Expliziter read-only Bestandsaudit und additiver S4-Metadaten-Backfill.

Dry-run und Apply verwenden ausschliesslich lesende Qdrant-Operationen.  Apply
persistiert alle Evidence-/Contractzeilen in genau einer PostgreSQL-
Transaktion.  Die Vektoren, Payloads, Collections und KIaras vier bestehende
Index-Datenschichten werden nie veraendert.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from kiara.retrieval.index_compatibility import (
    IndexCompatibilityError,
    IndexTarget,
    ResolvedIndexContract,
    _collection_state,
    _sha256_text,
    _validate_stored_contract,
    colbert_index_target,
    insert_contract_evidence,
    insert_index_contract,
    raw_qdrant_client,
    resolve_current_index_contract,
    stored_primary_index_target,
)
from kiara.web.index_contract_migration import INDEX_CONTRACT_CONFIG_LOCK_KEY

WITNESS_MIN_REFERENCES = 16
WITNESS_REFERENCE_COUNT = 32
WITNESS_CANDIDATE_LIMIT = 8192
WITNESS_ABS_TOLERANCE = 1e-3
WITNESS_COSINE_TOLERANCE = 1e-5

_READONLY_QDRANT_METHODS = frozenset(
    {"get_collections", "get_collection", "count", "scroll", "retrieve"}
)


class ReadOnlyQdrantClient:
    """Narrow client facade that makes accidental Qdrant writes impossible."""

    def __init__(self, raw_client) -> None:
        self._raw_client = raw_client

    @property
    def raw_client(self):
        return self._raw_client

    def __getattr__(self, name: str):
        if name not in _READONLY_QDRANT_METHODS:
            raise RuntimeError(f"index_backfill_qdrant_write_forbidden:{name}")
        return getattr(self._raw_client, name)


@dataclass(frozen=True)
class WitnessResult:
    state: str
    reason_code: str
    sample_count: int
    max_abs_error: float | None
    max_cosine_distance: float | None
    references_json_text: str
    references_sha256: str
    diagnostics_json_text: str
    diagnostics_sha256: str
    sample_set_sha256: str | None

    @property
    def passed(self) -> bool:
        return self.state == "verified"


@dataclass(frozen=True)
class BackfillAssessment:
    target: IndexTarget
    points_count: int
    collection_config_sha256: str
    resolved: ResolvedIndexContract
    verification_state: str
    evidence_type: str
    reason_code: str
    witness: WitnessResult | None
    action: str = "insert"

    def to_dict(self) -> dict[str, object]:
        witness = self.witness
        return {
            **self.target.to_dict(),
            "points_count": self.points_count,
            "collection_config_sha256": self.collection_config_sha256,
            "current_contract": self.resolved.diagnostic_dict(),
            "verification_state": self.verification_state,
            "compatibility_state": (
                "compatible" if self.verification_state == "verified" else self.verification_state
            ),
            "evidence_type": self.evidence_type,
            "reason_code": self.reason_code,
            "sample_count": witness.sample_count if witness else 0,
            "max_abs_error": witness.max_abs_error if witness else None,
            "max_cosine_distance": witness.max_cosine_distance if witness else None,
            "abs_tolerance": WITNESS_ABS_TOLERANCE if witness else None,
            "cosine_tolerance": WITNESS_COSINE_TOLERANCE if witness else None,
            "index_compatibility_id": (
                self.resolved.index_compatibility_id
                if self.verification_state == "verified"
                else None
            ),
            "query_compatibility_id": (
                self.resolved.query_compatibility_id
                if self.verification_state == "verified"
                else None
            ),
            "action": self.action,
        }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def collection_config_sha256(client: ReadOnlyQdrantClient, collection_name: str) -> str:
    info = client.get_collection(collection_name)
    snapshot = {
        "config": _jsonable(info.config),
        "payload_schema": _jsonable(info.payload_schema or {}),
    }
    return _sha256_text(_canonical_json(snapshot))


def _normalized_point_id(value: object) -> str:
    return str(value).replace("-", "")


def _text_length_bucket(length: int) -> str:
    if length < 256:
        return "lt256"
    if length < 1024:
        return "256_1023"
    if length < 4096:
        return "1024_4095"
    return "ge4096"


def _candidate_coverage(point) -> tuple[set[str], dict[str, object]] | None:
    payload = dict(point.payload or {})
    text = payload.get("_text")
    if text is None:
        text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    ds_raw = payload.get("ds_ids")
    if isinstance(ds_raw, list):
        ds_ids = sorted({int(value) for value in ds_raw if isinstance(value, (int, float))})
    elif isinstance(ds_raw, (int, float)):
        ds_ids = [int(ds_raw)]
    else:
        ds_ids = []
    file_type = str(payload.get("file_type") or "<missing>")
    directory = str(payload.get("directory") or "<missing>")
    bucket = _text_length_bucket(len(text))
    coverage = {
        *(f"ds:{value}" for value in ds_ids),
        f"file_type:{file_type}",
        f"text_length:{bucket}",
        f"directory:{directory}",
    }
    record = {
        "point_id": _normalized_point_id(point.id),
        "text": text,
        "source": str(payload.get("source") or ""),
        "ds_ids": ds_ids,
        "file_type": file_type,
        "directory": directory,
        "text_length": len(text),
        "text_length_bucket": bucket,
    }
    return coverage, record


def select_witness_references(
    client: ReadOnlyQdrantClient,
    target: IndexTarget,
    *,
    reference_count: int = WITNESS_REFERENCE_COUNT,
    candidate_limit: int = WITNESS_CANDIDATE_LIMIT,
) -> list[dict[str, object]]:
    """Deterministic greedy coverage over DS, type, directory and text length."""

    candidates: list[tuple[set[str], dict[str, object], str]] = []
    offset = None
    first = True
    while first or offset is not None:
        first = False
        remaining = candidate_limit - len(candidates)
        if remaining <= 0:
            break
        points, offset = client.scroll(
            collection_name=target.collection_name,
            limit=min(256, remaining),
            offset=offset,
            with_payload=["_text", "text", "source", "ds_ids", "file_type", "directory"],
            with_vectors=False,
        )
        for point in points:
            item = _candidate_coverage(point)
            if item is None:
                continue
            coverage, record = item
            tie = hashlib.sha256(
                f"{target.logical_index_key}:{record['point_id']}".encode()
            ).hexdigest()
            candidates.append((coverage, record, tie))
            if len(candidates) >= candidate_limit:
                break

    selected: list[dict[str, object]] = []
    covered: set[str] = set()
    remaining_candidates = list(candidates)
    while remaining_candidates and len(selected) < reference_count:
        remaining_candidates.sort(
            key=lambda item: (-len(item[0] - covered), item[2], str(item[1]["point_id"]))
        )
        coverage, record, _ = remaining_candidates.pop(0)
        selected.append(record)
        covered.update(coverage)
    return selected


def _vector_rows(value: object) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("witness_vector_shape_invalid")
    if all(isinstance(item, (int, float)) for item in value):
        return [[float(item) for item in value]]
    if all(isinstance(row, list) for row in value):
        result: list[list[float]] = []
        for row in value:
            if not row or not all(isinstance(item, (int, float)) for item in row):
                raise ValueError("witness_vector_shape_invalid")
            result.append([float(item) for item in row])
        return result
    raise ValueError("witness_vector_shape_invalid")


def compare_witness_vectors(
    stored: list[object],
    generated: list[object],
    *,
    abs_tolerance: float = WITNESS_ABS_TOLERANCE,
    cosine_tolerance: float = WITNESS_COSINE_TOLERANCE,
) -> tuple[bool, str, float | None, float | None, list[dict[str, object]]]:
    """Apply the normative S1 row-order, finite, abs and cosine criteria."""

    if len(stored) != len(generated):
        return False, "witness_cardinality_mismatch", None, None, []
    maximum_abs = 0.0
    maximum_cosine = 0.0
    measurements: list[dict[str, object]] = []
    try:
        for reference_index, (stored_value, generated_value) in enumerate(
            zip(stored, generated, strict=True)
        ):
            stored_rows = _vector_rows(stored_value)
            generated_rows = _vector_rows(generated_value)
            if len(stored_rows) != len(generated_rows):
                return False, "witness_shape_mismatch", None, None, measurements
            for row_index, (left, right) in enumerate(
                zip(stored_rows, generated_rows, strict=True)
            ):
                if len(left) != len(right):
                    return False, "witness_shape_mismatch", None, None, measurements
                if not all(math.isfinite(item) for item in (*left, *right)):
                    return False, "witness_non_finite", None, None, measurements
                row_abs = max(abs(a - b) for a, b in zip(left, right, strict=True))
                dot = sum(a * b for a, b in zip(left, right, strict=True))
                left_norm = math.sqrt(sum(a * a for a in left))
                right_norm = math.sqrt(sum(b * b for b in right))
                if left_norm == 0.0 or right_norm == 0.0:
                    return False, "witness_zero_norm", None, None, measurements
                cosine_distance = 1.0 - dot / (left_norm * right_norm)
                # Floating-point roundoff can produce tiny negative distances.
                cosine_distance = max(0.0, cosine_distance)
                maximum_abs = max(maximum_abs, row_abs)
                maximum_cosine = max(maximum_cosine, cosine_distance)
                measurements.append(
                    {
                        "reference_index": reference_index,
                        "row_index": row_index,
                        "dimension": len(left),
                        "max_abs_error": row_abs,
                        "cosine_distance": cosine_distance,
                    }
                )
    except (TypeError, ValueError):
        return False, "witness_vector_shape_invalid", None, None, measurements

    passed = maximum_abs <= abs_tolerance and maximum_cosine <= cosine_tolerance
    return (
        passed,
        "witness_verified" if passed else "witness_tolerance_exceeded",
        maximum_abs,
        maximum_cosine,
        measurements,
    )


def _stored_vectors_for_references(
    client: ReadOnlyQdrantClient,
    target: IndexTarget,
    references: list[dict[str, object]],
) -> list[object]:
    ids = [str(reference["point_id"]) for reference in references]
    points = client.retrieve(
        collection_name=target.collection_name,
        ids=ids,
        with_payload=["_text", "text"],
        with_vectors=True,
    )
    by_id = {_normalized_point_id(point.id): point for point in points}
    result: list[object] = []
    for reference in references:
        point_id = str(reference["point_id"])
        point = by_id.get(point_id)
        if point is None:
            raise ValueError("witness_reference_disappeared")
        payload = dict(point.payload or {})
        text = payload.get("_text")
        if text is None:
            text = payload.get("text")
        if text != reference["text"]:
            raise ValueError("witness_reference_text_changed")
        vector = point.vector
        if isinstance(vector, dict):
            raise ValueError("witness_named_vector_unsupported")
        result.append(vector)
    return result


def _request_witness_vectors(
    resolved: ResolvedIndexContract,
    references: list[dict[str, object]],
    *,
    timeout: float = 120.0,
) -> list[object]:
    if not resolved.verified or not resolved.backend_url or not resolved.profile_endpoint:
        raise ValueError("witness_profile_not_verified")
    payload: dict[str, object] = {
        "model": resolved.canonical_model_id,
        "input": [reference["text"] for reference in references],
        # S4-Ausnahme nur fuer den Witness. Produktionspayloads bleiben bis S5
        # unveraendert.
        "input_type": "search_document",
    }
    response = httpx.post(
        f"{resolved.backend_url.rstrip('/')}{resolved.profile_endpoint}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    value = response.json()
    embeddings = value.get("embeddings") if isinstance(value, dict) else None
    if not isinstance(embeddings, list):
        raise ValueError("witness_backend_payload_invalid")
    return embeddings


def run_vector_witness(
    client: ReadOnlyQdrantClient,
    target: IndexTarget,
    resolved: ResolvedIndexContract,
    *,
    reference_count: int = WITNESS_REFERENCE_COUNT,
) -> WitnessResult:
    references: list[dict[str, object]] = []
    measurements: list[dict[str, object]] = []
    max_abs: float | None = None
    max_cosine: float | None = None
    reason_code = "witness_internal_error"
    state = "evidence_failed"
    try:
        references = select_witness_references(
            client,
            target,
            reference_count=reference_count,
        )
        if len(references) < WITNESS_MIN_REFERENCES:
            reason_code = "witness_insufficient_references"
        else:
            stored = _stored_vectors_for_references(client, target, references)
            generated = _request_witness_vectors(resolved, references)
            passed, reason_code, max_abs, max_cosine, measurements = compare_witness_vectors(
                stored,
                generated,
            )
            state = "verified" if passed else "evidence_failed"
    except httpx.HTTPError as exc:
        reason_code = "witness_backend_error"
        measurements.append({"error_class": exc.__class__.__name__})
    except (TypeError, ValueError, RuntimeError) as exc:
        reason_code = str(exc) if str(exc).startswith("witness_") else "witness_internal_error"
        measurements.append({"error_class": exc.__class__.__name__})

    public_references: list[dict[str, object]] = []
    for index, reference in enumerate(references):
        public = {key: value for key, value in reference.items() if key != "text"}
        public["text_sha256"] = _sha256_text(str(reference["text"]))
        public["selection_order"] = index
        public_references.append(public)
    references_text = _canonical_json(public_references)
    sample_set_sha256 = _sha256_text(references_text) if public_references else None
    diagnostics_text = _canonical_json(
        {
            "state": state,
            "reason_code": reason_code,
            "selection": {
                "algorithm": "deterministic_greedy_coverage_v1",
                "candidate_limit": WITNESS_CANDIDATE_LIMIT,
                "requested_references": reference_count,
                "minimum_references": WITNESS_MIN_REFERENCES,
                "coverage_dimensions": [
                    "datasource",
                    "file_type",
                    "directory",
                    "text_length_bucket",
                ],
            },
            "criteria": {
                "shape": "exact",
                "row_order": "exact",
                "finite": True,
                "max_abs_error_lte": WITNESS_ABS_TOLERANCE,
                "cosine_distance_lte": WITNESS_COSINE_TOLERANCE,
                "both_required": True,
            },
            "measurements": measurements,
        }
    )
    return WitnessResult(
        state=state,
        reason_code=reason_code,
        sample_count=len(references),
        max_abs_error=max_abs,
        max_cosine_distance=max_cosine,
        references_json_text=references_text,
        references_sha256=_sha256_text(references_text),
        diagnostics_json_text=diagnostics_text,
        diagnostics_sha256=_sha256_text(diagnostics_text),
        sample_set_sha256=sample_set_sha256,
    )


def inventory_targets(config: dict, client: ReadOnlyQdrantClient) -> list[IndexTarget]:
    from kiara.retrieval.colbert import model_name_from_colbert_collection
    from kiara.retrieval.embedding import model_name_from_collection

    targets: list[IndexTarget] = []
    for collection in client.get_collections().collections:
        name = str(collection.name)
        if name.startswith("kb_"):
            targets.append(stored_primary_index_target(config, model_name_from_collection(name)))
        elif name.startswith("colbert_"):
            targets.append(colbert_index_target(config, model_name_from_colbert_collection(name)))
    return sorted(targets, key=lambda target: target.logical_index_key)


def _assess_target(
    conn,
    config: dict,
    client: ReadOnlyQdrantClient,
    target: IndexTarget,
    *,
    force_refresh: bool,
) -> BackfillAssessment:
    exists, points_count = _collection_state(client.raw_client, target.collection_name)
    if not exists:
        raise RuntimeError(f"index_collection_disappeared:{target.collection_name}")
    config_hash = collection_config_sha256(client, target.collection_name)
    resolved = resolve_current_index_contract(
        conn,
        config,
        target,
        force_refresh=force_refresh,
    )
    existing = conn.execute(
        "SELECT * FROM index_contracts WHERE logical_index_key = %s FOR SHARE",
        (target.logical_index_key,),
    ).fetchone()
    if existing is not None:
        existing_row = dict(existing)
        existing_state = str(existing_row["verification_state"])
        action = "skip_existing"
        existing_reason = "index_contract_already_persisted"
        if existing_state == "verified":
            try:
                _validate_stored_contract("backfill_audit", target, existing_row, resolved)
            except IndexCompatibilityError as exc:
                existing_state = "mismatch"
                existing_reason = exc.reason_code
                action = "conflict"
        return BackfillAssessment(
            target=target,
            points_count=points_count,
            collection_config_sha256=config_hash,
            resolved=resolved,
            verification_state=existing_state,
            evidence_type=str(existing["evidence_type"]),
            reason_code=existing_reason,
            witness=None,
            action=action,
        )

    if not resolved.verified:
        return BackfillAssessment(
            target=target,
            points_count=points_count,
            collection_config_sha256=config_hash,
            resolved=resolved,
            verification_state="unverified",
            evidence_type="unverified",
            reason_code=resolved.reason_code,
            witness=None,
        )
    if points_count == 0:
        return BackfillAssessment(
            target=target,
            points_count=0,
            collection_config_sha256=config_hash,
            resolved=resolved,
            verification_state="verified",
            evidence_type="empty_first_write",
            reason_code="index_empty_verified_backfill",
            witness=None,
        )

    witness = run_vector_witness(client, target, resolved)
    return BackfillAssessment(
        target=target,
        points_count=points_count,
        collection_config_sha256=config_hash,
        resolved=resolved,
        verification_state=witness.state,
        evidence_type="witness" if witness.passed else "evidence_failed",
        reason_code=witness.reason_code,
        witness=witness,
    )


def _evidence_values(assessment: BackfillAssessment, run_id: str) -> dict[str, object]:
    witness = assessment.witness
    if witness:
        references_text = witness.references_json_text
        diagnostics_text = witness.diagnostics_json_text
    else:
        references_text = "[]"
        diagnostics_text = _canonical_json(
            {
                "reason_code": assessment.reason_code,
                "target": assessment.target.to_dict(),
                "current_contract": assessment.resolved.diagnostic_dict(),
                "points_count": assessment.points_count,
                "qdrant_write_operations": 0,
            }
        )
    verified = assessment.verification_state == "verified"
    return {
        "logical_index_key": assessment.target.logical_index_key,
        "evidence_type": assessment.evidence_type,
        "result_state": assessment.verification_state,
        "reason_code": assessment.reason_code,
        "sample_count": witness.sample_count if witness else 0,
        "max_abs_error": witness.max_abs_error if witness else None,
        "max_cosine_distance": witness.max_cosine_distance if witness else None,
        "abs_tolerance": WITNESS_ABS_TOLERANCE if witness else None,
        "cosine_tolerance": WITNESS_COSINE_TOLERANCE if witness else None,
        "collection_points_count": assessment.points_count,
        "collection_config_sha256": assessment.collection_config_sha256,
        "sample_set_sha256": witness.sample_set_sha256 if witness else None,
        "references_json_text": references_text,
        "references_sha256": _sha256_text(references_text),
        "diagnostics_json_text": diagnostics_text,
        "diagnostics_sha256": _sha256_text(diagnostics_text),
        "candidate_index_compatibility_id": (
            assessment.resolved.index_compatibility_id if verified else None
        ),
        "candidate_query_compatibility_id": (
            assessment.resolved.query_compatibility_id if verified else None
        ),
        "contract_snapshot_sha256": assessment.resolved.contract_snapshot_sha256,
        "backfill_run_id": run_id,
    }


def run_index_contract_backfill(
    config: dict,
    *,
    apply: bool = False,
    force_refresh: bool = True,
) -> dict[str, object]:
    """Run deterministic audit; optionally persist only separate PG metadata."""

    from kiara.shared.db import db_transaction

    started = datetime.now(UTC)
    run_id = str(uuid.uuid4())
    readonly = ReadOnlyQdrantClient(raw_qdrant_client(config))
    with db_transaction() as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock_shared(%s)",
            (INDEX_CONTRACT_CONFIG_LOCK_KEY,),
        )
        targets = inventory_targets(config, readonly)
        # Deterministic order prevents lock inversion across parallel audits.
        for target in targets:
            lock_fn = "pg_advisory_xact_lock" if apply else "pg_advisory_xact_lock_shared"
            conn.execute(f"SELECT {lock_fn}(%s)", (target.lock_key,))

        assessments = [
            _assess_target(
                conn,
                config,
                readonly,
                target,
                force_refresh=force_refresh,
            )
            for target in targets
        ]
        inserted = 0
        skipped = 0
        if apply:
            conflicts = [item for item in assessments if item.action == "conflict"]
            if conflicts:
                names = ", ".join(item.target.collection_name for item in conflicts)
                raise RuntimeError(f"index_contract_backfill_conflict:{names}")
            for assessment in assessments:
                if assessment.action == "skip_existing":
                    skipped += 1
                    continue
                evidence_id = insert_contract_evidence(
                    conn,
                    _evidence_values(assessment, run_id),
                )
                insert_index_contract(
                    conn,
                    assessment.target,
                    assessment.resolved,
                    evidence_id=evidence_id,
                    evidence_type=assessment.evidence_type,
                    evidence_sample_count=(
                        assessment.witness.sample_count if assessment.witness else 0
                    ),
                    verification_state=assessment.verification_state,
                    reason_code=assessment.reason_code,
                    observed_points_count=assessment.points_count,
                    backfill_run_id=run_id,
                )
                inserted += 1

    finished = datetime.now(UTC)
    return {
        "schema": "kiara.index_contract_backfill.v1",
        "mode": "apply" if apply else "dry_run",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "transaction": "committed" if apply else "read_only_no_metadata_writes",
        "qdrant_write_operations": 0,
        "targets_total": len(assessments),
        "contracts_inserted": inserted,
        "contracts_skipped": skipped,
        "verified": sum(item.verification_state == "verified" for item in assessments),
        "unverified": sum(item.verification_state == "unverified" for item in assessments),
        "evidence_failed": sum(
            item.verification_state == "evidence_failed" for item in assessments
        ),
        "indexes": [item.to_dict() for item in assessments],
    }


def rollback_backfill_metadata(config: dict, run_id: str) -> dict[str, object]:
    """Remove one Apply run atomically; missing metadata remains fail-closed."""

    from kiara.shared.db import db_transaction

    if not run_id:
        raise ValueError("backfill_run_id_required")
    readonly = ReadOnlyQdrantClient(raw_qdrant_client(config))
    with db_transaction() as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock_shared(%s)",
            (INDEX_CONTRACT_CONFIG_LOCK_KEY,),
        )
        rows = conn.execute(
            "SELECT logical_index_key, lock_key, collection_name FROM index_contracts "
            "WHERE backfill_run_id = %s ORDER BY lock_key FOR UPDATE",
            (run_id,),
        ).fetchall()
        for row in rows:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (row["lock_key"],))
            # A read establishes that rollback itself did not replace/delete a
            # collection.  Missing collections do not make metadata deletion
            # unsafe: the post-state is still fail-closed.
            try:
                readonly.get_collection(str(row["collection_name"]))
            except Exception:
                pass
        evidence_ids = [
            int(row["evidence_id"])
            for row in conn.execute(
                "SELECT evidence_id FROM index_contracts WHERE backfill_run_id = %s",
                (run_id,),
            ).fetchall()
        ]
        deleted_contracts = conn.execute(
            "DELETE FROM index_contracts WHERE backfill_run_id = %s RETURNING logical_index_key",
            (run_id,),
        ).fetchall()
        deleted_evidence = 0
        if evidence_ids:
            deleted_evidence = conn.execute(
                "DELETE FROM index_contract_evidence WHERE id = ANY(%s) "
                "AND backfill_run_id = %s RETURNING id",
                (evidence_ids, run_id),
            ).fetchall()
            deleted_evidence = len(deleted_evidence)
    return {
        "schema": "kiara.index_contract_backfill_rollback.v1",
        "run_id": run_id,
        "contracts_removed": len(deleted_contracts),
        "evidence_removed": deleted_evidence,
        "qdrant_write_operations": 0,
        "post_state": "index_contract_missing_fail_closed",
    }
