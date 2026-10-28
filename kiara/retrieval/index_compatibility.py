"""Persistenter S4-Indexvertrag und zentraler fail-closed Laufzeit-Guard.

Der Guard bindet eine physische Qdrant-Collection an genau ein validiertes
KIron-Profil.  Modellname, Dimension und Collection-Schema sind dabei
ausdruecklich keine Kompatibilitaetsevidenz.  Eine PostgreSQL-Transaktion
haelt den gemeinsamen Config-Lock und den indexspezifischen Lock ueber das
gesamte kritische Qdrant-Fenster; Qdrant und PostgreSQL werden nicht als
gemeinsame Transaktion behandelt.

Der vollstaendige persistierte Discovery-Snapshot bleibt ein unveraenderter
Provenienzbeleg. Seine interne Raw-/Canonical-/SHA-Integritaet wird bei jedem
Guard geprueft. Laufzeitkompatibilitaet ist davon getrennt: Sie bindet das
frisch validierte Profil ueber Compatibility-IDs und die explizit
persistierten Modell-, Backend-, Profil-, Schema- und Artefaktfelder. Der
vollstaendige aktuelle Snapshot ist Diagnose, aber keine Vektorraumidentitaet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from qdrant_client import QdrantClient

from kiara.backend_selector import normalize_model_name
from kiara.retrieval.embedding_contract import (
    SCHEMA_SHA256,
    SCHEMA_VERSION,
    ContractState,
    ProfileKind,
    canonical_json,
    decode_json_document,
)
from kiara.retrieval.embedding_discovery import (
    discover_server_contracts,
    resolve_embedding_profile,
    select_configured_embedding_profile_id,
)
from kiara.shared.constants import (
    DEFAULT_QDRANT_URL,
    INDEX_CONTRACT_DISCOVERY_ATTEMPTS_DEFAULT,
    INDEX_CONTRACT_DISCOVERY_BACKOFF_INITIAL_SECS_DEFAULT,
    INDEX_CONTRACT_DISCOVERY_BACKOFF_MAX_SECS_DEFAULT,
    INDEX_CONTRACT_DISCOVERY_TIMEOUT_SECS_DEFAULT,
)
from kiara.web.index_contract_migration import INDEX_CONTRACT_CONFIG_LOCK_KEY

logger = logging.getLogger(__name__)


class IndexVectorKind(str, Enum):
    DENSE = "dense"
    LATE_CHUNKING = "late_chunking"
    MULTI_VECTOR = "multi_vector"


class IndexOperation(str, Enum):
    READ = "read"
    QUERY_EMBEDDING = "query_embedding"
    DOCUMENT_EMBEDDING = "document_embedding"
    WRITE = "write"
    DELETE = "delete"
    COLLECTION_ADMIN = "collection_admin"
    JOB_ENQUEUE = "job_enqueue"
    JOB_CLAIM = "job_claim"
    JOB_START = "job_start"
    FULL_REINGEST_CLEANUP = "full_reingest_cleanup"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    FIRST_WRITE = "first_write"


class IndexValidationScope(str, Enum):
    """Tiefe der Vertragspruefung fuer eine Indexoperation.

    Semantische Vektoroperationen brauchen das aktuell von KIron bestaetigte
    Profil. Reine Payload-/Loeschoperationen veraendern keinen Vektorraum und
    duerfen sich deshalb auf die lokal verifizierte, atomar gefencete
    Indexbindung stuetzen.
    """

    LIVE_PROFILE = "live_profile"
    STORED_BINDING = "stored_binding"


@dataclass(frozen=True)
class IndexTarget:
    store_type: str
    store_locator: str
    collection_name: str
    vector_kind: IndexVectorKind
    requested_model: str
    logical_index_key: str
    lock_key: int

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_index_key": self.logical_index_key,
            "store_type": self.store_type,
            "store_locator": self.store_locator,
            "collection_name": self.collection_name,
            "vector_kind": self.vector_kind.value,
            "requested_model": self.requested_model,
        }


@dataclass(frozen=True)
class ResolvedIndexContract:
    state: ContractState
    reason_code: str
    target: IndexTarget
    backend: dict[str, Any] | None = None
    backend_id: int | None = None
    backend_provider: str | None = None
    backend_url: str | None = None
    canonical_model_id: str | None = None
    profile_id: str | None = None
    profile_kind: str | None = None
    profile_endpoint: str | None = None
    profile_supported_roles: tuple[str, ...] = ()
    profile_role_sensitive: bool | None = None
    profile_input_type_required: bool | None = None
    profile_input_type_default: str | None = None
    profile_input_type_missing_role_behavior: str | None = None
    profile_formatting_owner: str | None = None
    contract_schema_version: int | None = None
    contract_schema_sha256: str | None = None
    contract_snapshot_json_text: str | None = None
    contract_snapshot_canonical_text: str | None = None
    contract_snapshot_sha256: str | None = None
    index_compatibility_id: str | None = None
    query_compatibility_id: str | None = None
    backend_type: str | None = None
    backend_implementation: str | None = None
    backend_implementation_revision: str | None = None
    artifact_repository: str | None = None
    artifact_revision: str | None = None
    artifact_manifest_digest: str | None = None
    artifact_json_text: str | None = None
    issues: tuple[dict[str, Any], ...] = ()
    discovery_attempts: int = 0

    @property
    def verified(self) -> bool:
        return (
            self.state is ContractState.VALID
            and self.index_compatibility_id is not None
            and self.query_compatibility_id is not None
        )

    def diagnostic_dict(self) -> dict[str, object]:
        return {
            **self.target.to_dict(),
            "state": self.state.value,
            "reason_code": self.reason_code,
            "backend_id": self.backend_id,
            "backend_provider": self.backend_provider,
            "backend_url": self.backend_url,
            "canonical_model_id": self.canonical_model_id,
            "profile_id": self.profile_id,
            "profile_kind": self.profile_kind,
            "profile_endpoint": self.profile_endpoint,
            "profile_supported_roles": list(self.profile_supported_roles),
            "profile_role_sensitive": self.profile_role_sensitive,
            "profile_input_type_required": self.profile_input_type_required,
            "request_enforcement": {
                "required": self.profile_input_type_required,
                "default": self.profile_input_type_default,
                "missing_role_behavior": self.profile_input_type_missing_role_behavior,
            },
            "profile_formatting_owner": self.profile_formatting_owner,
            "contract_schema_version": self.contract_schema_version,
            "contract_schema_sha256": self.contract_schema_sha256,
            "contract_snapshot_sha256": self.contract_snapshot_sha256,
            "index_compatibility_id": self.index_compatibility_id,
            "query_compatibility_id": self.query_compatibility_id,
            "issues": [dict(issue) for issue in self.issues],
            "discovery_attempts": self.discovery_attempts,
        }


@dataclass(frozen=True)
class ContractSnapshotDrift:
    """Redigierte Diagnose; Snapshot-Inhalte werden nie dupliziert."""

    raw_changed: bool
    canonical_changed: bool
    sha256_changed: bool
    stored_sha256: str
    current_sha256: str | None

    @property
    def detected(self) -> bool:
        return self.raw_changed or self.canonical_changed or self.sha256_changed

    def to_dict(self) -> dict[str, object]:
        return {
            "detected": self.detected,
            "reason_code": (
                "index_contract_snapshot_drift"
                if self.detected
                else "index_contract_snapshot_unchanged"
            ),
            "raw_changed": self.raw_changed,
            "canonical_changed": self.canonical_changed,
            "sha256_changed": self.sha256_changed,
            "stored_sha256": self.stored_sha256,
            "current_sha256": self.current_sha256,
        }


@dataclass(frozen=True)
class IndexGuardToken:
    target: IndexTarget
    operation: str
    write: bool
    validation_scope: IndexValidationScope
    require_configured_default: bool
    resolved: ResolvedIndexContract
    stored: dict[str, Any]
    snapshot_drift: ContractSnapshotDrift
    fence_version: int
    connection: Any

    def job_fields(self) -> dict[str, object]:
        return {
            "contract_logical_index_key": self.target.logical_index_key,
            "contract_index_compatibility_id": self.stored.get("index_compatibility_id"),
            "contract_query_compatibility_id": self.stored.get("query_compatibility_id"),
            "contract_canonical_model_id": self.stored.get("canonical_model_id"),
            "contract_profile_id": self.stored.get("profile_id"),
            "contract_profile_kind": self.stored.get("profile_kind"),
            "contract_profile_endpoint": self.stored.get("profile_endpoint"),
            # Jobs bleiben an den historischen Index-Provenienzbeleg gebunden.
            # Der operative aktuelle Discovery-Snapshot kann sich durch eine
            # Enforcement-Phase aendern, ohne einen anderen Vektorraum zu
            # bezeichnen.
            "contract_snapshot_sha256": self.stored.get("contract_snapshot_sha256"),
            "contract_fence_version": self.fence_version,
        }


class IndexCompatibilityError(RuntimeError):
    """Kontrollierter, nicht degradierbarer Indexvertragsfehler."""

    def __init__(
        self,
        reason_code: str,
        operation: str,
        target: IndexTarget,
        detail: str,
        *,
        stored_state: str | None = None,
        current_state: str | None = None,
        issues: tuple[dict[str, Any], ...] = (),
        discovery_attempts: int | None = None,
    ) -> None:
        super().__init__(
            f"Index-Kompatibilitaetsguard blockiert '{operation}' "
            f"fuer {target.collection_name}: {reason_code} ({detail})"
        )
        self.reason_code = reason_code
        self.operation = operation
        self.target = target
        self.detail = detail
        self.stored_state = stored_state
        self.current_state = current_state
        self.issues = tuple(dict(issue) for issue in issues)
        self.discovery_attempts = discovery_attempts
        self.rollback_diagnostics: list[dict[str, object]] = []

    def add_rollback_diagnostic(
        self,
        *,
        phase: str,
        error: Exception,
    ) -> None:
        """Haengt einen Rollbackfehler an, ohne die Primaerdiagnose zu ersetzen."""
        if isinstance(error, IndexCompatibilityError):
            diagnostic: dict[str, object] = error.to_dict()
        else:
            diagnostic = {
                "error": "rollback_failed",
                "exception_type": error.__class__.__name__,
                "detail": str(error),
            }
        self.rollback_diagnostics.append(
            {
                "phase": phase,
                "diagnostic": diagnostic,
            }
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "error": "index_compatibility_blocked",
            "reason_code": self.reason_code,
            "operation": self.operation,
            "detail": self.detail,
            "stored_state": self.stored_state,
            "current_state": self.current_state,
            **self.target.to_dict(),
            "operator_action": (
                "Contract-Discovery und Evidence pruefen; anschliessend den "
                "expliziten Index-Contract-Backfill/Audit ausfuehren."
            ),
        }
        if self.rollback_diagnostics:
            result["rollback_diagnostics"] = list(self.rollback_diagnostics)
        if self.issues:
            result["issues"] = [dict(issue) for issue in self.issues]
        if self.discovery_attempts is not None:
            result["discovery_attempts"] = self.discovery_attempts
        return result


class IndexContractUnavailableError(IndexCompatibilityError):
    """Transiente Index-Abhaengigkeitsstoerung ohne Contract-Mismatch."""

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update(
            {
                "error": "index_contract_temporarily_unavailable",
                "retryable": True,
                "operator_action": (
                    "Erreichbarkeit der betroffenen Index-Abhaengigkeit pruefen; "
                    "der Worker wartet fail-closed auf eine erneut vollstaendig "
                    "verifizierte Indexbindung."
                ),
            }
        )
        return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signed_lock_key(value: str) -> int:
    raw = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(raw, "big", signed=False)
    return number - (1 << 64) if number >= (1 << 63) else number


def safe_store_locator(raw_url: str) -> str:
    """Kanonische Qdrant-Instanz ohne Credentials, Query oder Fragment."""

    parsed = urlsplit((raw_url or DEFAULT_QDRANT_URL).strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("vectordb.qdrant_url ist keine gueltige HTTP(S)-URL")
    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("vectordb.qdrant_url enthaelt einen ungueltigen Port") from exc
    default_port = 80 if parsed.scheme == "http" else 443
    authority = host + (f":{port}" if port is not None and port != default_port else "")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), authority, path, "", ""))


def make_index_target(
    config: Mapping[str, Any],
    *,
    collection_name: str,
    vector_kind: IndexVectorKind | str,
    requested_model: str,
) -> IndexTarget:
    kind = IndexVectorKind(vector_kind)
    model = normalize_model_name(requested_model)
    if not model:
        raise ValueError("IndexTarget benoetigt einen nicht-leeren Modellnamen")
    locator = safe_store_locator(
        str(config.get("vectordb", {}).get("qdrant_url", DEFAULT_QDRANT_URL))
    )
    identity = {
        "version": 1,
        "store_type": "qdrant",
        "store_locator": locator,
        "collection_name": collection_name,
        "vector_kind": kind.value,
    }
    digest = _sha256_text(_canonical_json(identity))
    logical_key = f"qdrant-index-v1:sha256:{digest}"
    return IndexTarget(
        store_type="qdrant",
        store_locator=locator,
        collection_name=collection_name,
        vector_kind=kind,
        requested_model=model,
        logical_index_key=logical_key,
        lock_key=_signed_lock_key(logical_key),
    )


def dense_index_target(config: Mapping[str, Any], model: str) -> IndexTarget:
    from kiara.retrieval.embedding import collection_name_for_model

    normalized = normalize_model_name(model)
    return make_index_target(
        config,
        collection_name=collection_name_for_model(normalized),
        vector_kind=IndexVectorKind.DENSE,
        requested_model=normalized,
    )


def late_index_target(config: Mapping[str, Any], model: str) -> IndexTarget:
    from kiara.retrieval.embedding import collection_name_for_model

    normalized = normalize_model_name(model)
    return make_index_target(
        config,
        collection_name=collection_name_for_model(normalized),
        vector_kind=IndexVectorKind.LATE_CHUNKING,
        requested_model=normalized,
    )


def primary_index_target(config: Mapping[str, Any], model: str) -> IndexTarget:
    """Select the configured document/query processing for the existing model slot.

    The physical collection remains unchanged. Its persisted binding must still
    match; switching the configuration never reclassifies an existing index.
    """
    if (config.get("enrichment") or {}).get("late_chunking", False):
        return late_index_target(config, model)
    return dense_index_target(config, model)


def stored_primary_index_target(config: Mapping[str, Any], model: str) -> IndexTarget:
    """Resolve existing identity for maintenance, without choosing a live profile.

    The following stored-binding guard remains authoritative and fences the
    operation. Without a record, return the historical Dense candidate; a
    maintenance guard still rejects its missing proof. Current configuration
    must not reinterpret unbound historical vectors as Late.
    """
    from kiara.shared.db import db_query

    selected = dense_index_target(config, model)
    for token in reversed(_active_tokens.get()):
        target = token.target
        if (
            target.store_type == selected.store_type
            and target.store_locator == selected.store_locator
            and target.collection_name == selected.collection_name
        ):
            return target
    with db_query() as conn:
        row = conn.execute(
            "SELECT vector_kind FROM index_contracts WHERE store_type = %s "
            "AND store_locator = %s AND collection_name = %s",
            (selected.store_type, selected.store_locator, selected.collection_name),
        ).fetchone()
    if row is None:
        return selected
    kind = IndexVectorKind(row["vector_kind"])
    if kind not in (IndexVectorKind.DENSE, IndexVectorKind.LATE_CHUNKING):
        raise IndexCompatibilityError(
            "index_primary_binding_kind_invalid",
            "maintenance_target_resolution",
            selected,
            f"Primaerer Modellindex ist als {kind.value!r} gebunden",
        )
    return make_index_target(
        config,
        collection_name=selected.collection_name,
        vector_kind=kind,
        requested_model=selected.requested_model,
    )


def colbert_index_target(config: Mapping[str, Any], model: str) -> IndexTarget:
    from kiara.retrieval.colbert import colbert_collection_name

    normalized = normalize_model_name(model)
    return make_index_target(
        config,
        collection_name=colbert_collection_name(normalized),
        vector_kind=IndexVectorKind.MULTI_VECTOR,
        requested_model=normalized,
    )


def raw_qdrant_client(config: Mapping[str, Any]) -> QdrantClient:
    qdrant_cfg = config.get("vectordb", {})
    return QdrantClient(
        url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
        api_key=qdrant_cfg.get("qdrant_api_key") or None,
        timeout=qdrant_cfg.get("timeout", 60),
        prefer_grpc=bool(qdrant_cfg.get("qdrant_grpc")),
    )


def _backend_rows(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, category, provider, name, url, base_url, model, api_key, "
        "is_active, priority, extra_config FROM ki_backends "
        "WHERE category = 'embedding' AND is_active = true "
        "ORDER BY priority ASC, id ASC"
    ).fetchall()
    return [dict(row) for row in rows]


def _colbert_backend(
    config: Mapping[str, Any], rows: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, str]:
    explicit = str(config.get("retrieval", {}).get("colbert", {}).get("url") or "").rstrip("/")
    if explicit:
        matching = [row for row in rows if str(row.get("url") or "").rstrip("/") == explicit]
        return (matching[0] if matching else None), explicit

    # Ohne explizite ColBERT-URL ist der konfigurierte Backend-Prioritaets-
    # sieger bindend. Kein Health-basiertes get_next_backend(): Ein stiller
    # Wechsel auf einen anderen Server waere ein anderes Erzeugungsprofil.
    if rows and str(rows[0].get("url") or "").strip():
        return rows[0], str(rows[0]["url"]).rstrip("/")
    return None, ""


_TRANSIENT_DISCOVERY_HTTP_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _contract_discovery_settings(config: Mapping[str, Any]) -> tuple[int, float, float, float]:
    cfg = config.get("index_contract") or {}
    return (
        int(cfg.get("discovery_attempts", INDEX_CONTRACT_DISCOVERY_ATTEMPTS_DEFAULT)),
        float(
            cfg.get(
                "discovery_timeout_seconds",
                INDEX_CONTRACT_DISCOVERY_TIMEOUT_SECS_DEFAULT,
            )
        ),
        float(
            cfg.get(
                "discovery_backoff_initial_seconds",
                INDEX_CONTRACT_DISCOVERY_BACKOFF_INITIAL_SECS_DEFAULT,
            )
        ),
        float(
            cfg.get(
                "discovery_backoff_max_seconds",
                INDEX_CONTRACT_DISCOVERY_BACKOFF_MAX_SECS_DEFAULT,
            )
        ),
    )


def _issue_is_transient_discovery(issue: Mapping[str, Any]) -> bool:
    reason = str(issue.get("reason_code") or "")
    if reason == "contract_discovery_request_failed":
        return True
    if reason != "contract_discovery_http_error":
        return False
    detail = str(issue.get("detail") or "")
    if not detail.startswith("HTTP "):
        return False
    try:
        status = int(detail.removeprefix("HTTP ").strip())
    except ValueError:
        return False
    return status in _TRANSIENT_DISCOVERY_HTTP_STATUS


def _issues_are_transient_discovery(issues: tuple[dict[str, Any], ...]) -> bool:
    substantive = tuple(
        issue
        for issue in issues
        if str(issue.get("reason_code") or "") != "contract_server_invalid"
    )
    return bool(substantive) and all(_issue_is_transient_discovery(issue) for issue in substantive)


def _resolution_is_transient_discovery(resolved: ResolvedIndexContract) -> bool:
    return _issues_are_transient_discovery(resolved.issues)


def resolve_current_index_contract(
    conn,
    config: Mapping[str, Any],
    target: IndexTarget,
    *,
    force_refresh: bool = False,
) -> ResolvedIndexContract:
    """Loest genau das aktuell konfigurierte S3-Profil fail-closed auf."""

    rows = _backend_rows(conn)
    endpoint_by_kind = {
        IndexVectorKind.DENSE: "/api/embed",
        IndexVectorKind.LATE_CHUNKING: "/api/embed_late",
        IndexVectorKind.MULTI_VECTOR: "/api/embed_colbert",
    }
    profile_kind = ProfileKind(target.vector_kind.value)
    endpoint = endpoint_by_kind[target.vector_kind]
    profile_id: str | None = None

    if target.vector_kind is IndexVectorKind.MULTI_VECTOR:
        backend, backend_url = _colbert_backend(config, rows)
    else:
        matching = [
            row for row in rows if normalize_model_name(row.get("model")) == target.requested_model
        ]
        backend = matching[0] if matching else None
        backend_url = str(backend.get("url") or "").rstrip("/") if backend else ""
        if backend and target.vector_kind is IndexVectorKind.DENSE:
            profile_id, config_issues = select_configured_embedding_profile_id(backend)
            if config_issues:
                return ResolvedIndexContract(
                    state=ContractState.INCONSISTENT,
                    reason_code=config_issues[0].reason_code,
                    target=target,
                    backend=backend,
                    backend_id=int(backend["id"]),
                    backend_provider=str(backend.get("provider") or ""),
                    backend_url=backend_url,
                    issues=tuple(issue.to_dict() for issue in config_issues),
                )

    if backend is None and not backend_url:
        return ResolvedIndexContract(
            state=ContractState.MISSING,
            reason_code="index_backend_missing",
            target=target,
        )

    provider = str(backend.get("provider") or "ollama") if backend else "ollama"
    backend_id = int(backend["id"]) if backend and backend.get("id") is not None else None
    if provider != "ollama":
        return ResolvedIndexContract(
            state=ContractState.NOT_APPLICABLE,
            reason_code="contract_not_applicable_non_kiron_backend",
            target=target,
            backend=backend,
            backend_id=backend_id,
            backend_provider=provider,
            backend_url=backend_url,
        )
    if not backend_url:
        return ResolvedIndexContract(
            state=ContractState.MISSING,
            reason_code="contract_backend_url_missing",
            target=target,
            backend=backend,
            backend_id=backend_id,
            backend_provider=provider,
        )

    # The normalized target names the logical index; discovery requires the
    # exact configured backend reference, including case-sensitive model tags.
    discovery_model = (
        target.requested_model
        if target.vector_kind is IndexVectorKind.MULTI_VECTOR
        else str(backend["model"])
    )
    attempts, timeout, backoff_initial, backoff_max = _contract_discovery_settings(config)
    resolution = None
    attempts_used = 0
    for attempt in range(attempts):
        attempts_used = attempt + 1
        server = discover_server_contracts(
            backend_url,
            timeout=timeout,
            force_refresh=force_refresh or attempt > 0,
        )
        resolution = resolve_embedding_profile(
            server,
            discovery_model,
            endpoint=endpoint,
            profile_id=profile_id,
            kind=profile_kind,
        )
        issue_dicts = tuple(issue.to_dict() for issue in resolution.issues)
        if not _issues_are_transient_discovery(issue_dicts):
            break
        if attempts_used >= attempts:
            break
        delay = min(backoff_initial * (2**attempt), backoff_max)
        logger.warning(
            "Embedding-Vertragsdiscovery transient fehlgeschlagen "
            "(Versuch %d/%d, Backend=%s, Grund=%s); Retry in %.2fs",
            attempts_used,
            attempts,
            server.backend_url,
            issue_dicts[0].get("detail") if issue_dicts else "unbekannt",
            delay,
        )
        time.sleep(delay)

    if resolution is None:  # pragma: no cover - Configvalidierung erzwingt attempts >= 1
        raise AssertionError("Contract-Discovery ohne Versuch")
    reason_code = (
        resolution.issues[0].reason_code
        if resolution.issues
        else (
            "index_contract_verified"
            if resolution.rag_verified
            else f"contract_{resolution.state.value}"
        )
    )
    issues = tuple(issue.to_dict() for issue in resolution.issues)
    if resolution.capabilities is None or resolution.profile is None:
        return ResolvedIndexContract(
            state=resolution.state,
            reason_code=reason_code,
            target=target,
            backend=backend,
            backend_id=backend_id,
            backend_provider=provider,
            backend_url=backend_url,
            issues=issues,
            discovery_attempts=attempts_used,
        )

    capabilities = resolution.capabilities
    profile = resolution.profile
    artifact = {
        "repository": profile.artifact.repository,
        "revision": profile.artifact.revision,
        "manifest_digest": profile.artifact.manifest_digest,
        "weights": [item.__dict__ for item in profile.artifact.weights],
        "auxiliary": [item.__dict__ for item in profile.artifact.auxiliary],
    }
    return ResolvedIndexContract(
        state=resolution.state,
        reason_code=reason_code,
        target=target,
        backend=backend,
        backend_id=backend_id,
        backend_provider=provider,
        backend_url=backend_url,
        canonical_model_id=capabilities.canonical_model_id,
        profile_id=profile.profile_id,
        profile_kind=profile.kind.value,
        profile_endpoint=profile.endpoint,
        profile_supported_roles=profile.input_type.supported,
        profile_role_sensitive=profile.input_type.role_sensitive,
        profile_input_type_required=profile.input_type.required,
        profile_input_type_default=profile.input_type.default,
        profile_input_type_missing_role_behavior=profile.input_type.missing_role_behavior,
        profile_formatting_owner=profile.formatting_owner,
        contract_schema_version=SCHEMA_VERSION,
        contract_schema_sha256=SCHEMA_SHA256,
        contract_snapshot_json_text=capabilities.snapshot.json_text,
        contract_snapshot_canonical_text=capabilities.snapshot.canonical_json,
        contract_snapshot_sha256=capabilities.snapshot.canonical_sha256,
        index_compatibility_id=profile.index_compatibility_id,
        query_compatibility_id=profile.query_compatibility_id,
        backend_type=profile.backend.backend_type,
        backend_implementation=profile.backend.implementation,
        backend_implementation_revision=profile.backend.implementation_revision,
        artifact_repository=profile.artifact.repository,
        artifact_revision=profile.artifact.revision,
        artifact_manifest_digest=profile.artifact.manifest_digest,
        artifact_json_text=_canonical_json(artifact),
        issues=issues,
        discovery_attempts=attempts_used,
    )


_QDRANT_RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


def _qdrant_unavailability_cause(exc: Exception) -> Exception | None:
    """Liefert die transiente Ursache eines Qdrant-Fehlers, falls vorhanden.

    Der HTTP-Client von qdrant-client kapselt sowohl echte httpx-Transportfehler
    als auch lokale Response-Validierungsfehler in ``ResponseHandlingException``.
    Nur erstere sind ein Dienstausfall. Ein pauschaler Catch der Wrapper-Klasse
    wuerde inkompatible oder korrupte Qdrant-Antworten faelschlich endlos
    wiederholen.
    """
    from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    if isinstance(exc, ResourceExhaustedResponse):
        return exc
    if isinstance(exc, UnexpectedResponse):
        return exc if exc.status_code in _QDRANT_RETRYABLE_HTTP_STATUSES else None

    cause = exc.source if isinstance(exc, ResponseHandlingException) else exc
    if isinstance(cause, (httpx.TransportError, OSError)):
        return cause
    return None


def _collection_state(client: QdrantClient, collection_name: str) -> tuple[bool, int]:
    try:
        info = client.get_collection(collection_name)
    except Exception as exc:
        from qdrant_client.http.exceptions import UnexpectedResponse

        if isinstance(exc, UnexpectedResponse) and exc.status_code == 404:
            return False, 0
        raise
    count = info.points_count
    if count is None:
        count = client.count(collection_name=collection_name, exact=True).count
    if count is None:
        raise RuntimeError(f"Qdrant-Count fuer {collection_name} ist unbekannt")
    return True, int(count)


def _empty_evidence_payload(
    target: IndexTarget, resolved: ResolvedIndexContract, operation: str
) -> dict[str, Any]:
    references = "[]"
    diagnostics = _canonical_json(
        {
            "operation": operation,
            "reason_code": "index_empty_first_write_verified",
            "target": target.to_dict(),
            "resolved": resolved.diagnostic_dict(),
        }
    )
    return {
        "logical_index_key": target.logical_index_key,
        "evidence_type": "empty_first_write",
        "result_state": "verified",
        "reason_code": "index_empty_first_write_verified",
        "sample_count": 0,
        "collection_points_count": 0,
        "references_json_text": references,
        "references_sha256": _sha256_text(references),
        "diagnostics_json_text": diagnostics,
        "diagnostics_sha256": _sha256_text(diagnostics),
        "candidate_index_compatibility_id": resolved.index_compatibility_id,
        "candidate_query_compatibility_id": resolved.query_compatibility_id,
        "contract_snapshot_sha256": resolved.contract_snapshot_sha256,
    }


def insert_contract_evidence(conn, values: Mapping[str, Any]) -> int:
    columns = (
        "logical_index_key",
        "evidence_type",
        "result_state",
        "reason_code",
        "sample_count",
        "max_abs_error",
        "max_cosine_distance",
        "abs_tolerance",
        "cosine_tolerance",
        "collection_points_count",
        "collection_config_sha256",
        "sample_set_sha256",
        "references_json_text",
        "references_sha256",
        "diagnostics_json_text",
        "diagnostics_sha256",
        "candidate_index_compatibility_id",
        "candidate_query_compatibility_id",
        "contract_snapshot_sha256",
        "backfill_run_id",
    )
    row = conn.execute(
        f"INSERT INTO index_contract_evidence ({', '.join(columns)}) VALUES "
        f"({', '.join(['%s'] * len(columns))}) RETURNING id",
        tuple(values.get(column) for column in columns),
    ).fetchone()
    return int(row["id"])


def insert_index_contract(
    conn,
    target: IndexTarget,
    resolved: ResolvedIndexContract,
    *,
    evidence_id: int,
    evidence_type: str,
    evidence_sample_count: int,
    verification_state: str,
    reason_code: str,
    observed_points_count: int,
    backfill_run_id: str | None = None,
) -> dict[str, Any]:
    verified = verification_state == "verified"
    index_id = resolved.index_compatibility_id if verified else None
    query_id = resolved.query_compatibility_id if verified else None
    columns = (
        "logical_index_key",
        "lock_key",
        "store_type",
        "store_locator",
        "collection_name",
        "vector_kind",
        "requested_model",
        "backend_id",
        "backend_provider",
        "backend_url",
        "canonical_model_id",
        "profile_id",
        "profile_kind",
        "profile_endpoint",
        "contract_schema_version",
        "contract_schema_sha256",
        "contract_snapshot_json_text",
        "contract_snapshot_canonical_text",
        "contract_snapshot_sha256",
        "index_compatibility_id",
        "query_compatibility_id",
        "backend_type",
        "backend_implementation",
        "backend_implementation_revision",
        "artifact_repository",
        "artifact_revision",
        "artifact_manifest_digest",
        "artifact_json_text",
        "verification_state",
        "reason_code",
        "evidence_id",
        "evidence_type",
        "evidence_sample_count",
        "evidence_at",
        "observed_points_count",
        "backfill_run_id",
    )
    values = (
        target.logical_index_key,
        target.lock_key,
        target.store_type,
        target.store_locator,
        target.collection_name,
        target.vector_kind.value,
        target.requested_model,
        resolved.backend_id,
        resolved.backend_provider,
        resolved.backend_url,
        resolved.canonical_model_id,
        resolved.profile_id,
        resolved.profile_kind,
        resolved.profile_endpoint,
        resolved.contract_schema_version,
        resolved.contract_schema_sha256,
        resolved.contract_snapshot_json_text,
        resolved.contract_snapshot_canonical_text,
        resolved.contract_snapshot_sha256,
        index_id,
        query_id,
        resolved.backend_type,
        resolved.backend_implementation,
        resolved.backend_implementation_revision,
        resolved.artifact_repository,
        resolved.artifact_revision,
        resolved.artifact_manifest_digest,
        resolved.artifact_json_text,
        verification_state,
        reason_code,
        evidence_id,
        evidence_type,
        evidence_sample_count,
        "now()",
        observed_points_count,
        backfill_run_id,
    )
    placeholders = ["%s"] * len(columns)
    placeholders[33] = "now()"
    sql_values = values[:33] + values[34:]
    row = conn.execute(
        f"INSERT INTO index_contracts ({', '.join(columns)}) VALUES "
        f"({', '.join(placeholders)}) RETURNING *",
        sql_values,
    ).fetchone()
    return dict(row)


def _mismatch(
    operation: str,
    target: IndexTarget,
    reason_code: str,
    field: str,
    stored: object,
    current: object,
) -> IndexCompatibilityError:
    return IndexCompatibilityError(
        reason_code,
        operation,
        target,
        f"{field}: gespeichert={stored!r}, aktuell={current!r}",
        stored_state="verified",
        current_state="mismatch",
    )


def _validate_stored_snapshot_integrity(
    operation: str,
    target: IndexTarget,
    stored: Mapping[str, Any],
    resolved: ResolvedIndexContract,
) -> str:
    """Validate only the immutable historical provenance bundle.

    This deliberately does not compare the historical snapshot with current
    discovery. The evidence row is the independent persisted hash anchor, so
    a consistently rewritten Raw/Canonical/SHA bundle still fails closed.
    """

    state = str(stored.get("verification_state") or "")
    raw_snapshot = stored.get("contract_snapshot_json_text")
    stored_canonical = stored.get("contract_snapshot_canonical_text")
    stored_snapshot_hash = stored.get("contract_snapshot_sha256")
    if not all(
        isinstance(value, str) and bool(value)
        for value in (raw_snapshot, stored_canonical, stored_snapshot_hash)
    ):
        raise IndexCompatibilityError(
            "index_stored_snapshot_malformed",
            operation,
            target,
            "persistierter Contract-Snapshot-Bundle ist unvollstaendig",
            stored_state=state,
            current_state=resolved.state.value,
        )
    try:
        parsed_snapshot = decode_json_document(raw_snapshot.encode("utf-8"))
        if not isinstance(parsed_snapshot, Mapping):
            raise ValueError("snapshot root is not an object")
        recomputed_canonical = canonical_json(parsed_snapshot)
    except Exception as exc:
        raise IndexCompatibilityError(
            "index_stored_snapshot_malformed",
            operation,
            target,
            f"persistierter Contract-Snapshot ist ungueltig: {exc.__class__.__name__}",
            stored_state=state,
            current_state=resolved.state.value,
        ) from exc
    if recomputed_canonical != stored_canonical:
        raise IndexCompatibilityError(
            "index_stored_snapshot_canonical_mismatch",
            operation,
            target,
            "persistierter Rohsnapshot und kanonischer Snapshot stimmen nicht ueberein",
            stored_state=state,
            current_state=resolved.state.value,
        )
    if _sha256_text(recomputed_canonical) != stored_snapshot_hash:
        raise IndexCompatibilityError(
            "index_stored_snapshot_hash_mismatch",
            operation,
            target,
            "persistierter kanonischer Snapshot und SHA-256 stimmen nicht ueberein",
            stored_state=state,
            current_state=resolved.state.value,
        )

    evidence_snapshot_hash = stored.get("evidence_contract_snapshot_sha256")
    if not isinstance(evidence_snapshot_hash, str) or not evidence_snapshot_hash:
        raise IndexCompatibilityError(
            "index_stored_snapshot_evidence_hash_missing",
            operation,
            target,
            "persistierter Snapshot besitzt keinen Evidence-Hashanker",
            stored_state=state,
            current_state=resolved.state.value,
        )
    if evidence_snapshot_hash != stored_snapshot_hash:
        raise IndexCompatibilityError(
            "index_stored_snapshot_evidence_hash_mismatch",
            operation,
            target,
            "Contract- und Evidence-Snapshot-Hash stimmen nicht ueberein",
            stored_state=state,
            current_state=resolved.state.value,
        )
    return stored_snapshot_hash


def _validate_runtime_identity_binding(
    operation: str,
    target: IndexTarget,
    stored: Mapping[str, Any],
    resolved: ResolvedIndexContract,
) -> None:
    """Compare only semantic runtime identity, never request enforcement."""

    comparisons = (
        ("requested_model", target.requested_model, "index_requested_model_mismatch"),
        ("backend_id", resolved.backend_id, "index_backend_mismatch"),
        ("backend_provider", resolved.backend_provider, "index_backend_provider_mismatch"),
        ("backend_url", resolved.backend_url, "index_backend_url_mismatch"),
        ("canonical_model_id", resolved.canonical_model_id, "index_canonical_model_mismatch"),
        ("profile_id", resolved.profile_id, "index_profile_mismatch"),
        ("profile_kind", resolved.profile_kind, "index_profile_kind_mismatch"),
        ("profile_endpoint", resolved.profile_endpoint, "index_profile_endpoint_mismatch"),
        (
            "contract_schema_version",
            resolved.contract_schema_version,
            "index_contract_schema_version_mismatch",
        ),
        (
            "contract_schema_sha256",
            resolved.contract_schema_sha256,
            "index_contract_schema_hash_mismatch",
        ),
        (
            "index_compatibility_id",
            resolved.index_compatibility_id,
            "index_compatibility_id_mismatch",
        ),
        (
            "query_compatibility_id",
            resolved.query_compatibility_id,
            "index_query_compatibility_id_mismatch",
        ),
        ("backend_type", resolved.backend_type, "index_backend_type_mismatch"),
        (
            "backend_implementation",
            resolved.backend_implementation,
            "index_backend_implementation_mismatch",
        ),
        (
            "backend_implementation_revision",
            resolved.backend_implementation_revision,
            "index_backend_implementation_revision_mismatch",
        ),
        (
            "artifact_repository",
            resolved.artifact_repository,
            "index_artifact_repository_mismatch",
        ),
        (
            "artifact_revision",
            resolved.artifact_revision,
            "index_artifact_revision_mismatch",
        ),
        (
            "artifact_manifest_digest",
            resolved.artifact_manifest_digest,
            "index_artifact_manifest_mismatch",
        ),
        ("artifact_json_text", resolved.artifact_json_text, "index_artifact_mismatch"),
    )
    for field, current, reason_code in comparisons:
        stored_value = stored.get(field)
        if stored_value != current:
            raise _mismatch(operation, target, reason_code, field, stored_value, current)


def _snapshot_drift(
    stored: Mapping[str, Any], resolved: ResolvedIndexContract
) -> ContractSnapshotDrift:
    """Return hashes/booleans only; never copy contract contents into diagnostics."""

    return ContractSnapshotDrift(
        raw_changed=(
            stored.get("contract_snapshot_json_text") != resolved.contract_snapshot_json_text
        ),
        canonical_changed=(
            stored.get("contract_snapshot_canonical_text")
            != resolved.contract_snapshot_canonical_text
        ),
        sha256_changed=(
            stored.get("contract_snapshot_sha256") != resolved.contract_snapshot_sha256
        ),
        stored_sha256=str(stored["contract_snapshot_sha256"]),
        current_sha256=resolved.contract_snapshot_sha256,
    )


def _resolved_from_stored_binding(
    target: IndexTarget,
    stored: Mapping[str, Any],
) -> ResolvedIndexContract:
    """Lokale Vertragsansicht fuer nicht-semantische Indexoperationen."""

    return ResolvedIndexContract(
        state=ContractState.VALID,
        reason_code="index_stored_binding_verified",
        target=target,
        backend_id=stored.get("backend_id"),
        backend_provider=stored.get("backend_provider"),
        backend_url=stored.get("backend_url"),
        canonical_model_id=stored.get("canonical_model_id"),
        profile_id=stored.get("profile_id"),
        profile_kind=stored.get("profile_kind"),
        profile_endpoint=stored.get("profile_endpoint"),
        contract_schema_version=stored.get("contract_schema_version"),
        contract_schema_sha256=stored.get("contract_schema_sha256"),
        contract_snapshot_json_text=stored.get("contract_snapshot_json_text"),
        contract_snapshot_canonical_text=stored.get("contract_snapshot_canonical_text"),
        contract_snapshot_sha256=stored.get("contract_snapshot_sha256"),
        index_compatibility_id=stored.get("index_compatibility_id"),
        query_compatibility_id=stored.get("query_compatibility_id"),
        backend_type=stored.get("backend_type"),
        backend_implementation=stored.get("backend_implementation"),
        backend_implementation_revision=stored.get("backend_implementation_revision"),
        artifact_repository=stored.get("artifact_repository"),
        artifact_revision=stored.get("artifact_revision"),
        artifact_manifest_digest=stored.get("artifact_manifest_digest"),
        artifact_json_text=stored.get("artifact_json_text"),
    )


def _validate_stored_binding(
    operation: str,
    target: IndexTarget,
    stored: Mapping[str, Any],
) -> tuple[ResolvedIndexContract, ContractSnapshotDrift]:
    """Validiert die persistierte Indexbindung ohne KIron-Netzwerkzugriff."""

    resolved = _resolved_from_stored_binding(target, stored)
    state = str(stored.get("verification_state") or "")
    if state == "unverified":
        raise IndexCompatibilityError(
            "index_contract_unverified",
            operation,
            target,
            str(stored.get("reason_code") or "unverified"),
            stored_state=state,
            current_state="stored_binding",
        )
    if state == "evidence_failed":
        raise IndexCompatibilityError(
            "index_contract_evidence_failed",
            operation,
            target,
            str(stored.get("reason_code") or "evidence_failed"),
            stored_state=state,
            current_state="stored_binding",
        )
    if state != "verified":
        raise IndexCompatibilityError(
            "index_contract_state_invalid",
            operation,
            target,
            f"unbekannter persistierter Zustand {state!r}",
            stored_state=state,
            current_state="stored_binding",
        )

    _validate_stored_snapshot_integrity(operation, target, stored, resolved)
    comparisons = (
        ("store_type", target.store_type, "index_store_type_mismatch"),
        ("store_locator", target.store_locator, "index_store_locator_mismatch"),
        ("collection_name", target.collection_name, "index_collection_name_mismatch"),
        ("vector_kind", target.vector_kind.value, "index_vector_kind_mismatch"),
        ("requested_model", target.requested_model, "index_requested_model_mismatch"),
    )
    for field, current, reason_code in comparisons:
        stored_value = stored.get(field)
        if stored_value != current:
            raise _mismatch(operation, target, reason_code, field, stored_value, current)
    if not resolved.verified:
        raise IndexCompatibilityError(
            "index_stored_binding_incomplete",
            operation,
            target,
            "persistierte Compatibility-IDs fehlen",
            stored_state=state,
            current_state="stored_binding",
        )

    stored_hash = str(stored["contract_snapshot_sha256"])
    return resolved, ContractSnapshotDrift(
        raw_changed=False,
        canonical_changed=False,
        sha256_changed=False,
        stored_sha256=stored_hash,
        current_sha256=stored_hash,
    )


def _validate_stored_contract(
    operation: str,
    target: IndexTarget,
    stored: Mapping[str, Any],
    resolved: ResolvedIndexContract,
) -> ContractSnapshotDrift:
    state = str(stored.get("verification_state") or "")
    if state == "unverified":
        raise IndexCompatibilityError(
            "index_contract_unverified",
            operation,
            target,
            str(stored.get("reason_code") or "unverified"),
            stored_state=state,
            current_state=resolved.state.value,
        )
    if state == "evidence_failed":
        raise IndexCompatibilityError(
            "index_contract_evidence_failed",
            operation,
            target,
            str(stored.get("reason_code") or "evidence_failed"),
            stored_state=state,
            current_state=resolved.state.value,
        )
    if state != "verified":
        raise IndexCompatibilityError(
            "index_contract_state_invalid",
            operation,
            target,
            f"unbekannter persistierter Zustand {state!r}",
            stored_state=state,
            current_state=resolved.state.value,
        )
    _validate_stored_snapshot_integrity(operation, target, stored, resolved)

    if not resolved.verified:
        error_type = (
            IndexContractUnavailableError
            if _resolution_is_transient_discovery(resolved)
            else IndexCompatibilityError
        )
        raise error_type(
            "index_current_profile_not_verified",
            operation,
            target,
            resolved.reason_code,
            stored_state=state,
            current_state=resolved.state.value,
            issues=resolved.issues,
            discovery_attempts=resolved.discovery_attempts or None,
        )

    # ``resolve_current_index_contract`` only exposes a profile after the S3
    # parser has validated Schema/JCS and recomputed both normative IDs. The
    # complete discovery bytes can still change for non-identity policy or
    # diagnostics and therefore must not participate in this comparison.
    _validate_runtime_identity_binding(operation, target, stored, resolved)
    return _snapshot_drift(stored, resolved)


_active_tokens: ContextVar[tuple[IndexGuardToken, ...]] = ContextVar(
    "index_compatibility_guard_tokens", default=()
)

_source_cleanup_connection: ContextVar[Any] = ContextVar("source_cleanup_connection", default=None)


@contextmanager
def source_cleanup_fence():
    """Freeze index writers/creation while resolving and consuming source scope.

    The exclusive config fence must precede every index lock. It also covers
    as-yet unknown ColBERT collections: locking only discovered candidates
    would leave a collection-creation race. Nested guards reuse this connection
    but still perform their complete contract validation. No guard token is
    issued by this inventory fence.

    Only source-local ID inventories may be inspected without a verified
    binding. Semantic reads and every mutation still require guard_index.
    """
    from kiara.shared.db import db_transaction

    if _active_tokens.get() or _source_cleanup_connection.get() is not None:
        raise RuntimeError("source_cleanup_fence_requires_no_active_index_guard")
    with db_transaction() as conn:
        # Do not queue an exclusive waiter: an existing multi-index operation
        # may still need a shared config lock on its next guard connection.
        # A queued exclusive waiter would block that nested guard while waiting
        # for its parent to finish (an application-level deadlock).
        timeout = float(
            conn.execute(
                "SELECT EXTRACT(EPOCH FROM current_setting('statement_timeout')::interval) "
                "AS seconds"
            ).fetchone()["seconds"]
        )
        deadline = time.monotonic() + timeout if timeout else None
        while not conn.execute(
            "SELECT pg_try_advisory_xact_lock(%s) AS acquired", (INDEX_CONTRACT_CONFIG_LOCK_KEY,)
        ).fetchone()["acquired"]:
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError("source_cleanup_fence_busy: Zeitlimit fuer Indexsperre erreicht")
            time.sleep(0.05)
        reset = _source_cleanup_connection.set(conn)
        try:
            yield
        finally:
            _source_cleanup_connection.reset(reset)


def require_source_cleanup_fence() -> None:
    """Inventory access is permitted only inside the exclusive cleanup fence."""
    if _source_cleanup_connection.get() is None:
        raise RuntimeError("source_cleanup_inventory_without_fence")


@contextmanager
def guard_index(
    config: Mapping[str, Any],
    target: IndexTarget,
    *,
    operation: IndexOperation | str,
    write: bool = False,
    allow_initialize: bool = False,
    allow_missing_for_retire: bool = False,
    require_configured_default: bool = False,
    qdrant_client: QdrantClient | None = None,
    force_refresh: bool = False,
    validation_scope: IndexValidationScope | str = IndexValidationScope.LIVE_PROFILE,
) -> Iterator[IndexGuardToken]:
    """Prueft und fenced einen Index ueber das gesamte kritische Fenster."""

    operation_value = operation.value if isinstance(operation, IndexOperation) else str(operation)
    scope = IndexValidationScope(validation_scope)
    for active in reversed(_active_tokens.get()):
        if active.target.logical_index_key != target.logical_index_key:
            continue
        if write and not active.write:
            raise IndexCompatibilityError(
                "index_guard_lock_escalation",
                operation_value,
                target,
                "ein laufender Read-Guard kann nicht zum Write-Guard hochgestuft werden",
            )
        if require_configured_default and not active.require_configured_default:
            raise IndexCompatibilityError(
                "index_guard_default_fence_missing",
                operation_value,
                target,
                "laufender Guard bindet nicht das aktuell konfigurierte Default-Modell",
            )
        if (
            scope is IndexValidationScope.LIVE_PROFILE
            and active.validation_scope is IndexValidationScope.STORED_BINDING
        ):
            raise IndexCompatibilityError(
                "index_guard_validation_scope_escalation",
                operation_value,
                target,
                "ein Stored-Binding-Guard kann keine semantische Live-Pruefung ersetzen",
                stored_state="verified",
                current_state="stored_binding",
            )
        yield active
        return

    from kiara.shared.db import db_transaction

    # Nested guards return above, without building a client. An owned client
    # is created before and closed after the DB fence, never under extra locks.
    client_context = (
        nullcontext(qdrant_client)
        if qdrant_client is not None
        else closing(raw_qdrant_client(config))
    )
    cleanup_connection = _source_cleanup_connection.get()
    transaction = (
        nullcontext(cleanup_connection) if cleanup_connection is not None else db_transaction()
    )
    with client_context as client, transaction as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock_shared(%s)",
            (INDEX_CONTRACT_CONFIG_LOCK_KEY,),
        )
        if require_configured_default:
            if target.vector_kind not in (IndexVectorKind.DENSE, IndexVectorKind.LATE_CHUNKING):
                raise IndexCompatibilityError(
                    "index_default_guard_kind_invalid",
                    operation_value,
                    target,
                    "Default-Modell-Fence ist nur fuer primaere Dense-/Late-Indizes definiert",
                )
            configured_rows = _backend_rows(conn)
            configured_model = (
                normalize_model_name(configured_rows[0].get("model")) if configured_rows else ""
            )
            if configured_model != target.requested_model:
                raise IndexCompatibilityError(
                    "index_configured_default_model_mismatch",
                    operation_value,
                    target,
                    (
                        f"Operation waehlt {target.requested_model!r}, "
                        f"aktuell konfiguriert ist {configured_model!r}"
                    ),
                    current_state="configured_model_changed",
                )
        lock_fn = "pg_advisory_xact_lock" if write else "pg_advisory_xact_lock_shared"
        conn.execute(f"SELECT {lock_fn}(%s)", (target.lock_key,))

        physical = conn.execute(
            "SELECT * FROM index_contracts WHERE store_type = %s AND store_locator = %s "
            "AND collection_name = %s FOR SHARE",
            (target.store_type, target.store_locator, target.collection_name),
        ).fetchone()
        if physical and physical["logical_index_key"] != target.logical_index_key:
            raise IndexCompatibilityError(
                "index_physical_binding_conflict",
                operation_value,
                target,
                (
                    f"Collection ist bereits als {physical['vector_kind']} an "
                    f"{physical['logical_index_key']} gebunden"
                ),
                stored_state=str(physical.get("verification_state") or ""),
            )

        stored_row = conn.execute(
            "SELECT c.*, "
            "e.contract_snapshot_sha256 AS evidence_contract_snapshot_sha256 "
            "FROM index_contracts c "
            "JOIN index_contract_evidence e ON e.id = c.evidence_id "
            "WHERE c.logical_index_key = %s FOR SHARE OF c, e",
            (target.logical_index_key,),
        ).fetchone()
        try:
            exists, points_count = _collection_state(client, target.collection_name)
        except Exception as exc:
            transient_cause = _qdrant_unavailability_cause(exc)
            if transient_cause is None:
                raise
            cause_detail = f"{transient_cause.__class__.__name__}: {transient_cause}"
            raise IndexContractUnavailableError(
                "qdrant_temporarily_unavailable",
                operation_value,
                target,
                f"Qdrant unter {target.store_locator} nicht erreichbar ({cause_detail})",
                stored_state=(
                    str(stored_row.get("verification_state") or "") if stored_row else None
                ),
                current_state="unavailable",
                issues=(
                    {
                        "reason_code": "qdrant_request_failed",
                        "field_path": target.store_locator,
                        "detail": cause_detail,
                    },
                ),
            ) from exc
        resolved: ResolvedIndexContract | None = None
        if scope is IndexValidationScope.LIVE_PROFILE:
            resolved = resolve_current_index_contract(
                conn,
                config,
                target,
                force_refresh=force_refresh,
            )

        if stored_row is None:
            if scope is IndexValidationScope.STORED_BINDING:
                detail = (
                    f"Collection hat {points_count} Punkte"
                    if exists
                    else "Collection existiert nicht"
                )
                raise IndexCompatibilityError(
                    "index_contract_missing",
                    operation_value,
                    target,
                    detail,
                    current_state="stored_binding",
                )
            if resolved is None:  # pragma: no cover - Scope-Zweig ist vollstaendig
                raise AssertionError("Live-Guard ohne aufgeloesten Vertrag")
            if not (write and allow_initialize):
                detail = (
                    f"Collection hat {points_count} Punkte"
                    if exists
                    else "Collection existiert nicht"
                )
                raise IndexCompatibilityError(
                    "index_contract_missing",
                    operation_value,
                    target,
                    detail,
                    current_state=resolved.state.value,
                )
            if points_count != 0:
                raise IndexCompatibilityError(
                    "index_collection_nonempty_without_contract",
                    operation_value,
                    target,
                    f"Bestandscollection hat {points_count} Punkte; expliziter Witness erforderlich",
                    current_state=resolved.state.value,
                )
            if not resolved.verified:
                raise IndexCompatibilityError(
                    "index_first_write_profile_not_verified",
                    operation_value,
                    target,
                    resolved.reason_code,
                    current_state=resolved.state.value,
                )
            evidence_values = _empty_evidence_payload(target, resolved, operation_value)
            evidence_id = insert_contract_evidence(conn, evidence_values)
            stored = insert_index_contract(
                conn,
                target,
                resolved,
                evidence_id=evidence_id,
                evidence_type="empty_first_write",
                evidence_sample_count=0,
                verification_state="verified",
                reason_code="index_empty_first_write_verified",
                observed_points_count=0,
            )
            stored["evidence_contract_snapshot_sha256"] = evidence_values[
                "contract_snapshot_sha256"
            ]
            snapshot_drift = _snapshot_drift(stored, resolved)
        else:
            stored = dict(stored_row)
            if not exists:
                pending_first_write = (
                    write
                    and allow_initialize
                    and stored.get("verification_state") == "verified"
                    and stored.get("evidence_type") == "empty_first_write"
                    and int(stored.get("observed_points_count") or 0) == 0
                )
                explicit_retire_recovery = write and allow_missing_for_retire
                if not pending_first_write and not explicit_retire_recovery:
                    raise IndexCompatibilityError(
                        "index_collection_missing",
                        operation_value,
                        target,
                        "persistierter Indexvertrag verweist auf eine fehlende Collection",
                        stored_state=str(stored.get("verification_state") or ""),
                        current_state=(
                            resolved.state.value if resolved is not None else "stored_binding"
                        ),
                    )
            if scope is IndexValidationScope.STORED_BINDING:
                resolved, snapshot_drift = _validate_stored_binding(
                    operation_value,
                    target,
                    stored,
                )
            else:
                if resolved is None:  # pragma: no cover - Scope-Zweig ist vollstaendig
                    raise AssertionError("Live-Guard ohne aufgeloesten Vertrag")
                snapshot_drift = _validate_stored_contract(
                    operation_value,
                    target,
                    stored,
                    resolved,
                )

        if resolved is None:  # pragma: no cover - alle Erfolgszweige setzen resolved
            raise AssertionError("Index-Guard ohne Vertragsansicht")

        token = IndexGuardToken(
            target=target,
            operation=operation_value,
            write=write,
            validation_scope=scope,
            require_configured_default=require_configured_default,
            resolved=resolved,
            stored=stored,
            snapshot_drift=snapshot_drift,
            fence_version=int(stored["fence_version"]),
            connection=conn,
        )
        previous = _active_tokens.get()
        reset_token = _active_tokens.set((*previous, token))
        try:
            yield token
        finally:
            _active_tokens.reset(reset_token)


def current_guard_token(
    *, target: IndexTarget | None = None, model: str | None = None
) -> IndexGuardToken | None:
    normalized = normalize_model_name(model) if model is not None else None
    for token in reversed(_active_tokens.get()):
        if target is not None and token.target.logical_index_key != target.logical_index_key:
            continue
        if normalized is not None and token.target.requested_model != normalized:
            continue
        return token
    return None


def retire_index_contract(token: IndexGuardToken) -> None:
    """Retire only the active binding after an explicit collection delete.

    Evidence remains immutable.  The deletion uses the same transaction and
    fence version as the Qdrant critical window; a later access therefore
    observes either the old verified binding or the safe missing state.
    """

    if not token.write:
        raise IndexCompatibilityError(
            "index_contract_retire_requires_write_guard",
            token.operation,
            token.target,
            "aktiver Guard ist read-only",
        )
    deleted = token.connection.execute(
        "DELETE FROM index_contracts WHERE logical_index_key = %s AND fence_version = %s "
        "RETURNING logical_index_key",
        (token.target.logical_index_key, token.fence_version),
    ).fetchone()
    if deleted is None:
        raise IndexCompatibilityError(
            "index_contract_retire_fence_mismatch",
            token.operation,
            token.target,
            f"Fence-Version {token.fence_version} konnte nicht atomar entfernt werden",
        )


def exact_backend_from_guard(model: str, *, allow_colbert: bool = False) -> dict[str, Any] | None:
    normalized = normalize_model_name(model)
    for token in reversed(_active_tokens.get()):
        if token.target.requested_model != normalized:
            continue
        if token.validation_scope is not IndexValidationScope.LIVE_PROFILE:
            continue
        if token.target.vector_kind is IndexVectorKind.MULTI_VECTOR and not allow_colbert:
            continue
        return dict(token.resolved.backend) if token.resolved.backend else None
    return None


@contextmanager
def guard_job_contract(
    config: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    operation: IndexOperation | str,
    write: bool,
    allow_initialize: bool = False,
    require_configured_default: bool = False,
    qdrant_client: QdrantClient | None = None,
    force_refresh: bool = False,
    validation_scope: IndexValidationScope | str = IndexValidationScope.LIVE_PROFILE,
) -> Iterator[IndexGuardToken]:
    model = normalize_model_name(str(job.get("embedding_model") or ""))
    target = primary_index_target(config, model)
    with guard_index(
        config,
        target,
        operation=operation,
        write=write,
        allow_initialize=allow_initialize,
        require_configured_default=require_configured_default,
        qdrant_client=qdrant_client,
        force_refresh=force_refresh,
        validation_scope=validation_scope,
    ) as token:
        validate_job_contract_snapshot(job, token, operation=operation)
        yield token


@contextmanager
def guard_index_for_active_job(
    config: Mapping[str, Any],
    target: IndexTarget,
    *,
    operation: IndexOperation | str,
    write: bool,
    allow_initialize: bool = False,
    allow_missing_for_retire: bool = False,
    require_configured_default: bool = False,
    force_refresh: bool = False,
    qdrant_client: QdrantClient | None = None,
    validation_scope: IndexValidationScope | str = IndexValidationScope.LIVE_PROFILE,
) -> Iterator[IndexGuardToken]:
    """Fence an index and, inside a worker run, its immutable job snapshot.

    The worker stores the selected primary contract at enqueue time. Every primary
    embedding and Qdrant operation therefore re-reads and row-locks that job
    before entering its external critical window. ColBERT remains a separate
    secondary contract and can never borrow this primary job proof.
    """
    from kiara.ingestion.runtime import _get_current_job_id

    job_id = _get_current_job_id()
    if job_id is None or target.vector_kind is IndexVectorKind.MULTI_VECTOR:
        with guard_index(
            config,
            target,
            operation=operation,
            write=write,
            allow_initialize=allow_initialize,
            allow_missing_for_retire=allow_missing_for_retire,
            require_configured_default=require_configured_default,
            force_refresh=force_refresh,
            qdrant_client=qdrant_client,
            validation_scope=validation_scope,
        ) as token:
            yield token
        return

    from kiara.shared.db import db_query

    operation_value = operation.value if isinstance(operation, IndexOperation) else str(operation)
    with db_query() as read_db:
        row = read_db.execute(
            "SELECT * FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
    if not row:
        raise IndexCompatibilityError(
            "index_active_job_missing",
            operation_value,
            target,
            f"Aktiver Worker-Job #{job_id} existiert nicht",
            stored_state="job_snapshot",
        )
    job = dict(row)
    if job.get("contract_logical_index_key") != target.logical_index_key:
        raise IndexCompatibilityError(
            "index_job_target_mismatch",
            operation_value,
            target,
            (
                f"Job #{job_id}: gespeichert={job.get('contract_logical_index_key')!r}, "
                f"Operation={target.logical_index_key!r}"
            ),
            stored_state="job_snapshot",
        )

    with guard_job_contract(
        config,
        job,
        operation=operation,
        write=write,
        allow_initialize=allow_initialize,
        require_configured_default=require_configured_default,
        qdrant_client=qdrant_client,
        force_refresh=force_refresh,
        validation_scope=validation_scope,
    ) as token:
        locked = token.connection.execute(
            "SELECT * FROM ingestion_jobs WHERE id = %s FOR UPDATE",
            (job_id,),
        ).fetchone()
        if not locked:
            raise IndexCompatibilityError(
                "index_active_job_disappeared",
                operation_value,
                target,
                f"Aktiver Worker-Job #{job_id} verschwand vor dem kritischen Fenster",
                stored_state="job_snapshot",
            )
        validate_job_contract_snapshot(dict(locked), token, operation=operation)
        if token.target.logical_index_key != target.logical_index_key:
            raise IndexCompatibilityError(
                "index_job_target_mismatch",
                operation_value,
                target,
                (
                    f"Job #{job_id}: Guard={token.target.logical_index_key!r}, "
                    f"Operation={target.logical_index_key!r}"
                ),
                stored_state="job_snapshot",
            )
        yield token


def validate_job_contract_snapshot(
    job: Mapping[str, Any],
    token: IndexGuardToken,
    *,
    operation: IndexOperation | str,
) -> None:
    """Validate a freshly row-locked job against the active fence token."""

    operation_value = operation.value if isinstance(operation, IndexOperation) else str(operation)
    job_model = normalize_model_name(str(job.get("embedding_model") or ""))
    if job_model != token.target.requested_model:
        raise IndexCompatibilityError(
            "index_job_embedding_model_mismatch",
            operation_value,
            token.target,
            (f"Job-Modell gespeichert={job_model!r}, Guard={token.target.requested_model!r}"),
            stored_state="job_snapshot",
            current_state="verified",
        )
    for field, value in token.job_fields().items():
        if job.get(field) != value:
            reason = (
                "index_job_contract_missing"
                if job.get(field) is None
                else "index_job_contract_mismatch"
            )
            raise IndexCompatibilityError(
                reason,
                operation_value,
                token.target,
                f"Job-Feld {field}: gespeichert={job.get(field)!r}, aktuell={value!r}",
                stored_state="job_snapshot",
                current_state="verified",
            )


class GuardedQdrantClient:
    """Qdrant-Proxy: jeder Verbraucherzugriff passiert den zentralen Guard."""

    _READ_METHODS = frozenset(
        {
            "get_collection",
            "get_collections",
            "collection_exists",
            "count",
            "retrieve",
            "scroll",
            "query_points",
            "query_batch_points",
            "search",
            "search_batch",
            "facet",
        }
    )
    _STORED_BINDING_METHODS = frozenset(
        {
            "get_collection",
            "get_collections",
            "collection_exists",
            "count",
            "retrieve",
            "scroll",
            "facet",
            "delete",
            "delete_collection",
            "set_payload",
            "overwrite_payload",
            "delete_payload",
            "clear_payload",
            "create_payload_index",
            "delete_payload_index",
        }
    )

    def __init__(
        self,
        raw_client: QdrantClient,
        config: Mapping[str, Any],
        target: IndexTarget,
        *,
        require_configured_default: bool = False,
    ) -> None:
        self._raw_client = raw_client
        self._config = config
        self._target = target
        self._require_configured_default = require_configured_default

    @property
    def raw_client(self) -> QdrantClient:
        return self._raw_client

    def close(self, *args, **kwargs):
        """Schliesst nur den Transport, auch wenn das Modell nicht verfuegbar ist."""
        return self._raw_client.close(*args, **kwargs)

    def __getattr__(self, name: str):
        attribute = getattr(self._raw_client, name)
        if not callable(attribute):
            return attribute

        def guarded_call(*args, **kwargs):
            write = name not in self._READ_METHODS
            operation = IndexOperation.WRITE if write else IndexOperation.READ
            validation_scope = (
                IndexValidationScope.STORED_BINDING
                if name in self._STORED_BINDING_METHODS
                else IndexValidationScope.LIVE_PROFILE
            )
            with guard_index_for_active_job(
                self._config,
                self._target,
                operation=operation,
                write=write,
                require_configured_default=self._require_configured_default,
                qdrant_client=self._raw_client,
                validation_scope=validation_scope,
            ):
                return attribute(*args, **kwargs)

        return guarded_call


def list_persisted_contracts(conn) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT c.*, e.max_abs_error, e.max_cosine_distance, e.abs_tolerance, "
        "e.cosine_tolerance, e.collection_config_sha256, e.sample_set_sha256, "
        "e.contract_snapshot_sha256 AS evidence_contract_snapshot_sha256, "
        "e.created_at AS evidence_created_at "
        "FROM index_contracts c JOIN index_contract_evidence e ON e.id = c.evidence_id "
        "ORDER BY c.store_locator, c.collection_name"
    ).fetchall()
    return [dict(row) for row in rows]
