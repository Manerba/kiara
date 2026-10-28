"""Operationsbasierte Rollen fuer vertragsgestuetzte Embedding-Requests.

KIara entscheidet ausschliesslich anhand des fachlichen Vorgangs, ob ein
Dokument oder eine Suchanfrage vektorisiert wird. Modellformatierung bleibt
vollstaendig beim Server; Modellnamen sind weder Rollen- noch
Formatierungsquelle.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Any

from kiara.retrieval.embedding_contract import ContractState, ProfileKind

logger = logging.getLogger(__name__)

SEARCH_DOCUMENT = "search_document"
SEARCH_QUERY = "search_query"
_CANONICAL_ROLES = frozenset({SEARCH_DOCUMENT, SEARCH_QUERY})


@dataclass(frozen=True)
class EmbeddingRequestProfile:
    """Nicht-sensibler, validierter Profilbezug eines Embedding-Requests."""

    requested_model: str
    canonical_model_id: str
    profile_id: str
    kind: str
    endpoint: str
    input_type: str
    role_sensitive: bool
    index_compatibility_id: str
    query_compatibility_id: str

    @property
    def active_compatibility_id(self) -> str:
        if self.input_type == SEARCH_QUERY:
            return self.query_compatibility_id
        return self.index_compatibility_id

    def telemetry(self) -> dict[str, str | bool]:
        return {
            "requested_model": self.requested_model,
            "canonical_model_id": self.canonical_model_id,
            "profile_id": self.profile_id,
            "kind": self.kind,
            "endpoint": self.endpoint,
            "input_type": self.input_type,
            "role_sensitive": self.role_sensitive,
            "compatibility_id": self.active_compatibility_id,
        }


class EmbeddingRoleContractError(RuntimeError):
    """Kontrollierter Fehler fuer direkte, noch nicht indexgebundene Probes."""

    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        requested_model: str,
        endpoint: str,
        input_type: str,
        profile_id: str | None = None,
    ) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail
        self.requested_model = requested_model
        self.endpoint = endpoint
        self.input_type = input_type
        self.profile_id = profile_id

    def to_dict(self) -> dict[str, str | None]:
        return {
            "reason_code": self.reason_code,
            "detail": self.detail,
            "requested_model": self.requested_model,
            "endpoint": self.endpoint,
            "input_type": self.input_type,
            "profile_id": self.profile_id,
        }


def normalize_semantic_query_input(value: str) -> str:
    """Normalisiert genau den fachlichen Text, der Query-Vektor und Cache bindet.

    Unicode-Komposition, Zeilenenden und aeussere Leerzeichen werden
    kanonisiert. Interne Leerzeichen, Grossschreibung und Interpunktion bleiben
    erhalten, damit fachlich unterschiedliche Queries nie kollidieren.
    """

    if not isinstance(value, str):
        raise TypeError("Embedding-Query muss ein String sein")
    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("Query darf nicht leer sein")
    return normalized


def _validate_role(role: str) -> None:
    if role not in _CANONICAL_ROLES:
        raise ValueError(f"Ungueltige kanonische Embedding-Rolle: {role!r}")


def voyage_input_type_for_role(role: str) -> str:
    """Mappt KIaras kanonische Operation auf das native Voyage-Protokoll."""

    _validate_role(role)
    return "document" if role == SEARCH_DOCUMENT else "query"


def _request_profile_from_parts(
    *,
    requested_model: str,
    canonical_model_id: str | None,
    profile_id: str | None,
    kind: str | None,
    endpoint: str | None,
    supported_roles: tuple[str, ...],
    role_sensitive: bool | None,
    formatting_owner: str | None,
    index_compatibility_id: str | None,
    query_compatibility_id: str | None,
    role: str,
    expected_kind: ProfileKind,
    expected_endpoint: str,
    fail,
) -> EmbeddingRequestProfile:
    if role not in _CANONICAL_ROLES:
        fail(
            "embedding_role_invalid",
            f"Rolle {role!r} ist keine kanonische RAG-Rolle",
        )
    if kind != expected_kind.value:
        fail(
            "embedding_profile_kind_mismatch",
            f"Profilart {kind!r} statt {expected_kind.value!r}",
        )
    if endpoint != expected_endpoint:
        fail(
            "embedding_profile_endpoint_mismatch",
            f"Profilendpoint {endpoint!r} statt {expected_endpoint!r}",
        )
    if role not in supported_roles:
        fail(
            "embedding_role_unsupported",
            f"Rolle {role!r} ist im Profil nicht deklariert",
        )
    if formatting_owner != "server":
        fail(
            "embedding_formatting_owner_mismatch",
            f"formatting_owner={formatting_owner!r} statt 'server'",
        )
    if role_sensitive is None:
        fail(
            "embedding_role_sensitivity_unverified",
            "role_sensitive ist nicht verifiziert",
        )
    if not canonical_model_id or not profile_id:
        fail(
            "embedding_profile_identity_missing",
            "kanonische Modell-/Profilidentitaet fehlt",
        )
    if not index_compatibility_id or not query_compatibility_id:
        fail(
            "embedding_compatibility_id_missing",
            "Index- oder Query-Compatibility-ID fehlt",
        )
    return EmbeddingRequestProfile(
        requested_model=requested_model,
        canonical_model_id=canonical_model_id,
        profile_id=profile_id,
        kind=kind,
        endpoint=endpoint,
        input_type=role,
        role_sensitive=role_sensitive,
        index_compatibility_id=index_compatibility_id,
        query_compatibility_id=query_compatibility_id,
    )


def request_profile_from_guard(
    token: Any,
    *,
    role: str,
    expected_kind: ProfileKind,
    expected_endpoint: str,
) -> EmbeddingRequestProfile:
    """Bindet einen Request an den bereits atomar geprueften Indexvertrag."""

    from kiara.retrieval.index_compatibility import IndexCompatibilityError

    if token is None:
        raise RuntimeError("Embedding-Rollenvalidierung ohne aktiven Index-Guard")
    resolved = token.resolved
    state = getattr(resolved, "state", ContractState.VALID)

    def fail(reason_code: str, detail: str) -> None:
        raise IndexCompatibilityError(
            reason_code,
            token.operation,
            token.target,
            detail,
            current_state=getattr(state, "value", str(state)),
        )

    if state is not ContractState.VALID:
        fail(
            "embedding_profile_unverified",
            f"Vertragsstatus ist {getattr(state, 'value', state)!r}",
        )
    return _request_profile_from_parts(
        requested_model=token.target.requested_model,
        canonical_model_id=getattr(resolved, "canonical_model_id", None),
        profile_id=getattr(resolved, "profile_id", None),
        kind=getattr(resolved, "profile_kind", None),
        endpoint=getattr(resolved, "profile_endpoint", None),
        supported_roles=tuple(getattr(resolved, "profile_supported_roles", ())),
        role_sensitive=getattr(resolved, "profile_role_sensitive", None),
        formatting_owner=getattr(resolved, "profile_formatting_owner", None),
        index_compatibility_id=getattr(resolved, "index_compatibility_id", None),
        query_compatibility_id=getattr(resolved, "query_compatibility_id", None),
        role=role,
        expected_kind=expected_kind,
        expected_endpoint=expected_endpoint,
        fail=fail,
    )


def resolve_server_request_profile(
    base_url: str,
    requested_model: str,
    *,
    role: str,
    kind: ProfileKind,
    endpoint: str,
    profile_id: str | None = None,
    force_refresh: bool = False,
) -> EmbeddingRequestProfile:
    """Validiert einen direkten Warmup-/Health-Request gegen Discovery."""

    from kiara.retrieval.embedding_discovery import (
        discover_server_contracts,
        resolve_embedding_profile,
    )

    _validate_role(role)
    server = discover_server_contracts(base_url, force_refresh=force_refresh)
    resolution = resolve_embedding_profile(
        server,
        requested_model,
        endpoint=endpoint,
        profile_id=profile_id,
        kind=kind,
        role=role,
    )

    def fail(reason_code: str, detail: str) -> None:
        raise EmbeddingRoleContractError(
            reason_code,
            detail,
            requested_model=requested_model,
            endpoint=endpoint,
            input_type=role,
            profile_id=profile_id,
        )

    if not resolution.rag_verified or resolution.capabilities is None or resolution.profile is None:
        issue = resolution.issues[0] if resolution.issues else None
        fail(
            issue.reason_code if issue else "embedding_profile_unverified",
            issue.detail if issue else f"Vertragsstatus ist {resolution.state.value!r}",
        )
    profile = resolution.profile
    return _request_profile_from_parts(
        requested_model=requested_model,
        canonical_model_id=resolution.capabilities.canonical_model_id,
        profile_id=profile.profile_id,
        kind=profile.kind.value,
        endpoint=profile.endpoint,
        supported_roles=profile.input_type.supported,
        role_sensitive=profile.input_type.role_sensitive,
        formatting_owner=profile.formatting_owner,
        index_compatibility_id=profile.index_compatibility_id,
        query_compatibility_id=profile.query_compatibility_id,
        role=role,
        expected_kind=kind,
        expected_endpoint=endpoint,
        fail=fail,
    )


def backend_supports_embedding_operation(
    backend: dict[str, Any],
    *,
    role: str,
    kind: ProfileKind,
    endpoint: str,
    profile_id: str | None = None,
) -> tuple[bool, str]:
    """Nicht-werfende Admin-Diagnose fuer einen konkreten Backendpfad."""

    if str(backend.get("provider") or "") != "ollama":
        return False, "contract_not_applicable_non_kiron_backend"
    base_url = str(backend.get("url") or "").rstrip("/")
    requested_model = str(backend.get("model") or "")
    if not base_url or not requested_model:
        return False, "contract_backend_identity_missing"
    try:
        resolve_server_request_profile(
            base_url,
            requested_model,
            role=role,
            kind=kind,
            endpoint=endpoint,
            profile_id=profile_id,
        )
    except EmbeddingRoleContractError as exc:
        return False, exc.reason_code
    except Exception as exc:
        logger.debug(
            "Embedding-Profil fuer Admin-Diagnose nicht abrufbar: %s",
            exc.__class__.__name__,
        )
        return False, "embedding_contract_discovery_failed"
    return True, "embedding_profile_verified"


def emit_embedding_role_telemetry(
    profile: EmbeddingRequestProfile,
    *,
    operation: str,
    input_count: int,
) -> None:
    """Loggt nur Rollen-/Profilmetadaten, niemals fachliche Eingaben."""

    logger.info(
        "Embedding-Request operation=%s input_type=%s profile_id=%s "
        "canonical_model_id=%s kind=%s endpoint=%s compatibility_id=%s input_count=%d",
        operation,
        profile.input_type,
        profile.profile_id,
        profile.canonical_model_id,
        profile.kind,
        profile.endpoint,
        profile.active_compatibility_id,
        input_count,
    )
