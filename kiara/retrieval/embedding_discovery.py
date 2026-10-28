"""Central KIron embedding-contract discovery and profile resolution.

``/api/tags`` supplies canonical model groups.  Every V1 group is validated
independently, then checked against ``/api/show`` by its exact canonical name.
The resulting immutable snapshots are diagnostic/read-only in S3; existing
embedding request and index paths are deliberately not changed here.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from urllib.parse import urlsplit, urlunsplit

import httpx

from kiara.retrieval.embedding_contract import (
    CANONICAL_ROLES,
    SCHEMA_SHA256,
    SCHEMA_VERSION,
    ContractIssue,
    ContractSnapshot,
    ContractState,
    ContractValidationError,
    EmbeddingCapabilities,
    EmbeddingProfileContract,
    JSONDocumentError,
    ProfileKind,
    capabilities_equal,
    decode_json_document,
    parse_capabilities,
    snapshot_json_object,
)

logger = logging.getLogger(__name__)

DISCOVERY_CACHE_TTL_SECONDS = 30.0
DISCOVERY_TIMEOUT_SECONDS = 10.0
_MAX_DISCOVERY_RESPONSE_BYTES = 16 * 1024 * 1024
_SHOW_WORKERS = 4


@dataclass(frozen=True)
class ModelGroupDiscovery:
    displayed_name: str
    displayed_model: str
    reported_canonical_model_id: str | None
    state: ContractState
    capabilities: EmbeddingCapabilities | None
    raw_snapshot: ContractSnapshot | None
    show_checked: bool
    issues: tuple[ContractIssue, ...] = ()

    @property
    def identities(self) -> tuple[str, ...]:
        if self.capabilities is None:
            return ()
        return (self.capabilities.canonical_model_id, *self.capabilities.aliases)

    @property
    def discovery_consistent(self) -> bool:
        return self.state is ContractState.VALID and self.capabilities is not None

    def to_dict(self) -> dict:
        result: dict = {
            "displayed_identity": {
                "name": self.displayed_name,
                "model": self.displayed_model,
            },
            "reported_canonical_model_id": self.reported_canonical_model_id,
            "state": self.state.value,
            "show_checked": self.show_checked,
            "reason_codes": [issue.reason_code for issue in self.issues],
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if self.capabilities is not None:
            result.update(
                self.capabilities.to_diagnostic_dict(discovery_consistent=self.discovery_consistent)
            )
        elif self.raw_snapshot is not None:
            result["contract_snapshot"] = self.raw_snapshot.as_dict()
            result["contract_snapshot_sha256"] = self.raw_snapshot.canonical_sha256
        return result


@dataclass(frozen=True)
class ServerContractDiscovery:
    backend_url: str
    state: ContractState
    groups: tuple[ModelGroupDiscovery, ...]
    plain_model_names: tuple[str, ...]
    issues: tuple[ContractIssue, ...] = ()

    @property
    def cacheable(self) -> bool:
        return self.state is ContractState.VALID

    def matching_groups(self, requested_model: str) -> tuple[ModelGroupDiscovery, ...]:
        return tuple(group for group in self.groups if requested_model in group.identities)

    def to_dict(self) -> dict:
        return {
            "backend_url": self.backend_url,
            "state": self.state.value,
            "schema_version": SCHEMA_VERSION,
            "schema_sha256": SCHEMA_SHA256,
            "group_count": len(self.groups),
            "plain_model_count": len(self.plain_model_names),
            "reason_codes": [issue.reason_code for issue in self.issues],
            "issues": [issue.to_dict() for issue in self.issues],
            "groups": [group.to_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class ProfileResolution:
    requested_model: str
    requested_endpoint: str
    requested_profile_id: str | None
    state: ContractState
    displayed_name: str | None = None
    displayed_model: str | None = None
    capabilities: EmbeddingCapabilities | None = None
    profile: EmbeddingProfileContract | None = None
    issues: tuple[ContractIssue, ...] = ()

    @property
    def rag_verified(self) -> bool:
        return self.state is ContractState.VALID and bool(
            self.profile and self.profile.rag_verified
        )

    def to_dict(self) -> dict:
        result: dict = {
            "requested_identity": {
                "model": self.requested_model,
                "endpoint": self.requested_endpoint,
                "profile_id": self.requested_profile_id,
            },
            "displayed_identity": {
                "name": self.displayed_name,
                "model": self.displayed_model,
            },
            "state": self.state.value,
            "rag_verified": self.rag_verified,
            "reason_codes": [issue.reason_code for issue in self.issues],
            "issues": [issue.to_dict() for issue in self.issues],
            "canonical_model_id": (
                self.capabilities.canonical_model_id if self.capabilities else None
            ),
            "profile": (
                self.profile.to_diagnostic_dict(
                    discovery_consistent=self.state is ContractState.VALID
                )
                if self.profile
                else None
            ),
        }
        if self.capabilities is not None:
            result["contract_snapshot"] = self.capabilities.snapshot.as_dict()
            result["contract_snapshot_sha256"] = self.capabilities.snapshot.canonical_sha256
        return result


@dataclass(frozen=True)
class BackendContractDiagnostic:
    backend_id: int | None
    backend_name: str
    provider: str
    requested_model: str
    backend_url: str
    state: ContractState
    resolution: ProfileResolution | None = None
    issues: tuple[ContractIssue, ...] = ()

    @property
    def rag_verified(self) -> bool:
        return self.state is ContractState.VALID and bool(
            self.resolution and self.resolution.rag_verified
        )

    def to_dict(self) -> dict:
        return {
            "backend_id": self.backend_id,
            "backend_name": self.backend_name,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "backend_url": self.backend_url,
            "state": self.state.value,
            "rag_verified": self.rag_verified,
            "reason_codes": [issue.reason_code for issue in self.issues],
            "issues": [issue.to_dict() for issue in self.issues],
            "resolution": self.resolution.to_dict() if self.resolution else None,
        }


@dataclass(frozen=True)
class EmbeddingContractInventory:
    servers: tuple[ServerContractDiscovery, ...]
    backends: tuple[BackendContractDiagnostic, ...]

    def to_dict(self) -> dict:
        return {
            "contract": {
                "schema_version": SCHEMA_VERSION,
                "schema_sha256": SCHEMA_SHA256,
                "source": "KIron /api/tags + /api/show",
            },
            "servers": [server.to_dict() for server in self.servers],
            "backends": [backend.to_dict() for backend in self.backends],
        }


@dataclass
class _DiscoveryFlight:
    epoch: int
    event: threading.Event
    result: ServerContractDiscovery | None = None


_cache_lock = threading.Lock()
_cache_epoch = 0
_server_cache: dict[str, tuple[float, ServerContractDiscovery]] = {}
_server_flights: dict[str, _DiscoveryFlight] = {}


def _safe_url(value: str) -> str:
    """Remove userinfo, query and fragment from diagnostic URL output."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<invalid-url>"
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "<invalid-url>"
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "<invalid-url>"
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, host + port, path, "", ""))


def _cache_key(value: str) -> str:
    return value.strip().rstrip("/")


def invalidate_embedding_contract_cache(backend_url: str | None = None) -> None:
    """Invalidate discovery snapshots after embedding-backend CRUD."""

    global _cache_epoch
    with _cache_lock:
        _cache_epoch += 1
        if backend_url is None:
            _server_cache.clear()
        else:
            _server_cache.pop(_cache_key(backend_url), None)


def _endpoint_url(base_url: str, endpoint: str) -> str:
    return _cache_key(base_url) + endpoint


def _request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    json_body: Mapping[str, object] | None = None,
) -> object:
    request_kwargs = {"json": json_body} if json_body is not None else {}
    response = client.request(method, url, **request_kwargs)
    if response.status_code < 200 or response.status_code >= 300:
        raise JSONDocumentError(
            "contract_discovery_http_error",
            f"HTTP {response.status_code}",
        )
    if len(response.content) > _MAX_DISCOVERY_RESPONSE_BYTES:
        raise JSONDocumentError(
            "contract_discovery_response_too_large",
            f"response exceeds {_MAX_DISCOVERY_RESPONSE_BYTES} bytes",
        )
    return decode_json_document(response.content)


def _snapshot_if_possible(raw: object) -> ContractSnapshot | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        return snapshot_json_object(raw)
    except ContractValidationError:
        return None


def _state_for_validation_error(exc: ContractValidationError) -> ContractState:
    if exc.issue.reason_code == "contract_unknown_schema_version":
        return ContractState.UNKNOWN_SCHEMA_VERSION
    return ContractState.MALFORMED


def _tags_group(row: object, index: int) -> ModelGroupDiscovery | None:
    if not isinstance(row, Mapping):
        issue = ContractIssue(
            "contract_tags_model_malformed",
            f"/models/{index}",
            "model entry is not an object",
        )
        return ModelGroupDiscovery(
            displayed_name="",
            displayed_model="",
            reported_canonical_model_id=None,
            state=ContractState.MALFORMED,
            capabilities=None,
            raw_snapshot=None,
            show_checked=False,
            issues=(issue,),
        )
    if "kiron_capabilities" not in row:
        return None

    displayed_name = row.get("name") if isinstance(row.get("name"), str) else ""
    displayed_model = row.get("model") if isinstance(row.get("model"), str) else ""
    raw = row.get("kiron_capabilities")
    reported = raw.get("canonical_model_id") if isinstance(raw, Mapping) else None
    if not isinstance(reported, str):
        reported = None
    try:
        capabilities = parse_capabilities(raw)
    except ContractValidationError as exc:
        return ModelGroupDiscovery(
            displayed_name=displayed_name,
            displayed_model=displayed_model,
            reported_canonical_model_id=reported,
            state=_state_for_validation_error(exc),
            capabilities=None,
            raw_snapshot=_snapshot_if_possible(raw),
            show_checked=False,
            issues=(exc.issue,),
        )

    return ModelGroupDiscovery(
        displayed_name=displayed_name,
        displayed_model=displayed_model,
        reported_canonical_model_id=capabilities.canonical_model_id,
        state=ContractState.VALID,
        capabilities=capabilities,
        raw_snapshot=capabilities.snapshot,
        show_checked=False,
    )


def _check_show(
    client: httpx.Client,
    base_url: str,
    group: ModelGroupDiscovery,
) -> ModelGroupDiscovery:
    capabilities = group.capabilities
    if capabilities is None:
        return group
    canonical = capabilities.canonical_model_id
    try:
        document = _request_json(
            client,
            "POST",
            _endpoint_url(base_url, "/api/show"),
            json_body={"model": canonical},
        )
    except (httpx.RequestError, JSONDocumentError) as exc:
        reason = (
            exc.reason_code
            if isinstance(exc, JSONDocumentError)
            else "contract_discovery_request_failed"
        )
        detail = exc.detail if isinstance(exc, JSONDocumentError) else type(exc).__name__
        issue = ContractIssue(
            reason,
            "/api/show",
            detail,
            canonical_model_id=canonical,
        )
        return replace(
            group,
            state=ContractState.INCONSISTENT,
            show_checked=True,
            issues=(*group.issues, issue),
        )

    if not isinstance(document, Mapping):
        issue = ContractIssue(
            "contract_show_response_malformed",
            "/api/show",
            "response root is not an object",
            canonical_model_id=canonical,
        )
        return replace(
            group,
            state=ContractState.INCONSISTENT,
            show_checked=True,
            issues=(*group.issues, issue),
        )
    if "kiron_capabilities" not in document:
        issue = ContractIssue(
            "contract_show_missing",
            "/api/show/kiron_capabilities",
            "show response has no capability object",
            canonical_model_id=canonical,
        )
        return replace(
            group,
            state=ContractState.INCONSISTENT,
            show_checked=True,
            issues=(*group.issues, issue),
        )
    try:
        shown = parse_capabilities(document["kiron_capabilities"])
    except ContractValidationError as exc:
        issue = replace(exc.issue, canonical_model_id=canonical)
        return replace(
            group,
            state=ContractState.INCONSISTENT,
            show_checked=True,
            issues=(*group.issues, issue),
        )
    if not capabilities_equal(capabilities, shown):
        issue = ContractIssue(
            "contract_tags_show_mismatch",
            "/kiron_capabilities",
            "complete JCS snapshots differ",
            canonical_model_id=canonical,
        )
        return replace(
            group,
            state=ContractState.INCONSISTENT,
            show_checked=True,
            issues=(*group.issues, issue),
        )
    return replace(group, show_checked=True)


def _apply_registry_identity_checks(
    groups: Sequence[ModelGroupDiscovery],
) -> tuple[tuple[ModelGroupDiscovery, ...], tuple[ContractIssue, ...]]:
    identity_owners: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        for identity in group.identities:
            identity_owners.setdefault(identity, []).append(index)

    affected: dict[int, list[ContractIssue]] = {}
    server_issues: list[ContractIssue] = []
    for identity, owners in sorted(identity_owners.items()):
        unique_owners = sorted(set(owners))
        if len(unique_owners) <= 1:
            continue
        issue = ContractIssue(
            "contract_alias_ambiguous",
            "/kiron_capabilities/aliases",
            f"identity {identity!r} resolves to multiple groups",
        )
        server_issues.append(issue)
        for owner in unique_owners:
            affected.setdefault(owner, []).append(issue)

    checked: list[ModelGroupDiscovery] = []
    for index, group in enumerate(groups):
        issues = affected.get(index)
        if issues:
            checked.append(
                replace(
                    group,
                    state=ContractState.AMBIGUOUS,
                    issues=(*group.issues, *issues),
                )
            )
        else:
            checked.append(group)
    return tuple(checked), tuple(server_issues)


def _server_state(groups: Sequence[ModelGroupDiscovery]) -> ContractState:
    if not groups:
        return ContractState.MISSING
    states = {group.state for group in groups}
    if ContractState.MALFORMED in states:
        return ContractState.MALFORMED
    if ContractState.UNKNOWN_SCHEMA_VERSION in states:
        return ContractState.UNKNOWN_SCHEMA_VERSION
    if ContractState.INCONSISTENT in states:
        return ContractState.INCONSISTENT
    if ContractState.AMBIGUOUS in states:
        return ContractState.AMBIGUOUS
    return ContractState.VALID


def _discovery_error(base_url: str, reason_code: str, detail: str) -> ServerContractDiscovery:
    issue = ContractIssue(reason_code, "/api/tags", detail)
    return ServerContractDiscovery(
        backend_url=_safe_url(base_url),
        state=ContractState.DISCOVERY_ERROR,
        groups=(),
        plain_model_names=(),
        issues=(issue,),
    )


def _discover_server_uncached(
    base_url: str,
    timeout: float,
    transport: httpx.BaseTransport | None = None,
) -> ServerContractDiscovery:
    safe_url = _safe_url(base_url)
    if safe_url == "<invalid-url>":
        return _discovery_error(base_url, "contract_backend_url_invalid", "invalid URL")

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as client:
            tags = _request_json(client, "GET", _endpoint_url(base_url, "/api/tags"))
            if not isinstance(tags, Mapping) or not isinstance(tags.get("models"), list):
                return _discovery_error(
                    base_url,
                    "contract_tags_response_malformed",
                    "response must contain a models array",
                )

            groups: list[ModelGroupDiscovery] = []
            plain_names: list[str] = []
            for index, row in enumerate(tags["models"]):
                group = _tags_group(row, index)
                if group is not None:
                    groups.append(group)
                    continue
                if isinstance(row, Mapping):
                    for field in ("name", "model"):
                        candidate = row.get(field)
                        if isinstance(candidate, str) and candidate:
                            plain_names.append(candidate)

            valid_indices = [
                index for index, group in enumerate(groups) if group.capabilities is not None
            ]
            if valid_indices:
                workers = min(_SHOW_WORKERS, len(valid_indices))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_check_show, client, base_url, groups[index]): index
                        for index in valid_indices
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            groups[index] = future.result()
                        except Exception as exc:
                            logger.warning(
                                "Embedding-Vertragsdiscovery /api/show fehlgeschlagen (%s)",
                                type(exc).__name__,
                            )
                            issue = ContractIssue(
                                "contract_discovery_internal_error",
                                "/api/show",
                                type(exc).__name__,
                                canonical_model_id=groups[index].reported_canonical_model_id,
                            )
                            groups[index] = replace(
                                groups[index],
                                state=ContractState.INCONSISTENT,
                                show_checked=True,
                                issues=(*groups[index].issues, issue),
                            )
    except httpx.RequestError as exc:
        return _discovery_error(
            base_url,
            "contract_discovery_request_failed",
            type(exc).__name__,
        )
    except JSONDocumentError as exc:
        return _discovery_error(base_url, exc.reason_code, exc.detail)
    except Exception as exc:
        logger.warning(
            "Embedding-Vertragsdiscovery unerwartet fehlgeschlagen (%s)",
            type(exc).__name__,
        )
        return _discovery_error(
            base_url,
            "contract_discovery_internal_error",
            type(exc).__name__,
        )

    groups_checked, registry_issues = _apply_registry_identity_checks(groups)
    groups_checked = tuple(
        sorted(
            groups_checked,
            key=lambda group: (
                group.reported_canonical_model_id or "",
                group.displayed_name,
            ),
        )
    )
    group_issues = tuple(issue for group in groups_checked for issue in group.issues)
    return ServerContractDiscovery(
        backend_url=safe_url,
        state=_server_state(groups_checked),
        groups=groups_checked,
        plain_model_names=tuple(sorted(set(plain_names))),
        issues=(*registry_issues, *group_issues),
    )


def discover_server_contracts(
    base_url: str,
    *,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
    force_refresh: bool = False,
    transport: httpx.BaseTransport | None = None,
) -> ServerContractDiscovery:
    """Discover one backend with TTL cache and per-URL singleflight.

    A supplied transport is a test seam and intentionally bypasses the shared
    cache.  Only a completely valid server inventory is cached; malformed or
    unavailable responses are always returned as explicit diagnostics.
    """

    if transport is not None:
        return _discover_server_uncached(base_url, timeout, transport)
    key = _cache_key(base_url)
    if force_refresh:
        invalidate_embedding_contract_cache(base_url)

    while True:
        now = time.monotonic()
        with _cache_lock:
            epoch = _cache_epoch
            cached = _server_cache.get(key)
            if cached and cached[0] > now:
                return cached[1]
            if cached:
                _server_cache.pop(key, None)

            flight = _server_flights.get(key)
            if flight is None or flight.epoch != epoch:
                flight = _DiscoveryFlight(epoch=epoch, event=threading.Event())
                _server_flights[key] = flight
                producer = True
            else:
                producer = False

        if not producer:
            flight.event.wait()
            with _cache_lock:
                if flight.epoch != _cache_epoch:
                    continue
                if flight.result is not None:
                    return flight.result
            continue

        result = _discover_server_uncached(base_url, timeout)
        with _cache_lock:
            current_epoch = _cache_epoch
            current_flight = _server_flights.get(key)
            if current_flight is flight:
                _server_flights.pop(key, None)
            if current_epoch == flight.epoch:
                flight.result = result
                if result.cacheable:
                    _server_cache[key] = (
                        time.monotonic() + DISCOVERY_CACHE_TTL_SECONDS,
                        result,
                    )
            flight.event.set()
            stale = current_epoch != flight.epoch
        if stale:
            continue
        return result


def resolve_embedding_profile(
    server: ServerContractDiscovery,
    requested_model: str,
    *,
    endpoint: str,
    profile_id: str | None = None,
    kind: ProfileKind | None = None,
    role: str | None = None,
) -> ProfileResolution:
    """Resolve canonical group, endpoint and exact/default profile fail-closed."""

    base = {
        "requested_model": requested_model,
        "requested_endpoint": endpoint,
        "requested_profile_id": profile_id,
    }
    if server.state not in (ContractState.VALID, ContractState.MISSING):
        issue = ContractIssue(
            "contract_server_invalid",
            "/api/tags",
            f"server discovery state is {server.state.value}",
        )
        return ProfileResolution(
            **base,
            state=server.state,
            issues=(*server.issues, issue),
        )

    matches = server.matching_groups(requested_model)
    if len(matches) > 1:
        issue = ContractIssue(
            "contract_alias_ambiguous",
            "/kiron_capabilities/aliases",
            f"model reference {requested_model!r} is ambiguous",
        )
        return ProfileResolution(**base, state=ContractState.AMBIGUOUS, issues=(issue,))
    if not matches:
        exact_plain = requested_model in server.plain_model_names
        issue = ContractIssue(
            "contract_missing" if exact_plain else "contract_model_not_found",
            "/kiron_capabilities",
            (
                "matching model row has no KIron capability contract"
                if exact_plain
                else "no exact canonical model or registered alias match"
            ),
        )
        return ProfileResolution(**base, state=ContractState.MISSING, issues=(issue,))

    group = matches[0]
    capabilities = group.capabilities
    if not group.discovery_consistent or capabilities is None:
        return ProfileResolution(
            **base,
            state=group.state,
            displayed_name=group.displayed_name,
            displayed_model=group.displayed_model,
            capabilities=capabilities,
            issues=group.issues,
        )

    if profile_id is not None:
        profile = capabilities.profile_by_id(profile_id)
        if profile is None:
            issue = ContractIssue(
                "contract_profile_not_found",
                "/kiron_capabilities/profiles",
                f"profile {profile_id!r} does not exist",
                canonical_model_id=capabilities.canonical_model_id,
                profile_id=profile_id,
            )
            return ProfileResolution(
                **base,
                state=ContractState.MISSING,
                displayed_name=group.displayed_name,
                displayed_model=group.displayed_model,
                capabilities=capabilities,
                issues=(issue,),
            )
    else:
        profile = capabilities.default_profile(endpoint)
        if profile is None:
            issue = ContractIssue(
                "contract_endpoint_not_available",
                "/kiron_capabilities/profiles",
                f"endpoint {endpoint!r} has no default profile",
                canonical_model_id=capabilities.canonical_model_id,
            )
            return ProfileResolution(
                **base,
                state=ContractState.MISSING,
                displayed_name=group.displayed_name,
                displayed_model=group.displayed_model,
                capabilities=capabilities,
                issues=(issue,),
            )

    issues: list[ContractIssue] = []
    if profile.endpoint != endpoint:
        issues.append(
            ContractIssue(
                "contract_profile_endpoint_mismatch",
                "/kiron_capabilities/profiles/endpoint",
                f"profile endpoint {profile.endpoint!r} differs from {endpoint!r}",
                canonical_model_id=capabilities.canonical_model_id,
                profile_id=profile.profile_id,
            )
        )
    if kind is not None and profile.kind is not kind:
        issues.append(
            ContractIssue(
                "contract_profile_kind_mismatch",
                "/kiron_capabilities/profiles/kind",
                f"profile kind {profile.kind.value!r} differs from {kind.value!r}",
                canonical_model_id=capabilities.canonical_model_id,
                profile_id=profile.profile_id,
            )
        )
    if role is not None and (
        role not in CANONICAL_ROLES or role not in profile.input_type.supported
    ):
        issues.append(
            ContractIssue(
                "contract_role_unsupported",
                "/kiron_capabilities/profiles/input_type/supported",
                f"role {role!r} is not supported",
                canonical_model_id=capabilities.canonical_model_id,
                profile_id=profile.profile_id,
            )
        )
    if issues:
        state = ContractState.INCONSISTENT
    elif not profile.rag_verified:
        state = ContractState.UNVERIFIED
        issues.append(
            ContractIssue(
                "profile_unverified",
                "/kiron_capabilities/profiles/verification",
                "profile is visible but not eligible for verified RAG selection",
                canonical_model_id=capabilities.canonical_model_id,
                profile_id=profile.profile_id,
            )
        )
    else:
        state = ContractState.VALID

    return ProfileResolution(
        **base,
        state=state,
        displayed_name=group.displayed_name,
        displayed_model=group.displayed_model,
        capabilities=capabilities,
        profile=profile,
        issues=tuple(issues),
    )


def select_configured_embedding_profile_id(
    backend: Mapping[str, object],
) -> tuple[str | None, tuple[ContractIssue, ...]]:
    """Validiert und waehlt die explizite Profil-ID eines Backends.

    Dies ist die einzige Auswahlquelle fuer S3-Discovery und S4-Resolver.
    Nur ``None`` und ein leeres Mapping bedeuten, dass kein Profil explizit
    konfiguriert ist. Jeder andere malformed/mehrdeutige Wert bleibt
    fail-closed und darf nicht auf das Defaultprofil zurueckfallen.
    """
    raw_extra = backend.get("extra_config")
    if raw_extra is None or (isinstance(raw_extra, Mapping) and not raw_extra):
        return None, ()
    if isinstance(raw_extra, str):
        try:
            extra = json.loads(raw_extra)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, (
                ContractIssue(
                    "backend_extra_config_malformed",
                    "/ki_backends/extra_config",
                    "extra_config is not valid JSON",
                ),
            )
    elif isinstance(raw_extra, Mapping):
        extra = raw_extra
    else:
        return None, (
            ContractIssue(
                "backend_extra_config_malformed",
                "/ki_backends/extra_config",
                "extra_config is not an object",
            ),
        )

    if not isinstance(extra, Mapping):
        return None, (
            ContractIssue(
                "backend_extra_config_malformed",
                "/ki_backends/extra_config",
                "extra_config is not an object",
            ),
        )

    configured: list[tuple[str, str]] = []
    issues: list[ContractIssue] = []
    for key in ("embedding_profile_id", "profile_id"):
        if key not in extra:
            continue
        value = extra[key]
        if isinstance(value, str) and value and value == value.strip():
            configured.append((key, value))
        else:
            issues.append(
                ContractIssue(
                    "backend_profile_id_malformed",
                    f"/ki_backends/extra_config/{key}",
                    "configured profile ID must be a non-empty string",
                )
            )

    if issues:
        return None, tuple(issues)
    values = {value for _, value in configured}
    if len(values) > 1:
        return None, (
            ContractIssue(
                "backend_profile_id_ambiguous",
                "/ki_backends/extra_config",
                "embedding_profile_id and profile_id disagree",
            ),
        )
    return (configured[0][1] if configured else None), ()


def discover_configured_embedding_backends(
    backends: Sequence[Mapping[str, object]],
    *,
    timeout: float = DISCOVERY_TIMEOUT_SECONDS,
) -> EmbeddingContractInventory:
    """Build read-only diagnostics for all active embedding backends."""

    ollama_urls = sorted(
        {
            str(backend.get("url") or "")
            for backend in backends
            if backend.get("provider") == "ollama" and backend.get("url")
        }
    )
    discovered: dict[str, ServerContractDiscovery] = {}
    if ollama_urls:
        workers = min(_SHOW_WORKERS, len(ollama_urls))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(discover_server_contracts, url, timeout=timeout): url
                for url in ollama_urls
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    discovered[url] = future.result()
                except Exception as exc:
                    logger.warning(
                        "Embedding-Vertragsinventar fehlgeschlagen (%s)",
                        type(exc).__name__,
                    )
                    discovered[url] = _discovery_error(
                        url,
                        "contract_discovery_internal_error",
                        type(exc).__name__,
                    )

    diagnostics: list[BackendContractDiagnostic] = []
    ordered_backends = sorted(
        backends,
        key=lambda item: (
            int(item.get("priority") or 0),
            int(item.get("id") or 0),
        ),
    )
    for backend in ordered_backends:
        provider = str(backend.get("provider") or "")
        raw_url = str(backend.get("url") or "")
        safe_url = _safe_url(raw_url) if raw_url else ""
        common = {
            "backend_id": (int(backend["id"]) if isinstance(backend.get("id"), int) else None),
            "backend_name": str(backend.get("name") or ""),
            "provider": provider,
            "requested_model": str(backend.get("model") or ""),
            "backend_url": safe_url,
        }
        if provider != "ollama":
            issue = ContractIssue(
                "contract_not_applicable_non_kiron_backend",
                "/ki_backends/provider",
                f"provider {provider!r} does not expose KIron discovery",
            )
            diagnostics.append(
                BackendContractDiagnostic(
                    **common,
                    state=ContractState.NOT_APPLICABLE,
                    issues=(issue,),
                )
            )
            continue
        if not raw_url or raw_url not in discovered:
            issue = ContractIssue(
                "contract_backend_url_missing",
                "/ki_backends/url",
                "Ollama/KIron backend has no usable URL",
            )
            diagnostics.append(
                BackendContractDiagnostic(
                    **common,
                    state=ContractState.MISSING,
                    issues=(issue,),
                )
            )
            continue

        profile_id, config_issues = select_configured_embedding_profile_id(backend)
        if config_issues:
            diagnostics.append(
                BackendContractDiagnostic(
                    **common,
                    state=ContractState.INCONSISTENT,
                    issues=config_issues,
                )
            )
            continue

        resolution = resolve_embedding_profile(
            discovered[raw_url],
            common["requested_model"],
            endpoint="/api/embed",
            profile_id=profile_id,
            kind=ProfileKind.DENSE,
        )
        diagnostics.append(
            BackendContractDiagnostic(
                **common,
                state=resolution.state,
                resolution=resolution,
            )
        )

    return EmbeddingContractInventory(
        servers=tuple(discovered[url] for url in ollama_urls),
        backends=tuple(diagnostics),
    )
