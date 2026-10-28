"""Typed consumer for KIron embedding capability contract V1.

The wire shape is validated against the byte-identical S1 JSON Schema.  The
schema is intentionally combined with ADR-0009 behavioural checks and RFC 8785
canonicalisation: schema-valid data alone is not sufficient for a verified
profile.

This module only models and validates discovery data.  It does not select a
production embedding backend, alter request payloads or persist contract data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_VERSION = 1
SCHEMA_FILENAME = "kiron-embedding-capabilities-v1.schema.json"
SCHEMA_SHA256 = "d4d2698831e16bafe54dc03ffcb5b927933b9439111242e0d5997f0815dea477"
CANONICAL_ROLES = ("search_document", "search_query")
I_JSON_SAFE_INTEGER = 9_007_199_254_740_991

_DOMAIN_FOR_ROLE = {
    "search_document": "kiron.embedding.index_compatibility_id",
    "search_query": "kiron.embedding.query_compatibility_id",
}
_ENDPOINT_FOR_KIND = {
    "dense": "/api/embed",
    "late_chunking": "/api/embed_late",
    "multi_vector": "/api/embed_colbert",
}
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContractState(str, Enum):
    """Stable high-level discovery/validation states."""

    VALID = "valid"
    UNVERIFIED = "unverified"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN_SCHEMA_VERSION = "unknown_schema_version"
    MALFORMED = "malformed"
    INCONSISTENT = "inconsistent"
    AMBIGUOUS = "ambiguous"
    DISCOVERY_ERROR = "discovery_error"


class ProfileKind(str, Enum):
    DENSE = "dense"
    LATE_CHUNKING = "late_chunking"
    MULTI_VECTOR = "multi_vector"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ContractIssue:
    """Machine-readable, non-secret contract diagnostic."""

    reason_code: str
    field_path: str
    detail: str
    canonical_model_id: str | None = None
    profile_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "reason_code": self.reason_code,
            "field_path": self.field_path,
            "detail": self.detail,
            "canonical_model_id": self.canonical_model_id,
            "profile_id": self.profile_id,
        }


class ContractValidationError(ValueError):
    """Controlled fail-closed error for one capability snapshot."""

    def __init__(
        self,
        reason_code: str,
        field_path: str,
        detail: str,
        *,
        canonical_model_id: str | None = None,
        profile_id: str | None = None,
    ) -> None:
        super().__init__(f"{reason_code} at {field_path}: {detail}")
        self.issue = ContractIssue(
            reason_code=reason_code,
            field_path=field_path,
            detail=detail,
            canonical_model_id=canonical_model_id,
            profile_id=profile_id,
        )


class JSONDocumentError(ValueError):
    """Malformed JSON response before capability validation."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


class JCSCanonicalizationError(ValueError):
    """Value cannot be represented in the normative JCS domain."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class ContractSnapshot:
    """Detached immutable representation retaining all extension fields."""

    json_text: str
    canonical_json: str
    canonical_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return a fresh mutable copy; callers cannot mutate validated state."""

        value = json.loads(self.json_text)
        if not isinstance(value, dict):
            raise AssertionError("Capability snapshot root must remain an object")
        return value


@dataclass(frozen=True)
class DigestFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class BackendIdentity:
    backend_type: str
    implementation: str
    implementation_revision: str | None


@dataclass(frozen=True)
class ArtifactIdentity:
    repository: str | None
    revision: str | None
    manifest_digest: str | None
    weights: tuple[DigestFile, ...]
    auxiliary: tuple[DigestFile, ...]


@dataclass(frozen=True)
class RoleLimits:
    search_document: int | None
    search_query: int | None


@dataclass(frozen=True)
class MaxInputTokens:
    unit: str
    counting: str
    truncation: str
    overflow: str
    by_role: RoleLimits


@dataclass(frozen=True)
class InputTypeAlias:
    alias: str
    canonical: str


@dataclass(frozen=True)
class InputTypeContract:
    supported: tuple[str, ...]
    additional_task_roles: tuple[str, ...]
    aliases: tuple[InputTypeAlias, ...]
    role_sensitive: bool | None
    required: bool
    default: str | None
    missing_role_behavior: str


@dataclass(frozen=True)
class FormattingContract:
    template: str | None
    parameters: ContractSnapshot


@dataclass(frozen=True)
class TokenizerContract:
    repository: str | None
    revision: str | None


@dataclass(frozen=True)
class PoolingContract:
    method: str
    parameters: ContractSnapshot


@dataclass(frozen=True)
class NormalizationContract:
    method: str
    dtype: str
    scope: str


@dataclass(frozen=True)
class PipelineContract:
    tokenizer: TokenizerContract
    search_document_formatting: FormattingContract
    search_query_formatting: FormattingContract
    pooling: PoolingContract
    normalization: NormalizationContract
    parameters: ContractSnapshot


@dataclass(frozen=True)
class VerificationContract:
    status: VerificationStatus
    evidence: tuple[str, ...]
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True)
class EmbeddingProfileContract:
    profile_id: str
    kind: ProfileKind
    endpoint: str
    default_for_endpoint: bool
    backend: BackendIdentity
    artifact: ArtifactIdentity
    dimensions: int | None
    max_input_tokens: MaxInputTokens
    output_normalized: bool | None
    similarity: str
    input_type: InputTypeContract
    formatting_owner: str
    pipeline: PipelineContract
    verification: VerificationContract
    index_compatibility_id: str | None
    query_compatibility_id: str | None
    snapshot: ContractSnapshot

    @property
    def rag_verified(self) -> bool:
        """Whether this V1 profile is eligible in the contract view."""

        return (
            self.verification.status is VerificationStatus.VERIFIED
            and self.index_compatibility_id is not None
            and self.query_compatibility_id is not None
            and self.endpoint == _ENDPOINT_FOR_KIND[self.kind.value]
            and self.formatting_owner == "server"
            and self.input_type.supported == CANONICAL_ROLES
        )

    def to_diagnostic_dict(self, *, discovery_consistent: bool = True) -> dict[str, Any]:
        rag_verified = self.rag_verified and discovery_consistent
        reason_codes: list[str] = []
        if not discovery_consistent:
            reason_codes.append("contract_discovery_inconsistent")
        if self.verification.status is VerificationStatus.UNVERIFIED:
            reason_codes.append("profile_unverified")
        return {
            "profile_id": self.profile_id,
            "kind": self.kind.value,
            "endpoint": self.endpoint,
            "default_for_endpoint": self.default_for_endpoint,
            "backend": {
                "type": self.backend.backend_type,
                "implementation": self.backend.implementation,
                "implementation_revision": self.backend.implementation_revision,
            },
            "artifact": {
                "repository": self.artifact.repository,
                "revision": self.artifact.revision,
                "manifest_digest": self.artifact.manifest_digest,
                "weights": [
                    {"path": item.path, "sha256": item.sha256} for item in self.artifact.weights
                ],
                "auxiliary": [
                    {"path": item.path, "sha256": item.sha256} for item in self.artifact.auxiliary
                ],
            },
            "dimensions": self.dimensions,
            "max_input_tokens": {
                "unit": self.max_input_tokens.unit,
                "counting": self.max_input_tokens.counting,
                "truncation": self.max_input_tokens.truncation,
                "overflow": self.max_input_tokens.overflow,
                "by_role": {
                    "search_document": self.max_input_tokens.by_role.search_document,
                    "search_query": self.max_input_tokens.by_role.search_query,
                },
            },
            "output_normalized": self.output_normalized,
            "similarity": self.similarity,
            "input_type": {
                "supported": list(self.input_type.supported),
                "additional_task_roles": list(self.input_type.additional_task_roles),
                "aliases": [
                    {"alias": item.alias, "canonical": item.canonical}
                    for item in self.input_type.aliases
                ],
                "role_sensitive": self.input_type.role_sensitive,
                "required": self.input_type.required,
                "default": self.input_type.default,
                "missing_role_behavior": self.input_type.missing_role_behavior,
            },
            "formatting_owner": self.formatting_owner,
            "verification": {
                "status": self.verification.status.value,
                "evidence": list(self.verification.evidence),
                "blocking_reasons": list(self.verification.blocking_reasons),
            },
            "index_compatibility_id": self.index_compatibility_id,
            "query_compatibility_id": self.query_compatibility_id,
            "rag_verified": rag_verified,
            "reason_codes": reason_codes,
            "contract_snapshot": self.snapshot.as_dict(),
        }


@dataclass(frozen=True)
class EmbeddingCapabilities:
    schema_version: int
    canonical_model_id: str
    aliases: tuple[str, ...]
    profiles: tuple[EmbeddingProfileContract, ...]
    snapshot: ContractSnapshot

    def profile_by_id(self, profile_id: str) -> EmbeddingProfileContract | None:
        return next((item for item in self.profiles if item.profile_id == profile_id), None)

    def default_profile(self, endpoint: str) -> EmbeddingProfileContract | None:
        matches = [
            item
            for item in self.profiles
            if item.endpoint == endpoint and item.default_for_endpoint
        ]
        return matches[0] if len(matches) == 1 else None

    def to_diagnostic_dict(self, *, discovery_consistent: bool = True) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "canonical_model_id": self.canonical_model_id,
            "aliases": list(self.aliases),
            "profiles": [
                item.to_diagnostic_dict(discovery_consistent=discovery_consistent)
                for item in self.profiles
            ],
            "contract_snapshot": self.snapshot.as_dict(),
            "contract_snapshot_sha256": self.snapshot.canonical_sha256,
        }


def _raise_contract(reason_code: str, field_path: str, detail: str) -> None:
    raise ContractValidationError(reason_code, field_path, detail)


def _assert_valid_unicode_string(value: str) -> None:
    for code_point in map(ord, value):
        if 0xD800 <= code_point <= 0xDFFF:
            raise JCSCanonicalizationError(
                "jcs_invalid_unicode_scalar",
                f"lone surrogate U+{code_point:04X}",
            )


def _assert_jcs_value(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JCSCanonicalizationError(
                    "jcs_non_string_object_name",
                    f"object name has type {type(key).__name__}",
                )
            _assert_valid_unicode_string(key)
            _assert_jcs_value(item)
    elif isinstance(value, list):
        for item in value:
            _assert_jcs_value(item)
    elif isinstance(value, str):
        _assert_valid_unicode_string(value)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise JCSCanonicalizationError("jcs_non_finite_number", repr(value))
    else:
        raise JCSCanonicalizationError("jcs_value_type_forbidden", type(value).__name__)


def _assert_jcs_hash_value(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JCSCanonicalizationError(
                    "jcs_non_string_object_name",
                    f"object name has type {type(key).__name__}",
                )
            _assert_valid_unicode_string(key)
            _assert_jcs_hash_value(item)
    elif isinstance(value, list):
        for item in value:
            _assert_jcs_hash_value(item)
    elif isinstance(value, str):
        _assert_valid_unicode_string(value)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        if abs(value) > I_JSON_SAFE_INTEGER:
            raise JCSCanonicalizationError("jcs_integer_outside_i_json_safe_range", str(value))
    else:
        raise JCSCanonicalizationError("jcs_v1_value_type_forbidden", type(value).__name__)


def _utf16_sort_key(value: str) -> bytes:
    _assert_valid_unicode_string(value)
    return value.encode("utf-16-be")


def _jcs_quote(value: str) -> str:
    escapes = {
        0x08: "\\b",
        0x09: "\\t",
        0x0A: "\\n",
        0x0C: "\\f",
        0x0D: "\\r",
    }
    serialized: list[str] = []
    for character in value:
        code_point = ord(character)
        if code_point in escapes:
            serialized.append(escapes[code_point])
        elif code_point <= 0x1F:
            serialized.append(f"\\u{code_point:04x}")
        elif character == '"':
            serialized.append('\\"')
        elif character == "\\":
            serialized.append("\\\\")
        else:
            serialized.append(character)
    return '"' + "".join(serialized) + '"'


def _serialize_jcs_number(value: int | float) -> str:
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise JCSCanonicalizationError("jcs_non_finite_number", repr(value)) from exc
    if not math.isfinite(number):
        raise JCSCanonicalizationError("jcs_non_finite_number", repr(value))
    if number == 0:
        return "0"

    rendered = repr(number).lower()
    sign = ""
    if rendered.startswith("-"):
        sign = "-"
        rendered = rendered[1:]

    exponent_text = ""
    exponent = 0
    if "e" in rendered:
        rendered, raw_exponent = rendered.split("e", 1)
        exponent = int(raw_exponent)
        exponent_text = f"e{'+' if exponent >= 0 else ''}{exponent}"

    first, separator, last = rendered.partition(".")
    if last == "0":
        separator = ""
        last = ""

    if 0 < exponent < 21:
        first += last
        last = ""
        separator = ""
        exponent_text = ""
        zero_count = exponent - len(first) + 1
        if zero_count > 0:
            first += "0" * zero_count
    elif -7 < exponent < 0:
        last = first + last
        first = "0"
        separator = "."
        exponent_text = ""
        last = "0" * (-exponent - 1) + last

    return sign + first + separator + last + exponent_text


def _serialize_jcs(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_quote(value)
    if isinstance(value, (int, float)):
        return _serialize_jcs_number(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize_jcs(item) for item in value) + "]"
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: _utf16_sort_key(item[0]))
        return (
            "{"
            + ",".join(_jcs_quote(key) + ":" + _serialize_jcs(item) for key, item in ordered)
            + "}"
        )
    raise AssertionError("_assert_jcs_value must reject unsupported values")


def _serialize_jcs_hash(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _jcs_quote(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(_serialize_jcs_hash(item) for item in value) + "]"
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: _utf16_sort_key(item[0]))
        return (
            "{"
            + ",".join(_jcs_quote(key) + ":" + _serialize_jcs_hash(item) for key, item in ordered)
            + "}"
        )
    raise AssertionError("_assert_jcs_hash_value must reject unsupported values")


def canonical_json(value: object) -> str:
    """Return RFC 8785/JCS text for a complete capability value."""

    _assert_jcs_value(value)
    return _serialize_jcs(value)


def canonical_hash_json(value: object) -> str:
    """Return JCS text for the restricted V1 compatibility preimage."""

    _assert_jcs_hash_value(value)
    return _serialize_jcs_hash(value)


def _no_json_constant(value: str) -> None:
    raise JSONDocumentError("contract_json_non_finite_number", value)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JSONDocumentError("contract_json_duplicate_key", key)
        result[key] = value
    return result


def decode_json_document(content: bytes) -> Any:
    """Decode strict UTF-8 JSON and reject duplicate keys/non-finite values."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JSONDocumentError("contract_json_invalid_utf8", str(exc)) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_no_json_constant,
        )
    except JSONDocumentError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise JSONDocumentError("contract_json_invalid", str(exc)) from exc


@lru_cache(maxsize=1)
def _schema_bundle() -> tuple[dict[str, Any], Draft202012Validator]:
    schema_path = files("kiara.retrieval").joinpath("schemas", SCHEMA_FILENAME)
    raw = schema_path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != SCHEMA_SHA256:
        raise RuntimeError(
            f"KIron capability schema hash mismatch: {actual_hash} != {SCHEMA_SHA256}"
        )
    schema = decode_json_document(raw)
    if not isinstance(schema, dict):
        raise RuntimeError("KIron capability schema root is not an object")
    Draft202012Validator.check_schema(schema)
    return schema, Draft202012Validator(schema)


def schema_snapshot() -> dict[str, Any]:
    """Return a detached copy of the authoritative local S1 schema."""

    schema, _ = _schema_bundle()
    return json.loads(json.dumps(schema, ensure_ascii=False))


def _json_pointer(parts: Sequence[object]) -> str:
    encoded = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    suffix = "/".join(encoded)
    return "/kiron_capabilities" + (f"/{suffix}" if suffix else "")


def _validate_schema(value: Mapping[str, Any]) -> None:
    _, validator = _schema_bundle()
    errors = sorted(
        validator.iter_errors(value),
        # JSON-Schema paths may mix array indices and object names.  Convert
        # every segment before ordering so diagnostics remain deterministic
        # without ever comparing ``int`` and ``str`` directly.
        key=lambda item: (
            tuple(str(segment) for segment in item.absolute_path),
            item.message,
        ),
    )
    if not errors:
        return
    error = errors[0]
    detail = error.message
    if len(detail) > 500:
        detail = detail[:497] + "..."
    _raise_contract(
        "contract_schema_invalid",
        _json_pointer(error.absolute_path),
        detail,
    )


def _require_sorted_unique_strings(values: list[str], path: str) -> None:
    if len(values) != len(set(values)):
        _raise_contract("contract_invariant_invalid", path, "duplicate values")
    if values != sorted(values):
        _raise_contract("contract_invariant_invalid", path, "values are not sorted")


def _validate_digest_order(profile: Mapping[str, Any], profile_path: str) -> None:
    artifact = profile["artifact"]
    paths_by_field: dict[str, list[str]] = {}
    for field in ("weights", "auxiliary"):
        paths = [item["path"] for item in artifact[field]]
        paths_by_field[field] = paths
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            _raise_contract(
                "contract_invariant_invalid",
                f"{profile_path}/artifact/{field}",
                "digest entries must be unique and sorted by path",
            )
    if set(paths_by_field["weights"]).intersection(paths_by_field["auxiliary"]):
        _raise_contract(
            "contract_invariant_invalid",
            f"{profile_path}/artifact",
            "weight and auxiliary paths overlap",
        )


def _validate_input_type_behavior(profile: Mapping[str, Any], profile_path: str) -> None:
    input_type = profile["input_type"]
    additional = input_type["additional_task_roles"]
    _require_sorted_unique_strings(additional, f"{profile_path}/input_type/additional_task_roles")

    aliases = input_type["aliases"]
    alias_names = [item["alias"] for item in aliases]
    if len(alias_names) != len(set(alias_names)):
        _raise_contract(
            "contract_invariant_invalid",
            f"{profile_path}/input_type/aliases",
            "alias names must be unique",
        )
    alias_order = [(item["alias"], item["canonical"]) for item in aliases]
    if alias_order != sorted(alias_order):
        _raise_contract(
            "contract_invariant_invalid",
            f"{profile_path}/input_type/aliases",
            "aliases are not sorted by alias and canonical role",
        )

    role_sensitive = input_type["role_sensitive"]
    required = input_type["required"]
    default = input_type["default"]
    behavior = input_type["missing_role_behavior"]
    if role_sensitive is True:
        provider_phase = (
            required is False and default == "search_document" and behavior == "use_default"
        )
        strict_phase = required is True and default is None and behavior == "reject"
        if not (provider_phase or strict_phase):
            _raise_contract(
                "contract_invariant_invalid",
                f"{profile_path}/input_type",
                "role-sensitive enforcement fields do not form a valid phase",
            )
    elif role_sensitive is False:
        if required or default is not None or behavior != "no_op":
            _raise_contract(
                "contract_invariant_invalid",
                f"{profile_path}/input_type",
                "role-independent profile must use no_op without a default",
            )
        formatting = profile["pipeline"]["formatting"]
        if canonical_json(formatting["search_document"]) != canonical_json(
            formatting["search_query"]
        ):
            _raise_contract(
                "contract_invariant_invalid",
                f"{profile_path}/pipeline/formatting",
                "role-independent templates must be identical",
            )
        by_role = profile["max_input_tokens"]["by_role"]
        if by_role["search_document"] != by_role["search_query"]:
            _raise_contract(
                "contract_invariant_invalid",
                f"{profile_path}/max_input_tokens/by_role",
                "role-independent limits must be identical",
            )
    elif required or default is not None or behavior != "unknown":
        _raise_contract(
            "contract_invariant_invalid",
            f"{profile_path}/input_type",
            "unknown role sensitivity must remain fail-closed",
        )


def _build_preimage(
    capabilities: Mapping[str, Any], profile_index: int, role: str
) -> dict[str, Any]:
    profile = capabilities["profiles"][profile_index]
    artifact = profile["artifact"]
    pipeline = profile["pipeline"]
    formatting = pipeline["formatting"][role]
    limits = profile["max_input_tokens"]
    return {
        "artifact": {
            "auxiliary": [
                {"path": item["path"], "sha256": item["sha256"]} for item in artifact["auxiliary"]
            ],
            "manifest_digest": artifact["manifest_digest"],
            "repository": artifact["repository"],
            "revision": artifact["revision"],
            "weights": [
                {"path": item["path"], "sha256": item["sha256"]} for item in artifact["weights"]
            ],
        },
        "backend": {
            "implementation": profile["backend"]["implementation"],
            "implementation_revision": profile["backend"]["implementation_revision"],
            "type": profile["backend"]["type"],
        },
        "domain": _DOMAIN_FOR_ROLE[role],
        "pipeline": {
            "formatting": {
                "owner": profile["formatting_owner"],
                "parameters": formatting["parameters"],
                "template": formatting["template"],
            },
            "max_input_tokens": {
                "counting": limits["counting"],
                "limit": limits["by_role"][role],
                "overflow": limits["overflow"],
                "truncation": limits["truncation"],
                "unit": limits["unit"],
            },
            "normalization": {
                "dtype": pipeline["normalization"]["dtype"],
                "method": pipeline["normalization"]["method"],
                "scope": pipeline["normalization"]["scope"],
            },
            "parameters": pipeline["parameters"],
            "pooling": {
                "method": pipeline["pooling"]["method"],
                "parameters": pipeline["pooling"]["parameters"],
            },
            "role": role,
            "tokenizer": {
                "repository": pipeline["tokenizer"]["repository"],
                "revision": pipeline["tokenizer"]["revision"],
            },
        },
        "preimage_version": 1,
        "profile": {
            "dimensions": profile["dimensions"],
            "endpoint": profile["endpoint"],
            "kind": profile["kind"],
            "output_normalized": profile["output_normalized"],
            "profile_id": profile["profile_id"],
            "similarity": profile["similarity"],
        },
    }


def _compatibility_id(capabilities: Mapping[str, Any], profile_index: int, role: str) -> str:
    preimage = _build_preimage(capabilities, profile_index, role)
    canonical = canonical_hash_json(preimage)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_behavior(value: Mapping[str, Any]) -> None:
    aliases = value["aliases"]
    _require_sorted_unique_strings(aliases, "/kiron_capabilities/aliases")
    if value["canonical_model_id"] in aliases:
        _raise_contract(
            "contract_invariant_invalid",
            "/kiron_capabilities/aliases",
            "canonical model ID is repeated as an alias",
        )

    profiles = value["profiles"]
    profile_ids = [profile["profile_id"] for profile in profiles]
    if profile_ids != sorted(profile_ids) or len(profile_ids) != len(set(profile_ids)):
        _raise_contract(
            "contract_invariant_invalid",
            "/kiron_capabilities/profiles",
            "profile IDs must be unique and sorted",
        )

    endpoints = {profile["endpoint"] for profile in profiles}
    for endpoint in endpoints:
        defaults = [
            profile
            for profile in profiles
            if profile["endpoint"] == endpoint and profile["default_for_endpoint"]
        ]
        if len(defaults) != 1:
            _raise_contract(
                "contract_invariant_invalid",
                "/kiron_capabilities/profiles",
                f"endpoint {endpoint} does not have exactly one default profile",
            )

    for index, profile in enumerate(profiles):
        profile_path = f"/kiron_capabilities/profiles/{index}"
        _validate_digest_order(profile, profile_path)
        _require_sorted_unique_strings(
            profile["verification"]["evidence"],
            f"{profile_path}/verification/evidence",
        )
        _require_sorted_unique_strings(
            profile["verification"]["blocking_reasons"],
            f"{profile_path}/verification/blocking_reasons",
        )
        _validate_input_type_behavior(profile, profile_path)

        if profile["verification"]["status"] != "verified":
            continue
        limits = profile["max_input_tokens"]
        if (limits["truncation"], limits["overflow"]) not in (
            ("right", "truncate"),
            ("right", "late_chunk_fallback"),
            ("none", "reject"),
        ):
            _raise_contract(
                "contract_invariant_invalid",
                f"{profile_path}/max_input_tokens",
                "verified profiles require a known input-limit policy pair",
            )
        expected_index = _compatibility_id(value, index, "search_document")
        expected_query = _compatibility_id(value, index, "search_query")
        if profile["index_compatibility_id"] != expected_index:
            _raise_contract(
                "contract_compatibility_id_mismatch",
                f"{profile_path}/index_compatibility_id",
                "ID does not match the normative document preimage",
            )
        if profile["query_compatibility_id"] != expected_query:
            _raise_contract(
                "contract_compatibility_id_mismatch",
                f"{profile_path}/query_compatibility_id",
                "ID does not match the normative query preimage",
            )


def _make_snapshot(value: Mapping[str, Any]) -> ContractSnapshot:
    try:
        canonical = canonical_json(value)
        detached_json = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except JCSCanonicalizationError as exc:
        raise ContractValidationError(
            "contract_jcs_invalid", "/kiron_capabilities", str(exc)
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            "contract_jcs_invalid", "/kiron_capabilities", str(exc)
        ) from exc
    return ContractSnapshot(
        json_text=detached_json,
        canonical_json=canonical,
        canonical_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def snapshot_json_object(value: Mapping[str, Any]) -> ContractSnapshot:
    """Detach a JSON object for diagnostics without declaring it V1-valid."""

    return _make_snapshot(value)


def _parameter_snapshot(value: Mapping[str, Any]) -> ContractSnapshot:
    canonical = canonical_json(value)
    return ContractSnapshot(
        json_text=json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        canonical_json=canonical,
        canonical_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def _parse_profile(raw: Mapping[str, Any]) -> EmbeddingProfileContract:
    backend_raw = raw["backend"]
    artifact_raw = raw["artifact"]
    limits_raw = raw["max_input_tokens"]
    input_raw = raw["input_type"]
    pipeline_raw = raw["pipeline"]
    formatting_raw = pipeline_raw["formatting"]
    verification_raw = raw["verification"]
    return EmbeddingProfileContract(
        profile_id=raw["profile_id"],
        kind=ProfileKind(raw["kind"]),
        endpoint=raw["endpoint"],
        default_for_endpoint=raw["default_for_endpoint"],
        backend=BackendIdentity(
            backend_type=backend_raw["type"],
            implementation=backend_raw["implementation"],
            implementation_revision=backend_raw["implementation_revision"],
        ),
        artifact=ArtifactIdentity(
            repository=artifact_raw["repository"],
            revision=artifact_raw["revision"],
            manifest_digest=artifact_raw["manifest_digest"],
            weights=tuple(
                DigestFile(path=item["path"], sha256=item["sha256"])
                for item in artifact_raw["weights"]
            ),
            auxiliary=tuple(
                DigestFile(path=item["path"], sha256=item["sha256"])
                for item in artifact_raw["auxiliary"]
            ),
        ),
        dimensions=raw["dimensions"],
        max_input_tokens=MaxInputTokens(
            unit=limits_raw["unit"],
            counting=limits_raw["counting"],
            truncation=limits_raw["truncation"],
            overflow=limits_raw["overflow"],
            by_role=RoleLimits(**limits_raw["by_role"]),
        ),
        output_normalized=raw["output_normalized"],
        similarity=raw["similarity"],
        input_type=InputTypeContract(
            supported=tuple(input_raw["supported"]),
            additional_task_roles=tuple(input_raw["additional_task_roles"]),
            aliases=tuple(
                InputTypeAlias(alias=item["alias"], canonical=item["canonical"])
                for item in input_raw["aliases"]
            ),
            role_sensitive=input_raw["role_sensitive"],
            required=input_raw["required"],
            default=input_raw["default"],
            missing_role_behavior=input_raw["missing_role_behavior"],
        ),
        formatting_owner=raw["formatting_owner"],
        pipeline=PipelineContract(
            tokenizer=TokenizerContract(
                repository=pipeline_raw["tokenizer"]["repository"],
                revision=pipeline_raw["tokenizer"]["revision"],
            ),
            search_document_formatting=FormattingContract(
                template=formatting_raw["search_document"]["template"],
                parameters=_parameter_snapshot(formatting_raw["search_document"]["parameters"]),
            ),
            search_query_formatting=FormattingContract(
                template=formatting_raw["search_query"]["template"],
                parameters=_parameter_snapshot(formatting_raw["search_query"]["parameters"]),
            ),
            pooling=PoolingContract(
                method=pipeline_raw["pooling"]["method"],
                parameters=_parameter_snapshot(pipeline_raw["pooling"]["parameters"]),
            ),
            normalization=NormalizationContract(
                method=pipeline_raw["normalization"]["method"],
                dtype=pipeline_raw["normalization"]["dtype"],
                scope=pipeline_raw["normalization"]["scope"],
            ),
            parameters=_parameter_snapshot(pipeline_raw["parameters"]),
        ),
        verification=VerificationContract(
            status=VerificationStatus(verification_raw["status"]),
            evidence=tuple(verification_raw["evidence"]),
            blocking_reasons=tuple(verification_raw["blocking_reasons"]),
        ),
        index_compatibility_id=raw["index_compatibility_id"],
        query_compatibility_id=raw["query_compatibility_id"],
        snapshot=_make_snapshot(raw),
    )


def parse_capabilities(raw: object) -> EmbeddingCapabilities:
    """Validate and type one complete ``kiron_capabilities`` V1 object."""

    if not isinstance(raw, Mapping):
        _raise_contract("contract_schema_invalid", "/kiron_capabilities", "must be an object")
    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, (int, float)):
        _raise_contract(
            "contract_schema_invalid",
            "/kiron_capabilities/schema_version",
            "must be JSON number 1",
        )
    if not math.isfinite(float(version)) or version != SCHEMA_VERSION:
        _raise_contract(
            "contract_unknown_schema_version",
            "/kiron_capabilities/schema_version",
            f"unsupported schema version {version!r}",
        )

    # Detach before validation/model construction so no caller-owned mutable
    # reference can modify a successfully validated snapshot afterwards.
    try:
        detached = json.loads(
            json.dumps(raw, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ContractValidationError(
            "contract_jcs_invalid", "/kiron_capabilities", str(exc)
        ) from exc
    if not isinstance(detached, dict):
        _raise_contract("contract_schema_invalid", "/kiron_capabilities", "must be an object")

    _validate_schema(detached)
    try:
        canonical_json(detached)
    except JCSCanonicalizationError as exc:
        raise ContractValidationError(
            "contract_jcs_invalid", "/kiron_capabilities", str(exc)
        ) from exc
    _validate_behavior(detached)
    snapshot = _make_snapshot(detached)
    return EmbeddingCapabilities(
        schema_version=SCHEMA_VERSION,
        canonical_model_id=detached["canonical_model_id"],
        aliases=tuple(detached["aliases"]),
        profiles=tuple(_parse_profile(profile) for profile in detached["profiles"]),
        snapshot=snapshot,
    )


def capabilities_equal(left: EmbeddingCapabilities, right: EmbeddingCapabilities) -> bool:
    """Compare independently validated complete snapshots by JCS bytes."""

    return left.snapshot.canonical_json == right.snapshot.canonical_json


def is_sha256_compatibility_id(value: object) -> bool:
    return isinstance(value, str) and _SHA256_ID.fullmatch(value) is not None
