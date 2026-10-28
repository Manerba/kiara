"""S3 regression suite for generic KIron embedding-contract consumption."""

from __future__ import annotations

import copy
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from kiara.retrieval.embedding_contract import (
    SCHEMA_SHA256,
    ContractState,
    ContractValidationError,
    JSONDocumentError,
    ProfileKind,
    VerificationStatus,
    decode_json_document,
    parse_capabilities,
    schema_snapshot,
)
from kiara.retrieval.embedding_discovery import (
    EmbeddingContractInventory,
    ServerContractDiscovery,
    discover_configured_embedding_backends,
    discover_server_contracts,
    invalidate_embedding_contract_cache,
    resolve_embedding_profile,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "kiron-embedding-capabilities-v1.examples.json"


@pytest.fixture(scope="module")
def examples() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _valid(examples: dict, name: str) -> dict:
    return copy.deepcopy(next(item["value"] for item in examples["valid"] if item["name"] == name))


def _json_pointer_parent(document: object, pointer: str) -> tuple[object, str]:
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
    current = document
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current, parts[-1]


def _json_pointer_get(document: object, pointer: str) -> object:
    parent, key = _json_pointer_parent(document, pointer)
    return parent[int(key)] if isinstance(parent, list) else parent[key]


def _materialize_invalid_example(examples: dict, example: dict) -> dict:
    value = _valid(examples, example["base_valid"])
    for operation in example["operations"]:
        parent, key = _json_pointer_parent(value, operation["path"])
        if operation["op"] == "replace":
            replacement = copy.deepcopy(operation["value"])
            if isinstance(parent, list):
                parent[int(key)] = replacement
            else:
                parent[key] = replacement
        elif operation["op"] == "remove":
            parent.pop(int(key)) if isinstance(parent, list) else parent.pop(key)
        elif operation["op"] == "copy":
            copied = copy.deepcopy(_json_pointer_get(value, operation["from"]))
            if isinstance(parent, list):
                parent.append(copied) if key == "-" else parent.insert(int(key), copied)
            else:
                parent[key] = copied
        else:  # pragma: no cover - authoritative fixture only uses these operations
            raise AssertionError(f"unsupported fixture operation: {operation['op']}")
    return value


def _model_row(contract: dict) -> dict:
    canonical = contract["canonical_model_id"]
    return {
        "name": canonical,
        "model": canonical,
        "details": {},
        "kiron_capabilities": contract,
    }


def _mock_transport(
    tags_rows: list[dict], show_contracts: dict[str, object]
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": tags_rows})
        if request.url.path == "/api/show":
            payload = json.loads(request.content)
            value = show_contracts[payload["model"]]
            if isinstance(value, httpx.Response):
                return value
            return httpx.Response(
                200,
                json={
                    "model": payload["model"],
                    "kiron_capabilities": value,
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _discover(contract: dict, *, shown: object | None = None) -> ServerContractDiscovery:
    shown = copy.deepcopy(contract) if shown is None else shown
    transport = _mock_transport(
        [_model_row(copy.deepcopy(contract))],
        {contract["canonical_model_id"]: shown},
    )
    return discover_server_contracts("http://contract.test", transport=transport)


def _run_ki_modelle_diagnostic_failure(failure_mode: str) -> dict:
    source = (
        Path(__file__).parents[1] / "kiara" / "web" / "static" / "js" / "tab_ki_modelle.js"
    ).read_text(encoding="utf-8")
    harness = f"""
const failureMode = {json.dumps(failure_mode)};
const contentContainer = {{ innerHTML: '', setAttribute() {{}} }};
const categoryFilters = {{ innerHTML: '' }};
const filterBar = {{
    innerHTML: '',
    style: {{}},
    classList: {{ remove() {{}}, add() {{}} }},
    appendChild() {{}},
}};
globalThis.document = {{
    body: {{ _kiModelleActionsBound: false, addEventListener() {{}} }},
    createElement() {{ return {{ className: '', id: '', innerHTML: '' }}; }},
    getElementById(id) {{
        if (id === 'contentContainer') return contentContainer;
        if (id === 'filterBar') return filterBar;
        if (id === 'backendCategoryFilters') return categoryFilters;
        return null;
    }},
    querySelectorAll() {{ return []; }},
}};
globalThis.adminCurrentSubTab = 'ki-modelle';
globalThis._runTabCleanups = () => {{}};
globalThis.registerTabCleanup = () => {{}};
const requiredResponses = {{
    '/api/admin/ki/backends/categories': {{
        ok: true,
        json: async () => ({{ categories: [{{ id: 'embedding', label: 'Embedding' }}] }}),
    }},
    '/api/admin/ki/backends/': {{
        ok: true,
        json: async () => ({{ backends: [{{
            id: 7,
            name: 'Funktionierendes Backend',
            category: 'embedding',
            provider: 'ollama',
            model: 'nomic-embed-text:latest',
            url: 'http://backend.test',
            base_url: '',
            priority: 10,
            is_active: true,
            last_test_ok: null,
            last_test_at: null,
            last_test_error: '',
        }}] }}),
    }},
}};
globalThis.SharedUi = {{
    fetch: {{
        safeFetch: async (url) => {{
            if (url !== '/api/admin/embedding-contracts') return requiredResponses[url];
            if (failureMode === 'network') throw new Error('offline');
            if (failureMode === 'non_2xx') return {{ ok: false, json: async () => ({{}}) }};
            return {{
                ok: true,
                json: async () => {{ throw new SyntaxError('malformed JSON'); }},
            }};
        }},
    }},
    runtime: {{
        escapeHtml: (value) => String(value),
        escapeAttr: (value) => String(value),
    }},
    modal: {{}},
    notify: {{}},
}};
(async () => {{
    await initKiModelleTab();
    process.stdout.write(JSON.stringify({{
        html: contentContainer.innerHTML,
        category_html: categoryFilters.innerHTML,
    }}));
}})().catch((error) => {{
    console.error(error.stack || error);
    process.exit(1);
}});
"""
    completed = subprocess.run(
        ["node", "-"],
        input=source + "\n" + harness,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


class TestAuthoritativeContractModel:
    def test_schema_is_byte_identical_and_loadable(self):
        schema_path = (
            Path(__file__).parents[1]
            / "kiara"
            / "retrieval"
            / "schemas"
            / "kiron-embedding-capabilities-v1.schema.json"
        )
        import hashlib

        assert hashlib.sha256(schema_path.read_bytes()).hexdigest() == SCHEMA_SHA256
        assert schema_snapshot()["$id"].endswith("kiron-embedding-capabilities-v1.schema.json")

    def test_all_authoritative_valid_examples_parse(self, examples):
        parsed = [parse_capabilities(item["value"]) for item in examples["valid"]]
        assert len(parsed) == 7
        assert all(item.schema_version == 1 for item in parsed)

    def test_all_authoritative_invalid_examples_fail_closed(self, examples):
        rejected = []
        for item in examples["invalid"]:
            with pytest.raises(ContractValidationError):
                parse_capabilities(_materialize_invalid_example(examples, item))
            rejected.append(item["name"])
        assert len(rejected) == 15

    def test_provider_and_strict_phase_keep_same_ids(self, examples):
        provider = parse_capabilities(_valid(examples, "role_sensitive_provider_phase"))
        strict = parse_capabilities(_valid(examples, "role_sensitive_strict_phase_same_ids"))
        assert provider.profiles[0].input_type.required is False
        assert strict.profiles[0].input_type.required is True
        assert (
            provider.profiles[0].index_compatibility_id == strict.profiles[0].index_compatibility_id
        )
        assert (
            provider.profiles[0].query_compatibility_id == strict.profiles[0].query_compatibility_id
        )

    def test_role_independent_dense_profile_is_typed_and_verified(self, examples):
        contract = parse_capabilities(
            _valid(examples, "role_independent_canonical_roles_are_no_op")
        )
        profile = contract.profiles[0]
        assert profile.kind is ProfileKind.DENSE
        assert profile.verification.status is VerificationStatus.VERIFIED
        assert profile.input_type.role_sensitive is False
        assert profile.input_type.missing_role_behavior == "no_op"
        assert profile.rag_verified is True

    def test_multiple_profiles_remain_distinct(self, examples):
        contract = parse_capabilities(_valid(examples, "merged_service_ollama_and_late_profiles"))
        assert [(item.profile_id, item.kind, item.endpoint) for item in contract.profiles] == [
            ("ollama-dense-unverified-v1", ProfileKind.DENSE, "/api/embed"),
            ("service-dense-v1", ProfileKind.DENSE, "/api/embed"),
            ("service-late-unverified-v1", ProfileKind.LATE_CHUNKING, "/api/embed_late"),
        ]
        assert contract.default_profile("/api/embed").profile_id == "service-dense-v1"
        assert (
            contract.default_profile("/api/embed_late").profile_id == "service-late-unverified-v1"
        )

    def test_derived_colbert_is_not_dense(self, examples):
        value = _valid(examples, "unverified_unknown_pipeline_has_null_ids")
        value["canonical_model_id"] = "example/colbert:1"
        value["aliases"] = ["example-colbert"]
        profile = value["profiles"][0]
        profile["profile_id"] = "colbert-multivector-v1"
        profile["kind"] = "multi_vector"
        profile["endpoint"] = "/api/embed_colbert"
        contract = parse_capabilities(value)
        assert contract.profiles[0].kind is ProfileKind.MULTI_VECTOR
        assert contract.profiles[0].endpoint == "/api/embed_colbert"
        assert contract.default_profile("/api/embed") is None

    def test_unknown_extensions_and_float_are_preserved_but_detached(self, examples):
        value = _valid(examples, "unknown_extensions_are_preserved_and_hash_irrelevant")
        value["x_runtime_score"] = 0.5
        value["profiles"][0]["pipeline"]["tokenizer"]["x_ratio"] = 0.5
        contract = parse_capabilities(value)
        first = contract.snapshot.as_dict()
        assert first["x_runtime_score"] == 0.5
        assert first["profiles"][0]["pipeline"]["tokenizer"]["x_ratio"] == 0.5
        first["x_runtime_score"] = 99
        assert contract.snapshot.as_dict()["x_runtime_score"] == 0.5
        value["x_runtime_score"] = 123
        assert contract.snapshot.as_dict()["x_runtime_score"] == 0.5

    def test_typed_objects_are_frozen(self, examples):
        contract = parse_capabilities(_valid(examples, "role_sensitive_provider_phase"))
        with pytest.raises(FrozenInstanceError):
            contract.canonical_model_id = "changed"  # type: ignore[misc]

    def test_diagnostic_exports_are_fully_detached(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        server = _discover(value)
        profile = server.groups[0].capabilities.profiles[0]

        first = server.to_dict()
        exported_profile = first["groups"][0]["profiles"][0]
        exported_profile["artifact"]["weights"][0]["path"] = "MUTATED-WEIGHT"
        exported_profile["artifact"]["auxiliary"][0]["path"] = "MUTATED-AUX"
        exported_profile["input_type"]["aliases"][0]["alias"] = "MUTATED-ALIAS"

        assert profile.artifact.weights[0].path == "model.safetensors"
        assert profile.artifact.auxiliary[0].path == "adapter.safetensors"
        assert profile.input_type.aliases[0].alias == "document"

        second_profile = server.to_dict()["groups"][0]["profiles"][0]
        assert second_profile["artifact"]["weights"][0]["path"] == "model.safetensors"
        assert second_profile["artifact"]["auxiliary"][0]["path"] == "adapter.safetensors"
        assert second_profile["input_type"]["aliases"][0]["alias"] == "document"

    @pytest.mark.parametrize(
        ("version", "reason"),
        [
            (True, "contract_schema_invalid"),
            (False, "contract_schema_invalid"),
            ("1", "contract_schema_invalid"),
            (None, "contract_schema_invalid"),
            (0.5, "contract_unknown_schema_version"),
            (2, "contract_unknown_schema_version"),
        ],
    )
    def test_schema_version_is_type_safe(self, examples, version, reason):
        value = _valid(examples, "role_sensitive_provider_phase")
        value["schema_version"] = version
        with pytest.raises(ContractValidationError) as caught:
            parse_capabilities(value)
        assert caught.value.issue.reason_code == reason
        assert caught.value.issue.field_path == "/kiron_capabilities/schema_version"

    def test_json_number_one_point_zero_is_schema_equivalent(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        value["schema_version"] = 1.0
        assert parse_capabilities(value).schema_version == 1

    @pytest.mark.parametrize(
        "mutation",
        [
            "verified_null_id",
            "unverified_non_null_id",
            "unverified_without_blocker",
            "kind_endpoint_mismatch",
        ],
    )
    def test_schema_semantics_fail_closed(self, examples, mutation):
        if mutation.startswith("unverified"):
            value = _valid(examples, "unverified_unknown_pipeline_has_null_ids")
        else:
            value = _valid(examples, "role_sensitive_provider_phase")
        profile = value["profiles"][0]
        if mutation == "verified_null_id":
            profile["index_compatibility_id"] = None
        elif mutation == "unverified_non_null_id":
            profile["index_compatibility_id"] = "sha256:" + "a" * 64
        elif mutation == "unverified_without_blocker":
            profile["verification"]["blocking_reasons"] = []
        else:
            profile["endpoint"] = "/api/embed_late"
        with pytest.raises(ContractValidationError):
            parse_capabilities(value)

    @pytest.mark.parametrize(
        ("path", "invalid"),
        [
            (("profiles", 0, "max_input_tokens", "by_role", "query"), 512),
            (("profiles", 0, "kind"), "sparse"),
            (("profiles", 0, "dimensions"), "768"),
            (("profiles", 0, "index_compatibility_id"), "sha256:not-a-digest"),
        ],
    )
    def test_closed_role_map_enums_types_and_ids_fail_schema(self, examples, path, invalid):
        value = _valid(examples, "role_sensitive_provider_phase")
        target = value
        for segment in path[:-1]:
            target = target[segment]
        target[path[-1]] = invalid
        with pytest.raises(ContractValidationError) as caught:
            parse_capabilities(value)
        assert caught.value.issue.reason_code == "contract_schema_invalid"

    def test_duplicate_profiles_and_defaults_fail_behavioral_gate(self, examples):
        value = _valid(examples, "merged_service_ollama_and_late_profiles")
        value["profiles"][0]["default_for_endpoint"] = True
        with pytest.raises(ContractValidationError) as caught:
            parse_capabilities(value)
        assert caught.value.issue.reason_code == "contract_invariant_invalid"

        value = _valid(examples, "merged_service_ollama_and_late_profiles")
        value["profiles"][1]["profile_id"] = value["profiles"][0]["profile_id"]
        with pytest.raises(ContractValidationError):
            parse_capabilities(value)

    def test_stale_compatibility_id_is_recomputed_and_rejected(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        value["profiles"][0]["query_compatibility_id"] = "sha256:" + "0" * 64
        with pytest.raises(ContractValidationError) as caught:
            parse_capabilities(value)
        assert caught.value.issue.reason_code == "contract_compatibility_id_mismatch"
        assert caught.value.issue.field_path.endswith("/query_compatibility_id")

    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_extension_is_rejected(self, examples, invalid):
        value = _valid(examples, "role_sensitive_provider_phase")
        value["x_runtime"] = invalid
        with pytest.raises(ContractValidationError) as caught:
            parse_capabilities(value)
        assert caught.value.issue.reason_code == "contract_jcs_invalid"

    def test_lone_surrogate_is_rejected(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        value["x_text"] = "\ud800"
        with pytest.raises(ContractValidationError) as caught:
            parse_capabilities(value)
        assert caught.value.issue.reason_code == "contract_jcs_invalid"

    def test_duplicate_json_keys_are_rejected_before_modeling(self):
        with pytest.raises(JSONDocumentError) as caught:
            decode_json_document(b'{"kiron_capabilities":{},"kiron_capabilities":{}}')
        assert caught.value.reason_code == "contract_json_duplicate_key"


class TestCentralDiscovery:
    def test_tags_show_and_exact_alias_resolution(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        server = _discover(value)
        assert server.state is ContractState.VALID
        assert server.groups[0].show_checked is True

        canonical = resolve_embedding_profile(
            server, value["canonical_model_id"], endpoint="/api/embed", kind=ProfileKind.DENSE
        )
        alias = resolve_embedding_profile(
            server, value["aliases"][0], endpoint="/api/embed", kind=ProfileKind.DENSE
        )
        not_exact = resolve_embedding_profile(
            server,
            value["canonical_model_id"].upper(),
            endpoint="/api/embed",
            kind=ProfileKind.DENSE,
        )
        assert canonical.rag_verified is True
        assert alias.rag_verified is True
        assert canonical.capabilities.canonical_model_id == alias.capabilities.canonical_model_id
        assert not_exact.state is ContractState.MISSING

    def test_explicit_profile_must_match_exact_endpoint_and_kind(self, examples):
        value = _valid(examples, "merged_service_ollama_and_late_profiles")
        server = _discover(value)
        resolution = resolve_embedding_profile(
            server,
            value["canonical_model_id"],
            endpoint="/api/embed_late",
            profile_id="service-dense-v1",
            kind=ProfileKind.LATE_CHUNKING,
        )
        assert resolution.state is ContractState.INCONSISTENT
        assert set(item.reason_code for item in resolution.issues) == {
            "contract_profile_endpoint_mismatch",
            "contract_profile_kind_mismatch",
        }

    def test_unverified_profile_remains_visible_but_not_rag_verified(self, examples):
        value = _valid(examples, "merged_service_ollama_and_late_profiles")
        server = _discover(value)
        resolution = resolve_embedding_profile(
            server,
            value["canonical_model_id"],
            endpoint="/api/embed_late",
            kind=ProfileKind.LATE_CHUNKING,
        )
        assert resolution.profile.profile_id == "service-late-unverified-v1"
        assert resolution.state is ContractState.UNVERIFIED
        assert resolution.rag_verified is False

    def test_plain_ollama_model_has_explicit_missing_state(self):
        transport = _mock_transport(
            [{"name": "plain:latest", "model": "plain:latest", "details": {}}],
            {},
        )
        server = discover_server_contracts("http://plain.test", transport=transport)
        result = resolve_embedding_profile(
            server, "plain:latest", endpoint="/api/embed", kind=ProfileKind.DENSE
        )
        assert server.state is ContractState.MISSING
        assert result.state is ContractState.MISSING
        assert result.issues[0].reason_code == "contract_missing"

    @pytest.mark.parametrize(
        "mutation",
        ["id", "known_field", "extension", "extension_array_order"],
    )
    def test_any_tags_show_difference_is_detected(self, examples, mutation):
        tags_value = _valid(examples, "role_sensitive_provider_phase")
        shown = copy.deepcopy(tags_value)
        if mutation == "id":
            shown["profiles"][0]["query_compatibility_id"] = "sha256:" + "0" * 64
        elif mutation == "known_field":
            shown = _valid(examples, "role_sensitive_strict_phase_same_ids")
        elif mutation == "extension":
            shown["x_runtime_score"] = 0.5
        else:
            tags_value["x_order"] = [1, 2]
            shown["x_order"] = [2, 1]

        server = _discover(tags_value, shown=shown)
        assert server.state is ContractState.INCONSISTENT
        assert server.groups[0].state is ContractState.INCONSISTENT
        assert server.groups[0].show_checked is True
        resolution = resolve_embedding_profile(
            server, tags_value["canonical_model_id"], endpoint="/api/embed"
        )
        assert resolution.rag_verified is False

    def test_unknown_and_boolean_schema_versions_are_visible(self, examples):
        unknown = _valid(examples, "role_sensitive_provider_phase")
        unknown["schema_version"] = 2
        server = _discover(unknown)
        assert server.state is ContractState.UNKNOWN_SCHEMA_VERSION
        assert server.groups[0].raw_snapshot.as_dict()["schema_version"] == 2

        malformed = _valid(examples, "role_sensitive_provider_phase")
        malformed["schema_version"] = True
        server = _discover(malformed)
        assert server.state is ContractState.MALFORMED
        assert server.groups[0].issues[0].field_path == "/kiron_capabilities/schema_version"

    def test_duplicate_alias_across_groups_blocks_entire_registry(self, examples):
        first = _valid(examples, "role_sensitive_provider_phase")
        second = _valid(examples, "role_sensitive_provider_phase")
        first["canonical_model_id"] = "example/first:1"
        first["aliases"] = ["shared-alias"]
        second["canonical_model_id"] = "example/second:1"
        second["aliases"] = ["shared-alias"]
        transport = _mock_transport(
            [_model_row(first), _model_row(second)],
            {
                first["canonical_model_id"]: copy.deepcopy(first),
                second["canonical_model_id"]: copy.deepcopy(second),
            },
        )
        server = discover_server_contracts("http://ambiguous.test", transport=transport)
        assert server.state is ContractState.AMBIGUOUS
        assert all(group.state is ContractState.AMBIGUOUS for group in server.groups)
        assert "contract_alias_ambiguous" in {item.reason_code for item in server.issues}

    def test_discovery_calls_only_tags_and_show(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [_model_row(value)]})
            if request.url.path == "/api/show":
                return httpx.Response(200, json={"kiron_capabilities": value})
            raise AssertionError(f"unexpected write/vector request: {request.url.path}")

        server = discover_server_contracts(
            "http://readonly.test", transport=httpx.MockTransport(handler)
        )
        assert server.state is ContractState.VALID
        assert paths == ["/api/tags", "/api/show"]


class TestDiscoveryCache:
    def setup_method(self):
        invalidate_embedding_contract_cache()

    def teardown_method(self):
        invalidate_embedding_contract_cache()

    def test_concurrent_calls_share_one_complete_snapshot(self, examples):
        result = _discover(_valid(examples, "role_sensitive_provider_phase"))
        calls = 0
        calls_lock = threading.Lock()

        def fake_fetch(*args, **kwargs):
            nonlocal calls
            with calls_lock:
                calls += 1
            threading.Event().wait(0.05)
            return result

        with patch("kiara.retrieval.embedding_discovery._discover_server_uncached", fake_fetch):
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(discover_server_contracts, "http://cache.test")
                    for _ in range(8)
                ]
                returned = [future.result(timeout=2) for future in futures]
        assert calls == 1
        assert all(item is returned[0] for item in returned)

    def test_invalidation_prevents_stale_flight_writeback(self, examples):
        fresh = _discover(_valid(examples, "role_sensitive_provider_phase"))
        stale = replace(fresh, backend_url="http://stale.test")
        fresh = replace(fresh, backend_url="http://fresh.test")
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def fake_fetch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                assert release.wait(timeout=2)
                return stale
            return fresh

        with patch("kiara.retrieval.embedding_discovery._discover_server_uncached", fake_fetch):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(discover_server_contracts, "http://race.test")
                assert started.wait(timeout=2)
                invalidate_embedding_contract_cache("http://race.test")
                release.set()
                returned = future.result(timeout=2)
        assert calls == 2
        assert returned.backend_url == "http://fresh.test"

    def test_explicit_discovery_errors_are_not_cached(self):
        error = ServerContractDiscovery(
            backend_url="http://error.test",
            state=ContractState.DISCOVERY_ERROR,
            groups=(),
            plain_model_names=(),
        )
        with patch(
            "kiara.retrieval.embedding_discovery._discover_server_uncached",
            return_value=error,
        ) as fetch:
            discover_server_contracts("http://error.test")
            discover_server_contracts("http://error.test")
        assert fetch.call_count == 2

    def test_backend_selector_invalidation_reaches_contract_cache(self):
        from kiara.backend_selector import invalidate_cache

        with patch(
            "kiara.retrieval.embedding_discovery.invalidate_embedding_contract_cache"
        ) as invalidate:
            invalidate_cache("embedding")
        invalidate.assert_called_once_with()


class TestBackendAndAdminDiagnostics:
    def test_cloud_is_not_applicable_and_kiron_keeps_requested_identity(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        server = _discover(value)
        backends = [
            {
                "id": 1,
                "name": "KIron",
                "provider": "ollama",
                "url": "http://kiron.test",
                "model": value["aliases"][0],
                "priority": 10,
                "extra_config": "{}",
            },
            {
                "id": 2,
                "name": "OpenAI",
                "provider": "openai",
                "url": "",
                "model": "text-embedding-3-large",
                "priority": 20,
                "extra_config": "{}",
            },
        ]
        with patch(
            "kiara.retrieval.embedding_discovery.discover_server_contracts",
            return_value=server,
        ):
            inventory = discover_configured_embedding_backends(backends)
        assert inventory.backends[0].requested_model == value["aliases"][0]
        assert (
            inventory.backends[0].resolution.capabilities.canonical_model_id
            == value["canonical_model_id"]
        )
        assert inventory.backends[0].rag_verified is True
        assert inventory.backends[1].state is ContractState.NOT_APPLICABLE
        assert inventory.backends[1].rag_verified is False

    @pytest.mark.parametrize(
        ("extra_config", "expected_reason_codes"),
        [
            ("", {"backend_extra_config_malformed"}),
            ("{", {"backend_extra_config_malformed"}),
            (json.dumps({"profile_id": 123}), {"backend_profile_id_malformed"}),
            (json.dumps({"profile_id": ""}), {"backend_profile_id_malformed"}),
            (
                json.dumps(
                    {
                        "embedding_profile_id": "service-dense-v1",
                        "profile_id": "other-dense-v1",
                    }
                ),
                {"backend_profile_id_ambiguous"},
            ),
            (
                json.dumps(
                    {
                        "embedding_profile_id": "service-dense-v1",
                        "profile_id": None,
                    }
                ),
                {"backend_profile_id_malformed"},
            ),
        ],
    )
    def test_invalid_profile_configuration_never_resolves_verified_default(
        self, examples, extra_config, expected_reason_codes
    ):
        value = _valid(examples, "role_sensitive_provider_phase")
        server = _discover(value)
        backend = {
            "id": 1,
            "name": "KIron",
            "provider": "ollama",
            "url": "http://kiron.test",
            "model": value["canonical_model_id"],
            "priority": 10,
            "extra_config": extra_config,
        }
        with patch(
            "kiara.retrieval.embedding_discovery.discover_server_contracts",
            return_value=server,
        ):
            diagnostic = discover_configured_embedding_backends([backend]).backends[0]
        assert diagnostic.state is ContractState.INCONSISTENT
        assert diagnostic.rag_verified is False
        assert diagnostic.resolution is None
        assert {issue.reason_code for issue in diagnostic.issues} == expected_reason_codes

    def test_missing_explicit_profile_does_not_fall_back_to_default(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        server = _discover(value)
        backend = {
            "id": 1,
            "name": "KIron",
            "provider": "ollama",
            "url": "http://kiron.test",
            "model": value["canonical_model_id"],
            "priority": 10,
            "extra_config": json.dumps({"profile_id": "does-not-exist"}),
        }
        with patch(
            "kiara.retrieval.embedding_discovery.discover_server_contracts",
            return_value=server,
        ):
            diagnostic = discover_configured_embedding_backends([backend]).backends[0]
        assert diagnostic.state is ContractState.MISSING
        assert diagnostic.rag_verified is False
        assert diagnostic.resolution.requested_profile_id == "does-not-exist"
        assert diagnostic.resolution.profile is None

    def test_no_explicit_profile_still_resolves_verified_default(self, examples):
        value = _valid(examples, "role_sensitive_provider_phase")
        server = _discover(value)
        backend = {
            "id": 1,
            "name": "KIron",
            "provider": "ollama",
            "url": "http://kiron.test",
            "model": value["canonical_model_id"],
            "priority": 10,
            "extra_config": "{}",
        }
        with patch(
            "kiara.retrieval.embedding_discovery.discover_server_contracts",
            return_value=server,
        ):
            diagnostic = discover_configured_embedding_backends([backend]).backends[0]
        assert diagnostic.state is ContractState.VALID
        assert diagnostic.rag_verified is True
        assert diagnostic.resolution.profile.profile_id == "service-dense-v1"

        inconsistent = replace(diagnostic, state=ContractState.INCONSISTENT)
        assert inconsistent.resolution.rag_verified is True
        assert inconsistent.rag_verified is False

    @pytest.mark.asyncio
    async def test_admin_api_is_additive_read_only_diagnostic(self):
        from kiara.web.api_datasource_features import list_embedding_contracts

        expected = {
            "contract": {"schema_version": 1, "schema_sha256": SCHEMA_SHA256},
            "servers": [],
            "backends": [],
        }
        inventory = MagicMock(spec=EmbeddingContractInventory)
        inventory.to_dict.return_value = expected
        with (
            patch("kiara.backend_selector.get_all_active", return_value=[{"id": 1}]) as active,
            patch(
                "kiara.retrieval.embedding_discovery.discover_configured_embedding_backends",
                return_value=inventory,
            ) as discover,
            patch("kiara.retrieval.vectorstore.VectorStore.list_collections") as vectors,
            patch("kiara.retrieval.embedding._ollama_embed") as embed,
        ):
            result = await list_embedding_contracts(user={"id": 1})
        assert result == expected
        active.assert_called_once_with("embedding")
        discover.assert_called_once()
        vectors.assert_not_called()
        embed.assert_not_called()

    def test_ui_contains_required_contract_diagnostics(self):
        source = (
            Path(__file__).parents[1] / "kiara" / "web" / "static" / "js" / "tab_ki_modelle.js"
        ).read_text(encoding="utf-8")
        assert "/api/admin/embedding-contracts" in source
        for field in (
            "schema_version",
            "canonical_model_id",
            "profile_id",
            "role_sensitive",
            "required",
            "index_compatibility_id",
            "query_compatibility_id",
            "blocking_reasons",
            "rag_verified",
        ):
            assert field in source
        assert "SharedUi.runtime.escapeHtml(reasons)" in source

    @pytest.mark.parametrize("failure_mode", ["malformed_2xx", "network", "non_2xx"])
    def test_optional_contract_diagnostic_failure_keeps_backend_crud_visible(self, failure_mode):
        rendered = _run_ki_modelle_diagnostic_failure(failure_mode)
        assert "Funktionierendes Backend" in rendered["html"]
        assert 'data-action="toggle-backend"' not in rendered["html"]
        assert 'data-change-action="toggle-backend"' in rendered["html"]
        assert 'data-action="open-edit-backend-modal"' in rendered["html"]
        assert 'data-action="delete-backend"' in rendered["html"]
        assert "Vertragsdiagnose konnte nicht geladen werden." in rendered["html"]
        assert "Backend-Daten konnten nicht geladen werden." not in rendered["html"]
        assert "Embedding" in rendered["category_html"]


class TestS5OperationsBasedPayloadRegression:
    _BACKEND = {
        "provider": "ollama",
        "url": "http://embed.test",
        "api_key": "",
        "base_url": "",
    }

    def test_dense_document_and_query_payload_roles_are_explicit(self, s4_guard_passthrough):
        from kiara.retrieval.embedding import embed_query, embed_texts

        with (
            patch("kiara.retrieval.embedding._resolve_backend", return_value=self._BACKEND),
            patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2),
            patch(
                "kiara.retrieval.embedding._ollama_embed",
                side_effect=[[[0.1, 0.2]], [[0.3, 0.4]]],
            ) as request,
        ):
            embed_texts({}, "nomic-embed-text", ["Dokument"])
            embed_query({}, "nomic-embed-text", "Suche")
        assert request.call_args_list[0].args == (
            "http://embed.test",
            "nomic-embed-text",
            ["Dokument"],
            "search_document",
        )
        assert request.call_args_list[1].args == (
            "http://embed.test",
            "nomic-embed-text",
            ["Suche"],
            "search_query",
        )

    def test_no_local_model_prefix_and_late_document_role(self, s4_guard_passthrough):
        from kiara.retrieval.embedding import embed_query, embed_texts_late

        with (
            patch("kiara.retrieval.embedding._resolve_backend", return_value=self._BACKEND),
            patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2),
            patch(
                "kiara.retrieval.embedding._ollama_embed", return_value=[[0.1, 0.2]]
            ) as dense_request,
        ):
            embed_query({}, "mxbai-embed-large", "Suche")
        assert dense_request.call_args.args[2] == ["Suche"]
        assert dense_request.call_args.args[3] == "search_query"

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        with (
            patch("kiara.retrieval.embedding._resolve_backend", return_value=self._BACKEND),
            patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2),
            patch("kiara.retrieval.embedding.httpx.post", return_value=response) as post,
        ):
            embed_texts_late({}, "nomic-embed-text", "Dokument", ["Chunk"], [(0, 5)])
        assert post.call_args.kwargs["json"] == {
            "model": "nomic-embed-text",
            "input_type": "search_document",
            "document": "Dokument",
            "chunks": [{"text": "Chunk", "char_start": 0, "char_end": 5}],
        }

    def test_colbert_payload_has_explicit_document_role(self, s4_guard_passthrough):
        from kiara.retrieval.colbert import embed_colbert

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"embeddings": [[[[0.1, 0.2]]]]}
        with (
            patch(
                "kiara.retrieval.colbert._resolve_colbert_url",
                return_value="http://embed.test",
            ),
            patch("kiara.retrieval.colbert.httpx.post", return_value=response) as post,
        ):
            embed_colbert(
                {},
                "colbert-xm",
                ["Text"],
                input_type="search_document",
                retries=0,
            )
        assert post.call_args.args[0] == "http://embed.test/api/embed_colbert"
        assert post.call_args.kwargs["json"] == {
            "model": "colbert-xm",
            "input": ["Text"],
            "input_type": "search_document",
        }
