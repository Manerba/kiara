"""Pure regressions for the authorized KIron input-policy schema transition.

The fixtures are captured complete producer contracts. Their published IDs
are fixed expectations; this suite never regenerates them with consumer code.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from kiara.retrieval.embedding_contract import (
    SCHEMA_SHA256,
    ContractValidationError,
    parse_capabilities,
)

_NEW_SCHEMA_SHA256 = "d4d2698831e16bafe54dc03ffcb5b927933b9439111242e0d5997f0815dea477"
_OLD_SCHEMA_SHA256 = "2dafc62888439ef3be15d7ae4fac29d2718ee16ddcfb6f82f61bf87149b14264"
_HELLORD = "hellord/e5-mistral-7b-instruct:Q4_0"
_NOMIC = "nomic-embed-text:latest"


def _contract(model):
    fixture = Path(__file__).parent / "fixtures" / "p64-kiron-profiles-20260928.json"
    return json.loads(fixture.read_text(encoding="utf-8"))["contracts"][model]


def test_official_schema_and_historical_schema_have_pinned_bytes():
    schema_dir = Path(__file__).parents[1] / "kiara" / "retrieval" / "schemas"
    assert SCHEMA_SHA256 == _NEW_SCHEMA_SHA256
    for filename, expected in (
        ("kiron-embedding-capabilities-v1.schema.json", _NEW_SCHEMA_SHA256),
        ("kiron-embedding-capabilities-v1.pre-reject.schema.json", _OLD_SCHEMA_SHA256),
    ):
        assert hashlib.sha256((schema_dir / filename).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize(
    ("model", "profile_id", "policy", "dimension", "endpoint", "index_id", "query_id"),
    [
        (
            _HELLORD,
            "ollama-e5-mistral-dense-v1",
            ("none", "reject"),
            4096,
            "/api/embed",
            "sha256:1f92b3b8a653901de9059e8fa032bd33e85d85a0ccea391677f71ccb7cae2e19",
            "sha256:78273146de3d7ad5e3a24b319113a8ab32463a5bf04a41cc76a0b46e9da121ab",
        ),
        (
            _NOMIC,
            "kiron-nomic-dense-v1",
            ("right", "truncate"),
            768,
            "/api/embed",
            "sha256:bcd5ff914e77b0472a1ea5785c67eb93ed9d327870b4df0500b728070ad4f7f7",
            "sha256:d1768f20ad55fcaef6b3a5ddf8aa474e3da33f15d5e4e616f2fe55b9522d8bcf",
        ),
        (
            _NOMIC,
            "kiron-nomic-late-v1",
            ("right", "late_chunk_fallback"),
            768,
            "/api/embed_late",
            "sha256:dd9b890356d523fb89ce53913c58b5c03a37b8b4f15f1d9ff8458da124c94533",
            "sha256:15b064fcb667e5b3936ad98bd93cef87e9bb33511d2645054bc033566facc21a",
        ),
    ],
)
def test_verified_producer_profiles_preserve_published_ids(
    model, profile_id, policy, dimension, endpoint, index_id, query_id
):
    raw = _contract(model)
    original = copy.deepcopy(raw)
    parsed = parse_capabilities(raw)
    profile = parsed.profile_by_id(profile_id)
    assert profile is not None and profile.rag_verified
    assert (profile.max_input_tokens.truncation, profile.max_input_tokens.overflow) == policy
    assert profile.dimensions == dimension
    assert profile.endpoint == endpoint
    assert profile.index_compatibility_id == index_id
    assert profile.query_compatibility_id == query_id
    assert raw == original


@pytest.mark.parametrize(
    ("truncation", "overflow"),
    [
        ("right", "reject"),
        ("none", "truncate"),
        ("none", "late_chunk_fallback"),
        ("unknown", "reject"),
        ("none", "unknown"),
        ("unknown", "unknown"),
        ("left", "reject"),
        ("none", "discard"),
    ],
)
def test_verified_invalid_policies_fail_before_compatibility_id_check(truncation, overflow):
    raw = _contract(_HELLORD)
    raw["profiles"][0]["max_input_tokens"].update(truncation=truncation, overflow=overflow)
    with pytest.raises(ContractValidationError) as caught:
        parse_capabilities(raw)
    assert caught.value.issue.reason_code in {
        "contract_schema_invalid",
        "contract_invariant_invalid",
    }
    assert caught.value.issue.field_path.startswith(
        "/kiron_capabilities/profiles/0/max_input_tokens"
    )


@pytest.mark.parametrize("field", ["index_compatibility_id", "query_compatibility_id"])
def test_accepted_reject_policy_still_requires_unchanged_compatibility_ids(field):
    raw = _contract(_HELLORD)
    raw["profiles"][0][field] = "sha256:" + "0" * 64
    with pytest.raises(ContractValidationError) as caught:
        parse_capabilities(raw)
    assert caught.value.issue.reason_code == "contract_compatibility_id_mismatch"
    assert caught.value.issue.field_path.endswith("/" + field)


def test_unverified_nomic_profile_is_not_promoted_by_schema_expansion():
    profile = parse_capabilities(_contract(_NOMIC)).profile_by_id("ollama-nomic-dense-v1")
    assert profile is not None and not profile.rag_verified
    assert profile.index_compatibility_id is None
    assert profile.query_compatibility_id is None
    assert profile.verification.blocking_reasons


def test_hellord_has_only_its_published_dense_profile():
    parsed = parse_capabilities(_contract(_HELLORD))
    assert [(profile.profile_id, profile.kind.value) for profile in parsed.profiles] == [
        ("ollama-e5-mistral-dense-v1", "dense")
    ]
    profile = parsed.profiles[0]
    assert profile.max_input_tokens.by_role.search_document == 4096
    assert profile.max_input_tokens.by_role.search_query == 4096
