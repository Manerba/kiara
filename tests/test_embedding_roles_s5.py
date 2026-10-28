"""S5-Vertragstests fuer operationsbasierte Embedding-Rollen."""

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kiara.retrieval.embedding_contract import ContractState, ProfileKind
from kiara.retrieval.embedding_roles import (
    SEARCH_DOCUMENT,
    SEARCH_QUERY,
    EmbeddingRequestProfile,
    emit_embedding_role_telemetry,
    request_profile_from_guard,
)
from kiara.retrieval.index_compatibility import (
    IndexCompatibilityError,
    IndexVectorKind,
    make_index_target,
)

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")


def _target():
    return make_index_target(
        {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
        collection_name="kb_s5_roles",
        vector_kind=IndexVectorKind.DENSE,
        requested_model="model-name-is-not-a-role",
    )


def _token(**overrides):
    values = {
        "state": ContractState.VALID,
        "canonical_model_id": "registry.example/model@sha256:canonical",
        "profile_id": "dense-v1",
        "profile_kind": "dense",
        "profile_endpoint": "/api/embed",
        "profile_supported_roles": (SEARCH_DOCUMENT, SEARCH_QUERY),
        "profile_role_sensitive": True,
        "profile_input_type_required": False,
        "profile_formatting_owner": "server",
        "index_compatibility_id": "sha256:" + "1" * 64,
        "query_compatibility_id": "sha256:" + "2" * 64,
    }
    values.update(overrides)
    return SimpleNamespace(
        operation="query_embedding",
        target=_target(),
        resolved=SimpleNamespace(**values),
    )


@pytest.mark.parametrize(
    ("overrides", "role", "reason_code"),
    [
        ({"state": ContractState.UNVERIFIED}, SEARCH_QUERY, "embedding_profile_unverified"),
        (
            {"profile_supported_roles": (SEARCH_DOCUMENT,)},
            SEARCH_QUERY,
            "embedding_role_unsupported",
        ),
        ({"profile_kind": "multi_vector"}, SEARCH_QUERY, "embedding_profile_kind_mismatch"),
        (
            {"profile_endpoint": "/api/embed_late"},
            SEARCH_QUERY,
            "embedding_profile_endpoint_mismatch",
        ),
        (
            {"profile_formatting_owner": "client"},
            SEARCH_QUERY,
            "embedding_formatting_owner_mismatch",
        ),
        (
            {"profile_role_sensitive": None},
            SEARCH_QUERY,
            "embedding_role_sensitivity_unverified",
        ),
        (
            {"query_compatibility_id": None},
            SEARCH_QUERY,
            "embedding_compatibility_id_missing",
        ),
        ({}, "passage", "embedding_role_invalid"),
    ],
)
def test_profile_role_mismatches_fail_closed(overrides, role, reason_code):
    with pytest.raises(IndexCompatibilityError) as caught:
        request_profile_from_guard(
            _token(**overrides),
            role=role,
            expected_kind=ProfileKind.DENSE,
            expected_endpoint="/api/embed",
        )

    assert caught.value.reason_code == reason_code


def test_role_independent_profile_accepts_both_canonical_noop_roles():
    token = _token(profile_role_sensitive=False, profile_input_type_required=False)

    document = request_profile_from_guard(
        token,
        role=SEARCH_DOCUMENT,
        expected_kind=ProfileKind.DENSE,
        expected_endpoint="/api/embed",
    )
    query = request_profile_from_guard(
        token,
        role=SEARCH_QUERY,
        expected_kind=ProfileKind.DENSE,
        expected_endpoint="/api/embed",
    )

    assert document.input_type == SEARCH_DOCUMENT
    assert query.input_type == SEARCH_QUERY
    assert document.role_sensitive is False
    assert query.role_sensitive is False


def test_query_compatibility_mismatch_stops_before_backend_request(monkeypatch):
    from kiara.retrieval import embedding, index_compatibility

    target = _target()
    error = IndexCompatibilityError(
        "index_query_compatibility_id_mismatch",
        "query_embedding",
        target,
        "query compatibility changed",
        stored_state="verified",
        current_state="valid",
    )

    @contextmanager
    def mismatch_guard(*_args, **_kwargs):
        raise error
        yield  # pragma: no cover - contextmanager form requires a generator

    monkeypatch.setattr(index_compatibility, "guard_index_for_active_job", mismatch_guard)
    monkeypatch.setattr(index_compatibility, "raw_qdrant_client", lambda _config: MagicMock())

    with (
        patch.object(embedding, "_resolve_backend") as backend,
        patch.object(embedding, "_get_ollama_http_client") as request,
        pytest.raises(IndexCompatibilityError) as caught,
    ):
        embedding.embed_query({}, target.requested_model, "sensitive query")

    assert caught.value is error
    backend.assert_not_called()
    request.assert_not_called()


def test_colbert_rejects_noncanonical_role_before_http_request():
    from kiara.retrieval.colbert import embed_colbert

    with (
        patch("kiara.retrieval.colbert.httpx.post") as request,
        pytest.raises(IndexCompatibilityError) as caught,
    ):
        embed_colbert({}, "colbert-model", ["document"], input_type="passage")

    assert caught.value.reason_code == "embedding_role_invalid"
    request.assert_not_called()


def test_role_telemetry_contains_contract_but_never_input(caplog):
    secret_input = "Kundengeheimnis Auftrag 4711"
    profile = EmbeddingRequestProfile(
        requested_model="alias",
        canonical_model_id="registry.example/model@sha256:canonical",
        profile_id="dense-v1",
        kind="dense",
        endpoint="/api/embed",
        input_type=SEARCH_QUERY,
        role_sensitive=True,
        index_compatibility_id="sha256:" + "1" * 64,
        query_compatibility_id="sha256:" + "2" * 64,
    )

    with caplog.at_level(logging.INFO, logger="kiara.retrieval.embedding_roles"):
        emit_embedding_role_telemetry(
            profile,
            operation="query_embedding",
            input_count=1,
        )

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "input_type=search_query" in rendered
    assert "profile_id=dense-v1" in rendered
    assert "canonical_model_id=registry.example/model@sha256:canonical" in rendered
    assert "compatibility_id=sha256:" in rendered
    assert secret_input not in rendered
