"""Logical index keys must not replace exact KIron backend model references."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from test_index_compatibility_s4 import _backend_connection, _server_discovery

from kiara.backend_selector import normalize_model_name
from kiara.retrieval import embedding
from kiara.retrieval import index_compatibility as contracts
from kiara.retrieval.embedding_contract import parse_capabilities
from kiara.retrieval.embedding_roles import EmbeddingRequestProfile

HELLORD = "hellord/e5-mistral-7b-instruct:Q4_0"
NOMIC = "nomic-embed-text:latest"


def captured(model, kind):
    path = Path(__file__).parent / "fixtures/p64-kiron-profiles-20260928.json"
    raw = json.loads(path.read_text())["contracts"][model]
    profile = next(
        item
        for item in parse_capabilities(raw).profiles
        if item.kind.value == kind and item.rag_verified
    )
    return raw, profile


def resolver_case(monkeypatch, model, kind, *, backend_model=None):
    raw, profile = captured(model, kind)
    config = {"vectordb": {"qdrant_url": "http://model-reference.invalid:6333"}}
    backend = {
        "id": 12,
        "provider": "ollama",
        "url": "http://kiron-reference.invalid:11434",
        "model": model if backend_model is None else backend_model,
        "extra_config": json.dumps({"embedding_profile_id": profile.profile_id}),
    }
    target = contracts.make_index_target(
        config,
        collection_name=embedding.collection_name_for_model(model),
        vector_kind=kind,
        requested_model=model,
    )
    discovery = _server_discovery(raw)
    monkeypatch.setattr(contracts, "discover_server_contracts", lambda *a, **kw: discovery)
    return (
        backend,
        target,
        profile,
        contracts.resolve_current_index_contract(
            _backend_connection(backend), config, target, force_refresh=True
        ),
    )


@pytest.mark.parametrize(
    "model,kind", [(HELLORD, "dense"), (NOMIC, "dense"), (NOMIC, "late_chunking")]
)
def test_official_backend_reference_preserves_profile_and_logical_identity(
    monkeypatch, model, kind
):
    backend, target, profile, resolved = resolver_case(monkeypatch, model, kind)
    assert resolved.verified, resolved.issues
    assert target.requested_model == normalize_model_name(model)
    assert resolved.target is target
    assert resolved.backend == backend
    assert resolved.backend["model"] == model
    assert resolved.canonical_model_id == model
    assert resolved.profile_id == profile.profile_id
    assert resolved.index_compatibility_id == profile.index_compatibility_id
    assert resolved.query_compatibility_id == profile.query_compatibility_id
    assert target.collection_name == embedding.collection_name_for_model(
        normalize_model_name(model)
    )


@pytest.mark.parametrize("model,kind", [(HELLORD, "dense"), (NOMIC, "late_chunking")])
def test_unregistered_backend_spelling_is_not_accepted_as_casefold_alias(monkeypatch, model, kind):
    invalid = model.swapcase()
    assert normalize_model_name(invalid) == normalize_model_name(model)
    _, _, _, resolved = resolver_case(monkeypatch, model, kind, backend_model=invalid)
    assert not resolved.verified
    assert resolved.reason_code == "contract_model_not_found"


@pytest.mark.parametrize(
    "model,kind,role",
    [
        (HELLORD, "dense", "search_document"),
        (HELLORD, "dense", "search_query"),
        (NOMIC, "dense", "search_document"),
        (NOMIC, "dense", "search_query"),
        (NOMIC, "late_chunking", "search_document"),
        (NOMIC, "late_chunking", "search_query"),
    ],
)
def test_http_payload_uses_exact_guard_backend_for_document_and_query(
    monkeypatch, model, kind, role
):
    # Isolate transport after admission; the resolver tests above separately
    # exercise real strict discovery using the unchanged captured contracts.
    _, profile = captured(model, kind)
    backend = {"model": model, "provider": "ollama", "url": "http://kiron-reference.invalid"}
    original_backend = copy.deepcopy(backend)
    guarded_backend = Mock(return_value=backend)
    monkeypatch.setattr(contracts, "exact_backend_from_guard", guarded_backend)
    fallback = Mock(side_effect=AssertionError("A guarded request must retain its backend"))
    monkeypatch.setattr(embedding, "get_embedding_backend_for_model", fallback)
    monkeypatch.setattr(embedding, "get_embedding_dimension", lambda *_args: profile.dimensions)
    request_profile = EmbeddingRequestProfile(
        requested_model=normalize_model_name(model),
        canonical_model_id=model,
        profile_id=profile.profile_id,
        kind=kind,
        endpoint=profile.endpoint,
        input_type=role,
        role_sensitive=profile.input_type.role_sensitive,
        index_compatibility_id=profile.index_compatibility_id,
        query_compatibility_id=profile.query_compatibility_id,
    )
    payloads = []
    vector = [0.0] * profile.dimensions

    def transport(request):
        payloads.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"embeddings": [vector]})

    with httpx.Client(base_url=backend["url"], transport=httpx.MockTransport(transport)) as client:
        monkeypatch.setattr(embedding, "_get_ollama_http_client", lambda _url: client)
        monkeypatch.setattr(embedding.httpx, "post", client.post)
        logical_model = normalize_model_name(model)
        if role == "search_query":
            result = embedding.embed_query.__wrapped__(
                {}, logical_model, "Sachliche Suchanfrage", _request_profile=request_profile
            )
            assert result == vector
        elif kind == "dense":
            result = embedding.embed_texts.__wrapped__(
                {}, logical_model, ["Dokument"], _request_profile=request_profile
            )
            assert result == [vector]
        else:
            result = embedding.embed_texts_late.__wrapped__(
                {},
                logical_model,
                "Dokument",
                ["Dokument"],
                [(0, 8)],
                _request_profile=request_profile,
            )
            assert result == [vector]
    assert len(payloads) == 1
    endpoint, payload = payloads[0]
    assert endpoint == profile.endpoint
    assert payload["model"] == model
    assert payload["input_type"] == role
    assert backend == original_backend
    guarded_backend.assert_called_once_with(normalize_model_name(model))
    fallback.assert_not_called()


def test_backend_reference_falls_back_only_when_row_has_no_model(monkeypatch):
    monkeypatch.setattr(
        contracts,
        "exact_backend_from_guard",
        lambda _model: {"provider": "ollama", "url": "http://kiron-reference.invalid"},
    )
    assert embedding._resolve_backend("nomic-embed-text")["model"] == "nomic-embed-text"


@pytest.mark.parametrize("provider", ["openai", "voyage"])
@pytest.mark.parametrize("role", ["search_document", "search_query"])
def test_cloud_provider_call_model_is_unchanged(monkeypatch, provider, role):
    backend = {
        "model": "BACKEND-Row-Reference",
        "provider": provider,
        "url": "https://cloud.invalid",
        "api_key": "test",
        "base_url": "https://cloud.invalid",
    }
    monkeypatch.setattr(embedding, "_resolve_backend", lambda _model: backend)
    monkeypatch.setattr(embedding, "get_embedding_dimension", lambda *_args: 2)
    monkeypatch.setattr(embedding, "emit_embedding_role_telemetry", lambda *a, **kw: None)
    client = object()
    monkeypatch.setattr(embedding, f"_get_{provider}_client", lambda *_args: client)
    call = Mock(return_value=[[0.1, 0.2]])
    monkeypatch.setattr(embedding, f"_embed_{provider}", call)
    request_profile = SimpleNamespace(kind="dense")
    if role == "search_document":
        embedding.embed_texts.__wrapped__(
            {}, "CALLER-Model:latest", ["Dokument"], _request_profile=request_profile
        )
    else:
        embedding.embed_query.__wrapped__(
            {}, "CALLER-Model:latest", "Frage", _request_profile=request_profile
        )
    assert call.call_args.args[:2] == (client, "caller-model")
