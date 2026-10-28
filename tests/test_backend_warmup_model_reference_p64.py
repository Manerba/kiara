"""Worker warmup retains the exact backend reference through strict discovery."""

import copy
import json
from unittest.mock import Mock

import httpx
import pytest
from test_embedding_model_reference_p64 import HELLORD, NOMIC, captured
from test_index_compatibility_s4 import _server_discovery

from kiara import backend_selector, backend_warmup
from kiara.retrieval import embedding_discovery
from kiara.retrieval.embedding_roles import EmbeddingRoleContractError
from kiara.shared.constants import WARMUP_DUMMY_TEXT, WARMUP_KEEP_ALIVE


def warmup_case(monkeypatch, model, *, backend_model=None, provider="ollama"):
    raw, profile = captured(model, "dense")
    backend = {
        "model": model if backend_model is None else backend_model,
        "provider": provider,
        "url": "http://warmup-reference.invalid",
        "name": "captured backend",
        "extra_config": json.dumps({"embedding_profile_id": profile.profile_id}),
    }
    monkeypatch.setattr(
        backend_selector,
        "get_all_active",
        lambda category: [backend] if category == "embedding" else [],
    )
    server = _server_discovery(raw)
    monkeypatch.setattr(embedding_discovery, "discover_server_contracts", lambda *a, **kw: server)
    return backend, {"id": 1, "type": "local", "ds_ddu_enabled": 0, "ds_vlm_enabled": 0}


@pytest.mark.parametrize("model", [HELLORD, NOMIC])
def test_warmup_uses_exact_backend_reference_for_strict_discovery_and_http(monkeypatch, model):
    backend, datasource = warmup_case(monkeypatch, model)
    original = copy.deepcopy(backend)
    tasks = backend_warmup.build_warmup_tasks(
        datasource, backend_selector.normalize_model_name(model), {}
    )
    assert len(tasks) == 1
    requests = []

    def transport(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        monkeypatch.setattr(backend_warmup.httpx, "post", client.post)
        backend_warmup.BackendWarmer()._warm_embedding(tasks[0], 60)
    assert tasks[0].model == model
    assert requests == [
        (
            "/api/embed",
            {
                "model": model,
                "input": [WARMUP_DUMMY_TEXT],
                "input_type": "search_document",
                "keep_alive": WARMUP_KEEP_ALIVE,
            },
        )
    ]
    assert backend == original


@pytest.mark.parametrize("model", [HELLORD, NOMIC])
def test_warmup_unregistered_backend_spelling_still_blocks_before_transport(monkeypatch, model):
    _, datasource = warmup_case(monkeypatch, model, backend_model=model.swapcase())
    tasks = backend_warmup.build_warmup_tasks(
        datasource, backend_selector.normalize_model_name(model), {}
    )
    transport = Mock(side_effect=AssertionError("Unregistered reference reached embedding HTTP"))
    monkeypatch.setattr(backend_warmup.httpx, "post", transport)
    with pytest.raises(EmbeddingRoleContractError) as error:
        backend_warmup.BackendWarmer()._warm_embedding(tasks[0], 60)
    assert error.value.reason_code == "contract_model_not_found"
    transport.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "voyage"])
def test_cloud_warmup_selection_remains_empty(monkeypatch, provider):
    _, datasource = warmup_case(monkeypatch, NOMIC, provider=provider)
    assert backend_warmup.build_warmup_tasks(datasource, "nomic-embed-text", {}) == []
