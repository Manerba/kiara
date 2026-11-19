"""Feste Ollama-Runner-Optionen je LLM-Backend (ki_backends.extra_config.runner_options)."""

import json
from unittest.mock import MagicMock, patch

from kiara.generation.adapters.ollama import OllamaToolAdapter
from kiara.generation.backends.ollama import OllamaBackend
from kiara.generation.runner_options import fixed_runner_options

PROFILE = {"num_ctx": 65536, "num_batch": 128, "num_thread": 4, "num_gpu": 65}


def _backends(extra):
    return [
        {"provider": "ollama", "url": "http://kiron:11434/", "model": "big", "extra_config": extra},
        {"provider": "ollama", "url": "http://kiron:11434", "model": "small", "extra_config": "{}"},
    ]


def _response():
    return {"message": {"content": "OK"}, "done": True, "prompt_eval_count": 1, "eval_count": 1}


@patch("kiara.backend_selector.get_all_active")
def test_options_match_url_and_model(mock_active):
    mock_active.return_value = _backends(json.dumps({"num_ctx": 40960, "runner_options": PROFILE}))
    assert fixed_runner_options("http://kiron:11434", "big") == PROFILE
    assert fixed_runner_options("http://kiron:11434", "small") == {}
    assert fixed_runner_options("http://other:11434", "big") == {}


@patch("kiara.backend_selector.get_all_active")
def test_invalid_values_are_ignored(mock_active):
    mock_active.return_value = _backends(
        {"runner_options": {"num_ctx": "65536", "num_gpu": True, "num_batch": 128, "seed": 1}}
    )
    assert fixed_runner_options("http://kiron:11434", "big") == {"num_batch": 128}


@patch("kiara.backend_selector.get_all_active", side_effect=RuntimeError("db down"))
def test_lookup_failure_falls_back_to_call_options(_mock_active):
    assert fixed_runner_options("http://kiron:11434", "big") == {}


@patch("kiara.generation.backends.ollama.fixed_runner_options", return_value=PROFILE)
@patch("kiara.generation.backends.ollama._import_httpx")
@patch("kiara.generation.backends.ollama._import_ollama")
def test_backend_sends_fixed_options_instead_of_call_num_ctx(mock_ollama, _httpx, _fixed):
    client = MagicMock()
    mock_ollama.return_value.Client.return_value = client
    client.chat.return_value = _response()

    backend = OllamaBackend(url="http://kiron:11434", model="big", num_ctx=8192)
    backend.generate("System", "User", num_ctx=2048, timeout=30)
    backend.generate("System", "User")

    for call in client.chat.call_args_list:
        assert {k: call.kwargs["options"][k] for k in PROFILE} == PROFILE
    assert backend.last_usage["num_ctx"] == 65536


@patch("kiara.generation.adapters.ollama.fixed_runner_options", return_value=PROFILE)
def test_agent_adapter_sends_fixed_options(_fixed):
    client = MagicMock()
    with patch("ollama.Client", return_value=client):
        adapter = OllamaToolAdapter({"url": "http://kiron:11434", "model": "big"}, {})
    assert adapter._num_ctx == 65536
    assert adapter._runner_options == PROFILE
