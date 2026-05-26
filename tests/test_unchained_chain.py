"""Tests fuer kiara.generation.unchained_chain."""

from unittest.mock import MagicMock

import pytest

from kiara.generation.backends.base import LLMBackend
from kiara.generation.unchained_chain import UnchainedChain
from kiara.shared.constants import DEFAULT_LLM_NUM_CTX, DEFAULT_UNCHAINED_TEMPERATURE


def _make_mock_backend(provider="ollama", stream_chunks=None, usage=None):
    """Erzeugt ein Mock-LLMBackend fuer Tests."""
    mock = MagicMock(spec=LLMBackend)
    mock.provider = provider
    mock.model = "test-model"
    chunks = stream_chunks if stream_chunks is not None else ["Test", "-", "Antwort"]
    mock.generate_stream.return_value = iter(chunks)
    mock.last_usage = usage or {
        "prompt_tokens": 10,
        "response_tokens": 5,
        "model": "test-model",
        "num_ctx": 4096,
        "total_duration_ms": 100,
    }
    return mock


@pytest.fixture
def base_config():
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
        },
        "chat": {
            "unchained": {
                "temperature": 0.8,
                "system_prompt": "",
            },
        },
    }


class TestUnchainedChain:
    def test_init_defaults(self, base_config):
        backend = _make_mock_backend()
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert chain._temperature == 0.8
        assert chain._system_prompt == "Test-System-Prompt"
        assert chain.backend.model == "test-model"

    def test_init_no_system_prompt(self, base_config):
        backend = _make_mock_backend()
        chain = UnchainedChain(base_config, backend=backend, system_prompt=None)
        assert chain._system_prompt == ""

    def test_query_stream_yields_chunks(self, base_config):
        backend = _make_mock_backend(stream_chunks=["Hallo", " Welt", "!"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        chunks = list(chain.query_stream("Was ist Python?"))

        assert chunks == ["Hallo", " Welt", "!"]
        backend.generate_stream.assert_called_once()

        call_args = backend.generate_stream.call_args
        # System-Prompt wird jetzt gehaertet (Praeambel + Original)
        assert "SICHERHEITSHINWEIS" in call_args.args[0]
        assert "Test-System-Prompt" in call_args.args[0]
        assert "Was ist Python?" in call_args.args[1]  # user_message

    def test_query_stream_empty_response(self, base_config):
        backend = _make_mock_backend(stream_chunks=[])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        chunks = list(chain.query_stream("Test"))
        assert chunks == []

    def test_no_retriever_dependency(self, base_config):
        """UnchainedChain hat keinen Retriever — kein RAG-Kontext."""
        backend = _make_mock_backend()
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert not hasattr(chain, "_retriever")

    def test_temperature_from_config(self, base_config):
        backend = _make_mock_backend(stream_chunks=["ok"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        list(chain.query_stream("Test"))

        call_kwargs = backend.generate_stream.call_args.kwargs
        assert call_kwargs["temperature"] == 0.8


class TestUnchainedChainPrompt:
    """Tests fuer den Prompt-Aufbau."""

    def test_prompt_contains_question(self, base_config):
        """Prompt enthaelt die Frage."""
        backend = _make_mock_backend(stream_chunks=["ok"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        list(chain.query_stream("Erklaere DNS"))

        # user_message ist das zweite positional arg
        user_message = backend.generate_stream.call_args.args[1]
        assert "Erklaere DNS" in user_message

    def test_custom_system_prompt_used_in_llm_call(self, base_config):
        """Konfigurierter System-Prompt wird tatsaechlich im Backend-Call genutzt."""
        backend = _make_mock_backend(stream_chunks=["Arr!"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Du bist ein Pirat.")
        list(chain.query_stream("Hallo"))

        system_prompt = backend.generate_stream.call_args.args[0]
        # System-Prompt wird jetzt gehaertet (Praeambel + Original)
        assert "SICHERHEITSHINWEIS" in system_prompt
        assert "Du bist ein Pirat." in system_prompt

    def test_prompt_no_context_section(self, base_config):
        """Unchained-Prompt darf keinen RAG-Kontext enthalten."""
        backend = _make_mock_backend(stream_chunks=["ok"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        list(chain.query_stream("Test"))

        user_message = backend.generate_stream.call_args.args[1]
        assert "Kontext" not in user_message
        assert "Quelle" not in user_message


class TestUnchainedChainConfigFallbacks:
    """Tests fuer Config-Defaults und fehlende Sektionen."""

    def test_missing_chat_section_uses_defaults(self):
        """Fehlende chat-Sektion → Konstanten-Defaults."""
        config = {
            "ollama": {
                "url": "http://localhost:11434",
                "llm_url": "",
                "embedding_url": "",
                "llm_model": "m",
                "embedding_model": "e",
            },
        }
        backend = _make_mock_backend()
        chain = UnchainedChain(config, backend=backend, system_prompt=None)
        assert chain._temperature == DEFAULT_UNCHAINED_TEMPERATURE
        assert chain._system_prompt == ""

    def test_missing_unchained_section_uses_defaults(self):
        """Vorhandene chat-Sektion ohne 'unchained' Key."""
        config = {
            "ollama": {
                "url": "http://localhost:11434",
                "llm_url": "",
                "embedding_url": "",
                "llm_model": "m",
                "embedding_model": "e",
            },
            "chat": {},
        }
        backend = _make_mock_backend()
        chain = UnchainedChain(config, backend=backend, system_prompt=None)
        assert chain._temperature == DEFAULT_UNCHAINED_TEMPERATURE
        assert chain._system_prompt == ""

    def test_num_ctx_passed_for_ollama_backend(self, base_config):
        """num_ctx wird als kwarg an Ollama-Backend weitergereicht."""
        backend = _make_mock_backend(provider="ollama", stream_chunks=["ok"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        list(chain.query_stream("Test"))

        call_kwargs = backend.generate_stream.call_args.kwargs
        assert call_kwargs["num_ctx"] == DEFAULT_LLM_NUM_CTX

    def test_num_ctx_not_passed_for_cloud_backend(self, base_config):
        """num_ctx wird NICHT an Cloud-Backends weitergereicht."""
        backend = _make_mock_backend(provider="anthropic", stream_chunks=["ok"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        list(chain.query_stream("Test"))

        call_kwargs = backend.generate_stream.call_args.kwargs
        assert "num_ctx" not in call_kwargs

    def test_model_from_backend(self, base_config):
        """Model wird korrekt aus dem Backend gelesen."""
        backend = _make_mock_backend()
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert chain.backend.model == "test-model"


class TestUnchainedChainStreamBehavior:
    """Tests fuer Streaming-Edge-Cases."""

    def test_all_chunks_yielded(self, base_config):
        """Alle Chunks vom Backend werden durchgereicht."""
        backend = _make_mock_backend(stream_chunks=["A", "B"])
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        chunks = list(chain.query_stream("Test"))
        assert chunks == ["A", "B"]

    def test_long_streaming_response(self, base_config):
        """Viele kleine Chunks werden alle korrekt durchgereicht."""
        many_chunks = [f"word{i} " for i in range(50)]
        backend = _make_mock_backend(stream_chunks=many_chunks)
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        result = list(chain.query_stream("Lang"))
        assert len(result) == 50
        assert result[0] == "word0 "
        assert result[49] == "word49 "

    def test_generate_stream_called_with_correct_model(self, base_config):
        """Das konfigurierte Modell ist im Backend gesetzt."""
        backend = _make_mock_backend()
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert chain.backend.model == "test-model"


class TestUnchainedChainRouteAndUsage:
    """Tests fuer route_used und last_usage Delegation."""

    def test_route_local_for_ollama(self, base_config):
        backend = _make_mock_backend(provider="ollama")
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert chain.route_used == "local"

    def test_route_cloud_for_anthropic(self, base_config):
        backend = _make_mock_backend(provider="anthropic")
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert chain.route_used == "cloud"

    def test_route_cloud_for_openai(self, base_config):
        backend = _make_mock_backend(provider="openai")
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert chain.route_used == "cloud"

    def test_last_usage_delegates_to_backend(self, base_config):
        usage = {
            "prompt_tokens": 42,
            "response_tokens": 7,
            "model": "m",
            "num_ctx": 8192,
            "total_duration_ms": 200,
        }
        backend = _make_mock_backend(usage=usage)
        chain = UnchainedChain(base_config, backend=backend, system_prompt="Test-System-Prompt")
        assert chain.last_usage == usage
        assert chain.last_usage["prompt_tokens"] == 42
