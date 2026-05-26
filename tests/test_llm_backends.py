"""Umfassende Unit-Tests fuer das LLM-Backend-Abstraction-Modul (Sprint 1).

Kein Netzwerk, kein Ollama, kein Cloud-API-Zugriff — alles gemockt.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from kiara.generation.backends.anonymizing import AnonymizingBackend
from kiara.generation.backends.anthropic import AnthropicBackend
from kiara.generation.backends.base import LLMBackend, build_usage_estimated
from kiara.generation.backends.factory import (
    create_backend,
    wrap_with_anonymization,
)
from kiara.generation.backends.ollama import OllamaBackend, build_usage_from_ollama
from kiara.generation.backends.openai import OpenAIBackend

# ─── Helpers ───────────────────────────────────────────────────────────


def _make_ollama_stubs():
    """Erzeugt Stub-Module fuer ollama und httpx."""
    mock_httpx = MagicMock()
    mock_httpx.Timeout.return_value = "timeout-obj"

    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_ollama.Client.return_value = mock_client

    return mock_ollama, mock_httpx, mock_client


def _make_anthropic_stubs():
    """Erzeugt Stub-Modul fuer anthropic.

    BadRequestError wird als echte Exception-Klasse hinterlegt, damit
    isinstance-Pruefungen im Backend (Thinking-Fallback) funktionieren.
    """
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    mock_anthropic.BadRequestError = type("BadRequestError", (Exception,), {})
    return mock_anthropic, mock_client


def _make_openai_stubs():
    """Erzeugt Stub-Modul fuer openai."""
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai, mock_client


def _make_anonymizer_mock(clean_text="anon_text", mapping=None, stats=None):
    """Erzeugt einen Anonymizer-Mock mit korrekter API-Konvention."""
    if mapping is None:
        mapping = {}
    if stats is None:
        stats = {"total": 0}
    mock = MagicMock()
    mock.anonymize.return_value = clean_text
    mock.last_result = SimpleNamespace(
        clean_text=clean_text,
        mapping=mapping,
        stats=stats,
    )
    mock.deanonymize.side_effect = lambda text: text
    return mock


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 1: LLMBackend ABC
# ═══════════════════════════════════════════════════════════════════════


class TestLLMBackendABC:
    def test_llm_backend_is_abstract(self):
        """LLMBackend kann nicht direkt instanziiert werden."""
        with pytest.raises(TypeError):
            LLMBackend()

    def test_llm_backend_last_usage_default(self):
        """Default-Implementation von last_usage gibt {} zurueck."""

        class MinimalBackend(LLMBackend):
            def generate(self, system_prompt, user_message, **kwargs):
                return ""

            def generate_stream(self, system_prompt, user_message, **kwargs):
                yield ""

            @property
            def provider(self):
                return "test"

            @property
            def model(self):
                return "test-model"

        instance = MinimalBackend()
        assert instance.last_usage == {}


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 2: OllamaBackend
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaBackend:
    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_provider_and_model(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, _ = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        assert backend.provider == "ollama"
        assert backend.model == "qwen3:8b"

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = {
            "message": {"content": "Antwort"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
            "total_duration": 2_000_000_000,
        }

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        result = backend.generate("System", "User")

        assert result == "Antwort"
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "qwen3:8b"
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User"},
        ]
        assert call_kwargs["stream"] is False
        assert backend.last_usage["prompt_tokens"] == 10
        assert backend.last_usage["response_tokens"] == 5

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_strips_think_block(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = {
            "message": {"content": "<think>Plan</think>Antwort"},
            "done": True,
        }

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        result = backend.generate("System", "User")
        assert result == "Antwort"

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_empty_system_prompt(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = {"message": {"content": "OK"}, "done": True}

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        backend.generate("", "User")

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "User"}]

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_none_content(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = {"message": {"content": None}, "done": True}

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        result = backend.generate("System", "User")
        assert result == ""

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_stream(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = iter(
            [
                {"message": {"content": "Hallo "}, "done": False},
                {
                    "message": {"content": "Welt"},
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                    "total_duration": 1_000_000_000,
                },
            ]
        )

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        chunks = list(backend.generate_stream("System", "User"))
        assert chunks == ["Hallo ", "Welt"]
        assert backend.last_usage["prompt_tokens"] == 10
        assert backend.last_usage["response_tokens"] == 5

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_stream_with_think(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = iter(
            [
                {"message": {"content": "", "thinking": "Hmm..."}, "done": False},
                {"message": {"content": "", "thinking": "Also..."}, "done": False},
                {
                    "message": {"content": "Die Antwort"},
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                    "total_duration": 1_000_000_000,
                },
            ]
        )

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        chunks = list(backend.generate_stream("System", "User", think=True))
        assert chunks == ["<think>", "Hmm...", "Also...", "</think>", "Die Antwort"]

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_stream_think_close_on_done(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = iter(
            [
                {"message": {"content": "", "thinking": "Hmm..."}, "done": False},
                {
                    "message": {"content": "", "thinking": "Fertig"},
                    "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                    "total_duration": 500_000_000,
                },
            ]
        )

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        chunks = list(backend.generate_stream("System", "User", think=True))
        assert chunks[-1] == "</think>"

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_kwargs_override(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = {"message": {"content": "OK"}, "done": True}

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        backend.generate("System", "User", temperature=0.9, num_ctx=16384)

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.9
        assert call_kwargs["options"]["num_ctx"] == 16384

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_per_call_timeout(self, mock_imp_ollama, mock_imp_httpx):
        """generate(timeout=...) baut einen kurzlebigen Client mit hartem Timeout."""
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx
        mock_client.chat.return_value = {"message": {"content": "OK"}, "done": True}

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        backend.generate("System", "User", timeout=2.5)

        assert mock_ollama.Client.call_count == 2
        assert mock_httpx.Timeout.call_args_list[-1].kwargs == {
            "connect": 2.5,
            "read": 2.5,
            "write": 2.5,
            "pool": 2.5,
        }

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_generate_think_kwarg(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_client.chat.return_value = {"message": {"content": "OK"}, "done": True}

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")

        # think=True soll weitergereicht werden
        backend.generate("System", "User", think=True)
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["think"] is True

        # think=None soll NICHT weitergereicht werden
        backend.generate("System", "User")
        call_kwargs = mock_client.chat.call_args[1]
        assert "think" not in call_kwargs

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_ollama_stream_close(self, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "message": {"content": "OK"},
                        "done": True,
                        "prompt_eval_count": 0,
                        "eval_count": 0,
                        "total_duration": 0,
                    },
                ]
            )
        )
        mock_stream.close = MagicMock()
        mock_client.chat.return_value = mock_stream

        backend = OllamaBackend(url="http://localhost:11434", model="qwen3:8b")
        list(backend.generate_stream("System", "User"))
        mock_stream.close.assert_called_once()

    @patch("kiara.generation.backends.ollama.import_module")
    def test_ollama_missing_dependency(self, mock_import_module):
        def side_effect(name):
            if name == "ollama":
                raise ImportError("No module named 'ollama'")
            return MagicMock()

        mock_import_module.side_effect = side_effect

        with pytest.raises(RuntimeError, match="Dependency"):
            OllamaBackend(url="http://localhost:11434", model="qwen3:8b")


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 3: AnthropicBackend
# ═══════════════════════════════════════════════════════════════════════


class TestAnthropicBackend:
    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_provider_and_model(self, mock_imp):
        mock_anthropic, _ = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        assert backend.provider == "anthropic"
        assert backend.model == "claude-sonnet-4-6"

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="Claude-Antwort")]
        resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = resp

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        result = backend.generate("System-Prompt", "User-Message")

        assert result == "Claude-Antwort"
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "System-Prompt"
        assert call_kwargs["messages"] == [{"role": "user", "content": "User-Message"}]

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_joins_text_blocks(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        resp = MagicMock()
        resp.content = [
            MagicMock(type="text", text="Teil 1"),
            MagicMock(type="text", text="Teil 2"),
        ]
        resp.usage = MagicMock(input_tokens=50, output_tokens=30)
        mock_client.messages.create.return_value = resp

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        result = backend.generate("System", "User")
        assert result == "Teil 1\nTeil 2"

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_with_think(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        resp = MagicMock()
        resp.content = [
            MagicMock(type="thinking", thinking="Plan"),
            MagicMock(type="text", text="Claude-Antwort"),
        ]
        resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = resp

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        result = backend.generate("System", "User", think=True)

        assert result.startswith("<think>Plan</think>")
        assert "Claude-Antwort" in result

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive"}
        assert "temperature" not in call_kwargs

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_old_model_uses_budgeted_thinking(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="Claude-Antwort")]
        resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = resp

        backend = AnthropicBackend(api_key="sk-test", model="claude-3-7-sonnet-20250219")
        backend.generate("System", "User", think=True)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }
        assert call_kwargs["max_tokens"] == 4096
        assert "temperature" not in call_kwargs

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_think_fallback(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        # Erster Call wirft thinking-Fehler (BadRequestError), zweiter liefert Ergebnis
        resp_ok = MagicMock()
        resp_ok.content = [MagicMock(type="text", text="Claude-Antwort")]
        resp_ok.usage = MagicMock(input_tokens=50, output_tokens=30)
        mock_client.messages.create.side_effect = [
            mock_anthropic.BadRequestError("thinking is not supported"),
            resp_ok,
        ]

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        result = backend.generate("System", "User", think=True)

        assert result == "Claude-Antwort"
        assert mock_client.messages.create.call_count == 2
        # Zweiter Call hat temperature aber kein thinking
        second_call = mock_client.messages.create.call_args_list[1][1]
        assert "thinking" not in second_call
        assert "temperature" in second_call

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_empty_response(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        resp = MagicMock()
        resp.content = []
        resp.usage = MagicMock(input_tokens=50, output_tokens=0)
        mock_client.messages.create.return_value = resp

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        result = backend.generate("System", "User")
        assert result == ""

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_stream(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        # Mock context manager fuer messages.stream()
        mock_stream_ctx = MagicMock()
        mock_stream_obj = MagicMock()
        mock_stream_obj.text_stream = iter(["Chunk1", "Chunk2"])
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_obj)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream_ctx

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        chunks = list(backend.generate_stream("System", "User"))
        assert chunks == ["Chunk1", "Chunk2"]

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_stream_with_think(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        # Events fuer thinking + text
        events = [
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="thinking"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(thinking="Plan-Text"),
            ),
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="text"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="Antwort-Text"),
            ),
        ]

        mock_stream_ctx = MagicMock()
        mock_stream_obj = MagicMock()
        mock_stream_obj.__iter__ = MagicMock(return_value=iter(events))
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_obj)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream_ctx

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        chunks = list(backend.generate_stream("System", "User", think=True))
        assert chunks == ["<think>", "Plan-Text", "</think>", "Antwort-Text"]

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_think_fallback(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        # Erster Stream-Call wirft thinking-Fehler (BadRequestError)
        mock_stream_err = MagicMock()
        mock_stream_err.__enter__ = MagicMock(
            side_effect=mock_anthropic.BadRequestError("thinking budget exceeded")
        )
        mock_stream_err.__exit__ = MagicMock(return_value=False)

        # Zweiter Call (Fallback ohne think) liefert Text
        mock_stream_ok = MagicMock()
        mock_stream_obj = MagicMock()
        mock_stream_obj.text_stream = iter(["Fallback-Antwort"])
        mock_stream_ok.__enter__ = MagicMock(return_value=mock_stream_obj)
        mock_stream_ok.__exit__ = MagicMock(return_value=False)

        mock_client.messages.stream.side_effect = [mock_stream_err, mock_stream_ok]

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        chunks = list(backend.generate_stream("System", "User", think=True))
        assert "Fallback-Antwort" in chunks

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_think_no_retry_after_partial_output(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        # Stream yieldet erst einen Text-Chunk, dann Fehler
        events = [
            SimpleNamespace(
                type="content_block_start",
                content_block=SimpleNamespace(type="text"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="Partial"),
            ),
        ]

        mock_stream_ctx = MagicMock()
        mock_stream_obj = MagicMock()

        # Simuliert: nach Events kommt BadRequestError (thinking-bezogen)
        def iterate_then_raise():
            yield from events
            raise mock_anthropic.BadRequestError("thinking budget exceeded")

        mock_stream_obj.__iter__ = MagicMock(side_effect=lambda: iterate_then_raise())
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_obj)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream_ctx

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        with pytest.raises(Exception, match="thinking budget"):
            list(backend.generate_stream("System", "User", think=True))

        # Nur ein stream-Call (kein Retry)
        assert mock_client.messages.stream.call_count == 1

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_generate_kwargs_override(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="OK")]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = resp

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        backend.generate("System", "User", max_tokens=8192, temperature=0.9)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 8192
        assert call_kwargs["temperature"] == 0.9

    @patch("kiara.generation.backends.anthropic.import_module")
    def test_anthropic_missing_dependency(self, mock_import_module):
        mock_import_module.side_effect = ImportError("No module named 'anthropic'")
        with pytest.raises(RuntimeError, match="Dependency"):
            AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_usage_tracking(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="OK")]
        resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = resp

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        backend.generate("System", "User")

        assert backend.last_usage["model"] == "claude-sonnet-4-6"
        assert backend.last_usage["prompt_tokens"] == 100
        assert backend.last_usage["response_tokens"] == 50

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_anthropic_stream_usage_fallback(self, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        mock_stream_ctx = MagicMock()
        mock_stream_obj = MagicMock()
        mock_stream_obj.text_stream = iter(["Ein langer Text"])
        # get_final_message() raises so fallback estimation is triggered
        mock_stream_obj.get_final_message.side_effect = Exception("no usage")
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_obj)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_client.messages.stream.return_value = mock_stream_ctx

        backend = AnthropicBackend(api_key="sk-test", model="claude-sonnet-4-6")
        list(backend.generate_stream("System", "User"))

        assert backend.last_usage["prompt_tokens"] > 0
        assert backend.last_usage["response_tokens"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 4: OpenAIBackend
# ═══════════════════════════════════════════════════════════════════════


class TestOpenAIBackend:
    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_provider_and_model(self, mock_imp):
        mock_openai, _ = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        assert backend.provider == "openai"
        assert backend.model == "gpt-4o-mini"

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_generate(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="GPT-Antwort"))]
        resp.usage = MagicMock(prompt_tokens=80, completion_tokens=20)
        mock_client.chat.completions.create.return_value = resp

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        result = backend.generate("System", "User")

        assert result == "GPT-Antwort"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["max_tokens"] == 4096
        assert "max_completion_tokens" not in call_kwargs

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_o_series_uses_max_completion_tokens(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="GPT-Antwort"))]
        resp.usage = MagicMock(prompt_tokens=80, completion_tokens=20)
        mock_client.chat.completions.create.return_value = resp

        backend = OpenAIBackend(api_key="sk-test", model="o3-mini", max_tokens=1234)
        backend.generate("System", "User")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 1234
        assert "max_tokens" not in call_kwargs

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_generate_empty_choices(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        resp = MagicMock()
        resp.choices = []
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=0)
        mock_client.chat.completions.create.return_value = resp

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        result = backend.generate("System", "User")
        assert result == ""

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_generate_none_content(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=None))]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=0)
        mock_client.chat.completions.create.return_value = resp

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        result = backend.generate("System", "User")
        assert result == ""

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_generate_stream(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hallo "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Welt"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        result = list(backend.generate_stream("System", "User"))
        assert result == ["Hallo ", "Welt"]

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_generate_stream_skip_none_content(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hallo "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Welt"))]),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        result = list(backend.generate_stream("System", "User"))
        assert result == ["Hallo ", "Welt"]

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_usage_tracking(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="OK"))]
        resp.usage = MagicMock(prompt_tokens=80, completion_tokens=20)
        mock_client.chat.completions.create.return_value = resp

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        backend.generate("System", "User")

        assert backend.last_usage["prompt_tokens"] == 80
        assert backend.last_usage["response_tokens"] == 20

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_stream_usage_fallback(self, mock_imp):
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        chunk = MagicMock(choices=[MagicMock(delta=MagicMock(content="Ein langer Text"))])
        chunk.usage = None  # No real usage data → triggers estimation fallback
        mock_client.chat.completions.create.return_value = iter([chunk])

        backend = OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")
        list(backend.generate_stream("System", "User"))

        assert backend.last_usage["prompt_tokens"] > 0
        assert backend.last_usage["response_tokens"] > 0

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_base_url(self, mock_imp):
        mock_openai, _ = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        OpenAIBackend(api_key="sk-test", model="gpt-4o-mini", base_url="https://custom.endpoint/v1")
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["base_url"] == "https://custom.endpoint/v1"

    @patch("kiara.generation.backends.openai._import_openai")
    def test_openai_base_url_empty_becomes_none(self, mock_imp):
        mock_openai, _ = _make_openai_stubs()
        mock_imp.return_value = mock_openai

        OpenAIBackend(api_key="sk-test", model="gpt-4o-mini", base_url="")
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["base_url"] is None

    @patch("kiara.generation.backends.openai.import_module")
    def test_openai_missing_dependency(self, mock_import_module):
        mock_import_module.side_effect = ImportError("No module named 'openai'")
        with pytest.raises(RuntimeError, match="Dependency"):
            OpenAIBackend(api_key="sk-test", model="gpt-4o-mini")


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 5: AnonymizingBackend
# ═══════════════════════════════════════════════════════════════════════


class TestAnonymizingBackend:
    def test_anonymizing_delegates_provider_model(self):
        inner = MagicMock(spec=LLMBackend)
        inner.provider = "anthropic"
        inner.model = "claude"
        anon = _make_anonymizer_mock()

        wrapper = AnonymizingBackend(inner, anon)
        assert wrapper.provider == "anthropic"
        assert wrapper.model == "claude"

    def test_anonymizing_generate(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate.return_value = "Hallo [Person_001]"

        anon = _make_anonymizer_mock(
            clean_text="[Person_001]",
            mapping={"[Person_001]": "Herr Mueller"},
            stats={"total": 1},
        )
        anon.deanonymize.side_effect = lambda text: text.replace("[Person_001]", "Herr Mueller")

        wrapper = AnonymizingBackend(inner, anon)
        result = wrapper.generate("System", "Herr Mueller")

        assert result == "Hallo Herr Mueller"
        # Inner wurde mit anonymisierter Message aufgerufen
        inner.generate.assert_called_once_with("System", "[Person_001]")
        # System-Prompt wurde NICHT anonymisiert
        anon.anonymize.assert_called_once_with("Herr Mueller")

    def test_anonymizing_generate_stream(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate_stream.return_value = iter(["Hallo ", "[Person_001]"])

        anon = _make_anonymizer_mock(
            clean_text="[Person_001]",
            mapping={"[Person_001]": "Herr Mueller"},
            stats={"total": 1},
        )
        anon.deanonymize.side_effect = lambda text: text.replace("[Person_001]", "Herr Mueller")

        wrapper = AnonymizingBackend(inner, anon)
        chunks = list(wrapper.generate_stream("System", "Herr Mueller"))

        assert chunks == ["Hallo Herr Mueller"]

    def test_anonymizing_generate_stream_split_placeholder(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate_stream.return_value = iter(["Hallo [Person", "_001]"])

        anon = _make_anonymizer_mock(
            clean_text="[Person_001]",
            mapping={"[Person_001]": "Herr Mueller"},
            stats={"total": 1},
        )
        anon.deanonymize.side_effect = lambda text: text.replace("[Person_001]", "Herr Mueller")

        wrapper = AnonymizingBackend(inner, anon)
        chunks = list(wrapper.generate_stream("System", "Herr Mueller"))
        assert chunks == ["Hallo Herr Mueller"]

    def test_anonymizing_generate_stream_inner_exception_propagates(self):
        """Exception im inneren Backend-Stream wird transparent propagiert."""
        inner = MagicMock(spec=LLMBackend)
        inner.generate_stream.return_value = iter([])
        inner.generate_stream.side_effect = RuntimeError("Connection reset")

        anon = _make_anonymizer_mock()

        wrapper = AnonymizingBackend(inner, anon)
        with pytest.raises(RuntimeError, match="Connection reset"):
            list(wrapper.generate_stream("System", "Frage"))

    def test_anonymizing_generate_stream_multiple_entity_occurrences(self):
        """De-Anonymisierung ersetzt mehrfach vorkommende Platzhalter korrekt."""
        inner = MagicMock(spec=LLMBackend)
        inner.generate_stream.return_value = iter(
            [
                "[Person_001] sagte, dass ",
                "[Person_001] recht hat.",
            ]
        )

        anon = _make_anonymizer_mock(
            clean_text="[Person_001]",
            mapping={"[Person_001]": "Herr Mueller"},
            stats={"total": 1},
        )
        anon.deanonymize.side_effect = lambda text: text.replace("[Person_001]", "Herr Mueller")

        wrapper = AnonymizingBackend(inner, anon)
        chunks = list(wrapper.generate_stream("System", "Herr Mueller"))
        full = "".join(chunks)

        assert full.count("Herr Mueller") == 2
        assert "[Person_001]" not in full

    def test_anonymizing_stats_updated_after_streaming(self):
        """anonymizer_stats ist nach generate_stream() aktuell."""
        inner = MagicMock(spec=LLMBackend)
        inner.generate_stream.return_value = iter(["Antwort"])

        anon = _make_anonymizer_mock(
            clean_text="[Firma_001]",
            mapping={"[Firma_001]": "ACME"},
            stats={"total": 2},
        )
        anon.stats = {"entities_replaced": 2, "total": 2}

        wrapper = AnonymizingBackend(inner, anon)
        list(wrapper.generate_stream("System", "ACME"))

        assert wrapper.last_anon_answer == "Antwort"
        assert wrapper.anonymizer_stats["total"] == 2

    def test_anonymizing_system_prompt_not_anonymized(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate.return_value = "OK"

        anon = _make_anonymizer_mock()

        wrapper = AnonymizingBackend(inner, anon)
        wrapper.generate("System mit PII", "User-Text")

        anon.anonymize.assert_called_once_with("User-Text")

    def test_anonymizing_last_anon_question_answer(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate.return_value = "Hallo [Person_001]"

        anon = _make_anonymizer_mock(
            clean_text="[Person_001]",
            mapping={"[Person_001]": "Herr Mueller"},
            stats={"total": 1},
        )

        wrapper = AnonymizingBackend(inner, anon)
        wrapper.generate("System", "Herr Mueller")

        assert wrapper.last_anon_question == "[Person_001]"
        assert wrapper.last_anon_answer == "Hallo [Person_001]"

    def test_anonymizing_delegates_last_usage(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate.return_value = "OK"
        type(inner).last_usage = PropertyMock(return_value={"prompt_tokens": 42})

        anon = _make_anonymizer_mock()
        wrapper = AnonymizingBackend(inner, anon)
        wrapper.generate("System", "User")

        assert wrapper.last_usage == {"prompt_tokens": 42}

    def test_anonymizing_entities_count(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate.return_value = "OK"

        anon = _make_anonymizer_mock(stats={"total": 3})

        wrapper = AnonymizingBackend(inner, anon)
        wrapper.generate("System", "User")

        assert wrapper.last_entities_anonymized == 3

    def test_anonymizing_state_resets_between_calls(self):
        inner = MagicMock(spec=LLMBackend)
        inner.generate.return_value = "OK"

        anon = MagicMock()
        # Erster Call
        anon.anonymize.return_value = "[Person_001]"
        anon.last_result = SimpleNamespace(mapping={"[Person_001]": "Mueller"}, stats={"total": 1})
        anon.deanonymize.side_effect = lambda text: text

        wrapper = AnonymizingBackend(inner, anon)
        wrapper.generate("System", "Mueller")

        assert wrapper.last_entities_anonymized == 1

        # Zweiter Call mit anderen Daten
        anon.anonymize.return_value = "[Firma_001]"
        anon.last_result = SimpleNamespace(mapping={"[Firma_001]": "ACME"}, stats={"total": 5})

        wrapper.generate("System", "ACME")

        assert wrapper.last_entities_anonymized == 5
        assert wrapper.last_anon_question == "[Firma_001]"

    def test_anonymizing_stream_state_resets_between_calls(self):
        inner = MagicMock(spec=LLMBackend)

        anon = MagicMock()
        anon.deanonymize.side_effect = lambda text: text

        wrapper = AnonymizingBackend(inner, anon)

        # Erster Stream-Call
        inner.generate_stream.return_value = iter(["Antwort1"])
        anon.anonymize.return_value = "[Person_001]"
        anon.last_result = SimpleNamespace(mapping={"[Person_001]": "Mueller"}, stats={"total": 1})
        list(wrapper.generate_stream("System", "Mueller"))
        assert wrapper.last_entities_anonymized == 1
        assert wrapper.last_anon_answer == "Antwort1"

        # Zweiter Stream-Call
        inner.generate_stream.return_value = iter(["Antwort2"])
        anon.anonymize.return_value = "[Firma_001]"
        anon.last_result = SimpleNamespace(mapping={"[Firma_001]": "ACME"}, stats={"total": 7})
        list(wrapper.generate_stream("System", "ACME"))
        assert wrapper.last_entities_anonymized == 7
        assert wrapper.last_anon_answer == "Antwort2"


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 6: create_backend() Factory
# ═══════════════════════════════════════════════════════════════════════


class TestCreateBackend:
    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    @patch("kiara.config.get_llm_backend")
    def test_create_backend_ollama_default(self, mock_get_llm, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, _ = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx
        mock_get_llm.return_value = {"url": "http://localhost:11434", "model": "qwen3:8b"}

        result = create_backend({})
        assert isinstance(result, OllamaBackend)
        assert result.provider == "ollama"
        assert result.model == "qwen3:8b"

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_anthropic_by_id(self, mock_get_by_id, mock_imp):
        mock_anthropic, _ = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic
        mock_get_by_id.return_value = {
            "provider": "anthropic",
            "api_key": "sk-test",
            "model": "claude-sonnet-4-6",
            "category": "cloud",
        }

        result = create_backend({}, backend_id=42)
        assert isinstance(result, AnthropicBackend)
        assert result.model == "claude-sonnet-4-6"

    @patch("kiara.generation.backends.openai._import_openai")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_openai_by_id(self, mock_get_by_id, mock_imp):
        mock_openai, _ = _make_openai_stubs()
        mock_imp.return_value = mock_openai
        mock_get_by_id.return_value = {
            "provider": "openai",
            "api_key": "sk-test",
            "model": "gpt-4o-mini",
            "base_url": "https://custom.api",
            "category": "cloud",
        }

        result = create_backend({}, backend_id=43)
        assert isinstance(result, OpenAIBackend)

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_ollama_by_id(self, mock_get_by_id, mock_imp_ollama, mock_imp_httpx):
        mock_ollama, mock_httpx, _ = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx
        mock_get_by_id.return_value = {
            "provider": "ollama",
            "url": "http://gpu-server.local:11434",
            "model": "qwen3:8b",
            "category": "llm",
        }

        result = create_backend({}, backend_id=44)
        assert isinstance(result, OllamaBackend)

    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_invalid_id(self, mock_get_by_id):
        mock_get_by_id.return_value = None
        with pytest.raises(ValueError, match="nicht gefunden"):
            create_backend({}, backend_id=999)

    @patch("kiara.config.get_llm_backend")
    def test_create_backend_no_active_llm(self, mock_get_llm):
        mock_get_llm.side_effect = RuntimeError("Kein aktives LLM-Backend")
        with pytest.raises(RuntimeError):
            create_backend({})

    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_ollama_missing_url_raises(self, mock_get_by_id):
        """Fehlendes 'url'-Feld bei Ollama-Backend wirft ValueError fail-closed."""
        mock_get_by_id.return_value = {"provider": "ollama", "model": "qwen3:8b", "category": "llm"}
        with pytest.raises(ValueError, match="URL"):
            create_backend({}, backend_id=1)

    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_anthropic_missing_model_raises(self, mock_get_by_id):
        """Fehlendes 'model'-Feld bei Anthropic-Backend wirft KeyError."""
        mock_get_by_id.return_value = {
            "provider": "anthropic",
            "api_key": "sk-test",
            "category": "cloud",
        }
        with pytest.raises(KeyError):
            create_backend({}, backend_id=2)

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_unknown_provider_defaults_ollama(
        self, mock_get_by_id, mock_imp_ollama, mock_imp_httpx
    ):
        """Unbekannter Provider faellt auf OllamaBackend zurueck."""
        mock_ollama, mock_httpx, _ = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx
        mock_get_by_id.return_value = {
            "provider": "unknown-llm",
            "url": "http://x:11434",
            "model": "test",
            "category": "llm",
        }

        result = create_backend({}, backend_id=99)
        assert isinstance(result, OllamaBackend)
        assert result.provider == "ollama"

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_cloud_max_tokens_from_config(self, mock_get_by_id, mock_imp):
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic
        mock_get_by_id.return_value = {
            "provider": "anthropic",
            "api_key": "sk-test",
            "model": "claude-sonnet-4-6",
            "category": "cloud",
        }

        config = {"cloud": {"max_tokens": 8192, "temperature": 0.5}}
        result = create_backend(config, backend_id=42)
        assert isinstance(result, AnthropicBackend)
        assert result._max_tokens == 8192
        assert result._temperature == 0.5

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_cloud_empty_api_key(self, mock_get_by_id, mock_imp):
        mock_anthropic, _ = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic
        mock_get_by_id.return_value = {
            "provider": "anthropic",
            "api_key": "",
            "model": "claude-sonnet-4-6",
            "category": "cloud",
        }

        result = create_backend({}, backend_id=42)
        assert isinstance(result, AnthropicBackend)

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    @patch("kiara.backend_selector.get_backend_by_id")
    def test_create_backend_cloud_defaults_without_cloud_config(self, mock_get_by_id, mock_imp):
        mock_anthropic, _ = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic
        mock_get_by_id.return_value = {
            "provider": "anthropic",
            "api_key": "sk-test",
            "model": "claude-sonnet-4-6",
            "category": "cloud",
        }

        result = create_backend({}, backend_id=42)
        assert result._max_tokens == 4096  # DEFAULT_MAX_TOKENS
        assert result._temperature == 0.3  # DEFAULT_LLM_TEMPERATURE


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 7: wrap_with_anonymization()
# ═══════════════════════════════════════════════════════════════════════


class TestWrapWithAnonymization:
    def test_wrap_ollama_no_anonymization(self):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "ollama"
        result = wrap_with_anonymization(backend, {})
        assert result is backend

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources", return_value=MagicMock())
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_wrap_cloud_with_anonymization(self, MockAnonymizer, mock_build):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"

        mock_anon = MagicMock()
        MockAnonymizer.return_value = mock_anon

        config = {"anonymization": {"enabled": True}}

        result = wrap_with_anonymization(backend, config)

        assert isinstance(result, AnonymizingBackend)

    def test_wrap_cloud_anonymization_disabled(self):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        config = {"anonymization": {"enabled": False}}
        result = wrap_with_anonymization(backend, config)
        assert result is backend

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources", return_value=MagicMock())
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_wrap_cloud_anonymization_default_enabled(self, MockAnonymizer, mock_build):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        mock_anon = MagicMock()
        MockAnonymizer.return_value = mock_anon

        config = {"anonymization": {}}  # Kein "enabled"-Key
        result = wrap_with_anonymization(backend, config)
        assert isinstance(result, AnonymizingBackend)

    @patch(
        "kiara.anonymization.stammdaten.build_automaton_from_sources",
        side_effect=Exception("LDAP down"),
    )
    def test_wrap_cloud_build_automaton_failure_default_fail_closed(self, mock_build):
        """Issue #1056: Build-Failure bei Default-True (anonymization {} oder fehlend)
        wirft RuntimeError statt lautlos auf Raw-Cloud zurueckzufallen."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"

        # anonymization {} -> enabled-Default ist True -> Fail-Closed bei Init-Error
        with pytest.raises(RuntimeError, match="aktiviert"):
            wrap_with_anonymization(backend, {"anonymization": {}})

        # Auch ohne anonymization-Sektion (komplett fehlend) gilt Default True
        with pytest.raises(RuntimeError, match="aktiviert"):
            wrap_with_anonymization(backend, {})

    @patch(
        "kiara.anonymization.stammdaten.build_automaton_from_sources",
        side_effect=Exception("LDAP down"),
    )
    def test_wrap_cloud_build_automaton_failure_explicit_disabled_graceful(self, mock_build):
        """Issue #1056: Bei explizit enabled=False darf Build-Failure das Backend
        ohne Anonymisierung durchlassen (Operator-Entscheidung)."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"

        # Sollte nie zum try-Block kommen, aber wenn doch: graceful, kein Throw
        result = wrap_with_anonymization(backend, {"anonymization": {"enabled": False}})
        assert result is backend

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources", return_value=MagicMock())
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_wrap_cloud_uses_build_automaton(self, MockAnonymizer, mock_build):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        mock_anon = MagicMock()
        MockAnonymizer.return_value = mock_anon

        config = {"anonymization": {"enabled": True}}
        wrap_with_anonymization(backend, config)
        mock_build.assert_called_once()

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources", return_value=MagicMock())
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_wrap_cloud_normalizes_anonymizer_config(self, MockAnonymizer, mock_build):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        mock_anon = MagicMock()
        MockAnonymizer.return_value = mock_anon

        config = {
            "anonymization": {
                "enabled": True,
                "erp_source": {"enabled": True, "type": "csv", "path": "/data/stamm"},
                "ldap_source": {"server": "ldap.example.com"},
                "custom_entries": ["Test-Eintrag"],
                "custom_patterns": [r"\d{4}"],
            }
        }
        wrap_with_anonymization(backend, config)

        # Config wurde normalisiert an build_automaton_from_sources uebergeben
        call_config = mock_build.call_args[0][0]
        anon_cfg = call_config["anonymization"]
        assert anon_cfg["erp"]["enabled"] is True
        assert anon_cfg["erp"]["csv_dir"] == "/data/stamm"
        assert anon_cfg["ldap"] == {"server": "ldap.example.com"}
        assert anon_cfg["custom_entries"] == ["Test-Eintrag"]
        assert anon_cfg["custom_patterns"] == [r"\d{4}"]

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources", return_value=MagicMock())
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_wrap_cloud_prefers_erp_source_when_present(self, MockAnonymizer, mock_build):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        mock_anon = MagicMock()
        MockAnonymizer.return_value = mock_anon

        config = {
            "anonymization": {
                "enabled": True,
                "erp_source": {"enabled": False},
                "erp": {"enabled": True, "backend": "csv", "csv_dir": "/old"},
            }
        }
        wrap_with_anonymization(backend, config)

        call_config = mock_build.call_args[0][0]
        anon_cfg = call_config["anonymization"]
        assert anon_cfg["erp"]["enabled"] is False

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources", return_value=MagicMock())
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_wrap_cloud_normalizes_pure_erp_config(self, MockAnonymizer, mock_build):
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        mock_anon = MagicMock()
        MockAnonymizer.return_value = mock_anon

        config = {
            "anonymization": {
                "enabled": True,
                "erp": {"backend": "csv", "csv_dir": "/tmp/stammdaten"},
            }
        }
        wrap_with_anonymization(backend, config)

        call_config = mock_build.call_args[0][0]
        anon_cfg = call_config["anonymization"]
        assert anon_cfg["erp"]["enabled"] is True


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 7b: wrap_with_anonymization() mit restriction_levels
# ═══════════════════════════════════════════════════════════════════════


class TestWrapWithAnonymizationRestrictionLevels:
    def test_restriction_levels_none_fallback_behavior(self):
        """restriction_levels=None nutzt bisherigen Fallback auf anonymization.enabled."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        config = {"anonymization": {"enabled": False}}
        result = wrap_with_anonymization(backend, config, restriction_levels=None)
        assert result is backend  # Kein Wrapping (disabled)

    def test_restriction_levels_public_no_wrapping(self):
        """restriction_levels=["public"] fuehrt zu keinem Wrapping."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        result = wrap_with_anonymization(backend, {}, restriction_levels=["public"])
        assert result is backend

    def test_restriction_levels_empty_list_no_wrapping(self):
        """Leere restriction_levels fuehrt zu keinem Wrapping."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        result = wrap_with_anonymization(backend, {}, restriction_levels=[])
        assert result is backend

    @patch("kiara.anonymization.stammdaten.build_automaton_from_sources", return_value=MagicMock())
    @patch("kiara.anonymization.anonymizer.Anonymizer")
    def test_restriction_levels_anonymize_wraps(self, MockAnonymizer, mock_build):
        """restriction_levels=["anonymize"] fuehrt zu Wrapping."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        mock_anon = MagicMock()
        MockAnonymizer.return_value = mock_anon

        result = wrap_with_anonymization(
            backend,
            {"anonymization": {"enabled": True}},
            restriction_levels=["anonymize"],
        )
        assert isinstance(result, AnonymizingBackend)

    def test_restriction_levels_blocked_raises_value_error(self):
        """restriction_levels=["blocked"] wirft ValueError."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        with pytest.raises(ValueError, match="blocked"):
            wrap_with_anonymization(
                backend,
                {},
                restriction_levels=["blocked"],
            )

    def test_restriction_levels_anonymize_init_error_raises_runtime(self):
        """Init-Fehler bei restriction_levels=["anonymize"] wirft RuntimeError (nicht fail-open)."""
        backend = MagicMock(spec=LLMBackend)
        backend.provider = "anthropic"
        with patch(
            "kiara.anonymization.stammdaten.build_automaton_from_sources",
            side_effect=Exception("LDAP down"),
        ):
            with pytest.raises(RuntimeError, match="anonymize-Level"):
                wrap_with_anonymization(
                    backend,
                    {"anonymization": {"enabled": True}},
                    restriction_levels=["anonymize"],
                )


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 8: Hilfs-Funktionen
# ═══════════════════════════════════════════════════════════════════════


class TestHilfsFunktionen:
    def test_build_usage_from_ollama(self):
        response = {
            "prompt_eval_count": 100,
            "eval_count": 50,
            "total_duration": 5_000_000_000,
        }
        result = build_usage_from_ollama(response, 8192, "qwen3:8b")
        assert result == {
            "prompt_tokens": 100,
            "response_tokens": 50,
            "total_duration_ms": 5000,
            "num_ctx": 8192,
            "model": "qwen3:8b",
        }

    def test_build_usage_from_ollama_missing_fields(self):
        result = build_usage_from_ollama({}, 8192, "test")
        assert result == {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_duration_ms": 0,
            "num_ctx": 8192,
            "model": "test",
        }

    def test_build_usage_estimated(self):
        result = build_usage_estimated(350, 175, "test-model")
        assert result["prompt_tokens"] == 100
        assert result["response_tokens"] == 50
        assert result["model"] == "test-model"
        assert result["total_duration_ms"] == 0
        assert result["num_ctx"] == 0


# ═══════════════════════════════════════════════════════════════════════
# Gruppe 9: Backend Network-Error Propagation
# ═══════════════════════════════════════════════════════════════════════


class TestOllamaBackendNetworkErrors:
    """Ollama-Backend: Netzwerk-/API-Fehler werden transparent propagiert."""

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_generate_connection_error_propagates(self, mock_imp_ollama, mock_imp_httpx):
        """ConnectionError bei client.chat() wird nicht verschluckt."""
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx
        mock_client.chat.side_effect = ConnectionError("Connection refused")

        backend = OllamaBackend(url="http://offline:11434", model="test")
        with pytest.raises(ConnectionError, match="Connection refused"):
            backend.generate("", "Test")

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    def test_generate_stream_timeout_propagates(self, mock_imp_ollama, mock_imp_httpx):
        """Timeout bei Streaming wird transparent propagiert."""
        mock_ollama, mock_httpx, mock_client = _make_ollama_stubs()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx
        mock_client.chat.side_effect = TimeoutError("Read timed out")

        backend = OllamaBackend(url="http://slow:11434", model="test")
        with pytest.raises(TimeoutError, match="Read timed out"):
            list(backend.generate_stream("", "Test"))


class TestAnthropicBackendNetworkErrors:
    """Anthropic-Backend: API-Fehler werden transparent propagiert."""

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_generate_api_error_propagates(self, mock_imp):
        """API-Fehler ohne Think-Bezug wird nicht abgefangen."""
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic
        mock_client.messages.create.side_effect = RuntimeError("500 Internal Server Error")

        backend = AnthropicBackend(api_key="sk-test", model="claude-test")
        with pytest.raises(RuntimeError, match="500"):
            backend.generate("System", "Test")

    @patch("kiara.generation.backends.anthropic._import_anthropic")
    def test_generate_thinking_error_retries_without_thinking(self, mock_imp):
        """Thinking-Fehler (BadRequestError) wird einmal ohne Thinking retryed."""
        mock_anthropic, mock_client = _make_anthropic_stubs()
        mock_imp.return_value = mock_anthropic

        # Erster Call: BadRequestError mit thinking-Bezug, Zweiter Call: Erfolg
        ok_resp = MagicMock()
        ok_resp.content = [MagicMock(type="text", text="Fallback-Antwort")]
        ok_resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.side_effect = [
            mock_anthropic.BadRequestError("thinking budget exceeded"),
            ok_resp,
        ]

        backend = AnthropicBackend(api_key="sk-test", model="claude-test")
        result = backend.generate("", "Test", think=True)
        assert result == "Fallback-Antwort"
        assert mock_client.messages.create.call_count == 2


class TestOpenAIBackendNetworkErrors:
    """OpenAI-Backend: API-Fehler werden transparent propagiert."""

    @patch("kiara.generation.backends.openai._import_openai")
    def test_generate_authentication_error_propagates(self, mock_imp):
        """AuthenticationError bei ungueltigem API-Key wird nicht verschluckt."""
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai
        mock_client.chat.completions.create.side_effect = RuntimeError(
            "401 Unauthorized: Invalid API key"
        )

        backend = OpenAIBackend(api_key="sk-bad", model="gpt-test")
        with pytest.raises(RuntimeError, match="401"):
            backend.generate("System", "Test")

    @patch("kiara.generation.backends.openai._import_openai")
    def test_generate_timeout_propagates(self, mock_imp):
        """Timeout wird transparent propagiert."""
        mock_openai, mock_client = _make_openai_stubs()
        mock_imp.return_value = mock_openai
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timed out")

        backend = OpenAIBackend(api_key="sk-test", model="gpt-test")
        with pytest.raises(TimeoutError, match="timed out"):
            backend.generate("System", "Test")
