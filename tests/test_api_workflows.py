"""Tests fuer kiara.web.api_workflows: _init_llm() und Lazy Singletons.

Prueft die Initialisierung des Ollama-Backends fuer Workflow-Module
(TemplateEngine, DocumentSummarizer) inkl. Error-Handling.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestInitLlm:
    """Tests fuer _init_llm() in api_workflows.py."""

    @patch("kiara.generation.backends.ollama._import_httpx")
    @patch("kiara.generation.backends.ollama._import_ollama")
    @patch("kiara.config.get_llm_backend")
    @patch("kiara.config.load_config")
    def test_init_llm_success(self, mock_load, mock_get_llm, mock_imp_ollama, mock_imp_httpx):
        """Erfolgreiche Initialisierung gibt (config, OllamaBackend) zurueck."""
        from kiara.web.api_workflows import _init_llm

        mock_load.return_value = {"ollama": {"url": "http://x:11434"}}
        mock_get_llm.return_value = {"url": "http://x:11434", "model": "qwen3:8b"}

        mock_ollama = MagicMock()
        mock_httpx = MagicMock()
        mock_imp_ollama.return_value = mock_ollama
        mock_imp_httpx.return_value = mock_httpx

        config, backend = _init_llm()

        assert config is not None
        assert backend is not None
        assert backend.provider == "ollama"
        assert backend.model == "qwen3:8b"

    @patch("kiara.config.get_llm_backend")
    @patch("kiara.config.load_config")
    def test_init_llm_no_backend_returns_none(self, mock_load, mock_get_llm):
        """Wenn get_llm_backend() fehlschlaegt, ist backend=None."""
        from kiara.web.api_workflows import _init_llm

        mock_load.return_value = {}
        mock_get_llm.side_effect = RuntimeError("Kein aktives LLM-Backend")

        config, backend = _init_llm()

        assert config is not None
        assert backend is None


class TestSummarizeDocument:
    """Tests fuer /api/workflows/summarize Budget-Grenzen."""

    @pytest.mark.asyncio
    @patch("kiara.web.api_workflows.load_config")
    @patch("kiara.retrieval.retriever.Retriever")
    @patch("kiara.web.api_workflows._get_summarizer")
    @patch("kiara.web.api_chat._build_datasource_filter", return_value=None)
    async def test_summarize_truncates_input_before_llm(
        self,
        mock_filter,
        mock_get_summarizer,
        mock_retriever_cls,
        mock_config,
    ):
        from kiara.web.api_workflows import SummarizeRequest, summarize_document

        mock_config.return_value = {"workflows": {"summarize_max_input_tokens": 1}}
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "abcdef", "metadata": {"source": "a.txt"}},
        ]
        mock_retriever_cls.return_value = retriever
        summarizer = MagicMock()
        summarizer.summarize = AsyncMock(return_value="Kurzfassung")
        mock_get_summarizer.return_value = summarizer

        result = await summarize_document(
            SummarizeRequest(source="a.txt"),
            user={"id": 1, "is_admin": True},
        )

        assert result["truncated"] is True
        assert result["input_chars"] == 6
        assert result["max_input_chars"] == 3
        assert "gekuerzt" in result["summary"]
        assert summarizer.summarize.call_args.kwargs["text"] == "abc"

    @patch("kiara.generation.backends.ollama.OllamaBackend")
    @patch("kiara.config.get_llm_backend")
    @patch("kiara.config.load_config")
    def test_init_llm_ollama_connection_error_returns_none(
        self, mock_load, mock_get_llm, mock_backend_cls
    ):
        """Wenn OllamaBackend()-Konstruktor fehlschlaegt, ist backend=None."""
        from kiara.web.api_workflows import _init_llm

        mock_load.return_value = {}
        mock_get_llm.return_value = {"url": "http://offline:11434", "model": "test"}
        mock_backend_cls.side_effect = RuntimeError("Connection refused")

        config, backend = _init_llm()

        assert config is not None
        assert backend is None
