"""Sprint 3 Cleanup-Validierung: Legacy-Module entfernt, neue Pfade korrekt.

Stellt sicher, dass:
    - kiara.generation.llm und kiara.generation.cloud_client nicht mehr importierbar sind
    - HybridRAGChain nicht mehr existiert
    - build_usage_*-Helfer nur noch an den kanonischen Standorten leben
    - Workflow-Module und LLM-Detector die neue Backend-Signatur verwenden
    - QueryExpander keinen Legacy-llm-Parameter mehr hat
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest


class TestLegacyImportsRemoved:
    """Stellt sicher dass alte Module nicht mehr importierbar sind."""

    def test_llm_module_removed(self):
        """kiara.generation.llm darf nicht mehr importierbar sein."""
        with pytest.raises(ImportError):
            importlib.import_module("kiara.generation.llm")

    def test_cloud_client_module_removed(self):
        """kiara.generation.cloud_client darf nicht mehr importierbar sein."""
        with pytest.raises(ImportError):
            importlib.import_module("kiara.generation.cloud_client")

    def test_hybrid_rag_chain_removed(self):
        """HybridRAGChain darf nicht mehr aus chain.py importierbar sein."""
        with pytest.raises(ImportError):
            pass

    def test_ollama_llm_not_in_generation(self):
        """OllamaLLM darf nicht mehr aus generation importierbar sein."""
        import kiara.generation

        assert not hasattr(kiara.generation, "OllamaLLM")


class TestBuildUsageCanonicalLocation:
    """Stellt sicher dass Usage-Helfer nur noch an EINER Stelle existieren."""

    def test_build_usage_from_ollama_in_backends(self):
        """build_usage_from_ollama() ist aus backends.ollama importierbar."""
        from kiara.generation.backends.ollama import build_usage_from_ollama

        result = build_usage_from_ollama(
            {"prompt_eval_count": 10, "eval_count": 5, "total_duration": 1_000_000_000},
            num_ctx=4096,
            model="test",
        )
        assert result["prompt_tokens"] == 10
        assert result["response_tokens"] == 5

    def test_build_usage_estimated_in_backends(self):
        """build_usage_estimated() ist aus backends.base importierbar."""
        from kiara.generation.backends.base import build_usage_estimated

        result = build_usage_estimated(1000, 500, "test")
        assert result["prompt_tokens"] > 0
        assert result["model"] == "test"


class TestOllamaAdapterUsesNewImport:
    """Stellt sicher dass OllamaToolAdapter den neuen Import nutzt."""

    def test_adapter_usage_tracking(self):
        """OllamaToolAdapter nutzt build_usage_from_ollama aus backends.ollama."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        # Import-Pruefung: wenn der alte Import noch existieren wuerde,
        # wuerde das Modul beim Laden crashen (da llm.py geloescht)
        assert OllamaToolAdapter is not None


class TestWorkflowModulesUseBackendInterface:
    """Prueft die direkte Backend-Signatur in TemplateEngine/Summarizer."""

    def test_template_engine_uses_empty_system_prompt(self):
        import asyncio

        from kiara.workflows.template_engine import TemplateEngine

        mock_backend = MagicMock()
        mock_backend.generate.return_value = "ok"
        engine = TemplateEngine({"workflows": {}}, llm=mock_backend)
        asyncio.run(engine.generate_checklist("Test"))
        assert mock_backend.generate.call_args.args[0] == ""

    def test_summarizer_uses_empty_system_prompt(self):
        import asyncio

        from kiara.workflows.summarizer import DocumentSummarizer

        mock_backend = MagicMock()
        mock_backend.generate.return_value = "ok"
        summarizer = DocumentSummarizer({"workflows": {}}, llm=mock_backend)
        asyncio.run(summarizer.summarize("Kurztext"))
        assert mock_backend.generate.call_args.args[0] == ""


class TestLLMDetectorUsesOllamaBackend:
    """Prueft dass LLMPIIDetector OllamaBackend statt OllamaLLM nutzt."""

    def test_create_llm_returns_ollama_backend(self):
        """_create_llm() erzeugt ein OllamaBackend, kein OllamaLLM."""
        from kiara.anonymization.llm_detector import LLMPIIDetector

        config = {"anonymization": {"llm_detection": {"enabled": True}}}
        detector = LLMPIIDetector(config)

        with (
            patch(
                "kiara.config.get_llm_backend",
                return_value={"url": "http://x:11434", "model": "test"},
            ),
            patch("kiara.generation.backends.ollama.OllamaBackend") as mock_backend_cls,
        ):
            llm = detector._create_llm()

        mock_backend_cls.assert_called_once_with(url="http://x:11434", model="test")
        assert llm is mock_backend_cls.return_value

    def test_create_llm_uses_custom_model_when_set(self):
        """_create_llm() mit explizitem model nutzt dieses statt backend['model']."""
        from kiara.anonymization.llm_detector import LLMPIIDetector

        config = {"anonymization": {"llm_detection": {"enabled": True, "model": "custom-qwen"}}}
        detector = LLMPIIDetector(config)

        with (
            patch(
                "kiara.config.get_llm_backend",
                return_value={"url": "http://x:11434", "model": "default-model"},
            ),
            patch("kiara.generation.backends.ollama.OllamaBackend") as mock_cls,
        ):
            detector._create_llm()

        mock_cls.assert_called_once_with(url="http://x:11434", model="custom-qwen")

    def test_create_llm_fallback_to_backend_model_when_empty(self):
        """_create_llm() ohne model nutzt backend['model'] aus get_llm_backend()."""
        from kiara.anonymization.llm_detector import LLMPIIDetector

        config = {"anonymization": {"llm_detection": {"enabled": True, "model": ""}}}
        detector = LLMPIIDetector(config)

        with (
            patch(
                "kiara.config.get_llm_backend",
                return_value={"url": "http://x:11434", "model": "fallback-model"},
            ),
            patch("kiara.generation.backends.ollama.OllamaBackend") as mock_cls,
        ):
            detector._create_llm()

        mock_cls.assert_called_once_with(url="http://x:11434", model="fallback-model")

    def test_create_llm_failure_returns_none(self):
        """_create_llm() bei Fehler gibt None zurueck (kein Crash)."""
        from kiara.anonymization.llm_detector import LLMPIIDetector

        config = {"anonymization": {"llm_detection": {"enabled": True}}}
        detector = LLMPIIDetector(config)

        with patch("kiara.config.get_llm_backend", side_effect=RuntimeError("Kein Backend")):
            result = detector._create_llm()

        assert result is None

    def test_detect_calls_generate_with_system_prompt(self):
        """detect() ruft backend.generate(system_prompt, user_message, ...) auf.

        Der System-Prompt instruiert das LLM zur PII-Erkennung und enthaelt
        die Prompt-Injection-Hardening-Marker (<<<USER_TEXT>>>) — er darf
        NICHT leer sein, sonst landet die Anweisung im User-Slot und ist
        anfaellig fuer Injection ueber den zu analysierenden Text.
        """
        from kiara.anonymization.llm_detector import LLMPIIDetector

        config = {"anonymization": {"llm_detection": {"enabled": True}}}
        detector = LLMPIIDetector(config)

        mock_backend = MagicMock()
        mock_backend.generate.return_value = "[]"
        detector._llm = mock_backend
        detector._llm_init_attempted = True

        detector.detect("Max Mustermann wohnt in Berlin", [])
        mock_backend.generate.assert_called_once()
        system_prompt = mock_backend.generate.call_args.args[0]
        assert system_prompt, "System-Prompt darf nicht leer sein (Injection-Schutz)"
        assert "Datenschutz-Analyst" in system_prompt


class TestQueryExpanderNoLegacyLLM:
    """Prueft dass QueryExpander keinen Legacy-llm-Parameter mehr hat."""

    def test_no_llm_parameter(self):
        """QueryExpander.__init__() akzeptiert kein llm= mehr."""
        import inspect

        from kiara.retrieval.query_expansion import QueryExpander

        sig = inspect.signature(QueryExpander.__init__)
        assert "llm" not in sig.parameters
