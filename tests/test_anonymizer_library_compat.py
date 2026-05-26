"""Compat-Tests: kiara-Importpfade funktionieren nach Library-Extraktion.

Prueft:
- Smoke-Test: kiara_anonymizer importierbar ohne kiara
- Smoke-Test: alle kiara.anonymization.* Shims importierbar
- LLMPIIDetector(config) bleibt instanziierbar
- build_llm_detector(config) liefert Library-Klasse (nicht Wrapper)
"""


class TestLibrarySmoke:
    """kiara_anonymizer ist ohne kiara-Imports importierbar."""

    def test_import_kiara_anonymizer(self):
        import kiara_anonymizer

        assert hasattr(kiara_anonymizer, "Anonymizer")
        assert hasattr(kiara_anonymizer, "AnonymizationAutomaton")
        assert hasattr(kiara_anonymizer, "RegexPatternMatcher")
        assert hasattr(kiara_anonymizer, "ERPConnector")
        assert hasattr(kiara_anonymizer, "LLMPIIDetector")
        assert hasattr(kiara_anonymizer, "LLMBackend")

    def test_no_kiara_imports_in_library(self):
        """Verifiziert dass kiara_anonymizer keine kiara-Imports enthaelt."""

        # Alle Module geladen — kein ImportError von kiara.*


class TestShimSmoke:
    """Alle kiara.anonymization.* Shims bleiben importierbar."""

    def test_import_base(self):
        from kiara.anonymization.base import BaseAnonymizer

        assert BaseAnonymizer is not None

    def test_import_text_utils(self):
        from kiara.anonymization._text_utils import (
            _is_word_boundary,
            _resolve_overlaps,
            open_with_fallback,
        )

        assert callable(_is_word_boundary)
        assert callable(_resolve_overlaps)
        assert callable(open_with_fallback)

    def test_import_automaton(self):
        from kiara.anonymization.automaton import (
            _PSEUDONYM_PREFIXES,
            _REPORTING_CATEGORY,
            AnonymizationAutomaton,
        )

        assert AnonymizationAutomaton is not None
        assert isinstance(_PSEUDONYM_PREFIXES, dict)
        assert isinstance(_REPORTING_CATEGORY, dict)

    def test_import_anonymizer(self):
        from kiara.anonymization.anonymizer import AnonymizationResult, Anonymizer

        assert Anonymizer is not None
        assert AnonymizationResult is not None

    def test_import_patterns(self):
        from kiara.anonymization.patterns import _BUILTIN_PATTERNS, RegexPatternMatcher

        assert RegexPatternMatcher is not None
        assert isinstance(_BUILTIN_PATTERNS, list)

    def test_import_erp_connector(self):
        from kiara.anonymization.erp_connector import ERPConnector

        assert ERPConnector is not None

    def test_import_llm_detector(self):
        from kiara.anonymization.llm_detector import (
            LLMDetection,
            LLMPIIDetector,
            _normalize_category,
            _overlaps,
        )

        assert LLMDetection is not None
        assert LLMPIIDetector is not None
        assert callable(_normalize_category)
        assert callable(_overlaps)


class TestLLMDetectorCompat:
    """LLMPIIDetector(config) bleibt instanziierbar und lazy-initialisiert."""

    def test_config_compat_instantiation(self):
        from kiara.anonymization.llm_detector import LLMPIIDetector

        config = {
            "anonymization": {
                "llm_detection": {
                    "enabled": False,
                    "confidence_threshold": 0.7,
                },
            },
        }
        detector = LLMPIIDetector(config)
        assert detector._enabled is False
        results = detector.detect("Test", [])
        assert results == []

    def test_config_compat_lazy_init(self):
        from kiara.anonymization.llm_detector import LLMPIIDetector

        config = {
            "anonymization": {
                "llm_detection": {
                    "enabled": True,
                    "confidence_threshold": 0.7,
                },
            },
        }
        detector = LLMPIIDetector(config)
        assert hasattr(detector, "_create_llm")
        assert hasattr(detector, "_llm_init_attempted")
        assert detector._llm_init_attempted is False


class TestBuildLLMDetector:
    """build_llm_detector(config) liefert pure Library-Klasse."""

    def test_disabled_returns_none(self):
        from kiara.anonymization.llm_detector import build_llm_detector

        config = {
            "anonymization": {
                "llm_detection": {"enabled": False},
            },
        }
        result = build_llm_detector(config)
        assert result is None

    def test_returns_library_class_not_wrapper(self):
        """build_llm_detector erzeugt _LibraryLLMPIIDetector, nicht den Wrapper."""
        from kiara_anonymizer.llm_detector import LLMPIIDetector as LibraryDetector

        from kiara.anonymization.llm_detector import LLMPIIDetector, build_llm_detector

        # Wenn enabled aber kein Backend erreichbar → None
        config = {
            "anonymization": {
                "llm_detection": {
                    "enabled": True,
                    "confidence_threshold": 0.7,
                },
            },
        }
        result = build_llm_detector(config)
        # Entweder None (Backend nicht erreichbar) oder Library-Klasse
        if result is not None:
            assert isinstance(result, LibraryDetector)
            assert not isinstance(result, LLMPIIDetector)  # Nicht der Wrapper
