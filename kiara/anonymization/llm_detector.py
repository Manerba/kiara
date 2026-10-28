"""LLM-PII-Detector: kiara-Adapter mit config-kompatibler API.

Re-exportiert die Library-Klassen und stellt zusaetzlich bereit:
    - LLMPIIDetector(config): config-kompatibler Wrapper mit lazy LLM-Init
    - build_llm_detector(config): Factory fuer die pure Library-Klasse
"""

import logging
import time

from kiara_anonymizer.llm_detector import (
    LLMBackend,
    LLMDetection,
    _normalize_category,
    _overlaps,
)
from kiara_anonymizer.llm_detector import (
    LLMPIIDetector as _LibraryLLMPIIDetector,
)

logger = logging.getLogger(__name__)


def build_llm_detector(config: dict) -> _LibraryLLMPIIDetector | None:
    """Factory: Erzeugt eine pure Library-LLMPIIDetector-Instanz.

    Liest die kiara-Config und erstellt ein OllamaBackend als LLM-Backend.
    Gibt None zurueck wenn die LLM-Detection deaktiviert ist oder das
    Backend nicht erstellt werden kann.
    """
    llm_cfg = config.get("anonymization", {}).get("llm_detection", {})
    if not llm_cfg.get("enabled", False):
        return None

    try:
        from kiara.config import get_llm_backend
        from kiara.generation.backends.ollama import OllamaBackend

        backend_cfg = get_llm_backend(config)
        backend = OllamaBackend(
            url=backend_cfg["url"],
            model=llm_cfg.get("model") or backend_cfg["model"],
        )
        return _LibraryLLMPIIDetector(
            llm_backend=backend,
            enabled=True,
            confidence_threshold=llm_cfg.get("confidence_threshold", 0.7),
            temperature=llm_cfg.get("temperature", 0.1),
            num_ctx=llm_cfg.get("num_ctx", 4096),
            cache_size=llm_cfg.get("cache_size", 256),
            timeout_seconds=llm_cfg.get("timeout_seconds", 10),
        )
    except Exception as exc:
        logger.warning("LLM-PII-Detector konnte nicht initialisiert werden: %s", exc)
        return None


class LLMPIIDetector(_LibraryLLMPIIDetector):
    """Config-kompatibler Wrapper fuer bestehende kiara-Tests und -Imports.

    Behaelt die bisherige API bei: LLMPIIDetector(config) mit lazy LLM-Init.
    Intern delegiert an die Library-Klasse via super().
    """

    _LLM_INIT_RETRY_INTERVAL_SEC = 60.0

    def __init__(self, config: dict):
        self._config = config
        self._llm_init_attempted = False
        self._llm_init_last_failure: float | None = None
        llm_cfg = config.get("anonymization", {}).get("llm_detection", {})
        super().__init__(
            llm_backend=None,
            enabled=llm_cfg.get("enabled", False),
            confidence_threshold=llm_cfg.get("confidence_threshold", 0.7),
            temperature=llm_cfg.get("temperature", 0.1),
            num_ctx=llm_cfg.get("num_ctx", 4096),
            cache_size=llm_cfg.get("cache_size", 256),
            timeout_seconds=llm_cfg.get("timeout_seconds", 10),
        )
        self._model = llm_cfg.get("model", "")

    def _create_llm(self):
        """Lazy-Init des LLM-Backends. Bei Fehler Retry nach Cool-Down."""
        if self._llm is not None:
            return self._llm
        if (
            self._llm_init_attempted
            and self._llm_init_last_failure is not None
            and time.monotonic() - self._llm_init_last_failure < self._LLM_INIT_RETRY_INTERVAL_SEC
        ):
            return None
        self._llm_init_attempted = True

        try:
            from kiara.config import get_llm_backend
            from kiara.generation.backends.ollama import OllamaBackend

            backend = get_llm_backend(self._config)
            url = backend["url"]
            model = self._model if self._model else backend["model"]

            self._llm = OllamaBackend(url=url, model=model)
            self._llm_init_last_failure = None
        except Exception as exc:
            logger.warning("LLM-PII-Detector: LLM-Backend konnte nicht erstellt werden: %s", exc)
            self._llm = None
            self._llm_init_last_failure = time.monotonic()

        return self._llm

    def detect(self, text: str, covered_ranges: list[tuple[int, int]]):
        """Detect mit lazy LLM-Init vor dem Library-Aufruf."""
        if self._enabled and self._llm is None:
            self._create_llm()
        return super().detect(text, covered_ranges)
