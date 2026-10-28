"""Gemeinsame Basisklasse fuer RAGChain und UnchainedChain.

Kapselt die 7 identischen Properties beider Chains:
last_prompt, last_usage, route_used, last_anon_question,
last_anon_answer, last_entities_anonymized, backend.

Interne Klasse — wird nicht direkt von externen Konsumenten verwendet.
"""

from kiara.generation.backends.base import LLMBackend


class _ChainBase:
    """Gemeinsame Properties fuer RAGChain und UnchainedChain."""

    _backend: LLMBackend
    _last_prompt: str

    @property
    def last_prompt(self) -> str:
        """Vollstaendiger Prompt des letzten Aufrufs (fuer Audit-Trail)."""
        return self._last_prompt

    @property
    def last_usage(self) -> dict:
        """Token-Usage-Daten des letzten LLM-Aufrufs."""
        return self._backend.last_usage

    @property
    def route_used(self) -> str:
        """Route des letzten Aufrufs: 'local' oder 'cloud'."""
        return "cloud" if self._backend.provider in ("anthropic", "openai") else "local"

    @property
    def last_anon_question(self) -> str:
        """Anonymisierte Frage des letzten Aufrufs (leer wenn kein AnonymizingBackend)."""
        return getattr(self._backend, "last_anon_question", "")

    @property
    def last_anon_answer(self) -> str:
        """Anonymisierte Antwort des letzten Aufrufs (leer wenn kein AnonymizingBackend)."""
        return getattr(self._backend, "last_anon_answer", "")

    @property
    def last_entities_anonymized(self) -> int:
        """Anzahl anonymisierter Entities des letzten Aufrufs."""
        val = getattr(self._backend, "last_entities_anonymized", 0) or 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    @property
    def backend(self) -> LLMBackend:
        """Public accessor fuer das LLM-Backend (fuer Audit-Zugriff)."""
        return self._backend
