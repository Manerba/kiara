"""Anonymisierungs-Decorator fuer LLM-Backends.

Wrapt ein beliebiges LLMBackend mit transparenter Anonymisierung (Input)
und De-Anonymisierung (Output). Verwendet Anonymizer (nicht Pseudonymizer),
damit pro Call ein eigenes bijektives Mapping fuer die Rueckersetzung existiert.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import TYPE_CHECKING

from kiara.generation.backends.base import LLMBackend

if TYPE_CHECKING:
    from kiara.anonymization.anonymizer import Anonymizer

logger = logging.getLogger(__name__)


class AnonymizingBackend(LLMBackend):
    """Decorator: Anonymisiert den Input, de-anonymisiert den Output.

    Anonymisiert NUR die user_message, NICHT den system_prompt.
    Grund: In den bestehenden Pfaden ist der System-Prompt ein trusted Template
    aus DB/Admin-Konfiguration. Falls spaeter untrusted/user-generierter
    System-Prompt eingefuehrt wird, muss diese Annahme neu bewertet werden.

    WICHTIG — Streaming-Semantik:
    De-Anonymisierung erfordert Pufferung des gesamten Outputs, da Platzhalter
    ueber Token-Grenzen hinweg aufgeteilt sein koennen (z.B. "[Person" + "_001]").
    Daher gibt generate_stream() den gesamten de-anonymisierten Text als
    einen einzigen Chunk zurueck. Das entspricht der bestehenden gepufferten
    Cloud-De-Anonymisierung in generation/unchained_chain.py, Zeilen 178-186.
    """

    def __init__(self, inner: LLMBackend, anonymizer: Anonymizer) -> None:
        """Erstellt einen anonymisierenden Backend-Wrapper.

        Args:
            inner: Das zu wrappende LLM-Backend.
            anonymizer: Initialisierter Anonymizer mit geladenen Entities.
        """
        self._inner = inner
        self._anonymizer = anonymizer
        self._last_anon_question: str = ""
        self._last_anon_answer: str = ""
        self._last_mapping: dict[str, str] = {}
        self._last_entities_anonymized: int = 0

    @property
    def provider(self) -> str:
        """Delegiert an das innere Backend."""
        return self._inner.provider

    @property
    def model(self) -> str:
        """Delegiert an das innere Backend."""
        return self._inner.model

    @property
    def last_usage(self) -> dict:
        """Delegiert an das innere Backend."""
        return self._inner.last_usage

    @property
    def last_anon_question(self) -> str:
        """Anonymisierte Frage des letzten Aufrufs (fuer Audit-Logging)."""
        return self._last_anon_question

    @property
    def last_anon_answer(self) -> str:
        """Anonymisierte Antwort des letzten Aufrufs (fuer Audit-Logging)."""
        return self._last_anon_answer

    @property
    def last_entities_anonymized(self) -> int:
        """Anzahl anonymisierter Entities des letzten Aufrufs."""
        return self._last_entities_anonymized

    @property
    def anonymizer_stats(self) -> dict:
        """Audit-kompatible Statistiken des letzten Anonymisierungslaufs."""
        return self._anonymizer.stats

    def _anonymize_input(self, user_message: str) -> str:
        """Anonymisiert die Eingabe und aktualisiert Metriken.

        Raises:
            RuntimeError: Wenn keine Stammdaten geladen sind (Automaton leer).
                Verhindert dass PII-haltiger Klartext an Cloud-LLMs gesendet wird.
        """
        self._last_anon_question = ""
        self._last_anon_answer = ""
        self._last_mapping = {}
        self._last_entities_anonymized = 0

        if (
            not hasattr(self._anonymizer, "automaton")
            or self._anonymizer.automaton is None
            or self._anonymizer.automaton.entry_count == 0
        ):
            logger.warning("Anonymisierung ohne Stammdaten — Cloud-Route blockiert")
            raise RuntimeError("Anonymisierung nicht verfuegbar (keine Stammdaten geladen)")

        clean_text = self._anonymizer.anonymize(user_message)
        result = self._anonymizer.last_result
        self._last_anon_question = clean_text
        self._last_mapping = result.mapping
        self._last_entities_anonymized = result.stats.get("total", 0)
        return clean_text

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Anonymisiert user_message, generiert, de-anonymisiert Antwort.

        Args:
            system_prompt: Wird NICHT anonymisiert (trusted Template/Admin-Konfiguration).
            user_message: Wird anonymisiert vor dem Senden.
            **kwargs: Durchgereicht an das innere Backend.

        Returns:
            De-anonymisierter Antwort-Text.

        Raises:
            RuntimeError: Wenn keine Stammdaten geladen sind (Automaton leer) — fail-closed
                gegen ungewollten PII-Versand an Cloud-LLMs.
            ConnectionError: Vom inneren Backend bei Provider-Fehlern (siehe LLMBackend-ABC).
        """
        clean_text = self._anonymize_input(user_message)

        raw = self._inner.generate(system_prompt, clean_text, **kwargs)
        self._last_anon_answer = raw

        try:
            return self._anonymizer.deanonymize(raw)
        except Exception as exc:
            logger.warning("De-Anonymisierung fehlgeschlagen: %s", exc)
            raise RuntimeError("De-Anonymisierung fehlgeschlagen") from exc

    def generate_stream(
        self, system_prompt: str, user_message: str, **kwargs
    ) -> Generator[str, None, None]:
        """Anonymisiert, generiert (gepuffert), de-anonymisiert.

        Puffert den gesamten Stream fuer korrekte De-Anonymisierung.
        Gibt den de-anonymisierten Text als einen einzigen Chunk zurueck.

        Args:
            system_prompt: Wird NICHT anonymisiert.
            user_message: Wird anonymisiert.
            **kwargs: Durchgereicht an das innere Backend.

        Yields:
            Ein einziger Chunk mit dem de-anonymisierten Text.

        Raises:
            RuntimeError: Wenn keine Stammdaten geladen sind (Automaton leer) — fail-closed
                gegen ungewollten PII-Versand an Cloud-LLMs.
            ConnectionError: Vom inneren Backend bei Provider-Fehlern (siehe LLMBackend-ABC).
        """
        clean_text = self._anonymize_input(user_message)

        # Gesamten Stream puffern fuer korrekte De-Anonymisierung
        chunks = list(
            self._inner.generate_stream(
                system_prompt,
                clean_text,
                **kwargs,
            )
        )
        full = "".join(chunks)
        self._last_anon_answer = full

        try:
            yield self._anonymizer.deanonymize(full)
        except Exception as exc:
            logger.warning("De-Anonymisierung (Stream) fehlgeschlagen: %s", exc)
            raise RuntimeError("De-Anonymisierung fehlgeschlagen") from exc
