"""LLM-Backend ABC und gemeinsame Usage-Helfer.

Definiert das einheitliche Interface fuer alle LLM-Backends (Ollama, Anthropic,
OpenAI) sowie eine Hilfsfunktion fuer geschaetzte Token-Zahlen.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Generator


def _clean_think_blocks(text: str) -> str:
    """Entfernt <think>...</think>-Bloecke aus LLM-Antworten (Non-Streaming).

    Behandelt drei Faelle:
    1. Verschachtelte Bloecke: iteratives Entfernen bis Fixpunkt erreicht (qwen3 nested)
    2. Geschlossene Tags: <think>...</think> (greedy-minimal)
    3. Nicht geschlossene Tags: <think>... bis zum Textende
    """
    while True:
        new_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def build_usage_estimated(prompt_chars: int, response_chars: int, model: str) -> dict:
    """Baut ein Usage-Dict mit geschaetzten Token-Zahlen.

    Wird von Cloud-Backends als Fallback benutzt, wenn die Streaming-API
    keine belastbaren Usage-Felder liefert.
    """
    from kiara.shared.constants import CHARS_PER_TOKEN

    return {
        "prompt_tokens": int(prompt_chars / CHARS_PER_TOKEN),
        "response_tokens": int(response_chars / CHARS_PER_TOKEN),
        "total_duration_ms": 0,
        "num_ctx": 0,
        "model": model,
    }


class LLMBackend(ABC):
    """Einheitliches Interface fuer alle LLM-Backends.

    Alle Backends (Ollama, Anthropic, OpenAI) implementieren dieses Interface.
    Die Chains verwenden ausschliesslich dieses Interface — kein Provider-spezifischer Code.

    kwargs-Konvention (durchgereicht von den Chains):
        temperature (float): Sampling-Temperatur. Alle Provider.
        num_ctx (int): Kontextfenster-Groesse. Nur Ollama (ignoriert von Cloud-Backends).
        think (bool | None): Thinking-Toggle. Ollama (qwen3 <think>-Tags),
            Anthropic (Extended Thinking). OpenAI ignoriert den Wert.
        max_tokens (int): Maximale Antwort-Token. Nur Cloud-Backends
            (Ollama nutzt num_ctx stattdessen).
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Nicht-streamende Generierung.

        Args:
            system_prompt: System-Prompt (Rolle, Kontext-Anweisungen).
            user_message: User-Nachricht (Frage + ggf. RAG-Kontext + History).
            **kwargs: temperature, num_ctx, think, max_tokens (je nach Provider).

        Returns:
            Generierter Antwort-Text.

        Raises:
            ConnectionError: Bei Netzwerk-Fehlern zum Backend.
            RuntimeError: Bei API-Fehlern (ungueltige Credentials, Modell nicht gefunden).
        """

    @abstractmethod
    def generate_stream(
        self, system_prompt: str, user_message: str, **kwargs
    ) -> Generator[str, None, None]:
        """Streaming-Generierung.

        Args:
            system_prompt: System-Prompt (Rolle, Kontext-Anweisungen).
            user_message: User-Nachricht (Frage + ggf. RAG-Kontext + History).
            **kwargs: temperature, num_ctx, think, max_tokens (je nach Provider).

        Yields:
            Text-Chunks der LLM-Antwort.

        Raises:
            ConnectionError: Bei Netzwerk-Fehlern zum Backend.
            RuntimeError: Bei API-Fehlern.
        """

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider-Kennung: 'ollama', 'anthropic' oder 'openai'."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Modell-ID (z.B. 'qwen3:8b', 'claude-sonnet-4-6', 'gpt-4o-mini')."""

    @property
    def last_usage(self) -> dict:
        """Token-Usage-Daten des letzten generate/generate_stream Aufrufs.

        Returns:
            Dict mit Keys: prompt_tokens, response_tokens, total_duration_ms,
            num_ctx, model. Leeres Dict wenn noch kein Aufruf erfolgt ist.
        """
        return {}
