"""Basis-Interface und gemeinsame Datenklassen fuer LLM-Tool-Adapter.

Definiert das abstrakte Adapter-Interface, ToolStep-Ergebnisse und
ToolAdapterError. Alle konkreten Adapter (Ollama, Anthropic, OpenAI)
erben von BaseToolAdapter.

Beziehungen:
    - kiara.generation.adapters.ollama / anthropic / openai erben BaseToolAdapter
    - kiara.generation.agent_chain nutzt BaseToolAdapter + ToolStep
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from kiara.shared.constants import AGENT_RESULT_TRUNCATE_CLOUD
from kiara.tools.base import ToolDefinition

# ---------------------------------------------------------------------------
# Gemeinsame Hilfsfunktion fuer TOOL_CALL-Parsing
# ---------------------------------------------------------------------------


def _extract_balanced_params(text: str, start: int) -> str | None:
    """Extrahiert Klammer-Inhalt mit Klammerbalancierung + Quote-Awareness.

    Liest ab *start* (Position direkt nach der oeffnenden '(') und zaehlt
    Klammern. Klammern innerhalb von Quotes ("..." oder '...') werden
    ignoriert. Gibt den Inhalt zwischen den balancierten Klammern zurueck
    oder None wenn die Balance nicht aufgeht.
    """
    if start < 0 or start > len(text):
        return None
    depth = 1
    in_quote: str | None = None
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if in_quote:
            if ch == "\\" and i + 1 < len(text):
                i += 1  # Skip escaped character
            elif ch == in_quote:
                in_quote = None
        elif ch in ('"', "'"):
            in_quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def _extract_balanced_value(text: str, start: int) -> str | None:
    """Extrahiert einen balancierten JSON-Wert ([...] oder {...}) ab *start*.

    Funktioniert analog zu _extract_balanced_params, nutzt aber das
    Zeichen an *start* als Oeffner und bestimmt den passenden Schliesser
    automatisch. Gibt den vollstaendigen Wert inkl. Klammern zurueck.
    """
    if start >= len(text):
        return None
    openers = {"[": "]", "{": "}"}
    opener = text[start]
    closer = openers.get(opener)
    if closer is None:
        return None
    depth = 1
    in_quote: str | None = None
    i = start + 1
    while i < len(text) and depth > 0:
        ch = text[i]
        if in_quote:
            if ch == "\\" and i + 1 < len(text):
                i += 1
            elif ch == in_quote:
                in_quote = None
        elif ch in ('"', "'"):
            in_quote = ch
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i]


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass
class ToolStep:
    """Ergebnis eines Adapter-Schritts.

    Attributes:
        tool_call: (tool_name, params_dict) oder None wenn kein Tool-Call.
        text: LLM-Text (Reasoning bei Tool-Calls, finale Antwort bei is_final).
        tool_use_id: Provider-spezifische ID (Anthropic tool_use_id, OpenAI tool_call_id).
        is_final: True wenn text die komplette (de-anonymisierte) finale Antwort ist.
        parallel_calls: Zusaetzliche parallele Tool-Calls (ohne den ersten).
                        Liste von (tool_name, params, tool_use_id).
    """

    tool_call: tuple[str, dict] | None = None
    text: str = ""
    tool_use_id: str | None = None
    is_final: bool = False
    parallel_calls: list[tuple[str, dict, str | None]] | None = None


class ToolAdapterError(Exception):
    """Fehler bei der LLM-Adapter-Kommunikation.

    Attributes:
        retryable: True wenn ein Retry sinnvoll ist (Rate-Limit, Netzwerk).
    """

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Basis-Interface
# ---------------------------------------------------------------------------


class BaseToolAdapter(ABC):
    """Abstrakte Basisklasse fuer LLM-Tool-Adapter."""

    @property
    def result_truncate_limit(self) -> int:
        """Maximale Zeichenlaenge fuer einzelne Tool-Ergebnisse.

        Cloud-Adapter: AGENT_RESULT_TRUNCATE_CLOUD (24000)
        Ollama-Adapter: ueberschreibt mit AGENT_RESULT_TRUNCATE_OLLAMA (4000)

        Wird vom Budget-Manager in agent_chain.py als Per-Result-Limit verwendet.
        """
        return AGENT_RESULT_TRUNCATE_CLOUD

    @abstractmethod
    def init_conversation(
        self,
        question: str,
        system_prompt: str,
        tools: list[ToolDefinition],
        history_block: str = "",
    ) -> None:
        """Initialisiert eine neue Konversation mit Frage und Tools."""
        ...

    @abstractmethod
    def step(self) -> ToolStep:
        """Fuehrt einen LLM-Schritt aus und gibt das Ergebnis zurueck."""
        ...

    @abstractmethod
    def add_tool_result(
        self,
        tool_name: str,
        result_text: str,
        tool_use_id: str | None = None,
        raw_chunks: list[dict] | None = None,
    ) -> None:
        """Fuegt ein Tool-Ergebnis zum Konversationskontext hinzu.

        Args:
            tool_name: Name des aufgerufenen Tools.
            result_text: Display-Text des Ergebnisses.
            tool_use_id: Provider-spezifische ID (Anthropic/OpenAI).
            raw_chunks: Optionale strukturierte Retrieval-Ergebnisse
                        (nur von SearchDocumentsTool). Werden fuer
                        sauberes Final-Prompt-Formatting verwendet.
        """
        ...

    def add_tool_results(
        self,
        results: list[tuple[str, str, str | None]],
        raw_chunks_map: dict[str | None, list[dict]] | None = None,
    ) -> None:
        """Fuegt mehrere Tool-Ergebnisse auf einmal zum Kontext hinzu.

        Default-Implementation: Sequentielle Delegation an add_tool_result().
        Cloud-Adapter ueberschreiben dies fuer natives Batch-Format.

        Args:
            results: Liste von (tool_name, result_text, tool_use_id)
            raw_chunks_map: Optional, tool_use_id -> raw_chunks
                            (Keys koennen None sein — Ollama hat keine tool_use_ids)
        """
        for tool_name, result_text, tool_use_id in results:
            chunks = (raw_chunks_map or {}).get(tool_use_id)
            self.add_tool_result(tool_name, result_text, tool_use_id, raw_chunks=chunks)

    @abstractmethod
    def deanonymize_params(self, params: dict) -> dict:
        """De-anonymisiert String-Parameter eines Tool-Calls.

        THREAD-SAFETY: Diese Methode wird bei parallelen Tool-Calls in
        mehreren Threads gleichzeitig aufgerufen. Implementierungen:
        1. Duerfen keinen shared mutable State aendern (kein self._xxx = ...).
        2. Muessen IMMER ein neues Dict zurueckgeben (kein return params),
           damit parallele Threads keine Referenz auf dasselbe Objekt erhalten.
        """
        ...

    @abstractmethod
    def stream_final(self, additional_context: str = "", think: bool = False) -> Iterator[str]:
        """Streamt eine finale Antwort basierend auf dem bisherigen Kontext.

        Args:
            additional_context: Optionaler Zusatzkontext.
            think: Wenn True, Thinking aktivieren und <think>-Bloecke durchstreamen.
        """
        ...

    @property
    def last_conversation(self) -> str:
        """Vollstaendiger Konversationsverlauf als JSON-String (fuer Audit).

        Enthaelt exakt die Daten, die an das KI-Backend gesendet wurden.
        Bei Cloud-Backends sind PII bereits anonymisiert.
        """
        return ""

    @property
    @abstractmethod
    def last_usage(self) -> dict:
        """Token-Usage aller bisherigen Calls (kumuliert)."""
        ...
