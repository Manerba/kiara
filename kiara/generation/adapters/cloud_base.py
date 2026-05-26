"""Gemeinsame Basisklasse fuer Cloud-Adapter (Anthropic, OpenAI).

Extrahiert identische Logik aus AnthropicToolAdapter und OpenAIToolAdapter:
Anonymisierung/De-Anonymisierung, Usage-Tracking, Parameter-Handling,
gemeinsame __init__-Initialisierung und Truncation.

Beziehungen:
    - Erbt von kiara.generation.adapters.base.BaseToolAdapter
    - kiara.generation.adapters.anthropic / openai erben CloudAdapterBase
    - Ollama bleibt bei BaseToolAdapter (keine Anonymisierung, anderes Pattern)
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kiara.anonymization.base import BaseAnonymizer

from kiara.generation.adapters.base import (
    BaseToolAdapter,
    ToolAdapterError,
)
from kiara.shared.constants import (
    AGENT_RESULT_TRUNCATE_CLOUD,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
from kiara.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class CloudAdapterBase(BaseToolAdapter):
    """Gemeinsame Basisklasse fuer Cloud-LLM-Adapter (Anthropic, OpenAI).

    Stellt bereit:
    - Anonymisierung/De-Anonymisierung Hilfsmethoden
    - Gemeinsame __init__-Logik (Config-Parsing, Usage-Dict)
    - deanonymize_params() — identisch fuer alle Cloud-Adapter
    - last_usage — identisches Format fuer alle Cloud-Adapter
    - Truncation-Hilfe fuer add_tool_result()

    Subklassen muessen implementieren:
    - _create_client(backend) — Provider-spezifischen Client erstellen
    - init_conversation(), step(), add_tool_result(), add_tool_results()
    - stream_final(), last_conversation
    """

    def __init__(
        self, backend: dict, config: dict, anonymizer: BaseAnonymizer | None = None
    ) -> None:
        self._model = backend["model"]
        self._anonymizer: BaseAnonymizer | None = anonymizer
        self._requires_anonymizer = bool(config.get("_kiara_cloud_requires_anonymizer", False))

        cloud_cfg = config.get("cloud", {})
        self._max_tokens = cloud_cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        self._temperature = cloud_cfg.get("temperature", DEFAULT_LLM_TEMPERATURE)

        self._messages: list[dict] = []
        self._tool_defs: list[dict] = []
        self._usage: dict = {"input": 0, "output": 0}
        self._think: bool = False  # Extended Thinking (gesetzt von AgentChain)

    # -- Anonymisierung Hilfsmethoden --

    def _require_anonymizer(self, operation: str) -> None:
        """Bricht Cloud-Pfade ab, wenn Anonymisierung policy-seitig Pflicht ist."""
        if self._anonymizer is None and self._requires_anonymizer:
            raise ToolAdapterError(
                f"Cloud-{operation} erfordert Anonymisierung, aber kein "
                "Anonymizer ist initialisiert",
                retryable=False,
            )

    def _anonymize(self, text: str) -> str:
        """Anonymisiert Text falls ein Anonymizer konfiguriert ist."""
        if self._anonymizer:
            return self._anonymizer.anonymize(text)
        self._require_anonymizer("Anonymisierung")
        return text

    def _deanonymize(self, text: str) -> str:
        """De-anonymisiert Text falls ein Anonymizer konfiguriert ist."""
        if self._anonymizer:
            try:
                return self._anonymizer.deanonymize(text)
            except Exception as exc:
                logger.warning("De-Anonymisierung (Adapter) fehlgeschlagen: %s", exc)
                raise ToolAdapterError(
                    "Cloud-De-Anonymisierung fehlgeschlagen",
                    retryable=False,
                ) from exc
        self._require_anonymizer("De-Anonymisierung")
        return text

    # -- Gemeinsame Methoden --

    def deanonymize_params(self, params: dict) -> dict:
        """De-anonymisiert Tool-Parameter rekursiv (Strings, Listen, Dicts)."""
        if not self._anonymizer:
            self._require_anonymizer("Tool-Parameter-De-Anonymisierung")
            return {**params}  # IMMER neues Dict — kein Shared-Reference-Risiko

        def _deep(val):
            if isinstance(val, str):
                return self._anonymizer.deanonymize(val)
            if isinstance(val, dict):
                return {k: _deep(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_deep(item) for item in val]
            return val

        return {k: _deep(v) for k, v in params.items()}

    @property
    def last_usage(self) -> dict:
        """Gibt aggregierte Token-Usage-Statistiken zurueck."""
        return {
            "prompt_tokens": self._usage["input"],
            "response_tokens": self._usage["output"],
            "total_duration_ms": 0,
            "num_ctx": 0,
            "model": self._model,
        }

    def _build_tool_name_map(self, tools: list[ToolDefinition]) -> dict[str, str]:
        """Baut Tool-Name-Map: API-konformer Name → Original-Name.

        Cloud-APIs (Anthropic, OpenAI) erlauben nur [a-zA-Z0-9_-] in
        Tool-Namen. Admin-Funktionen nutzen Punkt-Notation (z.B. "backup.list")
        → Punkte werden fuer die API durch Unterstriche ersetzt.

        Returns:
            Dict {api_name: original_name} fuer Rueck-Mapping im step().
        """
        name_map = {}
        for td in tools:
            api_name = td.name.replace(".", "_")
            name_map[api_name] = td.name
        return name_map

    def _truncate_result(self, result_text: str) -> str:
        """Truncation fuer Single-Call-Pfad (add_tool_result).

        Budget-basierte Truncation fuer den Batch-Pfad (add_tool_results)
        erfolgt bereits in agent_chain.py.
        """
        display = result_text[:AGENT_RESULT_TRUNCATE_CLOUD]
        if len(result_text) > AGENT_RESULT_TRUNCATE_CLOUD:
            display += "..."
        return display

    def _collect_and_deanonymize(self, chunks: list[str]) -> str:
        """Joined gesammelte Stream-Chunks und de-anonymisiert das Ergebnis."""
        full_text = "".join(chunks)
        return self._deanonymize(full_text)

    # -- Template Methods (gemeinsames Skelett, Provider-spezifische Hooks) --

    @abstractmethod
    def _is_retryable(self, exc: Exception) -> bool:
        """Prueft ob ein Provider-Fehler retryable ist (Rate-Limit, Netzwerk)."""

    @abstractmethod
    def _append_context(self, anon_ctx: str) -> None:
        """Fuegt anonymisierten Zusatzkontext in die Messages ein.

        Provider-spezifisch: Anthropic nutzt content-block-Listen,
        OpenAI nutzt String-Concatenation.
        """

    @abstractmethod
    def _stream_chunks(self) -> Iterator[str]:
        """Streamt Text-Chunks vom Provider (provider-spezifische API).

        Wird innerhalb von stream_final() aufgerufen. Darf Exceptions
        werfen — die Fehlerbehandlung erfolgt im Template.
        """

    def stream_final(self, additional_context: str = "", think: bool = False) -> Iterator[str]:
        """Streamt die finale Antwort nach dem Tool-Loop (Template Method).

        Gemeinsames Skelett fuer alle Cloud-Adapter:
        1. Optional: Zusatzkontext anonymisieren und anhaengen
        2. Stream starten und Chunks sammeln
        3. Ergebnis de-anonymisieren und yielden
        """
        self._think = think
        if additional_context:
            self._append_context(self._anonymize(additional_context))
        try:
            chunks = list(self._stream_chunks())
        except Exception as exc:
            raise ToolAdapterError(str(exc), retryable=self._is_retryable(exc)) from exc
        final_anon_text = "".join(chunks)
        self._messages.append({"role": "assistant", "content": final_anon_text})
        yield self._deanonymize(final_anon_text)
