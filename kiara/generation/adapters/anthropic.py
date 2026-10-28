"""Anthropic-Adapter — native tool_use API mit Pseudonymisierung.

Verwaltet Messages im Anthropic-Format, anonymisiert Fragen/Ergebnisse
und de-anonymisiert die finale Antwort gepuffert.

Beziehungen:
    - Erbt von kiara.generation.adapters.cloud_base.CloudAdapterBase
    - Nutzt kiara.anonymization.pseudonymizer fuer PII-Schutz
"""

import logging
from collections.abc import Iterator

from kiara.generation.adapters.base import (
    ToolAdapterError,
    ToolStep,
)
from kiara.generation.adapters.cloud_base import CloudAdapterBase
from kiara.generation.backends.anthropic import _thinking_params
from kiara.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class AnthropicToolAdapter(CloudAdapterBase):
    """Adapter fuer Anthropic Claude — native tool_use API.

    Verwaltet Messages im Anthropic-Format, anonymisiert Fragen/Ergebnisse
    und de-anonymisiert die finale Antwort gepuffert.
    """

    def __init__(self, backend: dict, config: dict, anonymizer=None) -> None:
        super().__init__(backend, config, anonymizer)
        import anthropic
        import httpx

        from kiara.shared.constants import (
            CLOUD_CONNECT_TIMEOUT_SECS,
            CLOUD_MAX_RETRIES,
            CLOUD_READ_TIMEOUT_SECS,
        )

        self._client = anthropic.Anthropic(
            api_key=backend["api_key"],
            base_url=backend.get("base_url") or None,
            timeout=httpx.Timeout(CLOUD_READ_TIMEOUT_SECS, connect=CLOUD_CONNECT_TIMEOUT_SECS),
            max_retries=CLOUD_MAX_RETRIES,
        )
        self._system: str = ""

    def _is_retryable(self, exc: Exception) -> bool:
        """Prueft ob ein Anthropic-Fehler retryable ist."""
        try:
            import anthropic

            return isinstance(
                exc,
                (
                    anthropic.RateLimitError,
                    anthropic.APIConnectionError,
                    anthropic.InternalServerError,
                    anthropic.APITimeoutError,
                    anthropic.OverloadedError,
                ),
            )
        except ImportError:
            return False

    def _append_context(self, anon_ctx: str) -> None:
        """Fuegt Kontext in die Anthropic-Messages ein (content-block-Liste)."""
        # Letzte User-Message erweitern statt neue hinzuzufuegen,
        # um zwei aufeinanderfolgende user-Messages zu vermeiden
        # (Anthropic API erwartet abwechselnde Rollen).
        if (
            self._messages
            and self._messages[-1]["role"] == "user"
            and isinstance(self._messages[-1]["content"], list)
        ):
            self._messages[-1]["content"].append({"type": "text", "text": anon_ctx})
        else:
            self._messages.append({"role": "user", "content": anon_ctx})

    def _stream_chunks(self) -> Iterator[str]:
        """Streamt Text-Chunks via Anthropic Messages Stream API.

        Bei Extended Thinking (self._think=True): Event-basiertes Streaming
        mit <think>-Tags fuer Thinking-Bloecke.
        """
        params = self._build_api_params(include_tools=False)
        try:
            with self._client.messages.stream(**params) as stream:
                if not self._think:
                    yield from stream.text_stream
                    final_msg = stream.get_final_message()
                    if final_msg.usage is not None:
                        self._usage["input"] += final_msg.usage.input_tokens
                        self._usage["output"] += final_msg.usage.output_tokens
                    return
                # Event-basiertes Streaming fuer Thinking-Bloecke
                in_thinking = False
                for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "thinking":
                            in_thinking = True
                            yield "<think>"
                        elif event.content_block.type == "text" and in_thinking:
                            yield "</think>"
                            in_thinking = False
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "thinking"):
                            yield event.delta.thinking
                        elif hasattr(event.delta, "text"):
                            yield event.delta.text
                if in_thinking:
                    yield "</think>"
                final_msg = stream.get_final_message()
                if final_msg.usage is not None:
                    self._usage["input"] += final_msg.usage.input_tokens
                    self._usage["output"] += final_msg.usage.output_tokens
        except Exception as exc:
            # Fallback bei Thinking-Fehler
            if self._think and ("thinking" in str(exc).lower() or "budget" in str(exc).lower()):
                logger.warning("Extended Thinking Stream-Fehler, Fallback: %s", exc)
                self._think = False
                params = self._build_api_params(include_tools=False)
                with self._client.messages.stream(**params) as stream:
                    yield from stream.text_stream
                    final_msg = stream.get_final_message()
                    if final_msg.usage is not None:
                        self._usage["input"] += final_msg.usage.input_tokens
                        self._usage["output"] += final_msg.usage.output_tokens
            else:
                raise

    def init_conversation(
        self,
        question: str,
        system_prompt: str,
        tools: list[ToolDefinition],
        history_block: str = "",
    ) -> None:
        """Initialisiert eine neue Agent-Konversation mit Anthropic Messages API."""
        effective_question = f"{history_block}\n\n{question}" if history_block else question
        anon_question = self._anonymize(effective_question)
        self._system = system_prompt
        self._messages = [{"role": "user", "content": anon_question}]
        self._tool_name_map = self._build_tool_name_map(tools)
        self._tool_defs = [
            {"name": n, "description": td.description, "input_schema": td.parameters}
            for td, n in zip(tools, self._tool_name_map, strict=False)
        ]
        self._usage = {"input": 0, "output": 0}

    def _build_api_params(self, *, include_tools: bool = True) -> dict:
        """Baut API-Parameter mit optionalem Extended Thinking."""
        params = dict(
            model=self._model,
            system=self._system,
            messages=self._messages,
            max_tokens=self._max_tokens,
        )
        if include_tools and self._tool_defs:
            params["tools"] = self._tool_defs
        if self._think:
            # Temperature darf NICHT gesetzt werden
            params.update(_thinking_params(self._model, self._max_tokens))
        else:
            params["temperature"] = self._temperature
        return params

    def step(self) -> ToolStep:
        """Fuehrt einen Agent-Schritt aus: sendet Messages, parst Tool-Calls."""
        params = self._build_api_params()
        try:
            resp = self._client.messages.create(**params)
        except Exception as exc:
            # Fallback: Extended Thinking nicht unterstuetzt → ohne Thinking
            if self._think and ("thinking" in str(exc).lower() or "budget" in str(exc).lower()):
                logger.warning("Extended Thinking nicht unterstuetzt, Fallback: %s", exc)
                self._think = False
                params = self._build_api_params()
                try:
                    resp = self._client.messages.create(**params)
                except Exception as fallback_exc:
                    retryable = self._is_retryable(fallback_exc)
                    raise ToolAdapterError(str(fallback_exc), retryable=retryable) from fallback_exc
            else:
                retryable = self._is_retryable(exc)
                raise ToolAdapterError(str(exc), retryable=retryable) from exc

        # Thinking- und Text-Bloecke extrahieren (defensiv via getattr,
        # analog zu backends/anthropic.py — schuetzt vor Schema-Drift)
        thinking_parts = [
            getattr(b, "thinking", "")
            for b in resp.content
            if getattr(b, "type", None) == "thinking"
        ]
        text_parts = [
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        ]
        reasoning = "\n".join(text_parts)
        if thinking_parts:
            reasoning = f"<think>{''.join(thinking_parts)}</think>{reasoning}"

        # Parallele Tool-Calls: ALLE zurueckgeben (nicht mehr nur den ersten)
        tool_use_blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]

        # Bei stop_reason="max_tokens" liefert Anthropic ggf. einen abgeschnittenen
        # tool_use-Block mit unvollstaendigem input-JSON. Fail-closed VOR dem
        # _messages-Append, damit kein inkonsistenter History-State entsteht und
        # kein Tool mit truncated Argumenten ausgefuehrt wird.
        if tool_use_blocks and getattr(resp, "stop_reason", None) == "max_tokens":
            raise ToolAdapterError(
                "Anthropic-Response wurde durch max_tokens abgeschnitten waehrend "
                f"tool_use-Block (max_tokens={self._max_tokens}). Fail-closed, "
                "Tool nicht ausgefuehrt — max_tokens erhoehen oder Kontext kuerzen.",
                retryable=False,
            )

        # Anthropic-Schema: id ist Pflicht bei tool_use. Fail-fast VOR _messages-Append
        # und Usage-Buchung, damit kein invalider Assistant-Turn die History/das Audit
        # kontaminiert und keine Token-Statistik fuer einen fail-closed Schritt entsteht.
        if tool_use_blocks:
            for b in tool_use_blocks:
                if not getattr(b, "id", None):
                    raise ToolAdapterError(
                        f"Anthropic tool_use-Block ohne id (name={getattr(b, 'name', '?')!r}) — "
                        "Schema-Drift oder partieller Response-Block."
                    )

        # Assistant-Response zu Messages hinzufuegen NACH erfolgreichem Parse
        # (verhindert inkonsistente History bei Parse-Fehler)
        self._messages.append({"role": "assistant", "content": resp.content})
        if resp.usage is not None:
            self._usage["input"] += resp.usage.input_tokens
            self._usage["output"] += resp.usage.output_tokens

        if tool_use_blocks:
            first = tool_use_blocks[0]
            # API-Namen zurueck auf Original-Namen mappen (z.B. backup_list → backup.list)
            _map = getattr(self, "_tool_name_map", {})
            first_name = _map.get(getattr(first, "name", ""), getattr(first, "name", ""))
            parallel = (
                [
                    (
                        _map.get(getattr(b, "name", ""), getattr(b, "name", "")),
                        getattr(b, "input", {}),
                        b.id,
                    )
                    for b in tool_use_blocks[1:]
                ]
                if len(tool_use_blocks) > 1
                else None
            )
            if parallel:
                logger.info("Anthropic: %d parallele Tool-Calls", len(parallel) + 1)
            # Reasoning (inkl. <think>-Block) wird in agent_chain.py an den
            # User geyielded — daher hier de-anonymisieren, analog is_final.
            # _messages bleibt anonymisiert (LLM-Kontext) durch Append von resp.content oben.
            user_visible_text = self._deanonymize(reasoning)
            return ToolStep(
                tool_call=(first_name, getattr(first, "input", {})),
                text=user_visible_text,
                tool_use_id=first.id,
                parallel_calls=parallel,
            )

        # Kein Tool-Call — finale Antwort
        full_text = self._deanonymize(reasoning)
        return ToolStep(tool_call=None, text=full_text, is_final=True)

    def add_tool_result(
        self,
        tool_name: str,
        result_text: str,
        tool_use_id: str | None = None,
        raw_chunks: list[dict] | None = None,
    ) -> None:
        """Uebertraegt den zentral budgetierten Inhalt als natives Tool-Ergebnis."""
        # Anthropic-Schema: tool_use_id ist Pflicht im content-Block. Fail-closed
        # symmetrisch zur Validierung eingehender tool_use-Bloecke in step().
        if tool_use_id is None:
            raise ToolAdapterError(
                f"Anthropic add_tool_result ohne tool_use_id (tool={tool_name!r}) — "
                "tool_use_id ist im Anthropic content-Block Pflicht.",
                retryable=False,
            )
        anon_text = self.prepare_tool_text(result_text)
        content_blocks = [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": anon_text},
        ]
        self._messages.append({"role": "user", "content": content_blocks})

    def add_tool_results(
        self,
        results: list[tuple[str, str, str | None]],
        raw_chunks_map: dict[str | None, list[dict]] | None = None,
    ) -> None:
        """Sendet alle Tool-Ergebnisse in EINER user-Message (Anthropic-Pflicht).

        Truncation erfolgt bereits in agent_chain.py (Budget-basiert).
        Zentral vorbereitete Texte werden unveraendert uebernommen.
        """
        content_blocks = []
        for tool_name, result_text, tool_use_id in results:
            # Anthropic-Schema: tool_use_id ist Pflicht im content-Block. Fail-closed
            # symmetrisch zur Validierung eingehender tool_use-Bloecke in step().
            if tool_use_id is None:
                raise ToolAdapterError(
                    f"Anthropic add_tool_results enthaelt tool_use_id=None "
                    f"(tool={tool_name!r}) — tool_use_id ist im Anthropic content-Block Pflicht.",
                    retryable=False,
                )
            anon_text = self.prepare_tool_text(result_text)
            content_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": anon_text,
                }
            )
        self._messages.append({"role": "user", "content": content_blocks})

    def replace_tool_result(self, old_text, new_text, tool_use_id=None) -> bool:
        for message in self._messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") == tool_use_id
                ):
                    block["content"] = self.prepare_tool_text(new_text)
                    return True
        return False

    @property
    def last_conversation(self) -> str:
        """Serialisiert die gesamte Konversation als JSON (fuer Debugging/Audit)."""
        import json

        def _serialize(obj):
            """ContentBlock-Objekte (Pydantic BaseModel) zu dicts konvertieren."""
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            return str(obj)

        return json.dumps(
            {
                "system": self._system,
                "messages": self._messages,
                "tools": self._tool_defs,
            },
            ensure_ascii=False,
            default=_serialize,
        )
