"""OpenAI-Adapter — native tool_calls API mit Pseudonymisierung.

Verwaltet Messages im OpenAI-Chat-Format, speichert Assistant-Messages
explizit (Pflicht fuer OpenAI), anonymisiert und de-anonymisiert gepuffert.

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
from kiara.generation.backends.openai import _completion_token_kwargs
from kiara.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class OpenAIToolAdapter(CloudAdapterBase):
    """Adapter fuer OpenAI GPT — native tool_calls API.

    Verwaltet Messages im OpenAI-Chat-Format, speichert Assistant-Messages
    explizit (Pflicht fuer OpenAI), anonymisiert und de-anonymisiert gepuffert.
    """

    def __init__(self, backend: dict, config: dict, anonymizer=None) -> None:
        super().__init__(backend, config, anonymizer)
        import httpx
        import openai

        from kiara.shared.constants import (
            CLOUD_CONNECT_TIMEOUT_SECS,
            CLOUD_MAX_RETRIES,
            CLOUD_READ_TIMEOUT_SECS,
        )

        self._client = openai.OpenAI(
            api_key=backend["api_key"],
            base_url=backend.get("base_url") or None,
            timeout=httpx.Timeout(CLOUD_READ_TIMEOUT_SECS, connect=CLOUD_CONNECT_TIMEOUT_SECS),
            max_retries=CLOUD_MAX_RETRIES,
        )

    def _is_retryable(self, exc: Exception) -> bool:
        """Prueft ob ein OpenAI-Fehler retryable ist."""
        try:
            import openai

            return isinstance(
                exc,
                (
                    openai.RateLimitError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                    openai.APITimeoutError,
                ),
            )
        except ImportError:
            return False

    def _append_context(self, anon_ctx: str) -> None:
        """Fuegt Kontext in die OpenAI-Messages ein (String-Concatenation)."""
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages[-1]["content"] += "\n\n" + anon_ctx
        else:
            self._messages.append({"role": "user", "content": anon_ctx})

    def _stream_chunks(self) -> Iterator[str]:
        """Streamt Text-Chunks via OpenAI Chat Completion Stream."""
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            **_completion_token_kwargs(self._model, self._max_tokens),
            temperature=self._temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                self._usage["input"] += chunk.usage.prompt_tokens or 0
                self._usage["output"] += chunk.usage.completion_tokens or 0

    @staticmethod
    def _parse_tool_arguments(raw_args: str | None, tool_name: str) -> dict:
        """Parst OpenAI-tool_call.arguments fail-closed als JSON-Objekt."""
        import json as _json

        try:
            params = _json.loads(raw_args or "{}")
        except (_json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ToolAdapterError(
                f"Ungueltige Tool-Argumente fuer {tool_name}: {exc}",
                retryable=False,
            ) from exc
        if not isinstance(params, dict):
            raise ToolAdapterError(
                f"Ungueltige Tool-Argumente fuer {tool_name}: JSON-Objekt erwartet",
                retryable=True,
            )
        return params

    def init_conversation(
        self,
        question: str,
        system_prompt: str,
        tools: list[ToolDefinition],
        history_block: str = "",
    ) -> None:
        """Initialisiert eine neue Agent-Konversation mit OpenAI Chat API."""
        effective_question = f"{history_block}\n\n{question}" if history_block else question
        anon_question = self._anonymize(effective_question)
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": anon_question},
        ]
        self._tool_name_map = self._build_tool_name_map(tools)
        self._tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": td.name.replace(".", "_"),
                    "description": td.description,
                    "parameters": td.parameters,
                },
            }
            for td in tools
        ]
        self._usage = {"input": 0, "output": 0}

    def step(self) -> ToolStep:
        """Fuehrt einen Agent-Schritt aus: sendet Chat-Completion, parst tool_calls."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=self._tool_defs,
                **_completion_token_kwargs(self._model, self._max_tokens),
                temperature=self._temperature,
            )
        except Exception as exc:
            retryable = self._is_retryable(exc)
            raise ToolAdapterError(str(exc), retryable=retryable) from exc

        if not resp.choices:
            raise ToolAdapterError(
                "OpenAI lieferte leere choices-Liste",
                retryable=True,
            )
        choice = resp.choices[0]
        if resp.usage:
            self._usage["input"] += resp.usage.prompt_tokens or 0
            self._usage["output"] += resp.usage.completion_tokens or 0

        # Inkonsistenter API-State: tool_calls vorhanden, aber finish_reason != "tool_calls"
        # (typisch: finish_reason="length" wegen Token-Limit -> partielle/korrupte Tool-Args).
        # Fail-closed: History nicht mutieren, sonst landet korrupter Assistant-Frame ohne
        # tool-Response in _messages und der naechste Step laeuft auf 400.
        if choice.message.tool_calls and choice.finish_reason != "tool_calls":
            raise ToolAdapterError(
                f"OpenAI lieferte tool_calls mit finish_reason='{choice.finish_reason}' "
                f"(erwartet 'tool_calls'); vermutlich Token-Limit, Tool-Args potenziell korrupt.",
                retryable=False,
            )

        # Assistant-Response vorbereiten (Append erst nach erfolgreichem Parsing)
        msg = choice.message.model_dump()
        if msg.get("tool_calls") is None:
            msg.pop("tool_calls", None)
        # function_call=None ebenfalls bereinigen (Legacy-Feld)
        if msg.get("function_call") is None:
            msg.pop("function_call", None)

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            # API-Namen zurueck auf Original-Namen mappen (z.B. backup_list → backup.list)
            _map = getattr(self, "_tool_name_map", {})
            # Arguments VOR State-Mutation parsen — JSONDecodeError darf _messages nicht kontaminieren
            first_name = _map.get(tc.function.name, tc.function.name)
            params = self._parse_tool_arguments(tc.function.arguments, first_name)
            parallel = None
            if len(choice.message.tool_calls) > 1:
                parallel = []
                for t in choice.message.tool_calls[1:]:
                    tool_name = _map.get(t.function.name, t.function.name)
                    parallel.append(
                        (
                            tool_name,
                            self._parse_tool_arguments(t.function.arguments, tool_name),
                            t.id,
                        )
                    )
            if parallel:
                logger.info("OpenAI: %d parallele Tool-Calls", len(parallel) + 1)
            self._messages.append(msg)
            reasoning = choice.message.content or ""
            return ToolStep(
                tool_call=(first_name, params),
                text=reasoning,
                tool_use_id=tc.id,
                parallel_calls=parallel,
            )

        # Kein Tool-Call — finale Antwort
        self._messages.append(msg)
        full_text = self._deanonymize(choice.message.content or "")
        return ToolStep(tool_call=None, text=full_text, is_final=True)

    def add_tool_result(
        self,
        tool_name: str,
        result_text: str,
        tool_use_id: str | None = None,
        raw_chunks: list[dict] | None = None,
    ) -> None:
        """Single-Call-Pfad (Forced-Search, Tests, Abwaertskompatibilitaet)."""
        display = self._truncate_result(result_text)
        anon_text = self._anonymize(display)
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": anon_text,
            }
        )

    def add_tool_results(
        self,
        results: list[tuple[str, str, str | None]],
        raw_chunks_map: dict[str | None, list[dict]] | None = None,
    ) -> None:
        """Sendet alle Tool-Ergebnisse als separate tool-Messages (OpenAI-Format).

        Truncation erfolgt bereits in agent_chain.py (Budget-basiert).
        Hier wird nur anonymisiert und formatiert.
        """
        for _tool_name, result_text, tool_use_id in results:
            anon_text = self._anonymize(result_text)
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": anon_text,
                }
            )

    @property
    def last_conversation(self) -> str:
        """Serialisiert die gesamte Konversation als JSON (fuer Debugging/Audit)."""
        import json

        return json.dumps(
            {
                "messages": self._messages,
                "tools": self._tool_defs,
            },
            ensure_ascii=False,
            default=str,
        )
