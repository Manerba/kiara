"""OpenAI-Adapter — native Function Calls mit Pseudonymisierung.

Verwendet fuer Responses-only-Modelle das Item-Datenmodell der Responses API
und fuer OpenAI-kompatible Anbieter weiterhin Chat Completions.

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
from kiara.generation.backends.openai import (
    _completion_token_kwargs,
    _response_error_message,
    _temperature_kwargs,
    _uses_responses_api,
)
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
        self._responses_api = _uses_responses_api(self._model)
        self._instructions = ""

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
        if self._messages and self._messages[-1].get("role") == "user":
            self._messages[-1]["content"] += "\n\n" + anon_ctx
        else:
            self._messages.append({"role": "user", "content": anon_ctx})

    def _stream_chunks(self) -> Iterator[str]:
        """Streamt Text-Chunks ueber die zum Modell passende OpenAI-API."""
        if self._responses_api:
            yield from self._stream_responses_chunks()
            return

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            **_completion_token_kwargs(self._model, self._max_tokens),
            **_temperature_kwargs(self._model, self._temperature),
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                self._usage["input"] += chunk.usage.prompt_tokens or 0
                self._usage["output"] += chunk.usage.completion_tokens or 0

    def _responses_params(self, *, include_tools: bool) -> dict:
        """Baut gemeinsame Parameter fuer Responses-Agent-Aufrufe."""
        params = {
            "model": self._model,
            "input": self._messages,
            "max_output_tokens": self._max_tokens,
            **_temperature_kwargs(self._model, self._temperature),
        }
        if self._instructions:
            params["instructions"] = self._instructions
        if include_tools:
            params["tools"] = self._tool_defs
        return params

    def _track_responses_usage(self, usage) -> None:
        """Aggregiert Usage aus einer Responses-Antwort oder einem Abschluss-Event."""
        if usage is None:
            return
        self._usage["input"] += getattr(usage, "input_tokens", 0) or 0
        self._usage["output"] += getattr(usage, "output_tokens", 0) or 0

    def _stream_responses_chunks(self) -> Iterator[str]:
        """Streamt typisierte Text-Events aus der Responses API."""
        stream = self._client.responses.create(
            **self._responses_params(include_tools=False),
            stream=True,
        )
        try:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    yield event.delta
                elif event_type == "response.completed":
                    self._track_responses_usage(getattr(event.response, "usage", None))
                elif event_type in {"response.incomplete", "response.failed"}:
                    message = _response_error_message(event.response)
                    raise ToolAdapterError(
                        message or "OpenAI-Response fehlgeschlagen",
                        retryable=False,
                    )
                elif event_type == "error":
                    raise ToolAdapterError(
                        f"OpenAI-Streaming-Fehler: {event.message}",
                        retryable=False,
                    )
        finally:
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as exc:
                    logger.debug("OpenAI Responses stream.close() fehlgeschlagen: %s", exc)

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
        """Initialisiert eine neue Agent-Konversation mit der passenden API."""
        effective_question = f"{history_block}\n\n{question}" if history_block else question
        anon_question = self._anonymize(effective_question)
        self._tool_name_map = self._build_tool_name_map(tools)
        self._instructions = system_prompt
        if self._responses_api:
            self._messages = [{"role": "user", "content": anon_question}]
            self._tool_defs = [
                {
                    "type": "function",
                    "name": td.name.replace(".", "_"),
                    "description": td.description,
                    "parameters": td.parameters,
                    # Chat Completions war nicht-strikt. Explizit beibehalten,
                    # weil Responses Function Tools sonst strict bevorzugen.
                    "strict": False,
                }
                for td in tools
            ]
        else:
            self._messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": anon_question},
            ]
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
        """Fuehrt einen Agent-Schritt mit der zum Modell passenden API aus."""
        if self._responses_api:
            return self._step_responses()
        return self._step_chat()

    @staticmethod
    def _response_item_value(item, name: str, default=None):
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _dump_response_item(item) -> dict:
        """Serialisiert ein Responses-Output-Item fuer den naechsten Turn."""
        if isinstance(item, dict):
            return {key: value for key, value in item.items() if value is not None}
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            return model_dump(exclude_none=True)
        raise ToolAdapterError(
            "OpenAI lieferte ein nicht serialisierbares Response-Item",
            retryable=False,
        )

    def _step_responses(self) -> ToolStep:
        """Sendet einen Responses-Request und parst Function-Call-Items."""
        try:
            resp = self._client.responses.create(**self._responses_params(include_tools=True))
        except Exception as exc:
            retryable = self._is_retryable(exc)
            raise ToolAdapterError(str(exc), retryable=retryable) from exc

        error_message = _response_error_message(resp)
        if error_message:
            raise ToolAdapterError(error_message, retryable=False)

        output_items = list(getattr(resp, "output", None) or [])
        if not output_items:
            raise ToolAdapterError(
                "OpenAI lieferte eine leere output-Liste",
                retryable=True,
            )
        self._track_responses_usage(getattr(resp, "usage", None))

        parsed_calls: list[tuple[str, dict, str]] = []
        for item in output_items:
            if self._response_item_value(item, "type") != "function_call":
                continue
            if self._response_item_value(item, "status") == "incomplete":
                raise ToolAdapterError(
                    "OpenAI lieferte einen unvollstaendigen Function Call; "
                    "Tool-Argumente potenziell korrupt.",
                    retryable=False,
                )
            api_name = self._response_item_value(item, "name", "")
            tool_name = self._tool_name_map.get(api_name, api_name)
            call_id = self._response_item_value(item, "call_id")
            if not call_id:
                raise ToolAdapterError(
                    f"OpenAI Function Call fuer {tool_name or '?'} ohne call_id",
                    retryable=False,
                )
            params = self._parse_tool_arguments(
                self._response_item_value(item, "arguments"),
                tool_name,
            )
            parsed_calls.append((tool_name, params, call_id))

        # Reasoning-, Message- und Function-Call-Items muessen gemeinsam fuer
        # den Folgeturn erhalten bleiben. Erst nach komplett erfolgreichem
        # Parsing mutieren, damit kaputte Tool-Argumente den State nicht vergiften.
        dumped_items = [self._dump_response_item(item) for item in output_items]
        self._messages.extend(dumped_items)

        text = getattr(resp, "output_text", "") or ""
        if parsed_calls:
            first_name, first_params, first_id = parsed_calls[0]
            parallel = parsed_calls[1:] or None
            if parallel:
                logger.info("OpenAI: %d parallele Tool-Calls", len(parsed_calls))
            return ToolStep(
                tool_call=(first_name, first_params),
                text=text,
                tool_use_id=first_id,
                parallel_calls=parallel,
            )

        return ToolStep(
            tool_call=None,
            text=self._deanonymize(text),
            is_final=True,
        )

    def _step_chat(self) -> ToolStep:
        """Sendet eine Chat Completion und parst deren tool_calls."""
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=self._tool_defs,
                **_completion_token_kwargs(self._model, self._max_tokens),
                **_temperature_kwargs(self._model, self._temperature),
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
        """Fuegt ein Tool-Ergebnis im Format der aktiven OpenAI-API hinzu."""
        anon_text = self.prepare_tool_text(result_text)
        if self._responses_api:
            self._messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": anon_text,
                }
            )
        else:
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
        Zentral vorbereitete Texte werden unveraendert uebernommen.
        """
        for _tool_name, result_text, tool_use_id in results:
            anon_text = self.prepare_tool_text(result_text)
            if self._responses_api:
                self._messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": anon_text,
                    }
                )
            else:
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": anon_text,
                    }
                )

    def replace_tool_result(self, old_text, new_text, tool_use_id=None) -> bool:
        for message in self._messages:
            if self._responses_api:
                if (
                    message.get("type") == "function_call_output"
                    and message.get("call_id") == tool_use_id
                ):
                    message["output"] = self.prepare_tool_text(new_text)
                    return True
            elif message.get("role") == "tool" and message.get("tool_call_id") == tool_use_id:
                message["content"] = self.prepare_tool_text(new_text)
                return True
        return False

    @property
    def last_conversation(self) -> str:
        """Serialisiert die gesamte Konversation als JSON (fuer Debugging/Audit)."""
        import json

        if self._responses_api:
            return json.dumps(
                {
                    "instructions": self._instructions,
                    "input": self._messages,
                    "tools": self._tool_defs,
                },
                ensure_ascii=False,
                default=str,
            )
        return json.dumps(
            {
                "messages": self._messages,
                "tools": self._tool_defs,
            },
            ensure_ascii=False,
            default=str,
        )
