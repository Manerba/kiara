"""Generischer Client fuer Cloud-KI-APIs (OpenAI, Anthropic).

Sendet nur anonymisierten Text. Unterstuetzt Streaming.
Nutzt httpx.AsyncClient — NICHT die offiziellen SDKs (minimale Abhaengigkeiten).
Retry mit exponentiellem Backoff (3 Versuche).
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

import httpx

from kiara.shared.constants import (
    ANTHROPIC_API_VERSION,
    CLOUD_CONNECT_TIMEOUT_SECS,
    CLOUD_MAX_RETRIES,
    CLOUD_READ_TIMEOUT_SECS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_OPENAI_BASE = "https://api.openai.com"
_ANTHROPIC_BASE = "https://api.anthropic.com"

_RETRY_STATUSES = {429, 500, 502, 503, 504}


async def _raise_for_status_with_detail(resp: httpx.Response) -> None:
    """raise_for_status() mit API-Error-Details im Message.

    Parst Anthropic/OpenAI JSON-Error-Bodies und fuegt error.message
    in die HTTPStatusError-Message ein.
    """
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            try:
                err = resp.json()
            except httpx.ResponseNotRead:
                await resp.aread()
                err = resp.json()
            if isinstance(err.get("error"), dict):
                detail = err["error"].get("message", "")
        except Exception:
            pass
        if detail:
            raise httpx.HTTPStatusError(
                f"{exc} — {detail}", request=exc.request, response=exc.response
            ) from None
        raise


class CloudKIClient:
    """Generischer Client fuer Cloud-KI-APIs (OpenAI, Anthropic).

    Sendet nur anonymisierten Text. Unterstuetzt Streaming.
    """

    def __init__(self, provider: str, api_key: str, model: str, **kwargs):
        """
        Args:
            provider: 'openai' oder 'anthropic'
            api_key: API-Schluessel
            model: Modell-ID (z.B. 'gpt-4o-mini', 'claude-sonnet-4-6')
            **kwargs: base_url, max_tokens, system_prompt
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = kwargs.get("base_url", "")
        self.max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.system_prompt = kwargs.get("system_prompt", "")
        self._last_tokens_in = 0
        self._last_tokens_out = 0
        self._last_duration_ms = 0

    async def generate(
        self, prompt: str, context: str = "", temperature: float = DEFAULT_LLM_TEMPERATURE
    ) -> str:
        """Nicht-streamende Anfrage.

        Args:
            prompt: Die Frage / der Prompt.
            context: Optionaler Kontext (RAG-Chunks).
            temperature: Sampling-Temperatur.

        Returns:
            Die generierte Antwort als String.
        """
        start = time.perf_counter()

        if self.provider == "openai":
            result = await self._openai_generate(prompt, context, temperature)
        elif self.provider == "anthropic":
            result = await self._anthropic_generate(prompt, context, temperature)
        else:
            from kiara.exceptions import GenerationError

            raise GenerationError(f"Unbekannter Provider: {self.provider}")

        self._last_duration_ms = int((time.perf_counter() - start) * 1000)
        return result

    async def generate_stream(
        self, prompt: str, context: str = "", temperature: float = DEFAULT_LLM_TEMPERATURE
    ) -> AsyncGenerator[str, None]:
        """Streamende Anfrage.

        Args:
            prompt: Die Frage / der Prompt.
            context: Optionaler Kontext.
            temperature: Sampling-Temperatur.

        Yields:
            Text-Chunks der Antwort.
        """
        start = time.perf_counter()

        if self.provider == "openai":
            async for chunk in self._openai_stream(prompt, context, temperature):
                yield chunk
        elif self.provider == "anthropic":
            async for chunk in self._anthropic_stream(prompt, context, temperature):
                yield chunk
        else:
            from kiara.exceptions import GenerationError

            raise GenerationError(f"Unbekannter Provider: {self.provider}")

        self._last_duration_ms = int((time.perf_counter() - start) * 1000)

    @property
    def last_usage(self) -> dict:
        """Token-Verbrauch der letzten Anfrage."""
        return {
            "tokens_in": self._last_tokens_in,
            "tokens_out": self._last_tokens_out,
            "duration_ms": self._last_duration_ms,
        }

    # -- OpenAI-kompatible API (/v1/chat/completions) --------------------------

    def _openai_url(self) -> str:
        base = self.base_url.rstrip("/") if self.base_url else _OPENAI_BASE
        return f"{base}/v1/chat/completions"

    def _openai_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _openai_body(self, prompt: str, context: str, temperature: float, stream: bool) -> dict:
        user_msg = f"Kontext:\n{context}\n\nFrage: {prompt}" if context else prompt
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "stream": stream,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ],
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def _openai_generate(self, prompt: str, context: str, temperature: float) -> str:
        body = self._openai_body(prompt, context, temperature, stream=False)
        data = await self._request("POST", self._openai_url(), self._openai_headers(), body)
        # Token-Tracking
        usage = data.get("usage", {})
        self._last_tokens_in = usage.get("prompt_tokens", 0)
        self._last_tokens_out = usage.get("completion_tokens", 0)

        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise RuntimeError(
                f"OpenAI-API: unerwartete Response-Struktur (keine choices): {str(data)[:200]}"
            )
        return choices[0].get("message", {}).get("content", "")

    async def _openai_stream(
        self, prompt: str, context: str, temperature: float
    ) -> AsyncGenerator[str, None]:
        body = self._openai_body(prompt, context, temperature, stream=True)
        self._last_tokens_in = 0
        self._last_tokens_out = 0
        chunks_yielded = 0

        async for line in self._stream_request(
            "POST", self._openai_url(), self._openai_headers(), body
        ):
            # SSE-Format: jede Zeile beginnt mit "data: ", Payload danach
            if line.startswith("data: "):
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("Malformed SSE-Payload uebersprungen: %s", payload[:100])
                    continue
                # Provider-Error-Event im Stream → Exception werfen statt silent failure
                if isinstance(chunk.get("error"), dict):
                    err = chunk["error"]
                    err_msg = err.get("message") or err.get("type") or "unknown error"
                    from kiara.exceptions import GenerationError

                    raise GenerationError(f"OpenAI-Stream-Fehler: {err_msg}")
                # Defensiver Zugriff: choices kann leer sein bei Usage-/Heartbeat-Chunks
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    chunks_yielded += 1
                    yield text
                # OpenAI liefert usage nur im allerletzten Chunk (stream_options)
                usage = chunk.get("usage")
                if usage:
                    self._last_tokens_in = usage.get("prompt_tokens", 0)
                    self._last_tokens_out = usage.get("completion_tokens", 0)

        if chunks_yielded > 0 and self._last_tokens_out == 0:
            logger.warning(
                "OpenAI-Stream: usage-Event fehlt "
                "(Streaming-Abbruch?). %d Text-Chunks empfangen, "
                "Token-Tracking unvollstaendig.",
                chunks_yielded,
            )

    # -- Anthropic Messages API (/v1/messages) ---------------------------------

    def _anthropic_url(self) -> str:
        base = self.base_url.rstrip("/") if self.base_url else _ANTHROPIC_BASE
        return f"{base}/v1/messages"

    def _anthropic_headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json",
        }

    def _anthropic_body(self, prompt: str, context: str, temperature: float, stream: bool) -> dict:
        user_msg = f"Kontext:\n{context}\n\nFrage: {prompt}" if context else prompt
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "stream": stream,
            "system": self.system_prompt,
            "messages": [
                {"role": "user", "content": user_msg},
            ],
        }

    async def _anthropic_generate(self, prompt: str, context: str, temperature: float) -> str:
        body = self._anthropic_body(prompt, context, temperature, stream=False)
        data = await self._request("POST", self._anthropic_url(), self._anthropic_headers(), body)
        # Token-Tracking
        usage = data.get("usage", {})
        self._last_tokens_in = usage.get("input_tokens", 0)
        self._last_tokens_out = usage.get("output_tokens", 0)

        content = data.get("content", [])
        text_blocks = [
            c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
        ]
        return "".join(text_blocks)

    async def _anthropic_stream(
        self, prompt: str, context: str, temperature: float
    ) -> AsyncGenerator[str, None]:
        body = self._anthropic_body(prompt, context, temperature, stream=True)
        self._last_tokens_in = 0
        self._last_tokens_out = 0
        chunks_yielded = 0

        async for line in self._stream_request(
            "POST", self._anthropic_url(), self._anthropic_headers(), body
        ):
            if line.startswith("data: "):
                payload = line[6:].strip()
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("Malformed SSE-Payload uebersprungen: %s", payload[:100])
                    continue
                event_type = event.get("type", "")

                # Anthropic SSE Event-Reihenfolge:
                # 1. message_start      → input_tokens (einmalig)
                # 2. content_block_delta → Text-Chunks (mehrfach)
                # 3. message_delta       → output_tokens (einmalig, am Ende)
                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        chunks_yielded += 1
                        yield text

                elif event_type == "message_delta":
                    usage = event.get("usage", {})
                    self._last_tokens_out = usage.get("output_tokens", 0)

                elif event_type == "message_start":
                    msg = event.get("message", {})
                    usage = msg.get("usage", {})
                    self._last_tokens_in = usage.get("input_tokens", 0)

                elif event_type == "error":
                    # Provider-Error-Event im Stream → Exception werfen statt silent failure
                    err = event.get("error", {}) if isinstance(event.get("error"), dict) else {}
                    err_msg = err.get("message") or err.get("type") or "unknown error"
                    from kiara.exceptions import GenerationError

                    raise GenerationError(f"Anthropic-Stream-Fehler: {err_msg}")

        if chunks_yielded > 0 and self._last_tokens_out == 0:
            logger.warning(
                "Anthropic-Stream: message_delta mit output_tokens fehlt "
                "(Netzwerkfehler?). %d Text-Chunks empfangen, "
                "Token-Tracking unvollstaendig.",
                chunks_yielded,
            )

    # -- HTTP-Schicht mit Retry ------------------------------------------------

    async def _request(self, method: str, url: str, headers: dict, body: dict) -> dict:
        """HTTP-Request mit exponentiellem Backoff-Retry."""
        timeout = httpx.Timeout(
            connect=CLOUD_CONNECT_TIMEOUT_SECS, read=CLOUD_READ_TIMEOUT_SECS, write=30.0, pool=30.0
        )

        # Exponentieller Backoff: attempt 1→2s, 2→4s, 3→8s (2^attempt)
        for attempt in range(1, CLOUD_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.request(method, url, headers=headers, json=body)

                    if resp.status_code in _RETRY_STATUSES and attempt < CLOUD_MAX_RETRIES:
                        wait = min(2**attempt, 300)
                        logger.warning(
                            "Cloud-API %d, Retry %d/%d in %ds",
                            resp.status_code,
                            attempt,
                            CLOUD_MAX_RETRIES,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    # Letzter Versuch oder kein Retry-Status → raise bei Fehler
                    await _raise_for_status_with_detail(resp)
                    return resp.json()

            except httpx.TimeoutException:
                if attempt < CLOUD_MAX_RETRIES:
                    wait = min(2**attempt, 300)
                    logger.warning(
                        "Cloud-API Timeout, Retry %d/%d in %ds", attempt, CLOUD_MAX_RETRIES, wait
                    )
                    await asyncio.sleep(wait)
                    continue
                raise  # Letzter Versuch: Exception propagieren

        from kiara.exceptions import GenerationError

        raise GenerationError("Max retries exceeded")

    async def _stream_request(
        self, method: str, url: str, headers: dict, body: dict
    ) -> AsyncGenerator[str, None]:
        """Streaming HTTP-Request mit SSE-Parsing und Connect-Retry.

        Retry erfolgt nur VOR dem ersten yield (Connect-Phase).
        Nach dem ersten gesendeten Chunk wird nicht mehr retried,
        um doppelten Output zu vermeiden.
        """
        timeout = httpx.Timeout(
            connect=CLOUD_CONNECT_TIMEOUT_SECS, read=CLOUD_READ_TIMEOUT_SECS, write=30.0, pool=30.0
        )

        for attempt in range(1, CLOUD_MAX_RETRIES + 1):
            yielded = False
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(method, url, headers=headers, json=body) as resp:
                        if resp.status_code in _RETRY_STATUSES and attempt < CLOUD_MAX_RETRIES:
                            wait = min(2**attempt, 300)
                            logger.warning(
                                "Cloud-API Streaming %d, Retry %d/%d in %ds",
                                resp.status_code,
                                attempt,
                                CLOUD_MAX_RETRIES,
                                wait,
                            )
                            await asyncio.sleep(wait)
                            continue

                        await _raise_for_status_with_detail(resp)
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if line:
                                yielded = True
                                yield line
                        return  # Erfolg — Generator beenden

            except httpx.TimeoutException:
                if yielded or attempt >= CLOUD_MAX_RETRIES:
                    raise  # Mid-stream oder letzter Versuch: propagieren
                wait = min(2**attempt, 300)
                logger.warning(
                    "Cloud-API Streaming Timeout, Retry %d/%d in %ds",
                    attempt,
                    CLOUD_MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
                continue

        from kiara.exceptions import GenerationError

        raise GenerationError("Max retries exceeded (streaming)")
