"""Ollama-Backend ueber die offizielle /api/chat-API (Messages-Format).

Nutzt /api/chat (Messages), NICHT /api/generate (Single-Prompt).
Der Vorteil: Nativer system/user-Message-Support ohne String-Konkatenation.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from importlib import import_module

from kiara.generation.backends.base import LLMBackend, _clean_think_blocks
from kiara.shared.constants import (
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_LLM_TEMPERATURE,
    OLLAMA_CONNECT_TIMEOUT_SECS,
    OLLAMA_POOL_TIMEOUT_SECS,
    OLLAMA_READ_TIMEOUT_SECS,
    OLLAMA_WRITE_TIMEOUT_SECS,
)

logger = logging.getLogger(__name__)


def _import_ollama():
    try:
        return import_module("ollama")
    except ImportError as exc:
        raise RuntimeError("Optionale Dependency 'ollama' ist nicht installiert") from exc


def _import_httpx():
    try:
        return import_module("httpx")
    except ImportError as exc:
        raise RuntimeError("Optionale Dependency 'httpx' ist nicht installiert") from exc


def build_usage_from_ollama(response: dict, num_ctx: int, model: str) -> dict:
    """Baut ein einheitliches Usage-Dict aus einer Ollama-Response.

    Wird sowohl fuer Streaming (done-Chunk) als auch Non-Streaming benutzt.

    Returns:
        Dict mit prompt_tokens, response_tokens, total_duration_ms, num_ctx, model.
    """
    return {
        "prompt_tokens": response.get("prompt_eval_count", 0),
        "response_tokens": response.get("eval_count", 0),
        "total_duration_ms": (response.get("total_duration", 0) or 0) // 1_000_000,
        "num_ctx": num_ctx,
        "model": model,
    }


class OllamaBackend(LLMBackend):
    """Ollama-Backend ueber /api/chat (Messages-Format).

    Nutzt das ollama-Python-SDK mit client.chat() statt client.generate().
    Unterstuetzt Think-Handling fuer qwen3-Modelle (<think>-Tags).
    """

    def __init__(
        self,
        url: str,
        model: str,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        num_ctx: int = DEFAULT_LLM_NUM_CTX,
    ) -> None:
        """Erstellt einen Ollama-Backend-Client.

        Args:
            url: Ollama-Server-URL (z.B. 'http://gpu-server.local:11434').
            model: Modell-ID (z.B. 'qwen3:8b').
            temperature: Default-Temperatur (ueberschreibbar per kwarg).
            num_ctx: Default-Kontextfenster (ueberschreibbar per kwarg).
        """
        self._url = url
        self._httpx = _import_httpx()
        self._ollama_client = _import_ollama()
        self._client = self._make_client()
        self._model = model
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._last_usage: dict = {}

    def _make_client(self, timeout: float | None = None):
        """Erstellt einen Ollama-Client mit Default- oder Call-spezifischem Timeout."""
        if timeout is None:
            http_timeout = self._httpx.Timeout(
                connect=OLLAMA_CONNECT_TIMEOUT_SECS,
                read=OLLAMA_READ_TIMEOUT_SECS,
                write=OLLAMA_WRITE_TIMEOUT_SECS,
                pool=OLLAMA_POOL_TIMEOUT_SECS,
            )
        else:
            if timeout <= 0:
                raise ValueError("timeout muss groesser 0 sein")
            http_timeout = self._httpx.Timeout(
                connect=timeout,
                read=timeout,
                write=timeout,
                pool=timeout,
            )
        return self._ollama_client.Client(host=self._url, timeout=http_timeout)

    @property
    def provider(self) -> str:
        """Gibt 'ollama' zurueck."""
        return "ollama"

    @property
    def model(self) -> str:
        """Gibt die Modell-ID zurueck."""
        return self._model

    @property
    def last_usage(self) -> dict:
        """Token-Usage-Daten des letzten Aufrufs."""
        return self._last_usage

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Nicht-streamende Generierung ueber Ollama /api/chat.

        kwargs:
            temperature (float): Ueberschreibt Default.
            num_ctx (int): Ueberschreibt Default.
            think (bool | None): Thinking-Toggle (qwen3). None = nicht senden.
            timeout (float | None): Optionaler HTTP-Timeout nur fuer diesen Aufruf.

        Returns:
            Generierter Text (ohne <think>-Bloecke).
        """
        self._last_usage = {}
        temperature = kwargs.get("temperature", self._temperature)
        num_ctx = kwargs.get("num_ctx", self._num_ctx)
        timeout = kwargs.get("timeout")

        # Messages bauen — leere System-Messages weglassen
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        chat_kwargs = dict(
            model=self._model,
            messages=messages,
            options={"temperature": temperature, "num_ctx": num_ctx},
            stream=False,
        )
        if "max_tokens" in kwargs:
            if type(kwargs["max_tokens"]) is not int or kwargs["max_tokens"] < 1:
                raise ValueError("max_tokens muss eine positive ganze Zahl sein")
            chat_kwargs["options"]["num_predict"] = kwargs["max_tokens"]

        # Think-Parameter nur senden wenn explizit gesetzt
        think = kwargs.get("think")
        if think is not None:
            chat_kwargs["think"] = think

        if timeout is not None:
            client = self._make_client(timeout)
            try:
                response = client.chat(**chat_kwargs)
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        logger.debug("Ollama temp-client close() fehlgeschlagen: %s", exc)
        else:
            response = self._client.chat(**chat_kwargs)

        self._last_usage = build_usage_from_ollama(response, num_ctx, self._model)

        text = response.get("message", {}).get("content") or ""
        return _clean_think_blocks(text)

    def generate_stream(
        self, system_prompt: str, user_message: str, **kwargs
    ) -> Generator[str, None, None]:
        """Streaming-Generierung ueber Ollama /api/chat.

        Implementiert Think-Handling fuer qwen3-Modelle:
        - chunk["message"]["content"] enthaelt den normalen Antwort-Text
        - chunk["message"]["thinking"] enthaelt den Think-Text (nur bei think=True)
        - Wrapping in <think>...</think> Tags fuer korrekte Frontend-Darstellung

        kwargs:
            temperature (float): Ueberschreibt Default.
            num_ctx (int): Ueberschreibt Default.
            think (bool | None): Thinking-Toggle. None = nicht senden.
            timeout (float | None): Optionaler HTTP-Timeout nur fuer diesen Stream.

        Yields:
            Text-Chunks (inkl. <think>...</think> Wrapping wenn think=True).
        """
        self._last_usage = {}
        temperature = kwargs.get("temperature", self._temperature)
        num_ctx = kwargs.get("num_ctx", self._num_ctx)
        timeout = kwargs.get("timeout")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        chat_kwargs = dict(
            model=self._model,
            messages=messages,
            options={"temperature": temperature, "num_ctx": num_ctx},
            stream=True,
        )

        think = kwargs.get("think")
        if think is not None:
            chat_kwargs["think"] = think

        temp_client = self._make_client(timeout) if timeout is not None else None
        client = temp_client if temp_client is not None else self._client
        stream = None

        try:
            stream = client.chat(**chat_kwargs)
            in_think = False
            for chunk in stream:
                message = chunk.get("message", {})
                # Think-Chunks (qwen3 mit think=True)
                # Bei /api/chat liegt thinking unter message, nicht auf Root-Level
                thinking = message.get("thinking", "")
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                # Regulaere Antwort-Chunks
                content = message.get("content", "")
                if content:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield content
                # Done-Chunk mit Usage-Daten
                if chunk.get("done"):
                    if in_think:
                        yield "</think>"
                        in_think = False
                    self._last_usage = build_usage_from_ollama(chunk, num_ctx, self._model)
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as exc:
                    logger.debug("Ollama stream.close() fehlgeschlagen: %s", exc)
            if temp_client is not None:
                close = getattr(temp_client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        logger.debug("Ollama temp-client close() fehlgeschlagen: %s", exc)
