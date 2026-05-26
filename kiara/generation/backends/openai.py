"""OpenAI-Backend (GPT) ueber die offizielle openai-SDK.

Unterstuetzt base_url fuer OpenAI-kompatible Endpoints (z.B. interne Proxies/Gateways).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from importlib import import_module

from kiara.generation.backends.base import LLMBackend, build_usage_estimated
from kiara.shared.constants import (
    CLOUD_CONNECT_TIMEOUT_SECS,
    CLOUD_MAX_RETRIES,
    CLOUD_READ_TIMEOUT_SECS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def _import_openai():
    try:
        return import_module("openai")
    except ImportError as exc:
        raise RuntimeError("Optionale Dependency 'openai' ist nicht installiert") from exc


def _uses_max_completion_tokens(model: str) -> bool:
    """OpenAI o-Series Chat-Completions-Modelle akzeptieren kein max_tokens."""
    model_name = (model or "").split("/")[-1].lower()
    return re.match(r"^o(?:1|3|4)(?:[-_].*)?$", model_name) is not None


def _completion_token_kwargs(model: str, max_tokens: int) -> dict:
    if _uses_max_completion_tokens(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


class OpenAIBackend(LLMBackend):
    """OpenAI-Backend (GPT) ueber die offizielle SDK.

    Unterstuetzt base_url fuer OpenAI-kompatible Endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        base_url: str = "",
    ) -> None:
        """Erstellt einen OpenAI-Backend-Client.

        Args:
            api_key: Entschluesselter API-Key (Klartext).
            model: Modell-ID (z.B. 'gpt-4o-mini').
            max_tokens: Maximale Antwort-Token.
            temperature: Default-Temperatur.
            base_url: Optionale Base-URL (leer = offizielle OpenAI-API).
        """
        openai = _import_openai()
        import httpx  # dependency of openai SDK

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=httpx.Timeout(CLOUD_READ_TIMEOUT_SECS, connect=CLOUD_CONNECT_TIMEOUT_SECS),
            max_retries=CLOUD_MAX_RETRIES,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._last_usage: dict = {}

    @property
    def provider(self) -> str:
        """Gibt 'openai' zurueck."""
        return "openai"

    @property
    def model(self) -> str:
        """Gibt die Modell-ID zurueck."""
        return self._model

    @property
    def last_usage(self) -> dict:
        """Token-Usage-Daten des letzten Aufrufs (real oder Fallback-Schaetzung)."""
        return self._last_usage

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Nicht-streamende Generierung ueber OpenAI Chat Completions API.

        kwargs:
            temperature (float): Ueberschreibt Default.
            max_tokens (int): Ueberschreibt Default.

        Returns:
            Generierter Text.

        Raises:
            openai.AuthenticationError: Bei ungueltigem API-Key.
            openai.APIError: Bei API-Fehlern.
        """
        self._last_usage = {}
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # Leere System-Messages weglassen (wie Ollama-Backend)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        resp = self._client.chat.completions.create(
            model=self._model,
            **_completion_token_kwargs(self._model, max_tokens),
            temperature=temperature,
            messages=messages,
        )

        text = (resp.choices[0].message.content or "") if resp.choices else ""

        # Usage-Tracking: reale Werte bevorzugen, sonst Schaetzung
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._last_usage = {
                "prompt_tokens": usage.prompt_tokens or 0,
                "response_tokens": usage.completion_tokens or 0,
                "total_duration_ms": 0,
                "num_ctx": 0,
                "model": self._model,
            }
        else:
            self._last_usage = build_usage_estimated(
                len(system_prompt) + len(user_message),
                len(text),
                self._model,
            )

        return text

    def generate_stream(
        self, system_prompt: str, user_message: str, **kwargs
    ) -> Generator[str, None, None]:
        """Streaming-Generierung ueber OpenAI Chat Completions API.

        kwargs:
            temperature (float): Ueberschreibt Default.
            max_tokens (int): Ueberschreibt Default.

        Yields:
            Text-Chunks der Antwort.
        """
        self._last_usage = {}
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        # Leere System-Messages weglassen (wie Ollama-Backend)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        response_chunks = []
        real_usage = None
        stream = self._client.chat.completions.create(
            model=self._model,
            **_completion_token_kwargs(self._model, max_tokens),
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            messages=messages,
        )
        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    response_chunks.append(text)
                    yield text
                if getattr(chunk, "usage", None) is not None:
                    real_usage = chunk.usage
        finally:
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as exc:
                    logger.debug("OpenAI stream.close() fehlgeschlagen: %s", exc)

        full_response = "".join(response_chunks)
        if real_usage is not None:
            self._last_usage = {
                "prompt_tokens": real_usage.prompt_tokens or 0,
                "response_tokens": real_usage.completion_tokens or 0,
                "total_duration_ms": 0,
                "num_ctx": 0,
                "model": self._model,
            }
        else:
            self._last_usage = build_usage_estimated(
                len(system_prompt) + len(user_message),
                len(full_response),
                self._model,
            )
