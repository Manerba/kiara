"""OpenAI-Backend (GPT) ueber die offizielle openai-SDK.

Unterstuetzt Chat Completions fuer OpenAI-kompatible Endpoints und die
Responses API fuer Responses-only-Modelle wie GPT-5.5 Pro.
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
    """Chat-Completions-Modelle, die kein ``max_tokens`` akzeptieren."""
    model_name = (model or "").split("/")[-1].lower()
    return (
        model_name.startswith("gpt-5")
        or re.match(r"^o(?:1|3|4)(?:[-_].*)?$", model_name) is not None
    )


def _uses_responses_api(model: str) -> bool:
    """True fuer OpenAI-Pro-Modelle, die nur die Responses API anbieten.

    GPT-5 Pro und seine versionierten Nachfolger werden von OpenAI nicht ueber
    ``/v1/chat/completions`` angeboten. Andere Provider mit OpenAI-kompatibler
    API (DeepSeek, Ollama-Gateways usw.) bleiben auf Chat Completions.
    """
    model_name = (model or "").split("/")[-1].lower()
    return re.match(r"^gpt-5(?:\.\d+)?-pro(?:-|$)", model_name) is not None


def _uses_default_temperature(model: str) -> bool:
    """Chat-Completions-Modelle, die keine Temperature-Overrides akzeptieren."""
    model_name = (model or "").split("/")[-1].lower()
    return (
        model_name.startswith("gpt-5")
        or re.match(r"^o(?:1|3|4)(?:[-_].*)?$", model_name) is not None
    )


def _completion_token_kwargs(model: str, max_tokens: int) -> dict:
    if _uses_max_completion_tokens(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def _temperature_kwargs(model: str, temperature: float) -> dict:
    if _uses_default_temperature(model):
        return {}
    return {"temperature": temperature}


def _reasoning_effort_kwargs(model: str, reasoning_effort: str | None) -> dict:
    if not reasoning_effort or not _uses_max_completion_tokens(model):
        return {}
    return {"reasoning_effort": reasoning_effort}


def _responses_reasoning_kwargs(model: str, reasoning_effort: str | None) -> dict:
    """Uebersetzt den KIara-Override in das Responses-API-Datenmodell."""
    if not reasoning_effort or not _uses_max_completion_tokens(model):
        return {}
    return {"reasoning": {"effort": reasoning_effort}}


def _response_error_message(response) -> str | None:
    """Liefert fuer nicht erfolgreiche Responses einen fail-closed Fehlertext."""
    status = getattr(response, "status", None)
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) or "unbekannter Grund"
        return f"OpenAI-Response unvollstaendig: {reason}"
    if status in {"failed", "cancelled"}:
        error = getattr(response, "error", None)
        message = getattr(error, "message", None) or status
        return f"OpenAI-Response fehlgeschlagen: {message}"
    return None


def _responses_request_kwargs(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None,
) -> dict:
    """Baut einen einfachen Text-Request fuer ``/v1/responses``."""
    params = {
        "model": model,
        "input": user_message,
        "max_output_tokens": max_tokens,
        **_temperature_kwargs(model, temperature),
        **_responses_reasoning_kwargs(model, reasoning_effort),
    }
    if system_prompt:
        params["instructions"] = system_prompt
    return params


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
        """Nicht-streamende Generierung ueber die passende OpenAI-API.

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
        reasoning_effort = kwargs.get("reasoning_effort")

        if _uses_responses_api(self._model):
            resp = self._client.responses.create(
                **_responses_request_kwargs(
                    self._model,
                    system_prompt,
                    user_message,
                    max_tokens,
                    temperature,
                    reasoning_effort,
                )
            )
            error_message = _response_error_message(resp)
            if error_message:
                raise RuntimeError(error_message)
            text = resp.output_text or ""
        else:
            # Leere System-Messages weglassen (wie Ollama-Backend)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})

            resp = self._client.chat.completions.create(
                model=self._model,
                **_completion_token_kwargs(self._model, max_tokens),
                **_temperature_kwargs(self._model, temperature),
                **_reasoning_effort_kwargs(self._model, reasoning_effort),
                messages=messages,
            )

            text = (resp.choices[0].message.content or "") if resp.choices else ""

        # Usage-Tracking: reale Werte bevorzugen, sonst Schaetzung
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._last_usage = {
                "prompt_tokens": (
                    getattr(usage, "input_tokens", None)
                    if _uses_responses_api(self._model)
                    else getattr(usage, "prompt_tokens", None)
                )
                or 0,
                "response_tokens": (
                    getattr(usage, "output_tokens", None)
                    if _uses_responses_api(self._model)
                    else getattr(usage, "completion_tokens", None)
                )
                or 0,
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
        """Streaming-Generierung ueber die passende OpenAI-API.

        kwargs:
            temperature (float): Ueberschreibt Default.
            max_tokens (int): Ueberschreibt Default.

        Yields:
            Text-Chunks der Antwort.
        """
        self._last_usage = {}
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        reasoning_effort = kwargs.get("reasoning_effort")

        response_chunks = []
        real_usage = None
        uses_responses = _uses_responses_api(self._model)
        if uses_responses:
            stream = self._client.responses.create(
                **_responses_request_kwargs(
                    self._model,
                    system_prompt,
                    user_message,
                    max_tokens,
                    temperature,
                    reasoning_effort,
                ),
                stream=True,
            )
        else:
            # Leere System-Messages weglassen (wie Ollama-Backend)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_message})

            stream = self._client.chat.completions.create(
                model=self._model,
                **_completion_token_kwargs(self._model, max_tokens),
                **_temperature_kwargs(self._model, temperature),
                **_reasoning_effort_kwargs(self._model, reasoning_effort),
                stream=True,
                stream_options={"include_usage": True},
                messages=messages,
            )
        try:
            if uses_responses:
                for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        text = event.delta
                        response_chunks.append(text)
                        yield text
                    elif event_type == "response.completed":
                        real_usage = getattr(event.response, "usage", None)
                    elif event_type in {"response.incomplete", "response.failed"}:
                        error_message = _response_error_message(event.response)
                        raise RuntimeError(error_message or "OpenAI-Response fehlgeschlagen")
                    elif event_type == "error":
                        raise RuntimeError(f"OpenAI-Streaming-Fehler: {event.message}")
            else:
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
                "prompt_tokens": (
                    getattr(real_usage, "input_tokens", None)
                    if uses_responses
                    else getattr(real_usage, "prompt_tokens", None)
                )
                or 0,
                "response_tokens": (
                    getattr(real_usage, "output_tokens", None)
                    if uses_responses
                    else getattr(real_usage, "completion_tokens", None)
                )
                or 0,
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
