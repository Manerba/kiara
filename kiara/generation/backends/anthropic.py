"""Anthropic-Backend (Claude) ueber die offizielle anthropic-SDK.

Nutzt client.messages.create() (non-streaming) und
client.messages.stream() (streaming) mit nativer System-Prompt-Trennung.
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
    PASSWORD_MASK,
)

logger = logging.getLogger(__name__)

# Maskiert API-Keys/Bearer-Tokens in Exception-Messages, falls die SDK/httpx
# Auth-Header oder Keys in einer Fehler-Repraesentation einbettet.
_SECRET_RE = re.compile(
    r"sk-ant-[A-Za-z0-9_\-]{20,}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|Bearer\s+[A-Za-z0-9._\-]{20,}",
    re.IGNORECASE,
)


def _scrub_secrets(text: str) -> str:
    return _SECRET_RE.sub(PASSWORD_MASK, text)


def _import_anthropic():
    try:
        return import_module("anthropic")
    except ImportError as exc:
        raise RuntimeError("Optionale Dependency 'anthropic' ist nicht installiert") from exc


def _supports_adaptive_thinking(model: str) -> bool:
    """Adaptive Thinking ist nur fuer die explizit aktivierten neuen Modelle."""
    normalized = (model or "").lower()
    return "4-6" in normalized or "4.6" in normalized


def _thinking_params(model: str, max_tokens: int) -> dict:
    if _supports_adaptive_thinking(model):
        return {
            "thinking": {"type": "adaptive"},
            "max_tokens": max(max_tokens, 16000),
        }

    budget_tokens = max(1024, min(8192, max_tokens // 2))
    return {
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        },
        "max_tokens": max(max_tokens, budget_tokens + 1024),
    }


class AnthropicBackend(LLMBackend):
    """Anthropic-Backend (Claude) ueber die offizielle SDK.

    Nutzt client.messages.create() (non-streaming) und
    client.messages.stream() (streaming) mit nativer System-Prompt-Trennung.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        base_url: str = "",
    ) -> None:
        """Erstellt einen Anthropic-Backend-Client.

        Args:
            api_key: Entschluesselter API-Key (Klartext, NICHT verschluesselt).
                     Die Entschluesselung passiert in der Factory, NICHT hier.
            model: Modell-ID (z.B. 'claude-sonnet-4-6').
            max_tokens: Maximale Antwort-Token (Default: 4096 aus constants.py).
            temperature: Default-Temperatur (Default: 0.3).
            base_url: Optionale Base-URL (leer = offizielle Anthropic-API).
        """
        anthropic = _import_anthropic()
        import httpx  # dependency of anthropic SDK

        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url or None,
            timeout=httpx.Timeout(CLOUD_READ_TIMEOUT_SECS, connect=CLOUD_CONNECT_TIMEOUT_SECS),
            max_retries=CLOUD_MAX_RETRIES,
        )
        self._anthropic = anthropic
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._last_usage: dict = {}

    @property
    def provider(self) -> str:
        """Gibt 'anthropic' zurueck."""
        return "anthropic"

    @property
    def model(self) -> str:
        """Gibt die Modell-ID zurueck."""
        return self._model

    @property
    def last_usage(self) -> dict:
        """Token-Usage-Daten des letzten Aufrufs (real oder Fallback-Schaetzung)."""
        return self._last_usage

    def generate(self, system_prompt: str, user_message: str, **kwargs) -> str:
        """Nicht-streamende Generierung ueber Anthropic Messages API.

        kwargs:
            temperature (float): Ueberschreibt Default.
            max_tokens (int): Ueberschreibt Default.
            think (bool | None): Extended Thinking fuer Claude.

        Returns:
            Generierter Text.

        Raises:
            anthropic.APIError: Bei API-Fehlern.
            anthropic.AuthenticationError: Bei ungueltigem API-Key.
        """
        self._last_usage = {}
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        think = kwargs.get("think")

        params = dict(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        if think:
            params.update(_thinking_params(self._model, max_tokens))
        else:
            params["temperature"] = temperature

        # Bei Thinking-Fehler genau einmal ohne Thinking retryen.
        # Strikt: nur BadRequestError (HTTP 400) gilt — Auth-, RateLimit- oder
        # Server-Fehler werden nicht als Thinking-Fehler missinterpretiert.
        try:
            resp = self._client.messages.create(**params)
        except Exception as exc:
            exc_msg = str(exc)
            is_thinking_error = (
                think
                and isinstance(exc, self._anthropic.BadRequestError)
                and ("thinking" in exc_msg.lower() or "budget" in exc_msg.lower())
            )
            if is_thinking_error:
                logger.warning(
                    "Anthropic Thinking nicht verfuegbar, Fallback ohne Thinking: %s",
                    _scrub_secrets(exc_msg),
                )
                params.pop("thinking", None)
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature
                resp = self._client.messages.create(**params)
            else:
                raise

        # Antwort extrahieren: Thinking-Bloecke und Text-Bloecke trennen
        thinking_parts = []
        text_parts = []
        for block in resp.content or []:
            if getattr(block, "type", None) == "thinking":
                thinking_parts.append(getattr(block, "thinking", ""))
            elif getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))

        text = "\n".join(text_parts)
        if thinking_parts:
            text = f"<think>{''.join(thinking_parts)}</think>{text}"

        # Usage-Tracking: reale Werte bevorzugen, sonst Schaetzung
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._last_usage = {
                "prompt_tokens": usage.input_tokens or 0,
                "response_tokens": usage.output_tokens or 0,
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
        """Streaming-Generierung ueber Anthropic Messages API.

        kwargs:
            temperature (float): Ueberschreibt Default.
            max_tokens (int): Ueberschreibt Default.
            think (bool | None): Extended Thinking fuer Claude.

        Yields:
            Text-Chunks der Antwort.
        """
        self._last_usage = {}
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        think = kwargs.get("think")

        params = dict(
            model=self._model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        if think:
            params.update(_thinking_params(self._model, max_tokens))
        else:
            params["temperature"] = temperature

        response_chunks = []
        yielded_any = False
        in_thinking = False
        final_usage = None
        try:
            with self._client.messages.stream(**params) as stream:
                if not think:
                    for text in stream.text_stream:
                        response_chunks.append(text)
                        yielded_any = True
                        yield text
                else:
                    in_thinking = False
                    for event in stream:
                        if event.type == "content_block_start":
                            if event.content_block.type == "thinking":
                                in_thinking = True
                                response_chunks.append("<think>")
                                yielded_any = True
                                yield "<think>"
                            elif event.content_block.type == "text" and in_thinking:
                                in_thinking = False
                                response_chunks.append("</think>")
                                yielded_any = True
                                yield "</think>"
                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "thinking"):
                                response_chunks.append(event.delta.thinking)
                                yielded_any = True
                                yield event.delta.thinking
                            elif hasattr(event.delta, "text"):
                                response_chunks.append(event.delta.text)
                                yielded_any = True
                                yield event.delta.text
                    if in_thinking:
                        response_chunks.append("</think>")
                        yielded_any = True
                        yield "</think>"
                try:
                    final_msg = stream.get_final_message()
                    usage = getattr(final_msg, "usage", None)
                    if usage is not None:
                        final_usage = usage
                except Exception as exc:
                    logger.warning(
                        "Anthropic stream.get_final_message() fehlgeschlagen, Token-Usage "
                        "wird via Schaetzung gefuellt: %s",
                        _scrub_secrets(str(exc)),
                    )
        except Exception as exc:
            if in_thinking:
                response_chunks.append("</think>")
                yield "</think>\n"
            exc_msg = str(exc)
            is_thinking_error = (
                think
                and not yielded_any
                and isinstance(exc, self._anthropic.BadRequestError)
                and ("thinking" in exc_msg.lower() or "budget" in exc_msg.lower())
            )
            if is_thinking_error:
                logger.warning(
                    "Anthropic Thinking nicht verfuegbar, Fallback ohne Thinking: %s",
                    _scrub_secrets(exc_msg),
                )
                yield from self.generate_stream(
                    system_prompt,
                    user_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    think=False,
                )
                return
            self._last_usage = build_usage_estimated(
                len(system_prompt) + len(user_message),
                len("".join(response_chunks)),
                self._model,
            )
            raise

        full_response = "".join(response_chunks)
        if final_usage is not None:
            self._last_usage = {
                "prompt_tokens": final_usage.input_tokens or 0,
                "response_tokens": final_usage.output_tokens or 0,
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
