"""Unchained-Chain: Freies LLM ohne RAG-Kontext.

Direkte Kommunikation mit dem LLM ohne Retrieval-Schritt.
Kein Kontext aus der Wissensbasis, keine Quellen.

Bekommt ein fertig konfiguriertes LLMBackend per Injection.
"""

import logging
from collections.abc import Generator

from kiara.generation._chain_base import _ChainBase
from kiara.generation.backends.base import LLMBackend
from kiara.shared.constants import DEFAULT_LLM_NUM_CTX, DEFAULT_UNCHAINED_TEMPERATURE

logger = logging.getLogger(__name__)


class UnchainedChain(_ChainBase):
    """LLM-Chain ohne Retriever — freies Antworten.

    Bekommt ein fertig konfiguriertes LLMBackend per Injection.
    Kein RAG-Kontext, keine Quellen.
    """

    def __init__(self, config: dict, backend: LLMBackend, system_prompt: str | None = None) -> None:
        """Erstellt eine UnchainedChain mit injiziertem Backend.

        Args:
            config: Globale Config.
            backend: Fertig konfiguriertes LLMBackend (ggf. AnonymizingBackend-Wrapper).
            system_prompt: System-Prompt fuer das LLM.
        """
        self._config = config
        chat_cfg = config.get("chat", {}).get("unchained", {})
        self._temperature = chat_cfg.get("temperature", DEFAULT_UNCHAINED_TEMPERATURE)
        self._backend = backend
        self._system_prompt = system_prompt or ""
        if not self._system_prompt:
            logger.warning("Kein Unchained-System-Prompt uebergeben — leerer Prompt wird verwendet")
        self._last_prompt = ""

    def query_stream(
        self,
        question: str,
        num_ctx: int | None = None,
        history_block: str = "",
        think: bool | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """Streamt eine Antwort ohne RAG-Kontext.

        Args:
            question: Die Benutzerfrage.
            num_ctx: Kontextfenster-Groesse (None = Default).
            history_block: Optionaler Gespraechsverlauf.
            think: Thinking-Toggle.
            temperature: Temperature-Override.
        """
        self._last_prompt = ""

        from kiara.shared.date_utils import get_german_date_string
        from kiara.shared.prompt_defense import (
            TAG_HISTORY,
            build_hardened_system_prompt,
            sanitize_with_report,
            wrap_external_data,
        )
        from kiara.shared.prompt_defense_audit import log_defense_event

        date_prefix = get_german_date_string()

        if history_block:
            history_report = sanitize_with_report(history_block)
            if history_report.severity != "none":
                log_defense_event("chat_history", history_report)
            history_block = wrap_external_data(history_report.clean_text, TAG_HISTORY)

        history_part = f"\n\n{history_block}" if history_block else ""
        user_message = f"{date_prefix}{history_part}\n\n{question}"

        hardened_prompt = build_hardened_system_prompt(self._system_prompt)
        self._last_prompt = (
            f"{hardened_prompt}\n\n{user_message}" if hardened_prompt else user_message
        )

        effective_ctx = num_ctx or DEFAULT_LLM_NUM_CTX
        effective_temp = temperature if temperature is not None else self._temperature
        kwargs = {"think": think, "temperature": effective_temp}
        if self._backend.provider == "ollama":
            kwargs["num_ctx"] = effective_ctx

        yield from self._backend.generate_stream(
            hardened_prompt,
            user_message,
            **kwargs,
        )
