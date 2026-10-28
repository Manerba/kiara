"""Rewriter-Chain: LLM-basiertes Query-Rewriting + Standard-RAG.

Kombiniert einen schnellen, nicht-streamenden Rewrite-Schritt (immer lokal
via OllamaBackend) mit der bestehenden RAGChain fuer Retrieval + Antwort.
Ergibt genau zwei LLM-Aufrufe pro Anfrage — der "Medium-Effort"-RAG-Modus.
"""

import logging
import re
from collections.abc import Generator

from kiara.generation.backends.base import LLMBackend
from kiara.shared.constants import DEFAULT_LLM_TEMPERATURE, DEFAULT_REWRITER_NUM_CTX
from kiara.shared.processing_warnings import report_processing_warning

logger = logging.getLogger(__name__)

DEFAULT_REWRITE_PROMPT = (
    "Du bist ein Suchbegriff-Optimierer. Deine Aufgabe ist es, aus einer "
    "natuerlichsprachigen Frage einen optimalen Suchbegriff fuer eine "
    "Hybrid-Suche (Volltextsuche + Vektorsuche) zu formulieren.\n\n"
    "Regeln:\n"
    "- Entferne Fuellwoerter, Hoeflichkeitsfloskeln und Satzzeichen\n"
    "- Behalte Eigennamen, Fachbegriffe und Zahlen\n"
    "- Nutze den Gespraechsverlauf um Pronomen und Rueckbezuege aufzuloesen\n"
    "- Antworte mit GENAU EINER Zeile: nur der optimierte Suchbegriff\n"
    "- Kein Markdown, keine Erklaerung, keine Anführungszeichen\n"
    "- Maximal 10 Woerter\n"
)


def _validate_rewrite(raw: str, original: str) -> str:
    """Validiert und bereinigt das Rewrite-Ergebnis.

    - Leerer Text → Fallback auf Originalfrage
    - Mehrzeilig → nur erste nicht-leere Zeile
    - Markdown-Formatierung entfernen
    - Laenger als 500 Zeichen → kuerzen
    """
    if not raw or not raw.strip():
        return original

    # Nur erste nicht-leere Zeile
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    result = lines[0] if lines else original

    # Markdown-/Tag-Formatierung entfernen
    result = re.sub(r"[`*#<>\[\](){}]", "", result).strip()
    # Anfuehrungszeichen entfernen
    result = result.strip("\"'")

    if not result:
        return original

    # Laengenbegrenzung an Wortgrenze
    if len(result) > 500:
        cut = result[:500]
        last_space = cut.rfind(" ")
        result = cut[:last_space] if last_space > 0 else cut

    return result


class RewriterChain:
    """Query-Rewriter + RAGChain: optimierter Suchbegriff via LLM.

    Schritt 1: OllamaBackend formuliert einen optimierten Suchbegriff (lokal, nicht-streamend).
    Schritt 2: RAGChain sucht mit dem Suchbegriff, antwortet auf die Originalfrage.
    """

    def __init__(
        self,
        config: dict,
        rag_backend: LLMBackend,
        embedding_model: str | None = None,
        prompt_template_override: str | None = None,
        system_prompt: str = "",
    ) -> None:
        """Erstellt eine RewriterChain.

        Args:
            config: Globale Config.
            rag_backend: Backend fuer die RAG-Antwort (ggf. AnonymizingBackend).
            embedding_model: Embedding-Modell-Override.
            prompt_template_override: Prompt-Template-Override.
            system_prompt: Optionaler System-Prompt fuer die Antwort-Chain.
        """
        from kiara.backend_selector import BackendUnavailableError, get_all_active
        from kiara.generation.backends.ollama import OllamaBackend
        from kiara.generation.chain import RAGChain

        self._config = config
        rewriter_cfg = config.get("chat", {}).get("rewriter", {})

        # RAGChain mit injiziertem Backend fuer die Antwort
        self._rag_chain = RAGChain(
            config,
            backend=rag_backend,
            embedding_model=embedding_model,
            prompt_template_override=prompt_template_override,
            system_prompt=system_prompt,
        )

        # Separates Ollama-Backend fuer den Rewrite-Schritt (MUSS lokal sein)
        ollama_backends = [
            b for b in (get_all_active("llm") or []) if b and b.get("provider") == "ollama"
        ]
        if not ollama_backends:
            raise BackendUnavailableError("Kein lokales Ollama-Backend fuer Rewrite verfuegbar")
        ob = ollama_backends[0]
        self._rewrite_backend = OllamaBackend(
            url=ob["url"],
            model=ob["model"],
        )

        # Rewrite-Konfiguration
        self._rewrite_num_ctx = rewriter_cfg.get("num_ctx", DEFAULT_REWRITER_NUM_CTX)
        self._rewrite_temperature = rewriter_cfg.get("temperature", DEFAULT_LLM_TEMPERATURE)
        self._rewrite_prompt = rewriter_cfg.get("rewrite_prompt", "") or DEFAULT_REWRITE_PROMPT

        # State
        self._rewritten_query: str = ""
        self._original_question: str = ""
        self._rewrite_usage: dict = {}

    @property
    def rewritten_query(self) -> str:
        """Der vom LLM formulierte optimierte Suchbegriff."""
        return self._rewritten_query

    @property
    def last_usage(self) -> dict:
        """Aggregierte Token-Usage beider LLM-Aufrufe."""
        rag_usage = self._rag_chain.last_usage
        if not self._rewrite_usage:
            return rag_usage
        if not rag_usage:
            return self._rewrite_usage
        return {
            "prompt_tokens": self._rewrite_usage.get("prompt_tokens", 0)
            + rag_usage.get("prompt_tokens", 0),
            "response_tokens": self._rewrite_usage.get("response_tokens", 0)
            + rag_usage.get("response_tokens", 0),
            "total_duration_ms": self._rewrite_usage.get("total_duration_ms", 0)
            + rag_usage.get("total_duration_ms", 0),
            "model": rag_usage.get("model", self._rewrite_usage.get("model", "")),
            "num_ctx": rag_usage.get("num_ctx", self._rewrite_usage.get("num_ctx", 0)),
        }

    @property
    def last_prompt(self) -> str:
        """Audit-Trail-Prompt mit Rewrite-Info."""
        rag_prompt = self._rag_chain.last_prompt
        return f"[REWRITE: {self._rewritten_query}]\n\n[ORIGINAL: {self._original_question}]\n\n[PROMPT: {rag_prompt}]"

    @property
    def last_trace_id(self) -> str | None:
        """Delegiert die persistierte rag_query-trace_id an api_chat.py."""
        return self._rag_chain.last_trace_id

    @property
    def route_used(self) -> str:
        """Route der Antwort-Generierung (delegiert an RAGChain)."""
        return self._rag_chain.route_used

    @property
    def context_warning(self) -> dict | None:
        """Kontext-Overflow-Warnung (delegiert an RAGChain)."""
        return self._rag_chain.context_warning

    @property
    def last_anon_question(self) -> str:
        """Delegiert an RAGChain."""
        return self._rag_chain.last_anon_question

    @property
    def last_anon_answer(self) -> str:
        """Delegiert an RAGChain."""
        return self._rag_chain.last_anon_answer

    @property
    def last_entities_anonymized(self) -> int:
        """Delegiert an RAGChain."""
        return self._rag_chain.last_entities_anonymized

    @property
    def last_expansions(self) -> list[str]:
        """Query-Expansion-Varianten (delegiert an RAGChain)."""
        return self._rag_chain.last_expansions

    @property
    def backend(self) -> LLMBackend:
        """Public accessor fuer Audit-Zugriff auf das Antwort-Backend."""
        return self._rag_chain.backend

    def query_stream(
        self,
        question: str,
        k: int = 5,
        num_ctx: int | None = None,
        filter_dir: str | None = None,
        filter_type: str | None = None,
        filter_datasource_ids: list[int] | None = None,
        history_block: str = "",
        think: bool | None = None,
        context_override: list[dict] | None = None,
    ) -> tuple[Generator[str, None, None], list[dict]]:
        """RAG-Query mit vorgeschaltetem LLM-Rewrite.

        Bei context_override (Search-Tab) wird der Rewrite uebersprungen.
        """
        self._original_question = question

        # Context-Override: Rewrite ueberspringen
        if context_override is not None:
            self._rewritten_query = question
            self._rewrite_usage = {}
            return self._rag_chain.query_stream(
                question,
                k=k,
                num_ctx=num_ctx,
                filter_dir=filter_dir,
                filter_type=filter_type,
                filter_datasource_ids=filter_datasource_ids,
                history_block=history_block,
                think=think,
                context_override=context_override,
            )

        # Prompt-Injection-Defense fuer history_block (analog chain.py:302)
        if history_block:
            from kiara.shared.prompt_defense import (
                TAG_HISTORY,
                sanitize_with_report,
                wrap_external_data,
            )
            from kiara.shared.prompt_defense_audit import log_defense_event

            history_report = sanitize_with_report(history_block)
            if history_report.severity != "none":
                log_defense_event("chat_history", history_report)
            history_block = wrap_external_data(history_report.clean_text, TAG_HISTORY)

        # Schritt 1: LLM-basierter Rewrite
        rewritten_query = question  # Fallback
        try:
            history_part = f"\nGespraechsverlauf:\n{history_block}\n" if history_block else ""
            rewrite_input = (
                f"{self._rewrite_prompt}\n"
                f"{history_part}\n"
                f"Frage: {question}\n\n"
                f"Optimierter Suchbegriff:"
            )
            raw_rewrite = self._rewrite_backend.generate(
                system_prompt="",
                user_message=rewrite_input,
                temperature=self._rewrite_temperature,
                num_ctx=self._rewrite_num_ctx,
            )
            self._rewrite_usage = self._rewrite_backend.last_usage
            rewritten_query = _validate_rewrite(raw_rewrite, question)
        except Exception as exc:
            logger.warning("Rewrite fehlgeschlagen, Fallback auf Originalfrage: %s", exc)
            report_processing_warning("rewrite")
            self._rewrite_usage = {
                "model": self._rewrite_backend.model,
                "error": str(exc),
            }

        self._rewritten_query = rewritten_query

        # Schritt 2: Retrieval mit optimiertem Suchbegriff
        results = self._rag_chain._retrieve_and_rerank(
            rewritten_query,
            k,
            filter_dir,
            filter_type,
            filter_datasource_ids,
        )

        # Schritt 3: Antwort mit Originalfrage + Retrieval-Ergebnissen
        return self._rag_chain.query_stream(
            question,
            num_ctx=num_ctx,
            history_block=history_block,
            think=think,
            context_override=results,
            filter_datasource_ids=filter_datasource_ids,
        )
