"""RAG chain: retrieval + prompt building + LLM generation.

Die Chain orchestriert Retrieval und Generierung — das LLM-Backend
wird per Konstruktor injiziert (Dependency Injection). Die Chain
entscheidet NICHT ueber Routing oder Anonymisierung; beides wird
vom Aufrufer (api_chat.py) ueber das Backend gesteuert.
"""

import logging
from collections.abc import Callable, Generator

from kiara.generation._chain_base import _ChainBase
from kiara.generation.backends.base import LLMBackend
from kiara.retrieval.query_expansion import QueryExpander
from kiara.retrieval.reranker import CrossEncoderReranker
from kiara.retrieval.retriever import HybridRetriever
from kiara.shared.constants import (
    CHARS_PER_TOKEN,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_MAX_CONTEXT_CHUNKS,
    EXCERPT_LENGTH,
)
from kiara.shared.tracing import get_trace_id_for_span, get_tracer

logger = logging.getLogger(__name__)

_tracer = get_tracer(__name__, pipeline="retrieval")

_DEFAULT_RAG_TEMPLATE = (
    "Nutze den folgenden Kontext um die Frage zu beantworten.\n\n"
    "Kontext:\n{context}\n\n"
    "Frage: {question}"
)


def expand_sibling_chunks(
    selected: list[dict],
    candidates: list[dict],
    max_expand: int = 5,
    max_total: int | None = None,
    load_source_chunks: Callable[[str, list[int] | None], list[dict]] | None = None,
    fallback_datasource_ids: list[int] | None = None,
) -> list[dict]:
    """Ergaenzt Geschwister-Chunks aus derselben Quelldatei.

    Bei kleinen Dokumenten (chunk_total <= max_expand) werden alle
    Geschwister-Chunks ergaenzt. Bei grossen Dokumenten nur die
    direkten Nachbarn (chunk_index +/- 1).

    So sieht das LLM den vollen Kontext auch wenn der Reranker
    einen weniger relevanten Chunk desselben Dokuments bevorzugt.
    """

    def _normalize_ds_ids(raw) -> list[int]:
        if raw is None:
            return []
        values = raw if isinstance(raw, (list, tuple, set)) else [raw]
        normalized = []
        for value in values:
            try:
                normalized.append(int(value))
            except (TypeError, ValueError):
                continue
        return normalized

    def _ds_key(meta: dict) -> tuple[int, ...]:
        ds_ids = _normalize_ds_ids(meta.get("ds_ids"))
        if not ds_ids and fallback_datasource_ids:
            ds_ids = _normalize_ds_ids(fallback_datasource_ids)
        return tuple(sorted(set(ds_ids)))

    def _add_to_index(chunk: dict) -> None:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "")
        ci = meta.get("chunk_index", -1)
        try:
            ci = int(ci)
        except (TypeError, ValueError):
            ci = -1
        if source and ci >= 0:
            candidate_index.setdefault((source, _ds_key(meta)), {})[ci] = chunk

    # Index: (source, ds_key) -> {chunk_index: candidate}
    # ds_key diskriminiert gleichnamige Dateien in verschiedenen Datasources.
    candidate_index: dict[tuple[str, tuple], dict[int, dict]] = {}
    for c in candidates:
        _add_to_index(c)
    for c in selected:
        _add_to_index(c)

    # Ausgewaehlte Chunks identifizieren
    selected_keys: set[tuple[str, int, tuple]] = set()
    for r in selected:
        meta = r.get("metadata", {})
        source = meta.get("source", "")
        ci = meta.get("chunk_index", -1)
        try:
            ci = int(ci)
        except (TypeError, ValueError):
            ci = -1
        ds_key = _ds_key(meta)
        selected_keys.add((source, ci, ds_key))

    loaded_sources: set[tuple[str, tuple]] = set()

    def _desired_indices(ci: int, ct: int) -> list[int]:
        if ct <= 0:
            return []
        if ct <= max_expand:
            return list(range(ct))
        return [idx for idx in (ci - 1, ci + 1) if 0 <= idx < ct]

    def _load_missing_source(source: str, ds_key: tuple[int, ...]) -> None:
        if load_source_chunks is None or (source, ds_key) in loaded_sources:
            return
        loaded_sources.add((source, ds_key))
        ds_ids = list(ds_key) if ds_key else (fallback_datasource_ids or None)
        try:
            loaded = load_source_chunks(source, ds_ids)
        except Exception as exc:
            logger.warning(
                "Sibling-Expansion: Quelle %r konnte nicht nachgeladen werden: %s",
                source,
                exc,
                exc_info=True,
            )
            return
        for chunk in loaded or []:
            meta = chunk.get("metadata", {})
            if meta.get("source") != source:
                continue
            loaded_ds_key = _ds_key(meta)
            if ds_key and loaded_ds_key != ds_key:
                continue
            _add_to_index(chunk)

    # Geschwister finden
    to_add: dict[tuple[str, int, tuple], dict] = {}
    for r in selected:
        meta = r.get("metadata", {})
        source = meta.get("source", "")
        ci = meta.get("chunk_index", -1)
        ct = meta.get("chunk_total", 1)
        try:
            ci = int(ci)
            ct = int(ct)
        except (TypeError, ValueError):
            continue
        ds_key = _ds_key(meta)

        if not source or ci < 0:
            continue

        wanted = _desired_indices(ci, ct)
        source_chunks = candidate_index.get((source, ds_key), {})
        if any(idx not in source_chunks for idx in wanted):
            _load_missing_source(source, ds_key)
            source_chunks = candidate_index.get((source, ds_key), {})

        for idx in wanted:
            key = (source, idx, ds_key)
            if key not in selected_keys and key not in to_add and idx in source_chunks:
                to_add[key] = source_chunks[idx]

    if not to_add:
        return selected

    # Primaerordnung: Quellen nach Relevanz (Reihenfolge in Rerank-Ergebnis).
    # Innerhalb einer Quelle nach chunk_index sortiert.
    source_rank: dict[tuple[str, tuple], int] = {}
    for i, r in enumerate(selected):
        meta = r.get("metadata", {})
        src_key = (meta.get("source", ""), _ds_key(meta))
        if src_key not in source_rank:
            source_rank[src_key] = i

    def _context_sort_key(r: dict) -> tuple[int, int]:
        return (
            source_rank.get(
                (r.get("metadata", {}).get("source", ""), _ds_key(r.get("metadata", {}))),
                len(selected),
            ),
            r.get("metadata", {}).get("chunk_index", 0),
        )

    siblings = sorted(to_add.values(), key=_context_sort_key)

    expanded = selected + siblings
    expanded.sort(key=_context_sort_key)

    if max_total is not None and len(expanded) > max_total:
        sibling_budget = max(0, max_total - len(selected))
        # Im capped RAG-Kontext duerfen Geschwister nie staerker gerankte
        # Rerank-Treffer aus dem Kontext verdraengen.
        expanded = list(selected) + siblings[:sibling_budget]
        expanded.sort(key=_context_sort_key)

    logger.debug(
        "Sibling-Expansion: %d -> %d Chunks (+%d Geschwister)",
        len(selected),
        len(expanded),
        len(to_add),
    )
    return expanded


def format_context(results: list[dict]) -> str:
    """Format retrieval results into a context string (mit Prompt-Defense)."""
    from kiara.shared.prompt_defense import (
        TAG_CONTEXT,
        sanitize_text,
        sanitize_with_report,
        wrap_external_data,
    )
    from kiara.shared.prompt_defense_audit import log_defense_event

    parts = []
    for i, r in enumerate(results, 1):
        source = sanitize_text(r.get("metadata", {}).get("source") or "unbekannt")
        report = sanitize_with_report(r.get("text") or "")
        if report.severity != "none":
            log_defense_event("retrieval_context", report, document_source=source)
        parts.append(f"[Quelle {i}: {source}]\n{report.clean_text}")
    raw_context = "\n\n---\n\n".join(parts)
    return wrap_external_data(raw_context, TAG_CONTEXT)


def build_sources(results: list[dict]) -> list[dict]:
    """Baut die Quell-Liste aus Retrieval-Ergebnissen auf.

    Wird von RAGChain und SearchDocumentsTool gemeinsam genutzt.
    """
    sources = []
    for r in results:
        # Erstes vorhandenes/nicht-None Feld (NICHT Truthiness — 0 ist ein gueltiger Score)
        score = r.get("rrf_score")
        if score is None:
            score = r.get("rerank_score")
        if score is None:
            score = r.get("score", 0)
        text = r.get("text") or ""
        sources.append(
            {
                "source": r.get("metadata", {}).get("source", ""),
                "score": score,
                "excerpt": text[:EXCERPT_LENGTH] + "..." if len(text) > EXCERPT_LENGTH else text,
            }
        )
    return sources


class RAGChain(_ChainBase):
    """Orchestriert Retrieval und Generierung fuer RAG-Queries.

    Bekommt ein fertig konfiguriertes LLMBackend per Injection.
    Die Chain entscheidet NICHT ueber Routing oder Anonymisierung —
    beides wird vom Aufrufer (api_chat.py) ueber das Backend gesteuert.
    """

    def __init__(
        self,
        config: dict,
        backend: LLMBackend,
        embedding_model: str | None = None,
        prompt_template_override: str | None = None,
        system_prompt: str = "",
    ):
        self._config = config
        self._backend = backend
        self._retriever = HybridRetriever(config, embedding_model=embedding_model)
        self._reranker = CrossEncoderReranker(config)
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template_override or _DEFAULT_RAG_TEMPLATE
        if not prompt_template_override:
            logger.debug("Kein RAG-Prompt-Template uebergeben — Standard-Template wird verwendet")
        self._context_warning: dict | None = None
        self._last_prompt = ""
        self._expander = QueryExpander(config, backend=self._backend)
        self._last_expansions: list[str] = []
        self._last_trace_id: str | None = None

    @property
    def last_trace_id(self) -> str | None:
        """Trace-ID des letzten `rag_query`-Spans (fuer query_log-Korrelation)."""
        return self._last_trace_id

    @property
    def last_expansions(self) -> list[str]:
        """Query-Expansion-Varianten des letzten Aufrufs (leer wenn deaktiviert)."""
        return self._last_expansions

    @property
    def retriever(self) -> HybridRetriever:
        """Public accessor for the retriever instance."""
        return self._retriever

    @property
    def context_warning(self) -> dict | None:
        """Warnung wenn Quellen-Kontext das Kontextfenster uebersteigt."""
        return self._context_warning

    def _check_context_overflow(self, prompt: str, num_ctx: int) -> None:
        """Prueft ob der Prompt das Kontextfenster uebersteigt und setzt ggf. Warnung."""
        estimated_tokens = (len(prompt) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
        if estimated_tokens > num_ctx:
            self._context_warning = {
                "estimated_tokens": estimated_tokens,
                "num_ctx": num_ctx,
                "overflow_pct": round((estimated_tokens / num_ctx - 1) * 100),
            }
            logger.warning(
                "Kontext-Ueberlauf: ~%d Tokens geschaetzt, num_ctx=%d",
                estimated_tokens,
                num_ctx,
            )
        else:
            self._context_warning = None

    @staticmethod
    def _build_sources(results: list[dict]) -> list[dict]:
        """Baut die Quell-Liste aus Retrieval-Ergebnissen auf."""
        return build_sources(results)

    def _retrieve_and_rerank(
        self,
        question: str,
        k: int,
        filter_dir: str | None,
        filter_type: str | None,
        filter_datasource_ids: list[int] | None,
    ) -> list[dict]:
        """Optionale Query-Expansion + Retrieval + Reranking + Sibling-Expansion.

        Bei aktivierter Query-Expansion (Config: retrieval.query_expansion)
        werden per LLM alternative Suchbegriffe generiert und fuer jede
        Variante ein separater Retrieve-Aufruf durchgefuehrt. Die Ergebnisse
        werden dedupliziert und per Cross-Encoder auf die besten k Chunks
        reduziert. Bei deaktivierter Expansion ist das Verhalten identisch
        zum bisherigen Single-Query-Ablauf.
        """
        # 1. Query-Expansion (nur mit Reranker sinnvoll — ohne Reranker
        #    werden Expansion-Treffer bei Dedup/Truncation verworfen)
        with _tracer.start_as_current_span("query_expansion") as exp_span:
            variants = self._expander.expand(question) if self._reranker.enabled else [question]
            self._last_expansions = variants
            exp_span.set_attribute("variant_count", len(variants))
            exp_span.set_attribute("enabled", len(variants) > 1)

        # 2. Reranking-Multiplier
        fetch_k = k
        if self._reranker.enabled:
            multiplier = self._config.get("retrieval", {}).get("rerank_fetch_multiplier", 3)
            fetch_k = k * multiplier

        # 3. Multi-Query-Retrieval mit Fehlertoleranz fuer Expansion-Varianten
        all_results: list[dict] = []
        for i, variant in enumerate(variants):
            try:
                results = self._retriever.retrieve(
                    query=variant,
                    k=fetch_k,
                    filter_dir=filter_dir,
                    filter_type=filter_type,
                    filter_datasource_ids=filter_datasource_ids,
                )
                all_results.extend(results)
            except Exception as exc:
                if i == 0:
                    raise
                logger.warning(
                    "Retrieve fuer Expansion-Variante %r fehlgeschlagen: %s",
                    variant,
                    exc,
                    exc_info=True,
                )
                continue

        # 4. Deduplizierung: Composite-Key aus source + chunk_index + ds_ids
        seen: set[tuple] = set()
        unique_results: list[dict] = []
        for r in all_results:
            meta = r.get("metadata", {})
            ds_ids_raw = meta.get("ds_ids")
            if ds_ids_raw is None:
                ds_ids_norm: tuple = ()
            elif isinstance(ds_ids_raw, (list, tuple, set)):
                ds_ids_norm = tuple(sorted(ds_ids_raw))
            else:
                ds_ids_norm = (ds_ids_raw,)
            key = (meta.get("source", ""), meta.get("chunk_index", -1), ds_ids_norm)
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        # 5. Reranking immer gegen die ORIGINALFRAGE (nicht gegen Expansion)
        reranked = self._reranker.rerank(question, unique_results, top_k=k)

        # 6. Sibling-Expansion
        if self._reranker.enabled:
            with _tracer.start_as_current_span("sibling_expansion") as sib_span:
                sib_span.set_attribute("input_count", len(reranked))
                reranked = self._expand_sibling_chunks(
                    reranked,
                    unique_results,
                    filter_datasource_ids=filter_datasource_ids,
                )
                sib_span.set_attribute("output_count", len(reranked))

        return reranked

    def _expand_sibling_chunks(
        self,
        selected: list[dict],
        candidates: list[dict],
        filter_datasource_ids: list[int] | None = None,
    ) -> list[dict]:
        max_expand = self._config.get("retrieval", {}).get(
            "max_sibling_chunks",
            5,
        )
        max_total = self._config.get("retrieval", {}).get(
            "max_context_chunks",
            DEFAULT_MAX_CONTEXT_CHUNKS,
        )
        return expand_sibling_chunks(
            selected,
            candidates,
            max_expand,
            max_total=max_total,
            load_source_chunks=self._load_sibling_chunks,
            fallback_datasource_ids=filter_datasource_ids,
        )

    def _load_sibling_chunks(
        self,
        source: str,
        filter_datasource_ids: list[int] | None = None,
    ) -> list[dict]:
        try:
            return self._retriever.load_chunks_by_source(
                source,
                filter_datasource_ids=filter_datasource_ids,
            )
        except Exception as exc:
            logger.warning(
                "Sibling-Expansion: Nachladen von %r fehlgeschlagen: %s",
                source,
                exc,
                exc_info=True,
            )
            return []

    def _expansion_prefix(self) -> str:
        """Gibt Expansion-Prefix fuer _last_prompt zurueck, oder '' wenn keine Expansion."""
        if len(self._last_expansions) > 1:
            exp_str = ", ".join(f'"{v}"' for v in self._last_expansions[1:])
            return f"[Expansion: {exp_str}]\n\n"
        return ""

    def _build_prompt(
        self, context: str, question: str, history_block: str = ""
    ) -> tuple[str, str]:
        """Baut (system_prompt, user_message) fuer das Backend.

        Das bestehende RAG-Template bleibt vollstaendig in der user_message,
        weil es historisch {context} und {question} in EINEM String enthaelt.
        Ein optionaler, bereits vom Aufrufer ausgewaehlter system_prompt
        wird separat an das Backend uebergeben.
        """
        import re

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

        # Single-Pass-Substitution verhindert Second-Order-Replace:
        # Bei verketteten .replace() wuerde ein {question}-Literal im Context
        # vom zweiten Aufruf in die User-Frage uebersetzt (Prompt-Injection).
        _placeholders = {"context": context, "question": question}
        filled_template = re.sub(
            r"\{(context|question)\}",
            lambda m: _placeholders[m.group(1)],
            self._prompt_template,
        )
        user_message = f"{date_prefix}{history_part}\n\n{filled_template}"
        system_prompt = build_hardened_system_prompt(self._system_prompt or "")
        return system_prompt, user_message

    def query(
        self,
        question: str,
        k: int = 5,
        num_ctx: int | None = None,
        filter_dir: str | None = None,
        filter_type: str | None = None,
        filter_datasource_ids: list[int] | None = None,
        think: bool | None = None,
    ) -> dict:
        """Run a RAG query (non-streaming).

        Returns:
            Dict with 'answer', 'sources', and 'context_chunks'.
        """
        self._last_prompt = ""
        self._context_warning = None
        self._last_expansions = []
        results = self._retrieve_and_rerank(
            question,
            k,
            filter_dir,
            filter_type,
            filter_datasource_ids,
        )

        if not results:
            return {
                "answer": "Keine relevanten Dokumente gefunden.",
                "sources": [],
                "context_chunks": [],
            }

        max_chunks = self._config.get("retrieval", {}).get(
            "max_context_chunks", DEFAULT_MAX_CONTEXT_CHUNKS
        )
        if len(results) > max_chunks:
            logger.info("Chunk-Cap: %d -> %d Chunks", len(results), max_chunks)
            results = results[:max_chunks]

        context = format_context(results)
        system_prompt, user_message = self._build_prompt(context, question)
        joined_prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        self._last_prompt = self._expansion_prefix() + joined_prompt
        effective_ctx = num_ctx or DEFAULT_LLM_NUM_CTX
        self._check_context_overflow(joined_prompt, effective_ctx)
        kwargs = {"think": think}
        if self._backend.provider == "ollama":
            kwargs["num_ctx"] = effective_ctx
        answer = self._backend.generate(system_prompt, user_message, **kwargs)

        return {
            "answer": answer,
            "sources": self._build_sources(results),
            "context_chunks": results,
        }

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
        """RAG-Query mit Streaming-Antwort.

        context_override: Vorgefertigte Chunks (z.B. aus Such-Tab),
            ueberspringt Retrieval+Reranking.
        """
        with _tracer.start_as_current_span("rag_query") as span:
            has_datasource_filter = filter_datasource_ids is not None
            self._last_trace_id = get_trace_id_for_span(span)
            span.set_attribute("question", question[:200])
            span.set_attribute("k", k)
            span.set_attribute(
                "has_filter",
                bool(filter_dir or filter_type) or has_datasource_filter,
            )
            span.set_attribute(
                "has_datasource_filter",
                has_datasource_filter,
            )
            span.set_attribute(
                "datasource_filter_count",
                len(filter_datasource_ids or []),
            )
            span.set_attribute("context_override", context_override is not None)

            self._last_prompt = ""
            self._context_warning = None
            self._last_expansions = []
            effective_ctx = num_ctx or DEFAULT_LLM_NUM_CTX

            # 1. Retrieval (oder Context-Override)
            if context_override is not None:
                results = context_override
            else:
                results = self._retrieve_and_rerank(
                    question,
                    k,
                    filter_dir,
                    filter_type,
                    filter_datasource_ids,
                )

            if not results:

                def empty():
                    yield "Keine relevanten Dokumente gefunden."

                return empty(), []

            max_chunks = self._config.get("retrieval", {}).get(
                "max_context_chunks", DEFAULT_MAX_CONTEXT_CHUNKS
            )
            if len(results) > max_chunks:
                logger.info("Chunk-Cap: %d -> %d Chunks", len(results), max_chunks)
                results = results[:max_chunks]

            sources = self._build_sources(results)

            with _tracer.start_as_current_span("context_build") as ctx_span:
                context = format_context(results)
                ctx_span.set_attribute("chunk_count", len(results))
                ctx_span.set_attribute("context_chars", len(context))

            # 2. Prompt bauen
            system_prompt, user_message = self._build_prompt(
                context,
                question,
                history_block,
            )
            joined_prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
            self._last_prompt = self._expansion_prefix() + joined_prompt
            self._check_context_overflow(joined_prompt, effective_ctx)

            # 3. Backend aufrufen
            # HINWEIS: Dieser Span misst nur die Generator-Erstellung, nicht die
            # tatsaechliche LLM-Inferenz. generate_stream() gibt einen Generator
            # zurueck — die Token-Generierung erfolgt erst beim Konsumieren.
            # Fuer End-to-End-Timing wird die Dauer im query_log erfasst.
            with _tracer.start_as_current_span("llm_setup") as llm_span:
                llm_span.set_attribute("provider", self._backend.provider)
                llm_span.set_attribute("model", getattr(self._backend, "model", ""))
                kwargs = {"think": think}
                if self._backend.provider == "ollama":
                    kwargs["num_ctx"] = effective_ctx
                gen = self._backend.generate_stream(system_prompt, user_message, **kwargs)

            span.set_attribute("source_count", len(sources))
            return gen, sources
