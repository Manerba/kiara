"""RAG chain: retrieval + prompt building + LLM generation.

Die Chain orchestriert Retrieval und Generierung — das LLM-Backend
wird per Konstruktor injiziert (Dependency Injection). Die Chain
entscheidet NICHT ueber Routing oder Anonymisierung; beides wird
vom Aufrufer (api_chat.py) ueber das Backend gesteuert.
"""

import logging
from collections.abc import Callable, Generator

from kiara.exceptions import RetrievalError
from kiara.generation._chain_base import _ChainBase
from kiara.generation.backends.base import LLMBackend
from kiara.retrieval.query_expansion import QueryExpander
from kiara.retrieval.reranker import create_reranker
from kiara.retrieval.retriever import HybridRetriever
from kiara.shared.constants import (
    ABSTAIN_INSTRUCTION_MAX_SCORE_DEFAULT,
    ABSTAIN_INSTRUCTION_TEXT,
    ABSTAIN_MESSAGE_DEFAULT,
    ABSTAIN_MIN_RERANK_SCORE_DEFAULT,
    CHARS_PER_TOKEN,
    DEFAULT_LLM_NUM_CTX,
    DEFAULT_MAX_CONTEXT_CHUNKS,
    EXCERPT_LENGTH,
)
from kiara.shared.processing_warnings import report_processing_warning
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
            candidate_index.setdefault(
                (source, _ds_key(meta), meta.get("parser") or "standard"), {}
            )[ci] = chunk

    # Quelle, Datenquelle und Parserdarstellung identifizieren ein Dokument.
    candidate_index: dict[tuple[str, tuple, str], dict[int, dict]] = {}
    for c in candidates:
        _add_to_index(c)
    for c in selected:
        _add_to_index(c)

    # Ausgewaehlte Chunks identifizieren
    selected_keys: set[tuple] = set()
    for r in selected:
        meta = r.get("metadata", {})
        source = meta.get("source", "")
        ci = meta.get("chunk_index", -1)
        try:
            ci = int(ci)
        except (TypeError, ValueError):
            ci = -1
        ds_key = _ds_key(meta)
        selected_keys.add((source, ci, ds_key, meta.get("parser") or "standard"))

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
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, (IndexCompatibilityError, RetrievalError)):
                raise
            logger.warning(
                "Sibling-Expansion: Quelle %r konnte nicht nachgeladen werden: %s",
                source,
                exc,
                exc_info=True,
            )
            report_processing_warning("document_expansion")
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
    to_add: dict[tuple, dict] = {}
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
        representation = (source, ds_key, meta.get("parser") or "standard")
        source_chunks = candidate_index.get(representation, {})
        if any(idx not in source_chunks for idx in wanted):
            _load_missing_source(source, ds_key)
            source_chunks = candidate_index.get(representation, {})

        for idx in wanted:
            key = (source, idx, ds_key, representation[2])
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
        self._reranker = create_reranker(config)
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template_override or _DEFAULT_RAG_TEMPLATE
        if not prompt_template_override:
            logger.debug("Kein RAG-Prompt-Template uebergeben — Standard-Template wird verwendet")
        self._context_warning: dict | None = None
        self._last_prompt = ""
        self._last_max_rerank_score: float | None = None
        self._expander = QueryExpander(config, backend=self._backend)
        self._last_expansions: list[str] = []
        self._last_trace_id: str | None = None
        self._last_retrieval_diagnostics: dict | None = None

    @property
    def last_trace_id(self) -> str | None:
        """Trace-ID des letzten `rag_query`-Spans (fuer query_log-Korrelation)."""
        return self._last_trace_id

    @property
    def last_expansions(self) -> list[str]:
        """Query-Expansion-Varianten des letzten Aufrufs (leer wenn deaktiviert)."""
        return self._last_expansions

    @property
    def last_retrieval_diagnostics(self) -> dict | None:
        """Pfad-Diagnostik des letzten Retrieval-/Rerank-Aufrufs."""
        if self._last_retrieval_diagnostics is None:
            return None
        return dict(self._last_retrieval_diagnostics)

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
            from kiara.shared.processing_warnings import report_processing_warning

            self._context_warning = {
                "estimated_tokens": estimated_tokens,
                "num_ctx": num_ctx,
                "overflow_pct": round((estimated_tokens / num_ctx - 1) * 100),
            }
            report_processing_warning(
                "context", "window_estimate_exceeded", details=self._context_warning
            )
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

    @staticmethod
    def _with_final_context_ranks(results: list[dict]) -> list[dict]:
        ranked = []
        for rank, result in enumerate(results, 1):
            item = result.copy()
            item["final_context_rank"] = rank
            ranked.append(item)
        return ranked

    def _snapshot_retriever_diagnostics(self, variant_index: int, variant: str) -> dict | None:
        getter = getattr(self._retriever, "get_last_retrieval_diagnostics", None)
        if not callable(getter):
            return None
        diagnostics = getter()
        if not isinstance(diagnostics, dict):
            return None
        snapshot = dict(diagnostics)
        snapshot["variant_index"] = variant_index
        snapshot["variant_query"] = variant
        return snapshot

    def _set_chain_retrieval_diagnostics(
        self,
        *,
        question: str,
        variants: list[str],
        variant_diagnostics: list[dict],
        retrieved_count: int,
        deduped_count: int,
        reranked_count: int,
    ) -> None:
        self._last_retrieval_diagnostics = {
            "schema": "kiara-rag-retrieval-diagnostics",
            "schema_version": 1,
            "query": question,
            "variant_count": len(variants),
            "variants": variant_diagnostics,
            "retrieved_count": retrieved_count,
            "deduped_candidate_count": deduped_count,
            "reranking_enabled": bool(self._reranker.enabled),
            "result_count_before_context_cap": reranked_count,
        }

    def _set_final_context_diagnostics(self, results: list[dict]) -> None:
        diagnostics = dict(self._last_retrieval_diagnostics or {})
        diagnostics["final_context_count"] = len(results)
        diagnostics["final_context_chunk_ids"] = [
            str(result.get("chunk_id") or result.get("point_id") or result.get("_doc_id") or "")
            for result in results
        ]
        self._last_retrieval_diagnostics = diagnostics

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
        variant_diagnostics: list[dict] = []
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
                diagnostics = self._snapshot_retriever_diagnostics(i + 1, variant)
                if diagnostics is not None:
                    variant_diagnostics.append(diagnostics)
            except Exception as exc:
                from kiara.retrieval.index_compatibility import IndexCompatibilityError

                if isinstance(exc, (IndexCompatibilityError, RetrievalError)):
                    raise
                diagnostics = self._snapshot_retriever_diagnostics(i + 1, variant)
                if diagnostics is not None:
                    diagnostics["retrieval_error"] = {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                    variant_diagnostics.append(diagnostics)
                if i == 0:
                    raise
                report_processing_warning("query_variant")
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

        self._set_chain_retrieval_diagnostics(
            question=question,
            variants=variants,
            variant_diagnostics=variant_diagnostics,
            retrieved_count=len(all_results),
            deduped_count=len(unique_results),
            reranked_count=len(reranked),
        )
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
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, (IndexCompatibilityError, RetrievalError)):
                raise
            logger.warning(
                "Sibling-Expansion: Nachladen von %r fehlgeschlagen: %s",
                source,
                exc,
                exc_info=True,
            )
            report_processing_warning("document_expansion")
            return []

    def _expansion_prefix(self) -> str:
        """Gibt Expansion-Prefix fuer _last_prompt zurueck, oder '' wenn keine Expansion."""
        if len(self._last_expansions) > 1:
            exp_str = ", ".join(f'"{v}"' for v in self._last_expansions[1:])
            return f"[Expansion: {exp_str}]\n\n"
        return ""

    def _build_prompt(
        self,
        context: str,
        question: str,
        history_block: str = "",
        suppress_abstain_instruction: bool = False,
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
        if not suppress_abstain_instruction and self._abstain_instruction_applies():
            filled_template = f"{filled_template}\n\n{ABSTAIN_INSTRUCTION_TEXT}"
        user_message = f"{date_prefix}{history_part}\n\n{filled_template}"
        system_prompt = build_hardened_system_prompt(self._system_prompt or "")
        return system_prompt, user_message

    def _abstain_gate(self, results: list[dict]) -> tuple[str, dict] | None:
        """Stufe-1-Abstain: Score-Gate auf dem besten Reranker-Score.

        Greift nur bei retrieval.abstain_enabled UND aktivem Reranker — ohne
        Reranking gibt es keinen kalibrierten Relevanz-Score. Ergebnisse ohne
        rerank_score (z.B. Reranker degraded) loesen bewusst KEIN Abstain aus:
        ein kaputter Reranker ist kein Beleg fuer fehlende Evidenz.

        Returns:
            (abstain_message, abstain_diagnostics) oder None (kein Abstain).
        """
        retrieval_cfg = self._config.get("retrieval", {})
        if not retrieval_cfg.get("abstain_enabled", False):
            return None
        if not self._reranker.enabled or not results:
            return None
        try:
            threshold = float(
                retrieval_cfg.get("abstain_min_rerank_score", ABSTAIN_MIN_RERANK_SCORE_DEFAULT)
            )
        except (TypeError, ValueError):
            threshold = ABSTAIN_MIN_RERANK_SCORE_DEFAULT
        scores = [
            r.get("rerank_score")
            for r in results
            if isinstance(r.get("rerank_score"), (int, float))
        ]
        if not scores:
            logger.debug("Abstain-Gate uebersprungen: keine rerank_scores vorhanden")
            return None
        max_score = max(scores)
        if max_score >= threshold:
            return None
        message = str(retrieval_cfg.get("abstain_message") or "").strip() or ABSTAIN_MESSAGE_DEFAULT
        diagnostics = {
            "abstained": True,
            "reason": "max_rerank_score_below_threshold",
            "max_rerank_score": round(max_score, 6),
            "threshold": threshold,
            "candidates": len(results),
        }
        logger.info(
            "Abstain-Gate: max_rerank_score %.4f < %.4f (%d Kandidaten) — antworte ohne LLM-Call",
            max_score,
            threshold,
            len(results),
        )
        return message, diagnostics

    def _annotate_abstain_diagnostics(self, abstain_diag: dict) -> None:
        """Haengt die Abstain-Info an die Retrieval-Diagnostik (best-effort)."""
        if isinstance(self._last_retrieval_diagnostics, dict):
            self._last_retrieval_diagnostics["abstain"] = abstain_diag

    @staticmethod
    def _is_abstain_answer(answer: str) -> bool:
        """Erkennt eine Abstain-Antwort ueber die Laufzeit-Marker."""
        from kiara.shared.constants import ABSTAIN_ANSWER_MARKERS

        lowered = (answer or "").strip().lower()
        if not lowered:
            return True
        return any(marker in lowered for marker in ABSTAIN_ANSWER_MARKERS)

    def _maybe_grounded_retry(
        self,
        answer: str,
        context: str,
        question: str,
        results: list[dict],
        gen_kwargs: dict,
        filter_datasource_ids: list[int] | None = None,
    ) -> tuple[str, dict | None]:
        """Abstain-Zweitmeinung mit NLI-Gate (Zyklus 3, Policy streng quellengebunden).

        Wenn das Modell trotz vorhandenem Kontext abstiniert, wird EIN Versuch
        ohne Abstain-Instruktion generiert. Die Zweitantwort wird nur
        ausgeliefert, wenn der Kiron-NLI-Check sie als vom Kontext entailt
        bestaetigt — sonst bleibt es fail-closed beim Abstain. Adressiert die
        False-Abstains (Evidenz im Kontext, Modell zu vorsichtig), ohne die
        Quellenbindung aufzuweichen.
        """
        from kiara.generation.grounding import check_entailment, resolve_grounding_config

        settings = resolve_grounding_config(self._config)
        if not settings["enabled"]:
            return answer, None
        if not results or not self._is_abstain_answer(answer):
            return answer, None

        diag: dict = {"attempted": True, "accepted": False}
        system_prompt, user_message = self._build_prompt(
            context, question, suppress_abstain_instruction=True
        )
        self._retriever.validate_context(results, filter_datasource_ids)
        try:
            retry_answer = self._backend.generate(system_prompt, user_message, **gen_kwargs)
        except Exception as exc:
            diag["error"] = f"retry_generate {type(exc).__name__}: {exc}"
            logger.warning("Grounded-Retry: Zweitgenerierung fehlgeschlagen: %s", exc)
            return answer, diag

        if self._is_abstain_answer(retry_answer):
            diag["retry_abstained"] = True
            return answer, diag

        grounding = check_entailment(self._config, retry_answer, results)
        diag["grounding"] = grounding
        if (
            grounding.get("status") == "success"
            and grounding.get("max_entailment", 0.0) >= settings["threshold"]
        ):
            diag["accepted"] = True
            logger.info(
                "Grounded-Retry akzeptiert (max_entailment %.3f >= %.2f)",
                grounding["max_entailment"],
                settings["threshold"],
            )
            return retry_answer, diag
        logger.info(
            "Grounded-Retry verworfen (status=%s, max_entailment=%s) — Abstain bleibt",
            grounding.get("status"),
            grounding.get("max_entailment"),
        )
        return answer, diag

    @staticmethod
    def _max_rerank_score(results: list[dict]) -> float | None:
        """Bester Reranker-Score der Kandidaten; None ohne verwertbare Scores."""
        scores = [
            r.get("rerank_score")
            for r in results
            if isinstance(r.get("rerank_score"), (int, float))
        ]
        return max(scores) if scores else None

    def _abstain_instruction_applies(self) -> bool:
        """Ob die Abstain-Instruktion in den Prompt eingeblendet wird.

        Ampel-Prinzip: Bei abstain_instruction_max_score > 0 nur unterhalb
        dieses Score-Bands — starke Treffer behalten den cleanen Prompt.
        Ohne Scores (Reranker aus/degraded, context_override) gilt die
        Bedingung konservativ als erfuellt.
        """
        retrieval_cfg = self._config.get("retrieval", {})
        if not retrieval_cfg.get("abstain_instruction", False):
            return False
        try:
            max_gate = float(
                retrieval_cfg.get(
                    "abstain_instruction_max_score", ABSTAIN_INSTRUCTION_MAX_SCORE_DEFAULT
                )
            )
        except (TypeError, ValueError):
            max_gate = ABSTAIN_INSTRUCTION_MAX_SCORE_DEFAULT
        if max_gate <= 0:
            return True
        if self._last_max_rerank_score is None:
            return True
        return self._last_max_rerank_score < max_gate

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
        self._last_retrieval_diagnostics = None
        results = self._retrieve_and_rerank(
            question,
            k,
            filter_dir,
            filter_type,
            filter_datasource_ids,
        )

        self._last_max_rerank_score = self._max_rerank_score(results)

        if not results:
            self._set_final_context_diagnostics([])
            return {
                "answer": "Keine relevanten Dokumente gefunden.",
                "sources": [],
                "context_chunks": [],
                "retrieval_diagnostics": self.last_retrieval_diagnostics,
            }

        abstain = self._abstain_gate(results)
        if abstain is not None:
            message, abstain_diag = abstain
            self._set_final_context_diagnostics([])
            self._annotate_abstain_diagnostics(abstain_diag)
            return {
                "answer": message,
                "sources": [],
                "context_chunks": [],
                "abstained": True,
                "retrieval_diagnostics": self.last_retrieval_diagnostics,
            }

        max_chunks = self._config.get("retrieval", {}).get(
            "max_context_chunks", DEFAULT_MAX_CONTEXT_CHUNKS
        )
        if len(results) > max_chunks:
            from kiara.shared.processing_warnings import report_processing_warning

            report_processing_warning(
                "context",
                "chunk_limit",
                details={"available_chunks": len(results), "included_chunks": max_chunks},
            )
            logger.info("Chunk-Cap: %d -> %d Chunks", len(results), max_chunks)
            results = results[:max_chunks]
        results = self._with_final_context_ranks(results)
        self._set_final_context_diagnostics(results)

        context = format_context(results)
        system_prompt, user_message = self._build_prompt(context, question)
        joined_prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        self._last_prompt = self._expansion_prefix() + joined_prompt
        effective_ctx = num_ctx or DEFAULT_LLM_NUM_CTX
        self._check_context_overflow(joined_prompt, effective_ctx)
        kwargs = {"think": think}
        if self._backend.provider == "ollama":
            kwargs["num_ctx"] = effective_ctx
        self._retriever.validate_context(results, filter_datasource_ids)
        answer = self._backend.generate(system_prompt, user_message, **kwargs)

        answer, retry_diag = self._maybe_grounded_retry(
            answer, context, question, results, kwargs, filter_datasource_ids
        )
        if retry_diag is not None and isinstance(self._last_retrieval_diagnostics, dict):
            self._last_retrieval_diagnostics["grounded_retry"] = retry_diag

        result = {
            "answer": answer,
            "sources": self._build_sources(results),
            "context_chunks": results,
            "retrieval_diagnostics": self.last_retrieval_diagnostics,
        }
        if retry_diag is not None:
            result["grounded_retry"] = retry_diag
        return result

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
            self._last_retrieval_diagnostics = None
            effective_ctx = num_ctx or DEFAULT_LLM_NUM_CTX

            # 1. Retrieval (oder Context-Override)
            if context_override is not None:
                results = self._retriever.load_context_chunks(
                    context_override, filter_datasource_ids
                )
            else:
                results = self._retrieve_and_rerank(
                    question,
                    k,
                    filter_dir,
                    filter_type,
                    filter_datasource_ids,
                )

            self._last_max_rerank_score = self._max_rerank_score(results)

            if not results:
                self._set_final_context_diagnostics([])

                def empty():
                    yield "Keine relevanten Dokumente gefunden."

                return empty(), []

            # Abstain-Gate nur fuer frisch retrievte Ergebnisse — ein
            # context_override (z.B. Such-Tab) bringt keine kalibrierten Scores.
            if context_override is None:
                abstain = self._abstain_gate(results)
                if abstain is not None:
                    message, abstain_diag = abstain
                    self._set_final_context_diagnostics([])
                    self._annotate_abstain_diagnostics(abstain_diag)
                    span.set_attribute("abstained", True)

                    def abstained():
                        yield message

                    return abstained(), []

            max_chunks = self._config.get("retrieval", {}).get(
                "max_context_chunks", DEFAULT_MAX_CONTEXT_CHUNKS
            )
            if len(results) > max_chunks:
                from kiara.shared.processing_warnings import report_processing_warning

                report_processing_warning(
                    "context",
                    "chunk_limit",
                    details={"available_chunks": len(results), "included_chunks": max_chunks},
                )
                logger.info("Chunk-Cap: %d -> %d Chunks", len(results), max_chunks)
                results = results[:max_chunks]
            results = self._with_final_context_ranks(results)
            self._retriever.validate_context(results, filter_datasource_ids)
            self._set_final_context_diagnostics(results)

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

                def generate():
                    # Der Aufrufer kann den Generator erst spaeter konsumieren.
                    # Auch dann muss genau dieser Kontext noch freigegeben sein.
                    self._retriever.validate_context(results, filter_datasource_ids)
                    yield from self._backend.generate_stream(system_prompt, user_message, **kwargs)

                gen = generate()

            span.set_attribute("source_count", len(sources))
            return gen, sources
