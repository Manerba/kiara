"""Agent-Chain: Backend-agnostischer ReAct-Loop mit Tool-Use und SSE-Streaming.

Implementiert einen iterativen Agent-Loop: Das LLM plant Schritte,
ruft Werkzeuge auf (Dokumentensuche, Datenquellen-Liste, Taschenrechner, ...)
und formuliert abschliessend eine Antwort. Events werden als AgentEvent
zurueckgegeben und koennen per SSE an das Frontend gestreamt werden.

Der AgentChain ist backend-agnostisch — die LLM-Kommunikation wird
vollstaendig an einen BaseToolAdapter delegiert (Ollama, Anthropic, OpenAI).

Architektur-Uebersicht:
    query_stream() ist der zentrale Orchestrator — ein synchroner Generator,
    der AgentEvents yieldet. Der Loop laeuft iterativ (max_iterations-begrenzt)
    und delegiert jede Aufgabe an fokussierte Helper-Methoden:

    1. _step_with_retry()          — LLM-Schritt mit einmaligem Retry
    2. _filter_tool_calls()        — Batch-Dedup + Loop-Detection
    3. _execute_tool_batch()       — Parallele Tool-Ausfuehrung via ThreadPool
    4. _build_executed_results()   — Event-Erzeugung + Budget-Truncation
    5. _stream_final_answer()      — Finale Antwort (Cloud: direkt, Ollama: stream_final)

Budget-System:
    _prepare_tool_result() waehlt Originalpassagen fuer Dokumentwerkzeuge
    innerhalb eines gemeinsamen, konfigurierbaren Zeichenbudgets aus. Ein
    Teil davon bleibt fuer read_document reserviert. Alle Adapter erhalten
    denselben fertig aufbereiteten Text einschliesslich Auslassungsangaben.
    Andere Tools verwenden _truncate_result() mit dem Adapterlimit und dem
    500-Zeichen-Floor. Alle Ergebnisse zaehlen zum 96k-Gesamtbudget.

Thread-Safety bei paralleler Tool-Ausfuehrung:
    _execute_tool_batch() nutzt einen ThreadPoolExecutor (max 4 Worker)
    fuer parallele Tool-Calls. Thread-Safety ist gewaehrleistet, weil:
    - _run_tool() arbeitet nur mit lokalen Variablen und dem call_tuple-Argument
    - deanonymize_params() erzeugt immer ein neues Dict (kein shared mutable State)
    - ToolRegistry.get_tool() ist read-only nach Initialisierung
    - Tool.execute() ist async und wird per asyncio.run() im eigenen Event-Loop
      isoliert (kein shared State zwischen Threads)
    - _tool_events.append() ist ein einzelner atomic Append auf eine Python-Liste
      (GIL-geschuetzt), Reihenfolge ist nicht kritisch fuer SSE-Events

Loop-Detection:
    _filter_tool_calls() erkennt drei Loop-Muster:
    - Identische Wiederholung: Exakt gleicher Call wie der vorherige
    - Parameter-History: Gleiche Parameter >= 2x in der History
    - Hard-Limit: Gleiches Tool >= AGENT_MAX_SAME_TOOL_CALLS (5) Aufrufe
    Bei komplettem Loop (alle Calls gefiltert) wird der Agent zur
    Antwort gezwungen.
"""

import concurrent.futures
import logging
import time
from collections.abc import Generator
from contextvars import copy_context
from dataclasses import dataclass

from kiara.exceptions import RetrievalError
from kiara.generation.tool_adapter import BaseToolAdapter, ToolAdapterError, ToolStep
from kiara.retrieval.reranker import RerankBackendError
from kiara.shared.constants import (
    AGENT_CONTEXT_BUDGET_CHARS,
    AGENT_CONTEXT_READ_RESERVE_CHARS,
    AGENT_CONTEXT_RESULT_CHARS,
    AGENT_MAX_PARALLEL_TOOL_WORKERS,
    AGENT_MAX_SAME_TOOL_CALLS,
    AGENT_RESULT_FLOOR_CHARS,
    AGENT_RETRY_SLEEP_SECONDS,
    AGENT_TOOL_MAX_RETRIES,
    AGENT_TOOL_RETRY_DELAY_SECS,
    AGENT_TOTAL_RESULT_BUDGET,
    DEFAULT_AGENT_MAX_ITERATIONS,
    DEFAULT_MAX_CONTEXT_CHUNKS,
)
from kiara.shared.secret_redact import redact_secrets_in_text, redact_secrets_recursive
from kiara.shared.tracing import (
    attach_context,
    detach_context,
    get_current_context,
    get_trace_id_for_span,
    get_tracer,
)
from kiara.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_tracer = get_tracer(__name__, pipeline="agent")

_MAX_PARALLEL_TOOL_WORKERS = AGENT_MAX_PARALLEL_TOOL_WORKERS


def _redacted_exc_repr(exc: BaseException) -> str:
    """`<ExcClass>: <redaktierte Message>` fuer sichere Log-Ausgabe.

    Tool-/Backend-Exceptions koennen Toolparameter, DSNs oder API-Keys in
    ``str(exc)`` enthalten. ``exc_info=True`` wuerde diese ueber den
    Traceback erneut in Logs leaken; daher loggen Tool-Pfade die Exception
    als kompaktes ``ClassName: redacted_text`` und verzichten auf
    ``exc_info`` in diesen Pfaden (siehe Plan V15 — Streaming-Section).
    """
    return f"{type(exc).__name__}: {redact_secrets_in_text(str(exc))}"


def _freeze_value(value: object) -> tuple:
    """Typsicher hashbare Repraesentation fuer Loop-Vergleiche."""
    if value is None:
        return ("none", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((str(k), _freeze_value(v)) for k, v in value.items())),
        )
    if isinstance(value, list):
        return ("list", tuple(_freeze_value(v) for v in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_value(v) for v in value))
    if isinstance(value, set):
        return ("set", tuple(sorted((_freeze_value(v) for v in value), key=repr)))
    return (type(value).__name__, repr(value))


def _freeze_params(params: dict) -> frozenset:
    """Macht ein Parameter-Dict hashbar fuer Loop-Vergleiche."""
    return frozenset(sorted((str(k), _freeze_value(v)) for k, v in params.items()))


def _deduplicate_sources(sources: list[dict]) -> list[dict]:
    """Dedupliziert Quellen nach Dateipfad und vorhandener Dokumentreferenz.

    Behaelt nur das erste Vorkommen jeder Quelle (hoechster Score,
    da frueheste Treffer typischerweise die relevantesten sind).
    """
    seen: set[tuple] = set()
    unique: list[dict] = []
    for src in sources:
        key = (src.get("source", ""), src.get("document_id"))
        if key[0] and key in seen:
            continue
        if key[0]:
            seen.add(key)
        unique.append(src)
    return unique


@dataclass
class AgentEvent:
    """Ein einzelnes Event aus dem Agent-Loop fuer SSE-Streaming.

    Attributes:
        event_type: Art des Events (tool_call, tool_result, sources, chunk, error, done).
        data: Event-Payload (dict, str, list oder None).
    """

    event_type: str
    data: dict | str | list | None = None


@dataclass
class _LoopDetectionResult:
    """Ergebnis der Loop-Detection in _filter_tool_calls()."""

    executable: list[tuple]
    looped: list[tuple]
    dup_tids: list[tuple]


class AgentChain:
    """Backend-agnostischer ReAct-Agent mit Tool-Loop und Streaming.

    Ablauf:
    1. adapter.init_conversation() — Frage, System-Prompt, Tools
    2. adapter.step() — LLM-Schritt (non-streaming fuer Tool-Erkennung)
    3. Bei Tool-Call: Tool lokal ausfuehren, Ergebnis zum Adapter, zurueck zu 2
    4. Bei finaler Antwort: Text yielden (is_final) oder stream_final() (Ollama)
    5. max_iterations als Sicherheitslimit
    """

    def __init__(
        self,
        config: dict,
        tool_registry: ToolRegistry,
        adapter: BaseToolAdapter,
        user_id: int | None = None,
        max_iterations: int = DEFAULT_AGENT_MAX_ITERATIONS,
        max_same_tool_calls: int = AGENT_MAX_SAME_TOOL_CALLS,
        system_prompt_override: str | None = None,
        user_groups: list[str] | None = None,
        force_initial_search: bool = True,
    ) -> None:
        self._config = config
        self._tool_registry = tool_registry
        self._adapter = adapter
        self._user_id = user_id
        self._max_iterations = max_iterations
        self._max_same_tool_calls = max_same_tool_calls
        self._tool_events: list[dict] = []
        self._user_groups = user_groups
        self._force_initial_search = force_initial_search
        self._retrieval_context_chars = 0
        self._document_read_chars = 0
        self._context_source_numbers: dict[tuple, int] = {}
        self._document_context_results: list[dict] = []

        self._system_prompt = system_prompt_override or ""
        if not self._system_prompt:
            logger.warning("Kein Agent-System-Prompt uebergeben — leerer Prompt wird verwendet")
        self._last_trace_id: str | None = None

    @property
    def last_trace_id(self) -> str | None:
        return self._last_trace_id

    @staticmethod
    def _filter_tool_calls(
        all_calls: list[tuple[str, dict, str | None]],
        recent_calls: list[tuple[str, frozenset]],
        max_same_tool_calls: int,
    ) -> _LoopDetectionResult:
        """Filtert duplizierte und Loop-erkannte Tool-Calls aus einem Batch.

        Zweistufige Filterung:
        1. Batch-interne Deduplizierung (identische Calls im selben Step)
        2. Per-Call Loop-Detection gegen recent_calls History

        Returns:
            _LoopDetectionResult mit:
            - executable: Calls die ausgefuehrt werden sollen (cn, cp, tid, frozen)
            - looped: Calls die als Loop erkannt wurden (cn, cp, tid)
            - dup_tids: Batch-interne Duplikate (cn, tid)
        """
        # Batch-interne Deduplizierung
        seen_in_batch: set[tuple] = set()
        unique_calls: list[tuple[str, dict, str | None, frozenset]] = []
        dup_tids: list[tuple[str, str | None]] = []
        for cn, cp, tid in all_calls:
            frozen = _freeze_params(cp)
            key = (cn, frozen)
            if key in seen_in_batch:
                dup_tids.append((cn, tid))
                logger.info("Paralleler Duplikat-Call '%s' im selben Step uebersprungen", cn)
            else:
                seen_in_batch.add(key)
                unique_calls.append((cn, cp, tid, frozen))

        # Per-Call Loop-Detection
        executable: list[tuple[str, dict, str | None, frozenset]] = []
        looped: list[tuple[str, dict, str | None]] = []

        # Prospektive History: recent_calls + bereits in diesem Batch akzeptierte Calls.
        # Verhindert dass parallele Calls im selben Batch das max_same_tool_calls Limit
        # gemeinsam ueberschreiten — jeder einzelne Call laege sonst unter dem Limit.
        prospective_calls: list[tuple[str, frozenset]] = list(recent_calls)

        for cn, cp, tid, frozen in unique_calls:
            current = (cn, frozen)

            identical_repeat = prospective_calls and prospective_calls[-1] == current
            same_params_count = sum(1 for n, fp in prospective_calls if n == cn and fp == frozen)
            same_tool_count = sum(1 for n, _ in prospective_calls if n == cn)

            is_loop = False
            loop_reason = ""
            if identical_repeat or same_params_count >= 2:
                is_loop = True
                if identical_repeat:
                    loop_reason = "identischen Parametern (direkte Wiederholung)"
                else:
                    loop_reason = (
                        f"identischen Parametern "
                        f"({same_params_count}x in History, "
                        f"dies waere Aufruf #{same_params_count + 1})"
                    )
            elif cn != "read_document" and same_tool_count >= max_same_tool_calls > 0:
                is_loop = True
                loop_reason = f"{same_tool_count}x aufgerufen (Hard-Limit)"

            if is_loop:
                logger.info(
                    "Loop-Filter: '%s' mit %s — Call uebersprungen",
                    cn,
                    loop_reason,
                )
                looped.append((cn, cp, tid))
            else:
                executable.append((cn, cp, tid, frozen))
                prospective_calls.append((cn, frozen))

        return _LoopDetectionResult(executable=executable, looped=looped, dup_tids=dup_tids)

    @property
    def last_usage(self) -> dict:
        """Token-Usage-Daten des letzten Adapter-Aufrufs."""
        return self._adapter.last_usage

    @property
    def last_system_prompt(self) -> str:
        """System-Prompt des Agent (fuer Audit-Trail)."""
        return self._system_prompt

    @property
    def last_conversation(self) -> str:
        """Vollstaendiger Konversationsverlauf des Adapters (fuer Audit-Trail)."""
        return self._adapter.last_conversation

    # ------------------------------------------------------------------
    # Extracted helpers for query_stream() — each handles one concern
    # ------------------------------------------------------------------

    def _step_with_retry(self) -> tuple["ToolStep | None", AgentEvent | None, bool]:
        """Fuehrt einen LLM-Schritt aus, mit einmaligem Retry bei retryable-Fehlern.

        Returns:
            (step, None, retried) bei Erfolg oder (None, error_event, retried) bei Fehler.

        ToolAdapterError kann Toolparameter, DSNs oder API-Keys in der
        Message tragen (z.B. wenn der Adapter die Original-HTTP-Antwort
        oder das Backend-Echo durchreicht). AgentEvent("error") wird vom
        SSE-Stream und vom security_log weitergegeben — daher muessen
        Exception-Texte hier vor Ausgabe redaktiert werden.
        """
        try:
            return self._adapter.step(), None, False
        except ToolAdapterError as exc:
            if exc.retryable:
                time.sleep(AGENT_RETRY_SLEEP_SECONDS)
                try:
                    return self._adapter.step(), None, True
                except ToolAdapterError as retry_exc:
                    return (
                        None,
                        AgentEvent(
                            "error",
                            {"message": redact_secrets_in_text(str(retry_exc))},
                        ),
                        True,
                    )
            return (
                None,
                AgentEvent("error", {"message": redact_secrets_in_text(str(exc))}),
                False,
            )

    def _execute_tool_batch(
        self,
        executable: list[tuple[str, dict, str | None, frozenset]],
        parent_ctx=None,
    ) -> list[tuple[str, dict, str | None, dict]]:
        """Fuehrt eine Liste von Tool-Calls aus (parallel oder sequentiell).

        Bei mehreren Calls wird ein ThreadPoolExecutor verwendet
        (max. AGENT_MAX_PARALLEL_TOOL_WORKERS Worker). Bei einem
        einzelnen Call wird direkt im Haupt-Thread ausgefuehrt.

        Args:
            executable: Liste von (tool_name, params, tool_use_id, frozen_params).
            parent_ctx: OTel-Context fuer Propagation in Worker-Threads (optional).

        Returns:
            Liste von (tool_name, de-anonymized_params, tool_use_id, result_dict).
        """
        if len(executable) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(executable), _MAX_PARALLEL_TOOL_WORKERS)
            ) as pool:
                submitted = [
                    (
                        pool.submit(copy_context().run, self._run_tool, (cn, cp, tid), parent_ctx),
                        cn,
                        tid,
                    )
                    for cn, cp, tid, _ in executable
                ]
                results: list[tuple[str, dict, str | None, dict]] = []
                for fut, cn, tid in submitted:
                    try:
                        results.append(fut.result())
                    except Exception as exc:
                        from kiara.retrieval.index_compatibility import IndexCompatibilityError

                        if isinstance(
                            exc, (IndexCompatibilityError, RetrievalError, RerankBackendError)
                        ):
                            raise
                        # str(exc) kann Toolparameter, DSNs oder Secrets
                        # enthalten — `exc_info=True` wuerde das Original ueber
                        # den Traceback erneut in Logs leaken. Kompakte
                        # redaktierte Repraesentation ist hier ausreichend
                        # (Plan V15: Tool-/SSE-Pfade duerfen str(exc) nicht
                        # unredaktiert ausgeben).
                        redacted_repr = _redacted_exc_repr(exc)
                        logger.error(
                            "Paralleler Tool-Future '%s' propagated unerwartet: %s",
                            cn,
                            redacted_repr,
                        )
                        results.append(
                            (
                                cn,
                                {},
                                tid,
                                {
                                    "tool": cn,
                                    "params": {},
                                    "error": redact_secrets_in_text(str(exc)),
                                },
                            )
                        )
                return results
        cn, cp, tid, _ = executable[0]
        return [self._run_tool((cn, cp, tid), parent_ctx)]

    def _truncate_result(
        self,
        display: str,
        total_result_chars: int,
    ) -> tuple[str, int]:
        """Kuerzt einen Tool-Ergebnis-Text basierend auf dem Gesamt-Zeichenbudget.

        Budget-System:
        - AGENT_TOTAL_RESULT_BUDGET (96k): Globales Zeichenlimit ueber alle
          Tool-Ergebnisse einer Agent-Session. Verhindert Context-Overflow.
        - adapter.result_truncate_limit: Per-Result-Limit (Cloud 24k, Ollama 4k).
        - AGENT_RESULT_FLOOR_CHARS (500): Minimale Zeichenanzahl pro Ergebnis,
          auch wenn das Budget bereits erschoepft ist. Stellt sicher, dass jedes
          Ergebnis mindestens eine nuetzliche Zusammenfassung enthaelt.

        Truncation-Logik:
        1. effective_limit = min(adapter_limit, remaining_budget)
        2. Wenn display > effective_limit: target = max(effective_limit, floor)
        3. Wenn display > target: Kuerzen + Suffix ("[Budget erschoepft]" oder "...")

        Args:
            display: Der zu kuerzende Ergebnis-Text.
            total_result_chars: Bisheriger Gesamt-Zeichenverbrauch.

        Returns:
            (moeglicherweise gekuerzter Text, neuer Gesamt-Zeichenverbrauch).
        """
        adapter_limit = self._adapter.result_truncate_limit
        remaining_budget = max(0, AGENT_TOTAL_RESULT_BUDGET - total_result_chars)
        effective_limit = min(adapter_limit, remaining_budget)
        if len(display) > effective_limit:
            target = max(effective_limit, AGENT_RESULT_FLOOR_CHARS)
            if len(display) > target:
                suffix = (
                    " [Budget erschoepft — gekuerzt]"
                    if effective_limit < AGENT_RESULT_FLOOR_CHARS
                    else "..."
                )
                display = display[: max(target - len(suffix), 0)].rstrip() + suffix
        total_result_chars += len(display)
        return display, total_result_chars

    def _prepare_tool_result(
        self,
        result: dict,
        tool_name: str,
        query: str,
        total_result_chars: int,
        tool_use_id: str | None = None,
    ) -> tuple[str, list[dict], int]:
        """Ein gemeinsamer, bereits budgetierter Inhalt fuer alle Adapter."""
        from functools import lru_cache

        from kiara.generation.adapters.base import PreparedToolText
        from kiara.generation.context_selection import (
            prepare_context,
            prepare_document_read,
            render_context,
        )
        from kiara.shared.prompt_defense import sanitize_with_report
        from kiara.shared.prompt_defense_audit import log_defense_event

        # Kandidaten werden auf der tatsaechlichen Modell-Darstellung gemessen.
        # Wiederholtes Rendern darf keine neuen Pseudonyme fuer denselben Text
        # erzeugen; der Cache lebt nur fuer dieses einzelne Werkzeug-Ergebnis.
        prepare_text = lru_cache(maxsize=128)(self._adapter.prepare_tool_text)

        raw_chunks = result.get("raw_chunks")
        if raw_chunks and tool_name in {"search_documents", "read_document"}:
            cfg = self._config.get("chat", {}).get("agent", {})
            total_budget = max(0, int(cfg.get("context_budget_chars", AGENT_CONTEXT_BUDGET_CHARS)))
            result_budget = max(0, int(cfg.get("context_result_chars", AGENT_CONTEXT_RESULT_CHARS)))
            reserve = min(
                total_budget,
                max(
                    0,
                    int(
                        cfg.get(
                            "context_read_reserve_chars",
                            AGENT_CONTEXT_READ_RESERVE_CHARS,
                        )
                    ),
                ),
            )
            replacements = []
            read_request = result.get("document_read") if tool_name == "read_document" else None
            if read_request:
                reference = read_request["document_id"]
                for previous in self._document_context_results:
                    if previous["tool"] != "search_documents":
                        continue
                    remaining = [
                        c
                        for c in previous["chunks"]
                        if c["metadata"].get("document_ref") != reference
                    ]
                    references = [
                        meta
                        for meta in previous["references"]
                        if meta.get("document_ref") != reference
                    ]
                    if len(remaining) == len(previous["chunks"]) and len(references) == len(
                        previous["references"]
                    ):
                        continue
                    replacement = prepare_text(
                        render_context(remaining, references)
                        if remaining or references
                        else "[Suchauszug durch read_document ersetzt.]"
                    )
                    replacements.append((previous, remaining, references, replacement))
            released = sum(
                len(previous["text"]) - len(text) for previous, _, _, text in replacements
            )
            available = total_budget - self._retrieval_context_chars + released
            if tool_name != "read_document":
                available -= max(0, reserve - self._document_read_chars)
            budget = max(
                0,
                min(
                    result_budget,
                    available,
                    AGENT_TOTAL_RESULT_BUDGET - total_result_chars + released,
                ),
            )
            reranker = None
            if not read_request and cfg.get("final_reranking", True):
                from kiara.retrieval.reranker import create_reranker

                reranker = create_reranker(self._config)
            if read_request:
                context_loader = getattr(raw_chunks, "load_context", None)
                context_chunks = (
                    context_loader()
                    if context_loader is not None
                    and getattr(self._adapter, "requires_document_context", False) is True
                    else None
                )
                prepared = prepare_document_read(
                    raw_chunks,
                    request=read_request,
                    budget=budget,
                    source_numbers=self._context_source_numbers,
                    prepare_text=prepare_text,
                    context_chunks=context_chunks,
                )
                if prepared.chunks:
                    for previous, remaining, references, replacement in replacements:
                        self._adapter.replace_tool_result(
                            previous["text"], replacement, previous["tool_use_id"]
                        )
                        previous.update(text=replacement, chunks=remaining, references=references)
                    self._retrieval_context_chars -= released
                    total_result_chars -= released
                    result["document_read"] = prepared.chunks[0]["metadata"]["document_read"]
                else:
                    result["document_read"] = {**read_request, "budget_exhausted": True}
            else:
                prepared = prepare_context(
                    raw_chunks,
                    budget=budget,
                    query=query,
                    max_sources=max(
                        1,
                        int(
                            self._config.get("retrieval", {}).get(
                                "max_context_chunks",
                                DEFAULT_MAX_CONTEXT_CHUNKS,
                            )
                        ),
                    ),
                    source_diversity=cfg.get("raw_chunk_compaction", True),
                    reranker=reranker,
                    source_numbers=self._context_source_numbers,
                    prepare_text=prepare_text,
                )
            display = prepared.text
            if prepared.truncated:
                from kiara.shared.processing_warnings import report_processing_warning

                report_processing_warning(
                    "context",
                    (
                        "budget_exhausted"
                        if not prepared.chunks
                        else "document_partial"
                        if read_request
                        else "budget_limited"
                    ),
                    details={
                        "tool": tool_name,
                        "budget_chars": budget,
                        "input_chars": prepared.input_chars,
                        "output_chars": len(display),
                    },
                )
            # Statusmeldungen enthalten keine Dokumentbelege. Sie zaehlen zum
            # allgemeinen Werkzeugbudget, ohne die Leserreserve aufzubrauchen.
            self._retrieval_context_chars += len(display)
            if tool_name == "read_document":
                self._document_read_chars += len(display)
            if not display:
                notice = "[Kontextbudget erschoepft; dieses Werkzeug-Ergebnis wurde ausgelassen.]"
                display = PreparedToolText(
                    prepare_text(notice)[: max(0, AGENT_TOTAL_RESULT_BUDGET - total_result_chars)]
                )
            self._document_context_results.append(
                {
                    "tool": tool_name,
                    "text": display,
                    "chunks": prepared.chunks,
                    "references": prepared.references,
                    "tool_use_id": tool_use_id,
                }
            )
            logger.info(
                "Agent-Kontext: tool=%s input_chars=%d output_chars=%d sources=%d "
                "budget=%d truncated=%s",
                tool_name,
                prepared.input_chars,
                len(display),
                len(prepared.chunks),
                budget,
                prepared.truncated,
            )
            return display, prepared.chunks, total_result_chars + len(display)

        display = result.get("result", result.get("error", ""))
        report = sanitize_with_report(display)
        if report.severity != "none":
            log_defense_event("tool_result", report)
        display, total_result_chars = self._truncate_result(
            prepare_text(report.clean_text), total_result_chars
        )
        return PreparedToolText(display), [], total_result_chars

    def _build_executed_results(
        self,
        executed: list[tuple[str, dict, str | None, dict]],
        iteration: int,
        all_sources: list[dict],
        total_result_chars: int,
    ) -> tuple[
        list[AgentEvent], list[tuple[str, str, str | None]], dict[str | None, list[dict]], int
    ]:
        """Verarbeitet ausgefuehrte Tool-Ergebnisse: Events, Budget, Quellen.

        Iteriert ueber die ausgefuehrten Tool-Calls und:
        1. Erzeugt tool_call/tool_result-Events fuer SSE-Streaming
        2. Sammelt Quellen (sources) aus SearchDocuments-Ergebnissen
        3. Wendet Budget-Truncation auf Ergebnis-Texte an
        4. Sammelt raw_chunks fuer Final-Prompt-Formatting

        Args:
            executed: Ergebnisse aus _execute_tool_batch().
            iteration: Aktuelle Loop-Iteration (fuer Event-Daten).
            all_sources: Quellen-Akkumulator (wird in-place erweitert).
            total_result_chars: Bisheriger Gesamt-Zeichenverbrauch.

        Returns:
            Tuple (events, batch_results, raw_chunks_map, new_total_chars):
            - events: Liste von AgentEvents zum Yielden
            - batch_results: (tool_name, display_text, tool_use_id) fuer add_tool_results()
            - raw_chunks_map: tool_use_id -> raw_chunks fuer add_tool_results()
            - new_total_chars: Aktualisierter Gesamt-Zeichenverbrauch
        """
        events: list[AgentEvent] = []
        batch_results: list[tuple[str, str, str | None]] = []
        raw_chunks_map: dict[str | None, list[dict]] = {}
        pending_context: dict[int, dict] = {}

        for call_name, params, call_tid, result in executed:
            tool_call_data = {
                "tool": call_name,
                "params": redact_secrets_recursive(params),
                "iteration": iteration + 1,
                "result": str(result.get("result", result.get("error", "")))[:500],
            }
            tool = self._tool_registry.get_tool(call_name)
            if call_name == "search_documents" and hasattr(tool, "source_diagnostics"):
                diagnostics = tool.source_diagnostics(params)
                if isinstance(diagnostics, dict):
                    tool_call_data["source_selection"] = diagnostics
            self._tool_events.append(tool_call_data)
            events.append(AgentEvent("tool_call", tool_call_data))
            if result.get("sources"):
                all_sources.extend(result["sources"])
            events.append(
                AgentEvent(
                    "tool_result",
                    {
                        "tool": call_name,
                        "success": "error" not in result,
                        "summary": result.get("result", result.get("error", ""))[:200],
                    },
                )
            )

            previous_count = len(self._document_context_results)
            display, raw_chunks, total_result_chars = self._prepare_tool_result(
                result,
                call_name,
                str(params.get("query") or ""),
                total_result_chars,
                tool_use_id=call_tid,
            )
            if len(self._document_context_results) > previous_count:
                pending_context[len(batch_results)] = self._document_context_results[-1]
            if call_name == "read_document" and result.get("document_read"):
                tool_call_data["document_read"] = result["document_read"]
                if result["document_read"].get("budget_exhausted"):
                    events[-1].data.update(success=False, summary=display)
                if raw_chunks:
                    from kiara.shared.constants import EXCERPT_LENGTH

                    reference = raw_chunks[0]["metadata"].get("document_ref")
                    for source in all_sources:
                        if reference and source.get("document_id") == reference:
                            source["excerpt"] = raw_chunks[0]["text"][:EXCERPT_LENGTH]
                            source["document_read"] = result["document_read"]
            batch_results.append((call_name, display, call_tid))
            if raw_chunks:
                raw_chunks_map.setdefault(call_tid, []).extend(raw_chunks)

        # Nachlesen kann auch einen Suchauszug aus demselben Batch ersetzen.
        for index, context in pending_context.items():
            name, _, tid = batch_results[index]
            batch_results[index] = (name, context["text"], tid)
        raw_chunks_map = {}
        for context in pending_context.values():
            if context["chunks"]:
                raw_chunks_map.setdefault(context["tool_use_id"], []).extend(context["chunks"])
        return events, batch_results, raw_chunks_map, total_result_chars

    def _build_loop_skip_events(
        self,
        looped: list[tuple[str, dict, str | None]],
        iteration: int,
    ) -> tuple[list[AgentEvent], list[tuple[str, str, str | None]]]:
        """Erzeugt Events und Ergebnisse fuer Loop-uebersprungene Tool-Calls.

        Args:
            looped: Calls die als Loop erkannt wurden (name, params, tool_use_id).
            iteration: Aktuelle Loop-Iteration (fuer Event-Daten).

        Returns:
            (events, batch_results) — Events zum Yielden, Ergebnisse fuer add_tool_results().
        """
        events: list[AgentEvent] = []
        batch_results: list[tuple[str, str, str | None]] = []
        loop_skip_msg = (
            "Uebersprungen — identische Parameter bereits in einem vorherigen Schritt verwendet."
        )
        for cn, cp, tid in looped:
            self._tool_events.append(
                {
                    "tool": cn,
                    "params": redact_secrets_recursive(cp),
                    "iteration": iteration + 1,
                    "skipped": True,
                    "summary": loop_skip_msg[:200],
                }
            )
            events.append(
                AgentEvent(
                    "tool_call",
                    {
                        "tool": cn,
                        "params": redact_secrets_recursive(self._adapter.deanonymize_params(cp)),
                        "iteration": iteration + 1,
                        "skipped": True,
                    },
                )
            )
            events.append(
                AgentEvent(
                    "tool_result",
                    {
                        "tool": cn,
                        "success": True,
                        "summary": loop_skip_msg[:200],
                    },
                )
            )
            batch_results.append((cn, loop_skip_msg, tid))
        return events, batch_results

    def _handle_loop_abort(
        self,
        unique_calls: list[tuple[str, dict, str | None, frozenset]],
        all_calls: list[tuple[str, dict, str | None]],
        iteration: int,
    ) -> list[AgentEvent]:
        """Behandelt den Fall, dass ALLE Calls als Loops erkannt wurden.

        Erzeugt skip-Events fuer alle Calls, fuegt eine Loop-Message als
        Tool-Ergebnis hinzu und signalisiert dem Adapter, zur finalen
        Antwort ueberzugehen.

        Loop-Detection-Algorithmus (in _filter_tool_calls):
        - Identische Wiederholung: Exakt gleicher Call wie der vorherige
        - Parameter-History: Gleiche Parameter >= 2x in der History
        - Hard-Limit: Gleiches Tool >= max_same_tool_calls Aufrufe gesamt

        Args:
            unique_calls: Alle unique Calls (nach Batch-Dedup) mit frozen_params.
            all_calls: Alle original Calls (fuer add_tool_results).
            iteration: Aktuelle Loop-Iteration.

        Returns:
            Liste von AgentEvents zum Yielden.
        """
        logger.warning(
            "Loop erkannt: ALLE %d Calls gefiltert — erzwinge Antwort (Iteration %d)",
            len(unique_calls),
            iteration + 1,
        )
        loop_msg = (
            "Du hast diese Werkzeuge bereits mit denselben "
            "Parametern aufgerufen. Beantworte die Frage jetzt "
            "mit den vorhandenen Ergebnissen."
        )

        events: list[AgentEvent] = []
        for cn, cp, _tid, _frozen in unique_calls:
            self._tool_events.append(
                {
                    "tool": cn,
                    "params": redact_secrets_recursive(cp),
                    "iteration": iteration + 1,
                    "skipped": True,
                    "summary": loop_msg[:200],
                }
            )
            events.append(
                AgentEvent(
                    "tool_call",
                    {
                        "tool": cn,
                        "params": redact_secrets_recursive(self._adapter.deanonymize_params(cp)),
                        "iteration": iteration + 1,
                        "skipped": True,
                    },
                )
            )
            events.append(
                AgentEvent(
                    "tool_result",
                    {
                        "tool": cn,
                        "success": True,
                        "summary": loop_msg[:200],
                    },
                )
            )

        loop_results = [(cn, loop_msg, tid) for cn, _, tid in all_calls]
        self._adapter.add_tool_results(loop_results)
        return events

    def _handle_forced_search(
        self,
        question: str,
        all_sources: list[dict],
        total_result_chars: int = 0,
    ) -> tuple[list[AgentEvent], bool, int]:
        """Erzwingt eine initiale Dokumentensuche (Safety-Net fuer Ollama).

        Wird aufgerufen wenn Iteration 0 keinen Tool-Call ergibt und die
        Antwort nicht final ist (= lokales Modell). Cloud-Adapter liefern
        is_final=True und erreichen diesen Pfad nicht.

        Args:
            question: Die Benutzerfrage (wird als Suchquery verwendet).
            all_sources: Quellen-Akkumulator (wird in-place erweitert).
            total_result_chars: Bisheriger Gesamt-Zeichenverbrauch.

        Returns:
            (events, triggered, total_result_chars) — Events zum Yielden;
            triggered=True wenn die erzwungene Suche ausgefuehrt wurde;
            total_result_chars ist der aktualisierte Budget-Zaehler.
        """
        search_tool = self._tool_registry.get_tool("search_documents")
        if not search_tool:
            return [], False, total_result_chars

        logger.info(
            "Kein Tool-Call in Iteration 0 — erzwinge search_documents(query='%s')", question[:80]
        )
        forced_params = {"query": question}
        forced_call_data = {
            "tool": "search_documents",
            "params": forced_params,
            "iteration": 1,
        }
        if hasattr(search_tool, "source_diagnostics"):
            diagnostics = search_tool.source_diagnostics(forced_params)
            if isinstance(diagnostics, dict):
                forced_call_data["source_selection"] = diagnostics
        self._tool_events.append(forced_call_data)

        try:
            with _tracer.start_as_current_span("tool_call: search_documents") as span:
                span.set_attribute("tool_name", "search_documents")
                span.set_attribute("params_summary", str(forced_params)[:200])
                result = self._execute_tool("search_documents", forced_params)
                span.set_attribute("success", "error" not in result)
            forced_call_data["result"] = str(result.get("result", result.get("error", "")))[:500]
            if result.get("sources"):
                all_sources.extend(result["sources"])

            events = [
                AgentEvent("tool_call", forced_call_data),
                AgentEvent(
                    "tool_result",
                    {
                        "tool": "search_documents",
                        "success": "error" not in result,
                        "summary": result.get("result", result.get("error", ""))[:200],
                    },
                ),
            ]

            display, raw_chunks, total_result_chars = self._prepare_tool_result(
                result,
                "search_documents",
                question,
                total_result_chars,
            )
            self._adapter.add_tool_result(
                "search_documents",
                display,
                None,
                raw_chunks=raw_chunks or None,
            )
            return events, True, total_result_chars
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, (IndexCompatibilityError, RetrievalError, RerankBackendError)):
                raise
            # Exception-Texte koennen Toolparameter/Secrets enthalten —
            # `exc_info=True` ist hier bewusst weggelassen, weil der
            # Traceback denselben Text im finalen Frame leakt. Stattdessen
            # nur redaktierte Klasse+Message loggen und identischen
            # redaktierten Text in SSE/Tool-Event durchreichen.
            redacted_exc = redact_secrets_in_text(str(exc))
            logger.error("Forced-Search fehlgeschlagen: %s", _redacted_exc_repr(exc))
            forced_call_data["result"] = f"Fehler: {redacted_exc}"
            return (
                [
                    AgentEvent("tool_call", forced_call_data),
                    AgentEvent(
                        "tool_result",
                        {
                            "tool": "search_documents",
                            "success": False,
                            "summary": f"Forced-Search Exception: {redacted_exc}"[:200],
                        },
                    ),
                ],
                True,
                total_result_chars,
            )

    def _stream_final_answer(
        self,
        all_sources: list[dict],
        step: "ToolStep | None",
        think: bool,
        additional_context: str = "",
    ) -> Generator[AgentEvent, None, None]:
        """Streamt die finale Antwort als AgentEvents.

        Zwei Pfade je nach Adapter-Typ:
        - Cloud-Adapter (step.is_final=True): Text ist bereits de-anonymisiert,
          wird als einzelner Chunk yielded.
        - Ollama-Adapter (step ist None oder nicht final): Separater
          stream_final()-Aufruf (Double-Generation-Pattern), yieldet
          Token fuer Token.

        Args:
            all_sources: Gesammelte Quellen aus allen Tool-Calls.
            step: Letzter ToolStep (None beim Post-Loop-/Loop-Abbruch-Pfad).
            think: Wenn True, <think>-Bloecke in der Antwort durchstreamen.
            additional_context: Optionaler Zusatzkontext fuer stream_final()
                                (z.B. "Verwende KEINEN TOOL_CALL" bei Max-Iterations).

        Yields:
            AgentEvent("sources", ...), AgentEvent("chunk", ...), AgentEvent("done").
        """
        if all_sources:
            yield AgentEvent("sources", _deduplicate_sources(all_sources))

        if step is not None and step.is_final:
            # Cloud-Adapter: text ist bereits die de-anonymisierte Antwort
            yield AgentEvent("chunk", {"text": step.text})
        else:
            # Ollama: Separaten Final-Prompt streamen (Double-Generation-Pattern)
            try:
                for chunk in self._adapter.stream_final(
                    additional_context=additional_context, think=think
                ):
                    yield AgentEvent("chunk", {"text": chunk})
            except ToolAdapterError as exc:
                # ToolAdapterError-Texte koennen Backend-Antworten mit
                # Toolparametern/Secrets durchreichen. `exc_info=True` haengt
                # denselben Text ueber den Traceback nochmal an — daher nur
                # redaktierte Klasse+Message loggen.
                logger.error("stream_final fehlgeschlagen: %s", _redacted_exc_repr(exc))
                yield AgentEvent("error", {"message": redact_secrets_in_text(str(exc))})
            except Exception as exc:
                # Unerwartete (nicht-ToolAdapter) Fehler ohne Traceback loggen,
                # weil unbekannte Backend-Antworten ebenfalls Toolparameter
                # tragen koennten. Nutzer erhaelt generischen Text.
                logger.error(
                    "stream_final unerwartet fehlgeschlagen: %s",
                    _redacted_exc_repr(exc),
                )
                yield AgentEvent(
                    "error",
                    {"message": "Finale Antwort konnte nicht erzeugt werden."},
                )

        yield AgentEvent("done")

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------

    def query_stream(
        self, question: str, history_block: str = "", think: bool = False
    ) -> Generator[AgentEvent, None, None]:
        """Fuehrt den Agent-Loop aus und yieldet AgentEvents.

        Orchestriert den ReAct-Loop durch Delegation an fokussierte Methoden:
        - _step_with_retry(): LLM-Schritt mit Retry-Logik
        - _filter_tool_calls(): Batch-Dedup + Loop-Detection
        - _handle_loop_abort(): Abbruch bei komplettem Loop
        - _execute_tool_batch(): Parallele/sequentielle Tool-Ausfuehrung
        - _build_executed_results(): Event-Erzeugung + Budget-Truncation
        - _build_loop_skip_events(): Events fuer uebersprungene Calls
        - _handle_forced_search(): Safety-Net fuer lokale Modelle
        - _stream_final_answer(): Finale Antwort-Streaming

        Args:
            question: Die Benutzerfrage.
            history_block: Optionaler Gespraechsverlauf als Prompt-Block.
            think: Wenn True, Thinking aktivieren und <think>-Bloecke
                   in der finalen Antwort durchstreamen.

        Yields:
            AgentEvent-Instanzen fuer jeden Schritt (tool_call, tool_result,
            sources, chunk, error, done).
        """
        with _tracer.start_as_current_span("agent_query") as root_span:
            self._last_trace_id = get_trace_id_for_span(root_span)
            root_span.set_attribute("question", question[:200])
            root_span.set_attribute("max_iterations", self._max_iterations)

            tools = self._tool_registry.list_tools(user_groups=self._user_groups)
            root_span.set_attribute("tool_count", len(tools))
            from kiara.shared.prompt_defense import (
                TAG_HISTORY,
                build_hardened_system_prompt,
                sanitize_with_report,
                wrap_external_data,
            )
            from kiara.shared.prompt_defense_audit import log_defense_event

            if history_block:
                history_report = sanitize_with_report(history_block)
                if history_report.severity != "none":
                    log_defense_event("chat_history", history_report)
                history_block = wrap_external_data(history_report.clean_text, TAG_HISTORY)

            hardened_system_prompt = build_hardened_system_prompt(self._system_prompt)
            self._adapter.init_conversation(
                question,
                hardened_system_prompt,
                tools,
                history_block,
            )
            self._tool_events = []
            if hasattr(self._adapter, "_think"):
                self._adapter._think = think

            all_sources: list[dict] = []
            recent_calls: list[tuple[str, frozenset]] = []
            total_result_chars = 0
            self._retrieval_context_chars = 0
            self._document_read_chars = 0
            self._context_source_numbers = {}
            self._document_context_results = []
            max_same_tool_calls = self._max_same_tool_calls
            total_tool_calls = 0

            for iteration in range(self._max_iterations):
                with _tracer.start_as_current_span("agent_iteration") as iter_span:
                    iter_span.set_attribute("iteration", iteration + 1)

                    # 2. LLM-Schritt
                    with _tracer.start_as_current_span("llm_step") as step_span:
                        step, error_event, retried = self._step_with_retry()
                        step_span.set_attribute("retried", retried)
                    if error_event is not None:
                        root_span.set_attribute("total_iterations", iteration + 1)
                        root_span.set_attribute("total_tool_calls", total_tool_calls)
                        yield error_event
                        yield AgentEvent("done")
                        return

                    if step.tool_call:
                        iter_span.set_attribute("has_tool_call", True)
                        if think and step.text and "<think>" in step.text:
                            import re

                            _think_re = re.compile(r"<think>([\s\S]*?)</think>")
                            for m in _think_re.finditer(step.text):
                                yield AgentEvent("chunk", {"text": f"<think>{m.group(1)}</think>"})

                        all_calls = [(step.tool_call[0], step.tool_call[1], step.tool_use_id)]
                        if step.parallel_calls:
                            all_calls.extend(step.parallel_calls)

                        filt = self._filter_tool_calls(
                            all_calls,
                            recent_calls,
                            max_same_tool_calls,
                        )
                        unique_calls = filt.executable + [
                            (cn, cp, tid, _freeze_params(cp)) for cn, cp, tid in filt.looped
                        ]

                        if filt.looped:
                            with _tracer.start_as_current_span("loop_detection") as loop_span:
                                loop_span.set_attribute("filtered_count", len(filt.looped))

                        if not filt.executable:
                            for event in self._handle_loop_abort(
                                unique_calls, all_calls, iteration
                            ):
                                yield event
                            break

                        # Tool-Ausfuehrung MIT Context-Propagation
                        total_tool_calls += len(filt.executable)
                        executed = self._execute_tool_batch(
                            filt.executable, parent_ctx=get_current_context()
                        )

                        # zip statt next()-Lookup: _execute_tool_batch liefert
                        # die Ergebnisse positionsgleich zu filt.executable zurueck
                        # ([f.result() for f in futures]). Ein name-basierter Lookup
                        # waere bei tid=None (Ollama) nicht eindeutig, wenn mehrere
                        # parallele Calls denselben Tool-Namen tragen.
                        for (cn, _cp, _tid, frozen), (_, _, _, result_for_call) in zip(
                            filt.executable,
                            executed,
                            strict=False,
                        ):
                            if result_for_call is not None:
                                recent_calls.append((cn, frozen))

                        exec_events, batch_results, raw_chunks_map, total_result_chars = (
                            self._build_executed_results(
                                executed,
                                iteration,
                                all_sources,
                                total_result_chars,
                            )
                        )
                        for event in exec_events:
                            yield event

                        loop_events, loop_batch = self._build_loop_skip_events(
                            filt.looped,
                            iteration,
                        )
                        for event in loop_events:
                            yield event
                        batch_results.extend(loop_batch)

                        for dup_name, dup_tid in filt.dup_tids:
                            batch_results.append(
                                (
                                    dup_name,
                                    "Duplikat — bereits im selben Schritt abgehandelt.",
                                    dup_tid,
                                )
                            )

                        if batch_results:
                            self._adapter.add_tool_results(batch_results, raw_chunks_map)
                        continue

                    # 3.5 Forced Initial Search
                    if self._force_initial_search and iteration == 0 and not step.is_final:
                        with _tracer.start_as_current_span("forced_search") as fs_span:
                            forced_events, triggered, total_result_chars = (
                                self._handle_forced_search(
                                    question,
                                    all_sources,
                                    total_result_chars,
                                )
                            )
                            fs_span.set_attribute("triggered", triggered)
                        for event in forced_events:
                            yield event
                        if triggered:
                            total_tool_calls += 1
                            recent_calls.append(
                                ("search_documents", _freeze_params({"query": question}))
                            )
                            continue

                    iter_span.set_attribute("has_tool_call", False)

                # 4. Finale Antwort
                # HINWEIS: final_answer-Span ist BEWUSST vor dem yield from —
                # gleiche Logik wie llm_setup in chain.py. Der Span markiert den
                # Entscheidungspunkt, nicht die Streaming-Dauer.
                with _tracer.start_as_current_span("final_answer") as fa_span:
                    fa_span.set_attribute("source_count", len(all_sources))
                    fa_span.set_attribute("is_cloud_final", step is not None and step.is_final)
                yield from self._stream_final_answer(all_sources, step, think)

                root_span.set_attribute("total_iterations", iteration + 1)
                root_span.set_attribute("total_tool_calls", total_tool_calls)
                return

            # 5. Max-Iterations
            root_span.set_attribute("total_iterations", self._max_iterations)
            root_span.set_attribute("total_tool_calls", total_tool_calls)

            with _tracer.start_as_current_span("final_answer") as fa_span:
                fa_span.set_attribute("source_count", len(all_sources))
                fa_span.set_attribute("is_cloud_final", False)
            yield from self._stream_final_answer(
                all_sources,
                step=None,
                think=think,
                additional_context=(
                    "WICHTIG: Du hast KEINEN Zugriff mehr auf Werkzeuge. "
                    "Verwende KEINEN TOOL_CALL. Beantworte die Frage JETZT "
                    "basierend auf den bisherigen Ergebnissen."
                ),
            )

    def _run_tool(
        self,
        call_tuple: tuple[str, dict, str | None],
        parent_ctx=None,
    ) -> tuple[str, dict, str | None, dict]:
        """Fuehrt einen einzelnen Tool-Call aus (thread-safe).

        Wird vom ThreadPoolExecutor in parallelen Threads aufgerufen.
        De-anonymisiert Parameter, fuehrt das Tool aus, faengt Exceptions ab.

        Args:
            call_tuple: (tool_name, raw_params, tool_use_id)
            parent_ctx: OTel-Context fuer Span-Propagation (optional).

        Returns:
            (tool_name, de-anonymized_params, tool_use_id, result_dict)
        """
        cn, cp, tid = call_tuple

        token = None
        if parent_ctx is not None:
            try:
                token = attach_context(parent_ctx)
            except Exception as exc:
                logger.warning(
                    "attach_context fuer paralleles Tool '%s' fehlgeschlagen: %s",
                    cn,
                    exc,
                    exc_info=True,
                )
        try:
            with _tracer.start_as_current_span(f"tool_call: {cn}") as span:
                span.set_attribute("tool_name", cn)
                span.set_attribute("params_summary", str(redact_secrets_recursive(cp))[:200])
                params = cp
                try:
                    params = self._adapter.deanonymize_params(cp)
                    result = self._execute_tool(cn, params)
                    span.set_attribute("success", "error" not in result)
                except Exception as exc:
                    from kiara.retrieval.index_compatibility import IndexCompatibilityError

                    if isinstance(
                        exc, (IndexCompatibilityError, RetrievalError, RerankBackendError)
                    ):
                        raise
                    # str(exc) kann Toolparameter enthalten — Redact bevor in
                    # Tool-Result-Dict ablegen (SSE/Audit/Trace lesen daraus).
                    # exc_info=True wuerde denselben Text via Traceback erneut
                    # leaken (Befund 2 aus V15-Review).
                    redacted_exc = redact_secrets_in_text(str(exc))
                    logger.error(
                        "Paralleler Tool-Call '%s' Fehler: %s",
                        cn,
                        _redacted_exc_repr(exc),
                    )
                    result = {"tool": cn, "params": params, "error": redacted_exc}
                return cn, params, tid, result
        finally:
            if token is not None:
                detach_context(token)

    def _execute_tool(self, tool_name: str, params: dict) -> dict:
        """Fuehrt ein Tool synchron aus.

        Args:
            tool_name: Name des Tools.
            params: Parameter-Dictionary.

        Returns:
            Dict mit 'tool', 'params' und 'result' oder 'error'.
            Optional 'sources' bei SearchDocuments.
        """
        # Serverseitige Berechtigungspruefung — verhindert Ausfuehrung
        # von Tools, die der User laut list_tools() nicht sehen darf.
        allowed_names = {
            t.name for t in self._tool_registry.list_tools(user_groups=self._user_groups)
        }
        if tool_name not in allowed_names:
            logger.warning(
                "Tool-Aufruf '%s' abgelehnt — keine Berechtigung (user_groups=%s).",
                tool_name,
                self._user_groups,
            )
            return {
                "tool": tool_name,
                "params": params,
                "error": f"Werkzeug '{tool_name}' nicht erlaubt",
            }

        tool = self._tool_registry.get_tool(tool_name)
        if tool is None:
            return {
                "tool": tool_name,
                "params": params,
                "error": f"Werkzeug '{tool_name}' nicht gefunden",
            }

        valid, msg = tool.validate_params(params)
        if not valid:
            return {
                "tool": tool_name,
                "params": params,
                "error": f"Ungueltige Parameter: {msg}",
            }

        # Tool.execute() ist async def, aber query_stream() ist ein sync
        # Generator — wird von FastAPI, MCP-Server und Tests sync konsumiert.
        # Auch die Adapter-Schicht (step(), stream_final()) ist komplett sync.
        # Wenn bereits eine Event-Loop laeuft (FastAPI/uvicorn), kann
        # asyncio.run() nicht direkt aufgerufen werden → isolierter Thread
        # mit eigenem Event-Loop.  Overhead (~1ms/Call) ist vernachlaessigbar
        # gegenueber der LLM-Latenz (500ms-5s pro step()).

        last_error = None
        for attempt in range(AGENT_TOOL_MAX_RETRIES):
            try:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # OTel-Context muss in den Sub-Worker-Thread propagiert werden,
                    # sonst starten Spans aus tool.execute() (z.B. hybrid_retrieve,
                    # rerank) als eigene Root-Traces statt als Kinder des
                    # tool_call-Spans. Pattern analog zu _run_tool().
                    sub_ctx = get_current_context()

                    def _exec_with_ctx(sub_ctx=sub_ctx):
                        sub_token = None
                        if sub_ctx is not None:
                            try:
                                sub_token = attach_context(sub_ctx)
                            except Exception as exc:
                                logger.warning(
                                    "attach_context fuer Sub-Pool '%s' fehlgeschlagen: %s",
                                    tool_name,
                                    exc,
                                    exc_info=True,
                                )
                        try:
                            return asyncio.run(tool.execute(params, user_id=self._user_id))
                        finally:
                            if sub_token is not None:
                                detach_context(sub_token)

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(copy_context().run, _exec_with_ctx).result()
                else:
                    result = asyncio.run(tool.execute(params, user_id=self._user_id))
            except Exception as exc:
                from kiara.retrieval.index_compatibility import IndexCompatibilityError

                if isinstance(exc, (IndexCompatibilityError, RetrievalError, RerankBackendError)):
                    raise
                # str(exc) kann Toolparameter enthalten — Redact, da last_error
                # spaeter im LLM-sichtbaren "Nach N Versuchen fehlgeschlagen"-
                # Fehlerstring landet. exc_info=True wuerde den Original-Text
                # via Traceback erneut leaken (Befund 2 aus V15-Review).
                last_error = redact_secrets_in_text(str(exc))
                logger.warning(
                    "Tool '%s' Versuch %d/%d fehlgeschlagen: %s",
                    tool_name,
                    attempt + 1,
                    AGENT_TOOL_MAX_RETRIES,
                    _redacted_exc_repr(exc),
                )
                if attempt < AGENT_TOOL_MAX_RETRIES - 1:
                    import time

                    time.sleep(AGENT_TOOL_RETRY_DELAY_SECS)
                continue

            if result.success:
                break

            # Fehlgeschlagener ToolResult — Retry nur bei explizit transienten Fehlern
            last_error = result.error or "Unbekannter Fehler"
            logger.warning(
                "Tool '%s' Versuch %d/%d: %s",
                tool_name,
                attempt + 1,
                AGENT_TOOL_MAX_RETRIES,
                last_error,
            )
            if not result.retryable:
                break
            if attempt < AGENT_TOOL_MAX_RETRIES - 1:
                import time

                time.sleep(AGENT_TOOL_RETRY_DELAY_SECS)
        else:
            # Alle Retries fehlgeschlagen
            logger.error(
                "Tool '%s' nach %d Versuchen fehlgeschlagen: %s",
                tool_name,
                AGENT_TOOL_MAX_RETRIES,
                last_error,
            )
            return {
                "tool": tool_name,
                "params": params,
                "error": f"Nach {AGENT_TOOL_MAX_RETRIES} Versuchen fehlgeschlagen: {last_error}",
            }

        entry = {
            "tool": tool_name,
            "params": params,
        }

        if result.success:
            # Strukturierte Daten serialisieren, damit das LLM die
            # tatsaechlichen Ergebnisse sieht — nicht nur den display_text.
            # display_text ist oft nur eine Kurzform ("KI-Backends"),
            # waehrend data die vollstaendigen Datensaetze enthaelt.
            if result.data is not None:
                import json as _json

                try:
                    entry["result"] = _json.dumps(result.data, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    entry["result"] = str(result.data)
            else:
                entry["result"] = result.display_text or ""
            if isinstance(result.data, dict):
                if "sources" in result.data:
                    entry["sources"] = result.data["sources"]
                if "results" in result.data:
                    entry["raw_chunks"] = result.data["results"]
                if "document_read" in result.data:
                    entry["document_read"] = result.data["document_read"]
        else:
            entry["error"] = result.error or "Unbekannter Fehler"

        return entry
