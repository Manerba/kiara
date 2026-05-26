"""Ollama-Adapter — prompt-basiertes TOOL_CALL:-Parsing (lokal).

Baut bei jedem step() einen vollstaendigen Prompt mit System-Prompt,
Tool-Beschreibungen, bisherigen Ergebnissen und der Frage. Parst die
LLM-Antwort per Regex nach TOOL_CALL: Mustern.

Thinking/Chain-of-Thought (CoT) Verhalten:
    Ollama-Modelle (z.B. qwen3) unterstuetzen einen ``think``-Parameter,
    der interne Reasoning-Bloecke (<think>...</think>) aktiviert.

    - step() setzt think=False: Tool-Auswahl braucht kein CoT, und
      <think>-Bloecke wuerden das TOOL_CALL:-Regex-Parsing stoeren.
      Falls das Modell trotzdem <think>-Bloecke erzeugt, werden sie
      per Regex entfernt (Fallback-Cleaning).

    - stream_final() respektiert den ``think``-Parameter des Callers:
      * think=True: <think>-Bloecke werden durchgestreamt. Das Frontend
        rendert sie als aufklappbare Reasoning-Bloecke.
      * think=False (Default): Keine <think>-Bloecke in der Ausgabe.

    Retry-Logik bei Thinking-Timeouts:
      Thinking kann die Generierung erheblich verlaengern und zu Timeouts
      fuehren. _generate_raw() behandelt das zweistufig:
      1. Timeout beim Stream-Start (kein Token empfangen): Automatischer
         Retry mit think=False — der User bekommt eine Antwort ohne Reasoning.
      2. Timeout nach partiellem Streaming: Kein Retry moeglich (Tokens
         wurden bereits an den Client gesendet), nur Error-Logging.

Double-Generation-Pattern:
    Im Gegensatz zu Cloud-Adaptern (die die finale Antwort direkt im
    letzten step() liefern) braucht Ollama zwei Generierungen pro
    Agent-Turn: step() fuer Tool-Auswahl (non-streaming, kurz) und
    stream_final() fuer die Antwort (streaming, mit Retrieval-Kontext).
    stream_final() baut einen eigenen Prompt mit format_context() und
    optional einem Final-Rerank aller gesammelten Chunks.

Beziehungen:
    - Erbt von kiara.generation.adapters.base.BaseToolAdapter
    - Nutzt kiara.generation.backends.ollama.build_usage_from_ollama fuer Usage-Tracking
    - Importiert format_context aus kiara.generation.chain fuer Final-Prompt
"""

import ast
import json
import logging
import re
from collections.abc import Iterator

from kiara.generation.adapters.base import (
    BaseToolAdapter,
    ToolAdapterError,
    ToolStep,
    _extract_balanced_params,
)
from kiara.generation.backends.base import _clean_think_blocks
from kiara.generation.backends.ollama import build_usage_from_ollama
from kiara.shared.constants import (
    AGENT_RESULT_TRUNCATE_OLLAMA,
    DEFAULT_AGENT_NUM_CTX,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_CONTEXT_CHUNKS,
    OLLAMA_CONNECT_TIMEOUT_SECS,
    OLLAMA_POOL_TIMEOUT_SECS,
    OLLAMA_READ_TIMEOUT_SECS,
    OLLAMA_WRITE_TIMEOUT_SECS,
)
from kiara.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


def _chunk_dedupe_key(chunk: dict) -> tuple:
    """Stable Dedupe-Key fuer normale und kompaktierte Retrieval-Chunks."""
    meta = chunk.get("metadata", {}) or {}
    compact_type = meta.get("compact_group_type")
    compact_value = meta.get("compact_group_value")
    if compact_type and compact_value:
        return ("compact", compact_type, compact_value)
    invoice_number = meta.get("invoice_number")
    if invoice_number:
        return ("invoice", invoice_number)
    return ("source", meta.get("source", ""), meta.get("chunk_index", -1))


class OllamaToolAdapter(BaseToolAdapter):
    """Adapter fuer Ollama — prompt-basiertes TOOL_CALL:-Parsing.

    Baut bei jedem step() einen vollstaendigen Prompt mit System-Prompt,
    Tool-Beschreibungen, bisherigen Ergebnissen und der Frage. Parst die
    LLM-Antwort per Regex nach TOOL_CALL: Mustern.

    Im Gegensatz zu den Cloud-Adaptern (Anthropic/OpenAI), die native
    Tool-Use-APIs nutzen, verwendet Ollama ein prompt-basiertes Verfahren:
    Tool-Definitionen werden als Text in den Prompt eingebettet, und die
    LLM-Antwort wird per Regex nach ``TOOL_CALL: name(params)`` geparst.

    Zustandsverwaltung:
        _previous_results: Akkumuliert Tool-Ergebnisse fuer den naechsten Prompt
        _search_chunks: Deduplizierte Retrieval-Chunks fuer den Final-Prompt
        _sent_prompts: Audit-Trail aller gesendeten Prompts (step + final)
    """

    @property
    def result_truncate_limit(self) -> int:
        """Ollama-spezifisches Truncation-Limit (4000 statt 24000)."""
        return AGENT_RESULT_TRUNCATE_OLLAMA

    def __init__(self, backend: dict, config: dict) -> None:
        import httpx
        import ollama as ollama_client

        self._client = ollama_client.Client(
            host=backend["url"],
            timeout=httpx.Timeout(
                connect=OLLAMA_CONNECT_TIMEOUT_SECS,
                read=OLLAMA_READ_TIMEOUT_SECS,
                write=OLLAMA_WRITE_TIMEOUT_SECS,
                pool=OLLAMA_POOL_TIMEOUT_SECS,
            ),
        )
        self._model = backend["model"]
        self._config = config

        agent_cfg = config.get("chat", {}).get("agent", {})
        self._num_ctx = agent_cfg.get("num_ctx", DEFAULT_AGENT_NUM_CTX)
        self._temperature = agent_cfg.get("temperature", DEFAULT_LLM_TEMPERATURE)
        self._max_context_chunks = config.get("retrieval", {}).get(
            "max_context_chunks", DEFAULT_MAX_CONTEXT_CHUNKS
        )

        self._question: str = ""
        self._system_prompt: str = ""
        self._tools_prompt: str = ""
        self._history_block: str = ""
        self._previous_results: list[dict] = []
        self._search_chunks: list[dict] = []
        self._last_usage: dict = {}
        self._final_prompt_override: str = ""
        self._sent_prompts: list[dict] = []

    def init_conversation(
        self,
        question: str,
        system_prompt: str,
        tools: list[ToolDefinition],
        history_block: str = "",
    ) -> None:
        """Initialisiert eine neue Agent-Konversation mit Ollama Prompt-Template."""
        self._question = question
        self._system_prompt = system_prompt
        self._tools_prompt = self._build_tools_prompt(tools)
        self._history_block = history_block
        self._previous_results = []
        self._search_chunks = []
        self._last_usage = {}
        self._sent_prompts = []

    def step(self) -> ToolStep:
        """Fuehrt einen Agent-Schritt aus: generiert Prompt, parst Tool-Call via Regex."""
        prompt = self._build_agent_prompt()
        self._sent_prompts.append({"type": "step", "prompt": prompt})
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                options={"temperature": self._temperature, "num_ctx": self._num_ctx},
                stream=False,
                think=False,  # Thinking deaktivieren — Tool-Auswahl braucht kein CoT
            )
        except Exception as exc:
            raise ToolAdapterError(str(exc), retryable=True) from exc

        llm_text = response.get("response", "")
        clean_text = _clean_think_blocks(llm_text)

        tool_call = self._parse_tool_call(clean_text)

        if tool_call:
            return ToolStep(tool_call=tool_call, text=clean_text)
        return ToolStep(tool_call=None, text=clean_text, is_final=False)

    def add_tool_result(
        self,
        tool_name: str,
        result_text: str,
        tool_use_id: str | None = None,
        raw_chunks: list[dict] | None = None,
    ) -> None:
        """Speichert ein Tool-Ergebnis fuer den naechsten Prompt-Bau."""
        limit = self.result_truncate_limit
        display = result_text[:limit]
        if len(result_text) > limit:
            display += "..."
        self._previous_results.append(
            {
                "tool": tool_name,
                "result": display,
            }
        )
        # Strukturierte Retrieval-Chunks dedupliziert akkumulieren
        if raw_chunks:
            seen = {_chunk_dedupe_key(c) for c in self._search_chunks}
            added = 0
            for chunk in raw_chunks:
                key = _chunk_dedupe_key(chunk)
                if key not in seen:
                    self._search_chunks.append(chunk)
                    seen.add(key)
                    added += 1
            if added:
                logger.debug("search_chunks: +%d (gesamt: %d)", added, len(self._search_chunks))

    def deanonymize_params(self, params: dict) -> dict:
        """Gibt Parameter unveraendert zurueck (Ollama laeuft lokal, keine Anonymisierung)."""
        return {**params}

    def stream_final(self, additional_context: str = "", think: bool = False) -> Iterator[str]:
        """Streamt die finale Antwort nach dem Tool-Loop (Ollama generate)."""
        prompt = self._build_final_prompt(additional_context)
        self._sent_prompts.append({"type": "final", "prompt": prompt})
        logger.info(
            "stream_final: %d search_chunks, %d previous_results, prompt=%d chars, think=%s",
            len(self._search_chunks),
            len(self._previous_results),
            len(prompt),
            think,
        )
        self._last_usage = {}

        if think:
            # User will Thinking sehen → Tags durchstreamen (wie Standard/Unchained)
            yield from self._generate_raw(prompt, think=True)
        else:
            # Default: kein Thinking, kein Filtern noetig
            yield from self._generate_raw(prompt, think=False)

    def _generate_raw(self, prompt: str, think: bool) -> Iterator[str]:
        """Streamt eine Ollama-Antwort und gibt alle Chunks direkt weiter.

        Bei think=True werden <think>-Bloecke durchgestreamt (Frontend
        rendert sie als aufklappbare Reasoning-Bloecke).
        Bei think=False erzeugt Ollama keine <think>-Bloecke.

        Bei Timeout mit think=True wird ein Retry mit think=False durchgefuehrt.
        """
        import httpx as _httpx

        try:
            stream = self._client.generate(
                model=self._model,
                prompt=prompt,
                options={"temperature": self._temperature, "num_ctx": self._num_ctx},
                stream=True,
                think=think,
            )
        except (_httpx.TimeoutException, Exception) as exc:
            if think:
                logger.warning(
                    "stream_final think=True fehlgeschlagen (%s), Retry mit think=False", exc
                )
                yield from self._generate_raw(prompt, think=False)
                return
            raise ToolAdapterError(str(exc), retryable=True) from exc

        any_yielded = False
        in_think = False
        try:
            for chunk in stream:
                if chunk.get("thinking"):
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield chunk["thinking"]
                    any_yielded = True
                text = chunk.get("response", "")
                if text:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield text
                    any_yielded = True
                if chunk.get("done"):
                    if in_think:
                        yield "</think>"
                        in_think = False
                    self._last_usage = build_usage_from_ollama(chunk, self._num_ctx, self._model)
        except (_httpx.TimeoutException, Exception) as exc:
            if think and not any_yielded:
                logger.warning(
                    "stream_final Timeout waehrend think-Phase (%s), Retry mit think=False", exc
                )
                yield from self._generate_raw(prompt, think=False)
                return
            if not any_yielded:
                raise ToolAdapterError(str(exc), retryable=True) from exc
            logger.error("stream_final Fehler nach partiellem Streaming: %s", exc)
            yield "\n\n[Fehler: Antwort unvollständig]"

    @property
    def last_conversation(self) -> str:
        """Serialisiert alle gesendeten Prompts als JSON (fuer Debugging/Audit)."""
        import json

        return json.dumps(self._sent_prompts, ensure_ascii=False)

    @property
    def last_usage(self) -> dict:
        """Gibt Ollama-Usage-Statistiken des letzten Calls zurueck."""
        return self._last_usage

    # -- Interne Hilfsmethoden --

    @staticmethod
    def _build_tools_prompt(tools: list[ToolDefinition]) -> str:
        """Generiert den Tool-Beschreibungs-Prompt fuer Ollama."""
        if not tools:
            return ""
        _TYPE_LABELS = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }
        lines = ["Verfuegbare Werkzeuge:"]
        for idx, td in enumerate(tools, 1):
            props = td.parameters.get("properties", {})
            required = td.parameters.get("required", [])
            parts = []
            for name, prop in props.items():
                raw_type = prop.get("type")
                if isinstance(raw_type, list):
                    t = "|".join(_TYPE_LABELS.get(item, str(item)) for item in raw_type)
                elif isinstance(raw_type, str) and raw_type:
                    t = _TYPE_LABELS.get(raw_type, raw_type)
                else:
                    t = "any"
                parts.append(f"{name}: {t}" if name in required else f"{name}: {t} = optional")
            lines.append(f"{idx}. {td.name}({', '.join(parts)}) – {td.description}")
        lines.append("")
        lines.append("Um ein Werkzeug aufzurufen, antworte mit:")
        lines.append('TOOL_CALL: tool_name(param1="wert1", param2="wert2")')
        return "\n".join(lines)

    def _build_agent_prompt(self) -> str:
        from kiara.shared.date_utils import get_german_date_string

        parts = [
            get_german_date_string(),
            "",
            self._system_prompt.replace("{tools_prompt}", self._tools_prompt),
        ]
        if self._history_block:
            parts.append(f"\n{self._history_block}")
        if self._previous_results:
            parts.append("\nBisherige Werkzeug-Ergebnisse:")
            for r in self._previous_results:
                if "error" in r:
                    parts.append(f"- {r['tool']}: FEHLER: {r['error']}")
                else:
                    result_text = r.get("result", "")
                    parts.append(f"- {r['tool']}: {result_text}")
        parts.append(f"\nFrage: {self._question}")
        if self._previous_results:
            parts.append(
                "\nPruefe: Kannst du die Frage mit den bisherigen Ergebnissen "
                "VOLLSTAENDIG beantworten? Falls ja, antworte direkt. "
                "Falls nicht oder falls Ergebnisse fehlen, nutze ein "
                "weiteres passendes Werkzeug."
            )
        return "\n".join(parts)

    def _build_final_prompt(self, additional_context: str = "") -> str:
        from kiara.shared.date_utils import get_german_date_string

        parts = [get_german_date_string(), ""]

        # Wenn strukturierte Retrieval-Chunks vorliegen: Standard-RAG-Format
        if self._search_chunks:
            from kiara.generation.chain import format_context

            # Final-Rerank: Alle gesammelten Chunks gegen die Original-Frage
            # reranken, damit irrelevante Ergebnisse aus nachfolgenden Suchen
            # den Kontext nicht verwässern.
            sorted_chunks = self._final_rerank(self._search_chunks)
            instruction = self._final_prompt_override or ""
            parts.append(instruction)
            if self._history_block:
                parts.append(f"\n{self._history_block}")
            # Nicht-Such-Ergebnisse (calculate, list_datasources)
            non_search = [r for r in self._previous_results if r["tool"] != "search_documents"]
            if non_search:
                parts.append("\nWeitere Werkzeug-Ergebnisse:")
                for r in non_search:
                    if "error" in r:
                        parts.append(f"- {r['tool']}: FEHLER: {r['error']}")
                    else:
                        parts.append(f"- {r['tool']}: {r.get('result', '')}")
            parts.append(f"\nKontext:\n{format_context(sorted_chunks)}")
            if additional_context:
                parts.append(f"\n{additional_context}")
            parts.append(f"\nFrage: {self._question}")
        else:
            # Fallback: kein strukturierter Kontext (z.B. nur calculate-Aufrufe)
            parts.append("Du bist KIara, ein hilfreicher KI-Agent.")
            if self._history_block:
                parts.append(f"\n{self._history_block}")
            parts.append(f"\nFrage: {self._question}")
            if self._previous_results:
                parts.append("\nGesammelte Informationen:")
                for r in self._previous_results:
                    if "error" in r:
                        parts.append(f"- {r['tool']}: FEHLER: {r['error']}")
                    else:
                        parts.append(f"- {r['tool']}: {r.get('result', '')}")
            if additional_context:
                parts.append(f"\n{additional_context}")
            parts.append(
                "\nBeantworte die Frage basierend auf den gesammelten "
                "Informationen. Antworte praezise und auf Deutsch."
            )
        return "\n".join(parts)

    # Mindestanzahl Chunks nach Score-Filter (Sicherheitsnetz)
    _FINAL_RERANK_MIN_KEEP = 5
    # Cross-Encoder-Score unter dem Chunks als irrelevant gelten
    # ms-marco-MiniLM: >0 = wahrscheinlich relevant, <0 = wahrscheinlich nicht
    _FINAL_RERANK_SCORE_FLOOR = 0.0

    def _final_rerank(self, chunks: list[dict]) -> list[dict]:
        """Rerankt alle gesammelten Chunks gegen die Original-Frage.

        Bei Multi-Search-Szenarien akkumuliert der Agent Chunks aus mehreren
        Suchen. Spaetere Suchen koennen vage Queries haben und irrelevante
        Ergebnisse liefern, die den Kontext verwässern. Sibling-Expansion
        verstaerkt das Problem zusaetzlich.

        Strategie:
        1. Alle Chunks mit dem Cross-Encoder gegen die Original-Frage scoren
        2. Chunks unter dem Score-Floor (0.0) als irrelevant entfernen
        3. Mindestens _FINAL_RERANK_MIN_KEEP Chunks behalten
        4. Auf max_context_chunks begrenzen
        5. Nach Quelle + Chunk-Index sortieren fuer kohaerente Darstellung
        """
        cap = self._max_context_chunks
        final_reranking = self._config.get("chat", {}).get("agent", {}).get("final_reranking", True)

        if not final_reranking:
            if len(chunks) > cap:
                logger.warning(
                    "Agent Chunk-Cap (Final-Reranking deaktiviert): %d -> %d Chunks",
                    len(chunks),
                    cap,
                )
                chunks = chunks[:cap]
            return sorted(
                chunks,
                key=lambda c: (
                    c.get("metadata", {}).get("source", ""),
                    c.get("metadata", {}).get("chunk_index", 0),
                ),
            )

        from kiara.retrieval.reranker import CrossEncoderReranker

        # Kompaktierte Rechnungs-Chunks repraesentieren bereits je einen Beleg.
        # Sie duerfen bei Aggregatsfragen nicht erneut durch Cross-Encoder-
        # Score-Flooring ausgeduennt werden; sonst fehlen einzelne Rechnungen
        # wieder im finalen Prompt. Nicht-Rechnungsgruppen bleiben Kandidaten
        # fuer den normalen Final-Rerank und koennen als Rauschen herausfallen.
        protected_chunks = [
            c
            for c in chunks
            if (
                c.get("metadata", {}).get("source_diverse_compact")
                and c.get("metadata", {}).get("compact_group_type") == "invoice"
            )
        ]
        if protected_chunks:
            protected_ids = {id(c) for c in protected_chunks}
            candidate_chunks = [c for c in chunks if id(c) not in protected_ids]
            candidate_cap = max(0, cap - len(protected_chunks))
        else:
            candidate_chunks = chunks
            candidate_cap = cap

        reranker = CrossEncoderReranker(self._config)
        total = len(candidate_chunks)
        if candidate_cap <= 0:
            chunks = []
        elif reranker.enabled and total > self._FINAL_RERANK_MIN_KEEP:
            # Alle Chunks scoren (top_k=total), dann selbst filtern
            scored = reranker.rerank(self._question, candidate_chunks, top_k=total)

            # Score-basiertes Filtern: irrelevante Chunks entfernen
            floor = self._FINAL_RERANK_SCORE_FLOOR
            min_keep = 0 if protected_chunks else min(self._FINAL_RERANK_MIN_KEEP, total)
            relevant = [c for c in scored if c.get("rerank_score", 0) >= floor]
            # Sicherheitsnetz: mindestens min_keep Chunks behalten
            if len(relevant) < min_keep:
                relevant = scored[:min_keep]
            # Dann auf Cap begrenzen
            if len(relevant) > candidate_cap:
                relevant = relevant[:candidate_cap]

            dropped = total - len(relevant)
            if dropped > 0:
                worst = scored[-1].get("rerank_score", 0) if scored else 0
                best = scored[0].get("rerank_score", 0) if scored else 0
                logger.info(
                    "Final-Rerank: %d -> %d Kandidaten (%d entfernt, "
                    "score_range=%.1f..%.1f, floor=%.1f)",
                    total,
                    len(relevant),
                    dropped,
                    worst,
                    best,
                    floor,
                )
            chunks = relevant
        elif total > candidate_cap:
            logger.warning(
                "Agent Chunk-Cap (kein Reranker): %d -> %d Chunks",
                total,
                candidate_cap,
            )
            chunks = candidate_chunks[:candidate_cap]
        else:
            chunks = candidate_chunks

        if protected_chunks:
            if len(protected_chunks) > cap:
                logger.warning(
                    "Final-Rerank: %d geschuetzte Rechnungs-Chunks auf Cap %d begrenzt",
                    len(protected_chunks),
                    cap,
                )
            chunks = protected_chunks[:cap] + chunks

        # Nach Quelle + Chunk-Index sortieren fuer kohaerente Darstellung
        return sorted(
            chunks[:cap],
            key=lambda c: (
                c.get("metadata", {}).get("source", ""),
                c.get("metadata", {}).get("chunk_index", 0),
            ),
        )

    @staticmethod
    def _parse_tool_call(llm_response: str) -> tuple[str, dict] | None:
        """Parst TOOL_CALL aus der LLM-Antwort (Klammerbalancierung).

        Nutzt einen Zeichenzaehler statt Regex fuer den Klammer-Inhalt,
        damit Klammern in Parameterwerten (z.B. sqrt(144), Server (Rack 5))
        korrekt behandelt werden.
        """
        # Dotted Namen zulassen (z.B. system.events.list), aber auf
        # [A-Za-z_][\w.]* beschraenken — keine fuehrenden/abschliessenden Punkte.
        header = re.search(r"(?i)TOOL_CALL\s*:\s*([A-Za-z_][\w.]*)\s*\(", llm_response)
        if not header:
            return None
        tool_name = header.group(1)
        # Undotted Namen (Standard-Tools) werden weiterhin lowercase normalisiert;
        # dotted AdminFunction-Namen bleiben case-sensitive, damit sie 1:1 in der
        # Admin-Registry gefunden werden.
        if "." not in tool_name:
            tool_name = tool_name.lower()
        if not tool_name:
            return None

        params_str = _extract_balanced_params(llm_response, header.end())
        if params_str is None:
            return None

        params: dict[str, str | int | float | bool | list | dict] = {}
        for token in _split_top_level_param_tokens(params_str):
            parsed = _parse_param_token(token)
            if parsed is None:
                continue
            key, value = parsed
            if key not in params:
                params[key] = value
        return (tool_name, params)


def _split_top_level_param_tokens(params_str: str) -> list[str]:
    """Split name=value tokens without looking inside quotes or brackets."""
    starts: list[int] = []
    quote: str | None = None
    escape = False
    depth = 0
    i = 0
    while i < len(params_str):
        ch = params_str[i]
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch in "([{":
            depth += 1
            i += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            i += 1
            continue

        if depth == 0 and (i == 0 or params_str[i - 1].isspace() or params_str[i - 1] == ","):
            m = re.match(r"[A-Za-z_]\w*", params_str[i:])
            if m:
                end = i + m.end()
                j = end
                while j < len(params_str) and params_str[j].isspace():
                    j += 1
                if j < len(params_str) and params_str[j] == "=":
                    starts.append(i)
                    i = j + 1
                    continue
        i += 1

    tokens: list[str] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(params_str)
        token = params_str[start:end].strip().rstrip(",").strip()
        if token:
            tokens.append(token)
    return tokens


def _parse_param_token(token: str) -> tuple[str, str | int | float | bool | list | dict] | None:
    m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", token, re.DOTALL)
    if not m:
        return None
    key, raw = m.group(1), m.group(2).strip()
    if not raw:
        return None

    if raw[0] in ("'", '"'):
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
        return (key, value) if isinstance(value, str) else None

    lower = raw.lower()
    if lower == "true":
        return key, True
    if lower == "false":
        return key, False
    if re.fullmatch(r"-?\d+", raw):
        return key, int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return key, float(raw)
    if raw[0] in "[{":
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
        if isinstance(value, (list, dict)):
            return key, value
    return None
