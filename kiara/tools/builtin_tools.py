"""Builtin-Tools fuer den Agent-Modus.

Diese Tools benoetigen keine besonderen Permissions und stehen
jedem User im Agent-Modus zur Verfuegung:
- SearchDocumentsTool: Dokumentensuche ueber den Retriever (mit Filtern)
- ListDatasourcesTool: Listet verfuegbare Datenquellen
- CalculateTool: Sichere arithmetische Berechnungen
"""

import logging
import math
import re
import threading
from typing import TYPE_CHECKING

from kiara.exceptions import RetrievalError
from kiara.generation.context_selection import document_key
from kiara.retrieval.reranker import RerankBackendError
from kiara.shared.constants import PARENT_DOC_MAX_CHUNKS
from kiara.shared.processing_warnings import report_processing_warning
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult

if TYPE_CHECKING:
    from kiara.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def _enrich_candidates_from_store(
    selected: list[dict],
    candidates: list[dict],
    retriever: "HybridRetriever",
    filter_ds_ids: list[int] | None = None,
    max_document_chunks: int = PARENT_DOC_MAX_CHUNKS,
) -> list[dict]:
    """Reichert die Kandidatenliste mit vollstaendigen Quelldokumenten an.

    Fuer jede Quelle in ``selected`` mit chunk_total <= PARENT_DOC_MAX_CHUNKS
    werden ALLE Chunks dieser Quelle aus der VectorStore-DB nachgeladen.
    So findet die anschliessende Sibling-Expansion auch Chunks, die nicht
    unter den urspruenglichen fetch_k Retriever-Ergebnissen waren.

    Dies implementiert das "Parent Document Retrieval"-Pattern: Wenn der
    Reranker einen beliebigen Chunk eines kleinen Dokuments als relevant
    einstuft, wird das gesamte Dokument verfuegbar gemacht.

    ``filter_ds_ids`` schraenkt nachgeladene Chunks auf die erlaubten
    Datasource-IDs ein (ACL-Pruefung).
    """
    # Quellen identifizieren, die nachgeladen werden muessen
    sources_to_load: dict[str, int] = {}  # source -> chunk_total
    for r in selected:
        src = r.get("metadata", {}).get("source", "")
        ct = r.get("metadata", {}).get("chunk_total", 1)
        if src and ct <= max_document_chunks and src not in sources_to_load:
            sources_to_load[src] = ct

    if not sources_to_load:
        return candidates

    # Bestehende Kandidaten-Keys sammeln
    existing_keys: set[tuple] = {
        (document_key(c), c.get("metadata", {}).get("chunk_index", -1)) for c in candidates
    }

    enriched = list(candidates)
    for src, ct in sources_to_load.items():
        try:
            loaded = retriever.load_chunks_by_source(src, filter_ds_ids)
            added = 0
            for chunk in loaded:
                key = (document_key(chunk), chunk.get("metadata", {}).get("chunk_index", -1))
                if key not in existing_keys:
                    enriched.append(chunk)
                    existing_keys.add(key)
                    added += 1
            if added:
                logger.info(
                    "Parent-Doc-Retrieval: %d Chunks fuer '%s' nachgeladen (chunk_total=%d)",
                    added,
                    src[-50:],
                    ct,
                )
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, (IndexCompatibilityError, RetrievalError)):
                raise
            logger.warning("Parent-Doc-Retrieval fuer '%s' fehlgeschlagen", src, exc_info=True)
            report_processing_warning("document_expansion")

    return enriched


# Whitelist sicherer Funktionen fuer eval()
_SAFE_MATH_NAMES: dict[str, object] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
}

# AST-basierter Safe-Evaluator (kein eval())
import ast
import operator

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float | int:
    """Wertet einen mathematischen Ausdruck sicher per AST aus.

    Erlaubt nur: Zahlen, arithmetische Operatoren und Funktionen
    aus _SAFE_MATH_NAMES. Kein Zugriff auf Objekte, Attribute oder Builtins.
    """
    if len(expr) > 500:
        raise ValueError("Ausdruck zu lang (max 500 Zeichen)")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Ungueltiger mathematischer Ausdruck") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> float | int:
    """Rekursiver AST-Walker fuer sichere Auswertung."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SAFE_MATH_NAMES:
            val = _SAFE_MATH_NAMES[node.id]
            if isinstance(val, (int, float)):
                return val
            raise ValueError(f"'{node.id}' ist eine Funktion, kein Wert")
        raise ValueError(f"Unbekannter Name: '{node.id}'")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"Unerlaubter Operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)):
            if abs(right) > 1000:
                raise ValueError("Exponent zu gross (max 1000)")
            if (
                isinstance(left, (int, float))
                and abs(left) > 1
                and math.log10(abs(left)) * abs(right) > 1000
            ):
                raise ValueError("Ergebnis zu gross (max 10^1000)")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if not op:
            raise ValueError(f"Unerlaubter Operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Nur einfache Funktionsaufrufe erlaubt")
        func_name = node.func.id
        if func_name not in _SAFE_MATH_NAMES:
            raise ValueError(f"Unbekannte Funktion: '{func_name}'")
        func = _SAFE_MATH_NAMES[func_name]
        if not callable(func):
            raise ValueError(f"'{func_name}' ist nicht aufrufbar")
        args = [_eval_node(a) for a in node.args]
        if func_name == "pow" and len(args) >= 2:
            if isinstance(args[1], (int, float)) and abs(args[1]) > 1000:
                raise ValueError("Exponent zu gross (max 1000)")
            if (
                isinstance(args[0], (int, float))
                and isinstance(args[1], (int, float))
                and abs(args[0]) > 1
                and math.log10(abs(args[0])) * abs(args[1]) > 1000
            ):
                raise ValueError("Ergebnis zu gross (max 10^1000)")
        return func(*args)
    raise ValueError(f"Unerlaubter Ausdruck: {type(node).__name__}")


class SearchDocumentsTool(BaseTool):
    """Durchsucht die Wissensbasis nach relevanten Dokumenten.

    Wrapper um den Retriever mit optionalem Reranking und
    Berechtigungsfiltern auf Datasource-Ebene.
    """

    def __init__(
        self,
        retriever,
        reranker,
        config: dict,
        allowed_datasource_ids: list[int] | None = None,
        default_k: int | None = None,
        default_filter_type: str | None = None,
        default_filter_dir: str | None = None,
        document_references=None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._config = config
        self._allowed_datasource_ids = allowed_datasource_ids
        from kiara.retrieval.source_policy import get_source_profiles

        profile_mode = config.get("_chat_source_policy", {}).get("mode", "agent")
        self._source_profile = get_source_profiles(config)[
            "research" if profile_mode == "research" else "agent"
        ]
        self._default_k = self._source_profile["default_k"] if default_k is None else default_k
        self._default_filter_type = default_filter_type
        self._default_filter_dir = default_filter_dir
        from kiara.tools.document_tools import DocumentReferences

        self._document_references = document_references or DocumentReferences()
        # Deduplizierung ueber mehrere Tool-Calls hinweg:
        # Merkt sich Dokumentdarstellung und Position bereits ausgegebener Chunks.
        self._seen_keys: set[tuple] = set()
        self._seen_lock = threading.Lock()  # Thread-Safety fuer parallele Tool-Calls

    def definition(self) -> ToolDefinition:
        definition = ToolDefinition(
            name="search_documents",
            description=(
                "Durchsucht die Wissensbasis nach relevanten Dokumenten. "
                "WICHTIG: Im Normalfall nur die inhaltliche 'query' setzen. "
                "Die Hybrid-Suche findet Inhalte ohne weitere Filter am besten. "
                "file_type, directory und source_contains sind enge Filter und "
                "duerfen NUR verwendet werden, wenn der Nutzer sie ausdruecklich "
                "nennt oder ein frueheres Suchergebnis den exakten Wert belegt; "
                "niemals Verzeichnisnamen, Dateitypen oder Dateinamensmuster "
                "raten — auch nicht das heutige Datum oder daraus abgeleitete "
                "Jahreszahlen als source_contains. datasource_id ist die "
                "NUMERISCHE ID aus list_datasources, kein Name. "
                "Fehlen Angaben in einer Fundstelle, mit read_document und der "
                "document_id sowie Chunk-Position aus dem Ergebnis gezielt nachlesen."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Der Suchbegriff",
                        "minLength": 1,
                    },
                    "datasource_id": {
                        "type": ["integer", "string"],
                        "description": "Nur in dieser Datenquelle suchen (ID aus list_datasources)",
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Nur diesen Dateityp suchen (z.B. .pdf, .eml, .docx)",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Nur in diesem Pfad suchen (Teilstring)",
                    },
                    "source_contains": {
                        "type": "string",
                        "description": (
                            "Nur Dateien suchen, deren Quellpfad/Dateiname diesen "
                            "Teilstring enthaelt. Enger Filter fuer explizite "
                            "Pfad-/Dateiname-Wuensche. Nicht fuer normale "
                            "Inhaltssuche, IDs, Assetnamen oder aus dem heutigen "
                            "Datum abgeleitete Jahreszahlen verwenden."
                        ),
                    },
                },
                "required": ["query"],
            },
            requires_confirmation=False,
            required_permissions=[],
            category="builtin",
        )
        if self._default_k == 0:
            profile = self._source_profile
            definition.parameters["properties"]["k"] = {
                "type": "integer",
                "description": (
                    f"Anzahl relevanter Textabschnitte pro Suche: {profile['auto_min_k']} "
                    f"bis {profile['auto_max_k']}, Standard {profile['auto_default_k']}. "
                    + profile["auto_instructions"]
                ),
                "minimum": profile["auto_min_k"],
                "maximum": profile["auto_max_k"],
                "default": profile["auto_default_k"],
            }
        else:
            definition.description += (
                f" Die Trefferanzahl ist vom Benutzer auf {self._default_k} festgelegt; "
                "setze keinen k-Parameter."
            )
        return definition

    def validate_params(self, params: dict) -> tuple[bool, str]:
        # Ein trotzdem generiertes k darf eine feste UI-Vorgabe nicht veraendern.
        if self._default_k != 0 and isinstance(params, dict):
            params = {key: value for key, value in params.items() if key != "k"}
        return super().validate_params(params)

    @staticmethod
    def _is_trivial_query(query: str) -> str | None:
        """Prüft ob eine Suchanfrage zu unspezifisch ist.

        Erkennt Queries die nur aus kurzen Zahlen bestehen (1-4 Stellen),
        insbesondere Jahreszahlen. Solche Queries matchen in grossen
        Datenbestaenden fast jedes Dokument und liefern kein brauchbares
        Ranking.

        Returns:
            Fehlermeldung wenn trivial, None wenn OK.
        """
        stripped = query.strip()
        # Reine Zahlen mit 1-4 Stellen (z.B. "2025", "42", "0")
        if re.match(r"^\d{1,4}$", stripped):
            return (
                f"Suchanfrage '{stripped}' ist zu unspezifisch (reine Zahl). "
                "Bitte eine inhaltliche Suchanfrage formulieren, die das "
                "gesuchte Thema beschreibt. Jahreszahlen nur dann ueber "
                "source_contains filtern, wenn der Nutzer explizit nach Dateien "
                "mit dieser Jahreszahl im Pfad oder Dateinamen gefragt hat."
            )
        return None

    @staticmethod
    def _effective_source_filter(params: dict, query: str) -> str | None:
        """Normalisiert den optionalen source_contains-Filter.

        Lokale Modelle leiten bei Agent-Prompts gelegentlich das aktuelle Jahr
        aus dem Datums-Praeambel ab und setzen damit einen zu engen
        source_contains-Filter. Ein nacktes Jahr darf deshalb nur filtern, wenn
        es auch in der eigentlichen Suchquery vorkommt.
        """
        value = params.get("source_contains")
        if value is None:
            return None
        source_filter = str(value).strip()
        if not source_filter:
            return None
        if re.fullmatch(r"20\d{2}", source_filter) and source_filter not in query:
            logger.info(
                "search_documents: source_contains=%r ignoriert, weil die Query "
                "keine entsprechende Jahreszahl enthaelt",
                source_filter,
            )
            return None
        return source_filter

    def _validate_datasource_filter(
        self,
        params: dict,
    ) -> tuple[list[int] | None, ToolResult | None]:
        """Validiert und bereitet den Datasource-Filter auf.

        Returns:
            (filter_ds_ids, error_result). error_result ist None bei Erfolg.
        """
        ds_id = params.get("datasource_id")
        filter_ds_ids = self._allowed_datasource_ids
        if ds_id is not None:
            try:
                if isinstance(ds_id, str):
                    ds_id = int(ds_id)
            except ValueError:
                return None, ToolResult(
                    success=False,
                    data=None,
                    error=f"Ungueltige datasource_id: {ds_id}",
                    display_text="Ungueltige Datenquellen-ID",
                )
            if (
                self._allowed_datasource_ids is not None
                and ds_id not in self._allowed_datasource_ids
            ):
                logger.warning(
                    "search_documents: DS %d nicht in allowed_ids=%s",
                    ds_id,
                    self._allowed_datasource_ids,
                )
                return None, ToolResult(
                    success=False,
                    data=None,
                    error=(
                        f"Datenquelle {ds_id} ist im aktuellen Filter nicht enthalten. "
                        f"Verfuegbare IDs: {self._allowed_datasource_ids}. "
                        "Suche ohne datasource_id, um alle erlaubten Quellen zu durchsuchen."
                    ),
                    display_text="Datenquelle nicht im Filter",
                )
            filter_ds_ids = [ds_id]
        return filter_ds_ids, None

    def _effective_k(self, params: dict) -> int:
        profile = self._source_profile
        if self._default_k != 0:
            k = self._default_k
            minimum, maximum = 1, profile["max_k"]
        else:
            if not profile["auto_enabled"]:
                raise ValueError("Automatische Quellenwahl ist deaktiviert")
            k = params.get("k", profile["auto_default_k"])
            minimum, maximum = profile["auto_min_k"], profile["auto_max_k"]
        if type(k) is not int or not minimum <= k <= maximum:
            raise ValueError(f"k muss eine ganze Zahl zwischen {minimum} und {maximum} sein")
        return k

    def source_diagnostics(self, params: dict) -> dict:
        if not isinstance(params, dict):
            params = {}
        try:
            effective_k = self._effective_k(params)
        except ValueError:
            effective_k = None
        policy = self._config.get("_chat_source_policy", {})
        return {
            "ui_k": policy.get("ui_k", self._default_k),
            "requested_k": self._default_k,
            "selection": "auto" if self._default_k == 0 else "fixed",
            "model_k": params.get("k"),
            "effective_k": effective_k,
            "model_k_ignored": self._default_k != 0 and "k" in params,
        }

    def _retrieve_and_rerank(
        self,
        query: str,
        k: int,
        params: dict,
        filter_ds_ids: list[int] | None,
    ) -> list[dict]:
        """Fuehrt Retrieval mit optionalem Reranking und Sibling-Expansion durch."""
        fetch_k = k
        if self._reranker and self._reranker.enabled:
            multiplier = self._config.get("retrieval", {}).get("rerank_fetch_multiplier", 3)
            fetch_k = k * multiplier

        try:
            results = self._retriever.retrieve(
                query=query,
                k=fetch_k,
                filter_dir=params.get("directory") or self._default_filter_dir,
                filter_type=params.get("file_type") or self._default_filter_type,
                filter_datasource_ids=filter_ds_ids,
                filter_source=self._effective_source_filter(params, query),
            )
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, (IndexCompatibilityError, RetrievalError)):
                raise
            raise RetrievalError(f"Dokumentensuche fehlgeschlagen: {exc}") from exc

        if self._reranker and self._reranker.enabled:
            reranked = self._reranker.rerank(query, results, top_k=k)
            logger.info(
                "search_documents: %d retriever -> %d reranked (top_k=%d)",
                len(results),
                len(reranked),
                k,
            )
            enriched = _enrich_candidates_from_store(
                reranked,
                results,
                self._retriever,
                filter_ds_ids,
                max_document_chunks=self._config.get("retrieval", {}).get(
                    "max_sibling_chunks", self._source_profile["max_sibling_chunks"]
                ),
            )
            from kiara.generation.chain import expand_sibling_chunks

            max_expand = self._config.get("retrieval", {}).get(
                "max_sibling_chunks", self._source_profile["max_sibling_chunks"]
            )
            # Der Agent setzt zusammengehoerige Dokumentteile erst danach
            # zusammen. max_context_chunks begrenzt dort die Quellen; hier
            # wuerde der Cap bereits benoetigte Nachbar-Chunks entfernen.
            results = expand_sibling_chunks(reranked, enriched, max_expand)
            logger.info("search_documents: nach expand_sibling: %d Ergebnisse", len(results))
        else:
            results = results[:k]

        self._retriever.validate_context(results, filter_ds_ids)
        return results

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """Entfernt Chunks, die in vorherigen Tool-Calls bereits zurueckgegeben wurden.

        Thread-safe ueber self._seen_lock fuer parallele Tool-Calls.
        """
        with self._seen_lock:
            if not self._config.get("retrieval", {}).get("agent_dedup", True):
                self._seen_keys.clear()
            before_dedup = len(results)
            results = [
                r
                for r in results
                if (
                    document_key(r),
                    r.get("metadata", {}).get("chunk_index", -1),
                )
                not in self._seen_keys
            ]
            dedup_removed = before_dedup - len(results)
            if dedup_removed:
                logger.info(
                    "search_documents: %d/%d Duplikate entfernt (vorherige Aufrufe)",
                    dedup_removed,
                    before_dedup,
                )

            for r in results:
                key = (
                    document_key(r),
                    r.get("metadata", {}).get("chunk_index", -1),
                )
                self._seen_keys.add(key)

        return results

    @staticmethod
    def _format_search_results(results: list[dict]) -> ToolResult:
        """Formatiert Suchergebnisse als ToolResult mit Quellen-Uebersicht."""
        from kiara.shared.constants import EXCERPT_LENGTH

        sources = []
        text_parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "unbekannt")
            # Score-Prioritaet: rrf_score > rerank_score > score (nicht Truthiness)
            score = r.get("rrf_score")
            if score is None:
                score = r.get("rerank_score")
            if score is None:
                score = r.get("score", 0)
            text = r["text"]
            text_parts.append(f"[{i}] {source} (Relevanz: {score:.3f}):\n{text}")
            sources.append(
                {
                    "source": source,
                    "score": score,
                    "document_id": r["metadata"].get("document_ref"),
                    "excerpt": text[:EXCERPT_LENGTH] + "..."
                    if len(text) > EXCERPT_LENGTH
                    else text,
                }
            )

        return ToolResult(
            success=True,
            data={"results": results, "sources": sources},
            error=None,
            display_text="\n\n".join(text_parts),
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        diagnostics = self.source_diagnostics(params)
        logger.info("search_documents Quellensteuerung: %s", diagnostics)
        result = await self._execute_search(params)
        result.diagnostics["source_selection"] = diagnostics
        return result

    async def _execute_search(self, params: dict) -> ToolResult:
        if error := self._validate_or_error(params):
            return error
        query = params["query"]
        try:
            k = self._effective_k(params)
        except ValueError as exc:
            return ToolResult(False, None, str(exc), "Ungueltige Trefferanzahl")
        logger.info(
            "search_documents: query=%r, k=%d, source_contains=%r",
            query[:80],
            k,
            params.get("source_contains"),
        )

        trivial_msg = self._is_trivial_query(query)
        if trivial_msg:
            logger.warning("search_documents: triviale Query abgelehnt: %r", query)
            return ToolResult(
                success=False,
                data=None,
                error=trivial_msg,
                display_text=f"Suchanfrage zu unspezifisch: {query}",
            )

        filter_ds_ids, error = self._validate_datasource_filter(params)
        if error:
            return error

        try:
            results = self._retrieve_and_rerank(query, k, params, filter_ds_ids)

            # Fallback gegen halluzinierte Filterwerte: Kleine Agent-Modelle
            # erfinden gelegentlich file_type/directory/source_contains (z.B.
            # directory='Finanzen', file_type='pdf'), die formal gueltig sind,
            # aber im Bestand nichts matchen. Statt stiller 0-Treffer wird
            # einmal ohne die opportunistischen Filter gesucht und das dem
            # Modell explizit mitgeteilt, damit es die Filter nicht wiederholt.
            dropped_filters = {
                key: params.get(key)
                for key in ("file_type", "directory", "source_contains")
                if str(params.get(key) or "").strip()
            }
            fallback_note = ""
            if not results and dropped_filters:
                logger.info(
                    "search_documents: 0 Treffer mit Filtern %s — Retry ohne diese Filter",
                    dropped_filters,
                )
                retry_params = {
                    key: value
                    for key, value in params.items()
                    if key not in ("file_type", "directory", "source_contains")
                }
                results = self._retrieve_and_rerank(query, k, retry_params, filter_ds_ids)
                if results:
                    filter_desc = ", ".join(f"{k2}={v!r}" for k2, v in dropped_filters.items())
                    fallback_note = (
                        f"Hinweis: Die Filter ({filter_desc}) ergaben 0 Treffer und wurden "
                        "ignoriert. Diese Filter bei weiteren Suchen weglassen, sofern der "
                        "Nutzer sie nicht ausdruecklich verlangt hat. Ergebnisse ohne Filter:\n\n"
                    )

            results = self._document_references.annotate(results)
            results = self._deduplicate_results(results)

            if not results:
                return ToolResult(
                    success=True,
                    data={"results": [], "sources": []},
                    error=None,
                    display_text="Keine neuen relevanten Dokumente gefunden "
                    "(alle Treffer bereits in vorherigen Suchen).",
                )

            tool_result = self._format_search_results(results)
            if fallback_note:
                tool_result.display_text = fallback_note + (tool_result.display_text or "")
            return tool_result

        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            if isinstance(exc, (RetrievalError, RerankBackendError)):
                raise
            logger.error("SearchDocuments Fehler: %s", exc)
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                display_text=f"Fehler bei der Dokumentensuche: {exc}",
            )


class ListDatasourcesTool(BaseTool):
    """Listet alle verfuegbaren Datenquellen mit ID, Name, Typ und Statistiken.

    Filtert automatisch nach den Berechtigungen des anfragenden Users.
    """

    def __init__(self, allowed_datasource_ids: list[int] | None = None) -> None:
        self._allowed_datasource_ids = allowed_datasource_ids

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_datasources",
            description=(
                "Listet alle verfuegbaren Datenquellen mit ID, Name, Typ "
                "und Anzahl der indexierten Chunks/Dateien."
            ),
            parameters={"type": "object", "properties": {}},
            requires_confirmation=False,
            required_permissions=[],
            category="builtin",
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        try:
            from kiara.repositories.datasource import DatasourceRepository

            ds_repo = DatasourceRepository()
            rows = ds_repo.list_active_with_chunks()

            if self._allowed_datasource_ids is not None:
                rows = [r for r in rows if r["id"] in self._allowed_datasource_ids]

            lines = [
                f"ID {r['id']}: {r['name']} ({r['type']}, "
                f"{r['chunk_count']} Chunks, {r['file_count']} Dateien)"
                for r in rows
            ]
            text = "\n".join(lines) if lines else "Keine Datenquellen verfuegbar."
            return ToolResult(
                success=True,
                data={"datasources": rows},
                error=None,
                display_text=text,
            )
        except Exception as exc:
            logger.error("ListDatasources Fehler: %s", exc)
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                display_text=f"Fehler beim Laden der Datenquellen: {exc}",
            )


class CalculateTool(BaseTool):
    """Fuehrt sichere arithmetische Berechnungen aus.

    Nutzt einen AST-basierten Safe-Evaluator (_safe_eval) ohne eval().
    Keine besonderen Permissions erforderlich.
    """

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculate",
            description="Berechnet einen mathematischen Ausdruck in Python-Syntax. Beispiele: 15 * 4500 / 100 (fuer 15% von 4500), sqrt(144), 2**10, sin(pi/2). KEIN %-Zeichen und keine Woerter verwenden.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Der mathematische Ausdruck",
                        "minLength": 1,
                    },
                },
                "required": ["expression"],
            },
            requires_confirmation=False,
            required_permissions=[],
            category="builtin",
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        if error := self._validate_or_error(params):
            return error
        expression = params["expression"]

        try:
            result = _safe_eval(expression)
            result_str = str(result)
            return ToolResult(
                success=True,
                data={"expression": expression, "result": result},
                error=None,
                display_text=f"{expression} = {result_str}",
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                data=None,
                error=str(exc),
                display_text=f"Berechnungsfehler: {exc}",
            )
