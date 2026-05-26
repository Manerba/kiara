"""Retriever with metadata filtering support."""

from dataclasses import dataclass

from kiara.retrieval.vectorstore import VectorStore


def _normalize_file_extension(ext: str | None) -> str | None:
    """Stellt sicher dass Dateiendungen mit Punkt beginnen (.pdf, .docx)."""
    if not ext:
        return ext
    return (ext if ext.startswith(".") else f".{ext}").lower()


def _escape_es_wildcard(value: str) -> str:
    """Escaped ES-Wildcard-Sonderzeichen (`*`, `?`, `\\`) fuer wildcard-Queries."""
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def _normalize_datasource_ids(raw) -> set[int]:
    """Normalisiert ds_ids-Metadaten fuer Zugriffskontrollen."""
    if raw is None:
        return set()
    values = raw if isinstance(raw, (list, tuple, set)) else [raw]
    normalized = set()
    for value in values:
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _metadata_matches_datasource_ids(
    metadata: dict,
    filter_datasource_ids: list[int] | None,
) -> bool:
    if filter_datasource_ids is None:
        return True
    allowed = _normalize_datasource_ids(filter_datasource_ids)
    if not allowed:
        return False
    present = _normalize_datasource_ids(metadata.get("ds_ids"))
    return bool(present & allowed)


@dataclass(frozen=True)
class _FilterParams:
    """Interne Filterkriterien fuer Retrieval-Methoden."""

    datasource_ids: list[int] | None = None
    source: str | None = None
    file_type: str | None = None
    directory: str | None = None

    def normalize(self) -> "_FilterParams":
        """Gibt eine Kopie mit normalisierter Dateiendung zurueck."""
        return _FilterParams(
            datasource_ids=self.datasource_ids,
            source=self.source,
            file_type=_normalize_file_extension(self.file_type),
            directory=self.directory,
        )


def _apply_metadata_filter(
    results: list[dict],
    filters: "_FilterParams",
) -> list[dict]:
    """Post-hoc Dateityp-, Verzeichnis- und Quellpfad-Filter auf Ergebnislisten.

    Wird fuer Retrieval-Pfade genutzt, die keine native Metadaten-
    Filterung unterstuetzen (z.B. Graph-Retrieval aus PostgreSQL).
    """
    if not filters.file_type and not filters.directory and not filters.source:
        return results
    type_lower = filters.file_type.lower() if filters.file_type else None
    dir_lower = filters.directory.lower() if filters.directory else None
    source_lower = filters.source.lower() if filters.source else None

    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if type_lower and meta.get("file_type", "").lower() != type_lower:
            continue
        if dir_lower and dir_lower not in meta.get("directory", "").lower():
            continue
        if source_lower and source_lower not in meta.get("source", "").lower():
            continue
        filtered.append(r)
    return filtered


class Retriever:
    """Retrieves relevant chunks from the vector store."""

    def __init__(self, config: dict, embedding_model: str | None = None):
        self._config = config
        self._store = VectorStore(config, embedding_model=embedding_model)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dir: str | None = None,
        filter_type: str | None = None,
        filter_datasource_ids: list[int] | None = None,
        filter_source: str | None = None,
    ) -> list[dict]:
        """Retrieve relevant document chunks.

        Args:
            query: The search query.
            k: Number of results to return.
            filter_dir: Filter by directory path (substring match).
            filter_type: Filter by file extension (e.g. '.ps1').
            filter_datasource_ids: If set, only return chunks from these datasource IDs.
            filter_source: Filter by source path (substring match).

        Returns:
            List of result dicts with 'text', 'metadata', 'score'.
        """
        if filter_datasource_ids is not None and len(filter_datasource_ids) == 0:
            return []
        where = self._build_filter(
            filter_dir, filter_type, filter_datasource_ids, filter_source=filter_source
        )
        return self._store.query(query_text=query, k=k, where=where)

    def load_chunks_by_source(
        self,
        source: str,
        filter_datasource_ids: list[int] | None = None,
    ) -> list[dict]:
        """Laedt alle Chunks einer Quelle fuer Parent-/Sibling-Kontext.

        Bei gesetztem Datasource-Filter werden Chunks ohne passende ds_ids
        nicht zurueckgegeben. Fehlende DS-Zuordnung ist fail-closed.
        """
        if filter_datasource_ids is not None and len(filter_datasource_ids) == 0:
            return []
        data = self._store.get_documents_by_source(source)
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        results: list[dict] = []
        for i, text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
            if not _metadata_matches_datasource_ids(metadata, filter_datasource_ids):
                continue
            result = {
                "text": text,
                "metadata": metadata,
                "score": 0.0,
            }
            if i < len(ids):
                result["_doc_id"] = ids[i]
            results.append(result)
        return results

    def _build_filter(
        self,
        filter_dir: str | None,
        filter_type: str | None,
        filter_datasource_ids: list[int] | None = None,
        filter_source: str | None = None,
    ) -> dict | None:
        """Baut einen backend-agnostischen Where-Filter aus den Suchparametern.

        filter_datasource_ids:
            None  -> kein DS-Filter (Admin, sieht alles)
            [1,2] -> ds_ids $in [1,2] (MatchAny Pre-Filter in Qdrant)
            []    -> wird VOR diesem Aufruf als Frueh-Abbruch behandelt
        """
        conditions = []

        if filter_source:
            conditions.append({"source": {"$contains": filter_source}})

        if filter_dir:
            conditions.append({"directory": {"$contains": filter_dir}})

        if filter_type:
            filter_type = _normalize_file_extension(filter_type)
            conditions.append({"file_type": filter_type})

        if filter_datasource_ids:
            conditions.append({"ds_ids": {"$in": filter_datasource_ids}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def get_stats(self) -> dict:
        """Return vector store statistics."""
        return self._store.get_stats()

    def get_directories(self) -> list[str]:
        """Return available directories."""
        return self._store.get_directories()

    def get_file_types(self) -> list[str]:
        """Return available file types."""
        return self._store.get_file_types()


# ---------------------------------------------------------------------------
# BM25 Import (optional dependency)
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore[assignment,misc]
    _HAS_BM25 = False

import logging
import re
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

from kiara.shared.constants import (
    BM25_CACHE_TTL_SECS,
    BM25_MAX_CORPUS_DOCS,
    DEFAULT_ELASTICSEARCH_URL,
    KEYWORD_OVERFETCH_FACTOR,
)
from kiara.shared.constants import (
    STOP_WORDS as _STOP_WORDS,
)
from kiara.shared.tracing import get_tracer, mark_current_span_error

_tracer = get_tracer(__name__, pipeline="retrieval")

_RETRIEVAL_OVERFETCH_FACTOR = 2
"""Overfetch-Multiplikator fuer Semantic/ColBERT/Graph-Retrieval.

Niedriger als KEYWORD_OVERFETCH_FACTOR (=4) weil Vektor-Suche
praezsiser ist und weniger Overfetch braucht.
"""


class HybridRetriever(Retriever):
    """Kombiniert semantische Suche (VectorDB) mit Keyword-Suche (BM25).

    Reciprocal Rank Fusion (RRF):
    score = sum(1 / (k + rank_i)) fuer alle Ranklisten

    Der BM25-Index wird class-level gecacht, damit er nicht bei jeder
    Request neu gebaut werden muss (Chain wird pro Request instanziiert).
    """

    _bm25_cache: dict = {}  # Class-level: collection_name -> (BM25Okapi, doc_ids, doc_texts, doc_metadatas, created_at)
    _bm25_building: bool = False
    _bm25_lock = threading.Lock()  # Thread-Safety fuer Cache + Building-Flag
    _bm25_condition = threading.Condition(_bm25_lock)
    _bm25_generations: dict = {}  # Class-level: collection_name -> Build-Generation (bumped pro invalidate)
    _es_fail_until: float = 0.0  # Class-level: Cooldown nach ES-Verbindungsfehler
    _es_lock = threading.Lock()  # Thread-Safety fuer ES-Cooldown + Client-Init

    def __init__(self, config: dict, embedding_model: str | None = None):
        from kiara.backend_selector import normalize_model_name
        from kiara.config import get_embedding_model
        from kiara.ingestion._file_utils import ChunkDatasources

        raw_model = (
            embedding_model
            if embedding_model and embedding_model.strip()
            else get_embedding_model(config)
        )
        norm_model = normalize_model_name(raw_model)
        if not norm_model:
            raise ValueError("Kein gueltiges Embedding-Modell fuer HybridRetriever")
        super().__init__(config, norm_model)

        self._chunk_ds = ChunkDatasources()
        self._embed_model_name = norm_model

        # ColBERT-Defaults (vor _setup_colbert, fuer getattr-Safety)
        self._colbert_enabled = False
        self._colbert_model = ""
        self._colbert_store = None

        retrieval_cfg = config.get("retrieval", {})
        self._setup_hybrid_mode(retrieval_cfg)
        self._setup_graph(config)
        self._setup_colbert(config)

    def _setup_hybrid_mode(self, retrieval_cfg: dict) -> None:
        """Initialisiert den Hybrid-Modus (BM25 oder Elasticsearch).

        Setzt self._hybrid_mode, self._bm25_enabled, self._es_enabled und
        alle zugehoerigen Konfigurationswerte (RRF-K, TTL, ES-URL/Index).
        Loggt Warnungen bei fehlenden Dependencies.

        Args:
            retrieval_cfg: Der 'retrieval'-Abschnitt der Konfiguration.
        """
        # Abwaertskompatibilitaet: True -> "bm25", False -> "none"
        hybrid_val = retrieval_cfg.get("hybrid_search", "none")
        if hybrid_val is True:
            hybrid_val = "bm25"
        elif hybrid_val is False or not hybrid_val:
            hybrid_val = "none"
        self._hybrid_mode = hybrid_val  # "none", "bm25", "elasticsearch"

        self._bm25_enabled = self._hybrid_mode == "bm25" and _HAS_BM25
        self._es_enabled = self._hybrid_mode == "elasticsearch"

        from kiara.shared.constants import RRF_K_DEFAULT

        self._rrf_k = retrieval_cfg.get("rrf_k", RRF_K_DEFAULT)
        self._bm25_ttl = retrieval_cfg.get("bm25_cache_ttl", BM25_CACHE_TTL_SECS)
        self._bm25_max_corpus = retrieval_cfg.get(
            "bm25_max_corpus_docs",
            BM25_MAX_CORPUS_DOCS,
        )

        # Elasticsearch-Client (lazy)
        self._es_client = None
        self._es_index = retrieval_cfg.get("elasticsearch_index", "kiara")
        self._es_url = retrieval_cfg.get("elasticsearch_url", DEFAULT_ELASTICSEARCH_URL)
        es_fuzz = retrieval_cfg.get("elasticsearch_fuzziness", "none")
        self._es_fuzziness = None if es_fuzz in ("none", None, "") else es_fuzz

        if self._hybrid_mode == "bm25" and not _HAS_BM25:
            logger.warning(
                "rank_bm25 nicht installiert – BM25 Hybrid Search deaktiviert. "
                "Installation: pip install rank-bm25"
            )
        elif self._es_enabled:
            logger.info("Hybrid-Suche: Elasticsearch (%s, Index: %s)", self._es_url, self._es_index)

    def _setup_graph(self, config: dict) -> None:
        """Initialisiert GraphRAG falls in der Konfiguration aktiviert.

        Setzt self._graph_enabled und self._graph_retriever. Bei Fehlern
        wird GraphRAG deaktiviert und eine Warnung geloggt.

        Args:
            config: Die vollstaendige Konfiguration.
        """
        self._graph_enabled = config.get("graph", {}).get("enabled", False)
        self._graph_retriever = None
        if self._graph_enabled:
            try:
                from kiara.graph.retriever import GraphRetriever
                from kiara.graph.store import GraphStore

                graph_store = GraphStore()
                self._graph_retriever = GraphRetriever(config, graph_store, self._embed_model_name)
                logger.info("GraphRAG aktiviert")
            except Exception as exc:
                logger.warning(
                    "GraphRAG konnte nicht initialisiert werden: %s",
                    exc,
                    exc_info=True,
                )

    def _setup_colbert(self, config: dict) -> None:
        """Initialisiert ColBERT-Retrieval falls konfiguriert.

        No-throw: bei Fehler Warning + self._colbert_enabled = False.
        """
        self._colbert_enabled = False
        self._colbert_model = ""
        self._colbert_store = None

        enabled = bool(config.get("retrieval", {}).get("colbert", {}).get("enabled", False))
        if not enabled:
            return

        try:
            from kiara.backend_selector import normalize_model_name
            from kiara.retrieval.colbert import ColBERTStore
            from kiara.shared.constants import COLBERT_DEFAULT_MODEL

            model = normalize_model_name(
                config.get("retrieval", {}).get("colbert", {}).get("model", COLBERT_DEFAULT_MODEL)
            )
            if not model:
                logger.warning("ColBERT: kein Modellname konfiguriert")
                return
            self._colbert_model = model
            self._colbert_store = ColBERTStore(config, model)
            self._colbert_enabled = True
            logger.info("ColBERT-Retrieval aktiviert (Modell: %s)", model)
        except Exception as exc:
            logger.warning("ColBERT konnte nicht initialisiert werden: %s", exc, exc_info=True)
            self._colbert_enabled = False

    def _colbert_retrieve(
        self,
        query: str,
        k: int,
        where: dict | None,
    ) -> list[dict]:
        """ColBERT Multi-Vector Retrieval via Qdrant MaxSim.

        Args:
            query: Suchanfrage.
            k: Basis-Anzahl; intern k*2 fuer RRF-Overfetch.
            where: Ergebnis von _build_filter().

        Returns:
            Ergebnisliste im Retrieval-Shape fuer RRF-Fusion.
            Leere Liste wenn ColBERT deaktiviert.
        """
        if not getattr(self, "_colbert_enabled", False) or not getattr(
            self, "_colbert_store", None
        ):
            return []
        try:
            from kiara.retrieval.colbert import embed_colbert_query

            query_tokens = embed_colbert_query(
                self._config,
                self._colbert_model,
                query,
            )
            if not query_tokens:
                return []
            return self._colbert_store.search(
                query_tokens, k=k * _RETRIEVAL_OVERFETCH_FACTOR, where=where
            )
        except Exception as exc:
            mark_current_span_error(exc)
            logger.warning("ColBERT-Retrieval fehlgeschlagen: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # BM25 Index Management
    # ------------------------------------------------------------------

    def build_bm25_index(self) -> None:
        """Baut BM25-Index aus VectorStore-Dokumenten auf.

        Wird automatisch beim ersten retrieve()-Aufruf als Background-Thread
        gestartet. Waehrend der Build laeuft, fallen Queries auf reine
        Vektorsuche zurueck (kein BM25). Kann nach Ingestion manuell
        aufgerufen werden.
        """
        if not self._try_start_bm25_build():
            return
        self._build_bm25_index_started()

    def _build_bm25_index_started(self) -> None:
        """Fuehrt den BM25-Build aus, nachdem _bm25_building bereits gesetzt ist."""
        try:
            coll_name_pre = self._store.get_collection_name()
            with HybridRetriever._bm25_lock:
                # setdefault: Eintrag muss existieren, damit invalidate_bm25_cache(None)
                # diese Collection findet und die Generation hochzaehlen kann.
                HybridRetriever._bm25_generations.setdefault(coll_name_pre, 0)
                captured_generation = HybridRetriever._bm25_generations[coll_name_pre]

            docs, corpus, coll_name = self._load_bm25_documents()
            if not corpus:
                with HybridRetriever._bm25_lock:
                    HybridRetriever._bm25_cache.pop(coll_name, None)
                return
            self._build_and_cache_bm25(docs, corpus, coll_name, captured_generation)
        finally:
            with HybridRetriever._bm25_condition:
                HybridRetriever._bm25_building = False
                HybridRetriever._bm25_condition.notify_all()

    def _start_bm25_rebuild(self, coll_name: str) -> None:
        """Startet einen asynchronen BM25-Rebuild und laesst stale Cache nutzbar."""
        if not _HAS_BM25:
            return
        with HybridRetriever._bm25_condition:
            if HybridRetriever._bm25_building:
                return
            HybridRetriever._bm25_building = True
        thread = threading.Thread(
            target=self._build_bm25_index_started,
            name=f"kiara-bm25-rebuild-{coll_name}",
            daemon=True,
        )
        thread.start()

    def _try_start_bm25_build(self) -> bool:
        """Prueft ob ein BM25-Build gestartet werden kann.

        Setzt _bm25_building-Flag unter Lock. Gibt False zurueck
        wenn BM25 nicht verfuegbar oder bereits ein Build laeuft.
        """
        if not _HAS_BM25:
            return False
        with HybridRetriever._bm25_lock:
            if HybridRetriever._bm25_building:
                return False
            HybridRetriever._bm25_building = True
            return True

    def _load_bm25_documents(self) -> tuple[dict, list[str], str]:
        """Laedt Dokumente aus dem VectorStore fuer den BM25-Index.

        Returns:
            Tupel (all_docs, doc_texts, coll_name). doc_texts ist leer
            bei Fehler oder leerer Collection.
        """
        coll_name = self._store.get_collection_name()
        try:
            total = self._store.get_stats().get("total_chunks", 0)
            # Memory-Schutz: BM25Okapi haelt Korpus + Tokens + Metadaten
            # vollstaendig im Prozess-Speicher. Bei Ueberschreiten der
            # Schwelle fail-closed skip statt Worker-OOM riskieren.
            max_corpus = self._bm25_max_corpus
            if max_corpus and total > max_corpus:
                logger.warning(
                    "BM25-Index fuer Collection '%s' nicht gebaut: "
                    "Korpus-Umfang %d uebersteigt bm25_max_corpus_docs=%d. "
                    "Hybrid-Suche faellt auf reine Vektor-/Graph-Suche zurueck. "
                    "Fuer grosse Collections retrieval.hybrid_search auf "
                    "'elasticsearch' umstellen.",
                    coll_name,
                    total,
                    max_corpus,
                )
                return {}, [], coll_name
            logger.info(
                "BM25-Index fuer Collection '%s' wird aufgebaut (%d Chunks) ...",
                coll_name,
                total,
            )
            all_docs = self._store.get_all_documents(
                include=["documents", "metadatas"],
            )
            if not isinstance(all_docs, dict):
                logger.warning(
                    "BM25-Index fuer Collection '%s' nicht gebaut: "
                    "VectorStore lieferte kein Dokument-Dict",
                    coll_name,
                )
                return {}, [], coll_name
            doc_texts: list = all_docs.get("documents", [])
            if not isinstance(doc_texts, list):
                logger.warning(
                    "BM25-Index fuer Collection '%s' nicht gebaut: documents ist keine Liste",
                    coll_name,
                )
                return {}, [], coll_name
            if not doc_texts:
                logger.info("Keine Dokumente in Collection '%s' – BM25-Index leer", coll_name)
            return all_docs, doc_texts, coll_name
        except Exception as exc:
            logger.error("BM25-Index konnte nicht gebaut werden: %s", exc, exc_info=True)
            return {}, [], coll_name

    def _build_and_cache_bm25(
        self,
        all_docs: dict,
        doc_texts: list[str],
        coll_name: str,
        captured_generation: int,
    ) -> None:
        """Tokenisiert Dokumente, baut BM25-Index und cached ihn.

        Args:
            all_docs: Ergebnis von get_all_documents().
            doc_texts: Nicht-leere Dokumenten-Texte.
            coll_name: Collection-Name fuer den Cache-Key.
            captured_generation: Build-Generation zum Zeitpunkt des Snapshots.
                Beim Writeback wird gegen die aktuelle Generation geprueft —
                bei Mismatch wird der Snapshot verworfen, weil zwischenzeitlich
                eine invalidate_bm25_cache passiert ist.
        """
        doc_ids: list = all_docs["ids"]
        doc_metadatas: list = all_docs.get("metadatas", []) or [{} for _ in doc_ids]
        tokenized_all = [self._tokenize(text) for text in doc_texts]
        # BM25Okapi wirft ZeroDivisionError bei avgdl=0 — Dokumente ohne Tokens
        # (nur Stoppwoerter/Ein-Zeichen-Tokens) parallel ausfiltern.
        keep_idx = [i for i, toks in enumerate(tokenized_all) if toks]
        if not keep_idx:
            with HybridRetriever._bm25_lock:
                HybridRetriever._bm25_cache.pop(coll_name, None)
            logger.warning(
                "BM25-Index fuer Collection '%s' nicht gebaut: keine Dokumente "
                "mit verwertbaren Tokens (alle nur Stoppwoerter/Kurz-Tokens)",
                coll_name,
            )
            return
        tokenized = [tokenized_all[i] for i in keep_idx]
        doc_ids = [doc_ids[i] for i in keep_idx]
        doc_texts = [doc_texts[i] for i in keep_idx]
        doc_metadatas = [doc_metadatas[i] if i < len(doc_metadatas) else {} for i in keep_idx]
        bm25 = BM25Okapi(tokenized)
        with HybridRetriever._bm25_lock:
            current_generation = HybridRetriever._bm25_generations.get(coll_name, 0)
            if current_generation != captured_generation:
                logger.info(
                    "BM25-Build fuer Collection '%s' verworfen: zwischenzeitlich "
                    "invalidiert (generation %d != captured %d)",
                    coll_name,
                    current_generation,
                    captured_generation,
                )
                return
            HybridRetriever._bm25_cache[coll_name] = (
                bm25,
                doc_ids,
                doc_texts,
                doc_metadatas,
                time.time(),
            )
        logger.info(
            "BM25-Index fuer Collection '%s' gebaut: %d Dokumente",
            coll_name,
            len(doc_ids),
        )

    @classmethod
    def invalidate_bm25_cache(cls, coll_name: str | None = None) -> None:
        """Invalidiert den BM25-Cache (z.B. nach Ingestion).

        Erhoeht die Build-Generation der betroffenen Collection(s), damit
        ein laufender Build seinen stale Snapshot beim Writeback nicht
        zurueckpublishen kann.

        Args:
            coll_name: Nur diese Collection invalidieren. Wenn None, alles.
        """
        with cls._bm25_lock:
            if coll_name:
                cls._bm25_cache.pop(coll_name, None)
                cls._bm25_generations[coll_name] = cls._bm25_generations.get(coll_name, 0) + 1
            else:
                # "Alle" muss alle Collections treffen, fuer die ein Build
                # laufen koennte — d.h. alles in _bm25_generations (ein
                # Build hat dort einen Eintrag via setdefault angelegt) und
                # zusaetzlich alle aktuell gecachten.
                affected = set(cls._bm25_generations.keys()) | set(cls._bm25_cache.keys())
                cls._bm25_cache.clear()
                for key in affected:
                    cls._bm25_generations[key] = cls._bm25_generations.get(key, 0) + 1
        logger.info("BM25-Cache invalidiert: %s", coll_name or "alle")

    # ------------------------------------------------------------------
    # Startup-Warmup
    # ------------------------------------------------------------------

    @classmethod
    def warmup_bm25_index(cls, config: dict) -> None:
        """BM25-Index beim Service-Startup aufbauen (blockierend).

        Wird in der FastAPI-Lifespan aufgerufen, BEVOR der Server
        Requests annimmt. So gibt es keine GIL-Konflikte mit laufenden
        Requests und der Index steht sofort bereit.
        """
        hybrid_val = config.get("retrieval", {}).get("hybrid_search", "none")
        # Abwaertskompatibilitaet
        if hybrid_val is True:
            hybrid_val = "bm25"
        elif hybrid_val is False or not hybrid_val:
            hybrid_val = "none"
        if hybrid_val != "bm25" or not _HAS_BM25:
            return
        try:
            retriever = cls(config)
            retriever.build_bm25_index()
        except Exception as exc:
            logger.error("BM25-Warmup fehlgeschlagen: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_dir: str | None = None,
        filter_type: str | None = None,
        filter_datasource_ids: list[int] | None = None,
        filter_source: str | None = None,
    ) -> list[dict]:
        """Hybrid-Retrieval mit ds_ids Pre-Filter (Qdrant MatchAny).

        Orchestriert bis zu vier Retrieval-Pfade und fusioniert per RRF:
        1. Semantisch: VectorDB mit ds_ids Pre-Filter (O(1) via Payload-Index)
        2. Keyword: BM25/ES mit Punkt-Lookup (chunk_ds.filter_allowed)
        3. Graph: GraphRetriever + Punkt-Lookup in HybridRetriever
        4. ColBERT: Multi-Vector MaxSim (Qdrant, colbert_<modell>)
        5. RRF-Fusion der Ranglisten
        """
        with _tracer.start_as_current_span("hybrid_retrieve") as span:
            span.set_attribute("query", query[:200])
            span.set_attribute("k", k)
            span.set_attribute("hybrid_mode", self._hybrid_mode)
            span.set_attribute("graph_enabled", self._graph_enabled)
            span.set_attribute("colbert_enabled", getattr(self, "_colbert_enabled", False))

            if filter_datasource_ids is not None and len(filter_datasource_ids) == 0:
                span.set_attribute("total_results", 0)
                return []

            fp = _FilterParams(
                datasource_ids=filter_datasource_ids,
                source=filter_source,
                file_type=filter_type,
                directory=filter_dir,
            ).normalize()

            self._check_bm25_ttl()
            coll_name = self._store.get_collection_name()
            use_bm25 = self._ensure_bm25_cache(coll_name) if self._bm25_enabled else False
            use_es = self._es_enabled
            use_graph = getattr(self, "_graph_enabled", False) and getattr(
                self, "_graph_retriever", None
            )
            use_colbert = getattr(self, "_colbert_enabled", False) and getattr(
                self, "_colbert_store", None
            )
            use_rrf = use_bm25 or use_es or use_graph or use_colbert

            where = self._build_filter(
                fp.directory,
                fp.file_type,
                fp.datasource_ids,
                fp.source,
            )

            if not use_rrf:
                results = self._store.query(query_text=query, k=k, where=where)
                span.set_attribute("total_results", len(results))
                return results

            with _tracer.start_as_current_span("semantic_retrieve") as sem:
                semantic_results = self._semantic_retrieve(query, k, where)
                sem.set_attribute("result_count", len(semantic_results))

            with _tracer.start_as_current_span("keyword_retrieve") as kw:
                keyword_results = self._keyword_retrieve(
                    query,
                    k,
                    use_bm25,
                    use_es,
                    filters=fp,
                )
                kw.set_attribute("result_count", len(keyword_results))
                kw.set_attribute("mode", self._hybrid_mode)

            with _tracer.start_as_current_span("graph_retrieve") as gr:
                graph_results = self._graph_retrieve(query, k, filters=fp)
                gr.set_attribute("result_count", len(graph_results))
                gr.set_attribute("enabled", self._graph_enabled)

            with _tracer.start_as_current_span("colbert_retrieve") as cb:
                colbert_results = self._colbert_retrieve(query, k, where)
                cb.set_attribute("result_count", len(colbert_results))
                cb.set_attribute("model", getattr(self, "_colbert_model", ""))

            with _tracer.start_as_current_span("rrf_fusion") as rrf:
                lists = [semantic_results, keyword_results, graph_results, colbert_results]
                rrf.set_attribute("source_count", sum(1 for l in lists if l))
                fused = self._rrf_fusion_multi(lists, k)
                rrf.set_attribute("result_count", len(fused))

            span.set_attribute("total_results", len(fused))
            return fused

    def load_chunks_by_source(
        self,
        source: str,
        filter_datasource_ids: list[int] | None = None,
    ) -> list[dict]:
        """Laedt Quell-Chunks und erzwingt DS-Rechte per PostgreSQL-Lookup."""
        if filter_datasource_ids is not None and len(filter_datasource_ids) == 0:
            return []
        results = super().load_chunks_by_source(
            source,
            filter_datasource_ids=None,
        )
        if filter_datasource_ids is None:
            return results
        return self._apply_ds_filter(results, filter_datasource_ids)

    def _check_bm25_ttl(self) -> None:
        """Prueft BM25-Cache-TTL und erneuert veraltete Eintraege asynchron.

        Veraltete Eintraege bleiben waehrend des Rebuilds nutzbar
        (stale-while-revalidate), damit Requests nicht synchron auf den
        kompletten Indexaufbau warten.
        """
        coll_name = self._store.get_collection_name()
        with HybridRetriever._bm25_lock:
            cached = HybridRetriever._bm25_cache.get(coll_name)
            needs_rebuild = bool(cached and time.time() - cached[4] > self._bm25_ttl)
        if needs_rebuild:
            self._start_bm25_rebuild(coll_name)

    def _ensure_bm25_cache(self, coll_name: str) -> bool:
        """Baut den BM25-Index synchron nur beim echten Cache-Miss nach.

        Wartet bei laufendem Parallel-Build auf dessen Ende und prueft den
        Cache erneut, statt Hybrid-Retrieval still auf reine Semantiksuche
        zurueckfallen zu lassen.
        """
        with HybridRetriever._bm25_condition:
            if coll_name in HybridRetriever._bm25_cache:
                return True
            if HybridRetriever._bm25_building:
                HybridRetriever._bm25_condition.wait_for(
                    lambda: not HybridRetriever._bm25_building,
                    timeout=30.0,
                )
                return coll_name in HybridRetriever._bm25_cache

        self.build_bm25_index()

        with HybridRetriever._bm25_condition:
            return coll_name in HybridRetriever._bm25_cache

    def _semantic_retrieve(
        self,
        query: str,
        k: int,
        where: dict | None,
    ) -> list[dict]:
        """Semantische Vektorsuche mit Pre-Filter.

        Holt k*2 Kandidaten fuer RRF-Fusion (Overfetch, damit nach
        Fusion genuegend Ergebnisse uebrig bleiben).

        Args:
            query: Suchanfrage.
            k: Basis-Anzahl (wird intern verdoppelt fuer RRF-Overfetch).
            where: Metadaten-Filter inkl. ds_ids Pre-Filter.
        """
        try:
            return self._store.query(
                query_text=query, k=k * _RETRIEVAL_OVERFETCH_FACTOR, where=where
            )
        except Exception as exc:
            mark_current_span_error(exc)
            logger.warning("Semantische Suche fehlgeschlagen: %s", exc, exc_info=True)
            return []

    def _keyword_retrieve(
        self,
        query: str,
        k: int,
        use_bm25: bool,
        use_es: bool,
        filters: _FilterParams,
    ) -> list[dict]:
        """Keyword-Suche via BM25 oder Elasticsearch.

        Delegiert an _bm25_search() oder _es_search() je nach aktivem
        Hybrid-Modus. DS-Filterung erfolgt per Punkt-Lookup in den
        jeweiligen Methoden.
        """
        if use_bm25:
            return self._bm25_search(
                query,
                k=k,
                filters=filters,
            )
        if use_es:
            return self._es_search(
                query,
                k=k,
                filters=filters,
            )
        return []

    def _graph_retrieve(
        self,
        query: str,
        k: int,
        filters: _FilterParams,
    ) -> list[dict]:
        """Graph-Retrieval mit DS-Filter per Punkt-Lookup.

        Graph-Ergebnisse kommen aus PostgreSQL und enthalten keine
        DS-Zuordnung — der Filter wird nachtraeglich per
        _apply_ds_filter() angewendet. Dateityp- und Verzeichnis-
        Filter werden ebenfalls post-hoc auf Metadaten geprueft.
        """
        if not self._graph_enabled or not self._graph_retriever:
            return []
        try:
            # Maximalen Pool des GraphRetrievers ausschoepfen, damit
            # post-hoc DS-/Type-/Directory-Filter nicht deterministisch
            # valide Treffer vor der RRF-Fusion verlieren. _max_chunks
            # ist der interne Hard-Cap im GraphRetriever; ein hoeheres k
            # bringt nichts, ein niedrigeres beschneidet den Pool.
            graph_k = max(k * _RETRIEVAL_OVERFETCH_FACTOR, self._graph_retriever._max_chunks)
            results = self._graph_retriever.retrieve(query, k=graph_k)
            results = self._apply_ds_filter(results, filters.datasource_ids)
            return _apply_metadata_filter(results, filters)
        except Exception as exc:
            mark_current_span_error(exc)
            logger.warning("Graph-Retrieval fehlgeschlagen: %s", exc, exc_info=True)
            return []

    def _bm25_search(
        self,
        query: str,
        k: int = 10,
        filters: _FilterParams | None = None,
    ) -> list[dict]:
        """BM25 Keyword-Suche ueber den aufgebauten Index.

        DS-Filterung per Punkt-Lookup (batched, eine SQL-Query fuer
        alle Kandidaten). Nutzt numpy argpartition fuer O(n) Top-K
        statt O(n log n) Vollsortierung.
        """
        if filters is None:
            filters = _FilterParams()
        cached = self._get_bm25_cached()
        if cached is None:
            return []

        bm25, doc_ids, doc_texts, doc_metadatas, _ts = cached
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores_arr = np.asarray(bm25.get_scores(query_tokens))
        eligible_indices = self._bm25_eligible_indices(
            scores_arr,
            doc_ids,
            doc_metadatas,
            filters,
        )
        top_indices = self._bm25_top_k_indices(
            scores_arr,
            k,
            candidate_indices=eligible_indices,
        )
        if top_indices is None:
            return []

        candidates = HybridRetriever._bm25_collect_candidates(
            top_indices,
            scores_arr,
            doc_metadatas,
            filters,
        )
        return self._bm25_build_results(candidates[:k], doc_texts, doc_ids)

    def _get_bm25_cached(self) -> tuple | None:
        """Gibt den gecachten BM25-Index oder None zurueck."""
        coll_name = self._store.get_collection_name()
        return HybridRetriever._bm25_cache.get(coll_name)

    def _bm25_eligible_indices(
        self,
        scores_arr: np.ndarray,
        doc_ids: list,
        doc_metadatas: list,
        filters: "_FilterParams",
    ) -> np.ndarray:
        """Wendet BM25-Filter vor der Top-K-Auswahl an."""
        source_filter_lower = filters.source.lower() if filters.source else None
        type_lower = filters.file_type.lower() if filters.file_type else None
        dir_lower = filters.directory.lower() if filters.directory else None
        positive_indices = np.flatnonzero(scores_arr > 0)
        if len(doc_ids) < len(scores_arr):
            positive_indices = positive_indices[positive_indices < len(doc_ids)]

        if not source_filter_lower and not type_lower and not dir_lower:
            eligible = [int(idx) for idx in positive_indices]
        else:
            eligible = []
            for idx in positive_indices:
                metadata = doc_metadatas[idx] if idx < len(doc_metadatas) else {}
                if source_filter_lower:
                    src = metadata.get("source", "")
                    if source_filter_lower not in src.lower():
                        continue
                if type_lower:
                    ft = metadata.get("file_type", "")
                    if ft.lower() != type_lower:
                        continue
                if dir_lower:
                    d = metadata.get("directory", "")
                    if dir_lower not in d.lower():
                        continue
                eligible.append(int(idx))

        if filters.datasource_ids is not None and eligible:
            candidate_ids = [doc_ids[idx] for idx in eligible]
            allowed = self._chunk_ds.filter_allowed(
                candidate_ids,
                filters.datasource_ids,
                self._embed_model_name,
            )
            eligible = [idx for idx in eligible if doc_ids[idx] in allowed]

        return np.asarray(eligible, dtype=int)

    @staticmethod
    def _bm25_build_results(
        candidates: list[tuple[int, float, dict]],
        doc_texts: list[str],
        doc_ids: list,
    ) -> list[dict]:
        """Baut Ergebnis-Dicts aus BM25-Kandidaten."""
        return [
            {
                "text": doc_texts[idx],
                "metadata": metadata,
                "score": score,
                "_doc_id": doc_ids[idx],
            }
            for idx, score, metadata in candidates
        ]

    def _bm25_top_k_indices(
        self,
        scores_arr: np.ndarray,
        k: int,
        candidate_indices: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Berechnet Top-K Indizes aus BM25-Scores per argpartition.

        Nutzt numpy argpartition fuer O(n) statt O(n log n)
        Vollsortierung. Gibt None zurueck wenn keine Kandidaten.

        Args:
            scores_arr: BM25-Scores als numpy-Array.
            k: Gewuenschte finale Ergebnis-Anzahl. Der Helper bildet daraus
               ein Kandidatenfenster bis min(k * KEYWORD_OVERFETCH_FACTOR,
               len(candidate_indices)).
            candidate_indices: Optional bereits vorgefilterte Kandidaten.

        Returns:
            Sortierte Indizes der overfetchten Kandidaten oder None bei
            leerem Score-Array. Die finale Ergebnisbegrenzung auf k bleibt
            in _bm25_search() via candidates[:k].
        """
        if k <= 0:
            return None
        if candidate_indices is None:
            candidate_indices = np.arange(len(scores_arr))
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=int)

        fetch_k = min(k * KEYWORD_OVERFETCH_FACTOR, len(candidate_indices))
        if fetch_k == 0:
            return None

        if fetch_k >= len(candidate_indices):
            order = np.argsort(-scores_arr[candidate_indices])
            return candidate_indices[order]

        candidate_scores = scores_arr[candidate_indices]
        top_local = np.argpartition(-candidate_scores, fetch_k)[:fetch_k]
        top_indices = candidate_indices[top_local]
        return top_indices[np.argsort(-scores_arr[top_indices])]

    @staticmethod
    def _bm25_collect_candidates(
        top_indices: np.ndarray,
        scores_arr: np.ndarray,
        doc_metadatas: list,
        filters: "_FilterParams",
    ) -> list[tuple[int, float, dict]]:
        """Sammelt BM25-Kandidaten nach Score-, Source-, Typ- und Dir-Filter.

        Filtert Ergebnisse mit Score <= 0 und wendet optional
        case-insensitive Filter auf Source, Dateityp und Verzeichnis an.
        """
        source_filter_lower = filters.source.lower() if filters.source else None
        type_lower = filters.file_type.lower() if filters.file_type else None
        dir_lower = filters.directory.lower() if filters.directory else None

        candidates = []
        for idx in top_indices:
            score = scores_arr[idx]
            if score <= 0:
                break
            metadata = doc_metadatas[idx] if idx < len(doc_metadatas) else {}
            if source_filter_lower:
                src = metadata.get("source", "")
                if source_filter_lower not in src.lower():
                    continue
            if type_lower:
                ft = metadata.get("file_type", "")
                if ft.lower() != type_lower:
                    continue
            if dir_lower:
                d = metadata.get("directory", "")
                if dir_lower not in d.lower():
                    continue
            candidates.append((idx, float(score), metadata))

        return candidates

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion_multi(
        self,
        result_lists: list[list[dict]],
        k: int,
    ) -> list[dict]:
        """RRF-Fusion ueber N Ergebnislisten (verallgemeinert _rrf_fusion).

        Formel: rrf_score = sum(1/(rrf_k + rank_i)) fuer alle Listen i
        """
        if k <= 0:
            logger.warning("RRF-Fusion: k=%d ist ungueltig, gebe leere Liste zurueck", k)
            return []

        scores = {}  # doc_key -> {"score": float, "doc": dict}

        for results in result_lists:
            if not results:
                continue
            # enumerate() ist 0-basiert, RRF braucht 1-basierte Raenge → rank + 1
            for rank, doc in enumerate(results):
                key = doc.get("_doc_id")
                if not key:
                    # Fail-closed: heterogene Keys (z.B. _doc_id-Hash vs.
                    # source+text-Fallback) wuerden denselben Chunk in zwei
                    # Eintraegen abbilden und Duplikate in Top-K erzeugen.
                    logger.warning(
                        "RRF-Fusion: Retriever-Ergebnis ohne _doc_id verworfen (source=%s)",
                        doc.get("metadata", {}).get("source", "<unknown>"),
                    )
                    continue
                rrf_contribution = 1.0 / (self._rrf_k + rank + 1)

                if key in scores:
                    scores[key]["score"] += rrf_contribution
                else:
                    scores[key] = {"score": rrf_contribution, "doc": doc}

        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        output = []
        for item in sorted_results[:k]:
            result = item["doc"].copy()
            result["rrf_score"] = round(item["score"], 6)
            result.pop("_doc_id", None)
            output.append(result)

        return output

    # ------------------------------------------------------------------
    # Elasticsearch
    # ------------------------------------------------------------------

    def _get_es_client(self):
        """Lazy-Init des Elasticsearch-Clients mit Cooldown bei Fehlern."""
        if self._es_client is None:
            if time.time() < HybridRetriever._es_fail_until:
                return None
            with HybridRetriever._es_lock:
                if self._es_client is not None:
                    return self._es_client
                if time.time() < HybridRetriever._es_fail_until:
                    return None
                try:
                    from kiara.retrieval.elasticsearch_service import _get_client

                    client = _get_client(self._config)
                    client.info()
                    self._es_client = client
                    HybridRetriever._es_fail_until = 0.0
                    logger.info("Elasticsearch verbunden: %s", self._es_url)
                except Exception as exc:
                    HybridRetriever._es_fail_until = time.time() + 30
                    logger.warning(
                        "Elasticsearch-Verbindung fehlgeschlagen (%s), Retry in 30s: %s",
                        self._es_url,
                        exc,
                        exc_info=True,
                    )
                    self._es_client = None
        return self._es_client

    def _es_search(
        self,
        query: str,
        k: int = 10,
        filters: _FilterParams | None = None,
    ) -> list[dict]:
        """Keyword-Suche ueber Elasticsearch mit Punkt-Lookup DS-Filter."""
        if filters is None:
            filters = _FilterParams()
        es = self._get_es_client()
        if es is None:
            return []
        try:
            body = self._build_es_query_body(query, k, filters)
            resp = es.search(index=self._es_index, body=body)
            raw_results = self._parse_es_hits(resp)
            raw_results = self._apply_ds_filter(raw_results, filters.datasource_ids)
            return raw_results[:k]
        except Exception as exc:
            logger.warning("Elasticsearch-Suche fehlgeschlagen: %s", exc, exc_info=True)
            return []

    def _build_es_query_body(
        self,
        query: str,
        k: int,
        filters: _FilterParams,
    ) -> dict:
        """Baut den Elasticsearch-Query-Body mit Filtern."""
        must = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "metadata.source^2"],
                    **({} if not self._es_fuzziness else {"fuzziness": self._es_fuzziness}),
                }
            }
        ]
        es_filter = []
        if filters.source:
            safe_source = _escape_es_wildcard(filters.source)
            es_filter.append({"wildcard": {"metadata.source.raw": f"*{safe_source}*"}})
        if filters.file_type:
            ft = _normalize_file_extension(filters.file_type)
            es_filter.append({"term": {"metadata.file_type": ft}})
        if filters.directory:
            safe_dir = _escape_es_wildcard(filters.directory)
            es_filter.append({"wildcard": {"metadata.directory": f"*{safe_dir}*"}})
        if filters.datasource_ids is not None:
            es_filter.append({"terms": {"metadata.ds_ids": filters.datasource_ids}})
        return {
            "size": k * KEYWORD_OVERFETCH_FACTOR,
            "query": {"bool": {"must": must, "filter": es_filter}},
            "_source": ["text", "metadata"],
        }

    @staticmethod
    def _parse_es_hits(resp: dict) -> list[dict]:
        """Parst Elasticsearch-Hits in das einheitliche Ergebnis-Format."""
        results = []
        for hit in resp.get("hits", {}).get("hits", []) or []:
            source = hit.get("_source", {}) or {}
            results.append(
                {
                    "text": source.get("text", ""),
                    "metadata": source.get("metadata", {}),
                    "score": float(hit.get("_score") or 0.0),
                    "_doc_id": hit.get("_id", ""),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Shared Filters
    # ------------------------------------------------------------------

    def _apply_ds_filter(
        self,
        results: list[dict],
        filter_datasource_ids: list[int] | None,
    ) -> list[dict]:
        """Wendet DS-Zugriffskontrolle per Punkt-Lookup an.

        Shared-Logik fuer _es_search(), _graph_retrieve() und aehnliche
        Retrieval-Pfade, die keine native DS-Filterung unterstuetzen.
        Nutzt chunk_ds.filter_allowed() fuer batched SQL-Lookup.

        Thread-Safety: Kein Lock noetig — chunk_ds nutzt eigene
        DB-Connections.

        Args:
            results: Ergebnisse mit '_doc_id'-Feld.
            filter_datasource_ids: Erlaubte DS-IDs (None = kein Filter).

        Returns:
            Gefilterte Ergebnisliste (unveraendert wenn kein Filter).
        """
        if filter_datasource_ids is None or not results:
            return results

        results_with_id = [r for r in results if r.get("_doc_id")]
        dropped = len(results) - len(results_with_id)
        if dropped:
            logger.warning(
                "DS-Filter: %d Retriever-Ergebnis(se) ohne _doc_id verworfen",
                dropped,
            )
        if not results_with_id:
            return []
        candidate_ids = [r["_doc_id"] for r in results_with_id]
        allowed = self._chunk_ds.filter_allowed(
            candidate_ids,
            filter_datasource_ids,
            self._embed_model_name,
        )
        return [r for r in results_with_id if r["_doc_id"] in allowed]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_key(result: dict) -> str:
        """Erzeuge einen eindeutigen Schluessel fuer ein Ergebnis.

        Nutzt source + erstes Stueck Text als Fallback-Identifikation.
        """
        source = result.get("metadata", {}).get("source", "")
        text_start = result.get("text", "")[:80]
        return f"{source}::{text_start}"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Einfache Tokenisierung: Lowercase, Split, Stoppwoerter entfernen."""
        # Lowercase + nur Wortzeichen behalten
        tokens = re.findall(r"\w+", text.lower())
        # Stoppwoerter und sehr kurze Token entfernen
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]
