"""File-Level State Machine fuer die Ingestion Pipeline.

Enthaelt die Einzeldatei-Verarbeitungslogik: Loading, Chunking, Embedding,
Circuit-Breaker, Retry und Cancel. Wird von pipeline.py orchestriert.

Extrahiert in Sprint 4 (Pipeline-Split) aus pipeline.py.
"""

import asyncio
import concurrent.futures
import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

from kiara_ingestion_core.models import FileLogEntry, ProcessingContext

from kiara.backend_selector import BackendUnavailableError
from kiara.exceptions import VectorDBConnectionError
from kiara.ingestion._file_utils import (
    ChunkDatasources,
    HashStore,
    _CancelledError,
)
from kiara.ingestion.chunker import chunk_document
from kiara.ingestion.loader import IMAGE_FORMATS, describe_image, load_file
from kiara.retrieval.embedding import embed_texts_late
from kiara.retrieval.vectorstore import VectorStore
from kiara.shared.constants import (
    BACKEND_WAIT_INTERVAL_SECS,
    DEFAULT_QDRANT_URL,
    ENRICHMENT_MAX_CONTEXT_CHARS,
    ENRICHMENT_MIN_CHUNK_CHARS,
    HASH_CHECKPOINT_INTERVAL,
    INGESTION_MAX_CONSECUTIVE_ERRORS,
    LATE_CHUNKING_SUPPORTED_MODELS,
)
from kiara.shared.date_utils import elapsed_ms
from kiara.shared.secret_redact import redact_secrets_in_text
from kiara.shared.tracing import get_tracer

_tracer = get_tracer(__name__, pipeline="ingestion")

# ---------------------------------------------------------------------------
# Thread-State (Job-ID)
# ---------------------------------------------------------------------------

_thread_state = threading.local()


def set_current_job_id(job_id: int | None) -> None:
    """Wird vom Worker-Modul aufgerufen, bevor run_ingestion() startet."""
    _thread_state.job_id = job_id


def _get_current_job_id() -> int | None:
    return getattr(_thread_state, "job_id", None)


# ---------------------------------------------------------------------------
# Netzwerk-/Timeout-Fehler die auf Backend-Probleme hindeuten.
# ---------------------------------------------------------------------------

try:
    from httpx import NetworkError as _HttpxNetwork
    from httpx import TimeoutException as _HttpxTimeout

    _NETWORK_ERRORS: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        _HttpxTimeout,
        _HttpxNetwork,
    )
except ImportError:
    _NETWORK_ERRORS = (ConnectionError, TimeoutError)

_CATEGORICAL_ERRORS: tuple[type[Exception], ...] = (
    BackendUnavailableError,
    VectorDBConnectionError,
    *_NETWORK_ERRORS,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class _CircuitBreakerResult(Enum):
    """Ergebnis der Circuit-Breaker-Pruefung."""

    CONTINUE = "continue"
    RETRY = "retry"
    ABORT = "abort"
    CANCEL = "cancel"
    SKIP = "skip"


class _FileAction(Enum):
    """High-Level-Entscheidung nach einer Verarbeitungsphase."""

    SUCCESS = "success"
    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"
    CANCEL = "cancel"


class _DedupResult(Enum):
    """Ergebnis der Chunk-Deduplication."""

    NO_DEDUP = "no_dedup"
    PARTIAL = "partial"
    FULL = "full"


class _RollbackFailedError(VectorDBConnectionError):
    """Rollback bereits persistierter Embedding-Seiteneffekte ist fehlgeschlagen."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _IngestionContext:
    """Buendelt den gesamten Shared State eines Ingestion-Laufs.

    Wird ausschliesslich intern verwendet. Die oeffentliche API von
    run_ingestion() aendert sich nicht.
    """

    config: dict
    ctx: ProcessingContext
    store: VectorStore
    chunk_ds: ChunkDatasources
    hash_store: HashStore
    batch_size: int
    datasource_id: int | None
    embedding_model: str | None
    hash_mode: str
    ds_overrides: dict | None
    file_hashes: dict[str, tuple[str, int]]
    stats: dict
    pending_hashes: dict[str, tuple[str, int]]
    consecutive_errors: int = 0
    skipped: int = 0
    total: int = 0
    progress_callback: Callable | None = None
    control_callback: Callable | None = None
    file_log_callback: Callable | None = None
    file_start_callback: Callable | None = None
    status_callback: Callable | None = None
    last_full_text: str = ""
    last_original_chunks: list[str] | None = None
    last_original_spans: list[tuple[int, int]] | None = None
    last_original_chunks_by_id: dict[str, str] | None = None
    last_original_spans_by_id: dict[str, tuple[int, int]] | None = None


@dataclass
class _ExistingBatch:
    """Deduplizierte Chunks fuer ColBERT-Backfill."""

    ids: list[str]
    texts: list[str]
    metas: list[dict]


# ---------------------------------------------------------------------------
# DRY-Helper (Pattern A-D)
# ---------------------------------------------------------------------------


def _record_error(
    ictx,
    file_log,
    file_path,
    exc,
    phase,
    *,
    is_colbert=False,
    is_vectordb=False,
    write_error_details=True,
):
    """Error-Recording + Stats + FileLog (Pattern A)."""
    ictx.stats["errors"] += 1
    if is_colbert:
        ictx.stats.setdefault("colbert_errors", 0)
        ictx.stats["colbert_errors"] += 1
    if is_vectordb:
        ictx.stats.setdefault("vectordb_errors", 0)
        ictx.stats["vectordb_errors"] += 1
    if write_error_details:
        ictx.stats.setdefault("error_details", [])
        ictx.stats["error_details"].append(
            {
                "file": Path(file_path).name,
                "error": str(exc)[:200],
                "phase": phase,
            }
        )
    if file_log:
        file_log.status = "error"
        file_log.error_message = str(exc)[:200]
        file_log.error_phase = phase


def _mark_file_processed(ictx, file_path):
    """Hash + Stats + Persist (Pattern B)."""
    ictx.stats["processed_files"] += 1
    ictx.pending_hashes[file_path] = ictx.file_hashes[file_path]
    _persist_hashes_checkpoint(ictx)


def _cleanup_file_hash(ictx, file_path):
    """Hash-Cleanup bei Fehler (Pattern C)."""
    ictx.hash_store.remove_files(ictx.datasource_id, ictx.embedding_model, [file_path])


def _normalize_written_vector_ids(result, fallback_ids: list[str]) -> list[str]:
    """Normalisiert Rueckgaben von _write_embeddings fuer Tests/alte Mocks."""
    if result is None:
        return list(fallback_ids)
    if isinstance(result, (list, tuple)):
        return list(result)
    return list(fallback_ids)


def _rollback_vector_writes(store, chunk_ids: list[str], *, context: str) -> int:
    """Loescht bereits geschriebene VectorDB-Chunks wieder."""
    ids = list(dict.fromkeys(chunk_ids))
    if not ids:
        return 0
    store.delete_by_ids(ids)
    logger.warning("VectorDB-Rollback (%s): %d Chunks geloescht", context, len(ids))
    return len(ids)


def _rollback_elasticsearch_write(ictx, file_path: str, *, context: str) -> int:
    """Loescht moegliche ES-Seiteneffekte fuer die aktuelle Datei."""
    from kiara.retrieval.elasticsearch_service import delete_by_sources, is_configured

    if not is_configured(ictx.config):
        return 0
    deleted = delete_by_sources(
        ictx.config,
        [file_path],
        datasource_id=ictx.datasource_id,
    )
    logger.warning(
        "Elasticsearch-Rollback (%s): %d Dokumente fuer %s geloescht",
        context,
        deleted,
        file_path,
    )
    return deleted


def _rollback_db_associations(ictx, chunk_ids: list[str], *, context: str) -> int:
    """Loescht chunk_datasources/Graph-Seiteneffekte fuer neue Chunks."""
    ids = list(dict.fromkeys(chunk_ids))
    if ictx.datasource_id is None or not ids:
        return 0

    from kiara.shared.db import db_transaction

    with db_transaction() as conn:
        ictx.chunk_ds.acquire_graph_cleanup_lock(conn)
        ictx.chunk_ds.delete_by_chunk_ids(ids, ictx.embedding_model, conn=conn)
        graph_chunk_ids = ictx.chunk_ds.filter_unreferenced_chunk_ids(ids, conn=conn)
        if graph_chunk_ids:
            from kiara.graph.store import GraphStore

            GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)

    logger.warning("DB-Rollback (%s): %d Chunk-Zuordnungen geloescht", context, len(ids))
    return len(ids)


def _rollback_embedding_persistence(
    ictx,
    file_path: str,
    vector_ids: list[str],
    *,
    rollback_elasticsearch: bool,
    rollback_db_associations: bool = False,
    context: str,
) -> bool:
    """Best-Effort-Rollback fuer Embedding-Phase; False bedeutet fail-closed."""
    ok = True
    if rollback_db_associations:
        try:
            deleted = _rollback_db_associations(
                ictx,
                vector_ids,
                context=context,
            )
            ictx.stats.setdefault("db_rollback_chunks", 0)
            ictx.stats["db_rollback_chunks"] += deleted
        except Exception as exc:
            ok = False
            logger.critical(
                "DB-Rollback fehlgeschlagen (%s, %s): %s",
                context,
                file_path,
                exc,
                exc_info=True,
            )
            ictx.stats.setdefault("rollback_errors", 0)
            ictx.stats["rollback_errors"] += 1

    if rollback_elasticsearch:
        try:
            _rollback_elasticsearch_write(ictx, file_path, context=context)
        except Exception as exc:
            ok = False
            logger.critical(
                "Elasticsearch-Rollback fehlgeschlagen (%s, %s): %s",
                context,
                file_path,
                exc,
                exc_info=True,
            )
            ictx.stats.setdefault("rollback_errors", 0)
            ictx.stats["rollback_errors"] += 1

    if vector_ids:
        try:
            deleted = _rollback_vector_writes(
                ictx.store,
                vector_ids,
                context=context,
            )
            ictx.stats.setdefault("vectordb_rollback_chunks", 0)
            ictx.stats["vectordb_rollback_chunks"] += deleted
        except Exception as exc:
            ok = False
            logger.critical(
                "VectorDB-Rollback fehlgeschlagen (%s, %s): %s",
                context,
                file_path,
                exc,
                exc_info=True,
            )
            ictx.stats.setdefault("rollback_errors", 0)
            ictx.stats["rollback_errors"] += 1

    return ok


def _emit_file_log(ictx, file_log, t0):
    """FileLog-Finalisierung + Callback (Pattern D)."""
    if file_log and ictx.file_log_callback:
        file_log.duration_ms = elapsed_ms(t0)
        ictx.file_log_callback(file_log)


# ---------------------------------------------------------------------------
# Low-Level-Funktionen
# ---------------------------------------------------------------------------


def _init_ingestion_stats(
    all_files: list[str],
    files_to_process: list[str],
    deleted_files: list[str],
    resumed_hashes: dict,
) -> dict:
    """Erzeugt das initiale Stats-Dict.

    Keine Seiteneffekte — berechnet nur Werte.
    """
    return {
        "total_files": len(all_files),
        "processed_files": 0,
        "skipped_files": len(all_files) - len(files_to_process),
        "resumed_skipped": len(resumed_hashes),
        "deleted_files": len(deleted_files),
        "total_chunks": 0,
        "errors": 0,
    }


def _persist_hashes_checkpoint(ictx: _IngestionContext, force: bool = False) -> None:
    """Persistiert pending_hashes wenn Checkpoint-Intervall erreicht oder force=True.

    Args:
        ictx: Ingestion-Context mit pending_hashes und hash_store.
        force: True bei Abort/Cancel/Finalisierung — flusht sofort.
    """
    if not ictx.pending_hashes:
        return

    if force or ictx.stats["processed_files"] % HASH_CHECKPOINT_INTERVAL == 0:
        ictx.hash_store.upsert_many(ictx.datasource_id, ictx.embedding_model, ictx.pending_hashes)
        ictx.pending_hashes.clear()


def _check_vectordb_health(config: dict) -> bool:
    """Minimaler Health-Check fuer Qdrant.

    Erzeugt pro Aufruf einen kurzlebigen QdrantClient und ruft
    ``get_collections()`` auf. True nur bei erfolgreicher Rueckgabe.
    ALLE Exception-Typen werden als False gewertet.

    Nutzt bewusst NICHT ``VectorStore.list_collections(config)``: Jene
    Methode schluckt non-TransportError-Exceptions silent zu leerer Liste —
    "leere Rueckgabe" waere dort mehrdeutig (up ohne Collections vs. down).

    Args:
        config: Konfigurationsdict. Fehlender/None-``vectordb``-Block wird
            wie ``{}`` behandelt und nutzt ``DEFAULT_QDRANT_URL``.

    Returns:
        True bei erfolgreichem ``get_collections()``-Aufruf, sonst False.
    """
    from qdrant_client import QdrantClient

    qdrant_cfg = config.get("vectordb") or {}
    try:
        client = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
            timeout=2,
        )
        client.get_collections()
        return True
    except Exception as exc:
        logger.debug("VectorDB-Health-Check fehlgeschlagen: %s", exc)
        return False


def _wait_for_backend(
    config, ctx, control_callback, status_callback, error_type="embedding", embedding_model=None
):
    """Wartet unbegrenzt bis ein Backend wieder erreichbar ist.

    Args:
        config: Konfigurationsdict.
        ctx: ProcessingContext (fuer DDU-Backends via ctx.ddu_backends).
        control_callback: Cancel-Pruefung (aus Worker).
            Gibt "cancel", "shutdown" oder None zurueck.
            Blockiert intern bei Pause (DB-basiert).
        status_callback: Phase-Update fuer Admin-UI (oder None).
        error_type: "embedding", "ddu" oder "vectordb" — bestimmt den Health-Check.
        embedding_model: Modellname fuer Embedding-Health-Check.
            Pflicht wenn error_type="embedding". Fuer "vectordb" ignoriert.

    Returns:
        True wenn Backend wieder erreichbar, False wenn Cancel/Shutdown.
    """
    from kiara.backend_selector import (
        get_embedding_backend_for_model,
        invalidate_cache,
    )
    from kiara.ingestion.ddu_client import check_ddu_health

    if status_callback:
        status_callback("Warte auf Backend...")

    logger.warning(
        "Backend (%s) nicht erreichbar — warte auf Recovery. Cancel zum Abbrechen.",
        error_type,
    )

    while True:
        # --- Cancel/Shutdown-Check (jede Sekunde) ---
        for _ in range(BACKEND_WAIT_INTERVAL_SECS):
            if control_callback:
                signal = control_callback()
                if signal in ("cancel", "shutdown"):
                    logger.info("Wait abgebrochen durch %s-Signal.", signal)
                    return False
            time.sleep(1.0)

        # --- Health-Check (alle BACKEND_WAIT_INTERVAL_SECS Sekunden) ---
        try:
            if error_type == "ddu":
                ok, _err_msg = check_ddu_health(ctx.ddu_backends)
            elif error_type == "vectordb":
                ok = _check_vectordb_health(config)
            else:
                # Embedding: Zwei-Stufen-Check
                # 1. Cache invalidieren (30s TTL wuerde stale Ergebnis liefern)
                invalidate_cache("embedding")
                # 2. Modell-Verfuegbarkeit pruefen (inkludiert Server-Erreichbarkeit)
                backend = get_embedding_backend_for_model(embedding_model)
                ok = backend is not None
        except Exception as exc:
            logger.debug("Health-Check fehlgeschlagen: %s", exc)
            ok = False

        if ok:
            logger.info("Backend (%s) wieder erreichbar — fahre fort.", error_type)
            return True


def _check_ddu_preflight(ctx: ProcessingContext, warmup_stats: dict | None) -> None:
    """Prueft DDU-Backend-Erreichbarkeit. Wirft RuntimeError wenn nicht erreichbar.

    Wird uebersprungen wenn:
    - ctx.ddu_enabled is False (DDU deaktiviert)
    - warmup_stats DDU als OK bestaetigen (Pre-flight Skip)
    """
    if not ctx.ddu_enabled:
        return

    ddu_warmup_ok = warmup_stats and any(
        d.get("category") == "ddu" and d.get("ok") for d in warmup_stats.get("details", [])
    )
    if ddu_warmup_ok:
        logger.debug("DDU Pre-flight Check uebersprungen (Warmup-OK)")
        return

    from kiara.ingestion.ddu_client import check_ddu_health

    ddu_ok, ddu_error = check_ddu_health(ctx.ddu_backends)
    if not ddu_ok:
        raise RuntimeError(
            f"DDU ist fuer diese Datasource aktiviert, aber kein DDU-Backend ist "
            f"erreichbar: {ddu_error}. Ingestion abgebrochen."
        )
    logger.info("DDU Pre-flight Check bestanden — Backend erreichbar")


# ---------------------------------------------------------------------------
# Content Loading + Chunking
# ---------------------------------------------------------------------------


def _load_content(file_path, ctx, stats, file_log=None):
    """Laedt eine Datei via DDU oder Standard-Loader.

    Behandelt DDU-Fehler (Timeout, Connection, HTTP) als Infra-Fehler
    und trackt erfolgreiche DDU-Verarbeitung in stats.

    Args:
        file_path: Pfad zur zu verarbeitenden Datei.
        ctx: ProcessingContext mit DDU-Config.
        stats: Stats-Dict (wird in-place aktualisiert).
        file_log: Optionaler FileLogEntry (wird in-place aktualisiert).

    Returns:
        Tuple (doc, is_ddu_error).
        doc ist None wenn die Datei nicht geladen werden konnte.
        is_ddu_error ist True wenn ein DDU-Infra-Fehler aufgetreten ist
        (stats und file_log sind dann bereits aktualisiert).
    """
    # DDU-OCR fuer reine Bilddateien unterdruecken wenn ddu_ocr_images=False
    ext = Path(file_path).suffix.lower()
    file_ddu_enabled = ctx.ddu_enabled
    if file_ddu_enabled and not ctx.ddu_ocr_images and ext in IMAGE_FORMATS:
        file_ddu_enabled = False

    doc = load_file(file_path, ddu_enabled=file_ddu_enabled, ddu_backends=ctx.ddu_backends)

    # File-Log: Parser-Info erfassen
    if file_log and doc:
        file_log.parser = doc["metadata"].get("parser", "")
        file_log.parser_fallback = bool(doc["metadata"].get("parser_fallback", False))
    elif file_log and doc is None:
        file_log.parser = "unknown"

    # DDU-Tracking (Remote + Fehler-Typen + Timeout)
    if doc:
        parser = doc["metadata"].get("parser", "")
        ddu_error = doc["metadata"].get("ddu_error", "")

        # DDU-Fehler: Datei als Fehler zaehlen und ueberspringen.
        # Kein lautloser Fallback auf Standard-Loader — wenn DDU aktiviert
        # ist aber das Backend nicht erreichbar, wird die Datei uebersprungen.
        if ddu_error in ("timeout", "connection", "http_error"):
            error_label = {
                "timeout": "DDU timeout",
                "connection": "DDU Backend nicht erreichbar",
                "http_error": "DDU HTTP-Fehler",
            }[ddu_error]
            if ddu_error == "timeout":
                stats.setdefault("ddu_timeout", 0)
                stats["ddu_timeout"] += 1
            else:
                stats.setdefault("ddu_remote_errors", 0)
                stats["ddu_remote_errors"] += 1
            stats["errors"] += 1
            if file_log:
                file_log.status = "error"
                file_log.error_message = error_label
                file_log.error_phase = "loading"
                file_log.is_infra_error = True
            return doc, True  # DDU-Infra-Fehler

        # Erfolgreiche DDU-Verarbeitung tracken
        if parser in ("docling", "docling-remote"):
            stats.setdefault("ddu_files", 0)
            stats["ddu_files"] += 1
        if parser == "docling-remote":
            stats.setdefault("ddu_remote_files", 0)
            stats["ddu_remote_files"] += 1
        if doc["metadata"].get("parser_fallback"):
            stats.setdefault("ddu_fallback_files", 0)
            stats["ddu_fallback_files"] += 1

    return doc, False


def _handle_vlm_content(file_path, ctx, stats):
    """Fuehrt VLM-Bildbeschreibung durch, falls aktiviert.

    Args:
        file_path: Pfad zur zu verarbeitenden Datei.
        ctx: ProcessingContext mit VLM-Config.
        stats: Stats-Dict (wird in-place aktualisiert).

    Returns:
        VLM-Dokument-Dict oder None wenn VLM deaktiviert oder fehlgeschlagen.
    """
    if not ctx.vlm_enabled or not ctx.vlm_model:
        return None

    vlm_doc = describe_image(
        file_path, vlm_model=ctx.vlm_model, ollama_url=ctx.vlm_url, prompt=ctx.vlm_prompt
    )
    ext = Path(file_path).suffix.lower()
    if vlm_doc is None and ext in IMAGE_FORMATS:
        stats.setdefault("vlm_errors", 0)
        stats["vlm_errors"] += 1
    elif vlm_doc is not None:
        stats.setdefault("vlm_chunks", 0)
        stats["vlm_chunks"] += 1

    return vlm_doc


def _determine_chunks(doc, ds_overrides, chunk_size, chunk_overlap):
    """Bestimmt Chunk-Strategie und fuehrt das Chunking fuer ein Dokument aus.

    VLM-Dokumente werden als einzelner Chunk behandelt.
    Andere Dokumente: Override > Auto-Erkennung > Global-Defaults.

    Returns:
        Liste von Chunk-Dicts mit 'text' und 'metadata'.
    """
    if doc["metadata"].get("parser") == "vlm":
        doc["metadata"]["chunk_strategy"] = "vlm_single"
        doc["metadata"]["chunk_index"] = 0
        doc["metadata"]["chunk_size_used"] = len(doc["text"])
        return [
            {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "char_start": 0,
                "char_end": len(doc["text"]),
            }
        ]

    cs, co = chunk_size, chunk_overlap
    if ds_overrides and ds_overrides.get("chunk_strategy") not in (None, "auto", "custom"):
        from kiara.ingestion.chunk_strategy import STRATEGIES

        strategy = ds_overrides["chunk_strategy"]
        if strategy in STRATEGIES:
            cs, co = STRATEGIES[strategy]
            doc["metadata"]["chunk_strategy"] = strategy
    elif ds_overrides and ds_overrides.get("chunk_size") is not None:
        cs, co = chunk_size, chunk_overlap
        doc["metadata"]["chunk_strategy"] = "custom"
    else:
        try:
            from kiara.ingestion.chunk_strategy import detect_chunk_params

            cs, co = detect_chunk_params(doc["text"], doc["metadata"])
        except Exception as exc:
            logger.warning("Chunk-Strategie-Erkennung fehlgeschlagen: %s", exc, exc_info=True)
            cs, co = chunk_size, chunk_overlap
            doc["metadata"]["chunk_strategy"] = "default"
    doc["metadata"]["chunk_size_used"] = cs

    return chunk_document(
        doc["text"],
        doc["metadata"],
        chunk_size=cs,
        chunk_overlap=co,
    )


def _chunk_and_collect(docs_to_chunk, ds_overrides, chunk_size, chunk_overlap, file_log=None):
    """Chunked eine Liste von Dokumenten und sammelt Texte, Metadaten, IDs und Spans.

    Args:
        docs_to_chunk: Liste von Dokument-Dicts mit 'text' und 'metadata'.
        ds_overrides: Per-Datasource Overrides.
        chunk_size: Standard-Chunk-Groesse.
        chunk_overlap: Standard-Chunk-Overlap.
        file_log: Optionaler FileLogEntry (wird in-place aktualisiert).

    Returns:
        Tuple (file_texts, file_metas, file_ids, file_spans).
    """
    file_texts = []
    file_metas = []
    file_ids = []
    file_spans = []

    for d in docs_to_chunk:
        chunks = _determine_chunks(d, ds_overrides, chunk_size, chunk_overlap)

        if file_log and chunks:
            if not file_log.chunk_strategy:
                file_log.chunk_strategy = chunks[0]["metadata"].get("chunk_strategy", "")
            file_log.chunk_count += len(chunks)

        for chunk in chunks:
            parser = chunk["metadata"].get("parser", "standard")
            chunk_id = hashlib.md5(
                f"{chunk['metadata']['source']}::{parser}::{chunk['metadata']['chunk_index']}".encode()
            ).hexdigest()
            file_texts.append(chunk["text"])
            file_metas.append(chunk["metadata"])
            file_ids.append(chunk_id)

            start = chunk.get("char_start")
            end = chunk.get("char_end")
            if isinstance(start, int) and isinstance(end, int):
                file_spans.append((start, end))
            else:
                file_spans.append(None)

    return file_texts, file_metas, file_ids, file_spans


def _process_single_file(file_path, config, ds_overrides, datasource_id, ctx, stats, file_log=None):
    """Laedt eine Datei, fuehrt DDU/VLM-Verarbeitung durch und chunked das Ergebnis.

    Orchestriert die drei Phasen: load -> VLM -> chunk.
    Aktualisiert stats in-place (DDU/VLM-Tracking, Fehler).

    Args:
        file_path: Pfad zur zu verarbeitenden Datei.
        config: Konfigurationsdict.
        ds_overrides: Per-Datasource Overrides.
        datasource_id: Optionale Datasource-ID.
        ctx: ProcessingContext mit DDU/VLM-Config.
        stats: Stats-Dict (wird in-place aktualisiert).
        file_log: Optionaler FileLogEntry (wird in-place aktualisiert).

    Returns:
        Tuple (file_texts, file_metas, file_ids, full_text, file_spans)
        oder ([], [], [], "", []) bei Fehlern.
    """
    chunk_size = config["ingestion"]["chunk_size"]
    chunk_overlap = config["ingestion"]["chunk_overlap"]

    with _tracer.start_as_current_span("load_content") as load_span:
        doc, is_ddu_error = _load_content(file_path, ctx, stats, file_log=file_log)
        load_span.set_attribute("parser", file_log.parser if file_log else "")
        load_span.set_attribute("parser_fallback", file_log.parser_fallback if file_log else False)
        load_span.set_attribute("ddu_used", ctx.ddu_enabled)
        load_span.set_attribute("ddu_error", is_ddu_error)

    if is_ddu_error:
        return [], [], [], "", []

    # full_text fuer Enrichment-Kontext extrahieren (doc ist ein dict, kein LangChain-Document)
    full_text = doc["text"] if doc else ""

    vlm_doc = _handle_vlm_content(file_path, ctx, stats)

    is_no_content = doc is not None and doc["metadata"].get("no_content")

    docs_to_chunk = []
    if doc is not None and not is_no_content:
        docs_to_chunk.append(doc)
    if vlm_doc is not None:
        docs_to_chunk.append(vlm_doc)

    if not docs_to_chunk:
        if is_no_content:
            if file_log:
                file_log.status = "skipped"
                file_log.error_message = "Kein extrahierbarer Inhalt"
                file_log.error_phase = ""
        elif doc is None:
            stats["errors"] += 1
            stats.setdefault("error_details", [])
            stats["error_details"].append(
                {
                    "file": Path(file_path).name,
                    "error": "Datei konnte nicht geladen werden",
                    "phase": "loading",
                }
            )
            if file_log:
                file_log.status = "error"
                file_log.error_message = "Datei konnte nicht geladen werden"
                file_log.error_phase = "loading"
        return [], [], [], "", []

    with _tracer.start_as_current_span("chunk_and_collect") as chunk_span:
        result = _chunk_and_collect(
            docs_to_chunk,
            ds_overrides,
            chunk_size,
            chunk_overlap,
            file_log=file_log,
        )
        chunk_span.set_attribute("chunk_count", len(result[0]) if result else 0)
        chunk_span.set_attribute("strategy", file_log.chunk_strategy if file_log else "")

    file_texts, file_metas, file_ids, file_spans = result
    return file_texts, file_metas, file_ids, full_text, file_spans


# ---------------------------------------------------------------------------
# Write Functions
# ---------------------------------------------------------------------------


def _write_embeddings(
    file_texts, file_metas, file_ids, store, batch_size, control_callback=None, embeddings=None
):
    """Schreibt Chunks in Batches in den VectorStore.

    Args:
        file_texts: Liste der Chunk-Texte.
        file_metas: Liste der Chunk-Metadaten.
        file_ids: Liste der Chunk-IDs.
        store: VectorStore-Instanz.
        batch_size: Groesse der Schreib-Batches.
        control_callback: Optionaler Cancel-Callback.
        embeddings: Vorab-berechnete Vektoren (Late Chunking). Wenn None,
                    berechnet store.add() die Embeddings intern.

    Returns:
        Liste der erfolgreich in die VectorDB geschriebenen Chunk-IDs.

    Raises:
        _CancelledError: Wenn der control_callback 'cancel' zurueckgibt.
        ValueError: Wenn embeddings-Laenge nicht zu file_texts passt.
    """
    if embeddings is not None and len(embeddings) != len(file_texts):
        raise ValueError("embeddings/file_texts length mismatch in _write_embeddings")

    written_ids: list[str] = []
    try:
        for i in range(0, len(file_texts), batch_size):
            if control_callback:
                signal = control_callback()
                if signal in ("cancel", "shutdown"):
                    raise _CancelledError()
            batch_texts = file_texts[i : i + batch_size]
            batch_metas = file_metas[i : i + batch_size]
            batch_ids = file_ids[i : i + batch_size]
            if embeddings is not None:
                batch_embeddings = embeddings[i : i + batch_size]
                store.add_with_embeddings(
                    batch_texts,
                    batch_metas,
                    batch_ids,
                    batch_embeddings,
                )
            else:
                store.add(batch_texts, batch_metas, batch_ids)
            written_ids.extend(batch_ids)
    except _CancelledError:
        raise
    except Exception:
        if written_ids:
            try:
                _rollback_vector_writes(
                    store,
                    written_ids,
                    context="partial_vector_write",
                )
            except Exception as rollback_exc:
                raise _RollbackFailedError(
                    "VectorDB-Partial-Write konnte nicht zurueckgerollt werden"
                ) from rollback_exc
        raise

    return written_ids


def _write_to_elasticsearch(config, file_texts, file_metas, file_ids, stats):
    """Schreibt Chunks nach Elasticsearch, falls als Hybrid-Suche konfiguriert.

    Bei konfiguriertem Elasticsearch sind Schreibfehler fatal fuer die Datei:
    der Hash darf sonst nicht als erfolgreich verarbeitet persistiert werden.
    """
    try:
        from kiara.retrieval.elasticsearch_service import bulk_index_chunks, is_configured

        if not is_configured(config):
            return

        count = bulk_index_chunks(config, file_texts, file_metas, file_ids)
        if count != len(file_ids):
            raise RuntimeError(
                f"Elasticsearch-Schreiben unvollstaendig: {count}/{len(file_ids)} Chunks indexiert"
            )
        stats.setdefault("es_indexed", 0)
        stats["es_indexed"] += count
        return count
    except Exception as exc:
        logger.warning("Elasticsearch-Schreiben fehlgeschlagen: %s", exc, exc_info=True)
        stats.setdefault("es_errors", 0)
        stats["es_errors"] += 1
        raise


def _extract_graph_entities(config, file_texts, file_ids, stats, *, conn=None):
    """Fuehrt Graph-Extraction durch und aktualisiert Stats.

    Args:
        config: Konfigurationsdict.
        file_texts: Liste der Chunk-Texte.
        file_ids: Liste der Chunk-IDs.
        stats: Stats-Dict (wird in-place aktualisiert).
    """
    graph_result = _run_graph_extraction(config, file_texts, file_ids, conn=conn)
    stats.setdefault("graph_entities", 0)
    stats.setdefault("graph_relations", 0)
    stats.setdefault("graph_errors", 0)
    stats["graph_entities"] += graph_result["entities"]
    stats["graph_relations"] += graph_result["relations"]
    stats["graph_errors"] += graph_result["errors"]


def _run_graph_extraction(config, file_texts, file_ids, *, conn=None):
    """Extrahiert Graph-Entities aus Chunks, falls in der Config aktiviert.

    Laeuft asynchron; Fehler werden geloggt, nicht propagiert.

    Returns:
        Dict mit {"entities": int, "relations": int, "errors": int}.
    """
    result_stats = {"entities": 0, "relations": 0, "errors": 0}

    graph_cfg = config.get("graph", {})
    if not graph_cfg.get("enabled", False) or not graph_cfg.get("extract_on_ingestion", True):
        return result_stats

    try:
        from kiara.graph.extractor import EntityExtractor
        from kiara.graph.store import GraphStore

        graph_store = GraphStore()
        extractor = EntityExtractor(config)

        from kiara.shared.constants import GRAPH_EXTRACTION_LIMIT

        existing = graph_store.list_nodes(limit=GRAPH_EXTRACTION_LIMIT)
        initial_existing_labels = [f"{n['id']}: {n['label']}" for n in existing]
        current_run_labels: list[str] = []
        current_run_ids: set[str] = set()

        for ct, ci in zip(file_texts, file_ids, strict=False):
            try:
                existing_labels = _bounded_existing_labels(
                    current_run_labels,
                    initial_existing_labels,
                    GRAPH_EXTRACTION_LIMIT,
                )
                try:
                    result = asyncio.run(extractor.extract(ct, ci, existing_labels))
                except RuntimeError:
                    # Event-Loop laeuft bereits (z.B. in async-Kontext) →
                    # isolierter Thread mit eigenem Loop (gleicher Ansatz
                    # wie AgentChain._execute_tool).
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run, extractor.extract(ct, ci, existing_labels)
                        ).result()
                for entity in result.get("entities", []):
                    eid = entity.get("id")
                    etype = entity.get("type")
                    elabel = entity.get("label")
                    if not eid or not etype or not elabel:
                        continue
                    # chunk_id mitgeben, damit Entity-only Chunks (ohne Relations)
                    # vom GraphRetriever ueber graph_node_chunks gefunden werden (#685).
                    if entity.get("preserve_existing"):
                        graph_store.link_node_to_chunk(eid, ci, conn=conn)
                    else:
                        graph_store.add_node(
                            eid,
                            etype,
                            elabel,
                            entity.get("properties"),
                            source="extraction",
                            chunk_id=ci,
                            conn=conn,
                        )
                    if eid not in current_run_ids:
                        current_run_ids.add(eid)
                        current_run_labels.append(f"{eid}: {elabel}")
                    result_stats["entities"] += 1
                for rel in result.get("relations", []):
                    rsrc = rel.get("source")
                    rtgt = rel.get("target")
                    rrel = rel.get("relation")
                    if not rsrc or not rtgt or not rrel:
                        continue
                    graph_store.add_edge(
                        rsrc,
                        rtgt,
                        rrel,
                        rel.get("properties"),
                        chunk_id=ci,
                        conn=conn,
                    )
                    result_stats["relations"] += 1
            except Exception as exc:
                logger.warning(
                    "Graph-Extraction fehlgeschlagen fuer Chunk %s: %s",
                    ci,
                    exc,
                    exc_info=True,
                )
                result_stats["errors"] += 1
    except Exception as exc:
        logger.warning(
            "Graph-Extraction Initialisierung fehlgeschlagen: %s",
            exc,
            exc_info=True,
        )
        result_stats["errors"] += 1

    return result_stats


def _bounded_existing_labels(
    current_run_labels: list[str],
    initial_existing_labels: list[str],
    limit: int,
) -> list[str]:
    """Current-run Entities vor altem Snapshot priorisieren und begrenzen."""
    result = []
    seen_ids = set()
    for entry in current_run_labels + initial_existing_labels:
        node_id = entry.split(": ", 1)[0]
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        result.append(entry)
        if len(result) >= limit:
            break
    return result


# ---------------------------------------------------------------------------
# Circuit-Breaker
# ---------------------------------------------------------------------------


def _handle_circuit_breaker(
    ictx: _IngestionContext,
    file_path: str,
    file_log: FileLogEntry | None,
    error_phase: str,
    error_description: str,
    is_infra: bool,
    file_retries: int,
    t0: float,
    *,
    is_colbert: bool = False,
) -> tuple[_CircuitBreakerResult, int]:
    """Einheitliche Circuit-Breaker-Logik fuer Loading- und Embedding-Phase.

    Prueft ob ein Fehler die Circuit-Breaker-Schwelle erreicht und entscheidet
    ueber die Reaktion: Weitermachen, Retry nach Backend-Recovery, Abort oder Skip.

    Unterschiede zwischen Phasen werden ueber Parameter gesteuert:
    - is_infra: Loading setzt dies bedingt (file_log.is_infra_error),
      Embedding setzt immer True (da except _CATEGORICAL_ERRORS nur Infra faengt)
    - error_phase: Steuert Stats-Korrektur bei RETRY ("loading" vs. "embedding")
    - error_description: Caller-spezifischer Text fuer abort_msg
    - is_colbert: Bei True wird auch colbert_errors bei RETRY zurueckkorrigiert

    Returns:
        (result, updated_file_retries)
    """
    if is_infra:
        ictx.consecutive_errors += 1

    # vectordb-Fehler sind globale Qdrant-Backend-Ausfaelle und warten im
    # Worker-Modus sofort (keine Schwellwert-Phase, in der Files geskippt
    # werden wuerden). CLI-Modus ohne control_callback faellt auf ABORT
    # (siehe unten).
    is_vectordb_phase = error_phase == "vectordb"

    if not is_infra or (
        not is_vectordb_phase and ictx.consecutive_errors < INGESTION_MAX_CONSECUTIVE_ERRORS
    ):
        return _CircuitBreakerResult.CONTINUE, file_retries

    # --- Ab hier: Schwelle erreicht ODER vectordb-Sofortreaktion ---
    file_retries += 1

    # Max Retries (>2) → SKIP
    if file_retries > 2:
        ictx.consecutive_errors = 0
        logger.warning("Datei '%s' nach %d Retries uebersprungen.", file_path, file_retries - 1)
        return _CircuitBreakerResult.SKIP, file_retries

    # CLI-Modus (kein control_callback) → ABORT
    if not ictx.control_callback:
        _emit_file_log(ictx, file_log, t0)
        _persist_hashes_checkpoint(ictx, force=True)
        abort_msg = (
            f"Abbruch: {ictx.consecutive_errors} aufeinanderfolgende "
            f"Backend-Fehler ({error_phase}). Letzter: {error_description}"
        )
        logger.error(abort_msg)
        ictx.stats["abort_reason"] = abort_msg
        return _CircuitBreakerResult.ABORT, file_retries

    # Worker-Modus → Wait + Retry
    _persist_hashes_checkpoint(ictx, force=True)

    error_type = {
        "loading": "ddu",
        "embedding": "embedding",
        "vectordb": "vectordb",
    }[error_phase]

    recovered = _wait_for_backend(
        ictx.config,
        ictx.ctx,
        ictx.control_callback,
        ictx.status_callback,
        error_type=error_type,
        embedding_model=ictx.embedding_model,
    )

    if recovered and is_vectordb_phase:
        ictx.stats.setdefault("vectordb_recoveries", 0)
        ictx.stats["vectordb_recoveries"] += 1

    if not recovered:
        return _CircuitBreakerResult.CANCEL, file_retries

    # consecutive_errors wird NICHT hier zurueckgesetzt — erst nach erfolgreichem
    # Retry durch die Success-Pfade. Vorzeitiger Reset wuerde den Counter bei
    # erneutem Fehler bei 1 statt beim vorherigen Wert starten (Livelock-Risiko).

    # Stats-Korrektur (phasen-abhaengig)
    ictx.stats["errors"] -= 1
    if error_phase == "loading":
        if file_log and file_log.error_message:
            if "timeout" in file_log.error_message.lower():
                ictx.stats["ddu_timeout"] = max(0, ictx.stats.get("ddu_timeout", 1) - 1)
            elif file_log.error_message.startswith("DDU"):
                ictx.stats["ddu_remote_errors"] = max(0, ictx.stats.get("ddu_remote_errors", 1) - 1)
    elif error_phase == "embedding":
        if is_colbert and ictx.stats.get("colbert_errors", 0) > 0:
            ictx.stats["colbert_errors"] -= 1
        if ictx.stats.get("error_details"):
            ictx.stats["error_details"].pop()
    elif error_phase == "vectordb":
        ictx.stats["vectordb_errors"] = max(0, ictx.stats.get("vectordb_errors", 1) - 1)
        if is_colbert and ictx.stats.get("colbert_errors", 0) > 0:
            ictx.stats["colbert_errors"] -= 1
        if ictx.stats.get("error_details"):
            ictx.stats["error_details"].pop()

    # file_log zuruecksetzen (BEIDE Phasen einheitlich — fixt Bug in Embedding-Phase)
    if file_log:
        file_log.status = "pending"
        file_log.error_message = ""
        file_log.error_phase = ""
        file_log.is_infra_error = False

    if ictx.status_callback:
        ictx.status_callback("Verarbeite Dateien...")

    return _CircuitBreakerResult.RETRY, file_retries


# ---------------------------------------------------------------------------
# Loading-Result
# ---------------------------------------------------------------------------


def _handle_loading_result(
    ictx: _IngestionContext,
    file_path: str,
    file_log: FileLogEntry | None,
    errors_before: int,
    file_retries: int,
    t0: float,
) -> tuple[_FileAction, int]:
    """Wertet das Ergebnis von _process_single_file() aus wenn file_texts leer ist.

    Wird NUR aufgerufen wenn ``not file_texts`` (Caller prueft dies vorab).

    Behandelt 2 Faelle:
    1. file_texts leer + Fehler (infra/content) → SKIP (nur file_log_callback, KEIN progress)
    2. file_texts leer + kein Fehler (z.B. leere Datei) → SKIP (progress + file_log_callback)

    Returns:
        (_FileAction, updated_file_retries)
    """
    is_error = ictx.stats["errors"] > errors_before

    if is_error:
        is_infra = file_log is not None and file_log.is_infra_error

        # Hash bei ALLEN Fehlern entfernen (infra UND content)
        _cleanup_file_hash(ictx, file_path)

        error_desc = (
            f"{file_log.error_message if file_log else 'unbekannt'} "
            f"(Phase: {file_log.error_phase if file_log else 'unbekannt'})"
        )
        cb_result, file_retries = _handle_circuit_breaker(
            ictx,
            file_path,
            file_log,
            error_phase="loading",
            error_description=error_desc,
            is_infra=is_infra,
            file_retries=file_retries,
            t0=t0,
        )

        if cb_result == _CircuitBreakerResult.RETRY:
            return _FileAction.RETRY, file_retries
        if cb_result == _CircuitBreakerResult.ABORT:
            return _FileAction.ABORT, file_retries
        if cb_result == _CircuitBreakerResult.CANCEL:
            return _FileAction.CANCEL, file_retries
        # CONTINUE (unter Schwelle) oder SKIP (Max Retries) → naechste Datei.
        # consecutive_errors wird hier bewusst NICHT zurueckgesetzt:
        #  - CONTINUE: Counter muss aufeinanderfolgende Infra-Fehler
        #    akkumulieren, sonst wird die Circuit-Breaker-Schwelle nie erreicht.
        #  - SKIP: bereits in _handle_circuit_breaker zurueckgesetzt (Livelock-
        #    Verhinderung fuer naechste Datei).
        # Reset erfolgt nur im No-Content-else-Branch (Backend hat erfolgreich
        # geantwortet) bzw. in den Erfolgs-Pfaden nach abgeschlossener Datei.
        ictx.skipped += 1
        if ictx.progress_callback:
            ictx.progress_callback(ictx.skipped + ictx.stats["processed_files"], ictx.total)

    else:
        # Kein Fehler, aber kein Inhalt (z.B. leere Datei, VLM-only ohne Text)
        # Hash NICHT persistieren — Datei soll beim naechsten Run erneut
        # geprueft werden, falls sie inzwischen Content bekommen hat.
        ictx.consecutive_errors = 0
        ictx.skipped += 1
        if ictx.progress_callback:
            ictx.progress_callback(ictx.skipped + ictx.stats["processed_files"], ictx.total)
        if file_log and file_log.status == "success":
            file_log.status = "skipped"

    # Gemeinsam fuer Error-SKIP und No-Content-SKIP
    _emit_file_log(ictx, file_log, t0)
    return _FileAction.SKIP, file_retries


# ---------------------------------------------------------------------------
# ColBERT-Integration (Phase 18b)
# ---------------------------------------------------------------------------


class _ColBERTWriteBlocked(RuntimeError):
    """Signalisiert, dass ein ColBERT-Write blockiert wurde.

    Gruende: DS soft-deleted, DB-Lesefehler, Model-Mismatch bei
    persistierter Config. Wird von _handle_colbert_step dediziert
    behandelt (ABORT mit abort_reason).
    """

    pass


def _datasource_accepts_colbert_write(datasource_id: int) -> bool:
    """Prueft ob die Datasource aktiv ist (fail-closed bei Fehler).

    Returns False bei: fehlende DS, inaktive DS, DB-Lesefehler.
    """
    from kiara.shared.db import db_query

    try:
        with db_query() as conn:
            row = conn.execute(
                "SELECT is_active FROM datasources WHERE id = %s",
                (datasource_id,),
            ).fetchone()
            if not row:
                return False
            return bool(row["is_active"])
    except Exception as exc:
        logger.warning(
            "DS-Active-Check fehlgeschlagen fuer DS %s (fail-closed): %s",
            datasource_id,
            exc,
            exc_info=True,
        )
        return False


def _ensure_colbert_model_is_current_for_write(config: dict, colbert_model: str) -> None:
    """Prueft ob das ColBERT-Modell noch dem persistierten Config-Stand entspricht.

    Nur aktiv wenn ``_kiara_enforce_persisted_colbert_model=True`` (Worker/CLI Default).
    Custom-Config-Aufrufer ohne Marker bleiben kompatibel.

    Raises:
        _ColBERTWriteBlocked: Bei Model-Mismatch oder Fresh-Read-Fehler.
    """
    if not config.get("_kiara_enforce_persisted_colbert_model"):
        return

    from kiara.backend_selector import normalize_model_name
    from kiara.config import load_config
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL

    try:
        fresh_config = load_config()
    except Exception as exc:
        raise _ColBERTWriteBlocked(f"Fresh-Config-Read fehlgeschlagen: {exc}") from exc

    fresh_model = normalize_model_name(
        fresh_config.get("retrieval", {}).get("colbert", {}).get("model", COLBERT_DEFAULT_MODEL)
    )
    if fresh_model != colbert_model:
        raise _ColBERTWriteBlocked(
            f"ColBERT-Modell-Mismatch: laufend={colbert_model}, aktuell konfiguriert={fresh_model}"
        )


def _colbert_enabled_for_run(ictx: _IngestionContext) -> bool:
    """Prueft ob ColBERT fuer diesen Ingestion-Lauf aktiv ist."""
    return bool(
        ictx.datasource_id is not None
        and ictx.config.get("retrieval", {}).get("colbert", {}).get("enabled", False)
        and (ictx.ds_overrides or {}).get("colbert_enabled", False)
    )


def _sync_colbert_chunks(
    ictx: _IngestionContext,
    chunk_ids: list[str],
    chunk_texts: list[str],
    chunk_metas: list[dict],
) -> tuple[int, str]:
    """Low-Level-ColBERT-Sync: Embed + Upsert + Association.

    Dual-Lock: aeusseres DS-Lock serialisiert gegen Model-Switch-Cleanup,
    inneres Key-Lock serialisiert gegen Cleanups desselben Keyspace.
    Guards NACH Lock-Erwerb pruefen DS-Status und Model-Aktualitaet.

    Kein Stats-Commit, kein Retry, kein Circuit-Breaker.
    Exceptions werden unveraendert nach oben gereicht.

    Returns:
        (written_count, collection_name)
    """
    if not _colbert_enabled_for_run(ictx) or not chunk_ids:
        return 0, ""

    from kiara.backend_selector import normalize_model_name
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_cleanup_lock,
        colbert_datasource_lock,
        colbert_ds_model_key,
        embed_colbert,
    )
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL

    colbert_model = normalize_model_name(
        ictx.config.get("retrieval", {}).get("colbert", {}).get("model", COLBERT_DEFAULT_MODEL)
    )
    ds_model_key = colbert_ds_model_key(colbert_model)

    with _tracer.start_as_current_span("colbert_index") as span:
        span.set_attribute("chunk_count", len(chunk_ids))
        span.set_attribute("model", colbert_model)

        # Aeusseres Lock: DS-weite Serialisierung gegen Cleanup
        with colbert_datasource_lock(ictx.datasource_id, wait=True):
            # Inneres Lock: Keyspace-Serialisierung
            with colbert_cleanup_lock(ds_model_key, wait=True):
                # Guards NACH Lock-Erwerb
                if not _datasource_accepts_colbert_write(ictx.datasource_id):
                    raise _ColBERTWriteBlocked(
                        f"DS {ictx.datasource_id} ist inaktiv oder nicht vorhanden"
                    )
                _ensure_colbert_model_is_current_for_write(ictx.config, colbert_model)

                # ColBERTStore erst NACH Lock + Guards konstruieren
                cstore = ColBERTStore(ictx.config, colbert_model)
                cstore.ensure_collection()
                collection_name = cstore.get_collection_name()

                existing_ids = ictx.chunk_ds.chunks_exist(chunk_ids, ds_model_key)
                missing_ids = [cid for cid in chunk_ids if cid not in existing_ids]
                written_count = 0
                if missing_ids:
                    missing_texts = [
                        t
                        for t, cid in zip(chunk_texts, chunk_ids, strict=False)
                        if cid not in existing_ids
                    ]
                    missing_metas = [
                        m
                        for m, cid in zip(chunk_metas, chunk_ids, strict=False)
                        if cid not in existing_ids
                    ]

                    # ds_ids aus Metadaten entfernen vor Add
                    # (Crash vor add_associations wuerde DS-sichtbare Points ohne DB-SoT erzeugen)
                    for meta in missing_metas:
                        meta.pop("ds_ids", None)

                    token_embeddings = embed_colbert(ictx.config, colbert_model, missing_texts)
                    # Laengen-Guard: Embedding-Count muss == Chunk-Count
                    if len(token_embeddings) != len(missing_ids):
                        raise BackendUnavailableError(
                            f"ColBERT embed_colbert lieferte {len(token_embeddings)} Embeddings "
                            f"fuer {len(missing_ids)} Chunks"
                        )
                    cstore.add(missing_texts, missing_metas, missing_ids, token_embeddings)
                    written_count = len(missing_ids)

                ictx.chunk_ds.add_associations(chunk_ids, ictx.datasource_id, ds_model_key)
                ds_ids_map = ictx.chunk_ds.get_datasource_ids_for_chunks(chunk_ids, ds_model_key)
                cstore.update_ds_ids(ds_ids_map, wait=True)
                span.set_attribute("written_count", written_count)
                return written_count, collection_name


def _handle_colbert_step(
    ictx: _IngestionContext,
    *,
    chunk_ids: list[str],
    chunk_texts: list[str],
    chunk_metas: list[dict],
    file_path: str,
    file_log: FileLogEntry | None,
    file_retries: int,
    t0: float,
) -> tuple[_FileAction, int]:
    """Ruft _sync_colbert_chunks() auf mit demselben Error-/Retry-Vertrag.

    Commitet colbert_chunks_written/colbert_collection erst nach Erfolg.
    """
    if not _colbert_enabled_for_run(ictx) or not chunk_ids:
        return _FileAction.SUCCESS, file_retries

    try:
        written, coll_name = _sync_colbert_chunks(
            ictx,
            chunk_ids,
            chunk_texts,
            chunk_metas,
        )
        ictx.stats.setdefault("colbert_chunks_written", 0)
        ictx.stats["colbert_chunks_written"] += written
        if coll_name:
            ictx.stats["colbert_collection"] = coll_name
        return _FileAction.SUCCESS, file_retries

    except _ColBERTWriteBlocked as exc:
        # Dedizierter Handler: ABORT mit abort_reason
        logger.warning(
            "ColBERT-Write blockiert fuer '%s': %s",
            file_path,
            exc,
        )
        _record_error(ictx, file_log, file_path, exc, "embedding", is_colbert=True)
        ictx.stats["abort_reason"] = f"ColBERT-Write blockiert: {exc}"
        ictx.pending_hashes.pop(file_path, None)
        _cleanup_file_hash(ictx, file_path)
        _persist_hashes_checkpoint(ictx, force=True)
        _emit_file_log(ictx, file_log, t0)
        return _FileAction.ABORT, file_retries

    except _CancelledError:
        # Hash-Cleanup-Invariante: pending_hashes entfernen nach Dense-Success
        ictx.pending_hashes.pop(file_path, None)
        _cleanup_file_hash(ictx, file_path)
        _persist_hashes_checkpoint(ictx, force=True)
        _emit_file_log(ictx, file_log, t0)
        return _FileAction.CANCEL, file_retries

    except VectorDBConnectionError as exc:
        label = "ColBERT-VectorDB-Fehler"
        logger.error(
            "%s bei '%s': %s (consecutive: %d/%d)",
            label,
            file_path,
            exc,
            ictx.consecutive_errors + 1,
            INGESTION_MAX_CONSECUTIVE_ERRORS,
        )
        _record_error(
            ictx,
            file_log,
            file_path,
            exc,
            "vectordb",
            is_colbert=True,
            is_vectordb=True,
        )
        ictx.pending_hashes.pop(file_path, None)
        _cleanup_file_hash(ictx, file_path)

        cb_result, file_retries = _handle_circuit_breaker(
            ictx,
            file_path,
            file_log,
            error_phase="vectordb",
            error_description=f"{label}: {exc}",
            is_infra=True,
            file_retries=file_retries,
            t0=t0,
            is_colbert=True,
        )

        if cb_result == _CircuitBreakerResult.RETRY:
            return _FileAction.RETRY, file_retries
        if cb_result == _CircuitBreakerResult.ABORT:
            return _FileAction.ABORT, file_retries
        if cb_result == _CircuitBreakerResult.CANCEL:
            return _FileAction.CANCEL, file_retries
        return _FileAction.SKIP, file_retries

    except _CATEGORICAL_ERRORS as exc:
        label = "ColBERT-Backend-Fehler"
        logger.error(
            "%s bei '%s': %s (consecutive: %d/%d)",
            label,
            file_path,
            exc,
            ictx.consecutive_errors + 1,
            INGESTION_MAX_CONSECUTIVE_ERRORS,
        )
        _record_error(ictx, file_log, file_path, exc, "embedding", is_colbert=True)
        ictx.pending_hashes.pop(file_path, None)
        _cleanup_file_hash(ictx, file_path)

        cb_result, file_retries = _handle_circuit_breaker(
            ictx,
            file_path,
            file_log,
            error_phase="embedding",
            error_description=f"{label}: {exc}",
            is_infra=True,
            file_retries=file_retries,
            t0=t0,
            is_colbert=True,
        )

        if cb_result == _CircuitBreakerResult.RETRY:
            return _FileAction.RETRY, file_retries
        if cb_result == _CircuitBreakerResult.ABORT:
            return _FileAction.ABORT, file_retries
        if cb_result == _CircuitBreakerResult.CANCEL:
            return _FileAction.CANCEL, file_retries
        return _FileAction.SKIP, file_retries

    except Exception as exc:
        logger.error(
            "ColBERT-Sync fehlgeschlagen fuer %s: %s",
            file_path,
            redact_secrets_in_text(str(exc)),
            exc_info=True,
        )
        _record_error(ictx, file_log, file_path, exc, "embedding", is_colbert=True)

        ictx.consecutive_errors = 0
        ictx.pending_hashes.pop(file_path, None)
        _cleanup_file_hash(ictx, file_path)
        return _FileAction.SKIP, file_retries


# ---------------------------------------------------------------------------
# Shared Helper: DS-ID-Sync (Qdrant + ES)
# ---------------------------------------------------------------------------


def _sync_ds_ids(ictx: _IngestionContext, chunk_ids: list[str], context: str = "sync") -> None:
    """DS-IDs in chunk_datasources + Qdrant + ES synchronisieren.

    Ersetzt die duplizierten Bloecke in _handle_chunk_dedup (dedup)
    und _handle_embedding_phase (embed). ColBERT-Spezialfall bleibt separat.
    """
    try:
        ictx.chunk_ds.add_associations(chunk_ids, ictx.datasource_id, ictx.embedding_model)
    except Exception as exc:
        logger.error("add_associations fehlgeschlagen (%s): %s", context, exc, exc_info=True)
        raise
    _update_external_ds_ids(ictx, chunk_ids, context=context)


def _update_external_ds_ids(
    ictx: _IngestionContext,
    chunk_ids: list[str],
    context: str = "sync",
) -> None:
    """Aktualisiert Qdrant/ES-ds_ids aus der chunk_datasources-SoT."""
    try:
        ds_ids_map = ictx.chunk_ds.get_datasource_ids_for_chunks(chunk_ids, ictx.embedding_model)
        ictx.store.update_ds_ids(ds_ids_map)
        try:
            from kiara.retrieval import elasticsearch_service as es_svc

            if es_svc.is_configured(ictx.config):
                es_svc.update_ds_ids(ictx.config, ds_ids_map)
        except Exception as exc:
            logger.warning(
                "ES ds_ids-Update fehlgeschlagen (%s): %s",
                context,
                exc,
                exc_info=True,
            )
    except Exception as exc:
        logger.error("update_ds_ids fehlgeschlagen (%s): %s", context, exc, exc_info=True)
        raise


def _persist_graph_and_associations(
    ictx: _IngestionContext,
    file_texts: list[str],
    file_ids: list[str],
) -> None:
    """Persistiert Graph-Junctions und chunk_datasources unter gemeinsamem Lock."""
    if ictx.datasource_id is None:
        _extract_graph_entities(ictx.config, file_texts, file_ids, ictx.stats)
        return
    from kiara.shared.db import db_transaction

    with db_transaction() as conn:
        ictx.chunk_ds.acquire_graph_cleanup_lock(conn)
        _extract_graph_entities(
            ictx.config,
            file_texts,
            file_ids,
            ictx.stats,
            conn=conn,
        )
        ictx.chunk_ds.add_associations(
            file_ids,
            ictx.datasource_id,
            ictx.embedding_model,
            conn=conn,
            lock_already_held=True,
        )
    _update_external_ds_ids(ictx, file_ids, context="embed")


# ---------------------------------------------------------------------------
# Shared Helper: ColBERT-Action-Dispatch
# ---------------------------------------------------------------------------


def _apply_colbert_action(cb_action, span, status_prefix: str) -> str:
    """Wertet ColBERT-Action aus, setzt span-Attribute.

    Returns: "abort" | "retry" | "skip" | "ok"
    """
    if cb_action in (_FileAction.ABORT, _FileAction.CANCEL):
        span.set_attribute("final_status", f"colbert_{cb_action.value}")
        return "abort"
    if cb_action == _FileAction.RETRY:
        span.set_attribute("final_status", f"retry_{status_prefix}")
        return "retry"
    if cb_action == _FileAction.SKIP:
        span.set_attribute("final_status", f"skipped_{status_prefix}")
        return "skip"
    return "ok"


# ---------------------------------------------------------------------------
# Chunk-Deduplication
# ---------------------------------------------------------------------------


def _handle_chunk_dedup(
    ictx: _IngestionContext,
    file_path: str,
    file_texts: list[str],
    file_metas: list[dict],
    file_ids: list[str],
    file_spans: list[tuple[int, int] | None],
    file_log: FileLogEntry | None,
    t0: float,
) -> tuple[
    _DedupResult,
    list[str],
    list[dict],
    list[str],
    list[tuple[int, int] | None],
    _ExistingBatch | None,
]:
    """Chunk-Deduplication mit synchronem Span-Filtering.

    Prueft ob file_ids bereits in chunk_datasources existieren.
    Bei FULL-Dedup: Dense-Seiteneffekte erledigt, aber FULL-Finalisierung
    (stats, hashes, callbacks) wird an den Caller delegiert damit der
    ColBERT-Step vorher noch laufen kann.
    Bei PARTIAL-Dedup: gefilterte Listen + existing_batch zurueckgegeben.
    Bei NO_DEDUP: Original-Listen unveraendert.

    Returns:
        (result, file_texts, file_metas, file_ids, file_spans, existing_batch)
        Bei FULL sind die Listen leer, existing_batch enthaelt die dedupten Chunks.
    """
    if ictx.datasource_id is None or not file_ids:
        return _DedupResult.NO_DEDUP, file_texts, file_metas, file_ids, file_spans, None

    try:
        existing_ids = ictx.chunk_ds.chunks_exist(file_ids, ictx.embedding_model)
    except Exception as exc:
        logger.warning("chunk_datasources Lookup fehlgeschlagen: %s", exc, exc_info=True)
        existing_ids = set()

    # chunks_exist befragt nur die PostgreSQL-Assoziationstabelle. Eine partielle
    # Failure aus einem frueheren Lauf (DB-Eintrag ohne Qdrant-Vektor) wuerde
    # sonst zu permanentem Skip + 4-Schichten-Drift fuehren — Qdrant set_payload
    # auf einen nicht-existenten Point ist ein stiller No-Op. Daher Punkt-
    # Existenz direkt im VectorStore verifizieren und Orphans neu embedden.
    if existing_ids:
        try:
            verified = ictx.store.get_documents_by_ids(
                list(existing_ids),
                include=[],
            )
            verified_ids = set(verified.get("ids", []))
        except VectorDBConnectionError:
            raise
        except Exception as exc:
            logger.warning(
                "Qdrant-Existenz-Check bei Dedup fehlgeschlagen — Re-Embedding: %s",
                exc,
                exc_info=True,
            )
            existing_ids = set()
        else:
            orphans = existing_ids - verified_ids
            if orphans:
                logger.warning(
                    "Dedup: %d/%d Chunks haben chunk_datasources-Eintrag aber keinen "
                    "Qdrant-Vektor — werden neu embedded (Datei: %s)",
                    len(orphans),
                    len(existing_ids),
                    file_path,
                )
                existing_ids = verified_ids

    existing_batch_ids = [id for id in file_ids if id in existing_ids]
    eb = None
    if existing_batch_ids:
        try:
            _sync_ds_ids(ictx, existing_batch_ids, context="dedup")
        except VectorDBConnectionError:
            # Qdrant-Ausfall: Caller (_process_file_in_pipeline) behandelt
            # diesen Pfad als vectordb-Circuit-Breaker-Fall, nicht als Dedup-Fallback.
            raise
        except Exception:
            logger.error(
                "DS-ID-Sync bei Dedup fehlgeschlagen — Fallback auf Re-Embedding",
                exc_info=True,
            )
            return _DedupResult.NO_DEDUP, file_texts, file_metas, file_ids, file_spans, None
        ictx.stats.setdefault("dedup_chunks", 0)
        ictx.stats["dedup_chunks"] += len(existing_batch_ids)

        # existing_batch fuer ColBERT-Backfill zusammenbauen
        eb_texts = [t for t, cid in zip(file_texts, file_ids, strict=False) if cid in existing_ids]
        eb_metas = [m for m, cid in zip(file_metas, file_ids, strict=False) if cid in existing_ids]
        eb = _ExistingBatch(ids=existing_batch_ids, texts=eb_texts, metas=eb_metas)

    # 100% Dedup — Finalisierung wird an Caller delegiert (fuer ColBERT-Step)
    if len(existing_batch_ids) == len(file_ids):
        return _DedupResult.FULL, [], [], [], [], eb

    # Partial Dedup
    if existing_ids:
        kept = [
            (text, meta, chunk_id, span)
            for text, meta, chunk_id, span in zip(
                file_texts, file_metas, file_ids, file_spans, strict=True
            )
            if chunk_id not in existing_ids
        ]
        file_texts = [text for text, _, _, _ in kept]
        file_metas = [meta for _, meta, _, _ in kept]
        file_ids = [chunk_id for _, _, chunk_id, _ in kept]
        file_spans = [span for _, _, _, span in kept]
        return _DedupResult.PARTIAL, file_texts, file_metas, file_ids, file_spans, eb

    return _DedupResult.NO_DEDUP, file_texts, file_metas, file_ids, file_spans, None


# ---------------------------------------------------------------------------
# Late Chunking
# ---------------------------------------------------------------------------


def _should_use_late_chunking(config: dict, model: str | None) -> bool:
    """Prueft ob Late Chunking fuer diesen Lauf verwendet werden soll.

    Bedingungen:
    1. config["enrichment"]["late_chunking"] == True
    2. model ist nicht None
    3. Normalisierter Modellname in LATE_CHUNKING_SUPPORTED_MODELS
    4. Backend-Provider ist "ollama"
    """
    enrichment_cfg = config.get("enrichment") or {}
    if not enrichment_cfg.get("late_chunking", False):
        return False
    if not model:
        return False
    from kiara.backend_selector import (
        get_embedding_backend_for_model,
        normalize_model_name,
    )

    normalized = normalize_model_name(model)
    if normalized not in LATE_CHUNKING_SUPPORTED_MODELS:
        return False
    try:
        backend = get_embedding_backend_for_model(model)
        if not backend:
            return False
        return backend.get("provider", "ollama") == "ollama"
    except Exception:
        return False


def _late_chunking_file_eligible(
    full_text: str,
    original_chunks: list[str] | None,
    original_spans: list[tuple[int, int]] | None,
    file_metas: list[dict],
) -> bool:
    """Datei-spezifischer Guard fuer Late Chunking.

    Late Chunking braucht einen echten Original-Dokumenttext plus ausgerichtete
    Original-Chunks und char spans. VLM-Chunks werden ausgeschlossen, weil sie
    nicht aus full_text stammen.
    """
    if not full_text or not original_chunks or not original_spans:
        return False
    if len(original_chunks) != len(original_spans):
        return False
    return not any(meta.get("parser") == "vlm" for meta in file_metas)


def _refresh_late_chunking_snapshots(
    ictx: _IngestionContext,
    file_ids: list[str],
    file_texts: list[str],
    file_spans: list[tuple[int, int] | None],
    *,
    enrichment_applied: bool,
) -> None:
    """Haelt Original-Chunks/-Spans bei Retries und Partial-Dedup synchron.

    Beim ersten Pass werden ID-Mappings aus den nicht-enrichteten Chunks gebaut.
    Bei spaeteren Retries werden daraus die aktuell verbleibenden Original-Chunks
    anhand der verbliebenen file_ids rekonstruiert.
    """
    if not enrichment_applied:
        ictx.last_original_chunks_by_id = dict(zip(file_ids, file_texts, strict=True))
        ictx.last_original_spans_by_id = {}
        for chunk_id, span in zip(file_ids, file_spans, strict=True):
            if span is None:
                ictx.last_original_chunks = None
                ictx.last_original_spans = None
                ictx.last_original_chunks_by_id = None
                ictx.last_original_spans_by_id = None
                return
            ictx.last_original_spans_by_id[chunk_id] = span

    if not ictx.last_original_chunks_by_id or not ictx.last_original_spans_by_id:
        ictx.last_original_chunks = None
        ictx.last_original_spans = None
        return

    try:
        ictx.last_original_chunks = [
            ictx.last_original_chunks_by_id[chunk_id] for chunk_id in file_ids
        ]
        ictx.last_original_spans = [
            ictx.last_original_spans_by_id[chunk_id] for chunk_id in file_ids
        ]
    except KeyError:
        logger.warning(
            "Late-Chunking-Snapshots fuer verbleibende file_ids unvollstaendig; "
            "falle auf Standard-Embedding zurueck"
        )
        ictx.last_original_chunks = None
        ictx.last_original_spans = None
        ictx.last_original_chunks_by_id = None
        ictx.last_original_spans_by_id = None


# ---------------------------------------------------------------------------
# Embedding-Phase
# ---------------------------------------------------------------------------


def _handle_embedding_phase(
    ictx: _IngestionContext,
    file_path: str,
    file_texts: list[str],
    file_metas: list[dict],
    file_ids: list[str],
    file_log: FileLogEntry | None,
    file_retries: int,
    t0: float,
) -> tuple[_FileAction, int]:
    """Embedding-Phase mit Circuit-Breaker und Retry-Logik.

    Schreibt Chunks in VectorStore, optional Elasticsearch und Graph.
    Bei Erfolg: stats aktualisieren, Hash persistieren.
    Bei _CancelledError: Hashes flushen, CANCEL zurueckgeben.
    Bei _CATEGORICAL_ERRORS: Circuit-Breaker konsultieren.
    Bei sonstigem Exception: Content-Fehler loggen, SKIP.

    Returns:
        (_FileAction, updated_file_retries)
    """
    with _tracer.start_as_current_span("embed_batch") as span:
        span.set_attribute("chunk_count", len(file_texts))
        written_vector_ids: list[str] = []
        es_write_attempted = False
        db_associations_attempted = False

        try:
            embeddings = None
            late_chunking_used = False
            if _should_use_late_chunking(
                ictx.config, ictx.embedding_model
            ) and _late_chunking_file_eligible(
                ictx.last_full_text,
                ictx.last_original_chunks,
                ictx.last_original_spans,
                file_metas,
            ):
                embeddings = embed_texts_late(
                    ictx.config,
                    ictx.embedding_model,
                    ictx.last_full_text,
                    ictx.last_original_chunks,
                    ictx.last_original_spans,
                )
                if embeddings is not None and len(embeddings) == len(file_texts):
                    late_chunking_used = True
                else:
                    embeddings = None

            span.set_attribute("late_chunking", late_chunking_used)

            if ictx.datasource_id is not None:
                for meta in file_metas:
                    meta.setdefault("ds_ids", [])
                    if ictx.datasource_id not in meta["ds_ids"]:
                        meta["ds_ids"].append(ictx.datasource_id)

            with _tracer.start_as_current_span("write_vectors"):
                write_result = _write_embeddings(
                    file_texts,
                    file_metas,
                    file_ids,
                    ictx.store,
                    ictx.batch_size,
                    ictx.control_callback,
                    embeddings=embeddings,
                )
                written_vector_ids = _normalize_written_vector_ids(
                    write_result,
                    file_ids,
                )

            with _tracer.start_as_current_span("write_elasticsearch") as es_span:
                es_span.set_attribute(
                    "configured",
                    ictx.config.get("retrieval", {}).get("hybrid_search") == "elasticsearch",
                )
                es_write_attempted = True
                try:
                    _write_to_elasticsearch(
                        ictx.config,
                        file_texts,
                        file_metas,
                        file_ids,
                        ictx.stats,
                    )
                except _CATEGORICAL_ERRORS:
                    raise
                except Exception as es_exc:
                    # ES-Schreibfehler ist Infra-Fehler (Hybrid-Search kritisch).
                    # Als BackendUnavailableError re-raisen, damit der
                    # _CATEGORICAL_ERRORS-Handler den Circuit-Breaker triggert
                    # statt als generischer Content-Fehler zu behandeln.
                    raise BackendUnavailableError(
                        f"Elasticsearch-Schreiben fehlgeschlagen: {es_exc}"
                    ) from es_exc

            with _tracer.start_as_current_span("graph_extraction") as graph_span:
                graph_enabled = ictx.config.get("graph", {}).get("enabled", False)
                graph_span.set_attribute("enabled", graph_enabled)
                db_associations_attempted = ictx.datasource_id is not None
                _persist_graph_and_associations(ictx, file_texts, file_ids)

            ictx.stats["total_chunks"] += len(file_texts)
            _mark_file_processed(ictx, file_path)

        except _RollbackFailedError as exc:
            span.record_exception(exc)
            logger.critical("Embedding-Rollback fehlgeschlagen fuer %s: %s", file_path, exc)
            _record_error(
                ictx,
                file_log,
                file_path,
                exc,
                "vectordb_rollback",
                is_vectordb=True,
            )
            ictx.stats["abort_reason"] = f"Embedding-Rollback fehlgeschlagen: {exc}"
            _cleanup_file_hash(ictx, file_path)
            _persist_hashes_checkpoint(ictx, force=True)
            _emit_file_log(ictx, file_log, t0)
            return _FileAction.ABORT, file_retries

        except _CancelledError:
            span.set_attribute("cancelled", True)
            if not _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="cancelled_embedding_phase",
            ):
                exc = RuntimeError("Embedding-Rollback nach Cancel fehlgeschlagen")
                _record_error(
                    ictx,
                    file_log,
                    file_path,
                    exc,
                    "vectordb_rollback",
                    is_vectordb=True,
                )
                ictx.stats["abort_reason"] = str(exc)
                _cleanup_file_hash(ictx, file_path)
                _persist_hashes_checkpoint(ictx, force=True)
                _emit_file_log(ictx, file_log, t0)
                return _FileAction.ABORT, file_retries
            _cleanup_file_hash(ictx, file_path)
            _persist_hashes_checkpoint(ictx, force=True)
            _emit_file_log(ictx, file_log, t0)
            return _FileAction.CANCEL, file_retries

        except VectorDBConnectionError as exc:
            span.record_exception(exc)
            label = "VectorDB-Fehler"
            logger.error(
                "%s bei '%s': %s (consecutive: %d/%d)",
                label,
                file_path,
                redact_secrets_in_text(str(exc)),
                ictx.consecutive_errors + 1,
                INGESTION_MAX_CONSECUTIVE_ERRORS,
            )
            _record_error(
                ictx,
                file_log,
                file_path,
                exc,
                "vectordb",
                is_vectordb=True,
            )
            if not _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="vectordb_error",
            ):
                ictx.stats["abort_reason"] = (
                    f"Abbruch: Rollback nach VectorDB-Fehler fehlgeschlagen: {exc}"
                )
                _cleanup_file_hash(ictx, file_path)
                _persist_hashes_checkpoint(ictx, force=True)
                _emit_file_log(ictx, file_log, t0)
                return _FileAction.ABORT, file_retries

            cb_result, file_retries = _handle_circuit_breaker(
                ictx,
                file_path,
                file_log,
                error_phase="vectordb",
                error_description=f"{label}: {exc}",
                is_infra=True,
                file_retries=file_retries,
                t0=t0,
            )

            # Hash-Cleanup auch bei RETRY: hash_store muss konsistent zu Qdrant
            # bleiben (rollback hat Vektoren entfernt). Pipeline-Crash zwischen
            # RETRY-Wait und Re-Embed darf keine Resume-Inkonsistenz hinterlassen.
            _cleanup_file_hash(ictx, file_path)
            if cb_result == _CircuitBreakerResult.RETRY:
                return _FileAction.RETRY, file_retries
            if cb_result == _CircuitBreakerResult.ABORT:
                return _FileAction.ABORT, file_retries
            if cb_result == _CircuitBreakerResult.CANCEL:
                return _FileAction.CANCEL, file_retries
            return _FileAction.SKIP, file_retries

        except _CATEGORICAL_ERRORS as exc:
            span.record_exception(exc)
            is_backend_error = isinstance(exc, BackendUnavailableError)
            label = "Backend-Fehler" if is_backend_error else "Netzwerk-/Timeout-Fehler"

            logger.error(
                "%s bei '%s': %s (consecutive: %d/%d)",
                label,
                file_path,
                redact_secrets_in_text(str(exc)),
                ictx.consecutive_errors + 1,
                INGESTION_MAX_CONSECUTIVE_ERRORS,
            )
            _record_error(ictx, file_log, file_path, exc, "embedding")
            if not _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="categorical_embedding_error",
            ):
                ictx.stats["abort_reason"] = (
                    f"Abbruch: Rollback nach Embedding-Fehler fehlgeschlagen: {exc}"
                )
                _cleanup_file_hash(ictx, file_path)
                _persist_hashes_checkpoint(ictx, force=True)
                _emit_file_log(ictx, file_log, t0)
                return _FileAction.ABORT, file_retries

            cb_result, file_retries = _handle_circuit_breaker(
                ictx,
                file_path,
                file_log,
                error_phase="embedding",
                error_description=f"{label}: {exc}",
                is_infra=True,
                file_retries=file_retries,
                t0=t0,
            )

            # Hash-Cleanup auch bei RETRY: hash_store muss konsistent zu Qdrant
            # bleiben (rollback hat Vektoren entfernt). Pipeline-Crash zwischen
            # RETRY-Wait und Re-Embed darf keine Resume-Inkonsistenz hinterlassen.
            _cleanup_file_hash(ictx, file_path)
            if cb_result == _CircuitBreakerResult.RETRY:
                return _FileAction.RETRY, file_retries
            if cb_result == _CircuitBreakerResult.ABORT:
                return _FileAction.ABORT, file_retries
            if cb_result == _CircuitBreakerResult.CANCEL:
                return _FileAction.CANCEL, file_retries
            return _FileAction.SKIP, file_retries

        except Exception as exc:
            span.record_exception(exc)
            logger.error(
                "Embedding fehlgeschlagen fuer %s: %s",
                file_path,
                exc,
                exc_info=True,
            )
            _record_error(ictx, file_log, file_path, exc, "embedding")
            if not _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="generic_embedding_error",
            ):
                ictx.stats["abort_reason"] = (
                    f"Abbruch: Rollback nach Embedding-Fehler fehlgeschlagen: {exc}"
                )
                _cleanup_file_hash(ictx, file_path)
                _persist_hashes_checkpoint(ictx, force=True)
                _emit_file_log(ictx, file_log, t0)
                return _FileAction.ABORT, file_retries

            ictx.consecutive_errors = 0
            _cleanup_file_hash(ictx, file_path)
            return _FileAction.SKIP, file_retries

        # consecutive_errors-Reset wandert in den Aufrufer (process_one_file),
        # damit er erst nach vollstaendigem File-Commit (Dense+ColBERT+Stats)
        # greift. Reset hier waere zu frueh: ColBERT kann nach Dense-Erfolg
        # noch scheitern, und der nachfolgende Counter haette dann faelschlich
        # bei 0 statt beim akkumulierten Vorwert begonnen (Circuit-Breaker-Lock-Bypass).
        return _FileAction.SUCCESS, file_retries


# ---------------------------------------------------------------------------
# Enrichment-Hilfsfunktionen (Phase B/C)
# ---------------------------------------------------------------------------


def _should_enrich(config: dict, ds_overrides: dict | None) -> bool:
    """Prueft ob Enrichment fuer diesen Ingestion-Lauf aktiv ist."""
    if not config.get("enrichment", {}).get("enabled", False):
        return False
    if ds_overrides is None:
        return False
    return bool(ds_overrides.get("contextual_retrieval"))


def _get_enrichment_prompt(ds_overrides: dict | None) -> tuple[str, str]:
    """Laedt den Enrichment-Prompt fuer diese Datasource.

    Returns:
        Tuple (template_content, content_hash).
    """
    import hashlib

    from kiara.shared.db import db_query

    prompt_id = (ds_overrides or {}).get("enrichment_prompt_id")
    if prompt_id is not None:
        with db_query() as db:
            row = db.execute(
                "SELECT content, chain_type FROM prompt_templates "
                "WHERE id = %s AND is_active = true",
                (prompt_id,),
            ).fetchone()
        if not row:
            raise ValueError(f"Enrichment-Template {prompt_id} nicht gefunden oder inaktiv")
        if row["chain_type"] != "enrichment":
            raise ValueError(
                f"Template {prompt_id} hat chain_type='{row['chain_type']}', erwartet 'enrichment'"
            )
        content = row["content"]
    else:
        with db_query() as db:
            row = db.execute(
                "SELECT content FROM prompt_templates "
                "WHERE name = 'enrichment_default' AND is_active = true",
            ).fetchone()
        if not row:
            raise ValueError("System-Template 'enrichment_default' nicht gefunden")
        content = row["content"]

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return content, content_hash


def _generate_with_audit(backend, backend_dict, config, system_prompt, user_prompt):
    """Ruft backend.generate() auf und schreibt Cloud-Audit-Eintrag (Bug #596).

    Auditiert system-initiierte Cloud-Enrichment-Calls analog zu Chat/MCP
    (user_id=NULL, eigene correlation_id pro Call).
    """
    import time as _time
    import uuid as _uuid

    t_start = _time.time()
    result = backend.generate(system_prompt, user_prompt)
    duration_ms = int((_time.time() - t_start) * 1000)
    try:
        from kiara.cloud.audit import AuditLogger

        correlation_id = str(_uuid.uuid4())
        usage = getattr(backend, "last_usage", None) or {}
        anon_question = getattr(backend, "last_anon_question", "") or "[nicht anonymisiert]"
        anon_answer = getattr(backend, "last_anon_answer", "") or "[nicht anonymisiert]"
        entities_anonymized = int(getattr(backend, "last_entities_anonymized", 0) or 0)

        AuditLogger(config).log_request(
            user_id=None,
            question_anonymized=anon_question,
            answer_anonymized=anon_answer,
            provider=backend_dict.get("provider", ""),
            model=backend_dict.get("model", ""),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("response_tokens", usage.get("completion_tokens", 0)),
            entities_anonymized=entities_anonymized,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            context_sources=[{"source": "enrichment"}],
        )

        try:
            from kiara.anonymization.audit import log_anonymization_event

            stats = getattr(backend, "anonymizer_stats", None) or {
                "total_entities": entities_anonymized,
                "entity_types": {},
            }
            log_anonymization_event(
                user_id=None,
                session_id=correlation_id,
                question=user_prompt,
                entity_count=stats.get("total_entities", entities_anonymized),
                entity_types=stats.get("entity_types", {}),
                route="enrichment:cloud",
                cloud_provider=backend_dict.get("provider", ""),
                cloud_model=backend_dict.get("model", ""),
                response_time_ms=duration_ms,
                correlation_id=correlation_id,
            )
        except Exception as exc:
            logger.warning(
                "Anonymisierungs-Audit fuer Enrichment fehlgeschlagen: %s",
                exc,
                exc_info=True,
            )
    except Exception as exc:
        logger.warning("Cloud-Audit-Log fuer Enrichment fehlgeschlagen: %s", exc, exc_info=True)
    return result


def _get_enrichment_llm(config, ds_overrides):
    """Erzeugt die LLM-Aufloesung fuer das Enrichment.

    Returns:
        Tuple (llm_generate_fn, llm_generate_factory, allow_parallel).
    """
    from kiara.generation.backends.factory import create_backend, wrap_with_anonymization

    overrides = ds_overrides or {}
    backend_id = overrides.get("enrichment_backend_id")
    restriction_level = overrides.get("restriction_level", "anonymize")

    # Backend aufloesen — kein stiller Fallback, Backend muss explizit konfiguriert sein
    if backend_id is None:
        raise ValueError(
            "Kein Enrichment-Backend konfiguriert. "
            "Bitte in den Datenquellen-Einstellungen ein Backend waehlen."
        )

    from kiara.backend_selector import get_backend_by_id

    backend_dict = get_backend_by_id(backend_id)
    if backend_dict is None:
        raise ValueError(f"Enrichment-Backend {backend_id} nicht gefunden oder inaktiv")
    cat = backend_dict.get("category", "")
    if cat not in ("llm", "cloud"):
        raise ValueError(
            f"Enrichment-Backend {backend_id} hat Kategorie '{cat}', erwartet 'llm' oder 'cloud'"
        )
    backend = create_backend(config, backend_id)
    resolved_backend_id = backend_id

    # Cloud-Guard: Anonymisierung
    is_cloud = backend_dict.get("provider", "ollama") != "ollama"
    if is_cloud:
        backend = wrap_with_anonymization(
            backend,
            config,
            restriction_levels=[restriction_level] if restriction_level else None,
        )

    def llm_generate_fn(system_prompt: str, user_prompt: str) -> str:
        if is_cloud:
            return _generate_with_audit(
                backend,
                backend_dict,
                config,
                system_prompt,
                user_prompt,
            )
        return backend.generate(system_prompt, user_prompt)

    # Factory fuer isolierte Parallelisierung (nur lokal, nicht-anonymisiert)
    allow_parallel = not is_cloud
    llm_generate_factory = None
    if allow_parallel and resolved_backend_id is not None:
        _pinned_config = config
        _pinned_id = resolved_backend_id

        def llm_generate_factory():
            b = create_backend(_pinned_config, _pinned_id)
            return lambda sys, usr: b.generate(sys, usr)

    return llm_generate_fn, llm_generate_factory, allow_parallel


# ---------------------------------------------------------------------------
# File-Finalisierung + Pipeline-Verarbeitung
# ---------------------------------------------------------------------------


def _finalize_file(ictx: _IngestionContext, file_log: FileLogEntry | None, t0: float) -> None:
    """Finalisiert File-Logging und Progress nach Embedding-Phase.

    Wird NUR nach der Embedding-Phase aufgerufen (Erfolg oder Content-Error).
    NICHT nach Loading-Phase (dort ruft _handle_loading_result Callbacks selbst auf).
    NICHT nach ABORT/CANCEL (dort ruft der Handler file_log_callback selbst auf).
    """
    _emit_file_log(ictx, file_log, t0)
    if ictx.progress_callback:
        ictx.progress_callback(ictx.skipped + ictx.stats["processed_files"], ictx.total)


def _run_enrichment(
    ictx: _IngestionContext,
    file_path: str,
    file_texts: list[str],
    file_metas: list[dict],
    full_text: str,
) -> tuple[list[str], list[dict], bool]:
    """Fuehrt die Enrichment-Phase fuer eine Datei durch.

    Returns:
        (enriched_texts, enriched_metas, enrichment_applied)
        Bei Fehler werden die Original-Texte/Metas unveraendert zurueckgegeben
        und enrichment_applied ist False.
    """
    with _tracer.start_as_current_span("enrich_chunks") as enrich_span:
        enrich_span.set_attribute("chunk_count", len(file_texts))
        file_texts_before = list(file_texts)
        file_metas_before = [dict(m) for m in file_metas]
        try:
            from kiara.ingestion.enrichment import enrich_chunks_batch

            prompt_template, prompt_hash = _get_enrichment_prompt(ictx.ds_overrides)
            llm_fn, llm_factory, allow_par = _get_enrichment_llm(ictx.config, ictx.ds_overrides)

            _ds_name = ""
            if ictx.datasource_id:
                from kiara.shared.db import db_query

                with db_query() as db:
                    _row = db.execute(
                        "SELECT name FROM datasources WHERE id = %s",
                        (ictx.datasource_id,),
                    ).fetchone()
                    _ds_name = _row["name"] if _row else ""

            file_metadata = {
                "file_type": file_metas[0].get("file_type", "") if file_metas else "",
                "source": file_path,
                "datasource_name": _ds_name,
            }

            _max_ctx = ictx.config.get("enrichment", {}).get(
                "max_context_chars", ENRICHMENT_MAX_CONTEXT_CHARS
            )
            _max_workers = ictx.config.get("enrichment", {}).get("max_workers", 2)

            enriched_texts, enrich_stats = enrich_chunks_batch(
                chunk_texts=file_texts,
                full_text=full_text,
                llm_generate_fn=llm_fn,
                prompt_template=prompt_template,
                max_context_chars=_max_ctx,
                file_metadata=file_metadata,
                max_workers=_max_workers,
                llm_generate_factory=llm_factory,
                allow_parallel=allow_par,
            )

            enrich_span.set_attribute("enriched_count", enrich_stats.get("enriched", 0))
            enrich_span.set_attribute("skipped_count", enrich_stats.get("skipped", 0))

            ictx.stats.setdefault("enrichment_chunks", 0)
            ictx.stats.setdefault("enrichment_skipped", 0)
            ictx.stats.setdefault("enrichment_errors", 0)
            ictx.stats.setdefault("enrichment_time_ms", 0)
            ictx.stats["enrichment_chunks"] += enrich_stats.get("enriched", 0)
            ictx.stats["enrichment_skipped"] += enrich_stats.get("skipped", 0)
            ictx.stats["enrichment_errors"] += enrich_stats.get("errors", 0)
            ictx.stats["enrichment_time_ms"] += enrich_stats.get("enrichment_time_ms", 0)

            enriched_metas = [dict(m) for m in file_metas]
            for i, meta in enumerate(enriched_metas):
                changed = i < len(enriched_texts) and enriched_texts[i] != file_texts_before[i]
                if changed:
                    meta["contextual_prefix"] = True
                    meta["enrichment_prompt_hash"] = prompt_hash

            return enriched_texts, enriched_metas, True

        except _CancelledError:
            raise
        except Exception:
            skipped_count = sum(
                1 for text in file_texts_before if len(text) < ENRICHMENT_MIN_CHUNK_CHARS
            )
            error_count = len(file_texts_before) - skipped_count
            ictx.stats.setdefault("enrichment_skipped", 0)
            ictx.stats.setdefault("enrichment_errors", 0)
            ictx.stats["enrichment_skipped"] += skipped_count
            ictx.stats["enrichment_errors"] += error_count
            logger.warning(
                "Enrichment fehlgeschlagen fuer %s — verwende Original-Texte",
                file_path,
                exc_info=True,
            )
            return file_texts_before, file_metas_before, False


def _process_file_in_pipeline(ictx: _IngestionContext, file_path: str) -> bool:
    """Verarbeitet eine Datei komplett (inkl. Retry-Schleife).

    Returns:
        True → naechste Datei verarbeiten.
        False → Pipeline abbrechen (Cancel/Abort).
    """
    with _tracer.start_as_current_span("ingest_file") as span:
        span.set_attribute("file_path", str(file_path))
        span.set_attribute("datasource_id", ictx.datasource_id or 0)
        span.set_attribute("job_id", _get_current_job_id() or 0)
        try:
            span.set_attribute("file_size", os.path.getsize(file_path))
        except OSError:
            span.set_attribute("file_size", 0)
        span.set_attribute("file_type", os.path.splitext(str(file_path))[1].lower())

        if ictx.control_callback:
            signal = ictx.control_callback()
            if signal in ("cancel", "shutdown"):
                span.set_attribute("final_status", "cancelled_before_start")
                return False

        file_log = None
        if ictx.file_log_callback:
            p = Path(file_path)
            try:
                fsize = p.stat().st_size
            except OSError:
                fsize = 0
            file_log = FileLogEntry(
                file_path=file_path,
                file_name=p.name,
                file_size=fsize,
                file_ext=p.suffix.lower(),
            )
        t0 = time.monotonic()

        if ictx.file_start_callback:
            ictx.file_start_callback(Path(file_path).name, file_path)

        file_retries = 0
        skip_loading = False
        errors_before = ictx.stats["errors"]
        enrichment_applied = False

        # File-lokale Idempotenz-Marker: trennen "Stats bereits gezählt"
        # und "Hash aktuell persistiert". Noetig fuer ColBERT-RETRY nach
        # erfolgreichem Dense-Write: _handle_embedding_phase hat bereits
        # _mark_file_processed() gerufen; ein anschliessender
        # ColBERT-VectorDBConnectionError→RETRY räumt Hash/Pending-Hash
        # via _cleanup_file_hash auf und reruns als FULL-Dedup, ohne dass
        # processed_files ein zweites Mal erhöht werden darf.
        stats_counted = False

        ictx.last_full_text = ""
        ictx.last_original_chunks = None
        ictx.last_original_spans = None
        ictx.last_original_chunks_by_id = None
        ictx.last_original_spans_by_id = None

        while True:
            if not skip_loading:
                errors_before = ictx.stats["errors"]
                file_texts, file_metas, file_ids, full_text, file_spans = _process_single_file(
                    file_path,
                    ictx.config,
                    ictx.ds_overrides,
                    ictx.datasource_id,
                    ictx.ctx,
                    ictx.stats,
                    file_log=file_log,
                )
                enrichment_applied = False
                ictx.last_full_text = full_text
            skip_loading = False

            if not file_texts:
                action, file_retries = _handle_loading_result(
                    ictx,
                    file_path,
                    file_log,
                    errors_before,
                    file_retries,
                    t0,
                )
                if action == _FileAction.RETRY:
                    span.set_attribute("final_status", "retry_loading")
                    continue
                if action == _FileAction.ABORT:
                    span.set_attribute("final_status", "aborted_loading")
                    return False
                if action == _FileAction.CANCEL:
                    span.set_attribute("final_status", "cancelled_loading")
                    return False
                span.set_attribute("final_status", "skipped_loading")
                return True

            try:
                dedup_result, file_texts, file_metas, file_ids, file_spans, existing_batch = (
                    _handle_chunk_dedup(
                        ictx,
                        file_path,
                        file_texts,
                        file_metas,
                        file_ids,
                        file_spans,
                        file_log,
                        t0,
                    )
                )
            except VectorDBConnectionError as exc:
                span.record_exception(exc)
                label = "VectorDB-Fehler (Dedup)"
                logger.error(
                    "%s bei '%s': %s (consecutive: %d/%d)",
                    label,
                    file_path,
                    exc,
                    ictx.consecutive_errors + 1,
                    INGESTION_MAX_CONSECUTIVE_ERRORS,
                )
                _record_error(
                    ictx,
                    file_log,
                    file_path,
                    exc,
                    "vectordb",
                    is_vectordb=True,
                )
                cb_result, file_retries = _handle_circuit_breaker(
                    ictx,
                    file_path,
                    file_log,
                    error_phase="vectordb",
                    error_description=f"{label}: {exc}",
                    is_infra=True,
                    file_retries=file_retries,
                    t0=t0,
                )
                if cb_result == _CircuitBreakerResult.RETRY:
                    span.set_attribute("final_status", "retry_dedup")
                    skip_loading = True
                    continue
                _cleanup_file_hash(ictx, file_path)
                if cb_result == _CircuitBreakerResult.ABORT:
                    span.set_attribute("final_status", "aborted_dedup")
                    return False
                if cb_result == _CircuitBreakerResult.CANCEL:
                    span.set_attribute("final_status", "cancelled_dedup")
                    return False
                # SKIP
                span.set_attribute("final_status", "skipped_dedup")
                ictx.skipped += 1
                break

            # FULL-Dedup: ColBERT-Backfill, dann finalisieren
            if dedup_result == _DedupResult.FULL:
                if existing_batch and _colbert_enabled_for_run(ictx):
                    cb_action, file_retries = _handle_colbert_step(
                        ictx,
                        chunk_ids=existing_batch.ids,
                        chunk_texts=existing_batch.texts,
                        chunk_metas=existing_batch.metas,
                        file_path=file_path,
                        file_log=file_log,
                        file_retries=file_retries,
                        t0=t0,
                    )
                    signal = _apply_colbert_action(cb_action, span, "colbert_dedup")
                    if signal == "abort":
                        return False
                    if signal == "retry":
                        skip_loading = True
                        continue
                    if signal == "skip":
                        break
                # Finalisierung (war vorher in _handle_chunk_dedup)
                ictx.consecutive_errors = 0
                if stats_counted:
                    # ColBERT-RETRY nach bereits gezähltem Dense-Erfolg:
                    # nur Hash re-persistieren, nicht nochmal processed_files hochzaehlen
                    ictx.pending_hashes[file_path] = ictx.file_hashes[file_path]
                    _persist_hashes_checkpoint(ictx)
                else:
                    _mark_file_processed(ictx, file_path)
                    stats_counted = True
                _emit_file_log(ictx, file_log, t0)
                if ictx.progress_callback:
                    ictx.progress_callback(ictx.skipped + ictx.stats["processed_files"], ictx.total)
                span.set_attribute("final_status", "deduped")
                return True

            _refresh_late_chunking_snapshots(
                ictx,
                file_ids,
                file_texts,
                file_spans,
                enrichment_applied=enrichment_applied,
            )

            # --- ENRICHMENT-PHASE ---
            if _should_enrich(ictx.config, ictx.ds_overrides) and not enrichment_applied:
                file_texts, file_metas, enrichment_applied = _run_enrichment(
                    ictx, file_path, file_texts, file_metas, full_text
                )

            action, file_retries = _handle_embedding_phase(
                ictx,
                file_path,
                file_texts,
                file_metas,
                file_ids,
                file_log,
                file_retries,
                t0,
            )

            if action == _FileAction.RETRY:
                span.set_attribute("final_status", "retry_embedding")
                skip_loading = True
                continue
            if action == _FileAction.ABORT:
                span.set_attribute("final_status", "aborted_embedding")
                return False
            if action == _FileAction.CANCEL:
                span.set_attribute("final_status", "cancelled_embedding")
                return False
            if action == _FileAction.SKIP:
                span.set_attribute("final_status", "skipped_embedding")
                ictx.skipped += 1
                break

            # Dense-Embedding war erfolgreich (Helper ruft _mark_file_processed)
            stats_counted = True

            # ColBERT-Step fuer neue Chunks (nach erfolgreichem Dense-Embedding)
            if _colbert_enabled_for_run(ictx):
                # Sammle alle Chunks: neue + existing_batch
                all_cb_ids = list(file_ids)
                all_cb_texts = list(file_texts)
                all_cb_metas = list(file_metas)
                if existing_batch:
                    all_cb_ids.extend(existing_batch.ids)
                    all_cb_texts.extend(existing_batch.texts)
                    all_cb_metas.extend(existing_batch.metas)

                cb_action, file_retries = _handle_colbert_step(
                    ictx,
                    chunk_ids=all_cb_ids,
                    chunk_texts=all_cb_texts,
                    chunk_metas=all_cb_metas,
                    file_path=file_path,
                    file_log=file_log,
                    file_retries=file_retries,
                    t0=t0,
                )
                signal = _apply_colbert_action(cb_action, span, "colbert")
                if signal == "abort":
                    return False
                if signal == "retry":
                    # ColBERT-Fehlerpfad hat Hash/Pending-Hash bereits entfernt;
                    # der FULL-Dedup-Retry muss den Hash wieder persistieren,
                    # aber processed_files darf nicht doppelt zaehlen.
                    skip_loading = True
                    continue
                if signal == "skip":
                    break

            # Vollstaendiger File-Commit: Dense+ColBERT(+Stats) erfolgreich
            ictx.consecutive_errors = 0
            span.set_attribute("final_status", "success")
            break

        _finalize_file(ictx, file_log, t0)
        return True
