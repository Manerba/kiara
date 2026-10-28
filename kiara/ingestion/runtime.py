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
from kiara.exceptions import ChunkingContentError, VectorDBConnectionError
from kiara.ingestion._file_utils import (
    ChunkDatasources,
    HashStore,
    _CancelledError,
)
from kiara.ingestion.chunker import chunk_document, chunk_document_with_full_text
from kiara.ingestion.loader import IMAGE_FORMATS, describe_image, load_file
from kiara.ingestion.processing_revision import stamp_document_versions
from kiara.retrieval.embedding import embed_texts_late
from kiara.retrieval.embedding_roles import SEARCH_DOCUMENT
from kiara.retrieval.index_compatibility import (
    IndexCompatibilityError,
    IndexContractUnavailableError,
)
from kiara.retrieval.vectorstore import VectorStore
from kiara.shared.constants import (
    BACKEND_WAIT_INTERVAL_SECS,
    DEFAULT_QDRANT_URL,
    ENRICHMENT_MAX_CONTEXT_CHARS,
    ENRICHMENT_MIN_CHUNK_CHARS,
    HASH_CHECKPOINT_INTERVAL,
    INDEX_CONTRACT_RECOVERY_POLL_SECS_DEFAULT,
    INGESTION_MAX_CONSECUTIVE_ERRORS,
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
    file_completion_callback: Callable[[str], None]
    file_preparation_callback: Callable[[str, list[dict], list[str]], bool] | None = None
    file_loading_context: Callable | None = None
    consecutive_errors: int = 0
    skipped: int = 0
    total: int = 0
    progress_callback: Callable | None = None
    control_callback: Callable | None = None
    file_log_callback: Callable | None = None
    file_start_callback: Callable | None = None
    status_callback: Callable | None = None
    activity_callback: Callable | None = None
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


def _abort_index_compatibility(
    ictx: _IngestionContext,
    file_path: str,
    file_log: FileLogEntry | None,
    exc: IndexCompatibilityError,
    t0: float,
    *,
    consumer: str,
    is_colbert: bool = False,
) -> None:
    """Persistiert den gemeinsamen, nicht degradierbaren Guard-ABORT-Vertrag."""
    diagnostic = exc.to_dict()
    logger.error(
        "Index-Kompatibilitaetsguard blockiert %s fuer %s: %s",
        consumer,
        file_path,
        redact_secrets_in_text(str(exc)),
    )
    _record_error(
        ictx,
        file_log,
        file_path,
        exc,
        "index_compatibility",
        is_colbert=is_colbert,
    )
    error_detail = ictx.stats["error_details"][-1]
    error_detail.update(
        {
            "reason_code": exc.reason_code,
            "operation": exc.operation,
            "consumer": consumer,
            "diagnostic": diagnostic,
        }
    )
    ictx.stats["index_compatibility_error"] = diagnostic
    if exc.rollback_diagnostics:
        rollback_diagnostics = list(exc.rollback_diagnostics)
        error_detail["rollback_diagnostics"] = rollback_diagnostics
        ictx.stats["index_compatibility_rollback_errors"] = rollback_diagnostics
        ictx.stats["abort_reason"] = (
            "Index-Kompatibilitaetsguard und Partial-Rollback fehlgeschlagen: "
            f"{exc.reason_code} ({exc.detail})"
        )
    else:
        ictx.stats["abort_reason"] = (
            f"Index-Kompatibilitaetsguard: {exc.reason_code} ({exc.detail})"
        )
    ictx.pending_hashes.pop(file_path, None)
    _cleanup_file_hash(ictx, file_path)
    _persist_hashes_checkpoint(ictx, force=True)
    _emit_file_log(ictx, file_log, t0)


def _mark_file_processed(ictx, file_path, document_ids):
    """Bestaetigt Hash und Statistik atomar, bevor Erfolg gezaehlt wird."""
    try:
        from kiara.ingestion.document_publication import stage_document_versions

        completed = False

        def complete(versions):
            nonlocal completed
            stage_document_versions(versions)
            ictx.file_completion_callback(file_path)
            completed = True

        ictx.store.finalize_document_versions(
            document_ids,
            publish_search=lambda texts, metas, ids: _publish_search_snapshot(
                ictx, texts, metas, ids
            ),
            on_complete=complete,
        )
        if not completed:
            raise RuntimeError("Dokumentabschluss wurde nicht bestaetigt")
    except Exception:
        # Bereits geschriebene Chunks bleiben als Wiederanlaufbasis erhalten.
        # Ein neuer Lauf durchlaeuft auch bei FULL-Dedup erneut den Abschluss.
        _cleanup_file_hash(ictx, file_path)
        raise
    ictx.stats["processed_files"] += 1


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


def _rollback_elasticsearch_write(
    ictx, file_path: str, chunk_ids: list[str], *, context: str
) -> int:
    """Loescht moegliche ES-Seiteneffekte fuer die aktuelle Datei."""
    from kiara.retrieval.elasticsearch_indices import document_address
    from kiara.retrieval.elasticsearch_service import delete_by_ids, model_uses_elasticsearch

    if not model_uses_elasticsearch(ictx.config, ictx.embedding_model):
        return 0
    deleted = delete_by_ids(
        ictx.config,
        [document_address(ictx.config, ictx.datasource_id, cid) for cid in chunk_ids],
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
        conn.execute(
            "DELETE FROM chunk_datasources WHERE datasource_id = %s "
            "AND embedding_model = %s AND chunk_id = ANY(%s)",
            (ictx.datasource_id, ictx.embedding_model, ids),
        )
        graph_chunk_ids = ictx.chunk_ds.filter_unreferenced_chunk_ids(ids, conn=conn)
        if graph_chunk_ids:
            from kiara.graph.store import GraphStore

            GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)

    logger.warning("DB-Rollback (%s): %d Chunk-Zuordnungen geloescht", context, len(ids))
    return len(ids)


def _reconcile_rollback_memberships(ictx, chunk_ids, *, exclude_owner, verify_absent=False):
    """Bounded live reconciliation without holding a DB lock over backend calls.

    Before PG cleanup the target excludes only the rolling-back datasource.
    Afterwards it is the complete current SoT, including concurrent changes.
    Acknowledged writes alone are not proof: reread both payloads and PG.
    Missing shared vectors/documents cannot be reconstructed here: fail closed.
    """
    from kiara.retrieval import elasticsearch_service as es_svc
    from kiara.retrieval.datasource_metadata import same_ds_ids
    from kiara.retrieval.elasticsearch_indices import document_address

    def read_memberships():
        mapping = ictx.chunk_ds.get_datasource_ids_for_chunks(chunk_ids, ictx.embedding_model)
        return {
            cid: sorted(
                ds for ds in mapping.get(cid, []) if not exclude_owner or ds != ictx.datasource_id
            )
            for cid in chunk_ids
        }

    use_es = es_svc.model_uses_elasticsearch(ictx.config, ictx.embedding_model)
    attempts = ictx.config.get("ingestion", {}).get("ds_ids_sync_attempts", 3)
    for _ in range(attempts):
        expected = read_memberships()
        shared = {cid: ds_ids for cid, ds_ids in expected.items() if ds_ids}
        orphan_ids = [cid for cid, ds_ids in expected.items() if not ds_ids]
        if shared:
            actual = ictx.store.get_ds_ids(list(shared))
            if read_memberships() != expected:
                continue
            changed = {
                cid: ids for cid, ids in shared.items() if not same_ds_ids(actual.get(cid), ids)
            }
            if changed:
                ictx.store.update_ds_ids(changed, wait=True)
            if read_memberships() != expected:
                continue
            es_shared = {
                document_address(ictx.config, ds, cid): owners
                for cid, owners in shared.items()
                for ds in owners
            }
            if use_es:
                es_svc.update_ds_ids(ictx.config, es_shared, refresh=False, skip_unchanged=True)
            actual_qdrant = ictx.store.get_ds_ids(list(shared))
            if any(
                not same_ds_ids(actual_qdrant.get(cid), owners) for cid, owners in shared.items()
            ):
                continue
            if use_es:
                actual_es = es_svc.get_ds_ids(ictx.config, list(es_shared))
                if any(
                    not same_ds_ids(actual_es.get(address), owners)
                    for address, owners in es_shared.items()
                ):
                    continue
        if verify_absent and orphan_ids:
            remaining = ictx.store.get_documents_by_ids(orphan_ids, include=[])["ids"]
            if remaining or (
                use_es
                and es_svc.get_ds_ids(
                    ictx.config,
                    [document_address(ictx.config, ictx.datasource_id, cid) for cid in orphan_ids],
                )
            ):
                raise BackendUnavailableError(
                    "Rollback hinterlaesst Chunks ohne PostgreSQL-Zuordnung"
                )
        if read_memberships() == expected:
            return orphan_ids
    raise BackendUnavailableError("DS-Zuordnungen/Payloads aendern sich waehrend des Rollbacks")


def _rollback_embedding_persistence(
    ictx,
    file_path: str,
    vector_ids: list[str],
    *,
    rollback_elasticsearch: bool,
    rollback_db_associations: bool = False,
    context: str,
    compatibility_error: IndexCompatibilityError | None = None,
) -> bool | IndexCompatibilityError:
    """Rollback extern-first; PostgreSQL-Zuordnungen bleiben letzter Retry-Anker."""
    rollback_contract_error: IndexCompatibilityError | None = None

    def record_rollback_failure(phase: str, error: Exception) -> None:
        nonlocal rollback_contract_error
        if compatibility_error is not None:
            compatibility_error.add_rollback_diagnostic(phase=phase, error=error)
        elif isinstance(error, IndexCompatibilityError):
            if rollback_contract_error is None:
                rollback_contract_error = error
            else:
                rollback_contract_error.add_rollback_diagnostic(
                    phase=phase,
                    error=error,
                )

    def rollback_failed(phase: str, exc: Exception, label: str) -> bool | IndexCompatibilityError:
        record_rollback_failure(phase, exc)
        logger.critical(
            "%s fehlgeschlagen (%s, %s): %s",
            label,
            context,
            file_path,
            exc,
            exc_info=True,
        )
        ictx.stats.setdefault("rollback_errors", 0)
        ictx.stats["rollback_errors"] += 1
        return rollback_contract_error or False

    # Externe, schwerer rekonstruierbare Schichten zuerst. Schlaegt Qdrant
    # fehl, bleiben DB und ES bewusst als Retry-Anker erhalten. Schlaegt ES
    # nach erfolgreichem Qdrant-Delete fehl, bleibt wenigstens die PostgreSQL-
    # SoT erhalten; der Dedup-Pfad erkennt den fehlenden Vektor und re-embedded.
    orphan_ids = vector_ids
    if vector_ids and ictx.datasource_id is not None:
        try:
            orphan_ids = _reconcile_rollback_memberships(
                ictx,
                vector_ids,
                exclude_owner=rollback_db_associations,
            )
        except Exception as exc:
            return rollback_failed(f"{context}:shared_memberships", exc, "Shared-Chunk-Rollback")
    if vector_ids:
        try:
            if orphan_ids and ictx.datasource_id is not None:
                # An ID classified as orphan can become shared before deletion.
                # Keep its retry anchors if either current layer has foreign owners.
                current = ictx.chunk_ds.get_datasource_ids_for_chunks(
                    orphan_ids, ictx.embedding_model
                )
                if any(ds != ictx.datasource_id for owners in current.values() for ds in owners):
                    raise BackendUnavailableError(
                        "Rollback-Orphans sind inzwischen mehrfach zugeordnet"
                    )
                documents = ictx.store.get_documents_by_ids(orphan_ids, include=["metadatas"])
                if len(documents["ids"]) != len(documents["metadatas"]):
                    raise BackendUnavailableError("Rollback-Orphan-Abfrage unvollstaendig")
                for metadata in documents["metadatas"]:
                    owners = metadata.get("ds_ids")
                    if not isinstance(owners, list) or any(
                        ds != ictx.datasource_id for ds in owners
                    ):
                        raise BackendUnavailableError(
                            "Rollback-Orphan hat fremde/ungeklaerte Payload-Zuordnungen"
                        )
            deleted = _rollback_vector_writes(
                ictx.store,
                orphan_ids,
                context=context,
            )
            ictx.stats.setdefault("vectordb_rollback_chunks", 0)
            ictx.stats["vectordb_rollback_chunks"] += deleted
        except Exception as exc:
            return rollback_failed(
                f"{context}:vectordb",
                exc,
                "VectorDB-Rollback",
            )

    if rollback_elasticsearch:
        try:
            from kiara.retrieval.elasticsearch_service import is_configured

            if orphan_ids and ictx.datasource_id is not None and is_configured(ictx.config):
                # Qdrant cleanup can outlive the initial orphan classification.
                # Preserve ES and PG retry anchors if PG now has foreign owners,
                # even when ES still contains only the rolling-back datasource.
                current = ictx.chunk_ds.get_datasource_ids_for_chunks(
                    orphan_ids, ictx.embedding_model
                )
                if any(ds != ictx.datasource_id for owners in current.values() for ds in owners):
                    raise BackendUnavailableError(
                        "ES-Rollback-Orphans sind inzwischen mehrfach zugeordnet"
                    )
            _rollback_elasticsearch_write(
                ictx,
                file_path,
                vector_ids if rollback_db_associations else orphan_ids,
                context=context,
            )
        except Exception as exc:
            return rollback_failed(
                f"{context}:elasticsearch",
                exc,
                "Elasticsearch-Rollback",
            )

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
            return rollback_failed(
                f"{context}:db_associations",
                exc,
                "DB-Rollback",
            )

    if vector_ids and ictx.datasource_id is not None:
        try:
            _reconcile_rollback_memberships(
                ictx,
                vector_ids,
                exclude_owner=False,
                verify_absent=True,
            )
        except Exception as exc:
            return rollback_failed(f"{context}:verification", exc, "Rollback-Abnahme")

    return rollback_contract_error or True


def _call_activity_callback(callback, event, **details):
    """Meldet einen Diagnose-Checkpoint, ohne die Ingestion zu beeinflussen."""
    if callback is None:
        return
    try:
        callback(event, **details)
    except Exception as exc:
        logger.debug("Ingestion-Aktivitaetscallback fehlgeschlagen: %s", exc, exc_info=True)


def _emit_activity(ictx, event, **details):
    _call_activity_callback(getattr(ictx, "activity_callback", None), event, **details)


def _emit_file_log(ictx, file_log, t0):
    """FileLog-Finalisierung + Callback (Pattern D)."""
    if file_log:
        file_log.duration_ms = elapsed_ms(t0)
        if ictx.file_log_callback:
            ictx.file_log_callback(file_log)
    _emit_activity(
        ictx,
        "file_finished",
        phase="file_complete",
        chunks_created=file_log.chunk_count if file_log else None,
    )


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


def _wait_for_index_contract_target_recovery(
    config: dict,
    error: IndexContractUnavailableError,
    *,
    control_callback,
    status_callback,
    stats: dict,
    require_configured_default: bool = False,
) -> bool:
    """Wartet fail-closed auf eine erneut vollstaendig verifizierte Indexbindung.

    ``False`` bedeutet ausschliesslich Cancel/Shutdown oder CLI-Modus. Echte
    Contract-Mismatches werden unveraendert als ``IndexCompatibilityError``
    propagiert und duerfen nie als Recovery-Erfolg gelten.
    """

    if not control_callback:
        return False

    from kiara.retrieval.index_compatibility import (
        guard_index_for_active_job,
        raw_qdrant_client,
    )

    cfg = config.get("index_contract") or {}
    poll_seconds = float(
        cfg.get("recovery_poll_seconds", INDEX_CONTRACT_RECOVERY_POLL_SECS_DEFAULT)
    )
    started = time.monotonic()
    stats.setdefault("contract_discovery_waits", 0)
    stats["contract_discovery_waits"] += 1
    if status_callback:
        status_callback("Warte auf Index-Abhaengigkeit...")
    logger.warning(
        "Index-Abhaengigkeit fuer %s nicht erreichbar; Worker wartet fail-closed",
        error.target.collection_name,
    )

    client = raw_qdrant_client(config)
    try:
        while True:
            signal = control_callback()
            if signal in ("cancel", "shutdown"):
                return False
            try:
                with guard_index_for_active_job(
                    config,
                    error.target,
                    operation="contract_recovery_probe",
                    write=False,
                    require_configured_default=require_configured_default,
                    force_refresh=True,
                    qdrant_client=client,
                ):
                    pass
            except IndexContractUnavailableError as retry_error:
                logger.warning(
                    "Index-Abhaengigkeit weiter nicht erreichbar (%s, naechster Versuch in %.1fs)",
                    retry_error.detail,
                    poll_seconds,
                )
                time.sleep(poll_seconds)
                continue

            elapsed = int(time.monotonic() - started)
            stats.setdefault("contract_discovery_recoveries", 0)
            stats["contract_discovery_recoveries"] += 1
            stats.setdefault("contract_discovery_wait_seconds", 0)
            stats["contract_discovery_wait_seconds"] += elapsed
            if status_callback:
                status_callback("Verarbeite Dateien...")
            logger.info(
                "Index-Abhaengigkeit fuer %s nach %ds wieder verifiziert",
                error.target.collection_name,
                elapsed,
            )
            return True
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _wait_for_index_contract_recovery(
    ictx: _IngestionContext,
    error: IndexContractUnavailableError,
) -> bool:
    """Context-Adapter fuer die dateilokale Contract-Recovery."""

    return _wait_for_index_contract_target_recovery(
        ictx.config,
        error,
        control_callback=ictx.control_callback,
        status_callback=ictx.status_callback,
        stats=ictx.stats,
        require_configured_default=(
            error.target.vector_kind.value in {"dense", "late_chunking"}
            and bool(
                getattr(
                    ictx.store,
                    "_require_configured_default",
                    False,
                )
            )
        ),
    )


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

    doc = load_file(
        file_path,
        ddu_enabled=file_ddu_enabled,
        ddu_backends=ctx.ddu_backends,
        mail_archive=ctx.mail_archive,
    )

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
        doc["metadata"]["chunk_total"] = 1
        doc["metadata"]["chunk_size_used"] = len(doc["text"])
        doc["metadata"]["chunk_overlap_used"] = 0
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
    doc["metadata"]["chunk_overlap_used"] = co

    if (
        doc["metadata"].get("content_format") == "markdown"
        and doc["metadata"].get("source_mime_type") == "text/html"
    ):
        doc["text"], chunks = chunk_document_with_full_text(
            doc["text"], doc["metadata"], chunk_size=cs, chunk_overlap=co
        )
        return chunks

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

        previous_end = 0
        for chunk in chunks:
            start, end = chunk.get("char_start"), chunk.get("char_end")
            if (
                type(start) is int
                and type(end) is int
                and 0 <= start < end <= len(d["text"])
                and d["text"][start:end] == chunk["text"]
            ):
                chunk["metadata"].update(
                    source_char_start=start,
                    source_char_end=end,
                    source_gap_before=d["text"][previous_end:start],
                    source_overlap_before=max(0, previous_end - start),
                    source_body_start=0,
                )
                previous_end = end

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

    stamp_document_versions(file_texts, file_metas)
    for meta in file_metas:
        meta["processing_input_version"] = meta["processing_version"]
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
    # HTML-Markdown kann beim Chunking wiederholte Tabellenköpfe erhalten.
    # Late Chunking muss denselben vorbereiteten Text wie die Spans verwenden.
    full_text = doc["text"] if doc else ""
    return file_texts, file_metas, file_ids, full_text, file_spans


# ---------------------------------------------------------------------------
# Write Functions
# ---------------------------------------------------------------------------


def _write_embeddings(
    file_texts,
    file_metas,
    file_ids,
    store,
    batch_size,
    control_callback=None,
    embeddings=None,
    activity_callback=None,
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
        activity_callback: Optionaler Diagnose-Callback nach erfolgreichem Batch.

    Returns:
        Liste der erfolgreich in die VectorDB geschriebenen Chunk-IDs.

    Raises:
        _CancelledError: Wenn der control_callback 'cancel' zurueckgibt.
        ValueError: Wenn embeddings-Laenge nicht zu file_texts passt.
    """
    if embeddings is not None and len(embeddings) != len(file_texts):
        raise ValueError("embeddings/file_texts length mismatch in _write_embeddings")

    written_ids: list[str] = []
    with store._index_contract_guard(operation="vector_batch_write", write=True):
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
                _call_activity_callback(
                    activity_callback,
                    "vector_batch_written",
                    phase="embedding",
                    chunks_written_delta=len(batch_ids),
                )
        except IndexCompatibilityError as exc:
            if written_ids:
                try:
                    _rollback_vector_writes(
                        store,
                        written_ids,
                        context="partial_vector_write",
                    )
                except Exception as rollback_exc:
                    exc.add_rollback_diagnostic(
                        phase="partial_vector_write:vectordb",
                        error=rollback_exc,
                    )
            raise
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
                except IndexCompatibilityError:
                    raise
                except Exception as rollback_exc:
                    raise _RollbackFailedError(
                        "VectorDB-Partial-Write konnte nicht zurueckgerollt werden"
                    ) from rollback_exc
            raise

    return written_ids


def _publish_search_snapshot(ictx, texts, metadatas, ids):
    """Publish the guarded canonical snapshot to every PG-confirmed owner."""
    from kiara.retrieval.datasource_metadata import same_ds_ids
    from kiara.retrieval.elasticsearch_indices import document_address
    from kiara.retrieval.elasticsearch_service import is_configured, model_uses_elasticsearch

    if not is_configured(ictx.config):
        return _write_to_elasticsearch(
            ictx.config, texts, metadatas, ids, ictx.stats, datasource_id=ictx.datasource_id
        )
    if not model_uses_elasticsearch(ictx.config, ictx.embedding_model):
        return
    memberships = ictx.chunk_ds.get_datasource_ids_for_chunks(ids, ictx.embedding_model)
    batches = {}
    for text, meta, cid in zip(texts, metadatas, ids, strict=True):
        owners = memberships.get(cid, [])
        if ictx.datasource_id not in owners or not same_ds_ids(meta.get("ds_ids"), owners):
            raise RuntimeError("ES-Publikation ohne bestaetigten PG-Snapshot")
        for ds in owners:
            document_address(ictx.config, ds, cid)
            batch = batches.setdefault(ds, ([], [], []))
            batch[0].append(text)
            batch[1].append(meta)
            batch[2].append(cid)
    for ds, (batch_texts, batch_metas, batch_ids) in batches.items():
        _write_to_elasticsearch(
            ictx.config, batch_texts, batch_metas, batch_ids, ictx.stats, datasource_id=ds
        )


def _write_to_elasticsearch(config, file_texts, file_metas, file_ids, stats, *, datasource_id):
    """Schreibt Chunks nach Elasticsearch, falls als Hybrid-Suche konfiguriert.

    Bei konfiguriertem Elasticsearch sind Schreibfehler fatal fuer die Datei:
    der Hash darf sonst nicht als erfolgreich verarbeitet persistiert werden.
    """
    try:
        from kiara.retrieval.elasticsearch_service import bulk_index_chunks, is_configured

        if not is_configured(config):
            return

        count = bulk_index_chunks(
            config, file_texts, file_metas, file_ids, datasource_id=datasource_id
        )
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


def _rollback_colbert_associations(
    ictx: _IngestionContext,
    chunk_ids: list[str],
    ds_model_key: str,
) -> int:
    """Entfernt nur die in diesem Sync neu angelegten ColBERT-Zuordnungen."""
    ids = list(dict.fromkeys(chunk_ids))
    if not ids or ictx.datasource_id is None:
        return 0

    from kiara.shared.db import db_transaction

    with db_transaction() as conn:
        ictx.chunk_ds.acquire_graph_cleanup_lock(conn)
        ictx.chunk_ds.remove_chunks_for_datasource(
            ictx.datasource_id,
            ds_model_key,
            ids,
            conn=conn,
        )
    logger.warning("ColBERT-DB-Rollback: %d Zuordnungen entfernt", len(ids))
    return len(ids)


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

    expected_texts = dict(zip(chunk_ids, chunk_texts, strict=True))
    if len(chunk_metas) != len(chunk_ids):
        raise ValueError("ColBERT-Sync: Metadaten passen nicht zu den Chunk-IDs")

    from kiara.backend_selector import normalize_model_name
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_cleanup_lock,
        colbert_datasource_lock,
        colbert_ds_model_key,
        embed_colbert,
    )
    from kiara.retrieval.index_compatibility import (
        colbert_index_target,
        guard_index,
        raw_qdrant_client,
    )
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL

    colbert_model = normalize_model_name(
        ictx.config.get("retrieval", {}).get("colbert", {}).get("model", COLBERT_DEFAULT_MODEL)
    )
    ds_model_key = colbert_ds_model_key(colbert_model)
    target = colbert_index_target(ictx.config, colbert_model)

    with _tracer.start_as_current_span("colbert_index") as span:
        span.set_attribute("chunk_count", len(chunk_ids))
        span.set_attribute("model", colbert_model)

        # Aeusseres Lock: DS-weite Serialisierung gegen Cleanup
        with colbert_datasource_lock(ictx.datasource_id, wait=True):
            # Inneres Lock: Keyspace-Serialisierung
            with colbert_cleanup_lock(ds_model_key, wait=True):
                if not _datasource_accepts_colbert_write(ictx.datasource_id):
                    raise _ColBERTWriteBlocked(
                        f"DS {ictx.datasource_id} ist inaktiv oder nicht vorhanden"
                    )
                _ensure_colbert_model_is_current_for_write(
                    ictx.config,
                    colbert_model,
                )
                # Ein Write-Fence bleibt ueber Embed, alle Batches, DB-SoT,
                # Payload-Sync und einen eventuell noetigen Rollback aktiv.
                with guard_index(
                    ictx.config,
                    target,
                    operation="colbert_ingestion_sync",
                    write=True,
                    allow_initialize=True,
                    qdrant_client=raw_qdrant_client(ictx.config),
                ):
                    cstore = None
                    written_ids: list[str] = []
                    added_association_ids: list[str] = []
                    prior_ds_ids_map: dict[str, list[int]] = {}
                    associations_attempted = False
                    try:
                        # Store erst NACH Lock + aeusserem Contract-Fence konstruieren.
                        cstore = ColBERTStore(ictx.config, colbert_model, intent="write")
                        cstore.ensure_collection()
                        collection_name = cstore.get_collection_name()

                        existing_ids = ictx.chunk_ds.chunks_exist(chunk_ids, ds_model_key)
                        # Ein erhaltener Vektor kann nach Dense-Reparatur noch
                        # auf altem Enrichment beruhen. Nur den aktuellen Text
                        # samt tatsaechlich gespeichertem Vektor deduplizieren.
                        existing_ids &= cstore.existing_ids(
                            chunk_ids, require_vectors=True, expected_texts=expected_texts
                        )
                        prior_ds_ids_map = ictx.chunk_ds.get_datasource_ids_for_chunks(
                            chunk_ids,
                            ds_model_key,
                        )
                        added_association_ids = [
                            cid
                            for cid in chunk_ids
                            if ictx.datasource_id not in prior_ds_ids_map.get(cid, [])
                        ]
                        missing_ids = [cid for cid in chunk_ids if cid not in existing_ids]
                        if missing_ids:
                            missing_texts = [
                                text
                                for text, cid in zip(chunk_texts, chunk_ids, strict=False)
                                if cid not in existing_ids
                            ]
                            missing_metas = [
                                dict(meta)
                                for meta, cid in zip(chunk_metas, chunk_ids, strict=False)
                                if cid not in existing_ids
                            ]

                            # Neue Owner erst nach dem PG-Commit veroeffentlichen.
                            # Reparaturen erhalten bereits bestaetigte Owner auch
                            # bei einem Abbruch mitten in den Upsert-Batches.
                            for cid, meta in zip(missing_ids, missing_metas, strict=True):
                                meta.pop("ds_ids", None)
                                if prior_ds_ids_map.get(cid):
                                    meta["ds_ids"] = prior_ds_ids_map[cid]

                            token_embeddings = embed_colbert(
                                ictx.config,
                                colbert_model,
                                missing_texts,
                                input_type=SEARCH_DOCUMENT,
                            )
                            if len(token_embeddings) != len(missing_ids):
                                raise BackendUnavailableError(
                                    "ColBERT embed_colbert lieferte "
                                    f"{len(token_embeddings)} Embeddings fuer "
                                    f"{len(missing_ids)} Chunks"
                                )
                            # Auch ein partiell fehlgeschlagener Upsert muss in
                            # den Rollback eingehen.
                            written_ids = list(missing_ids)
                            cstore.add(
                                missing_texts,
                                missing_metas,
                                missing_ids,
                                token_embeddings,
                            )

                        associations_attempted = True
                        ictx.chunk_ds.add_associations(
                            chunk_ids,
                            ictx.datasource_id,
                            ds_model_key,
                        )
                        ds_ids_map = ictx.chunk_ds.get_datasource_ids_for_chunks(
                            chunk_ids,
                            ds_model_key,
                        )
                        cstore.update_ds_ids(ds_ids_map, wait=True)
                        incomplete_ids = set(chunk_ids) - cstore.existing_ids(
                            chunk_ids, require_vectors=True, expected_texts=expected_texts
                        )
                        if incomplete_ids:
                            raise VectorDBConnectionError(
                                "ColBERT-Sync unvollstaendig: "
                                f"{len(incomplete_ids)} Chunks ohne aktuellen Text und Vektor"
                            )
                        span.set_attribute("written_count", len(written_ids))
                        return len(written_ids), collection_name
                    except IndexCompatibilityError as exc:
                        external_rollback_ok = True
                        # Bereits besessene Punkte bleiben als reparierter
                        # Bestand erhalten; Loeschen wuerde andere Owner treffen.
                        unowned_written_ids = [
                            cid for cid in written_ids if not prior_ds_ids_map.get(cid)
                        ]
                        if unowned_written_ids and cstore is not None:
                            try:
                                cstore.delete_by_ids(unowned_written_ids)
                            except Exception as rollback_exc:
                                external_rollback_ok = False
                                exc.add_rollback_diagnostic(
                                    phase="colbert_sync:vectordb",
                                    error=rollback_exc,
                                )
                        if associations_attempted and prior_ds_ids_map and cstore is not None:
                            try:
                                cstore.update_ds_ids(prior_ds_ids_map, wait=True)
                            except Exception as rollback_exc:
                                external_rollback_ok = False
                                exc.add_rollback_diagnostic(
                                    phase="colbert_sync:payload_restore",
                                    error=rollback_exc,
                                )
                        if (
                            external_rollback_ok
                            and associations_attempted
                            and added_association_ids
                        ):
                            try:
                                _rollback_colbert_associations(
                                    ictx,
                                    added_association_ids,
                                    ds_model_key,
                                )
                            except Exception as rollback_exc:
                                exc.add_rollback_diagnostic(
                                    phase="colbert_sync:db_associations",
                                    error=rollback_exc,
                                )
                        raise


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

    except IndexContractUnavailableError as exc:
        # Dense ist zu diesem Zeitpunkt bereits erfolgreich. Der Datei-Hash
        # darf waehrend eines laengeren Contract-Waits nicht als vollstaendig
        # committed persistiert bleiben, weil ColBERT noch fehlt.
        ictx.pending_hashes.pop(file_path, None)
        _cleanup_file_hash(ictx, file_path)
        _persist_hashes_checkpoint(ictx, force=True)
        try:
            recovered = _wait_for_index_contract_recovery(ictx, exc)
        except IndexCompatibilityError as hard_error:
            _abort_index_compatibility(
                ictx,
                file_path,
                file_log,
                hard_error,
                t0,
                consumer="colbert",
                is_colbert=True,
            )
            return _FileAction.ABORT, file_retries
        if recovered:
            if file_log:
                file_log.status = "pending"
                file_log.error_message = ""
                file_log.error_phase = ""
                file_log.is_infra_error = False
            return _FileAction.RETRY, file_retries
        if not ictx.control_callback:
            _abort_index_compatibility(
                ictx,
                file_path,
                file_log,
                exc,
                t0,
                consumer="colbert",
                is_colbert=True,
            )
            return _FileAction.ABORT, file_retries
        _emit_file_log(ictx, file_log, t0)
        return _FileAction.CANCEL, file_retries

    except IndexCompatibilityError as exc:
        _abort_index_compatibility(
            ictx,
            file_path,
            file_log,
            exc,
            t0,
            consumer="colbert",
            is_colbert=True,
        )
        return _FileAction.ABORT, file_retries

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
    """Gleicht Live-Payloads mit der SoT ab; wiederholt bei konkurrierendem Drift.

    Keine initialen Metadaten oder Cache-Eintraege gelten als Schreibnachweis.
    Die kurzen SoT-Lese-Kontexte liegen ausserhalb der Backend-Guards.
    """
    from kiara.retrieval import elasticsearch_service as es_svc
    from kiara.retrieval.datasource_metadata import same_ds_ids

    def read_memberships():
        mapping = ictx.chunk_ds.get_datasource_ids_for_chunks(chunk_ids, ictx.embedding_model)
        return {cid: sorted(mapping.get(cid, [])) for cid in chunk_ids}

    try:
        attempts = ictx.config.get("ingestion", {}).get("ds_ids_sync_attempts", 3)
        for _ in range(attempts):
            ds_ids_map = read_memberships()
            actual = ictx.store.get_ds_ids(chunk_ids)
            changed = {
                cid: ds_ids
                for cid, ds_ids in ds_ids_map.items()
                if not same_ds_ids(actual.get(cid), ds_ids)
            }
            if changed:
                ictx.store.update_ds_ids(changed, wait=True)
            ictx.stats["ds_metadata_unchanged"] = (
                ictx.stats.get("ds_metadata_unchanged", 0) + len(ds_ids_map) - len(changed)
            )
            if es_svc.model_uses_elasticsearch(ictx.config, ictx.embedding_model):
                from kiara.ingestion.document_publication import replayable_memberships
                from kiara.retrieval.elasticsearch_indices import document_address

                existing = es_svc.reconcile_memberships(
                    ictx.config,
                    ds_ids_map,
                    adopting_datasource_id=ictx.datasource_id,
                    replayable={
                        document_address(ictx.config, ds, cid)
                        for ds, cid in replayable_memberships(ds_ids_map)
                    },
                )
                es_svc.mark_search_data_adopted(ictx.config, {a.datasource_id for a in existing})
            if read_memberships() == ds_ids_map:
                return
        raise BackendUnavailableError("DS-Zuordnungen aendern sich waehrend des Abgleichs")
    except IndexCompatibilityError:
        raise
    except _CATEGORICAL_ERRORS:
        raise
    except Exception as exc:
        logger.error("update_ds_ids fehlgeschlagen (%s): %s", context, exc, exc_info=True)
        raise BackendUnavailableError(f"DS-Zuordnungsabgleich fehlgeschlagen: {exc}") from exc


def _persist_graph_and_associations(
    ictx: _IngestionContext,
    file_texts: list[str],
    file_ids: list[str],
) -> None:
    """Persistiert Graph-Junctions und chunk_datasources unter gemeinsamem Lock."""
    if ictx.datasource_id is None:
        _extract_graph_entities(ictx.config, file_texts, file_ids, ictx.stats)
        # Auch CLI-Reparaturen behalten bereits bestaetigte DS-Zuordnungen;
        # fuer neue, unzugeordnete Chunks liefert PostgreSQL eine leere Liste.
        _update_external_ds_ids(ictx, file_ids, context="embed_cli")
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
    if not file_ids:
        return _DedupResult.NO_DEDUP, file_texts, file_metas, file_ids, file_spans, None

    try:
        if ictx.datasource_id is None:
            # CLI-Chunks brauchen keine DS-Zuordnung, sind aber nach demselben
            # Eingangsvergleich wiederverwendbar. Den Bestand prueft Qdrant unten.
            existing_ids = set(file_ids)
        else:
            existing_ids = ictx.chunk_ds.chunks_exist(file_ids, ictx.embedding_model)
    except IndexCompatibilityError:
        raise
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
                include=["documents", "metadatas"] if _colbert_enabled_for_run(ictx) else [],
            )
            verified_ids = set(verified.get("ids", []))
        except IndexCompatibilityError:
            raise
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
                    "Dedup: %d/%d Kandidaten haben keinen Qdrant-Vektor "
                    "— werden neu embedded (Datei: %s)",
                    len(orphans),
                    len(existing_ids),
                    file_path,
                )
                existing_ids = verified_ids

    existing_batch_ids = [id for id in file_ids if id in existing_ids]
    eb = None
    if existing_batch_ids:
        try:
            if ictx.datasource_id is not None:
                _sync_ds_ids(ictx, existing_batch_ids, context="dedup")
        except IndexCompatibilityError:
            raise
        except _CATEGORICAL_ERRORS:
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
        # No metadata write is needed when live memberships already match.
        # Retain the visibility obligation even if a later phase raises before
        # the pipeline can return its stats (e.g. ColBERT or a callback).
        # Existing ES addresses were registered by membership reconciliation.
        # New DS documents are written by the canonical publication callback.

        # existing_batch fuer ColBERT-Backfill zusammenbauen
        if _colbert_enabled_for_run(ictx):
            # Die wiederverwendete Darstellung kann bereits angereichert sein.
            # ColBERT muss denselben gespeicherten Text wie Dense einbetten.
            try:
                stored = dict(
                    zip(
                        verified["ids"],
                        zip(verified["documents"], verified["metadatas"], strict=True),
                        strict=True,
                    )
                )
                eb_texts = [stored[cid][0] for cid in existing_batch_ids]
                eb_metas = [stored[cid][1] for cid in existing_batch_ids]
            except (KeyError, ValueError) as exc:
                raise VectorDBConnectionError(
                    "ColBERT-Backfill: gespeicherte Dense-Chunks sind unvollstaendig"
                ) from exc
        else:
            eb_texts = [
                t for t, cid in zip(file_texts, file_ids, strict=False) if cid in existing_ids
            ]
            eb_metas = [
                m for m, cid in zip(file_metas, file_ids, strict=False) if cid in existing_ids
            ]
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
    """Prueft nur die fachliche Aktivierung von Late Chunking.

    Bedingungen:
    1. config["enrichment"]["late_chunking"] == True
    2. model ist nicht None

    Profilart, Endpoint, Rolle und Verifikation prueft unmittelbar danach der
    atomare Late-Index-Guard in ``embed_texts_late``. Ein unbekanntes oder
    unverified Profil darf nicht still auf Dense zurueckfallen.
    """
    enrichment_cfg = config.get("enrichment") or {}
    if not enrichment_cfg.get("late_chunking", False):
        return False
    if not model:
        return False
    from kiara.backend_selector import normalize_model_name

    return bool(normalize_model_name(model))


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
    Bei Erfolg: geschriebene Chunks zaehlen; Dateiabschluss bleibt beim Caller.
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

        def abort_for_rollback_contract_error(
            result: bool | IndexCompatibilityError,
        ) -> bool:
            if not isinstance(result, IndexCompatibilityError):
                return False
            span.record_exception(result)
            _abort_index_compatibility(
                ictx,
                file_path,
                file_log,
                result,
                t0,
                consumer="embedding_rollback",
            )
            return True

        try:
            _emit_activity(
                ictx,
                "embedding_started",
                phase="embedding",
                chunks_created=len(file_texts),
            )
            embeddings = None
            late_chunking_used = False
            if _should_use_late_chunking(ictx.config, ictx.embedding_model):
                if not _late_chunking_file_eligible(
                    ictx.last_full_text,
                    ictx.last_original_chunks,
                    ictx.last_original_spans,
                    file_metas,
                ):
                    raise ValueError(
                        "Late-Index kann diese Datei ohne gueltigen Volltext und Chunk-Spans "
                        "nicht verarbeiten; Dense-Fallback ist nicht zulaessig"
                    )
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
                    raise ValueError(
                        "Late-Embedding lieferte keine vollstaendigen Vektoren; "
                        "Dense-Fallback in einen Late-Index ist nicht zulaessig"
                    )

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
                    activity_callback=getattr(ictx, "activity_callback", None),
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
                    from kiara.retrieval.elasticsearch_service import (
                        is_configured,
                        model_uses_elasticsearch,
                    )

                    if not is_configured(ictx.config) or model_uses_elasticsearch(
                        ictx.config, ictx.embedding_model
                    ):
                        _write_to_elasticsearch(
                            ictx.config,
                            file_texts,
                            file_metas,
                            file_ids,
                            ictx.stats,
                            datasource_id=ictx.datasource_id,
                        )
                except IndexCompatibilityError:
                    raise
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
                _emit_activity(ictx, "elasticsearch_phase_complete", phase="elasticsearch")

            with _tracer.start_as_current_span("graph_extraction") as graph_span:
                graph_enabled = ictx.config.get("graph", {}).get("enabled", False)
                graph_span.set_attribute("enabled", graph_enabled)
                db_associations_attempted = ictx.datasource_id is not None
                _persist_graph_and_associations(ictx, file_texts, file_ids)
                _emit_activity(ictx, "graph_phase_complete", phase="graph")

            ictx.stats["total_chunks"] += len(file_texts)

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
            rollback_result = _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="cancelled_embedding_phase",
            )
            if abort_for_rollback_contract_error(rollback_result):
                return _FileAction.ABORT, file_retries
            if not rollback_result:
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

        except IndexContractUnavailableError as exc:
            span.record_exception(exc)
            try:
                recovered = _wait_for_index_contract_recovery(ictx, exc)
            except IndexCompatibilityError as hard_error:
                _rollback_embedding_persistence(
                    ictx,
                    file_path,
                    written_vector_ids,
                    rollback_elasticsearch=es_write_attempted,
                    rollback_db_associations=db_associations_attempted,
                    context="index_contract_recovery_mismatch",
                    compatibility_error=hard_error,
                )
                _abort_index_compatibility(
                    ictx,
                    file_path,
                    file_log,
                    hard_error,
                    t0,
                    consumer="embedding",
                )
                return _FileAction.ABORT, file_retries

            if recovered:
                if file_log:
                    file_log.status = "pending"
                    file_log.error_message = ""
                    file_log.error_phase = ""
                    file_log.is_infra_error = False
                return _FileAction.RETRY, file_retries

            rollback_result = _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="index_contract_unavailable_cancel",
                compatibility_error=None if ictx.control_callback else exc,
            )
            if abort_for_rollback_contract_error(rollback_result):
                return _FileAction.ABORT, file_retries
            if not rollback_result:
                rollback_error = RuntimeError(
                    "Embedding-Rollback waehrend Vertragsausfall fehlgeschlagen"
                )
                _record_error(
                    ictx,
                    file_log,
                    file_path,
                    rollback_error,
                    "vectordb_rollback",
                    is_vectordb=True,
                )
                ictx.stats["abort_reason"] = str(rollback_error)
                _cleanup_file_hash(ictx, file_path)
                _persist_hashes_checkpoint(ictx, force=True)
                _emit_file_log(ictx, file_log, t0)
                return _FileAction.ABORT, file_retries

            if not ictx.control_callback:
                _abort_index_compatibility(
                    ictx,
                    file_path,
                    file_log,
                    exc,
                    t0,
                    consumer="embedding",
                )
                return _FileAction.ABORT, file_retries

            _cleanup_file_hash(ictx, file_path)
            _persist_hashes_checkpoint(ictx, force=True)
            _emit_file_log(ictx, file_log, t0)
            return _FileAction.CANCEL, file_retries

        except IndexCompatibilityError as exc:
            span.record_exception(exc)
            _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="index_compatibility_error",
                compatibility_error=exc,
            )
            _abort_index_compatibility(
                ictx,
                file_path,
                file_log,
                exc,
                t0,
                consumer="embedding",
            )
            return _FileAction.ABORT, file_retries

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
            rollback_result = _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="vectordb_error",
            )
            if abort_for_rollback_contract_error(rollback_result):
                return _FileAction.ABORT, file_retries
            if not rollback_result:
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
            rollback_result = _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="categorical_embedding_error",
            )
            if abort_for_rollback_contract_error(rollback_result):
                return _FileAction.ABORT, file_retries
            if not rollback_result:
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
            rollback_result = _rollback_embedding_persistence(
                ictx,
                file_path,
                written_vector_ids,
                rollback_elasticsearch=es_write_attempted,
                rollback_db_associations=db_associations_attempted,
                context="generic_embedding_error",
            )
            if abort_for_rollback_contract_error(rollback_result):
                return _FileAction.ABORT, file_retries
            if not rollback_result:
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
                    # Nur ein tatsaechlich erhaltener Originaltext traegt
                    # weiterhin die Positionsherkunft des Chunkers.
                    if enriched_texts[i].endswith(file_texts_before[i]):
                        meta["source_body_start"] = len(enriched_texts[i]) - len(
                            file_texts_before[i]
                        )
                    else:
                        meta.pop("source_body_start", None)

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
    from kiara.ingestion.document_publication import document_publication

    with document_publication(
        str(ictx.store.get_collection_name()),
        file_path,
        ictx.embedding_model,
        datasource_id=ictx.datasource_id,
    ) as publication:
        return _process_file_with_publication(ictx, file_path, publication)


def _process_file_with_publication(ictx: _IngestionContext, file_path: str, publication) -> bool:
    """Verarbeitet eine Datei komplett (inkl. Retry-Schleife).

    Returns:
        True → naechste Datei verarbeiten.
        False → Pipeline abbrechen (Cancel/Abort).
    """
    with _tracer.start_as_current_span("ingest_file") as span:
        span.set_attribute("file_path", str(file_path))
        span.set_attribute("datasource_id", ictx.datasource_id or 0)
        span.set_attribute("job_id", _get_current_job_id() or 0)
        span.set_attribute("file_type", os.path.splitext(str(file_path))[1].lower())

        if ictx.control_callback:
            signal = ictx.control_callback()
            if signal in ("cancel", "shutdown"):
                span.set_attribute("final_status", "cancelled_before_start")
                return False

        if ictx.file_preparation_callback is not None:
            # Auch ein Ladefehler darf einen geplanten Wiederanlauf nicht als
            # weiterhin vollstaendig freigeben. Das Cleanup folgt nach Parsing.
            publication.begin()

        t0 = time.monotonic()
        _emit_activity(
            ictx,
            "file_started",
            phase="loading",
            file_name=Path(file_path).name,
            file_path=str(file_path),
        )
        if ictx.file_start_callback:
            ictx.file_start_callback(Path(file_path).name, file_path)

        try:
            span.set_attribute("file_size", os.path.getsize(file_path))
        except OSError:
            span.set_attribute("file_size", 0)

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

        file_retries = 0
        skip_loading = False
        errors_before = ictx.stats["errors"]
        enrichment_applied = False

        ictx.last_full_text = ""
        ictx.last_original_chunks = None
        ictx.last_original_spans = None
        ictx.last_original_chunks_by_id = None
        ictx.last_original_spans_by_id = None

        while True:
            if not skip_loading:
                errors_before = ictx.stats["errors"]
                from contextlib import nullcontext

                try:
                    loading = (
                        ictx.file_loading_context(file_path)
                        if ictx.file_loading_context is not None
                        else nullcontext()
                    )
                    with loading:
                        file_texts, file_metas, file_ids, full_text, file_spans = (
                            _process_single_file(
                                file_path,
                                ictx.config,
                                ictx.ds_overrides,
                                ictx.datasource_id,
                                ictx.ctx,
                                ictx.stats,
                                file_log=file_log,
                            )
                        )
                except ChunkingContentError as exc:
                    logger.warning(
                        "Chunking fehlgeschlagen fuer '%s': %s — verarbeite naechste Datei",
                        file_path,
                        exc,
                        exc_info=True,
                    )
                    _record_error(ictx, file_log, file_path, exc, "chunking")
                    file_texts, file_metas, file_ids, full_text, file_spans = [], [], [], "", []
                except OSError as exc:
                    _record_error(ictx, file_log, file_path, exc, "loading")
                    file_texts, file_metas, file_ids, full_text, file_spans = [], [], [], "", []
                document_ids = list(file_ids)
                if (
                    file_texts or ictx.stats["errors"] == errors_before
                ) and ictx.file_preparation_callback is not None:
                    replaced = ictx.file_preparation_callback(file_path, file_metas, file_ids)
                    if replaced:
                        # Quellenweites Cleanup entwertet die bisherige Generation.
                        publication.generation = None
                enrichment_applied = False
                ictx.last_full_text = full_text
                _emit_activity(
                    ictx,
                    "content_chunked",
                    phase="chunking_complete",
                    chunks_created=len(file_texts),
                )
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

            publication.begin()

            # FULL-Dedup leert die Arbeitslisten. Ein ColBERT-Retry braucht
            # dennoch alle Kandidaten, auch zuvor deduplizierte Chunks.
            dedup_inputs = (file_texts, file_metas, file_ids, file_spans)
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
            except IndexContractUnavailableError as exc:
                span.record_exception(exc)
                try:
                    recovered = _wait_for_index_contract_recovery(ictx, exc)
                except IndexCompatibilityError as hard_error:
                    _abort_index_compatibility(
                        ictx,
                        file_path,
                        file_log,
                        hard_error,
                        t0,
                        consumer="dedup",
                    )
                    span.set_attribute("final_status", "aborted_dedup_contract")
                    return False
                if recovered:
                    skip_loading = True
                    span.set_attribute("final_status", "retry_dedup_contract")
                    continue
                if not ictx.control_callback:
                    _abort_index_compatibility(
                        ictx,
                        file_path,
                        file_log,
                        exc,
                        t0,
                        consumer="dedup",
                    )
                    span.set_attribute("final_status", "aborted_dedup_contract")
                    return False
                _cleanup_file_hash(ictx, file_path)
                _persist_hashes_checkpoint(ictx, force=True)
                _emit_file_log(ictx, file_log, t0)
                span.set_attribute("final_status", "cancelled_dedup_contract")
                return False
            except IndexCompatibilityError as exc:
                span.record_exception(exc)
                _abort_index_compatibility(
                    ictx,
                    file_path,
                    file_log,
                    exc,
                    t0,
                    consumer="dedup",
                )
                span.set_attribute("final_status", "aborted_dedup_contract")
                return False
            except _CATEGORICAL_ERRORS as exc:
                span.record_exception(exc)
                is_vector_error = isinstance(exc, VectorDBConnectionError)
                error_phase = "vectordb" if is_vector_error else "embedding"
                label = "Index-Zuordnungsfehler (Dedup)"
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
                    error_phase,
                    is_vectordb=is_vector_error,
                )
                cb_result, file_retries = _handle_circuit_breaker(
                    ictx,
                    file_path,
                    file_log,
                    error_phase=error_phase,
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
                        file_texts, file_metas, file_ids, file_spans = dedup_inputs
                        skip_loading = True
                        continue
                    if signal == "skip":
                        break
                _mark_file_processed(ictx, file_path, document_ids)
                if file_log:
                    file_log.status = "success"
                ictx.consecutive_errors = 0
                _finalize_file(ictx, file_log, t0)
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

            # Enrichment kann nur eine Teilmenge bearbeiten. Eine freigegebene
            # Version entsteht erst aus allen tatsaechlich gespeicherten Chunks.
            file_metas = [{**meta, "processing_pending": True} for meta in file_metas]

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
                if existing_batch:
                    # Der naechste Dedup-Pass muss auch die bisherigen Chunks
                    # erneut an ColBERT uebergeben. Die Eingaben stammen aus
                    # dem Stand vor Enrichment und muessen neu angereichert werden.
                    file_texts, file_metas, file_ids, file_spans = dedup_inputs
                    enrichment_applied = False
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
                    file_texts, file_metas, file_ids, file_spans = dedup_inputs
                    skip_loading = True
                    continue
                if signal == "skip":
                    break

            # Vollstaendiger File-Commit: Dense+ColBERT(+Stats) erfolgreich
            _mark_file_processed(ictx, file_path, document_ids)
            if file_log:
                file_log.status = "success"
            ictx.consecutive_errors = 0
            span.set_attribute("final_status", "success")
            break

        _finalize_file(ictx, file_log, t0)
        return True
