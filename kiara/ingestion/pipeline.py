"""Ingestion pipeline: scan, load, chunk, embed, store.

Ablauf:
    scan  ->  diff  ->  process  ->  embed
    (scan_files)  (_compute_file_diff)  (_process_single_file)  (_write_embeddings)

Die Pipeline verarbeitet Dateien inkrementell ueber File-Hashes. Jede Datei
durchlaeuft die Phasen Loading -> VLM -> Chunking -> Embedding. Pro Phase
entscheidet ein _FileAction-Enum ueber Weiterverarbeitung, Retry oder Abbruch.

State Machine (_FileAction):
    SUCCESS  - Datei erfolgreich verarbeitet, naechste Datei.
    SKIP     - Datei uebersprungen (leer, Fehler unter Schwelle), naechste Datei.
    RETRY    - Backend-Recovery abgeschlossen, Datei erneut verarbeiten.
    ABORT    - Circuit-Breaker ausgeloest (CLI), Pipeline beenden.
    CANCEL   - Cancel/Shutdown-Signal empfangen, Pipeline beenden.

Circuit-Breaker:
    Nach INGESTION_MAX_CONSECUTIVE_ERRORS aufeinanderfolgenden Infra-Fehlern
    (DDU-Timeout, Embedding-Backend nicht erreichbar) wird im CLI-Modus
    abgebrochen und im Worker-Modus auf Backend-Recovery gewartet.

File-Level State Machine (Retry, Circuit-Breaker, Embedding) in runtime.py.
File-Utilities (Hashing, Scanning) sind in _file_utils.py ausgelagert.
"""

import copy
import logging
import os

logger = logging.getLogger(__name__)

from kiara_ingestion_core.file_diff import _compute_file_diff, _filter_resumed_files
from kiara_ingestion_core.models import ProcessingContext
from tqdm import tqdm

from kiara.backend_selector import normalize_model_name
from kiara.config import get_embedding_model, load_config
from kiara.ingestion._file_utils import (
    ChunkDatasources,
    HashStore,
    scan_files,
)
from kiara.ingestion.datasource_runtime import build_datasource_overrides, load_connector_config
from kiara.ingestion.loader import IMAGE_FORMATS

# Re-Exports aus runtime.py (Worker-, Compat- und Test-Kompatibilitaet)
from kiara.ingestion.runtime import (  # noqa: F401
    _check_ddu_preflight,
    _determine_chunks,
    _get_current_job_id,
    _IngestionContext,
    _init_ingestion_stats,
    _persist_hashes_checkpoint,
    _process_file_in_pipeline,
    set_current_job_id,
)
from kiara.retrieval.vectorstore import VectorStore
from kiara.shared.db import db_transaction

# ---------------------------------------------------------------------------
# Config-Override-Mapping (Pattern E)
# ---------------------------------------------------------------------------

_OVERRIDE_MAPPINGS = [
    ("ingestion", "chunk_size", "chunk_size"),
    ("ingestion", "chunk_overlap", "chunk_overlap"),
    ("datasource", "file_types", "file_types"),
    ("datasource", "exclude_patterns", "exclude_patterns"),
]


def _apply_ds_overrides(config, ds_overrides):
    """Wendet Datasource-Overrides auf die Config an und loest Feature-Flags auf.

    Mutiert config in-place (Chunk-Groesse, File-Types, Bild-Extensions).

    Returns:
        ProcessingContext mit aufgeloesten DDU- und VLM-Einstellungen.
    """
    for section, key, override_key in _OVERRIDE_MAPPINGS:
        if ds_overrides and ds_overrides.get(override_key) is not None:
            config[section][key] = ds_overrides[override_key]

    # Attachment-Indexierung: deaktiviert → */attachments/* ausschließen
    if ds_overrides and not ds_overrides.get("index_attachments", True):
        exclude = config["datasource"].get("exclude_patterns", [])
        config["datasource"]["exclude_patterns"] = [*exclude, "*/attachments/*"]

    # Content-Detection: accept_all_text Flag in Config durchreichen
    accept_all = ds_overrides.get("accept_all_text", False) if ds_overrides else False
    if accept_all:
        config["datasource"]["accept_all_text"] = True

    ddu_enabled = ds_overrides.get("ddu_enabled", False) if ds_overrides else False
    ddu_backends = None
    if ddu_enabled:
        from kiara.config import get_ddu_backends

        ddu_backends = get_ddu_backends(config)
        if not ddu_backends:
            ddu_enabled = False
            logger.warning(
                "DDU aktiviert fuer Datasource, aber keine aktiven DDU-Backends "
                "konfiguriert — DDU deaktiviert"
            )

    ddu_ocr_images = ds_overrides.get("ddu_ocr_images", True) if ds_overrides else True

    vlm_enabled = ds_overrides.get("vlm_enabled", False) if ds_overrides else False
    vlm_model = ""
    vlm_url = ""
    if vlm_enabled:
        from kiara.config import get_vlm_backend

        vlm_backend = get_vlm_backend(config)
        if vlm_backend:
            vlm_model = vlm_backend["model"]
            vlm_url = vlm_backend["url"]
        else:
            vlm_enabled = False

    need_images = vlm_enabled or (ddu_enabled and ddu_ocr_images)
    if need_images:
        current_types = set(config["datasource"]["file_types"])
        config["datasource"]["file_types"] = list(current_types | IMAGE_FORMATS)

    vlm_prompt = config.get("ingestion", {}).get("vlm_prompt", "")

    return ProcessingContext(
        ddu_enabled=ddu_enabled,
        ddu_backends=ddu_backends,
        ddu_ocr_images=ddu_ocr_images,
        vlm_enabled=vlm_enabled,
        vlm_model=vlm_model,
        vlm_url=vlm_url,
        vlm_prompt=vlm_prompt,
    )


def _cleanup_stale_chunks(
    config, store, hash_store, chunk_ds, datasource_id, embedding_model, files_to_delete
):
    """Loescht Chunks fuer geaenderte/entfernte Dateien aus VectorStore, ColBERT und ES.

    Fuehrt Cross-DS-Cleanup durch: chunk_datasources-Eintraege loeschen und
    Hashes anderer betroffener Datasources invalidieren.
    Bereinigt Dense UND alle historisch vorhandenen ColBERT-Modelle.

    Args:
        config: Konfigurationsdict.
        store: VectorStore-Instanz (Dense).
        hash_store: HashStore-Instanz.
        chunk_ds: ChunkDatasources-Instanz (oder None).
        datasource_id: Optionale Datasource-ID.
        embedding_model: Optionales Embedding-Modell.
        files_to_delete: Liste der zu loeschenden Dateipfade.
    """
    from kiara.ingestion.vectordb_ops import (
        _recount_datasource_embeddings,
        _resolve_colbert_models_for_ds,
        delete_from_elasticsearch,
        invalidate_bm25,
    )

    if not files_to_delete:
        return

    # Cross-DS-Cleanup: chunk_datasources-Eintraege + Hash-Invalidierung
    # Reihenfolge (Dense-last-Retry-Anker): externer ColBERT-Cleanup zuerst,
    # dann DB-Transaktion mit chunk_datasources + Graph, dann ES-/Dense-Qdrant-
    # Delete; Dense-Qdrant ist der einzige dateipfadbasierte ID-Retry-Anker
    # (get_ids_by_sources) und darf erst zum Schluss mutiert werden. BM25
    # wird NACH der finalen Dense-Mutation invalidiert, damit ein paralleler
    # Query nicht zwischen Invalidation und Delete stale Dense-Dokumente in
    # den BM25-Cache zieht.
    if datasource_id is not None and chunk_ds is not None:
        old_chunk_ids = store.get_ids_by_sources(files_to_delete)
        if old_chunk_ids:
            affected_ds_ids = chunk_ds.get_other_datasources(
                old_chunk_ids, datasource_id, embedding_model
            )

            # ColBERT-Modelle ermitteln (alle historisch vorhandenen)
            colbert_models = _resolve_colbert_models_for_ds(config, datasource_id, chunk_ds)

            # 1. Externes Cleanup: ColBERT (vor DB/Dense). Fehler -> Abbruch,
            # damit DB- und Dense-Anker erhalten bleiben.
            if colbert_models:
                from kiara.retrieval.colbert import ColBERTStore, colbert_ds_model_key

                for colbert_model in colbert_models:
                    cstore = ColBERTStore(config, colbert_model)
                    cstore.delete_by_ids(old_chunk_ids)

            # 2. DB-Mutation: chunk_datasources + Hash-Invalidierung + Graph
            # atomar in einer Transaktion, damit stale graph_edges auf
            # geloeschte Chunks nicht zurueckbleiben koennen.
            with db_transaction() as conn:
                chunk_ds.acquire_graph_cleanup_lock(conn)
                chunk_ds.delete_by_chunk_ids(old_chunk_ids, embedding_model, conn=conn)
                for ds_id in affected_ds_ids:
                    hash_store.remove_files(ds_id, embedding_model, files_to_delete, conn=conn)
                # ColBERT chunk_datasources ebenfalls bereinigen
                if colbert_models:
                    from kiara.retrieval.colbert import colbert_ds_model_key

                    for colbert_model in colbert_models:
                        chunk_ds.delete_by_chunk_ids(
                            old_chunk_ids,
                            colbert_ds_model_key(colbert_model),
                            conn=conn,
                        )
                # Graph-Cleanup: modell-agnostisch, in derselben Transaktion.
                # Graph-Fehler rollen die DB-Transaktion zurueck, Dense-Qdrant
                # bleibt unmutiert als Retry-Anker.
                graph_chunk_ids = chunk_ds.filter_unreferenced_chunk_ids(
                    old_chunk_ids,
                    conn=conn,
                )
                if graph_chunk_ids:
                    from kiara.graph.store import GraphStore

                    GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)
                # datasource_embeddings-Recount fuer Cross-DS aus SoT.
                # Dense + alle ColBERT-Modelle pro betroffener DS.
                for ds_id in affected_ds_ids:
                    _recount_datasource_embeddings(conn, ds_id, embedding_model)
                    if colbert_models:
                        for colbert_model in colbert_models:
                            _recount_datasource_embeddings(
                                conn, ds_id, colbert_ds_model_key(colbert_model)
                            )

            # 3. Externes Cleanup: Elasticsearch
            delete_from_elasticsearch(config, files_to_delete, datasource_id=datasource_id)

            # 4. Externes Cleanup: Dense VectorDB (zuletzt als Retry-Anker).
            store.delete_by_sources(files_to_delete)
            # 5. BM25 erst NACH finaler Dense-Mutation invalidieren.
            invalidate_bm25(store.get_collection_name())
            return

        # old_chunk_ids leer: ID-basierte ColBERT-/DB-/Graph-Schritte entfallen,
        # aber dateipfadbasierte Schichten (ES + Dense source-basiert + BM25)
        # muessen trotzdem laufen, damit der File-Cleanup vollstaendig bleibt.
        delete_from_elasticsearch(config, files_to_delete, datasource_id=datasource_id)
        store.delete_by_sources(files_to_delete)
        invalidate_bm25(store.get_collection_name())
        return

    # Fallback: kein datasource_id / chunk_ds — nur dateipfadbasierte Schichten
    # plus BM25-Invalidierung NACH Dense-Mutation.
    delete_from_elasticsearch(config, files_to_delete, datasource_id=datasource_id)
    store.delete_by_sources(files_to_delete)
    invalidate_bm25(store.get_collection_name())


def _load_and_diff_files(
    config,
    store,
    incremental,
    datasource_id,
    embedding_model,
    resume_skip_files=None,
    hash_mode="mtime",
    chunk_ds=None,
):
    """Scannt Dateien, berechnet Hashes und ermittelt den Incremental-Diff.

    Loescht Chunks fuer geaenderte/entfernte Dateien aus dem VectorStore.
    Bei vorhandenem chunk_ds wird Cross-DS-Cleanup durchgefuehrt.

    Args:
        config: Konfigurationsdict.
        store: VectorStore-Instanz.
        incremental: Wenn True, nur geaenderte/neue Dateien verarbeiten.
        datasource_id: Optionale Datasource-ID.
        embedding_model: Optionales Embedding-Modell.
        resume_skip_files: Set von Dateipfaden, die im vorherigen Job bereits
            erfolgreich verarbeitet wurden (aus ingestion_file_logs).
        hash_mode: Hash-Strategie ("mtime" oder "content").
        chunk_ds: Optionale ChunkDatasources-Instanz fuer Cross-DS-Cleanup.

    Returns:
        Tuple (all_files, files_to_process, file_hashes, files_to_delete,
               hash_store, resumed_hashes).
    """
    hash_store = HashStore()

    all_files = scan_files(config)
    stored_hashes = hash_store.load(datasource_id, embedding_model)
    missing_size_paths = (
        hash_store.get_missing_size_paths(datasource_id) if incremental and stored_hashes else set()
    )

    # Hash-Mode-Migration: Wenn der Connector jetzt einen anderen Modus
    # verwendet als beim letzten Lauf, Hashes einmalig umrechnen.
    if incremental and stored_hashes:
        stored_mode = hash_store.get_hash_mode(datasource_id, embedding_model)
        if stored_mode != hash_mode:
            logger.info(
                "Hash-Mode-Wechsel erkannt: %s → %s (ds=%s) — migriere Hashes...",
                stored_mode,
                hash_mode,
                datasource_id,
            )
            hash_store.migrate_hash_mode(datasource_id, embedding_model, hash_mode)
            stored_hashes = hash_store.load(datasource_id, embedding_model)
            missing_size_paths = hash_store.get_missing_size_paths(datasource_id)

    files_to_process, file_hashes, deleted_files = _compute_file_diff(
        all_files,
        incremental,
        stored_hashes,
        hash_mode=hash_mode,
    )
    if not incremental and stored_hashes:
        current_paths = set(all_files)
        deleted_files = [old_path for old_path in stored_hashes if old_path not in current_paths]

    # Heal-Pfad: Legacy-Zeilen mit file_size IS NULL aktualisieren
    size_refresh_hashes = {
        fp: file_hashes[fp]
        for fp in missing_size_paths
        if fp in file_hashes and stored_hashes.get(fp) == file_hashes[fp][0]
    }
    if size_refresh_hashes:
        hash_store.upsert_many(datasource_id, embedding_model, size_refresh_hashes)

    # Resume: bereits verarbeitete Dateien herausfiltern (VOR der Chunk-Loeschung!)
    resumed_hashes = {}
    if resume_skip_files and incremental:
        files_to_process, resumed_hashes = _filter_resumed_files(
            files_to_process,
            file_hashes,
            resume_skip_files,
        )

    # Trennung geaenderte / neue Dateien:
    # Nur geaenderte + geloeschte Dateien loeschen (Cross-DS-Cleanup).
    # Bei Full-Reingest gelten alle bereits bekannten aktuellen Dateien als
    # geaendert, damit alte Chunk-IDs nicht per Dedupe weiterverwendet werden.
    # Neue Dateien (kein stored_hash) werden in run_ingestion() per chunks_exist deduped.
    changed_files = [f for f in files_to_process if f in stored_hashes]
    files_to_delete = changed_files + deleted_files

    _cleanup_stale_chunks(
        config, store, hash_store, chunk_ds, datasource_id, embedding_model, files_to_delete
    )

    return all_files, files_to_process, file_hashes, files_to_delete, hash_store, resumed_hashes


# ---------------------------------------------------------------------------
# ColBERT Model-Switch Cleanup (AP2 + AP3)
# ---------------------------------------------------------------------------


def _delete_colbert_datasource_embedding_counter(datasource_id: int, stale_key: str) -> None:
    """Loescht den datasource_embeddings-Counter fuer eine DS + stale ColBERT-Key.

    Fehlende Zeile ist ein erfolgreicher No-op (idempotent).
    Counter-Delete-Fehler werden geloggt und weitergeworfen, damit der Caller
    per ``except Exception: continue`` nur dieses Modell abbricht und
    chunk_datasources als Retry-Anker erhalten bleibt.
    """
    try:
        with db_transaction() as conn:
            conn.execute(
                "DELETE FROM datasource_embeddings "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (datasource_id, stale_key),
            )
    except Exception as exc:
        logger.warning(
            "datasource_embeddings Counter-Delete fehlgeschlagen "
            "fuer DS %s / %s: %s (chunk_datasources bleibt erhalten)",
            datasource_id,
            stale_key,
            exc,
            exc_info=True,
        )
        raise


def _delete_colbert_datasource_entries_with_graph_cleanup(
    config: dict,
    chunk_ds: ChunkDatasources,
    datasource_id: int,
    stale_key: str,
) -> int:
    """Loescht stale ColBERT-Referenzen und bereinigt Graph-Orphans atomar."""
    with db_transaction() as conn:
        chunk_ds.acquire_graph_cleanup_lock(conn)
        rows = conn.execute(
            "SELECT chunk_id FROM chunk_datasources "
            "WHERE datasource_id = %s AND embedding_model = %s",
            (datasource_id, stale_key),
        ).fetchall()
        candidate_ids = list(dict.fromkeys(r["chunk_id"] for r in rows))
        deleted = conn.execute(
            "DELETE FROM chunk_datasources WHERE datasource_id = %s AND embedding_model = %s",
            (datasource_id, stale_key),
        ).rowcount
        if candidate_ids:
            graph_chunk_ids = chunk_ds.filter_unreferenced_chunk_ids(
                candidate_ids,
                conn=conn,
            )
            if graph_chunk_ids:
                from kiara.graph.store import GraphStore

                GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)
        return deleted


def _cleanup_stale_colbert_models(
    config: dict,
    datasource_id: int,
    chunk_ds: ChunkDatasources,
    stale_models: list[str],
) -> None:
    """Bereinigt verwaiste ColBERT-Daten fuer stale Modelle einer DS.

    Pro stale Modell:
    1. Advisory-Lock (non-blocking, Skip bei Konflikt)
    2. Collection-Existenz pruefen
    3. Qdrant-Cleanup (Orphans loeschen, Shared-Points updaten)
    4. Collection droppen falls leer
    5. DB-Cleanup (Counter + chunk_datasources)

    Fehler sind nicht-kritisch — Self-Healing beim naechsten Lauf.
    """
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_cleanup_lock,
        colbert_ds_model_key,
    )

    for stale_model in stale_models:
        try:
            raw_model = str(stale_model) if stale_model is not None else ""
            if not raw_model:
                logger.warning("Leeres stale ColBERT-Modell uebersprungen")
                continue
            stale_key = colbert_ds_model_key(raw_model)
            with colbert_cleanup_lock(stale_key, wait=False) as acquired:
                if not acquired:
                    logger.info(
                        "ColBERT-Cleanup fuer %s uebersprungen (Lock-Konflikt)",
                        stale_key,
                    )
                    continue

                coll_info = ColBERTStore.get_colbert_collection_info(config, stale_model)

                if coll_info is None:
                    # Keine Collection — nur DB bereinigen
                    logger.info(
                        "ColBERT-Collection fuer %s existiert nicht — nur DB-Cleanup",
                        stale_model,
                    )
                    try:
                        _delete_colbert_datasource_embedding_counter(datasource_id, stale_key)
                    except Exception:
                        continue  # Counter-Fehler bricht nur dieses Modell ab
                    _delete_colbert_datasource_entries_with_graph_cleanup(
                        config,
                        chunk_ds,
                        datasource_id,
                        stale_key,
                    )
                    continue

                if coll_info["chunks"] == 0:
                    # Leere Collection droppen, keine Qdrant-Mutation
                    logger.info(
                        "ColBERT-Collection %s ist leer — droppe und bereinige DB",
                        coll_info["name"],
                    )
                    try:
                        cstore = ColBERTStore(config, stale_model)
                        cstore.drop_collection_if_empty()
                    except Exception as exc:
                        logger.warning(
                            "Drop leere ColBERT-Collection %s fehlgeschlagen: %s "
                            "(chunk_datasources bleibt als Retry-Anker)",
                            stale_key,
                            exc,
                            exc_info=True,
                        )
                        continue  # Drop-Fehler bricht nur dieses Modell ab
                    try:
                        _delete_colbert_datasource_embedding_counter(datasource_id, stale_key)
                    except Exception:
                        continue
                    _delete_colbert_datasource_entries_with_graph_cleanup(
                        config,
                        chunk_ds,
                        datasource_id,
                        stale_key,
                    )
                    continue

                # Collection existiert und hat Punkte — Qdrant-Cleanup
                orphan_ids, non_orphan_ds_ids = chunk_ds.preview_datasource_removal(
                    datasource_id, stale_key
                )

                cstore = ColBERTStore(config, stale_model)

                if orphan_ids:
                    cstore.delete_by_ids(orphan_ids)

                if non_orphan_ds_ids:
                    existing = cstore.existing_ids(list(non_orphan_ds_ids.keys()))
                    filtered_map = {
                        cid: ds_list
                        for cid, ds_list in non_orphan_ds_ids.items()
                        if cid in existing
                    }
                    if filtered_map:
                        cstore.update_ds_ids(filtered_map, wait=True)

                # Collection droppen falls nach Cleanup leer
                try:
                    cstore.drop_collection_if_empty()
                except Exception as exc:
                    logger.warning(
                        "drop_collection_if_empty fehlgeschlagen fuer %s: %s "
                        "(chunk_datasources bleibt als Retry-Anker)",
                        stale_key,
                        exc,
                        exc_info=True,
                    )
                    continue

                # DB-Cleanup: erst Counter, dann chunk_datasources
                try:
                    _delete_colbert_datasource_embedding_counter(datasource_id, stale_key)
                except Exception:
                    continue
                _delete_colbert_datasource_entries_with_graph_cleanup(
                    config,
                    chunk_ds,
                    datasource_id,
                    stale_key,
                )

                logger.info(
                    "ColBERT-Cleanup abgeschlossen: DS %s, Modell %s "
                    "(%d Orphans, %d Shared-Updates)",
                    datasource_id,
                    stale_model,
                    len(orphan_ids),
                    len(non_orphan_ds_ids),
                )

        except Exception as exc:
            logger.warning(
                "ColBERT-Cleanup fuer stale Modell %s / DS %s fehlgeschlagen "
                "(nicht-kritisch, naechster Lauf bereinigt): %s",
                stale_model,
                datasource_id,
                exc,
                exc_info=True,
            )


def cleanup_stale_colbert_models(
    config,
    datasource_id,
    chunk_ds,
    stale_models,
    *,
    allow_current_models: set[str] | None = None,
):
    """Public API fuer manuellen ColBERT-Cleanup (Admin-Chat, CLI).

    Returns:
        Stats-Dict fuer Admin/CLI-Reporting.
        ``allow_current_models`` ist nur fuer AP1-klassifizierte
        ``collection_missing``-DB-Anker des aktuellen Modells gedacht.
    """
    from kiara.backend_selector import normalize_model_name
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_cleanup_lock,
        colbert_datasource_lock,
        colbert_ds_model_key,
    )
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL

    requested_models = list(stale_models or [])
    colbert_cfg = config.get("retrieval", {}).get("colbert", {})
    colbert_enabled = bool(colbert_cfg.get("enabled", False))
    current_model = normalize_model_name(colbert_cfg.get("model", COLBERT_DEFAULT_MODEL))
    allowed_current = {
        normalize_model_name(m) for m in (allow_current_models or set()) if normalize_model_name(m)
    }
    stats = {
        "datasource_id": datasource_id,
        "requested_models": requested_models,
        "attempted_models": [],
        "skipped_invalid_models": [],
        "skipped_current_models": [],
        "skipped_recreated_current_models": [],
        "db_only_current_models": [],
        "key_lock_conflicts": [],
        "errors": [],
        "ds_lock_acquired": False,
        "lock_conflict": False,
        "cleanup_invoked": False,
    }

    clean_stale_models = []
    for model in requested_models:
        raw_model = str(model) if model is not None else ""
        normalized = normalize_model_name(raw_model)
        if not normalized or normalized.startswith("colbert::"):
            logger.warning("Ungueltiges stale ColBERT-Modell uebersprungen: %r", model)
            stats["skipped_invalid_models"].append(model)
            continue
        if (
            colbert_enabled
            and current_model
            and normalized == current_model
            and normalized not in allowed_current
        ):
            logger.warning("Aktuelles ColBERT-Modell nicht manuell bereinigt: %r", model)
            stats["skipped_current_models"].append(model)
            continue
        clean_stale_models.append(normalized)
    stats["attempted_models"] = clean_stale_models

    if not clean_stale_models:
        return stats

    db_only_current_models = [
        model
        for model in clean_stale_models
        if colbert_enabled and current_model and model == current_model and model in allowed_current
    ]
    normal_stale_models = [
        model for model in clean_stale_models if model not in set(db_only_current_models)
    ]

    with colbert_datasource_lock(datasource_id, wait=False) as acquired:
        stats["ds_lock_acquired"] = bool(acquired)
        if not acquired:
            logger.info(
                "Manueller ColBERT-Cleanup fuer DS %s uebersprungen (Lock-Konflikt)",
                datasource_id,
            )
            stats["lock_conflict"] = True
            return stats

        if normal_stale_models:
            _cleanup_stale_colbert_models(config, datasource_id, chunk_ds, normal_stale_models)
            stats["cleanup_invoked"] = True

        for model in db_only_current_models:
            stale_key = colbert_ds_model_key(model)
            try:
                with colbert_cleanup_lock(stale_key, wait=False) as key_acquired:
                    if not key_acquired:
                        stats["key_lock_conflicts"].append(model)
                        continue
                    coll_info = ColBERTStore.get_colbert_collection_info(config, model)
                    affected_ids = chunk_ds.get_chunk_ids(datasource_id, stale_key)
                    existing_recreated_ids = set()
                    if coll_info is not None and affected_ids:
                        cstore = ColBERTStore(config, model)
                        existing_recreated_ids = cstore.existing_ids(affected_ids)
                    if existing_recreated_ids:
                        logger.warning(
                            "Aktuelles ColBERT-Modell %s wurde seit AP1 mit %d "
                            "betroffenen Qdrant-Punkten neu angelegt; DB-only Cleanup "
                            "uebersprungen",
                            model,
                            len(existing_recreated_ids),
                        )
                        stats["skipped_recreated_current_models"].append(
                            {
                                "model": model,
                                "existing_ids": len(existing_recreated_ids),
                            }
                        )
                        continue
                    try:
                        _delete_colbert_datasource_embedding_counter(datasource_id, stale_key)
                    except Exception as exc:
                        stats["errors"].append({"model": model, "error": str(exc)})
                        continue
                    _delete_colbert_datasource_entries_with_graph_cleanup(
                        config,
                        chunk_ds,
                        datasource_id,
                        stale_key,
                    )
                    stats["db_only_current_models"].append(model)
                    stats["cleanup_invoked"] = True
            except Exception as exc:
                logger.warning(
                    "DB-only Current-Model-Cleanup fuer DS %s / %s fehlgeschlagen: %s",
                    datasource_id,
                    model,
                    exc,
                    exc_info=True,
                )
                stats["errors"].append({"model": model, "error": str(exc)})
        return stats


def cleanup_stale_colbert_collections_without_db(config, stale_collections):
    """Loescht stale ColBERT-Qdrant-Collections ohne chunk_datasources-Anker."""
    from kiara.backend_selector import normalize_model_name
    from kiara.retrieval.colbert import (
        colbert_cleanup_lock,
        colbert_collection_name,
        colbert_ds_model_key,
    )
    from kiara.shared.constants import DEFAULT_QDRANT_URL
    from kiara.shared.db import db_query

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return {"requested": [], "deleted": [], "skipped_with_datasources": [], "errors": []}

    qdrant_cfg = config.get("vectordb", {})
    client = QdrantClient(
        url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
        api_key=qdrant_cfg.get("qdrant_api_key") or None,
        timeout=qdrant_cfg.get("timeout", 60),
    )
    stats = {"requested": [], "deleted": [], "skipped_with_datasources": [], "errors": []}
    for item in stale_collections or []:
        datasources = item.get("datasources") or []
        stats["requested"].append(item)
        if datasources:
            stats["skipped_with_datasources"].append(item)
            continue
        model = normalize_model_name(item.get("model"))
        name = item.get("name")
        if not model or not name:
            stats["errors"].append({"collection": item, "error": "missing model/name"})
            continue
        try:
            expected_name = colbert_collection_name(model)
        except ValueError as exc:
            stats["errors"].append({"collection": name, "error": str(exc)})
            continue
        if name != expected_name:
            stats["errors"].append(
                {
                    "collection": name,
                    "error": f"name/model mismatch: expected {expected_name}",
                }
            )
            continue
        ds_key = colbert_ds_model_key(model)
        try:
            with colbert_cleanup_lock(ds_key, wait=False) as lock_acquired:
                if not lock_acquired:
                    stats["errors"].append({"collection": name, "error": "lock_conflict"})
                    continue
                with db_query() as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) AS cnt FROM chunk_datasources WHERE embedding_model = %s",
                        (ds_key,),
                    ).fetchone()
                anchor_count = row["cnt"] if row else 0
                if anchor_count > 0:
                    skipped = {**item, "anchor_count": anchor_count}
                    stats["skipped_with_datasources"].append(skipped)
                    continue
                client.delete_collection(collection_name=name)
                stats["deleted"].append(name)
        except Exception as exc:
            logger.warning(
                "Collection-only ColBERT-Cleanup fuer %s fehlgeschlagen: %s",
                name,
                exc,
                exc_info=True,
            )
            stats["errors"].append({"collection": name, "error": str(exc)})
    return stats


def _cleanup_colbert_model_switch_if_ready(
    config: dict,
    datasource_id: int | None,
    chunk_ds: ChunkDatasources,
    embedding_model: str,
    run_completed: bool,
    stats: dict,
) -> None:
    """Prueft Readiness-Guards und loest ColBERT-Model-Switch-Cleanup aus.

    Guards (alle muessen erfuellt sein):
    1. run_completed == True
    2. stats["errors"] == 0
    3. datasource_id gesetzt
    4. ColBERT fuer diese DS konfiguriert
    5. Aktuelles Modell hat DB-Eintraege im aktuellen colbert::-Key
    6. Alle Dense-Chunk-IDs existieren in der aktuellen ColBERT-Collection
       UND enthalten diese DS im ds_ids-Payload

    Bei Guard-Verletzung oder Fehler: Skip, kein Pipeline-Abort.
    """
    if not run_completed or stats.get("errors", 0) > 0:
        return
    if datasource_id is None:
        return
    if not config.get("retrieval", {}).get("colbert", {}).get("enabled", False):
        return

    from kiara.ingestion.vectordb_ops import _resolve_colbert_models_for_ds
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_datasource_lock,
        colbert_ds_model_key,
    )
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL

    try:
        current_model = normalize_model_name(
            config.get("retrieval", {}).get("colbert", {}).get("model", COLBERT_DEFAULT_MODEL)
        )
        current_key = colbert_ds_model_key(current_model)

        # Guard: aktuelles Modell hat DB-Eintraege fuer diese DS
        current_chunk_ids = chunk_ds.get_chunk_ids(datasource_id, current_key)
        if not current_chunk_ids:
            logger.debug(
                "ColBERT Cleanup: Kein aktuelles Modell (%s) fuer DS %s — Skip",
                current_key,
                datasource_id,
            )
            return

        # Guard: Collection existiert
        coll_info = ColBERTStore.get_colbert_collection_info(config, current_model)
        if coll_info is None:
            logger.debug(
                "ColBERT Cleanup: Collection fuer %s existiert nicht — Skip",
                current_model,
            )
            return

        # Guard: Dense-Chunk-IDs existieren in ColBERT UND enthalten DS
        dense_chunk_ids = chunk_ds.get_chunk_ids(datasource_id, embedding_model)
        if not dense_chunk_ids:
            return

        cstore = ColBERTStore(config, current_model)
        covered_ids = cstore.ids_with_datasource(dense_chunk_ids, datasource_id)
        if covered_ids != set(dense_chunk_ids):
            logger.debug(
                "ColBERT Cleanup: Coverage-Guard nicht erfuellt "
                "(%d/%d Dense-IDs in ColBERT) — Skip",
                len(covered_ids),
                len(dense_chunk_ids),
            )
            return

        # DS-Lock (non-blocking)
        with colbert_datasource_lock(datasource_id, wait=False) as acquired:
            if not acquired:
                logger.info(
                    "ColBERT Cleanup: DS %s uebersprungen (Lock-Konflikt)",
                    datasource_id,
                )
                return

            # Stale-Modelle ermitteln
            all_models = _resolve_colbert_models_for_ds(config, datasource_id, chunk_ds)
            stale_models = [m for m in all_models if m != current_model]
            if not stale_models:
                return

            logger.info(
                "ColBERT Model-Switch Cleanup: DS %s, aktuell=%s, stale=%s",
                datasource_id,
                current_model,
                stale_models,
            )
            _cleanup_stale_colbert_models(config, datasource_id, chunk_ds, stale_models)

    except Exception as exc:
        logger.warning(
            "ColBERT Model-Switch Detection fehlgeschlagen fuer DS %s (nicht-kritisch): %s",
            datasource_id,
            exc,
            exc_info=True,
        )


def run_ingestion(
    config: dict | None = None,
    incremental: bool = False,
    datasource_id: int | None = None,
    ds_overrides: dict | None = None,
    progress_callback=None,
    control_callback=None,
    embedding_model: str | None = None,
    file_log_callback=None,
    file_start_callback=None,
    resume_skip_files: set[str] | None = None,
    hash_mode: str = "mtime",
    status_callback=None,
    warmup_stats: dict | None = None,
) -> dict:
    """Run the full ingestion pipeline.

    Args:
        config: Configuration dict. Loaded from default if None.
        incremental: If True, only process changed/new files.
        datasource_id: If set, tag all chunks with this datasource ID.
        ds_overrides: Per-datasource config overrides (chunk_size, chunk_overlap, etc.).
        progress_callback: Optional callback(processed, total) called after each file.
        control_callback: Optional callback() returning "cancel" to abort or None to continue.
            The callback may block internally (e.g. for pause).
        file_start_callback: Optional callback(file_name, file_index, total_files) called
            before each file is processed.
        resume_skip_files: Set of file paths already processed in a previous job
            (from ingestion_file_logs). These files are skipped without re-processing
            and their hashes are persisted for future incremental runs.
        status_callback: Optional callback(phase_text) to update job phase in Admin-UI.
            Called with descriptive strings like "Warte auf Backend..." or
            "Verarbeite Dateien...".

    Returns:
        Stats dict with counts.
    """
    if config is None:
        config = load_config()
        # Marker: Config stammt aus load_config() — persistiert, re-readbar
        config["_kiara_enforce_persisted_colbert_model"] = True

    # Zentrale Embedding-Modell-Aufloesung (AP4)
    raw_embedding_model = (
        embedding_model
        if embedding_model and embedding_model.strip()
        else get_embedding_model(config)
    )
    embedding_model = normalize_model_name(raw_embedding_model)

    ctx = _apply_ds_overrides(config, ds_overrides)
    _check_ddu_preflight(ctx, warmup_stats)

    # Backend-Setup
    os.makedirs(config["vectordb"]["path"], exist_ok=True)

    store = VectorStore(config, embedding_model=embedding_model)
    chunk_ds = ChunkDatasources()
    batch_size = config["ingestion"]["batch_size"]

    all_files, files_to_process, file_hashes, deleted_files, hash_store, resumed_hashes = (
        _load_and_diff_files(
            config,
            store,
            incremental,
            datasource_id,
            embedding_model,
            resume_skip_files=resume_skip_files,
            hash_mode=hash_mode,
            chunk_ds=chunk_ds,
        )
    )

    if resumed_hashes:
        hash_store.upsert_many(datasource_id, embedding_model, resumed_hashes)

    # Hash-Cleanup fuer geloeschte Dateien — file_hashes UND
    # datasource_embeddings-Recount atomar, damit der 4-Schichten-Counter
    # nicht zwischen Mutationen divergiert (auch im CLI-Pfad ohne Worker-
    # Finalize und bei Pipeline-Crash vor _finalize_success).
    if deleted_files:
        from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings

        with db_transaction() as conn:
            hash_store.remove_files(
                datasource_id,
                embedding_model,
                deleted_files,
                conn=conn,
            )
            _recount_datasource_embeddings(conn, datasource_id, embedding_model)

    stats = _init_ingestion_stats(all_files, files_to_process, deleted_files, resumed_hashes)

    skipped = stats["skipped_files"]
    total = stats["total_files"]

    if progress_callback:
        progress_callback(skipped, total)

    if not files_to_process:
        hash_store.set_hash_mode(datasource_id, embedding_model, hash_mode)
        logger.info("No files to process.")
        # No-Files-Pfad gilt als run_completed=True
        _cleanup_colbert_model_switch_if_ready(
            config,
            datasource_id,
            chunk_ds,
            embedding_model,
            run_completed=True,
            stats=stats,
        )
        # CLI-Pfad: datasource_embeddings aus SoT (chunk_datasources +
        # file_hashes) recounten, damit der 4-Schichten-Counter ohne
        # Worker-Finalize konsistent bleibt. Im Worker-Pfad wird der Wert
        # direkt danach von _finalize_success ueberschrieben.
        if datasource_id is not None:
            from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings

            with db_transaction() as conn:
                _recount_datasource_embeddings(conn, datasource_id, embedding_model)
        return stats

    logger.info("Processing %d files...", len(files_to_process))

    # Context buendeln
    ictx = _IngestionContext(
        config=config,
        ctx=ctx,
        store=store,
        chunk_ds=chunk_ds,
        hash_store=hash_store,
        batch_size=batch_size,
        datasource_id=datasource_id,
        embedding_model=embedding_model,
        hash_mode=hash_mode,
        ds_overrides=ds_overrides,
        file_hashes=file_hashes,
        stats=stats,
        pending_hashes={},
        skipped=skipped,
        total=total,
        progress_callback=progress_callback,
        control_callback=control_callback,
        file_log_callback=file_log_callback,
        file_start_callback=file_start_callback,
        status_callback=status_callback,
    )

    # run_completed-Tracking: initial True, auf False bei Abort/Cancel
    run_completed = True
    for file_path in tqdm(files_to_process, desc="Loading & chunking"):
        if not _process_file_in_pipeline(ictx, file_path):
            run_completed = False
            break

    _persist_hashes_checkpoint(ictx, force=True)
    hash_store.set_hash_mode(datasource_id, embedding_model, hash_mode)

    # ColBERT Model-Switch Cleanup am Ende des Laufs
    _cleanup_colbert_model_switch_if_ready(
        config,
        datasource_id,
        chunk_ds,
        embedding_model,
        run_completed=run_completed,
        stats=stats,
    )

    # CLI-Pfad: datasource_embeddings aus SoT (chunk_datasources +
    # file_hashes) recounten, damit der 4-Schichten-Counter ohne
    # Worker-Finalize konsistent bleibt. Idempotent — im Worker-Pfad
    # ueberschreibt _finalize_success den Wert direkt danach via
    # _emb_repo.upsert_stats.
    if datasource_id is not None:
        from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings

        with db_transaction() as conn:
            _recount_datasource_embeddings(conn, datasource_id, embedding_model)

    return stats


def run_all_datasources(config: dict | None = None, incremental: bool = False) -> dict:
    """Run ingestion for all active datasources via Connector-Registry.

    Args:
        config: Base configuration dict. Loaded from default if None.
        incremental: If True, only process changed/new files.

    Returns:
        Aggregated stats dict.
    """
    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector

    ensure_loaded()

    if config is None:
        config = load_config()
        # Marker: Config stammt aus load_config() — persistiert
        config["_kiara_enforce_persisted_colbert_model"] = True

    from kiara.repositories.datasource import DatasourceRepository

    _ds_repo = DatasourceRepository()
    rows = _ds_repo.get_active()

    if not rows:
        logger.info("No active datasources found. Running default ingestion.")
        return run_ingestion(config, incremental=incremental)

    total_stats = {
        "total_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "deleted_files": 0,
        "total_chunks": 0,
        "errors": 0,
        "vlm_chunks": 0,
        "vlm_errors": 0,
        "ddu_files": 0,
        "ddu_remote_files": 0,
        "ddu_fallback_files": 0,
        "ddu_remote_errors": 0,
        "ddu_timeout": 0,
        "enrichment_chunks": 0,
        "enrichment_skipped": 0,
        "enrichment_errors": 0,
        "enrichment_time_ms": 0,
        "colbert_chunks_written": 0,
        "colbert_errors": 0,
        "vectordb_errors": 0,
        "vectordb_recoveries": 0,
        "datasources_processed": 0,
    }

    for row in rows:
        ds_config = copy.deepcopy(config)
        ds_type = row["type"]
        ds_id = row["id"]

        connector_cls = get_connector(ds_type)
        if not connector_cls:
            logger.error("Unknown datasource type '%s' for id=%s", ds_type, ds_id)
            total_stats["errors"] += 1
            continue

        hash_mode = getattr(connector_cls, "HASH_MODE", "mtime")

        conn_config = load_connector_config(row, connector_cls)

        path = connector_cls.safe_prepare_and_scan(conn_config, dict(row), ds_config)
        if path is None:
            total_stats["errors"] += 1
            continue

        ds_config["datasource"]["path"] = path
        logger.info("--- Datasource %s: %s (%s) -> %s ---", ds_id, row["name"], ds_type, path)

        ds_overrides = build_datasource_overrides(row)

        try:
            stats = run_ingestion(
                ds_config,
                incremental=incremental,
                datasource_id=ds_id,
                ds_overrides=ds_overrides or None,
                hash_mode=hash_mode,
            )
            for key in (
                "total_files",
                "processed_files",
                "skipped_files",
                "deleted_files",
                "total_chunks",
                "errors",
                "vlm_chunks",
                "vlm_errors",
                "ddu_files",
                "ddu_remote_files",
                "ddu_fallback_files",
                "ddu_remote_errors",
                "ddu_timeout",
                "enrichment_chunks",
                "enrichment_skipped",
                "enrichment_errors",
                "enrichment_time_ms",
                "colbert_chunks_written",
                "colbert_errors",
                "vectordb_errors",
                "vectordb_recoveries",
            ):
                total_stats[key] += stats.get(key, 0)
            if stats.get("colbert_collection"):
                total_stats["colbert_collection"] = stats["colbert_collection"]
            total_stats["datasources_processed"] += 1
        except Exception as exc:
            logger.error("Datasource %s failed: %s", ds_id, exc, exc_info=True)
            total_stats["errors"] += 1

    return total_stats
