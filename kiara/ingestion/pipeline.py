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
from contextlib import ExitStack, nullcontext
from functools import wraps

logger = logging.getLogger(__name__)


def _stale_chunk_owners(chunk_ds, chunk_ids, embedding_model, *, include_colbert):
    """Ermittelt echte Owner der betroffenen IDs, getrennt nach Modell."""
    owners = {embedding_model: set(chunk_ds.get_other_datasources(chunk_ids, -1, embedding_model))}
    if include_colbert and chunk_ids:
        for model in chunk_ds.distinct_models():
            if model.startswith("colbert::"):
                model_owners = set(chunk_ds.get_other_datasources(chunk_ids, -1, model))
                if model_owners:
                    owners[model] = model_owners
    return owners


def _guard_stale_chunk_cleanup(function):
    """Fence all affected vector keyspaces across four-layer cleanup."""

    @wraps(function)
    def wrapped(
        config,
        store,
        hash_store,
        chunk_ds,
        datasource_id,
        embedding_model,
        files_to_delete,
    ):
        from kiara.ingestion.html_migration import validate_run_selection
        from kiara.retrieval.colbert import model_name_from_colbert_ds_key
        from kiara.retrieval.index_compatibility import (
            colbert_index_target,
            guard_index,
            source_cleanup_fence,
            stored_primary_index_target,
        )

        selection = validate_run_selection(config, datasource_id, embedding_model)
        if selection is not None and not set(files_to_delete).issubset(selection["paths"]):
            raise ValueError("Datei-Cleanup liegt ausserhalb der HTML-Migrationsauswahl")
        if chunk_ds is None:
            chunk_ds = ChunkDatasources()
        if not files_to_delete:
            return
        with ExitStack() as stack:
            # Freeze writers, including collection creation, before discovery.
            # HTML migrations are explicitly primary-only and need no inventory.
            if selection is None:
                stack.enter_context(source_cleanup_fence())
            primary = stored_primary_index_target(config, embedding_model)
            stack.enter_context(
                guard_index(
                    config,
                    primary,
                    operation="incremental_reingest_cleanup",
                    write=True,
                    validation_scope="stored_binding",
                )
            )
            models = frozenset()
            if selection is None:
                from kiara.ingestion.cleanup_targets import resolve_source_colbert_keys

                models = resolve_source_colbert_keys(
                    config, store, chunk_ds, datasource_id, embedding_model, files_to_delete
                )
            targets = [
                colbert_index_target(config, model_name_from_colbert_ds_key(model))
                for model in models
            ]
            for target in sorted(targets, key=lambda item: item.logical_index_key):
                stack.enter_context(
                    guard_index(
                        config,
                        target,
                        operation="incremental_reingest_cleanup",
                        write=True,
                        validation_scope="stored_binding",
                    )
                )
            return function(
                config,
                store,
                hash_store,
                chunk_ds,
                datasource_id,
                embedding_model,
                files_to_delete,
                colbert_keys=models,
            )

    return wrapped


from kiara_ingestion_core.file_diff import _compute_file_diff
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
    _colbert_enabled_for_run,
    _determine_chunks,
    _get_current_job_id,
    _IngestionContext,
    _init_ingestion_stats,
    _persist_hashes_checkpoint,
    _process_file_in_pipeline,
    _wait_for_index_contract_target_recovery,
    set_current_job_id,
)
from kiara.retrieval.index_compatibility import IndexContractUnavailableError
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

    from kiara.config import get_mail_archive_config

    return ProcessingContext(
        mail_archive=get_mail_archive_config(config),
        ddu_enabled=ddu_enabled,
        ddu_backends=ddu_backends,
        ddu_ocr_images=ddu_ocr_images,
        vlm_enabled=vlm_enabled,
        vlm_model=vlm_model,
        vlm_url=vlm_url,
        vlm_prompt=vlm_prompt,
    )


@_guard_stale_chunk_cleanup
def _cleanup_stale_chunks(
    config,
    store,
    hash_store,
    chunk_ds,
    datasource_id,
    embedding_model,
    files_to_delete,
    *,
    colbert_keys=(),
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
    from kiara.ingestion.html_migration import validate_run_selection

    selection = validate_run_selection(config, datasource_id, embedding_model)
    if selection is not None and not set(files_to_delete).issubset(selection["paths"]):
        raise ValueError("Datei-Cleanup liegt ausserhalb der HTML-Migrationsauswahl")
    if not files_to_delete:
        return

    from kiara.ingestion.document_publication import (
        invalidate_documents,
        pending_cleanup_models,
        pending_source_cleanups,
    )

    collection = str(store.get_collection_name())
    if selection is not None:
        # Validate model scope without loading source chunk inventories.
        excluded_models = {
            key
            for key in pending_cleanup_models(collection, files_to_delete)
            if not key.startswith("es::")
        } - {embedding_model}
        if excluded_models:
            raise RuntimeError(
                "HTML-Migration mit offenem Cleanup-Inventar ausserhalb des Primaermodells: "
                f"{', '.join(sorted(excluded_models))}. "
                "Zuerst den vollstaendigen Quellen-Cleanup erneut ausfuehren."
            )

    if config.get("_html_reingest") is not None:
        from kiara.ingestion.html_reingest import (
            assert_exclusive_sources,
            validate_prepared_sources,
        )

        if chunk_ds is None or datasource_id is None:
            raise ValueError("HTML-Neuaufbereitung braucht eine bestaetigte DS-Zuordnung")
        validate_prepared_sources(config, files_to_delete)
        assert_exclusive_sources(store, chunk_ds, datasource_id, embedding_model, files_to_delete)

    # Auch HTML-Neuaufbereitung entwertet Hash und Dateizaehler erst zusammen
    # mit Lesesperre und dauerhaftem Cleanup-Auftrag, vor jeder Indexmutation.
    from kiara.retrieval.elasticsearch_service import model_uses_elasticsearch

    use_es = model_uses_elasticsearch(config, embedding_model)
    invalidate_documents(
        collection, files_to_delete, embedding_model, cleanup=True, cleanup_es=use_es
    )
    # Bounded batches, one invalidation for the entire job. Every batch keeps
    # PG deletion and counter maintenance atomic, including on a later failure.
    for offset in range(0, len(files_to_delete), 100):
        batch = files_to_delete[offset : offset + 100]
        pending = pending_source_cleanups(collection, sources=batch)
        _cleanup_source_batch(
            config,
            store,
            hash_store,
            chunk_ds,
            datasource_id,
            embedding_model,
            batch,
            selection,
            collection,
            use_es,
            pending,
            colbert_keys,
        )


def _cleanup_source_batch(
    config,
    store,
    hash_store,
    chunk_ds,
    datasource_id,
    embedding_model,
    files_to_delete,
    selection,
    collection,
    use_es,
    pending,
    colbert_keys,
):
    from kiara.ingestion.document_publication import finish_source_cleanup, remember_source_cleanup
    from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings, invalidate_bm25

    if chunk_ds is None:
        chunk_ds = ChunkDatasources()
    import json

    from kiara.retrieval.colbert import ColBERTStore, model_name_from_colbert_ds_key
    from kiara.retrieval.elasticsearch_indices import ElasticsearchDocumentAddress, document_address
    from kiara.retrieval.elasticsearch_service import (
        delete_by_ids,
        get_ids_by_source,
        targets_for_datasources,
    )

    es_key = f"es::{embedding_model}"
    es_addresses = set()

    colbert_stores = {
        key: ColBERTStore(config, model_name_from_colbert_ds_key(key), intent="maintenance")
        for key in sorted(colbert_keys)
    }
    old_chunk_ids = set()
    # Das gespeicherte Inventar bleibt erhalten, wenn ColBERT schon geloescht
    # wurde, aber PG/Graph, ES oder Dense fehlschlagen. Keine Texte/Vektoren kopieren.
    for source in files_to_delete:
        ids = set(store.get_ids_by_sources([source]))
        # Auch bei Verlust beider Vektorbestände kann ES die Quellen-IDs
        # noch belegen. Vor allen Deletes in das dauerhafte Inventar aufnehmen.
        saved = pending.get(source, {})
        source_addresses = {
            ElasticsearchDocumentAddress(*json.loads(value)) for value in saved.get(es_key, [])
        }
        ids.update(a.chunk_id for a in source_addresses)

        for model in (embedding_model, *colbert_stores):
            ids.update(saved.get(model, []))
        matched_keys = set(saved) & set(colbert_stores)
        for key, cstore in colbert_stores.items():
            colbert_ids = cstore.get_ids_by_source(source)
            if colbert_ids:
                ids.update(colbert_ids)
                matched_keys.add(key)
        owners = _stale_chunk_owners(
            chunk_ds, list(ids), embedding_model, include_colbert=selection is None
        )
        matched_keys.update(set(owners) - {embedding_model})
        if use_es:
            target_ids = (
                {int(ds) for ds in saved.get(es_key + "::owners", [])}
                | set(owners.get(embedding_model, []))
                | {a.datasource_id for a in source_addresses}
            )
            if datasource_id is not None:
                target_ids.add(datasource_id)
            targets = targets_for_datasources(config, target_ids, embedding_model)
            if source_addresses:
                # Durable addresses authorize finishing a previous deletion even
                # after its ES writes succeeded and vectors/PG cleanup failed.
                from kiara.retrieval.elasticsearch_indices import DatasourceIndexTarget

                targets = tuple(
                    DatasourceIndexTarget(t.datasource_id, t.index_name, 0) for t in targets
                )
            source_addresses.update(get_ids_by_source(config, source, targets=targets))
            # PG owns even currently missing ES documents; keep their complete
            # addresses so retry cannot lose the physical cleanup scope.
            memberships = chunk_ds.get_datasource_ids_for_chunks(list(ids), embedding_model)
            source_addresses.update(
                document_address(config, ds, cid)
                for cid, ds_ids in memberships.items()
                for ds in ds_ids
            )
            ids.update(a.chunk_id for a in source_addresses)
            es_addresses.update(source_addresses)
            # ES can be the last source-to-ID evidence after both vector stores
            # vanished. Include PG-confirmed ColBERT owners in the durable retry.
            owners = _stale_chunk_owners(
                chunk_ds, list(ids), embedding_model, include_colbert=selection is None
            )
            matched_keys.update(set(owners) - {embedding_model})
        if matched_keys - set(colbert_keys):
            raise RuntimeError("ColBERT-Cleanup-Ziel ausserhalb der geprueften Auswahl")
        remember_source_cleanup(
            collection,
            source,
            {
                **{model: sorted(ids) for model in (embedding_model, *matched_keys)},
                **(
                    {
                        es_key: [
                            json.dumps([a.datasource_id, a.index_name, a.chunk_id])
                            for a in source_addresses
                        ]
                    }
                    if use_es
                    else {}
                ),
            },
        )
        old_chunk_ids.update(ids)
    old_chunk_ids = sorted(old_chunk_ids)
    if old_chunk_ids:
        owners = _stale_chunk_owners(
            chunk_ds, old_chunk_ids, embedding_model, include_colbert=selection is None
        )
        if set(owners) - {embedding_model, *colbert_keys}:
            raise RuntimeError("ColBERT-Cleanup-Owner ausserhalb der geprueften Auswahl")
        # Auch CLI ohne DS ersetzt den ganzen Quellenbestand. PostgreSQL
        # bestimmt die Owner, einschliesslich reiner ColBERT-Zuordnungen.
        for cstore in colbert_stores.values():
            cstore.delete_by_ids(old_chunk_ids)
            if cstore.existing_ids(old_chunk_ids):
                raise RuntimeError("ColBERT-Quellen-Cleanup ist unvollstaendig")

        # PG-Zuordnungen, Graph und Statistiken aendern sich atomar. Dense
        # bleibt bei einem Fehler als dateipfadbasierter Retry-Anker erhalten.
        with db_transaction() as conn:
            chunk_ds.acquire_graph_cleanup_lock(conn)
            for model, ds_ids in owners.items():
                chunk_ds.delete_by_chunk_ids(old_chunk_ids, model, conn=conn)
                for ds_id in ds_ids:
                    hash_store.remove_files(ds_id, model, files_to_delete, conn=conn)
                    _recount_datasource_embeddings(conn, ds_id, model)
            graph_chunk_ids = chunk_ds.filter_unreferenced_chunk_ids(old_chunk_ids, conn=conn)
            if graph_chunk_ids:
                from kiara.graph.store import GraphStore

                GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)

    # Quellenweite ES-Loeschung, Dense zuletzt, BM25 nach der Dense-Mutation.
    # Ohne alte IDs bleiben diese dateipfadbasierten Schritte verpflichtend.
    if use_es:
        delete_by_ids(config, list(es_addresses))
    store.delete_by_sources(files_to_delete)
    invalidate_bm25(store.get_collection_name())
    if store.get_ids_by_sources(files_to_delete):
        raise RuntimeError("Dense-Quellen-Cleanup ist unvollstaendig")
    for model in (embedding_model, *colbert_stores):
        if chunk_ds.get_datasource_ids_for_chunks(old_chunk_ids, model):
            raise RuntimeError("PG-Quellen-Cleanup ist unvollstaendig")
    finish_source_cleanup(collection, files_to_delete)


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

    Entfernte Dateien und explizite Neuindizierungen werden bereinigt.
    Inkrementelle Kandidaten werden nach dem Laden mit ihrem gespeicherten
    Eingangstext verglichen, bevor vorhandene Chunks geloescht werden.

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
    from kiara.ingestion.document_publication import (
        pending_cleanup_sources,
        pending_document_sources,
    )
    from kiara.ingestion.html_migration import validate_run_selection
    from kiara.ingestion.processing_revision import processing_file_hash

    selection = validate_run_selection(config, datasource_id, embedding_model)
    hash_store = HashStore()

    file_stats = {}
    all_files = (
        list(selection["paths"])
        if selection is not None
        else scan_files(config, file_stats=file_stats)
    )
    # An explicit migration replays only its selected files, even when their
    # completion hashes already match. Unselected paths never enter the diff.
    stored_hashes = {} if selection is not None else hash_store.load(datasource_id, embedding_model)
    missing_size_paths = (
        hash_store.get_missing_size_paths(datasource_id) if incremental and stored_hashes else set()
    )

    # Hash-Mode-Migration: Wenn der Connector jetzt einen anderen Modus
    # verwendet als beim letzten Lauf, Dateiabschluesse erneut verifizieren.
    if incremental and stored_hashes:
        stored_mode = hash_store.get_hash_mode(datasource_id, embedding_model)
        if stored_mode != hash_mode:
            logger.info(
                "Hash-Mode-Wechsel erkannt: %s → %s (ds=%s) — pruefe Dateiabschluesse erneut",
                stored_mode,
                hash_mode,
                datasource_id,
            )
            hash_store.migrate_hash_mode(datasource_id, embedding_model, hash_mode)
            stored_hashes = hash_store.load(datasource_id, embedding_model)
            missing_size_paths = hash_store.get_missing_size_paths(datasource_id)

    _candidates, file_hashes, deleted_files = _compute_file_diff(
        all_files,
        incremental,
        stored_hashes,
        hash_mode=hash_mode,
        file_stats=file_stats,
    )
    if selection is not None and set(file_hashes) != set(all_files):
        missing = sorted(set(all_files) - set(file_hashes))
        raise ValueError(
            "Vorvalidierte HTML-Migrationsdatei fehlt oder ist keine regulaere Datei: "
            + ", ".join(missing)
        )

    file_hashes = {
        path: (processing_file_hash(config, path, digest), size)
        for path, (digest, size) in file_hashes.items()
    }
    pending_sources = pending_document_sources(str(store.get_collection_name()), list(file_hashes))
    if datasource_id == 0:
        # CLI-Hashes bleiben als Pfadinventar erhalten, auch nach Verlust des
        # Owners. Ein passender Hash allein bestaetigt dessen Zuordnung nicht.
        cli_chunk_ds = chunk_ds if chunk_ds is not None else ChunkDatasources()
        for path, (digest, _size) in file_hashes.items():
            if stored_hashes.get(path) != digest or path in pending_sources:
                continue
            with store._index_contract_guard(operation="cli_hash_memberships", write=False):
                ids = store.get_ids_by_sources([path])
                owners = cli_chunk_ds.get_datasource_ids_for_chunks(ids, embedding_model)
                if not ids or any(0 not in owners.get(cid, []) for cid in ids):
                    pending_sources.add(path)
    files_to_process = [
        path
        for path, (digest, _size) in file_hashes.items()
        if not incremental or stored_hashes.get(path) != digest or path in pending_sources
    ]
    # Disappeared/non-regular files are absent from the validated diff.
    file_stats.clear()
    all_files = list(file_hashes)
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

    # Ein Log ist kein Dateiabschluss. Auch ein passender Hash darf eine
    # noch offene dokumentweite Veroeffentlichung nicht ueberspringen.
    resumed_hashes = {}

    # Neue DS und Wiederanlaeufe koennen unveraenderte, bereits angereicherte
    # Chunks uebernehmen. Erst die Dateivorbereitung entscheidet ueber deren
    # Wiederverwendung. Ein expliziter Voll-/Migrationslauf erzwingt das Cleanup.
    forced_cleanup = files_to_process if not incremental or selection is not None else []
    files_to_delete = list(dict.fromkeys([*forced_cleanup, *deleted_files]))
    # Ein voriger Cleanup hat Hashes bereits entwertet. Auch inzwischen
    # entfernte Dateien brauchen dann den gespeicherten Wiederanlaufanker.
    root = os.path.abspath(config["datasource"]["path"])
    for source in pending_cleanup_sources(str(store.get_collection_name())):
        if selection is not None:
            in_scope = source in selection["paths"]
        else:
            in_scope = os.path.commonpath([root, os.path.abspath(source)]) == root
        if in_scope and source not in files_to_delete:
            files_to_delete.append(source)

    _cleanup_stale_chunks(
        config, store, hash_store, chunk_ds, datasource_id, embedding_model, files_to_delete
    )

    return all_files, files_to_process, file_hashes, files_to_delete, hash_store, resumed_hashes


def _prepare_file_chunks(
    config,
    store,
    hash_store,
    chunk_ds,
    datasource_id,
    embedding_model,
    path,
    metadatas,
    ids,
) -> bool:
    """Gesunde Teilbestaende behalten; andere Eingangsdarstellungen ersetzen.

    Der Aufrufer haelt die Quellsperre und hat ihre Freigabe bereits geoeffnet.
    IDs binden nur Quelle, Parser und Position. Die vor Enrichment gespeicherte
    Eingangs-Version bindet zusaetzlich Text und Chunk-Grenzen.
    """
    expected = dict(zip(ids, metadatas, strict=True))
    existing_ids = store.get_ids_by_sources([path])
    # Bei leerem Loader-Ergebnis koennen ausschliesslich ColBERT-Punkte und
    # PG-Zuordnungen uebrig sein. Auch ohne Dense-Anker quellenweit bereinigen.
    replace = not expected or bool(set(existing_ids) - set(expected))
    if not replace:
        from kiara.shared.constants import QDRANT_SCROLL_BATCH_SIZE

        for start in range(0, len(existing_ids), QDRANT_SCROLL_BATCH_SIZE):
            batch = existing_ids[start : start + QDRANT_SCROLL_BATCH_SIZE]
            stored = store.get_documents_by_ids(batch, include=["metadatas"])
            for cid, meta in zip(stored["ids"], stored["metadatas"], strict=True):
                current = expected[cid]
                if (
                    not meta.get("processing_input_version")
                    or meta["processing_input_version"] != current.get("processing_input_version")
                    or meta.get("modified") != current.get("modified")
                    or meta.get("size_bytes") != current.get("size_bytes")
                ):
                    replace = True
                    break
            if replace:
                break
    if replace:
        _cleanup_stale_chunks(
            config, store, hash_store, chunk_ds, datasource_id, embedding_model, [path]
        )
    return replace


def _complete_file(
    hash_store,
    datasource_id,
    embedding_model,
    file_path,
    file_hash,
    *,
    sync_model_sizes=True,
    additional_models=(),
):
    """Commit the completion hash and model counts after all required writes."""
    from kiara.ingestion.document_publication import complete_document_publication
    from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings

    with db_transaction() as conn:
        # Wie invalidate_documents zuerst die Freigabezeile sperren, dann
        # Hashes/Zaehler. Sichtbar wird alles gemeinsam beim Commit.
        complete_document_publication(conn, file_path, embedding_model, datasource_id=datasource_id)
        hash_store.upsert_many(
            datasource_id,
            embedding_model,
            {file_path: file_hash},
            conn=conn,
            sync_model_sizes=sync_model_sizes,
        )
        for model in dict.fromkeys((embedding_model, *additional_models)):
            _recount_datasource_embeddings(
                conn, 0 if datasource_id is None else int(datasource_id), model
            )


# ---------------------------------------------------------------------------
# ColBERT Model-Switch Cleanup (AP2 + AP3)
# ---------------------------------------------------------------------------


def _delete_colbert_datasource_embedding_counter(
    datasource_id: int, stale_key: str, *, conn=None
) -> None:
    """Loescht den datasource_embeddings-Counter fuer eine DS + stale ColBERT-Key.

    Fehlende Zeile ist ein erfolgreicher No-op (idempotent).
    Counter-Delete-Fehler werden geloggt und weitergeworfen, damit der Caller
    nur dieses Modell abbricht und
    chunk_datasources als Retry-Anker erhalten bleibt.
    """
    try:
        with nullcontext(conn) if conn is not None else db_transaction() as conn:
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
    *,
    conn=None,
) -> int:
    """Loescht stale ColBERT-Referenzen und bereinigt Graph-Orphans atomar."""
    with nullcontext(conn) if conn is not None else db_transaction() as conn:
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
    4. Collection droppen falls leer und ohne weitere PG-Owner
    5. DB-Cleanup inklusive Inventar und Vertragsentfernung atomar abschliessen

    Gewoehnliche Fehler sind nicht-kritisch — Self-Healing beim naechsten
    Lauf. IndexCompatibilityError ist dagegen ein nicht degradierbarer
    Integritaetsfehler und wird unveraendert weitergereicht.
    """
    from kiara.ingestion.document_publication import finish_colbert_model_cleanup
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_cleanup_lock,
        colbert_ds_model_key,
    )
    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        colbert_index_target,
        guard_index,
        retire_index_contract,
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

                target = colbert_index_target(config, raw_model)
                with guard_index(
                    config,
                    target,
                    operation="stale_colbert_model_cleanup",
                    write=True,
                    allow_missing_for_retire=True,
                    validation_scope="stored_binding",
                ) as token:
                    coll_info = ColBERTStore.get_colbert_collection_info(config, stale_model)
                    # Andere Owner brauchen die leere Collection und ihren
                    # Vertrag noch fuer ihre eigene Reparatur/Bereinigung.
                    other_owners = token.connection.execute(
                        "SELECT 1 FROM chunk_datasources "
                        "WHERE embedding_model = %s AND datasource_id <> %s LIMIT 1",
                        (stale_key, datasource_id),
                    ).fetchone()

                    if coll_info is None:
                        # Keine Collection — nur DB bereinigen
                        logger.info(
                            "ColBERT-Collection fuer %s existiert nicht — nur DB-Cleanup",
                            stale_model,
                        )
                        _delete_colbert_datasource_embedding_counter(
                            datasource_id, stale_key, conn=token.connection
                        )
                        _delete_colbert_datasource_entries_with_graph_cleanup(
                            config,
                            chunk_ds,
                            datasource_id,
                            stale_key,
                            conn=token.connection,
                        )
                        if not other_owners:
                            retire_index_contract(token)
                            finish_colbert_model_cleanup(token.connection, stale_key)
                        continue

                    if coll_info["chunks"] == 0:
                        # Leere Collection droppen, keine Qdrant-Mutation
                        logger.info(
                            "ColBERT-Collection %s ist leer — droppe und bereinige DB",
                            coll_info["name"],
                        )
                        try:
                            dropped = False
                            if not other_owners:
                                cstore = ColBERTStore(config, stale_model, intent="maintenance")
                                dropped = cstore.drop_collection_if_empty()
                        except IndexCompatibilityError:
                            raise
                        except Exception as exc:
                            logger.warning(
                                "Drop leere ColBERT-Collection %s fehlgeschlagen: %s "
                                "(chunk_datasources bleibt als Retry-Anker)",
                                stale_key,
                                exc,
                                exc_info=True,
                            )
                            raise
                        _delete_colbert_datasource_embedding_counter(
                            datasource_id, stale_key, conn=token.connection
                        )
                        _delete_colbert_datasource_entries_with_graph_cleanup(
                            config,
                            chunk_ds,
                            datasource_id,
                            stale_key,
                            conn=token.connection,
                        )
                        if dropped:
                            finish_colbert_model_cleanup(token.connection, stale_key)
                        continue

                    # Collection existiert und hat Punkte — Qdrant-Cleanup
                    orphan_ids, non_orphan_ds_ids = chunk_ds.preview_datasource_removal(
                        datasource_id, stale_key
                    )

                    cstore = ColBERTStore(config, stale_model, intent="maintenance")

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
                        dropped = cstore.drop_collection_if_empty() if not other_owners else False
                    except IndexCompatibilityError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "drop_collection_if_empty fehlgeschlagen fuer %s: %s "
                            "(chunk_datasources bleibt als Retry-Anker)",
                            stale_key,
                            exc,
                            exc_info=True,
                        )
                        raise

                    # DB-Cleanup: erst Counter, dann chunk_datasources
                    _delete_colbert_datasource_embedding_counter(
                        datasource_id, stale_key, conn=token.connection
                    )
                    _delete_colbert_datasource_entries_with_graph_cleanup(
                        config,
                        chunk_ds,
                        datasource_id,
                        stale_key,
                        conn=token.connection,
                    )
                    if dropped:
                        finish_colbert_model_cleanup(token.connection, stale_key)

                    logger.info(
                        "ColBERT-Cleanup abgeschlossen: DS %s, Modell %s "
                        "(%d Orphans, %d Shared-Updates)",
                        datasource_id,
                        stale_model,
                        len(orphan_ids),
                        len(non_orphan_ds_ids),
                    )

        except IndexCompatibilityError:
            raise
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
    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        colbert_index_target,
        guard_index,
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
                    target = colbert_index_target(config, model)
                    with guard_index(
                        config,
                        target,
                        operation="current_colbert_model_cleanup",
                        write=True,
                        allow_missing_for_retire=True,
                        validation_scope="stored_binding",
                    ) as token:
                        coll_info = ColBERTStore.get_colbert_collection_info(config, model)
                        affected_ids = chunk_ds.get_chunk_ids(datasource_id, stale_key)
                        existing_recreated_ids = set()
                        if coll_info is not None and affected_ids:
                            cstore = ColBERTStore(config, model, intent="maintenance")
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
                        _delete_colbert_datasource_embedding_counter(
                            datasource_id, stale_key, conn=token.connection
                        )
                        _delete_colbert_datasource_entries_with_graph_cleanup(
                            config,
                            chunk_ds,
                            datasource_id,
                            stale_key,
                            conn=token.connection,
                        )
                        if (
                            coll_info is None
                            and not token.connection.execute(
                                "SELECT 1 FROM chunk_datasources WHERE embedding_model = %s LIMIT 1",
                                (stale_key,),
                            ).fetchone()
                        ):
                            from kiara.ingestion.document_publication import (
                                finish_colbert_model_cleanup,
                            )
                            from kiara.retrieval.index_compatibility import retire_index_contract

                            retire_index_contract(token)
                            finish_colbert_model_cleanup(token.connection, stale_key)
                    stats["db_only_current_models"].append(model)
                    stats["cleanup_invoked"] = True
            except IndexCompatibilityError:
                raise
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
                from kiara.retrieval.index_compatibility import (
                    IndexCompatibilityError,
                    colbert_index_target,
                    guard_index,
                    retire_index_contract,
                )

                contract_target = colbert_index_target(config, model)
                with guard_index(
                    config,
                    contract_target,
                    operation="stale_colbert_collection_delete",
                    write=True,
                    allow_missing_for_retire=True,
                    qdrant_client=client,
                    validation_scope="stored_binding",
                ) as token:
                    # Die vorangehende Bestandsabfrage lag vor dem Index-Guard.
                    # Neue Zuordnungen duerfen nicht mit der Collection verschwinden.
                    if token.connection.execute(
                        "SELECT 1 FROM chunk_datasources WHERE embedding_model = %s LIMIT 1",
                        (ds_key,),
                    ).fetchone():
                        stats["skipped_with_datasources"].append(item)
                        continue
                    # Nach externem Delete und PG-Rollback existiert nur noch
                    # der verifizierte Vertrag. Auch diesen Abschluss wiederholen.
                    if client.collection_exists(collection_name=name):
                        deleted = client.delete_collection(collection_name=name)
                        if deleted is False:
                            raise RuntimeError(f"ColBERT-Collection {name!r} nicht geloescht")
                    retire_index_contract(token)
                    from kiara.ingestion.document_publication import finish_colbert_model_cleanup

                    finish_colbert_model_cleanup(token.connection, ds_key)
                stats["deleted"].append(name)
        except IndexCompatibilityError:
            raise
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

    Fachliche Readiness-Verletzungen und gewoehnliche Fehler bleiben ein Skip.
    IndexCompatibilityError wird unveraendert zur Pipeline-Orchestrierung
    weitergereicht und beendet den Job strukturiert.
    """
    if not run_completed or stats.get("errors", 0) > 0:
        return
    if datasource_id is None or datasource_id == 0:
        return
    if not config.get("retrieval", {}).get("colbert", {}).get("enabled", False):
        return

    from kiara.ingestion.vectordb_ops import _resolve_colbert_models_for_ds
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_datasource_lock,
        colbert_ds_model_key,
    )
    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        colbert_index_target,
        guard_index,
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

        # Guard: Dense-Chunk-IDs existieren in ColBERT UND enthalten DS
        dense_chunk_ids = chunk_ds.get_chunk_ids(datasource_id, embedding_model)
        if not dense_chunk_ids:
            return

        # DS-Lock (non-blocking) muss vor den Index-Fences liegen; Writer nutzen
        # dieselbe Reihenfolge und koennen so nicht mit dem Cleanup deadlocken.
        with colbert_datasource_lock(datasource_id, wait=False) as acquired:
            if not acquired:
                logger.info(
                    "ColBERT Cleanup: DS %s uebersprungen (Lock-Konflikt)",
                    datasource_id,
                )
                return

            # Stale-Modelle ermitteln
            all_models = _resolve_colbert_models_for_ds(config, datasource_id, chunk_ds)
            stale_models = sorted(m for m in all_models if m != current_model)
            if not stale_models:
                return

            targets = [colbert_index_target(config, current_model)]
            targets.extend(colbert_index_target(config, model) for model in stale_models)
            unique_targets = {target.logical_index_key: target for target in targets}
            with ExitStack() as stack:
                for target in sorted(
                    unique_targets.values(), key=lambda item: item.logical_index_key
                ):
                    stack.enter_context(
                        guard_index(
                            config,
                            target,
                            operation="colbert_model_switch_cleanup",
                            write=True,
                            validation_scope="stored_binding",
                        )
                    )

                # Collection und Coverage werden unter denselben Fences wie das
                # nachfolgende stale Cleanup gelesen. Kein TOCTOU zwischen
                # Freigabe des aktuellen Modells und erster Mutation.
                coll_info = ColBERTStore.get_colbert_collection_info(config, current_model)
                if coll_info is None:
                    logger.debug(
                        "ColBERT Cleanup: Collection fuer %s existiert nicht — Skip",
                        current_model,
                    )
                    return

                cstore = ColBERTStore(config, current_model, intent="maintenance")
                covered_ids = cstore.ids_with_datasource(dense_chunk_ids, datasource_id)
                if covered_ids != set(dense_chunk_ids):
                    logger.debug(
                        "ColBERT Cleanup: Coverage-Guard nicht erfuellt "
                        "(%d/%d Dense-IDs in ColBERT) — Skip",
                        len(covered_ids),
                        len(dense_chunk_ids),
                    )
                    return

                logger.info(
                    "ColBERT Model-Switch Cleanup: DS %s, aktuell=%s, stale=%s",
                    datasource_id,
                    current_model,
                    stale_models,
                )
                _cleanup_stale_colbert_models(config, datasource_id, chunk_ds, stale_models)

    except IndexCompatibilityError:
        raise
    except Exception as exc:
        logger.warning(
            "ColBERT Model-Switch Detection fehlgeschlagen fuer DS %s (nicht-kritisch): %s",
            datasource_id,
            exc,
            exc_info=True,
        )


def _run_colbert_model_switch_cleanup(
    config: dict,
    datasource_id: int | None,
    chunk_ds: ChunkDatasources,
    embedding_model: str,
    run_completed: bool,
    stats: dict,
) -> bool:
    """Orchestrierungsgrenze fuer einen strukturierten Compatibility-ABORT."""
    from kiara.retrieval.index_compatibility import IndexCompatibilityError

    try:
        _cleanup_colbert_model_switch_if_ready(
            config,
            datasource_id,
            chunk_ds,
            embedding_model,
            run_completed=run_completed,
            stats=stats,
        )
    except IndexCompatibilityError as exc:
        diagnostic = exc.to_dict()
        detail = {
            "phase": "index_compatibility",
            "reason_code": exc.reason_code,
            "operation": exc.operation,
            "consumer": "colbert_model_switch_cleanup",
            "diagnostic": diagnostic,
        }
        stats["errors"] = stats.get("errors", 0) + 1
        stats["index_compatibility_error"] = diagnostic
        stats.setdefault("error_details", []).append(detail)
        stats["abort_reason"] = f"Index-Kompatibilitaetsguard: {exc.reason_code} ({exc.detail})"
        logger.error(
            "ColBERT-Modellwechsel am Index-Kompatibilitaetsguard blockiert: %s",
            exc,
            exc_info=True,
        )
        return False
    return True


def _refresh_search_on_exit(function):
    @wraps(function)
    def wrapped(config=None, *args, **kwargs):
        from kiara.retrieval.elasticsearch_service import ingestion_visibility

        if config is None:
            config = load_config()
            config["_kiara_enforce_persisted_colbert_model"] = True
        with ingestion_visibility(config, refresh_on_success=False) as visibility:
            stats = function(config, *args, **kwargs)
            visibility["stats"] = stats
            # Hashes can outlive a failed refresh, including across processes.
            # Adopting them or existing chunks requires a completion barrier.
            # Early preflight/contract aborts do not authorize a backend write.
            visibility["completed"] = bool(
                (stats.get("skipped_files") or stats.get("dedup_chunks"))
                and not stats.get("errors")
                and not stats.get("abort_reason")
            )
            return stats

    return wrapped


def _ensure_cli_datasource() -> int:
    """Bindet den bestehenden CLI-Hash-Namensraum 0 an einen echten PG-Owner.

    Die Konfiguration bestimmt den Scan, deshalb nimmt diese Datenquelle
    nicht an automatischen Connector-Laeufen teil. Keine Retrieval-Ausnahme:
    ihre Chunks brauchen dieselben FK-gesicherten Zuordnungen wie jeder Owner.
    """
    with db_transaction() as conn:
        conn.execute(
            "INSERT INTO datasources (id, name, type, is_active) "
            "VALUES (0, 'CLI (Konfiguration)', 'local', false) "
            "ON CONFLICT (id) DO NOTHING"
        )
        row = conn.execute("SELECT name, type, is_active FROM datasources WHERE id = 0").fetchone()
        if not row or (row["name"], row["type"], row["is_active"]) != (
            "CLI (Konfiguration)",
            "local",
            False,
        ):
            raise ValueError("CLI-Datenquelle 0 ist anderweitig belegt oder veraendert")
    return 0


@_refresh_search_on_exit
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
    activity_callback=None,
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
        resume_skip_files: Previous file logs. Only a matching completed hash
            permits skipping; a log alone cannot complete a file.
        status_callback: Optional callback(phase_text) to update job phase in Admin-UI.
            Called with descriptive strings like "Warte auf Backend..." or
            "Verarbeite Dateien...".
        activity_callback: Optional callback(event, **details) fuer direkte
            Datei-/Chunk-/Persistenz-Checkpoints des Worker-Watchdogs.

    Returns:
        Stats dict with counts.
    """
    if config is None:
        config = load_config()
        # Marker: Config stammt aus load_config() — persistiert, re-readbar
        config["_kiara_enforce_persisted_colbert_model"] = True

    # Persistent S4 maintenance blocks a new CLI run before any PG/Qdrant write.
    from kiara.retrieval.elasticsearch_maintenance import assert_access

    assert_access(config)

    # Zentrale Embedding-Modell-Aufloesung (AP4)
    raw_embedding_model = (
        embedding_model
        if embedding_model and embedding_model.strip()
        else get_embedding_model(config)
    )
    embedding_model = normalize_model_name(raw_embedding_model)

    from kiara.ingestion.html_migration import validate_run_selection

    selection = validate_run_selection(config, datasource_id, embedding_model)
    if selection is not None:
        ds_overrides = {**(ds_overrides or {}), "colbert_enabled": False}
    if datasource_id is None or datasource_id == 0:
        datasource_id = _ensure_cli_datasource()
        # Ohne explizite DS wurde bisher kein ColBERT-Aufbau angefordert.
        ds_overrides = {**(ds_overrides or {}), "colbert_enabled": False}

    ctx = _apply_ds_overrides(config, ds_overrides)
    _check_ddu_preflight(ctx, warmup_stats)

    # Backend-Setup
    os.makedirs(config["vectordb"]["path"], exist_ok=True)

    contract_recovery_stats: dict[str, int] = {}
    while True:
        try:
            store = VectorStore(config, embedding_model=embedding_model, intent="write")
            break
        except IndexContractUnavailableError as exc:
            if not control_callback:
                raise
            recovered = _wait_for_index_contract_target_recovery(
                config,
                exc,
                control_callback=control_callback,
                status_callback=status_callback,
                stats=contract_recovery_stats,
            )
            if recovered:
                continue
            cancelled_stats = _init_ingestion_stats([], [], [], {})
            cancelled_stats.update(contract_recovery_stats)
            return cancelled_stats
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

    from kiara.retrieval.elasticsearch_service import (
        model_uses_elasticsearch,
        refresh_index,
        targets_for_datasources,
    )

    if model_uses_elasticsearch(config, embedding_model):
        # Run durable pending cleanup first; its recorded deletions can explain
        # an empty index. Before any new publication, reject lost live corpus.
        from kiara.ingestion.document_publication import can_replay_unpublished_datasource
        from kiara.retrieval.elasticsearch_indices import DatasourceIndexTarget

        targets = targets_for_datasources(config, [datasource_id], embedding_model)
        if can_replay_unpublished_datasource(
            store, chunk_ds, datasource_id, embedding_model, files_to_process
        ):
            targets = tuple(
                DatasourceIndexTarget(t.datasource_id, t.index_name, 0) for t in targets
            )
        refresh_index(config, targets=targets)

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
    stats.update(contract_recovery_stats)
    if stats.get("skipped_files"):
        from kiara.retrieval.elasticsearch_service import (
            mark_search_data_adopted,
            model_uses_elasticsearch,
        )

        if model_uses_elasticsearch(config, embedding_model):
            mark_search_data_adopted(config, [datasource_id])
    if selection is not None:
        stats["html_migration_completed_paths"] = []

    skipped = stats["skipped_files"]
    total = stats["total_files"]

    if progress_callback:
        progress_callback(skipped, total)

    if not files_to_process:
        logger.info("No files to process.")
        # No-Files-Pfad gilt als run_completed=True
        if selection is None and not _run_colbert_model_switch_cleanup(
            config,
            datasource_id,
            chunk_ds,
            embedding_model,
            run_completed=True,
            stats=stats,
        ):
            return stats
        if selection is None:
            hash_store.set_hash_mode(datasource_id, embedding_model, hash_mode)
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

    from contextlib import contextmanager

    from kiara.ingestion._file_utils import _LoadedFileVersion

    loaded_versions = {}

    @contextmanager
    def loading_file(path):
        loaded_versions.pop(path, None)
        version = _LoadedFileVersion.capture(path, hash_mode, config)
        yield
        version.validate()
        loaded_versions[path] = version

    def complete_file(path):
        version = loaded_versions[path]
        version.validate()
        completion_options = {"sync_model_sizes": selection is None}
        if _colbert_enabled_for_run(ictx):
            from kiara.retrieval.colbert import colbert_ds_model_key
            from kiara.shared.constants import COLBERT_DEFAULT_MODEL

            current_model = config["retrieval"]["colbert"].get("model", COLBERT_DEFAULT_MODEL)
            completion_options["additional_models"] = (colbert_ds_model_key(current_model),)
        _complete_file(
            hash_store,
            datasource_id,
            embedding_model,
            path,
            version.completion,
            **completion_options,
        )
        if selection is not None:
            stats["html_migration_completed_paths"].append(path)

    def prepare_file(path, metadatas, ids):
        return _prepare_file_chunks(
            config,
            store,
            hash_store,
            chunk_ds,
            datasource_id,
            embedding_model,
            path,
            metadatas,
            ids,
        )

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
        file_completion_callback=complete_file,
        file_preparation_callback=prepare_file,
        file_loading_context=loading_file,
        skipped=skipped,
        total=total,
        progress_callback=progress_callback,
        control_callback=control_callback,
        file_log_callback=file_log_callback,
        file_start_callback=file_start_callback,
        status_callback=status_callback,
        activity_callback=activity_callback,
    )

    # run_completed-Tracking: initial True, auf False bei Abort/Cancel
    run_completed = True
    for file_path in tqdm(files_to_process, desc="Loading & chunking"):
        if not _process_file_in_pipeline(ictx, file_path):
            run_completed = False
            break

    _persist_hashes_checkpoint(ictx, force=True)
    if selection is None:
        hash_store.set_hash_mode(datasource_id, embedding_model, hash_mode)

    # ColBERT Model-Switch Cleanup am Ende des Laufs
    if selection is None and not _run_colbert_model_switch_cleanup(
        config,
        datasource_id,
        chunk_ds,
        embedding_model,
        run_completed=run_completed,
        stats=stats,
    ):
        return stats

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
    from kiara.retrieval.index_compatibility import IndexCompatibilityError

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
        "contract_discovery_waits": 0,
        "contract_discovery_recoveries": 0,
        "contract_discovery_wait_seconds": 0,
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
                "contract_discovery_waits",
                "contract_discovery_recoveries",
                "contract_discovery_wait_seconds",
            ):
                total_stats[key] += stats.get(key, 0)
            if stats.get("colbert_collection"):
                total_stats["colbert_collection"] = stats["colbert_collection"]
            total_stats["datasources_processed"] += 1
        except IndexCompatibilityError:
            raise
        except Exception as exc:
            logger.error("Datasource %s failed: %s", ds_id, exc, exc_info=True)
            total_stats["errors"] += 1

    return total_stats
