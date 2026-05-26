"""Generische VectorDB-Operationen (Shared-Bausteine).

Zentrale Loesch-Funktion fuer Dateien aus allen Datenschichten
mit Orphan-Pruefung und Preview-First-Strategie fuer Crash-Sicherheit.
Unterstuetzt Dense (``kb_*``) und ColBERT (``colbert_*``) parallel.
"""

import logging

from kiara.ingestion._file_utils import ChunkDatasources, HashStore
from kiara.retrieval.vectorstore import VectorStore
from kiara.shared.db import db_transaction

logger = logging.getLogger(__name__)


def delete_from_elasticsearch(
    config: dict, file_paths: list[str], datasource_id: int | None = None
) -> None:
    """ES-Loeschung mit Konfigurationsguard.

    Bei konfiguriertem Elasticsearch sind Loeschfehler fatal: sonst koennen
    stale Suchdokumente uebrig bleiben, waehrend DB/Qdrant bereits Erfolg
    melden.
    """
    from kiara.retrieval.elasticsearch_service import delete_by_sources, is_configured

    if not is_configured(config):
        return

    try:
        delete_by_sources(config, file_paths, datasource_id=datasource_id)
    except Exception as exc:
        logger.warning(
            "ES-Loeschung fehlgeschlagen fuer DS %s (%d Pfade): %s",
            datasource_id,
            len(file_paths),
            exc,
            exc_info=True,
        )
        raise


def invalidate_bm25(coll_name: str) -> None:
    """BM25-Cache fuer eine Collection invalidieren."""
    try:
        from kiara.retrieval.retriever import HybridRetriever

        HybridRetriever.invalidate_bm25_cache(coll_name)
    except Exception as exc:
        logger.warning(
            "BM25-Cache-Invalidierung fuer %s fehlgeschlagen: %s",
            coll_name,
            exc,
            exc_info=True,
        )


def _resolve_colbert_models_for_ds(
    config: dict,
    datasource_id: int,
    chunk_ds: ChunkDatasources,
) -> list[str]:
    """Ermittelt ALLE fuer eine DS relevanten ColBERT-Modelle.

    Vereinigt:
    - Modelle aus existierenden ``colbert_*``-Collections in Qdrant
    - Das aktuell konfigurierte ``retrieval.colbert.model``
    - Historische ``colbert::``-Keys aus ``chunk_datasources``

    Filtert auf die fuer diese DS tatsaechlich vorhandenen Modelle.
    """
    from kiara.backend_selector import normalize_model_name
    from kiara.retrieval.colbert import (
        ColBERTStore,
        colbert_ds_model_key,
        model_name_from_colbert_ds_key,
    )
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL

    candidate_ds_keys: set[str] = set()

    # 1. Aus Qdrant-Collections
    for coll in ColBERTStore.list_colbert_collections(config):
        candidate_ds_keys.add(colbert_ds_model_key(coll["model"]))

    # 2. Aktuell konfiguriertes Modell
    configured = normalize_model_name(
        config.get("retrieval", {}).get("colbert", {}).get("model", COLBERT_DEFAULT_MODEL)
    )
    if configured:
        candidate_ds_keys.add(colbert_ds_model_key(configured))

    # 3. Historische DB-Keys
    candidate_ds_keys.update(k for k in chunk_ds.distinct_models() if k.startswith("colbert::"))

    if not candidate_ds_keys:
        return []

    # Auf fuer diese DS vorhandene Modelle einschraenken
    counts = chunk_ds.counts_by_datasource_all_models(datasource_id, sorted(candidate_ds_keys))
    return [model_name_from_colbert_ds_key(ds_key) for ds_key, count in counts.items() if count > 0]


def _recount_datasource_embeddings(
    conn,
    datasource_id: int,
    embedding_model: str,
) -> None:
    """Setzt ``datasource_embeddings`` aus der Source-of-Truth neu.

    ``chunk_count`` aus ``chunk_datasources`` + ``file_count`` aus
    ``file_hashes`` lesen und in einem UPDATE setzen. Laeuft auf demselben
    ``conn`` wie die vorherigen DELETEs, sieht also den post-delete State.

    Wenn beide Werte null sind, wird die Zeile geloescht.
    """
    chunk_row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM chunk_datasources "
        "WHERE datasource_id = %s AND embedding_model = %s",
        (datasource_id, embedding_model),
    ).fetchone()
    chunk_count = (chunk_row["cnt"] if chunk_row else 0) or 0

    file_row = conn.execute(
        "SELECT COUNT(DISTINCT file_path) AS cnt FROM file_hashes "
        "WHERE datasource_id = %s AND embedding_model = %s",
        (datasource_id, embedding_model),
    ).fetchone()
    file_count = (file_row["cnt"] if file_row else 0) or 0

    if chunk_count == 0 and file_count == 0:
        conn.execute(
            "DELETE FROM datasource_embeddings WHERE datasource_id = %s AND embedding_model = %s",
            (datasource_id, embedding_model),
        )
    else:
        conn.execute(
            "INSERT INTO datasource_embeddings "
            "(datasource_id, embedding_model, chunk_count, file_count) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (datasource_id, embedding_model) DO UPDATE "
            "SET chunk_count = EXCLUDED.chunk_count, "
            "file_count = EXCLUDED.file_count",
            (datasource_id, embedding_model, chunk_count, file_count),
        )


def delete_files(
    config: dict,
    datasource_id: int,
    embedding_model: str,
    file_paths: list[str],
    *,
    chunk_ds: ChunkDatasources,
    hash_store: HashStore,
) -> dict:
    """Loescht Dateien aus allen Datenschichten (Dense + ColBERT).

    Reihenfolge ist crash-/retry-sicher (Dense-last Retry-Anker):
    1. Read-only chunk_ids aus Dense-Qdrant (einziger dateipfadbasierter
       Retry-Anker).
    2. Preview Dense + ColBERT (read-only).
    3. ColBERT-Qdrant-Cleanup (vor DB/Dense).
    4. DB-/Hash-Transaktion: chunk_datasources (Dense + ColBERT),
       file_hashes und datasource_embeddings-Recount aus SoT in EINEM
       Commit.
    5. Elasticsearch (dateipfadbasiert).
    6. Dense-Qdrant-Mutation (zuletzt; bei Retry werden nicht mehr
       vorhandene Punkte idempotent ueberspungen).
    7. BM25 NACH finaler Dense-Mutation invalidieren.

    Bei ``not chunk_ids`` werden ID-basierte Dense-/ColBERT-Qdrant-Schritte
    uebersprungen, aber ES-Loeschung und BM25-Invalidation bleiben
    verpflichtend (dateipfadbasierte Schichten). Counter werden dort
    ebenfalls aus der SoT neu gesetzt.

    Returns:
        {"deleted_files", "deleted_chunks", "orphan_chunks_deleted",
         "shared_chunks_updated"}
    """
    from kiara.retrieval.colbert import ColBERTStore, colbert_ds_model_key

    # 1. Guard
    if not file_paths:
        raise ValueError("file_paths darf nicht leer sein")

    # 2. VectorStore oeffnen
    store = VectorStore(config, embedding_model=embedding_model)

    # 3. Alle chunk_ids ermitteln (DS-gefiltert). Dense-Qdrant ist der
    # einzige dateipfadbasierte Retry-Anker — read-only Schritt hier.
    chunk_ids = store.get_ids_by_sources(file_paths, ds_id=datasource_id)
    colbert_models = _resolve_colbert_models_for_ds(config, datasource_id, chunk_ds)

    # 4. not chunk_ids: ID-basierte Qdrant-Schritte entfallen, aber
    # dateipfadbasierte Schichten (ES, BM25) plus file_hashes + Counter
    # aus SoT muessen trotzdem laufen.
    if not chunk_ids:
        with db_transaction() as conn:
            hash_store.remove_files(datasource_id, embedding_model, file_paths, conn=conn)
            _recount_datasource_embeddings(conn, datasource_id, embedding_model)
            for colbert_model in colbert_models:
                _recount_datasource_embeddings(
                    conn, datasource_id, colbert_ds_model_key(colbert_model)
                )
        delete_from_elasticsearch(config, file_paths, datasource_id=datasource_id)
        invalidate_bm25(store.get_collection_name())
        return {
            "deleted_files": len(file_paths),
            "deleted_chunks": 0,
            "orphan_chunks_deleted": 0,
            "shared_chunks_updated": 0,
        }

    # 5. Preview-Phase: Read-only Orphan-/Non-Orphan-Analyse (Dense + ColBERT)
    dense_orphan_ids, dense_non_orphan_map = chunk_ds.preview_chunk_removal_for_datasource(
        datasource_id, embedding_model, chunk_ids
    )
    colbert_cleanup: dict[str, tuple[list[str], dict[str, list[int]]]] = {}
    for colbert_model in colbert_models:
        colbert_cleanup[colbert_model] = chunk_ds.preview_chunk_removal_for_datasource(
            datasource_id, colbert_ds_model_key(colbert_model), chunk_ids
        )

    # 6. ColBERT-Qdrant-Cleanup zuerst — Dense bleibt als Retry-Anker.
    for colbert_model, (colbert_orphan_ids, colbert_non_orphan_map) in colbert_cleanup.items():
        cstore = ColBERTStore(config, colbert_model)
        if colbert_orphan_ids:
            cstore.delete_by_ids(colbert_orphan_ids)
        if colbert_non_orphan_map:
            cstore.update_ds_ids(colbert_non_orphan_map, wait=True)

    # 7. DB-/Hash-Mutation in EINER Transaktion — chunk_datasources (Dense+
    # ColBERT), Graph-Cleanup, file_hashes und datasource_embeddings-Recount atomar.
    with db_transaction() as conn:
        chunk_ds.acquire_graph_cleanup_lock(conn)
        chunk_ds.remove_chunks_for_datasource(datasource_id, embedding_model, chunk_ids, conn=conn)
        hash_store.remove_files(datasource_id, embedding_model, file_paths, conn=conn)
        for colbert_model in colbert_models:
            chunk_ds.remove_chunks_for_datasource(
                datasource_id,
                colbert_ds_model_key(colbert_model),
                chunk_ids,
                conn=conn,
            )
            _recount_datasource_embeddings(conn, datasource_id, colbert_ds_model_key(colbert_model))
        graph_chunk_ids = chunk_ds.filter_unreferenced_chunk_ids(
            chunk_ids,
            conn=conn,
        )
        if graph_chunk_ids:
            from kiara.graph.store import GraphStore

            GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)
        _recount_datasource_embeddings(conn, datasource_id, embedding_model)

    # 8. Elasticsearch (DS-spezifisch)
    delete_from_elasticsearch(config, file_paths, datasource_id=datasource_id)

    # 9. Dense-Qdrant-Mutation zuletzt — idempotent bei Retry.
    if dense_orphan_ids:
        store.delete_by_ids(dense_orphan_ids)
    if dense_non_orphan_map:
        store.update_ds_ids(dense_non_orphan_map)

    # 10. BM25 NACH finaler Dense-Mutation invalidieren.
    invalidate_bm25(store.get_collection_name())

    return {
        "deleted_files": len(file_paths),
        "deleted_chunks": len(chunk_ids),
        "orphan_chunks_deleted": len(dense_orphan_ids),
        "shared_chunks_updated": len(dense_non_orphan_map),
    }
