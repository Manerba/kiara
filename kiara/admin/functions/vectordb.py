"""Admin-Funktionen fuer VektorDB-Dateiverwaltung."""

import asyncio
import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="vectordb.files",
    category="vectordb",
    description="Dateiliste einer Datenquelle mit Chunk-Counts",
    help="Listet alle indexierten Dateien einer Datenquelle auf, "
    "inklusive Anzahl der Chunks pro Datei.",
    examples=[
        "Welche Dateien hat Datenquelle 3 indexiert?",
        "Zeig mir die Chunks von DS 5",
        "Wie viele Chunks hat jede Datei in DS 1?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {
                "type": "integer",
                "description": "Datenquellen-ID",
            },
            "model": {
                "type": "string",
                "description": "Embedding-Modell-Name (z.B. nomic-embed-text)",
            },
        },
        "required": ["ds_id", "model"],
    },
    returns="Dateiliste mit Chunk-Counts und Gesamtsummen",
    raw_response=True,
    http=("GET", "/api/admin/vectordb/datasources/{ds_id}/files"),
)
async def vectordb_files(user: dict, ds_id: int, model: str) -> AdminResult:
    from kiara.backend_selector import normalize_model_name
    from kiara.repositories.datasource import DatasourceRepository

    model = normalize_model_name(model)
    if not model:
        return AdminResult.fail("Kein gueltiges Embedding-Modell angegeben")

    ds_repo = DatasourceRepository()
    ds = ds_repo.get_by_id(ds_id)
    if not ds:
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")

    from kiara.config import load_config
    from kiara.retrieval.vectorstore import VectorStore

    def _list_files_sync() -> list[dict]:
        vs = VectorStore(load_config(), embedding_model=model)
        return vs.list_sources_for_datasource(ds_id)

    try:
        files = await asyncio.to_thread(_list_files_sync)
    except Exception as exc:
        return AdminResult.fail(f"VektorDB-Abfrage fehlgeschlagen: {exc}")

    total_chunks = sum(f.get("chunk_count", 0) for f in files)
    return AdminResult.ok(
        f"{len(files)} Datei(en), {total_chunks} Chunks",
        data={"files": files, "total_chunks": total_chunks, "ds_id": ds_id},
    )


@admin_function(
    name="vectordb.delete_files",
    category="vectordb",
    description="Dateien aus VektorDB löschen",
    help="Löscht ausgewählte Dateien und ihre Chunks aus der VektorDB. "
    "Die Dateien werden beim nächsten Indexlauf neu verarbeitet.",
    examples=[
        "Lösche Datei X aus Datenquelle 3",
        "Entferne die fehlerhaften Dateien aus der VektorDB",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {
                "type": "integer",
                "description": "Datenquellen-ID",
            },
            "embedding_model": {
                "type": "string",
                "description": "Embedding-Modell-Name",
            },
            "file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Liste der zu löschenden Dateipfade",
            },
        },
        "required": ["ds_id", "embedding_model", "file_paths"],
    },
    returns="Ergebnis der Löschung",
    requires_confirmation=True,
    confirm_template="dateien aus datenquelle {ds_id} löschen",
    pk_param="ds_id",
    changelog_area="vectordb",
    changelog_action="delete",
    raw_response=True,
    http=("POST", "/api/admin/vectordb/datasources/{ds_id}/delete-files"),
)
async def vectordb_delete_files(
    user: dict, ds_id: int, embedding_model: str, file_paths: list
) -> AdminResult:
    if not file_paths:
        return AdminResult.fail("Keine Dateien zum Löschen angegeben")

    from kiara.backend_selector import normalize_model_name

    embedding_model = normalize_model_name(embedding_model)
    if not embedding_model:
        return AdminResult.fail("Kein gueltiges Embedding-Modell angegeben")

    from kiara.repositories.datasource import DatasourceRepository

    ds_repo = DatasourceRepository()
    ds = ds_repo.get_by_id(ds_id)
    if not ds:
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")

    # Aktive Jobs pruefen — Race-Condition mit parallelem Worker vermeiden
    from kiara.shared.db import db_query

    with db_query() as db:
        active_job = db.execute(
            "SELECT id, status FROM ingestion_jobs "
            "WHERE status IN ('pending', 'running') "
            "AND datasource_id = %s AND embedding_model = %s "
            "ORDER BY queued_at DESC LIMIT 1",
            (ds_id, embedding_model),
        ).fetchone()
    if active_job:
        return AdminResult.fail(
            f"Ingestion-Job #{active_job['id']} ist {active_job['status']} "
            f"fuer Datenquelle {ds_id}. Bitte zuerst abbrechen oder warten."
        )

    from kiara.config import load_config
    from kiara.ingestion._file_utils import ChunkDatasources, HashStore
    from kiara.ingestion.vectordb_ops import delete_files

    config = load_config()
    chunk_ds = ChunkDatasources()
    hash_store = HashStore()
    try:
        result = await asyncio.to_thread(
            delete_files,
            config,
            datasource_id=ds_id,
            embedding_model=embedding_model,
            file_paths=file_paths,
            chunk_ds=chunk_ds,
            hash_store=hash_store,
        )
    except Exception as exc:
        return AdminResult.fail(f"Löschen fehlgeschlagen: {exc}")

    return AdminResult.ok(
        f"{len(file_paths)} Datei(en) gelöscht",
        data=result,
    )


@admin_function(
    name="vectordb.delete_collection",
    category="vectordb",
    description="Gesamte Embedding-Collection löschen",
    help="Löscht eine komplette Dense-Qdrant-Collection für ein Embedding-Modell "
    "und bereinigt die zugehörigen Dense-Schichten (Qdrant, BM25-Cache, "
    "chunk_datasources, file_hashes, hash_mode_meta, datasource_embeddings). "
    "ColBERT hat eigene Modell-Keys und wird hier nicht gelöscht.",
    examples=[
        "Lösche die Collection für nomic-embed-text",
        "Entferne alle Embeddings des Modells mxbai-embed-large",
        "Collection des alten Modells komplett löschen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "embedding_model": {
                "type": "string",
                "description": "Name des Embedding-Modells dessen Collection gelöscht werden soll",
            },
        },
        "required": ["embedding_model"],
    },
    returns="Ergebnis der Löschung mit Statistiken (gelöschte Chunks, Dateien, Datenquellen)",
    requires_confirmation=True,
    confirm_template="collection {embedding_model} löschen",
    pk_param="embedding_model",
    changelog_area="vectordb",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/vectordb/collections/{embedding_model:path}"),
)
async def vectordb_delete_collection(user: dict, embedding_model: str) -> AdminResult:
    from kiara.backend_selector import normalize_model_name
    from kiara.config import load_config
    from kiara.retrieval.embedding import collection_name_for_model
    from kiara.shared.constants import DEFAULT_QDRANT_URL
    from kiara.shared.db import db_query, db_transaction

    # 1. Normalisieren + Validieren
    model = normalize_model_name(embedding_model)
    if not model:
        return AdminResult.fail("Kein gültiges Embedding-Modell angegeben")
    if model.startswith("colbert::"):
        return AdminResult.fail(
            "ColBERT-Modell-Keys können nicht über den Dense-Collection-Delete gelöscht werden",
            http_status=400,
        )

    config = load_config()
    collection_name = collection_name_for_model(model)

    # 2. Qdrant-Client + Existenzpruefung
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    qdrant_cfg = config.get("vectordb", {})
    client = await asyncio.to_thread(
        QdrantClient,
        url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
        api_key=qdrant_cfg.get("qdrant_api_key") or None,
        timeout=qdrant_cfg.get("timeout", 60),
    )

    try:
        collections_resp = await asyncio.to_thread(client.get_collections)
        existing_collections = {c.name for c in collections_resp.collections}
    except Exception as exc:
        return AdminResult.fail(
            f"Qdrant-Collections konnten nicht geladen werden: {exc}",
            http_status=503,
        )

    collection_exists = collection_name in existing_collections
    chunk_count = 0
    if collection_exists:
        try:
            info = await asyncio.to_thread(client.get_collection, collection_name)
            chunk_count = info.points_count or 0
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                logger.info(
                    "Qdrant-Collection %s war beim Lesen bereits weg",
                    collection_name,
                )
                collection_exists = False
                chunk_count = 0
            else:
                return AdminResult.fail(
                    f"Qdrant-Collection '{collection_name}' konnte nicht gelesen werden: {exc}",
                    http_status=503,
                )
        except Exception as exc:
            return AdminResult.fail(
                f"Qdrant-Collection '{collection_name}' konnte nicht gelesen werden: {exc}",
                http_status=503,
            )

    # 3. DB-Reste pruefen (Crash-Recovery, wenn Qdrant bereits geloescht ist)
    with db_query() as db:
        db_remains = db.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM datasource_embeddings WHERE embedding_model = %s) + "
            "(SELECT COUNT(*) FROM chunk_datasources WHERE embedding_model = %s) + "
            "(SELECT COUNT(*) FROM file_hashes WHERE embedding_model = %s) + "
            "(SELECT COUNT(*) FROM hash_mode_meta WHERE embedding_model = %s) "
            "AS cnt",
            (model, model, model, model),
        ).fetchone()["cnt"]

    if not collection_exists and db_remains == 0:
        return AdminResult.fail(
            f"Collection für Modell '{model}' nicht gefunden",
            http_status=404,
        )

    # 4. Aktive Jobs pruefen
    with db_query() as db:
        active_job = db.execute(
            "SELECT id, status FROM ingestion_jobs "
            "WHERE status IN ('pending', 'running') AND embedding_model = %s "
            "ORDER BY queued_at DESC LIMIT 1",
            (model,),
        ).fetchone()
    if active_job:
        return AdminResult.fail(
            f"Ingestion-Job #{active_job['id']} ist {active_job['status']} "
            f"für Modell '{model}'. Bitte zuerst abbrechen oder warten."
        )

    # Betroffene Datenquellen VOR dem Delete zaehlen
    with db_query() as db:
        affected_ds = db.execute(
            "SELECT COUNT(DISTINCT datasource_id) AS cnt FROM ("
            "  SELECT datasource_id FROM datasource_embeddings WHERE embedding_model = %s "
            "  UNION "
            "  SELECT datasource_id FROM chunk_datasources WHERE embedding_model = %s"
            ") sub",
            (model, model),
        ).fetchone()["cnt"]

    # --- Ab hier destruktiv ---
    # Reihenfolge: Qdrant-Collection-Drop → BM25 → DB. Dieser Pfad loescht
    # die gesamte Dense-Collection und ist damit NICHT mit der Dense-last
    # Reihenfolge von vectordb_ops.delete_files() vergleichbar — dort bleibt
    # Dense-Qdrant bis zum Schluss als ID-Retry-Anker erhalten. Hier gibt es
    # keinen Retry-Anker, weil der Caller explizit die ganze Collection
    # ausser Betrieb nimmt.

    # 5. Qdrant: Dense-Collection loeschen, wenn vorhanden
    dense_deleted = False
    if collection_exists:
        try:
            delete_ok = bool(
                await asyncio.to_thread(client.delete_collection, collection_name=collection_name)
            )
            if not delete_ok:
                return AdminResult.fail(
                    f"Qdrant-Collection '{collection_name}' wurde nicht gelöscht",
                    http_status=502,
                )
            dense_deleted = True
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                logger.info(
                    "Qdrant-Collection %s war beim Delete bereits weg",
                    collection_name,
                )
            else:
                return AdminResult.fail(
                    f"Qdrant-Collection '{collection_name}' konnte nicht gelöscht werden: {exc}",
                    http_status=502,
                )
        except Exception as exc:
            return AdminResult.fail(
                f"Qdrant-Collection '{collection_name}' konnte nicht gelöscht werden: {exc}",
                http_status=502,
            )

    # 6. BM25-Cache invalidieren
    from kiara.ingestion.vectordb_ops import invalidate_bm25

    await asyncio.to_thread(invalidate_bm25, collection_name)

    # 7. DB: Alle Dense-Schichten bereinigen (eine Transaktion)
    with db_transaction() as db:
        from kiara.ingestion._file_utils import ChunkDatasources

        chunk_ds = ChunkDatasources()
        chunk_ds.acquire_graph_cleanup_lock(db)
        candidate_rows = db.execute(
            "SELECT DISTINCT chunk_id FROM chunk_datasources WHERE embedding_model = %s",
            (model,),
        ).fetchall()
        graph_candidates = [r["chunk_id"] for r in candidate_rows]

        cd_count = db.execute(
            "DELETE FROM chunk_datasources WHERE embedding_model = %s",
            (model,),
        ).rowcount

        fh_count = db.execute(
            "DELETE FROM file_hashes WHERE embedding_model = %s",
            (model,),
        ).rowcount

        hm_count = db.execute(
            "DELETE FROM hash_mode_meta WHERE embedding_model = %s",
            (model,),
        ).rowcount

        de_count = db.execute(
            "DELETE FROM datasource_embeddings WHERE embedding_model = %s",
            (model,),
        ).rowcount

        if graph_candidates:
            from kiara.graph.store import GraphStore

            graph_chunk_ids = chunk_ds.filter_unreferenced_chunk_ids(
                graph_candidates,
                conn=db,
            )
            if graph_chunk_ids:
                GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=db)

    # 8. UI-/Dashboard-Caches aktualisieren (non-fatal)
    try:
        from kiara.web.cache_refresh import refresh_datasource_cache, refresh_vectordb_cache

        refresh_datasource_cache()
        refresh_vectordb_cache()
    except Exception as exc:
        logger.warning("Cache-Refresh nach Collection-Delete fehlgeschlagen: %s", exc)

    message = (
        f"Collection {collection_name} gelöscht"
        if dense_deleted
        else f"DB-Reste für Collection {collection_name} bereinigt"
    )

    return AdminResult.ok(
        message,
        data={
            "name": collection_name,
            "message": message,
            "embedding_model": model,
            "collection_name": collection_name,
            "deleted_chunks_qdrant": chunk_count if dense_deleted else 0,
            "deleted_chunk_datasources": cd_count,
            "deleted_file_hashes": fh_count,
            "deleted_hash_mode_meta": hm_count,
            "deleted_datasource_embeddings": de_count,
            "affected_datasources": affected_ds,
            "dense_deleted": dense_deleted,
            "colbert_untouched": True,
        },
    )
