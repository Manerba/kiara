"""Service-Funktionen fuer Datenquellen-CRUD.

Extrahiert aus api_datasource.py — sowohl Admin-Funktionen (Chat)
als auch REST-Endpunkte (Auto-Router/Legacy) nutzen diese Logik.
"""

import json
import logging

from kiara.config import load_config
from kiara.repositories.datasource import DatasourceRepository
from kiara.shared.constants import INGESTION_DATASOURCE_LOCK_NAMESPACE, PASSWORD_MASK
from kiara.shared.crypto import encrypt, encrypt_secret_fields, is_encrypted

logger = logging.getLogger(__name__)

_ACTIVE_INGESTION_JOB_STATUSES = ("pending", "running", "paused")


async def create_datasource(
    name: str,
    ds_type: str,
    connector_config: dict,
    ds_chunk_strategy: str = "auto",
    ds_chunk_size: int | None = None,
    ds_chunk_overlap: int | None = None,
    ds_file_types: list[str] | None = None,
    ds_exclude_patterns: list[str] | None = None,
    ds_ddu_enabled: bool = False,
    ds_ddu_ocr_images: bool = True,
    ds_vlm_enabled: bool = False,
    ds_accept_all_text: bool = False,
    ds_index_attachments: bool = True,
    sync_interval: str = "global",
    restriction_level: str = "anonymize",
    ds_contextual_retrieval: bool = False,
    ds_enrichment_prompt_id: int | None = None,
    ds_enrichment_backend_id: int | None = None,
    ds_colbert_enabled: bool = False,
) -> dict:
    """Erstellt eine neue Datenquelle mit verschluesselten Credentials.

    Returns: {"id": int, "message": str}
    Raises: ValueError bei unbekanntem Konnektor-Typ.
    """
    # Enrichment-ID-Validierung (analog zu datasource.update)
    for _name, _val in [
        ("ds_enrichment_prompt_id", ds_enrichment_prompt_id),
        ("ds_enrichment_backend_id", ds_enrichment_backend_id),
    ]:
        if _val is not None and (not isinstance(_val, int) or _val <= 0):
            raise ValueError(f"'{_name}' muss eine positive Ganzzahl oder null sein")

    if ds_contextual_retrieval and ds_enrichment_backend_id is None:
        raise ValueError(
            "Contextual Retrieval erfordert ein explizites Enrichment-Backend. "
            "Bitte ein LLM- oder Cloud-Backend waehlen."
        )

    # Chunk-Parameter-Validierung
    if ds_chunk_size is not None and ds_chunk_size <= 0:
        raise ValueError("ds_chunk_size muss groesser als 0 sein")
    if ds_chunk_overlap is not None and ds_chunk_overlap < 0:
        raise ValueError("ds_chunk_overlap darf nicht negativ sein")
    if (
        ds_chunk_size is not None
        and ds_chunk_overlap is not None
        and ds_chunk_overlap >= ds_chunk_size
    ):
        raise ValueError("ds_chunk_overlap muss kleiner als ds_chunk_size sein")

    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector

    ensure_loaded()
    try:
        connector_cls = get_connector(ds_type)
    except KeyError:
        # Registry wirft KeyError bei unbekanntem Typ (Auto-Fix-Haertung).
        raise ValueError(f"Unbekannter Konnektor-Typ: {ds_type}")
    if not connector_cls:
        raise ValueError(f"Unbekannter Konnektor-Typ: {ds_type}")

    encrypted_config = encrypt_secret_fields(connector_config, connector_cls.secret_fields())

    config = load_config()
    ds_repo = DatasourceRepository()
    ing_cfg = config.get("ingestion", {})
    name = name.strip() if name else name
    if not name:
        raise ValueError("Datenquellen-Name darf nicht leer sein")
    ds_id = ds_repo.create(
        name=name,
        type=ds_type,
        connector_config=json.dumps(encrypted_config),
        is_active=True,
        ds_chunk_strategy=ds_chunk_strategy,
        ds_chunk_size=(
            ds_chunk_size if ds_chunk_size is not None else ing_cfg.get("chunk_size", 1000)
        ),
        ds_chunk_overlap=(
            ds_chunk_overlap if ds_chunk_overlap is not None else ing_cfg.get("chunk_overlap", 200)
        ),
        ds_file_types=json.dumps(ds_file_types) if ds_file_types else "",
        ds_exclude_patterns=json.dumps(ds_exclude_patterns) if ds_exclude_patterns else "",
        ds_ddu_enabled=bool(ds_ddu_enabled),
        ds_ddu_ocr_images=int(ds_ddu_ocr_images),
        ds_vlm_enabled=bool(ds_vlm_enabled),
        ds_accept_all_text=int(ds_accept_all_text),
        ds_index_attachments=int(ds_index_attachments),
        sync_interval=sync_interval,
        restriction_level=restriction_level,
        ds_contextual_retrieval=int(ds_contextual_retrieval),
        ds_enrichment_prompt_id=ds_enrichment_prompt_id,
        ds_enrichment_backend_id=ds_enrichment_backend_id,
        ds_colbert_enabled=bool(ds_colbert_enabled),
    )

    from kiara.web.api_utils import safe_refresh
    from kiara.web.cache_refresh import refresh_datasource_cache

    safe_refresh(refresh_datasource_cache)

    return {"id": ds_id, "message": "Datenquelle erstellt"}


async def delete_datasource_full(ds_id: int) -> dict:
    """Loescht Datenquelle mit ALLEN Abhaengigkeiten.

    Reihenfolge (crash-sicher — externe Systeme VOR DB):
    1.  Qdrant-Vektoren loeschen (pro Collection)
    1b. Graph-Nodes/-Edges loeschen (Orphan-Chunks)
    2.  Elasticsearch-Eintraege loeschen
    3. File-Hash-Store bereinigen (PostgreSQL)
    4. DB-Cascade (11 Tabellen, in Transaktion)
    5. Mount-Points entfernen (SMB/WebDAV)
    6. Cron-Schedule entfernen
    7. Cache aktualisieren

    Returns: {"message": "Datenquelle geloescht"} (kompatibel mit bestehendem Endpunkt)
    Raises: ValueError wenn DS nicht existiert.
    """
    config = load_config()
    ds_repo = DatasourceRepository()
    ds = ds_repo.get_by_id(ds_id)
    if not ds:
        raise ValueError(f"Datenquelle {ds_id} nicht gefunden")

    old_file_paths: list[str] = []

    # Preflight + Soft-Delete in einer Transaktion. Der Advisory-Lock ist derselbe
    # wie in enqueue_job(); damit koennen Delete und neue Jobs nicht zwischen
    # Preflight und externem Cleanup aneinander vorbeilaufen. Der file_hashes-
    # Snapshot fuer ES wird unter demselben Lock gelesen.
    from kiara.shared.db import db_transaction

    try:
        with db_transaction() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(%s, %s)",
                (INGESTION_DATASOURCE_LOCK_NAMESPACE, ds_id),
            )
            active_job = conn.execute(
                "SELECT id, status FROM ingestion_jobs "
                "WHERE datasource_id = %s AND status IN (%s, %s, %s) "
                "ORDER BY id ASC LIMIT 1 FOR UPDATE",
                (ds_id, *_ACTIVE_INGESTION_JOB_STATUSES),
            ).fetchone()
            if active_job:
                raise ValueError(
                    f"Datenquelle hat einen aktiven Ingestion-Job "
                    f"(job_id={active_job['id']}, status={active_job['status']}). "
                    "Bitte den Job erst abbrechen oder abschliessen lassen."
                )
            rows = conn.execute(
                "SELECT DISTINCT file_path FROM file_hashes WHERE datasource_id = %s",
                (ds_id,),
            ).fetchall()
            old_file_paths = [row["file_path"] for row in rows]
            conn.execute(
                "UPDATE datasources SET is_active = false, name = %s WHERE id = %s",
                (f"[DELETING] {ds.get('name', '')}", ds_id),
            )
    except ValueError:
        raise
    except Exception:
        logger.exception("Soft-Delete fuer DS %d fehlgeschlagen — Abbruch", ds_id)
        raise ValueError(f"Datenquelle {ds_id} konnte nicht als geloescht markiert werden")

    # 1. Qdrant-Cleanup (pro Collection, non-fatal)
    #    Drei-Phasen-Ansatz: Preview → Qdrant → DB-Delete
    #    Bei Crash nach Qdrant aber vor DB-Delete: chunk_datasources existiert
    #    noch, Retry leitet Orphan-IDs erneut korrekt ab.
    try:
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.ingestion.vectordb_ops import invalidate_bm25
        from kiara.retrieval.vectorstore import VectorStore

        chunk_ds = ChunkDatasources()
        for coll in VectorStore.list_collections(config):
            store = VectorStore(config, coll["model"])
            # Phase 1: Read-only Vorschau (DB unveraendert)
            orphan_ids, non_orphan_ds_ids = chunk_ds.preview_datasource_removal(
                ds_id, coll["model"]
            )
            # Phase 2: Qdrant-Cleanup (non-fatal, idempotent)
            # DB-Cascade und Graph-Cleanup passieren spaeter atomar.
            try:
                if orphan_ids:
                    store.delete_by_ids(orphan_ids)
                if non_orphan_ds_ids:
                    store.update_ds_ids(non_orphan_ds_ids, wait=True)
                invalidate_bm25(store.get_collection_name())
            except Exception:
                logger.exception(
                    "Qdrant-Chunk-Update fuer DS %d, Collection %s fehlgeschlagen — "
                    "DS wird per Cascade geloescht, Dense-Reste muessen ggf. "
                    "ueber Integrity-Reports bereinigt werden",
                    ds_id,
                    coll["model"],
                )
    except Exception:
        logger.exception("Qdrant-Cleanup fuer DS %d fehlgeschlagen (fortfahrend)", ds_id)

    # 1a. ColBERT-Cleanup (separate colbert_* Collections, non-fatal)
    try:
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.retrieval.colbert import ColBERTStore, colbert_ds_model_key

        chunk_ds = ChunkDatasources()
        for colbert_coll in ColBERTStore.list_colbert_collections(config):
            try:
                ds_key = colbert_ds_model_key(colbert_coll["model"])
                if chunk_ds.count_by_datasource(ds_id, ds_key) == 0:
                    continue
                cstore = ColBERTStore(config, colbert_coll["model"])
                orphan_ids, non_orphan_ds_ids = chunk_ds.preview_datasource_removal(ds_id, ds_key)
                if orphan_ids:
                    cstore.delete_by_ids(orphan_ids)
                if non_orphan_ds_ids:
                    cstore.update_ds_ids(non_orphan_ds_ids, wait=True)
            except Exception:
                logger.exception(
                    "ColBERT-Cleanup fuer %s fehlgeschlagen — DS wird per "
                    "Cascade geloescht; ColBERT-Reste sind mit aktueller "
                    "Repair-Oberflaeche nicht automatisch reparierbar und "
                    "muessen manuell/zukuenftig bereinigt werden",
                    colbert_coll["name"],
                )
    except Exception:
        logger.exception("ColBERT-Cleanup fuer DS %d fehlgeschlagen (fortfahrend)", ds_id)

    # 2. Elasticsearch (optional, non-fatal)
    #    Nutzt den unter Delete-Lock erzeugten file_hashes-Snapshot.
    try:
        from kiara.ingestion.vectordb_ops import delete_from_elasticsearch

        if old_file_paths:
            delete_from_elasticsearch(config, old_file_paths, datasource_id=ds_id)
    except Exception:
        logger.exception("Elasticsearch-Cleanup fuer DS %d fehlgeschlagen", ds_id)

    # 3. DB-Cascade + Graph-Cleanup (FATAL, atomar):
    # ingestion_file_logs -> file_hashes -> hash_mode_meta -> chunk_datasources ->
    # datasource_embeddings -> datasource_groups -> datasource_users ->
    # ingestion_jobs -> cache_datasources -> cache_jobs -> datasources
    # Bei Fehler: DS bleibt als soft-deleted markiert, Admin wird informiert.
    from kiara.ingestion._file_utils import ChunkDatasources

    chunk_ds = ChunkDatasources()
    with db_transaction() as conn:
        chunk_ds.acquire_graph_cleanup_lock(conn)
        graph_chunk_ids = chunk_ds.preview_graph_orphan_chunk_ids_after_datasource_removal(
            ds_id,
            conn=conn,
        )
        if graph_chunk_ids:
            from kiara.graph.store import GraphStore

            GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)
        ds_repo.delete_cascade(ds_id, conn=conn)

    # 4. Mount-Points (SMB/WebDAV, non-fatal)
    ds_type = ds.get("type")
    if ds_type in ("smb", "webdav"):
        try:
            from kiara.connectors.registry import auto_mount_point

            mp = auto_mount_point(ds)
            if ds_type == "webdav":
                # WebDAV braucht Cleanup von /etc/davfs2/secrets
                from kiara.shared.mount import unmount_webdav

                ok, msg = unmount_webdav(mp)
            else:
                from kiara.shared.mount import unmount_smb

                ok, msg = unmount_smb(mp)
            if not ok:
                logger.warning(
                    "Mount-Cleanup fuer DS %d (%s) fehlgeschlagen: %s", ds_id, ds_type, msg
                )
        except Exception:
            logger.exception("Mount-Cleanup fuer DS %d fehlgeschlagen", ds_id)

    # 5. Cron-Schedule (non-fatal)
    try:
        from kiara.web.cron_manager import remove_schedule

        remove_schedule(ds_id)
    except Exception:
        logger.exception("Schedule-Cleanup fuer DS %d fehlgeschlagen", ds_id)

    # 6. Cache
    from kiara.web.cache_refresh import delete_datasource_cache

    delete_datasource_cache(ds_id)

    return {"message": "Datenquelle geloescht"}


def resolve_masked_config(new_config: dict, existing_config: dict, connector_cls) -> dict:
    """Loest PASSWORD_MASK in connector_config auf.

    Wenn der User ein Passwort-Feld nicht aendert, sendet das Frontend PASSWORD_MASK.
    In diesem Fall wird der bestehende (verschluesselte) Wert beibehalten.
    Neue Passwort-Werte werden mit encrypt() verschluesselt.
    """
    resolved = {**existing_config, **new_config}
    for field in connector_cls.secret_fields():
        if field not in resolved:
            # Nicht mitgesendet → bestehenden (verschluesselten) Wert beibehalten
            if field in existing_config:
                resolved[field] = existing_config[field]
        elif resolved[field] == PASSWORD_MASK:
            if field in existing_config:
                resolved[field] = existing_config[field]
            else:
                del resolved[field]
        elif resolved[field] and not is_encrypted(resolved[field]):
            resolved[field] = encrypt(resolved[field])
    return resolved
