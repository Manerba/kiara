"""KIara Admin - Cache-Refresh fuer Admin-UI Tabs.

Vorberechnete JSON-Payloads in cache_datasources / cache_jobs / cache_vectordb.
Wird sowohl vom Ingestion-Worker als auch von der Admin-API aufgerufen.

Refresh-Arten:
    - Anlassbezogen: nach Statusaenderungen (Worker, API-Endpoints)
    - Periodisch:    Background-Thread in admin_app.py (Fallback)

Drosselung:
    _maybe_refresh_*() Funktionen begrenzen Schreibfrequenz im Worker
    (max alle 3 Sekunden), da Progress-Callbacks pro Datei feuern.
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from kiara.repositories.cache import CacheRepository
from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
from kiara.repositories.job import IngestionJobRepository
from kiara.repositories.permission import PermissionRepository
from kiara.shared.constants import CACHE_THROTTLE_SECONDS
from kiara.shared.db import db_query, db_transaction
from kiara.shared.enums import ConnectorType

logger = logging.getLogger(__name__)

# Module-level repo instances
_cache_repo = CacheRepository()
_ds_repo = DatasourceRepository()
_emb_repo = DatasourceEmbeddingRepository()
_job_repo = IngestionJobRepository()
_perm_repo = PermissionRepository()

# ---------------------------------------------------------------------------
# Drosselungs-State (Worker-Prozess)
# ---------------------------------------------------------------------------

_throttle_timestamps: dict[str, float] = {}
_throttle_lock = threading.Lock()
_THROTTLE_SECONDS = CACHE_THROTTLE_SECONDS


def _maybe_refresh(
    cache_key: str,
    refresh_fn: Callable[..., Any],
    *args: Any,
    interval: float = _THROTTLE_SECONDS,
) -> None:
    """Generisches Throttling fuer Cache-Refresh-Funktionen.

    Ruft *refresh_fn(*args)* hoechstens alle *interval* Sekunden auf.
    Fehler werden als DEBUG geloggt, nicht propagiert.
    """
    now = time.time()
    with _throttle_lock:
        if now - _throttle_timestamps.get(cache_key, 0.0) < interval:
            return
        _throttle_timestamps[cache_key] = now
    try:
        refresh_fn(*args)
    except Exception as exc:
        logger.warning("%s-Cache-Refresh fehlgeschlagen (args=%s): %s", cache_key, args, exc)


def _maybe_refresh_datasource_cache(ds_id: int) -> None:
    """Gedrosselte Variante — max alle 3s (fuer Progress-Callbacks)."""
    _maybe_refresh("DS", refresh_datasource_cache, ds_id)


def _maybe_refresh_jobs_cache(job_id: int) -> None:
    """Gedrosselte Variante — max alle 3s (fuer Progress-Callbacks)."""
    _maybe_refresh("Jobs", refresh_jobs_cache, job_id)


# ---------------------------------------------------------------------------
# Datasource-Cache
# ---------------------------------------------------------------------------


def _load_ds_batch_data(db, ds_id: int | None) -> tuple[list, dict, dict, dict, dict, dict]:
    """Laedt alle Batch-Daten fuer den DS-Cache in einem Read-Kontext.

    Returns:
        Tuple (rows, running_jobs, group_counts, user_counts, emb_map, size_map).
    """
    from kiara.retrieval.embedding import collection_name_for_model

    if ds_id is not None:
        rows = db.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchall()
    else:
        rows = db.execute("SELECT * FROM datasources ORDER BY id").fetchall()

    # Laufende/pausierte Jobs vorladen (1 Query statt N)
    job_rows = db.execute(
        "SELECT datasource_id, id, status, embedding_model FROM ingestion_jobs "
        "WHERE status IN ('running', 'paused')"
    ).fetchall()
    running_jobs = {jr["datasource_id"]: dict(jr) for jr in job_rows}

    # Group/User-Counts batch-laden (1 Query statt N pro DS)
    gc_rows = db.execute(
        "SELECT datasource_id, COUNT(*) AS count FROM datasource_groups GROUP BY datasource_id"
    ).fetchall()
    group_counts = {r["datasource_id"]: r["count"] for r in gc_rows}

    uc_rows = db.execute(
        "SELECT datasource_id, COUNT(*) AS count FROM datasource_users GROUP BY datasource_id"
    ).fetchall()
    user_counts = {r["datasource_id"]: r["count"] for r in uc_rows}

    # Embeddings batch-laden (1 Query statt N)
    emb_rows = db.execute(
        "SELECT datasource_id, embedding_model, chunk_count, file_count, "
        "last_ingested_at FROM datasource_embeddings ORDER BY embedding_model"
    ).fetchall()
    emb_map: dict[int, list] = {}
    for er in emb_rows:
        entry = dict(er)
        entry["collection"] = collection_name_for_model(er["embedding_model"])
        emb_map.setdefault(er["datasource_id"], []).append(entry)

    # Live Chunk-Counts fuer laufende Jobs aus chunk_datasources lesen,
    # da datasource_embeddings erst nach Job-Abschluss aktualisiert wird.
    if running_jobs:
        try:
            from kiara.config import load_config
            from kiara.ingestion._file_utils import ChunkDatasources

            load_config()
            chunk_ds = ChunkDatasources()
            for ds_id_val, job_info in running_jobs.items():
                em = job_info.get("embedding_model", "")
                if not em:
                    continue
                live_count = chunk_ds.count_by_datasource(ds_id_val, em)
                if live_count <= 0:
                    continue
                entries = emb_map.setdefault(ds_id_val, [])
                updated = False
                for entry in entries:
                    if entry["embedding_model"] == em:
                        entry["chunk_count"] = live_count
                        updated = True
                        break
                if not updated:
                    entries.append(
                        {
                            "datasource_id": ds_id_val,
                            "embedding_model": em,
                            "chunk_count": live_count,
                            "file_count": 0,
                            "last_ingested_at": None,
                            "collection": collection_name_for_model(em),
                        }
                    )
        except Exception as exc:
            logger.debug("Live-Chunk-Count konnte nicht geladen werden: %s", exc)

    # Deduplizierte Dateigroessen-Aggregation aus file_hashes
    if ds_id is not None:
        size_rows = db.execute(
            """SELECT datasource_id, COALESCE(SUM(file_size), 0) AS source_size_bytes
               FROM (
                   SELECT datasource_id, file_path, MAX(file_size) AS file_size
                   FROM file_hashes
                   WHERE datasource_id = %s
                   GROUP BY datasource_id, file_path
               ) s
               GROUP BY datasource_id""",
            (ds_id,),
        ).fetchall()
    else:
        size_rows = db.execute(
            """SELECT datasource_id, COALESCE(SUM(file_size), 0) AS source_size_bytes
               FROM (
                   SELECT datasource_id, file_path, MAX(file_size) AS file_size
                   FROM file_hashes
                   GROUP BY datasource_id, file_path
               ) s
               GROUP BY datasource_id"""
        ).fetchall()

    size_map = {r["datasource_id"]: r["source_size_bytes"] for r in size_rows}

    return rows, running_jobs, group_counts, user_counts, emb_map, size_map


def _load_live_progress(db, worker_job_id: int | None) -> dict[int, dict]:
    """Laedt Live-Progress-Daten aus dem laufenden Job.

    Returns:
        Dict DS-ID -> {progress_processed, progress_total, started_at, phase, job_id}.
    """
    if not worker_job_id:
        return {}
    prow = db.execute(
        "SELECT datasource_id, progress_processed, progress_total, "
        "started_at, stats_json FROM ingestion_jobs WHERE id = %s",
        (worker_job_id,),
    ).fetchone()
    if not prow:
        return {}
    from kiara.web.config_utils import safe_json_loads

    stats = safe_json_loads(prow["stats_json"] or "{}", {})
    return {
        prow["datasource_id"]: {
            "progress_processed": prow["progress_processed"],
            "progress_total": prow["progress_total"],
            "started_at": prow["started_at"],
            "phase": stats.get("phase", ""),
            "job_id": worker_job_id,
        }
    }


# Alte Konnektor-Spalten, die aus dem Payload entfernt werden (Legacy-Migration).
_LEGACY_CONNECTOR_FIELDS = (
    "smb_server",
    "smb_share",
    "smb_username",
    "smb_password",
    "smb_domain",
    "smb_mount_point",
    "webdav_url",
    "webdav_username",
    "webdav_password",
    "webdav_base_path",
    "imap_server",
    "imap_port",
    "imap_ssl",
    "imap_username",
    "imap_password",
    "imap_folders",
    "imap_since_days",
    "imap_download_attachments",
    "imap_max_attachment_mb",
    "sql_dsn",
    "sql_driver",
    "sql_query",
    "sql_text_columns",
    "sql_id_column",
    "sql_timestamp_column",
    "sql_text_template",
)


def _build_ds_payload(
    row: dict,
    running_jobs: dict,
    group_counts: dict,
    user_counts: dict,
    emb_map: dict,
    size_map: dict,
    live_progress: dict,
    worker_job_id: int | None,
    worker_paused: bool,
    get_connector: Callable,
    safe_json_loads: Callable,
) -> tuple[int, str]:
    """Baut das JSON-Payload fuer eine einzelne Datenquelle.

    Returns:
        Tuple (datasource_id, payload_json_string).
    """
    from kiara.shared.constants import PASSWORD_MASK

    d = dict(row)
    rid = row["id"]

    # Passwort-Maskierung
    raw_cc = d.get("connector_config", "{}")
    config = safe_json_loads(raw_cc, {})
    connector_cls = get_connector(d["type"])
    if connector_cls:
        for field in connector_cls.secret_fields():
            if config.get(field):
                config[field] = PASSWORD_MASK
        d["connector_config"] = config
        d["source_display"] = connector_cls.source_display(config, d)
        d["effective_path"] = connector_cls.effective_path(config, d)
    else:
        d["connector_config"] = config
        d["source_display"] = d.get("path", "")
        d["effective_path"] = ""

    if d.get("type") == ConnectorType.LOCAL:
        d["path"] = config.get("path", "")

    # Alte Konnektor-Spalten entfernen
    for f in _LEGACY_CONNECTOR_FIELDS:
        d.pop(f, None)

    # JSON-Felder decodieren
    for jf in ("ds_file_types", "ds_exclude_patterns"):
        if d.get(jf):
            d[jf] = safe_json_loads(d[jf], d[jf])

    # Live-Status-Override
    rj = running_jobs.get(rid)
    if rj and rj["id"] == worker_job_id and worker_paused:
        d["ingestion_status"] = "paused"
    elif rj and rj["status"] == "running":
        d["ingestion_status"] = "running"

    d["group_count"] = group_counts.get(rid, 0)
    d["user_count"] = user_counts.get(rid, 0)
    d["embeddings"] = emb_map.get(rid, [])

    # ColBERT-Embeddings (separater Cache-Block, aus chunk_datasources)
    colbert_embs = []
    try:
        from kiara.config import load_config
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.retrieval.colbert import (
            ColBERTStore,
            colbert_ds_model_key,
        )

        chunk_ds_inst = ChunkDatasources()
        for coll in ColBERTStore.list_colbert_collections(load_config()):
            ds_key = colbert_ds_model_key(coll["model"])
            count = chunk_ds_inst.count_by_datasource(rid, ds_key)
            if count > 0:
                # last_ingested_at aus dem Dense-Erfolgszeitstempel ableiten
                last_ts = None
                for emb_entry in emb_map.get(rid, []):
                    ts = emb_entry.get("last_ingested_at")
                    if ts and (last_ts is None or ts > last_ts):
                        last_ts = ts
                colbert_embs.append(
                    {
                        "model": coll["model"],
                        "collection": coll["name"],
                        "chunk_count": count,
                        "last_ingested_at": last_ts,
                    }
                )
    except Exception as exc:
        logger.warning("ColBERT-Statistiken nicht ladbar: %s", exc)
    d["colbert_embeddings"] = colbert_embs

    d["source_size_bytes"] = int(size_map.get(rid, 0))

    # Live-Progress fuer laufende Jobs
    lp = live_progress.get(rid)
    if lp:
        d["progress_processed"] = lp["progress_processed"]
        d["progress_total"] = lp["progress_total"]
        d["started_at"] = lp["started_at"]
        d["phase"] = lp["phase"]
        d["job_id"] = lp["job_id"]

    return rid, json.dumps(d, ensure_ascii=False, default=str)


def refresh_datasource_cache(ds_id: int | None = None) -> None:
    """Berechnet und schreibt den Cache fuer eine oder alle Datenquellen.

    Enthaelt alle Felder die ``GET /api/admin/datasources`` braucht:
    source_display, group_count, user_count, embeddings[], ingestion_status,
    progress_processed, progress_total, phase.  KEIN current_file.
    """
    from kiara.connectors.registry import (
        ensure_loaded,
    )
    from kiara.connectors.registry import (
        get as get_connector,
    )
    from kiara.web.config_utils import safe_json_loads

    ensure_loaded()

    # Read + Payload-Build + Write in EINER Transaktion — verhindert TOCTOU
    # zwischen Read-Snapshot und Cache-Write (#1026).
    with db_transaction() as db:
        rows, running_jobs, group_counts, user_counts, emb_map, size_map = _load_ds_batch_data(
            db, ds_id
        )

        if not rows:
            if ds_id is not None:
                db.execute(
                    "DELETE FROM cache_datasources WHERE datasource_id = %s",
                    (ds_id,),
                )
            else:
                db.execute("DELETE FROM cache_datasources")
            return

        from kiara.web.ingestion_worker import get_running_job_id, is_job_paused

        worker_job_id = get_running_job_id()
        worker_paused = is_job_paused(worker_job_id) if worker_job_id else False
        live_progress = _load_live_progress(db, worker_job_id)

        payloads = [
            _build_ds_payload(
                row,
                running_jobs,
                group_counts,
                user_counts,
                emb_map,
                size_map,
                live_progress,
                worker_job_id,
                worker_paused,
                get_connector,
                safe_json_loads,
            )
            for row in rows
        ]

        if ds_id is None:
            db.execute("DELETE FROM cache_datasources")

        for ds_id_val, payload in payloads:
            db.execute(
                "INSERT INTO cache_datasources (datasource_id, payload_json, updated_at) "
                "VALUES (%s, %s, NOW()) "
                "ON CONFLICT(datasource_id) DO UPDATE SET "
                "payload_json = excluded.payload_json, updated_at = excluded.updated_at",
                (ds_id_val, payload),
            )


def delete_datasource_cache(ds_id: int) -> None:
    """Entfernt eine Datenquelle aus dem Cache (nach DELETE)."""
    with db_transaction() as tx:
        tx.execute("DELETE FROM cache_datasources WHERE datasource_id = %s", (ds_id,))


# ---------------------------------------------------------------------------
# Jobs-Cache
# ---------------------------------------------------------------------------

# SQL-Filter: Intermediates aus Resume-Chains ausblenden
_CHAIN_FILTER = (
    "j.id NOT IN (SELECT resumed_from_job FROM ingestion_jobs WHERE resumed_from_job IS NOT NULL)"
)


def _load_visible_jobs(db, job_id: int | None) -> list | None:
    """Laedt sichtbare Jobs (nicht-Intermediates) fuer den Cache.

    Returns:
        Liste von Job-Rows, oder None wenn ein Einzel-Job nicht sichtbar ist.
    """
    if job_id is not None:
        rows = db.execute(
            "SELECT j.*, d.name AS datasource_name "
            "FROM ingestion_jobs j "
            "LEFT JOIN datasources d ON j.datasource_id = d.id "
            "WHERE j.id = %s AND " + _CHAIN_FILTER,
            (job_id,),
        ).fetchall()
        if not rows:
            return None
        return rows
    return db.execute(
        "SELECT j.*, d.name AS datasource_name "
        "FROM ingestion_jobs j "
        "LEFT JOIN datasources d ON j.datasource_id = d.id "
        "WHERE " + _CHAIN_FILTER + " "
        "ORDER BY j.id DESC LIMIT 200"
    ).fetchall()


_MAX_RESUME_CHAIN_DEPTH = 100


def _load_job_ancestors(db, rows: list) -> dict[int, dict]:
    """Laedt alle Ancestor-Jobs fuer Resume-Chains in einem Batch.

    Returns:
        Dict ancestor_id -> ancestor_row_dict.
    """
    ancestor_ids: set[int] = set()
    for row in rows:
        if row["resumed_from_job"]:
            current = row["resumed_from_job"]
            seen: set[int] = set()
            depth = 0
            while current and current not in seen:
                if depth >= _MAX_RESUME_CHAIN_DEPTH:
                    logger.warning(
                        "Resume-Chain-Tiefe von %d ueberschritten bei job-ancestor=%s — Chain abgebrochen (DB-Korruption?)",
                        _MAX_RESUME_CHAIN_DEPTH,
                        current,
                    )
                    break
                seen.add(current)
                ancestor_ids.add(current)
                anc = db.execute(
                    "SELECT resumed_from_job FROM ingestion_jobs WHERE id = %s",
                    (current,),
                ).fetchone()
                current = anc["resumed_from_job"] if anc else None
                depth += 1

    if not ancestor_ids:
        return {}
    placeholders = ",".join("%s" for _ in ancestor_ids)
    anc_rows = db.execute(
        "SELECT id, status, started_at, finished_at, error_message, "
        "stats_json, resumed_from_job "
        f"FROM ingestion_jobs WHERE id IN ({placeholders})",
        list(ancestor_ids),
    ).fetchall()
    return {ar["id"]: dict(ar) for ar in anc_rows}


def _build_resume_chain(job: dict, ancestors_map: dict) -> None:
    """Baut die Resume-Chain fuer einen Job aus vorgeladenen Ancestors (in-place)."""
    if not job.get("resumed_from_job"):
        return
    chain = []
    ancestor_duration = 0
    current = job["resumed_from_job"]
    seen: set[int] = set()
    while current and current not in seen:
        seen.add(current)
        ancestor = ancestors_map.get(current)
        if not ancestor:
            break
        a_duration = 0
        if ancestor["started_at"] and ancestor["finished_at"]:
            from kiara.web.config_utils import safe_json_loads as _sjl

            a_stats = _sjl(ancestor["stats_json"], {})
            if isinstance(a_stats, dict):
                a_duration = a_stats.get("duration_seconds", 0)
        ancestor_duration += a_duration
        chain.append(
            {
                "id": ancestor["id"],
                "status": ancestor["status"],
                "started_at": ancestor["started_at"],
                "finished_at": ancestor["finished_at"],
                "error_message": ancestor["error_message"] or "",
                "duration_seconds": a_duration,
            }
        )
        current = ancestor.get("resumed_from_job")
    if chain:
        job["resume_chain_length"] = len(chain) + 1
        job["resume_chain_original_job_id"] = chain[-1]["id"]
        job["resume_chain_ancestor_duration"] = ancestor_duration
        job["resume_chain"] = chain


def _build_job_payload(
    row: dict,
    ancestors_map: dict,
    current_worker_id: int | None,
    worker_paused: bool,
) -> tuple[int, int, str, str]:
    """Baut das JSON-Payload fuer einen einzelnen Job.

    Returns:
        Tuple (job_id, datasource_id, status, payload_json_string).
    """
    from kiara.shared.enums import IngestionStatus
    from kiara.web.config_utils import safe_json_loads

    job = dict(row)
    # Stats aus JSON extrahieren (JSONB liefert dict, TEXT liefert str)
    job["stats"] = safe_json_loads(job.get("stats_json") or "{}", {})
    job.pop("stats_json", None)

    _build_resume_chain(job, ancestors_map)

    # Ghost/Pause-Detection
    if job["status"] == IngestionStatus.RUNNING:
        if job["id"] == current_worker_id:
            job["is_paused"] = worker_paused
        else:
            job["is_ghost"] = True

    return (
        job["id"],
        job.get("datasource_id", 0),
        job.get("status", ""),
        json.dumps(job, ensure_ascii=False, default=str),
    )


def refresh_jobs_cache(job_id: int | None = None) -> None:
    """Berechnet und schreibt den Cache fuer einen oder alle Jobs.

    Enthaelt Stats, Resume-Chain (vorberechnet), Ghost-Detection, is_paused.
    """
    from kiara.web.ingestion_worker import get_running_job_id, is_job_paused

    # Read + Payload-Build + Write in EINER Transaktion — verhindert TOCTOU
    # zwischen Read-Snapshot und Cache-Write (#1026).
    with db_transaction() as db:
        rows = _load_visible_jobs(db, job_id)
        if rows is None:
            db.execute("DELETE FROM cache_jobs WHERE job_id = %s", (job_id,))
            return
        ancestors_map = _load_job_ancestors(db, rows)

        current_worker_id = get_running_job_id()
        worker_paused = is_job_paused(current_worker_id) if current_worker_id else False
        payloads = [
            _build_job_payload(row, ancestors_map, current_worker_id, worker_paused) for row in rows
        ]

        if job_id is None:
            db.execute("DELETE FROM cache_jobs")
        for jid, ds_id_val, status, payload in payloads:
            db.execute(
                "INSERT INTO cache_jobs (job_id, datasource_id, status, payload_json, updated_at) "
                "VALUES (%s, %s, %s, %s, NOW()) "
                "ON CONFLICT(job_id) DO UPDATE SET "
                "datasource_id = excluded.datasource_id, "
                "status = excluded.status, payload_json = excluded.payload_json, "
                "updated_at = excluded.updated_at",
                (jid, ds_id_val, status, payload),
            )


# ---------------------------------------------------------------------------
# VektorDB-Cache
# ---------------------------------------------------------------------------


def refresh_vectordb_cache() -> None:
    """Berechnet und schreibt den VektorDB-Backend-Status (Singleton).

    Macht Qdrant-Calls — daher nur periodisch aufrufen (120s).
    """
    from kiara.config import load_config
    from kiara.retrieval.vectorstore_factory import list_collections_for_backend

    config = load_config()

    backends = []
    for backend_name in ("qdrant",):
        info = {
            "name": backend_name,
            "is_active": True,
            "reachable": False,
            "collections": [],
            "colbert_collections": [],
            "error": None,
        }
        try:
            collections = list_collections_for_backend(config)
            # model_name aus datasource_embeddings joinen
            model_map = {}
            try:
                active_models = _emb_repo.get_active_models()
                from kiara.retrieval.embedding import collection_name_for_model

                for model_name in active_models:
                    col_name = collection_name_for_model(model_name)
                    model_map[col_name] = model_name
            except Exception as _exc:
                logger.warning("model_map konnte nicht erstellt werden: %s", _exc)

            for col in collections:
                col["model_name"] = model_map.get(col.get("name", ""), col.get("model", ""))
            info["reachable"] = True
            info["collections"] = collections

            # ColBERT-Collections (separat)
            try:
                from kiara.retrieval.colbert import ColBERTStore

                info["colbert_collections"] = ColBERTStore.list_colbert_collections(config)
            except Exception:
                info["colbert_collections"] = []

            # chunks_by_model: Dense + ColBERT Union
            chunks_by_model = []
            for col in collections:
                chunks_by_model.append(
                    {
                        "kind": "kb",
                        "model": col.get("model_name", col.get("model", "")),
                        "chunks": col.get("chunks", 0),
                    }
                )
            for col in info["colbert_collections"]:
                chunks_by_model.append(
                    {
                        "kind": "colbert",
                        "model": col.get("model", ""),
                        "chunks": col.get("chunks", 0),
                    }
                )
            info["chunks_by_model"] = chunks_by_model

            # Gesamt-Chunks (Dense-only aus datasource_embeddings)
            try:
                with db_query() as _db2:
                    _total = _db2.execute(
                        "SELECT COALESCE(SUM(chunk_count), 0) AS total "
                        "FROM datasource_embeddings WHERE embedding_model != ''"
                    ).fetchone()
                info["total_chunks"] = _total["total"] if _total else 0
            except Exception:
                info["total_chunks"] = 0

            # total_unique_chunks: eindeutige chunk_ids ohne Doppelzaehlung
            try:
                with db_query() as _db3:
                    _unique = _db3.execute(
                        "SELECT COUNT(DISTINCT chunk_id) AS total FROM chunk_datasources"
                    ).fetchone()
                info["total_unique_chunks"] = _unique["total"] if _unique else 0
            except Exception:
                info["total_unique_chunks"] = info["total_chunks"]

        except Exception as exc:
            info["error"] = str(exc)
        backends.append(info)

    payload = json.dumps({"backends": backends}, ensure_ascii=False, default=str)
    with db_transaction() as tx:
        tx.execute(
            "INSERT INTO cache_vectordb (id, payload_json, updated_at) "
            "VALUES (1, %s, NOW()) "
            "ON CONFLICT(id) DO UPDATE SET "
            "payload_json = excluded.payload_json, updated_at = excluded.updated_at",
            (payload,),
        )


# ---------------------------------------------------------------------------
# Voll-Refresh (fuer Background-Thread)
# ---------------------------------------------------------------------------


def refresh_all() -> None:
    """Alle Caches aktualisieren (fuer periodischen Background-Thread)."""
    try:
        refresh_datasource_cache()
    except Exception as exc:
        logger.warning("DS-Cache-Refresh fehlgeschlagen: %s", exc)
    try:
        refresh_jobs_cache()
    except Exception as exc:
        logger.warning("Jobs-Cache-Refresh fehlgeschlagen: %s", exc)


def refresh_all_with_vectordb() -> None:
    """Alle Caches inkl. VektorDB aktualisieren."""
    refresh_all()
    try:
        refresh_vectordb_cache()
    except Exception as exc:
        logger.warning("VektorDB-Cache-Refresh fehlgeschlagen: %s", exc)
