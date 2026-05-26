"""
Integrity-Service — Background-Thread fuer Integritaetspruefungen und Repairs.
Analog zu elasticsearch_service.py.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_task: dict = {
    "running": False,
    "action": None,  # "check" | "repair_ds_ids" | "repair_orphans" | "repair_stats"
    "progress": None,  # {"step"|"processed": int, "total": int, "label": str}
    "result": None,  # {"status": "success"|"error", ...}
}

_VALID_REPAIRS = {
    "repair_ds_ids",
    "repair_orphans",
    "repair_stats",
    "repair_reset_meta",
    "repair_reconstruct",
    "repair_rebuild",
    "repair_cleanup_colbert",
}


def is_running() -> bool:
    with _lock:
        return _task["running"]


def get_task_status() -> dict:
    with _lock:
        return dict(_task)


def start_check() -> None:
    """Startet die Integritaetspruefung im Background-Thread."""
    with _lock:
        if _task["running"]:
            raise RuntimeError("Ein Integritaets-Task laeuft bereits")
        _task.update({"running": True, "action": "check", "progress": None, "result": None})

    thread = threading.Thread(target=_run_check, daemon=True)
    thread.start()


def start_repair(action: str) -> None:
    """Startet eine Repair-Aktion im Background-Thread.

    Args:
        action: "repair_ds_ids" | "repair_orphans" | "repair_stats"
    """
    if action not in _VALID_REPAIRS:
        raise ValueError(f"Unbekannte Repair-Aktion: {action}")

    # Issue #1058: Repairs duerfen NICHT parallel zu einer laufenden Ingestion laufen —
    # repair_orphans wuerde Qdrant-Punkte loeschen, deren chunk_datasources-Eintrag der
    # Ingestion-Worker gerade erst danach commitet (Race -> Datenverlust).
    from kiara.web.ingestion_worker import get_running_job_id

    running_job = get_running_job_id()
    if running_job is not None:
        raise RuntimeError(
            f"Integritaets-Repair kann nicht starten — Ingestion-Job #{running_job} laeuft. "
            f"Bitte erst die Ingestion abschliessen oder abbrechen."
        )

    with _lock:
        if _task["running"]:
            raise RuntimeError("Ein Integritaets-Task laeuft bereits")
        _task.update({"running": True, "action": action, "progress": None, "result": None})

    dispatch = {
        "repair_ds_ids": _run_repair_ds_ids,
        "repair_orphans": _run_repair_orphans,
        "repair_stats": _run_repair_stats,
        "repair_reset_meta": _run_repair_reset_meta,
        "repair_reconstruct": _run_repair_reconstruct,
        "repair_rebuild": _run_repair_rebuild,
        "repair_cleanup_colbert": _run_repair_cleanup_colbert,
    }
    thread = threading.Thread(target=dispatch[action], daemon=True)
    thread.start()


def _run_check() -> None:
    """Background-Thread: Fuehrt alle Integritaetschecks aus."""
    try:
        from kiara.config import load_config
        from kiara.integrity.checks import run_full_check

        config = load_config()

        def _progress(step: int, total: int, label: str) -> None:
            with _lock:
                _task["progress"] = {"step": step, "total": total, "label": label}

        result = run_full_check(config, progress_callback=_progress)
        with _lock:
            _task["result"] = {"status": "success", **result}
    except Exception as exc:
        logger.exception("Integritaetspruefung fehlgeschlagen")
        with _lock:
            _task["result"] = {"status": "error", "error": str(exc)}
    finally:
        with _lock:
            _task["running"] = False
            _task["action"] = None


# ---------------------------------------------------------------------------
# Repair: ds_ids in Qdrant synchronisieren
# ---------------------------------------------------------------------------


def _run_repair_ds_ids() -> None:
    """Background-Thread: Synchronisiert ds_ids in Qdrant mit chunk_datasources."""
    try:
        from kiara.config import load_config
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.retrieval.vectorstore import VectorStore

        config = load_config()
        chunk_ds = ChunkDatasources()

        # Multi-Collection-Iteration (alle Modelle, analog zu repair_orphans)
        collections = VectorStore.list_collections(config)
        non_empty = [c for c in collections if c.get("chunks", 0) > 0]

        updated = 0

        def _progress(label: str) -> None:
            with _lock:
                _task["progress"] = {"processed": updated, "total": 0, "label": label}

        from kiara.retrieval import elasticsearch_service as es_svc

        es_enabled = es_svc.is_configured(config)

        _progress("ds_ids-Sync gestartet...")

        for coll in non_empty:
            model = coll["model"]
            store = VectorStore(config, embedding_model=model)
            for batch_map in chunk_ds.get_all_ds_ids_grouped(model, batch_size=1000):
                store.update_ds_ids(batch_map)
                if es_enabled:
                    es_svc.update_ds_ids(config, batch_map)
                updated += len(batch_map)
                _progress(f"[{model}] {updated:,} Chunks aktualisiert")

        with _lock:
            _task["result"] = {
                "status": "success",
                "action": "repair_ds_ids",
                "updated": updated,
                "collections": len(non_empty),
                "message": f"ds_ids-Sync abgeschlossen: {updated:,} Chunks aktualisiert "
                f"({len(non_empty)} Collection(s))",
            }
    except Exception as exc:
        logger.exception("ds_ids-Repair fehlgeschlagen")
        with _lock:
            _task["result"] = {"status": "error", "action": "repair_ds_ids", "error": str(exc)}
    finally:
        with _lock:
            _task["running"] = False
            _task["action"] = None


# ---------------------------------------------------------------------------
# Repair: Verwaiste Chunks loeschen
# ---------------------------------------------------------------------------


def _run_repair_orphans() -> None:
    """Background-Thread: Loescht verwaiste Chunks aus Qdrant (vollstaendiger Scan).

    Iteriert ueber ALLE nicht-leeren Qdrant-Collections (Multi-Modell).
    """
    try:
        from kiara.config import load_config
        from kiara.retrieval.vectorstore import VectorStore
        from kiara.shared.db import db_query

        config = load_config()

        # Multi-Collection-Iteration
        collections = VectorStore.list_collections(config)
        non_empty = [c for c in collections if c.get("chunks", 0) > 0]
        grand_total = sum(c["chunks"] for c in non_empty)

        processed = 0
        total_deleted = 0

        def _progress(label: str) -> None:
            with _lock:
                _task["progress"] = {
                    "processed": min(processed, grand_total),
                    "total": grand_total,
                    "label": label,
                }

        _progress("Orphan-Scan gestartet...")

        for coll in non_empty:
            coll_label = coll["model"]
            store = VectorStore(config, embedding_model=coll["model"])
            orphan_ids = []  # Pro Collection zuruecksetzen!

            # Phase 1: Vollstaendiger Scan dieser Collection
            # Design-Entscheidung: KEIN embedding_model-Filter (gleiche Begruendung
            # wie bei check_orphan_chunks — siehe Kommentar dort).
            for batch in store.scroll_all(batch_size=1000):
                batch_ids = [cid for cid, _ in batch]
                known_ids = set()
                for i in range(0, len(batch_ids), 500):
                    sub = batch_ids[i : i + 500]
                    ph = ",".join(["%s"] * len(sub))
                    with db_query() as conn:
                        rows = conn.execute(
                            f"SELECT DISTINCT chunk_id FROM chunk_datasources "
                            f"WHERE chunk_id IN ({ph})",
                            sub,
                        ).fetchall()
                        known_ids.update(r["chunk_id"] for r in rows)

                for cid in batch_ids:
                    if cid not in known_ids:
                        orphan_ids.append(cid)
                processed += len(batch_ids)
                _progress(
                    f"[{coll_label}] Gescannt: {processed:,}/{grand_total:,}, "
                    f"Orphans: {len(orphan_ids):,}"
                )

            # Phase 2: Orphans dieser Collection loeschen
            if orphan_ids:
                _progress(f"[{coll_label}] Loesche {len(orphan_ids):,} verwaiste Chunks...")
                store.delete_by_ids(orphan_ids)
                total_deleted += len(orphan_ids)

        with _lock:
            _task["result"] = {
                "status": "success",
                "action": "repair_orphans",
                "scanned": processed,
                "deleted": total_deleted,
                "message": f"Orphan-Cleanup: {total_deleted:,} Chunks geloescht "
                f"(von {processed:,} gescannt, "
                f"{len(non_empty)} Collection(s))",
            }
    except Exception as exc:
        logger.exception("Orphan-Repair fehlgeschlagen")
        with _lock:
            _task["result"] = {"status": "error", "action": "repair_orphans", "error": str(exc)}
    finally:
        with _lock:
            _task["running"] = False
            _task["action"] = None


# ---------------------------------------------------------------------------
# Repair: Statistiken in datasource_embeddings korrigieren
# ---------------------------------------------------------------------------


def _run_repair_stats() -> None:
    """Background-Thread: Korrigiert chunk_count in datasource_embeddings."""
    try:
        from kiara.config import load_config
        from kiara.repositories.datasource import DatasourceEmbeddingRepository
        from kiara.retrieval.vectorstore import VectorStore
        from kiara.shared.db import db_query

        config = load_config()
        de_repo = DatasourceEmbeddingRepository()

        # Multi-Modell-Iteration (alle Modelle, analog zu repair_orphans)
        collections = VectorStore.list_collections(config)
        non_empty = [c for c in collections if c.get("chunks", 0) > 0]
        models = [c["model"] for c in non_empty]

        total_checked = 0
        total_corrected = 0

        def _progress(label: str) -> None:
            with _lock:
                _task["progress"] = {
                    "processed": total_corrected,
                    "total": total_checked,
                    "label": label,
                }

        _progress("Stats-Korrektur gestartet...")

        for model in models:
            # Tatsaechliche Counts aus chunk_datasources
            with db_query() as conn:
                actual_rows = conn.execute(
                    "SELECT datasource_id, COUNT(*) AS cnt "
                    "FROM chunk_datasources WHERE embedding_model = %s "
                    "GROUP BY datasource_id",
                    (model,),
                ).fetchall()
            actual_counts = {r["datasource_id"]: r["cnt"] for r in actual_rows}

            # Bestehende Counts aus datasource_embeddings
            with db_query() as conn:
                de_rows = conn.execute(
                    "SELECT datasource_id, chunk_count FROM datasource_embeddings "
                    "WHERE embedding_model = %s",
                    (model,),
                ).fetchall()
            existing_counts = {r["datasource_id"]: r["chunk_count"] for r in de_rows}

            all_ds_ids = sorted(set(actual_counts.keys()) | set(existing_counts.keys()))
            total_checked += len(all_ds_ids)

            for i, ds_id in enumerate(all_ds_ids):
                actual = actual_counts.get(ds_id, 0)
                existing = existing_counts.get(ds_id, -1)
                if actual != existing:
                    de_repo.upsert_chunk_count(ds_id, model, actual)
                    total_corrected += 1
                if (i + 1) % 5 == 0 or i + 1 == len(all_ds_ids):
                    _progress(
                        f"[{model}] {i + 1}/{len(all_ds_ids)} DS geprueft, "
                        f"gesamt {total_corrected} korrigiert"
                    )

        with _lock:
            _task["result"] = {
                "status": "success",
                "action": "repair_stats",
                "checked": total_checked,
                "corrected": total_corrected,
                "models": len(models),
                "message": f"Stats-Korrektur: {total_corrected} von {total_checked} "
                f"Datenquellen aktualisiert ({len(models)} Modell(e))",
            }
    except Exception as exc:
        logger.exception("Stats-Repair fehlgeschlagen")
        with _lock:
            _task["result"] = {"status": "error", "action": "repair_stats", "error": str(exc)}
    finally:
        with _lock:
            _task["running"] = False
            _task["action"] = None


# ---------------------------------------------------------------------------
# Repair: Metadaten-Reset (wenn Qdrant leer, aber Metadaten vorhanden)
# ---------------------------------------------------------------------------


def _run_repair_reset_meta() -> None:
    """Background-Thread: Loescht verwaiste Metadaten wenn Qdrant-Collection leer ist."""
    try:
        from kiara.config import load_config
        from kiara.retrieval.vectorstore import VectorStore
        from kiara.shared.db import db_transaction

        config = load_config()

        def _progress(label: str) -> None:
            with _lock:
                _task["progress"] = {"processed": 0, "total": 3, "label": label}

        # Safety-Check: ALLE Qdrant-Collections muessen leer sein
        collections = VectorStore.list_collections(config)
        qdrant_total = sum(c.get("chunks", 0) for c in collections)
        if qdrant_total > 0:
            non_empty = [
                f"{c['name']} ({c['chunks']:,})" for c in collections if c.get("chunks", 0) > 0
            ]
            with _lock:
                _task["result"] = {
                    "status": "error",
                    "action": "repair_reset_meta",
                    "error": f"Abbruch: Qdrant enthaelt {qdrant_total:,} Chunks in "
                    f"{', '.join(non_empty)}. "
                    f"Metadaten-Reset nur bei leeren Collections erlaubt.",
                }
            return

        _progress("Loesche Metadaten (chunk_datasources, file_hashes, datasource_embeddings)...")
        with db_transaction() as conn:
            from kiara.ingestion._file_utils import ChunkDatasources

            chunk_ds = ChunkDatasources()
            chunk_ds.acquire_graph_cleanup_lock(conn)
            candidate_rows = conn.execute(
                "SELECT DISTINCT chunk_id FROM chunk_datasources"
            ).fetchall()
            graph_candidates = [r["chunk_id"] for r in candidate_rows]
            r1 = conn.execute("DELETE FROM chunk_datasources").rowcount
            r2 = conn.execute("DELETE FROM file_hashes").rowcount
            r3 = conn.execute("DELETE FROM datasource_embeddings").rowcount
            if graph_candidates:
                from kiara.graph.store import GraphStore

                graph_chunk_ids = chunk_ds.filter_unreferenced_chunk_ids(
                    graph_candidates,
                    conn=conn,
                )
                if graph_chunk_ids:
                    GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)

        with _lock:
            _task["result"] = {
                "status": "success",
                "action": "repair_reset_meta",
                "deleted_chunk_ds": r1,
                "deleted_file_hashes": r2,
                "deleted_ds_embeddings": r3,
                "message": f"Metadaten-Reset: {r1:,} chunk_datasources, "
                f"{r2:,} file_hashes, {r3:,} datasource_embeddings geloescht",
            }
    except Exception as exc:
        logger.exception("Metadaten-Reset fehlgeschlagen")
        with _lock:
            _task["result"] = {"status": "error", "action": "repair_reset_meta", "error": str(exc)}
    finally:
        with _lock:
            _task["running"] = False
            _task["action"] = None


# ---------------------------------------------------------------------------
# Repair: Metadaten aus Qdrant-Daten rekonstruieren
# ---------------------------------------------------------------------------


def _load_datasource_paths(config: dict) -> list[dict]:
    """Laedt aktive Datasources mit normalisierten Pfaden fuer Longest-Prefix-Match."""
    from kiara.shared.db import db_query
    from kiara.web.api_datasource import _effective_path

    with db_query() as conn:
        ds_rows = conn.execute(
            "SELECT * FROM datasources WHERE is_active = true ORDER BY id"
        ).fetchall()

    ds_paths = []
    for row in ds_rows:
        try:
            path = _effective_path(dict(row))
        except Exception:
            path = row.get("path") or ""
        if path:
            norm = path.rstrip("/") + "/"
            ds_paths.append({"id": row["id"], "name": row["name"], "path": norm})

    ds_paths.sort(key=lambda d: len(d["path"]), reverse=True)
    return ds_paths


_CHUNK_DS_INSERT_SQL = (
    "INSERT INTO chunk_datasources "
    "(chunk_id, datasource_id, embedding_model) VALUES (%s, %s, %s) "
    "ON CONFLICT (chunk_id, datasource_id, embedding_model) DO NOTHING"
)


def _scan_qdrant_assignments(config, ds_paths, non_empty, progress_fn):
    """Scannt Qdrant-Collections und schreibt Zuordnungen in chunk_datasources.

    Returns: {"non_empty": list, "processed_total": int, "total_inserted": int,
              "unmatched": int, "grand_total": int}
    """
    from kiara.ingestion._file_utils import ChunkDatasources
    from kiara.retrieval.vectorstore import VectorStore
    from kiara.shared.db import db_transaction, get_db

    grand_total = sum(c["chunks"] for c in non_empty)
    processed_total = 0
    total_inserted = 0
    unmatched = 0
    # Kandidaten-Snapshot aller betroffenen chunk_ids, bevor chunk_datasources
    # pro Modell geleert wird — wird am Ende unter demselben Session-Lock
    # durch filter_unreferenced_chunk_ids + GraphStore.delete_by_chunk_ids
    # aufgearbeitet (ImplScan V28 Finding 4).
    graph_cleanup_candidates: set[str] = set()
    lock_conn = get_db()
    try:
        ChunkDatasources.acquire_graph_cleanup_session_lock(lock_conn)
        for coll_info in non_empty:
            coll_name = coll_info["name"]
            model = coll_info["model"]
            progress_fn(
                f"Collection {coll_name}: Entferne stale Eintraege...", processed_total, grand_total
            )

            with db_transaction() as conn:
                # Snapshot BEVOR die Rows verschwinden — diese Chunks koennen
                # nach dem Scan ohne Reinsertion enden (Qdrant-leere Payloads,
                # keine DS-Pfad-Treffer). Ihre graph_node_chunks- und
                # graph_edge_chunks-Links muessen dann am Ende geprueft werden.
                snapshot_rows = conn.execute(
                    "SELECT DISTINCT chunk_id FROM chunk_datasources WHERE embedding_model = %s",
                    (model,),
                ).fetchall()
                for row in snapshot_rows:
                    graph_cleanup_candidates.add(row["chunk_id"])
                conn.execute(
                    "DELETE FROM chunk_datasources WHERE embedding_model = %s",
                    (model,),
                )

            progress_fn(f"Collection {coll_name}: Starte Scan...", processed_total, grand_total)

            store = VectorStore(config, embedding_model=model)
            batch_inserts = []

            for batch in store.scroll_all(batch_size=1000):
                for chunk_id, payload in batch:
                    source = (payload or {}).get("source", "")
                    if not source:
                        unmatched += 1
                        continue

                    matched_ds = None
                    for ds in ds_paths:
                        if source.startswith(ds["path"]) or source == ds["path"].rstrip("/"):
                            matched_ds = ds
                            break

                    if matched_ds:
                        batch_inserts.append((chunk_id, matched_ds["id"], model))
                    else:
                        unmatched += 1

                if len(batch_inserts) >= 5000:
                    with db_transaction() as conn:
                        conn.executemany(_CHUNK_DS_INSERT_SQL, batch_inserts)
                    total_inserted += len(batch_inserts)
                    batch_inserts = []

                processed_total += len(batch)
                progress_fn(
                    f"Collection {coll_name}: {processed_total:,}/{grand_total:,} "
                    f"({total_inserted:,} zugeordnet, {unmatched:,} ohne Match)",
                    processed_total,
                    grand_total,
                )

            if batch_inserts:
                with db_transaction() as conn:
                    conn.executemany(_CHUNK_DS_INSERT_SQL, batch_inserts)
                total_inserted += len(batch_inserts)

        # Graph-Cleanup fuer Chunks, die nach dem Scan NICHT mehr in
        # chunk_datasources stehen (kein source-Match, leerer Payload).
        # Muss unter demselben Session-Lock laufen, damit zwischen
        # filter_unreferenced_chunk_ids und delete_by_chunk_ids kein
        # paralleler Writer Rows einschleust. Fehler werden geloggt aber nicht
        # weitergereicht — die Repair-Aktion bleibt erfolgreich, stale Graph-
        # Links sind self-healing beim naechsten Delete- oder Full-Reingest-
        # Aufruf (ImplScan V28 Finding 4).
        if graph_cleanup_candidates:
            try:
                chunk_ds = ChunkDatasources()
                with db_transaction() as conn:
                    unreferenced = chunk_ds.filter_unreferenced_chunk_ids(
                        sorted(graph_cleanup_candidates),
                        conn=conn,
                    )
                if unreferenced:
                    from kiara.graph.store import GraphStore

                    GraphStore().delete_by_chunk_ids(unreferenced)
                    logger.info(
                        "Integrity-Reconstruct: %d Graph-Chunks bereinigt (aus %d Kandidaten).",
                        len(unreferenced),
                        len(graph_cleanup_candidates),
                    )
            except Exception as exc:
                logger.warning(
                    "Integrity-Reconstruct Graph-Cleanup fehlgeschlagen "
                    "(stale Junctions bleiben, self-healing beim naechsten "
                    "Delete- oder Full-Reingest-Aufruf): %s",
                    exc,
                )
    finally:
        try:
            ChunkDatasources.release_graph_cleanup_session_lock(lock_conn)
        finally:
            lock_conn.close()

    return {
        "processed_total": processed_total,
        "total_inserted": total_inserted,
        "unmatched": unmatched,
        "grand_total": grand_total,
    }


def _recalc_datasource_embeddings(config: dict) -> None:
    """Berechnet datasource_embeddings aus chunk_datasources-Counts neu."""
    from kiara.repositories.datasource import DatasourceEmbeddingRepository
    from kiara.shared.db import db_transaction

    de_repo = DatasourceEmbeddingRepository()
    with db_transaction() as conn:
        conn.execute(
            "DELETE FROM chunk_datasources cd "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM datasources ds "
            "  WHERE ds.id = cd.datasource_id"
            ")"
        )
        count_rows = conn.execute(
            "SELECT cd.datasource_id, cd.embedding_model, COUNT(*) AS cnt, "
            "COUNT(DISTINCT cd.chunk_id) AS file_est "
            "FROM chunk_datasources cd "
            "JOIN datasources ds ON ds.id = cd.datasource_id "
            "GROUP BY cd.datasource_id, cd.embedding_model"
        ).fetchall()
        conn.execute(
            "DELETE FROM datasource_embeddings de "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM datasources ds "
            "  WHERE ds.id = de.datasource_id"
            ") OR NOT EXISTS ("
            "  SELECT 1 FROM chunk_datasources cd "
            "  WHERE cd.datasource_id = de.datasource_id "
            "  AND cd.embedding_model = de.embedding_model"
            ")"
        )
        for r in count_rows:
            de_repo.upsert_chunk_count(
                r["datasource_id"],
                r["embedding_model"],
                r["cnt"],
                conn=conn,
            )


def _apply_ds_id_updates(config, non_empty, progress_fn, processed_total, grand_total) -> int:
    """Aktualisiert ds_ids in Qdrant + ES fuer alle Collections."""
    from kiara.ingestion._file_utils import ChunkDatasources
    from kiara.retrieval import elasticsearch_service as es_svc
    from kiara.retrieval.vectorstore import VectorStore

    chunk_ds = ChunkDatasources()
    es_enabled = es_svc.is_configured(config)
    ds_ids_updated = 0
    for coll_info in non_empty:
        model = coll_info["model"]
        store = VectorStore(config, embedding_model=model)
        for batch_map in chunk_ds.get_all_ds_ids_grouped(model, batch_size=1000):
            try:
                store.update_ds_ids(batch_map)
                if es_enabled:
                    es_svc.update_ds_ids(config, batch_map)
            except Exception as exc:
                logger.warning(
                    "Batch update_ds_ids fehlgeschlagen (model=%s, %d chunks): %s",
                    model,
                    len(batch_map),
                    exc,
                )
                continue
            ds_ids_updated += len(batch_map)
            progress_fn(f"ds_ids aktualisiert: {ds_ids_updated:,}", processed_total, grand_total)
    return ds_ids_updated


def _run_repair_reconstruct() -> None:
    """Background-Thread: Rekonstruiert chunk_datasources + datasource_embeddings aus Qdrant."""
    try:
        from kiara.config import load_config
        from kiara.retrieval.vectorstore import VectorStore

        config = load_config()

        def _progress(label: str, processed: int = 0, total: int = 0) -> None:
            with _lock:
                _task["progress"] = {"processed": processed, "total": total, "label": label}

        _progress("Lade Datenquellen...")
        ds_paths = _load_datasource_paths(config)
        if not ds_paths:
            with _lock:
                _task["result"] = {
                    "status": "error",
                    "action": "repair_reconstruct",
                    "error": "Keine aktiven Datenquellen mit Pfaden gefunden",
                }
            return

        logger.info("Reconstruct: %d Datenquellen mit Pfaden geladen", len(ds_paths))
        collections = VectorStore.list_collections(config)
        non_empty = [c for c in collections if c.get("chunks", 0) > 0]
        if not non_empty:
            with _lock:
                _task["result"] = {
                    "status": "error",
                    "action": "repair_reconstruct",
                    "error": "Keine Qdrant-Collections mit Chunks gefunden",
                }
            return

        scan_result = _scan_qdrant_assignments(config, ds_paths, non_empty, _progress)

        _progress(
            "Berechne Statistiken...", scan_result["processed_total"], scan_result["grand_total"]
        )
        _recalc_datasource_embeddings(config)

        _progress(
            "Aktualisiere ds_ids in Qdrant + ES...",
            scan_result["processed_total"],
            scan_result["grand_total"],
        )
        ds_ids_updated = _apply_ds_id_updates(
            config, non_empty, _progress, scan_result["processed_total"], scan_result["grand_total"]
        )

        with _lock:
            _task["result"] = {
                "status": "success",
                "action": "repair_reconstruct",
                "collections_scanned": len(non_empty),
                "chunks_scanned": scan_result["processed_total"],
                "chunks_matched": scan_result["total_inserted"],
                "chunks_unmatched": scan_result["unmatched"],
                "ds_ids_updated": ds_ids_updated,
                "message": f"Rekonstruktion abgeschlossen: {scan_result['total_inserted']:,} Zuordnungen "
                f"aus {scan_result['processed_total']:,} Chunks ({len(non_empty)} Collections). "
                f"{scan_result['unmatched']:,} ohne DS-Match, {ds_ids_updated:,} ds_ids aktualisiert.",
            }
    except Exception as exc:
        logger.exception("Metadaten-Rekonstruktion fehlgeschlagen")
        with _lock:
            _task["result"] = {"status": "error", "action": "repair_reconstruct", "error": str(exc)}
    finally:
        with _lock:
            _task["running"] = False
            _task["action"] = None


# ---------------------------------------------------------------------------
# Repair: Collection-Rebuild (Snapshot → Drop → Restore)
# ---------------------------------------------------------------------------


def _set_result(status, action, **kwargs):
    with _lock:
        _task["result"] = {"status": status, "action": action, **kwargs}


def _clear_running():
    with _lock:
        _task["running"] = False
        _task["action"] = None


def _find_fragmented_collections(config, client) -> list[dict]:
    """Ermittelt fragmentierte Qdrant-Collections (segments > 5 AND chunks > 10000)."""
    from kiara.retrieval.vectorstore import VectorStore

    collections = VectorStore.list_collections(config)
    targets = []
    for coll in collections:
        if coll.get("chunks", 0) <= 10000:
            continue
        info = client.get_collection(coll["name"])
        if (info.segments_count or 0) > 5:
            targets.append(
                {
                    "name": coll["name"],
                    "chunks": coll["chunks"],
                    "segments_before": info.segments_count,
                }
            )
    return targets


def _rebuild_collection(client, target, created_snapshots, progress_fn) -> dict:
    """Rebuilt eine einzelne Collection via Snapshot → Drop → Restore."""
    import time

    from qdrant_client.models import SnapshotPriority

    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    name = target["name"]
    progress_fn(f"{name}: Snapshot erstellen...")

    snapshot_info = client.create_snapshot(collection_name=name)
    snapshot_name = snapshot_info.name
    snapshot_path = f"/var/lib/qdrant/snapshots/{name}/{snapshot_name}"
    created_snapshots.append({"collection": name, "snapshot": snapshot_name})

    if not os.path.isfile(snapshot_path):
        raise RuntimeError(
            f"Snapshot-Datei nicht gefunden: {snapshot_path}. Rebuild fuer {name} abgebrochen."
        )
    snap_size = os.path.getsize(snapshot_path)
    if snap_size == 0:
        raise RuntimeError(
            f"Snapshot-Datei ist leer (0 Bytes): {snapshot_path}. Rebuild fuer {name} abgebrochen."
        )
    progress_fn(f"{name}: Snapshot OK ({snap_size / 1024 / 1024:.0f} MB), Collection loeschen...")

    client.delete_collection(collection_name=name)
    progress_fn(f"{name}: Aus Snapshot wiederherstellen...")

    try:
        client.recover_snapshot(
            collection_name=name,
            location=f"file://{snapshot_path}",
            priority=SnapshotPriority.SNAPSHOT,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Snapshot-Restore fuer {name} fehlgeschlagen nach Collection-Loesch. "
            f"Snapshot erhalten: {snapshot_path} — manuelles Recovery moeglich."
        ) from exc

    restored_ok = False
    for _ in range(120):
        time.sleep(5)
        try:
            info = client.get_collection(name)
            opt = getattr(info.optimizer_status, "status", str(info.optimizer_status))
            if info.status.value == "green" and opt == "ok":
                restored_ok = True
                break
            progress_fn(
                f"{name}: Laden... ({info.points_count} Punkte, Status: {info.status.value})"
            )
        except Exception:
            pass

    if not restored_ok:
        raise RuntimeError(
            f"Collection {name} nach 10 Min nicht 'green'. "
            f"Snapshot erhalten: {snapshot_path} — manuelles Recovery moeglich."
        )

    progress_fn(f"{name}: Payload-Indexes erstellen...")
    for field, schema in QdrantVectorStore._PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )
        except Exception as exc:
            logger.warning("Payload-Index %s fuer %s fehlgeschlagen: %s", field, name, exc)

    info_after = client.get_collection(name)
    return {
        "name": name,
        "chunks": info_after.points_count,
        "segments_before": target["segments_before"],
        "segments_after": info_after.segments_count,
    }


def _run_repair_rebuild() -> None:
    """Background-Thread: Rebuilt fragmentierte Qdrant-Collections via Snapshot/Restore."""

    def _progress(label: str) -> None:
        with _lock:
            _task["progress"] = {"processed": 0, "total": 0, "label": label}

    created_snapshots = []
    client = None

    try:
        from qdrant_client import QdrantClient

        from kiara.config import load_config
        from kiara.shared.constants import DEFAULT_QDRANT_URL

        config = load_config()
        qdrant_cfg = config.get("vectordb", {})
        client = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
            timeout=600,
        )

        targets = _find_fragmented_collections(config, client)
        if not targets:
            _set_result(
                "success", "repair_rebuild", message="Keine fragmentierten Collections gefunden"
            )
            return

        results = [_rebuild_collection(client, t, created_snapshots, _progress) for t in targets]

        for snap in created_snapshots:
            try:
                client.delete_snapshot(
                    collection_name=snap["collection"],
                    snapshot_name=snap["snapshot"],
                )
            except Exception as exc:
                logger.warning(
                    "Snapshot-Loesch fehlgeschlagen (%s/%s): %s",
                    snap["collection"],
                    snap["snapshot"],
                    exc,
                )

        parts = [
            f"{r['name']}: {r['segments_before']}→{r['segments_after']} Segmente" for r in results
        ]
        _set_result(
            "success", "repair_rebuild", rebuilt=results, message="Rebuild: " + "; ".join(parts)
        )

    except Exception as exc:
        logger.exception("Collection-Rebuild fehlgeschlagen")
        snap_info = [f"{s['collection']}/{s['snapshot']}" for s in created_snapshots]
        if snap_info:
            logger.warning("Snapshots erhalten fuer manuelle Recovery: %s", ", ".join(snap_info))
        _set_result("error", "repair_rebuild", error=str(exc))
    finally:
        _clear_running()


def _run_repair_cleanup_colbert() -> None:
    """Background-Thread: Stale ColBERT-Collections/DB-Anker bereinigen."""
    try:
        from kiara.config import load_config
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.ingestion.pipeline import (
            cleanup_stale_colbert_collections_without_db,
            cleanup_stale_colbert_models,
        )
        from kiara.integrity.checks import check_stale_colbert_collections
        from kiara.retrieval.colbert import (
            colbert_ds_model_key,
            model_name_from_colbert_ds_key,
        )
        from kiara.shared.db import db_query

        config = load_config()
        chunk_ds = ChunkDatasources()

        def _progress(label: str, processed: int = 0, total: int = 0) -> None:
            with _lock:
                _task["progress"] = {
                    "processed": processed,
                    "total": total,
                    "label": label,
                }

        _progress("ColBERT-Cleanup: Detection...")
        detection = check_stale_colbert_collections(config)
        stale_collections = detection.get("stale_collections", [])
        orphaned_db_entries = detection.get("orphaned_db_entries", [])
        stale_set = {c["model"] for c in stale_collections} | {
            e["model"] for e in orphaned_db_entries
        }
        if detection.get("status") == "warning" and not stale_set:
            _set_result(
                "error",
                "repair_cleanup_colbert",
                skipped=True,
                error=(
                    "Cleanup uebersprungen: Detection ist warning und "
                    "keine stale Modelle sind sicher klassifiziert."
                ),
            )
            return

        allow_current_models = {
            e["model"] for e in orphaned_db_entries if e.get("reason") == "collection_missing"
        }
        collection_only_stale = [c for c in stale_collections if not c.get("datasources")]

        db_colbert_keys = [
            m for m in chunk_ds.distinct_models() if m and str(m).startswith("colbert::")
        ]
        raw_models = []
        for key in db_colbert_keys:
            try:
                raw_models.append(model_name_from_colbert_ds_key(key))
            except ValueError:
                logger.warning("Ungueltiger ColBERT-DS-Key uebersprungen: %r", key)
        stale_models = sorted({m for m in raw_models if m in stale_set})

        stale_ds_keys = [colbert_ds_model_key(m) for m in stale_models]
        if not stale_ds_keys:
            affected_ds_ids = []
        else:
            with db_query() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT datasource_id FROM chunk_datasources "
                    "WHERE embedding_model = ANY(%s)",
                    (stale_ds_keys,),
                ).fetchall()
            affected_ds_ids = [r["datasource_id"] for r in rows]

        total_steps = len(affected_ds_ids) + (1 if collection_only_stale else 0)
        _progress("ColBERT-Cleanup: Collection-only...", 0, total_steps)
        collection_only = cleanup_stale_colbert_collections_without_db(
            config, collection_only_stale
        )

        invoked = 0
        lock_skips = 0
        current_model_skips = 0
        db_only_current_cleanups = 0
        invalid_skips = 0
        wrapper_errors = 0
        for i, ds_id in enumerate(affected_ds_ids, start=1):
            _progress(f"ColBERT-Cleanup: DS {ds_id}...", i, total_steps)
            result = cleanup_stale_colbert_models(
                config,
                ds_id,
                chunk_ds,
                stale_models,
                allow_current_models=allow_current_models,
            )
            if result.get("lock_conflict"):
                lock_skips += 1
            if result.get("cleanup_invoked"):
                invoked += 1
            current_model_skips += len(result.get("skipped_current_models", []))
            db_only_current_cleanups += len(result.get("db_only_current_models", []))
            invalid_skips += len(result.get("skipped_invalid_models", []))
            wrapper_errors += len(result.get("errors", []))

        _set_result(
            "success",
            "repair_cleanup_colbert",
            datasources_attempted=len(affected_ds_ids),
            cleanup_invoked=invoked,
            lock_skips=lock_skips,
            current_model_skips=current_model_skips,
            db_only_current_cleanups=db_only_current_cleanups,
            invalid_model_skips=invalid_skips,
            wrapper_errors=wrapper_errors,
            collection_only_deleted=len(collection_only.get("deleted", [])),
            collection_only_skipped_with_datasources=len(
                collection_only.get("skipped_with_datasources", [])
            ),
            collection_only_errors=len(collection_only.get("errors", [])),
            message=(
                f"ColBERT-Cleanup: {invoked} Datasource(s), "
                f"Collection-only geloescht: "
                f"{len(collection_only.get('deleted', []))}, "
                f"Fehler: {wrapper_errors + len(collection_only.get('errors', []))}"
            ),
        )
    except Exception as exc:
        logger.exception("ColBERT-Cleanup-Repair fehlgeschlagen")
        _set_result("error", "repair_cleanup_colbert", error=str(exc))
    finally:
        _clear_running()
