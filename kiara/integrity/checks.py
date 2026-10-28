"""
Integritaetspruefungen fuer die KIara-Datenschichten.

Reine Funktionen ohne State. Jede Funktion bekommt config: dict
und gibt ein typisiertes Dict zurueck.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def check_layer_counts(config: dict) -> dict:
    """Zaehlt Chunks pro Datenschicht und vergleicht.

    Beruecksichtigt ALLE Qdrant-Collections (Multi-Modell-Betrieb),
    nicht nur die aktuell konfigurierte.

    Returns:
        Dict mit qdrant_total, collections, es_count, es_configured,
        chunk_ds_count, chunk_ds_total_rows, file_hashes_count,
        ds_embeddings_total, qdrant_empty, status, details.
    """
    from kiara.retrieval.vectorstore_factory import list_all_retrieval_collections
    from kiara.shared.db import db_query

    # 1. Qdrant — ALLE Retrieval-Collections zaehlen (kb_* + colbert_*)
    # Erreichbarkeit explizit pruefen, damit Verbindungsfehler nicht als
    # "Collections leer" fehlinterpretiert werden
    qdrant_reachable = True
    try:
        from qdrant_client import QdrantClient

        qdrant_cfg = config.get("vectordb", {})
        _qclient = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", "http://localhost:6333"),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
            timeout=10,
        )
        _qclient.get_collections()  # Wirft Exception bei Verbindungsfehler
    except Exception as exc:
        qdrant_reachable = False
        logger.warning("Qdrant nicht erreichbar: %s", exc)

    collections = list_all_retrieval_collections(config) if qdrant_reachable else []
    qdrant_total = sum(c.get("chunks", 0) for c in collections)
    qdrant_dense_total = sum(c.get("chunks", 0) for c in collections if c.get("kind") == "kb")
    qdrant_colbert_total = sum(
        c.get("chunks", 0) for c in collections if c.get("kind") == "colbert"
    )

    # 2. Elasticsearch (optional)
    from kiara.retrieval.elasticsearch_service import get_status, is_configured

    es_configured = is_configured(config)
    es_status = get_status(config) if es_configured else {}
    es_count = es_status.get("doc_count")

    # 3. chunk_datasources (mit Dense/ColBERT-Split + Keyspace-Distinct)
    with db_query() as conn:
        row = conn.execute(
            "SELECT "
            "  COUNT(DISTINCT chunk_id::text) AS distinct_chunks, "
            "  COUNT(DISTINCT (COALESCE(embedding_model, ''), chunk_id::text)) "
            "    AS chunk_ds_keyspace_distinct_chunks, "
            "  COUNT(*) AS total_rows, "
            "  COUNT(DISTINCT chunk_id::text) FILTER ("
            "    WHERE COALESCE(embedding_model, '') NOT LIKE 'colbert::%%'"
            "  ) AS dense_chunk_ds_distinct_chunks, "
            "  COUNT(*) FILTER ("
            "    WHERE COALESCE(embedding_model, '') NOT LIKE 'colbert::%%'"
            "  ) AS dense_chunk_ds_total_rows, "
            "  COUNT(DISTINCT chunk_id::text) FILTER ("
            "    WHERE embedding_model LIKE 'colbert::%%'"
            "  ) AS colbert_chunk_ds_distinct_chunks, "
            "  COUNT(*) FILTER ("
            "    WHERE embedding_model LIKE 'colbert::%%'"
            "  ) AS colbert_chunk_ds_total_rows "
            "FROM chunk_datasources"
        ).fetchone()
    chunk_ds_count = row["distinct_chunks"] if row else 0
    chunk_ds_keyspace_distinct_chunks = row["chunk_ds_keyspace_distinct_chunks"] if row else 0
    chunk_ds_total_rows = row["total_rows"] if row else 0
    dense_chunk_ds_total_rows = row["dense_chunk_ds_total_rows"] if row else 0
    colbert_chunk_ds_total_rows = row["colbert_chunk_ds_total_rows"] if row else 0

    # 4. file_hashes
    with db_query() as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM file_hashes").fetchone()
    file_hashes_count = row["cnt"] if row else 0

    # 5. datasource_embeddings (mit Dense/ColBERT-Split)
    with db_query() as conn:
        row = conn.execute(
            "SELECT "
            "  COALESCE(SUM(chunk_count), 0) AS total, "
            "  COALESCE(SUM(chunk_count) FILTER ("
            "    WHERE COALESCE(embedding_model, '') NOT LIKE 'colbert::%%'"
            "  ), 0) AS dense_total, "
            "  COALESCE(SUM(chunk_count) FILTER ("
            "    WHERE embedding_model LIKE 'colbert::%%'"
            "  ), 0) AS colbert_total "
            "FROM datasource_embeddings"
        ).fetchone()
    ds_embeddings_total = row["total"] if row else 0
    dense_ds_embeddings_total = row["dense_total"] if row else 0
    colbert_ds_embeddings_total = row["colbert_total"] if row else 0

    # 6. Bewertung
    meta_total = chunk_ds_count + file_hashes_count + ds_embeddings_total
    status = "ok"
    qdrant_empty = False
    details_parts = []

    if not qdrant_reachable:
        # Verbindungsfehler ≠ leere Collections
        status = "warning"
        details_parts.append("Qdrant nicht erreichbar — Schicht-Vergleich nicht moeglich")
    elif qdrant_total == 0 and meta_total > 0:
        status = "critical"
        qdrant_empty = True
        details_parts.append(
            f"Alle Qdrant-Collections leer — {chunk_ds_count:,} verwaiste "
            f"Metadaten in chunk_datasources, {file_hashes_count:,} in file_hashes. "
            f"Metadaten-Reset oder Neu-Indexierung erforderlich"
        )
    elif qdrant_total != chunk_ds_keyspace_distinct_chunks:
        status = "warning"
        diff = qdrant_total - chunk_ds_keyspace_distinct_chunks
        details_parts.append(
            f"Qdrant ({qdrant_total:,}) und chunk_datasources-Keyspace "
            f"({chunk_ds_keyspace_distinct_chunks:,} Modell/Chunk-Paare) "
            f"weichen um {abs(diff):,} Chunks ab"
        )
    if es_configured and (es_status.get("error") or not es_status.get("counts_match")):
        if status != "critical":
            status = "warning"
        details_parts.append(es_status.get("error") or "ES-/PG-Quellenmengen weichen ab")
    if not details_parts:
        details_parts.append("Alle Schichten konsistent")

    return {
        "qdrant_total": qdrant_total,
        "qdrant_dense_total": qdrant_dense_total,
        "qdrant_colbert_total": qdrant_colbert_total,
        "collections": collections,
        "es_count": es_count,
        "es_configured": es_configured,
        "es_inventory": es_status,
        "chunk_ds_count": chunk_ds_count,
        "chunk_ds_keyspace_distinct_chunks": chunk_ds_keyspace_distinct_chunks,
        "chunk_ds_total_rows": chunk_ds_total_rows,
        "dense_chunk_ds_total_rows": dense_chunk_ds_total_rows,
        "colbert_chunk_ds_total_rows": colbert_chunk_ds_total_rows,
        "file_hashes_count": file_hashes_count,
        "ds_embeddings_total": ds_embeddings_total,
        "dense_ds_embeddings_total": dense_ds_embeddings_total,
        "colbert_ds_embeddings_total": colbert_ds_embeddings_total,
        "qdrant_empty": qdrant_empty,
        "status": status,
        "details": "; ".join(details_parts),
    }


def _load_ds_data(config: dict) -> dict:
    """Laedt DS-Rows + Counts aus den 3 Datenschichten."""
    from kiara.shared.db import db_query

    with db_query() as conn:
        ds_rows = conn.execute(
            "SELECT id, name FROM datasources WHERE is_active = true ORDER BY id"
        ).fetchall()

    with db_query() as conn:
        count_rows = conn.execute(
            "SELECT datasource_id, embedding_model, COUNT(*) AS cnt "
            "FROM chunk_datasources GROUP BY datasource_id, embedding_model"
        ).fetchall()
    chunk_counts = {(r["datasource_id"], r["embedding_model"]): r["cnt"] for r in count_rows}

    with db_query() as conn:
        de_rows = conn.execute(
            "SELECT datasource_id, embedding_model, "
            "COALESCE(chunk_count, 0) AS chunk_count, "
            "COALESCE(file_count, 0) AS file_count "
            "FROM datasource_embeddings"
        ).fetchall()
    de_stats = {
        (r["datasource_id"], r["embedding_model"]): {
            "chunks": r["chunk_count"],
            "files": r["file_count"],
        }
        for r in de_rows
    }

    with db_query() as conn:
        fh_rows = conn.execute(
            "SELECT datasource_id, embedding_model, COUNT(*) AS cnt "
            "FROM file_hashes GROUP BY datasource_id, embedding_model"
        ).fetchall()
    fh_counts = {(r["datasource_id"], r["embedding_model"]): r["cnt"] for r in fh_rows}

    all_models = sorted(
        set(m for _, m in chunk_counts) | set(m for _, m in de_stats) | set(m for _, m in fh_counts)
    )

    return {
        "ds_rows": ds_rows,
        "chunk_counts": chunk_counts,
        "de_stats": de_stats,
        "fh_counts": fh_counts,
        "all_models": all_models,
    }


def _check_qdrant_coverage(ds_data: dict) -> tuple[list, int]:
    """Vergleicht Qdrant- vs. DB-Counts pro DS + Modell."""
    ds_rows = ds_data["ds_rows"]
    chunk_counts = ds_data["chunk_counts"]
    de_stats = ds_data["de_stats"]
    fh_counts = ds_data["fh_counts"]
    all_models = ds_data["all_models"]

    ds_warnings = 0
    datasources = []
    for ds in ds_rows:
        ds_id = ds["id"]
        for model in all_models:
            is_colbert = str(model).startswith("colbert::")
            actual = chunk_counts.get((ds_id, model), 0)
            de_info = de_stats.get((ds_id, model), {"chunks": 0, "files": 0})
            expected = de_info["chunks"]
            file_count = de_info["files"]
            fh_count = fh_counts.get((ds_id, model), 0)

            if actual == 0 and expected == 0 and fh_count == 0:
                continue

            if is_colbert:
                datasources.append(
                    {
                        "id": ds_id,
                        "name": ds["name"],
                        "embedding_model": model,
                        "kind": "colbert",
                        "chunk_ds_count": actual,
                        "de_chunk_count": None,
                        "de_file_count": None,
                        "fh_count": None,
                        "status": "ok",
                        "issues": [],
                    }
                )
                continue

            issues = []
            if actual != expected:
                diff = actual - expected
                issues.append(
                    f"chunk_datasources ({actual:,}) vs datasource_embeddings ({expected:,}): "
                    f"Differenz {diff:+,}"
                )
            if actual == 0 and expected == 0 and fh_count > 0:
                issues.append(f"{fh_count:,} file_hashes ohne Chunks")
            if file_count > 0 and fh_count != file_count:
                issues.append(
                    f"file_hashes ({fh_count:,}) vs datasource_embeddings.file_count ({file_count:,})"
                )

            ds_status = "warning" if issues else "ok"
            if issues:
                ds_warnings += 1

            datasources.append(
                {
                    "id": ds_id,
                    "name": ds["name"],
                    "embedding_model": model,
                    "kind": "kb",
                    "chunk_ds_count": actual,
                    "de_chunk_count": expected,
                    "de_file_count": file_count,
                    "fh_count": fh_count,
                    "status": ds_status,
                    "issues": issues,
                }
            )

    return datasources, ds_warnings


def _check_colbert_coverage(config: dict, ds_data: dict) -> tuple[list, int]:
    """Prueft ob DS mit aktiviertem ColBERT auch ColBERT-Chunks haben."""
    from kiara.shared.db import db_query

    chunk_counts = ds_data["chunk_counts"]
    de_stats = ds_data["de_stats"]
    fh_counts = ds_data["fh_counts"]

    colbert_global_enabled = bool(
        config.get("retrieval", {}).get("colbert", {}).get("enabled", False)
    )
    if not colbert_global_enabled:
        return [], 0

    dense_or_file_backed_ds_ids = (
        {did for (did, model) in chunk_counts if not str(model).startswith("colbert::")}
        | {did for (did, model) in de_stats if not str(model).startswith("colbert::")}
        | {did for (did, model) in fh_counts if not str(model).startswith("colbert::")}
    )

    with db_query() as conn:
        colbert_ds_rows = conn.execute(
            "SELECT id, name FROM datasources "
            "WHERE is_active = true AND ds_colbert_enabled = true "
            "ORDER BY id"
        ).fetchall()

    colbert_warnings = []
    for ds in colbert_ds_rows:
        if ds["id"] not in dense_or_file_backed_ds_ids:
            continue
        has_colbert = any(
            did == ds["id"] and str(model).startswith("colbert::") for (did, model) in chunk_counts
        )
        if not has_colbert:
            colbert_warnings.append({"id": ds["id"], "name": ds["name"]})

    return colbert_warnings, len(colbert_warnings)


def _check_es_coverage(config: dict) -> tuple[int, int]:
    from kiara.retrieval.elasticsearch_service import check_missing_ds_ids

    result = check_missing_ds_ids(config)
    if not result["safe_to_proceed"]:
        raise RuntimeError(result["error"] or "ES-Zuordnungen unvollstaendig")
    return result["missing_ds_ids"], result["total_docs"]


def check_ds_coverage(config: dict) -> dict:
    """Prueft DS-Zuordnung (chunk_datasources) und ES ds_ids-Abdeckung.

    Vergleicht pro Datenquelle UND Embedding-Modell die Chunk-Counts in drei Schichten:
    - chunk_datasources (Source of Truth)
    - datasource_embeddings (Soll-Statistik aus Ingestion)
    - file_hashes (indexierte Dateien)

    Returns:
        Dict mit datasources (Liste mit Vergleichsdaten), es_without_ds_ids,
        es_total, status ("ok"|"warning"|"critical").
    """
    ds_data = _load_ds_data(config)
    datasources, ds_warnings = _check_qdrant_coverage(ds_data)
    colbert_warnings, colbert_warning_count = _check_colbert_coverage(config, ds_data)
    es_without, es_total = _check_es_coverage(config)

    status = "ok"
    if es_without > 0 and es_total > 0:
        pct = es_without / es_total * 100
        if pct > 50:
            status = "critical"
        elif pct > 10:
            status = "warning"
    if ds_warnings > 0 and status == "ok":
        status = "warning"
    if colbert_warning_count > 0 and status == "ok":
        status = "warning"

    return {
        "datasources": datasources,
        "ds_warnings": ds_warnings,
        "colbert_warnings": colbert_warnings,
        "colbert_warning_count": colbert_warning_count,
        "es_without_ds_ids": es_without,
        "es_total": es_total,
        "status": status,
    }


def check_es_duplicates(config: dict, top_n: int = 20) -> dict:
    """Findet Dateien mit unverhaeltnismaessig vielen ES-Eintraegen.

    Returns:
        Dict mit es_configured, top_sources, total_unique_sources,
        total_docs, suspected_duplicates, status.
    """
    _empty = {
        "es_configured": False,
        "top_sources": [],
        "status": "ok",
        "total_unique_sources": 0,
        "total_docs": 0,
        "suspected_duplicates": 0,
    }
    try:
        from kiara.retrieval.elasticsearch_service import (
            _get_client,
            is_configured,
            search_explicit_targets,
            search_scope,
        )
    except ImportError:
        return _empty

    if not is_configured(config):
        return _empty

    total_docs = 0
    unique_sources = 0
    top_sources = []
    suspected = 0
    with search_scope(config, None) as targets:
        for target in targets:
            # Preserve the existing large-file diagnostic within a source index.
            # Adding regular copies from another DS must not invent a deviation.
            resp = search_explicit_targets(
                _get_client(config),
                (target,),
                {
                    "size": 0,
                    "track_total_hits": True,
                    "aggs": {
                        "sources": {
                            "terms": {
                                "field": "metadata.source.raw",
                                "size": top_n,
                                "order": {"_count": "desc"},
                            }
                        },
                        "unique_sources": {"cardinality": {"field": "metadata.source.raw"}},
                    },
                },
            )
            total_docs += resp["hits"]["total"]["value"]
            unique_sources += resp["aggregations"]["unique_sources"]["value"]
            for bucket in resp["aggregations"]["sources"]["buckets"]:
                count = bucket["doc_count"]
                top_sources.append(
                    {"datasource_id": target.datasource_id, "source": bucket["key"], "count": count}
                )
                if count > 5000:
                    suspected += 1
    top_sources = sorted(top_sources, key=lambda row: row["count"], reverse=True)[:top_n]

    status = "ok"
    if suspected > 5:
        status = "critical"
    elif suspected > 0:
        status = "warning"

    return {
        "es_configured": True,
        "top_sources": top_sources,
        "total_unique_sources": unique_sources,
        "source_count_unit": "approximate_datasource_source_pairs",
        "total_docs": total_docs,
        "suspected_duplicates": suspected,
        "status": status,
    }


def check_orphan_chunks(config: dict, sample_size: int = 10000) -> dict:
    """Prueft auf verwaiste Chunks (in Qdrant aber ohne DS-Zuordnung).

    Sampling-basiert fuer Performance bei grossen Collections.
    Iteriert ueber ALLE nicht-leeren Qdrant-Collections (Multi-Modell).
    Das sample_size-Budget wird proportional nach Chunk-Anteil verteilt
    (min. 100 pro Collection fuer statistische Signifikanz).

    Returns:
        Dict mit sampled, orphans_found, orphan_pct, sample_orphans, status.
    """
    from kiara.retrieval.vectorstore_factory import (
        create_retrieval_store,
        list_all_retrieval_collections,
    )
    from kiara.shared.db import db_query

    collections = list_all_retrieval_collections(config)
    non_empty = [c for c in collections if c.get("chunks", 0) > 0]

    if not non_empty:
        return {
            "sampled": 0,
            "orphans_found": 0,
            "orphan_pct": 0.0,
            "sample_orphans": [],
            "status": "ok",
        }

    grand_total = sum(c["chunks"] for c in non_empty)
    all_sample_ids = []
    all_sample_meta = {}

    for coll in non_empty:
        # Budget proportional verteilen (min. 100 pro Collection)
        coll_budget = max(100, int(sample_size * coll["chunks"] / grand_total))
        store = create_retrieval_store(config, kind=coll["kind"], model=coll["model"])

        offset = None
        batch_limit = min(coll_budget, 1000)
        collected = 0
        while collected < coll_budget:
            results, next_offset = store._client.scroll(
                collection_name=store._collection,
                limit=batch_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not results:
                break
            for point in results:
                cid = str(point.id).replace("-", "")
                all_sample_ids.append(cid)
                all_sample_meta[cid] = point.payload or {}
                collected += 1
                if collected >= coll_budget:
                    break
            offset = next_offset
            if offset is None:
                break

    # Design-Entscheidung: KEIN embedding_model-Filter in der Query.
    # chunk_id = MD5(source::parser::chunk_index) ist NICHT modell-spezifisch —
    # derselbe chunk_id existiert in mehreren Collections, wenn ein Dokument mit
    # mehreren Modellen indexiert wurde. Der Orphan-Check prueft daher bewusst
    # nur, ob ein chunk_id UEBERHAUPT in chunk_datasources vorkommt (egal fuer
    # welches Modell). Dies erkennt zuverlaessig Chunks, die komplett verwaist
    # sind. Modell-spezifische Inkonsistenzen (Chunk in Qdrant fuer Modell A,
    # aber nur chunk_datasources-Eintrag fuer Modell B) werden durch den
    # ds_coverage-Check abgedeckt.

    if not all_sample_ids:
        return {
            "sampled": 0,
            "orphans_found": 0,
            "orphan_pct": 0.0,
            "sample_orphans": [],
            "status": "ok",
        }

    # Batch-Lookup in chunk_datasources
    known_ids = set()
    batch_size = 500
    for i in range(0, len(all_sample_ids), batch_size):
        batch = all_sample_ids[i : i + batch_size]
        ph = ",".join(["%s"] * len(batch))
        with db_query() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT chunk_id FROM chunk_datasources WHERE chunk_id IN ({ph})",
                batch,
            ).fetchall()
            known_ids.update(r["chunk_id"] for r in rows)

    orphans_found = 0
    sample_orphans = []
    for cid in all_sample_ids:
        if cid not in known_ids:
            orphans_found += 1
            if len(sample_orphans) < 20:
                meta = all_sample_meta.get(cid, {})
                sample_orphans.append(
                    {
                        "chunk_id": cid,
                        "source": meta.get("source", "unbekannt"),
                    }
                )

    sampled = len(all_sample_ids)
    orphan_pct = round(orphans_found / sampled * 100, 1) if sampled > 0 else 0.0
    status = "warning" if orphan_pct > 5 else "ok"

    return {
        "sampled": sampled,
        "orphans_found": orphans_found,
        "orphan_pct": orphan_pct,
        "sample_orphans": sample_orphans,
        "status": status,
    }


def check_overlapping_datasources(config: dict) -> dict:
    """Erkennt Datenquellen mit ueberlappenden Pfaden (Unterverzeichnis-Beziehung).

    Returns:
        Dict mit overlaps (Liste) und status ("ok"|"warning").
    """
    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector
    from kiara.shared.db import db_query

    with db_query() as conn:
        rows = conn.execute(
            "SELECT id, name, type, connector_config "
            "FROM datasources WHERE is_active = true ORDER BY id"
        ).fetchall()

    ensure_loaded()

    def _connector_config(ds):
        keys = ds.keys() if hasattr(ds, "keys") else ()
        raw = ds["connector_config"] if "connector_config" in keys else "{}"
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            loaded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _effective_path(ds):
        connector_cls = get_connector(ds["type"])
        if not connector_cls:
            return None
        try:
            path = connector_cls.effective_path(_connector_config(ds), dict(ds))
        except Exception as exc:
            logger.warning(
                "Effective-Path fuer DS %s (%s) konnte nicht berechnet werden: %s",
                ds.get("id") if hasattr(ds, "get") else "?",
                ds.get("type") if hasattr(ds, "get") else "?",
                exc,
            )
            return None
        return os.path.normpath(path) if path else None

    ds_list = []
    for r in rows:
        p = _effective_path(r)
        if p:
            ds_list.append({"id": r["id"], "name": r["name"], "path": p})

    overlaps = []
    for i, a in enumerate(ds_list):
        for b in ds_list[i + 1 :]:
            pa, pb = a["path"], b["path"]
            if pa == pb:
                overlaps.append(
                    {
                        "ds_a": a,
                        "ds_b": b,
                        "relationship": "identical",
                    }
                )
            elif pb.startswith(pa + os.sep):
                overlaps.append(
                    {
                        "ds_a": a,
                        "ds_b": b,
                        "relationship": "parent",
                    }
                )
            elif pa.startswith(pb + os.sep):
                overlaps.append(
                    {
                        "ds_a": b,
                        "ds_b": a,
                        "relationship": "parent",
                    }
                )

    return {
        "overlaps": overlaps,
        "status": "warning" if overlaps else "ok",
    }


def check_qdrant_fragmentation(config: dict) -> dict:
    """Prueft Segment-Fragmentierung der Qdrant-Collections.

    Liest Segment-Anzahl und Speicherverbrauch pro Collection.
    Hohe Segment-Zahlen deuten auf Fragmentierung hin (z.B. nach
    massiven Payload-Updates), die durch Optimierung reduziert werden kann.

    Returns:
        Dict mit collections (Liste mit Segment-Details), status.
    """
    from kiara.retrieval.vectorstore import VectorStore

    collections_raw = VectorStore.list_collections(config)
    if not collections_raw:
        return {"collections": [], "status": "ok", "total_segments": 0}

    # Qdrant-Client fuer Detail-Abfrage
    try:
        from qdrant_client import QdrantClient

        qdrant_cfg = config.get("vectordb", {})
        client = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", "http://localhost:6333"),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
        )
    except Exception as exc:
        return {
            "collections": [],
            "status": "error",
            "error": f"Qdrant nicht erreichbar: {exc}",
            "total_segments": 0,
        }

    # Disk-Usage per Collection (falls Pfad existiert)
    qdrant_storage = "/var/lib/qdrant/storage/collections"
    collections = []
    total_segments = 0
    fragmented_count = 0

    for coll in collections_raw:
        name = coll["name"]
        chunks = coll.get("chunks", 0)

        # Segment-Count via API
        segments = 0
        optimizer_status = "unknown"
        try:
            info = client.get_collection(name)
            segments = info.segments_count or 0
            optimizer_status = (
                info.optimizer_status
                if isinstance(info.optimizer_status, str)
                else getattr(info.optimizer_status, "status", "ok")
            )
        except Exception:
            pass

        # Disk-Usage via Filesystem
        disk_mb = 0
        try:
            coll_path = os.path.join(qdrant_storage, name)
            if os.path.isdir(coll_path):
                total_bytes = 0
                for dirpath, _dirs, files in os.walk(coll_path):
                    for f in files:
                        total_bytes += os.path.getsize(os.path.join(dirpath, f))
                disk_mb = round(total_bytes / (1024 * 1024))
        except Exception:
            pass

        # Bewertung: >5 Segmente bei >10k Chunks = fragmentiert
        is_fragmented = segments > 5 and chunks > 10000
        if is_fragmented:
            fragmented_count += 1
        total_segments += segments

        collections.append(
            {
                "name": name,
                "model": coll.get("model", ""),
                "chunks": chunks,
                "segments": segments,
                "optimizer_status": optimizer_status,
                "disk_mb": disk_mb,
                "fragmented": is_fragmented,
            }
        )

    status = "ok"
    if fragmented_count > 0:
        status = "warning"

    return {
        "collections": collections,
        "total_segments": total_segments,
        "fragmented_count": fragmented_count,
        "status": status,
    }


def check_enrichment_prompt_consistency(config: dict) -> dict:
    """Prueft ob Enrichment-Chunks zum aktuellen Prompt-Hash passen.

    Fuer jede Datasource mit ds_contextual_retrieval=1:
    1. Aktuellen Prompt ermitteln (ds_enrichment_prompt_id oder Default)
    2. SHA256-Hash des aktuellen Prompt-Contents berechnen
    3. Alle Chunks der Qdrant-Collections scrollen, Python-seitig nach ds_ids filtern
    4. Pro Chunk klassifizieren: consistent / inconsistent / not_enriched
    5. Historische Hashes via prompt_template_history aufloesen

    Returns:
        Dict mit check, status, total_datasources_checked, total_chunks_checked,
        consistent, inconsistent, not_enriched, details (per-DS Aufschluesselung).
    """
    import hashlib

    from kiara.retrieval.vectorstore import VectorStore
    from kiara.shared.db import db_query

    # 1. Datasources mit Contextual Retrieval laden
    with db_query() as conn:
        ds_rows = conn.execute(
            "SELECT id, name, ds_enrichment_prompt_id "
            "FROM datasources "
            "WHERE is_active = true AND ds_contextual_retrieval = 1 "
            "ORDER BY id"
        ).fetchall()

    if not ds_rows:
        return {
            "check": "enrichment_prompt_consistency",
            "status": "ok",
            "total_datasources_checked": 0,
            "total_chunks_checked": 0,
            "consistent": 0,
            "inconsistent": 0,
            "not_enriched": 0,
            "details": [],
        }

    # 2. Aktuellen Prompt-Hash pro DS ermitteln (Batch-Load statt N+1)
    prompt_ids = {
        ds["ds_enrichment_prompt_id"] for ds in ds_rows if ds["ds_enrichment_prompt_id"] is not None
    }
    prompt_cache: dict[int, str] = {}
    default_content: str | None = None
    try:
        with db_query() as conn:
            if prompt_ids:
                ids_list = list(prompt_ids)
                ph = ",".join(["%s"] * len(ids_list))
                rows = conn.execute(
                    f"SELECT id, content FROM prompt_templates "
                    f"WHERE id IN ({ph}) AND is_active = true",
                    ids_list,
                ).fetchall()
                prompt_cache = {r["id"]: r["content"] for r in rows}

            default_row = conn.execute(
                "SELECT content FROM prompt_templates "
                "WHERE name = 'enrichment_default' AND is_active = true",
            ).fetchone()
            default_content = default_row["content"] if default_row else None
    except Exception as exc:
        logger.warning("Prompt-Templates nicht ladbar: %s", exc)

    ds_prompt_hashes = {}
    for ds in ds_rows:
        ds_id = ds["id"]
        prompt_id = ds["ds_enrichment_prompt_id"]
        content = prompt_cache.get(prompt_id) if prompt_id is not None else default_content
        if content:
            ds_prompt_hashes[ds_id] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        else:
            ds_prompt_hashes[ds_id] = None

    # 3. Bekannte historische Hashes laden
    known_hashes = set()
    try:
        with db_query() as conn:
            hash_rows = conn.execute(
                "SELECT DISTINCT content_hash FROM prompt_template_history"
            ).fetchall()
        known_hashes = {r["content_hash"] for r in hash_rows}
    except Exception as exc:
        logger.warning("prompt_template_history nicht lesbar: %s", exc)
    # Aktuelle Hashes ebenfalls als bekannt einstufen
    for h in ds_prompt_hashes.values():
        if h:
            known_hashes.add(h)

    # 4. Alle Collections scrollen und Chunks pro DS klassifizieren
    ds_ids_set = {ds["id"] for ds in ds_rows}
    ds_stats = {
        ds_id: {"consistent": 0, "inconsistent": 0, "not_enriched": 0, "found_hashes": set()}
        for ds_id in ds_ids_set
    }

    collections = VectorStore.list_collections(config)
    non_empty = [c for c in collections if c.get("chunks", 0) > 0]

    total_chunks_checked = 0
    for coll in non_empty:
        try:
            store = VectorStore(config, embedding_model=coll["model"], intent="maintenance")
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            logger.warning("Collection %s nicht oeffenbar: %s", coll["name"], exc)
            continue

        try:
            for batch in store.scroll_all():
                for _chunk_id, meta in batch:
                    # Python-seitig auf relevante DS filtern
                    chunk_ds_ids = meta.get("ds_ids", [])
                    if not isinstance(chunk_ds_ids, list):
                        chunk_ds_ids = [chunk_ds_ids] if chunk_ds_ids else []

                    matching_ds = ds_ids_set.intersection(chunk_ds_ids)
                    if not matching_ds:
                        continue

                    total_chunks_checked += 1
                    has_prefix = meta.get("contextual_prefix", False)
                    chunk_hash = meta.get("enrichment_prompt_hash")

                    for ds_id in matching_ds:
                        stats = ds_stats[ds_id]
                        current_hash = ds_prompt_hashes.get(ds_id)

                        if has_prefix and chunk_hash:
                            stats["found_hashes"].add(chunk_hash)
                            if current_hash and chunk_hash == current_hash:
                                stats["consistent"] += 1
                            else:
                                stats["inconsistent"] += 1
                        else:
                            stats["not_enriched"] += 1
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            logger.warning("Collection %s Scroll-Fehler: %s", coll["name"], exc)

    # 5. Ergebnis zusammenbauen
    total_consistent = 0
    total_inconsistent = 0
    total_not_enriched = 0
    details = []

    for ds in ds_rows:
        ds_id = ds["id"]
        stats = ds_stats[ds_id]
        total_consistent += stats["consistent"]
        total_inconsistent += stats["inconsistent"]
        total_not_enriched += stats["not_enriched"]

        # Historische Hashes aufloesen
        found = sorted(stats["found_hashes"])
        current_hash = ds_prompt_hashes.get(ds_id)
        hash_info = []
        for h in found:
            if h == current_hash:
                hash_info.append({"hash": h[:12], "status": "current"})
            elif h in known_hashes:
                hash_info.append({"hash": h[:12], "status": "historical"})
            else:
                hash_info.append({"hash": h[:12], "status": "unknown"})

        details.append(
            {
                "datasource_id": ds_id,
                "datasource_name": ds["name"],
                "current_prompt_hash": current_hash[:12] if current_hash else None,
                "consistent": stats["consistent"],
                "inconsistent": stats["inconsistent"],
                "not_enriched": stats["not_enriched"],
                "found_hashes": hash_info,
            }
        )

    status = "warning" if total_inconsistent > 0 else "ok"

    return {
        "check": "enrichment_prompt_consistency",
        "status": status,
        "total_datasources_checked": len(ds_rows),
        "total_chunks_checked": total_chunks_checked,
        "consistent": total_consistent,
        "inconsistent": total_inconsistent,
        "not_enriched": total_not_enriched,
        "details": details,
    }


def check_stale_colbert_collections(config: dict) -> dict:
    """Erkennt stale ColBERT-Collections und verwaiste colbert::*-DB-Eintraege.

    Stale = Collection fuer ein anderes Modell als das aktuell konfigurierte.
    Orphaned = DB-Eintraege in chunk_datasources ohne zugehoerige Qdrant-Collection.

    Bei ColBERT global deaktiviert: ALLE existierenden ColBERT-Collections als
    stale melden (Reason ``disabled``).

    Returns:
        Dict mit status, details, stale_collections, orphaned_db_entries.
    """
    from kiara.backend_selector import normalize_model_name
    from kiara.retrieval.colbert import (
        colbert_ds_model_key,
        model_name_from_colbert_collection,
        model_name_from_colbert_ds_key,
    )
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL
    from kiara.shared.db import db_query

    # Schritt 0: Qdrant fail-closed pruefen
    try:
        from qdrant_client import QdrantClient

        from kiara.shared.constants import DEFAULT_QDRANT_URL

        qdrant_cfg = config.get("vectordb", {})
        qclient = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
            timeout=10,
        )
        all_collections = qclient.get_collections().collections
        collections = []
        for coll in all_collections:
            if not coll.name.startswith("colbert_"):
                continue
            try:
                raw_model = model_name_from_colbert_collection(coll.name)
            except ValueError:
                logger.warning("Ungueltiger ColBERT-Collection-Name uebersprungen: %s", coll.name)
                continue
            info = qclient.get_collection(coll.name)
            points_count = info.points_count
            if points_count is None:
                count_result = qclient.count(collection_name=coll.name, exact=True)
                points_count = count_result.count
            if points_count is None:
                raise RuntimeError(f"Qdrant-Count fuer {coll.name} ist unbekannt")
            collections.append(
                {
                    "kind": "colbert",
                    "name": coll.name,
                    "model": raw_model,
                    "chunks": int(points_count),
                }
            )
    except Exception as exc:
        return {
            "status": "warning",
            "details": f"Qdrant nicht erreichbar — Stale-ColBERT-Pruefung uebersprungen: {exc}",
            "stale_collections": [],
            "orphaned_db_entries": [],
        }

    # Schritt 1: Aktuelles ColBERT-Modell + enabled-Flag
    colbert_cfg = config.get("retrieval", {}).get("colbert", {})
    colbert_enabled = bool(colbert_cfg.get("enabled", False))
    current_model = normalize_model_name(colbert_cfg.get("model", COLBERT_DEFAULT_MODEL))

    # Schritt 2: Helper fuer DS-Listen pro colbert-Key
    def _datasources_for_colbert_key(ds_key: str) -> list[dict]:
        with db_query() as conn:
            ds_rows = conn.execute(
                "SELECT DISTINCT cd.datasource_id, ds.name "
                "FROM chunk_datasources cd "
                "LEFT JOIN datasources ds ON ds.id = cd.datasource_id "
                "WHERE cd.embedding_model = %s "
                "ORDER BY cd.datasource_id",
                (ds_key,),
            ).fetchall()
        return [{"id": r["datasource_id"], "name": r.get("name")} for r in ds_rows]

    # Schritt 3: Stale ColBERT-Collections erkennen
    stale_collections = []
    for c in collections:
        raw_model = c["model"]
        normalized = normalize_model_name(raw_model)
        if not colbert_enabled:
            reason = "disabled"
        elif normalized != current_model:
            reason = "model_mismatch"
        else:
            continue  # Aktuelles Modell, nicht stale
        ds_key = colbert_ds_model_key(raw_model)
        datasources = _datasources_for_colbert_key(ds_key)
        stale_collections.append(
            {
                "model": raw_model,
                "name": c["name"],
                "chunks": c["chunks"],
                "reason": reason,
                "datasources": datasources,
            }
        )

    # Schritt 4: Orphaned DB-Eintraege erkennen
    from kiara.ingestion._file_utils import ChunkDatasources

    chunk_ds = ChunkDatasources()
    db_colbert_keys = [
        m for m in chunk_ds.distinct_models() if m and str(m).startswith("colbert::")
    ]
    existing_colbert_models = {c["model"] for c in collections if c.get("kind") == "colbert"}
    orphaned_db_entries = []
    for key in db_colbert_keys:
        try:
            raw_model = model_name_from_colbert_ds_key(key)
        except ValueError:
            logger.warning("Ungueltiger ColBERT-DS-Key uebersprungen: %r", key)
            continue
        if not raw_model:
            logger.warning("Leerer ColBERT-DS-Key uebersprungen: %r", key)
            continue
        if raw_model not in existing_colbert_models:
            orphaned_db_entries.append(
                {
                    "model": raw_model,
                    "reason": "collection_missing",
                    "datasources": _datasources_for_colbert_key(key),
                }
            )

    # Schritt 5: Status + Details
    total_stale = len(stale_collections)
    total_orphaned = len(orphaned_db_entries)
    if total_stale == 0 and total_orphaned == 0:
        status = "ok"
        details = "Keine stale ColBERT-Collections oder verwaisten DB-Eintraege"
    else:
        status = "warning"
        parts = []
        if total_stale > 0:
            parts.append(f"{total_stale} stale Collection(s)")
        if total_orphaned > 0:
            parts.append(f"{total_orphaned} verwaiste(r) DB-Eintrag/Eintraege")
        details = ", ".join(parts)

    return {
        "status": status,
        "details": details,
        "stale_collections": stale_collections,
        "orphaned_db_entries": orphaned_db_entries,
    }


def run_full_check(config: dict, progress_callback=None) -> dict:
    """Fuehrt alle Integritaetschecks aus.

    Args:
        progress_callback: Optional callback(step: int, total: int, label: str).

    Returns:
        Dict mit checks, overall_status, summary.
    """
    checks = {}
    total_steps = 8
    check_defs = [
        ("layer_counts", "Schicht-Zaehlung", check_layer_counts),
        ("ds_coverage", "DS-Abdeckung", check_ds_coverage),
        ("es_duplicates", "ES-Duplikate", check_es_duplicates),
        ("orphan_chunks", "Verwaiste Chunks", check_orphan_chunks),
        ("overlapping_ds", "Ueberlappende DS-Pfade", check_overlapping_datasources),
        ("qdrant_fragmentation", "Qdrant-Fragmentierung", check_qdrant_fragmentation),
        (
            "enrichment_prompt_consistency",
            "Enrichment-Prompt-Konsistenz",
            check_enrichment_prompt_consistency,
        ),
        ("stale_colbert", "Stale ColBERT-Collections", check_stale_colbert_collections),
    ]

    for i, (key, label, fn) in enumerate(check_defs):
        if progress_callback:
            try:
                progress_callback(i, total_steps, label)
            except Exception as cb_exc:
                logger.warning("progress_callback fehlgeschlagen: %s", cb_exc)
        try:
            checks[key] = fn(config)
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            checks[key] = {"status": "error", "error": str(exc)}

    if progress_callback:
        progress_callback(total_steps, total_steps, "Fertig")

    # Overall-Status: schlechtester Einzel-Status
    statuses = [c.get("status", "ok") for c in checks.values()]
    if "critical" in statuses:
        overall = "critical"
    elif "error" in statuses or "warning" in statuses:
        overall = "warning"
    else:
        overall = "ok"

    # Zusammenfassung
    problems = []
    lc = checks.get("layer_counts", {})
    if lc.get("status") != "ok":
        problems.append(lc.get("details", "Schicht-Inkonsistenz"))
    dc = checks.get("ds_coverage", {})
    if dc.get("es_without_ds_ids", 0) > 0:
        pct = round(dc["es_without_ds_ids"] / max(dc.get("es_total", 1), 1) * 100)
        problems.append(f"{pct}% der ES-Docs ohne ds_ids")
    warn_ds = [d for d in dc.get("datasources", []) if d.get("status") == "warning"]
    if warn_ds:
        names = ", ".join(f"DS {d['id']} ({d['name']})" for d in warn_ds[:5])
        suffix = f" (+{len(warn_ds) - 5} weitere)" if len(warn_ds) > 5 else ""
        problems.append(f"Inkonsistente DS: {names}{suffix}")
    colbert_warn = dc.get("colbert_warnings", [])
    if colbert_warn:
        names = ", ".join(f"DS {d['id']} ({d['name']})" for d in colbert_warn[:5])
        suffix = f" (+{len(colbert_warn) - 5} weitere)" if len(colbert_warn) > 5 else ""
        problems.append(f"ColBERT aktiviert aber ohne Chunks: {names}{suffix}")
    ed = checks.get("es_duplicates", {})
    if ed.get("suspected_duplicates", 0) > 0:
        problems.append(
            f"{ed['suspected_duplicates']} Dateien mit verdaechtig vielen ES-Eintraegen"
        )
    oc = checks.get("orphan_chunks", {})
    if oc.get("orphan_pct", 0) > 0:
        problems.append(f"{oc['orphan_pct']}% verwaiste Chunks (Stichprobe)")
    od = checks.get("overlapping_ds", {})
    if od.get("overlaps"):
        problems.append(f"{len(od['overlaps'])} ueberlappende DS-Pfade")
    qf = checks.get("qdrant_fragmentation", {})
    if qf.get("fragmented_count", 0) > 0:
        frags = [c for c in qf.get("collections", []) if c.get("fragmented")]
        names = ", ".join(f"{c['name']} ({c['segments']} Segmente)" for c in frags[:3])
        problems.append(f"Fragmentiert: {names}")
    sc = checks.get("stale_colbert", {})
    stale_colls = sc.get("stale_collections", [])
    orphaned_dbs = sc.get("orphaned_db_entries", [])
    if stale_colls or orphaned_dbs:
        parts = []
        if stale_colls:
            parts.append(f"{len(stale_colls)} stale ColBERT-Collection(s)")
        if orphaned_dbs:
            parts.append(f"{len(orphaned_dbs)} verwaiste ColBERT-DB-Eintraege")
        problems.append("; ".join(parts))

    summary = "; ".join(problems) if problems else "Alle Pruefungen bestanden"

    return {
        "checks": checks,
        "overall_status": overall,
        "summary": summary,
    }
