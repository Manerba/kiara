"""Exact model-bound ES inventory and bounded canonical reconciliation.

Calls require the service's default-model guard. Counts are physical copies;
verification is explicitly sampled unless full=True (the maintenance gate).
"""

from collections import Counter

from kiara.backend_selector import normalize_model_name
from kiara.config import get_embedding_model
from kiara.retrieval.elasticsearch_indices import document_address, resolve_datasource_targets
from kiara.shared.db import db_query


def exact_targets(config):
    model = normalize_model_name(get_embedding_model(config))
    with db_query() as conn:
        rows = conn.execute(
            "SELECT ds.id, COUNT(cd.chunk_id) AS count FROM datasources ds "
            "LEFT JOIN chunk_datasources cd ON cd.datasource_id = ds.id "
            "AND cd.embedding_model = %s GROUP BY ds.id ORDER BY ds.id",
            (model,),
        ).fetchall()
        orphan = conn.execute(
            "SELECT 1 FROM chunk_datasources cd WHERE cd.embedding_model = %s "
            "AND NOT EXISTS (SELECT 1 FROM datasources ds WHERE ds.id = cd.datasource_id) LIMIT 1",
            (model,),
        ).fetchone()
    if orphan:
        raise RuntimeError("PG-Zuordnung ohne Datenquellen-Owner")
    counts = {row["id"]: row["count"] for row in rows}
    return resolve_datasource_targets(
        config, None, allowed_ds_ids=list(counts), expected_document_counts=counts
    )


def canonical_batches(config, store, targets, batch_size=500):
    """One membership query and one publication query per canonical batch."""
    from kiara.ingestion.document_publication import assert_documents_published
    from kiara.retrieval.embedding import collection_name_for_model

    model = normalize_model_name(get_embedding_model(config))
    allowed = {t.datasource_id for t in targets}
    for batch in store.scroll_all_with_text(batch_size=batch_size):
        ids = [cid for cid, _, _ in batch]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Doppelte kanonische IDs im Batch")
        with db_query() as conn:
            rows = conn.execute(
                "SELECT chunk_id, datasource_id FROM chunk_datasources "
                "WHERE embedding_model = %s AND chunk_id = ANY(%s)",
                (model, ids),
            ).fetchall()
        owners = {cid: [] for cid in ids}
        for row in rows:
            if row["datasource_id"] not in allowed:
                raise RuntimeError("PG-Owner ausserhalb des Zielinventars")
            owners[row["chunk_id"]].append(row["datasource_id"])
        canonical = []
        for cid, text, meta in batch:
            memberships = sorted(owners[cid])
            if not memberships:
                raise RuntimeError(f"Chunk {cid} ohne bestaetigte PG-Zuordnung")
            if not isinstance(text, str) or not text:
                raise RuntimeError(f"Chunk {cid} ohne kanonischen Text")
            if sorted(meta.get("ds_ids", [])) != memberships:
                raise RuntimeError(f"Chunk {cid}: PG-/Qdrant-Zuordnungen widersprechen sich")
            canonical.append((cid, text, {**meta, "ds_ids": memberships}))
        assert_documents_published(
            collection_name_for_model(model), [{"metadata": m} for _, _, m in canonical]
        )
        yield canonical


def _store(config):
    from kiara.retrieval.vectorstore import VectorStore

    return VectorStore(config, embedding_model=get_embedding_model(config), intent="maintenance")


def _publication_blocker(config):
    """The ledger also covers interrupted sources that have no Qdrant points."""
    from kiara.retrieval.embedding import collection_name_for_model

    collection = collection_name_for_model(normalize_model_name(get_embedding_model(config)))
    with db_query() as conn:
        rows = conn.execute(
            "SELECT source FROM document_publications WHERE collection_name = %s "
            "AND completed_at IS NULL ORDER BY source LIMIT 21",
            (collection,),
        ).fetchall()
    if rows:
        return {
            "status": "error",
            "release_ready": False,
            "reason_code": "open_document_publications",
            "error": "Offene Dokumentpublikationen: kanonischer Bestand nicht freigegeben",
            "collection": collection,
            "open_publications": [row["source"] for row in rows[:20]],
            "open_publications_truncated": len(rows) > 20,
        }
    return None


def _binding(config):
    from kiara.retrieval.index_compatibility import current_guard_token

    token = current_guard_token(model=get_embedding_model(config))
    if token is None or not token.require_configured_default:
        raise RuntimeError("ES-Inventar ohne gebundene Default-Modellpruefung")
    fields = token.job_fields()
    return {
        key: fields[key]
        for key in (
            "contract_logical_index_key",
            "contract_index_compatibility_id",
            "contract_snapshot_sha256",
            "contract_fence_version",
        )
    }


def preview(config, batch_size=500):
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:
        raise ValueError("batch_size muss zwischen 1 und 1000 liegen")
    blocker = _publication_blocker(config)
    if blocker:
        return blocker
    targets = exact_targets(config)
    store = _store(config)
    counts = Counter()
    logical = 0
    for batch in canonical_batches(config, store, targets, batch_size):
        logical += len(batch)
        for _, _, meta in batch:
            counts.update(meta["ds_ids"])
    expected = {t.datasource_id: t.expected_document_count for t in targets}
    if any(counts[ds] != count for ds, count in expected.items()):
        raise RuntimeError("PG-/Qdrant-Mengen weichen ab")
    qdrant_count = store.get_stats()["total_chunks"]
    if logical != qdrant_count:
        raise RuntimeError("Unvollstaendiger kanonischer Scroll")
    from kiara.retrieval import elasticsearch_service as es

    # A preview must describe missing replacement indices, not pretend they exist.
    # Zero expectations here only permit observing absence; they are NOT a release gate.
    observed = es.inspect_datasource_indices(
        config, es._operation_targets(config, [t.datasource_id for t in targets])
    )
    absent = {t.datasource_id for t in observed.absent_empty}
    return {
        "status": "preview",
        "index_binding": _binding(config),
        "collection": store.get_stats()["collection"],
        "embedding_model": get_embedding_model(config),
        "qdrant_count": logical,
        "expected_physical_documents": sum(expected.values()),
        "unit": "physical_primary_documents",
        "batch_size": batch_size,
        "datasources": [
            {
                "datasource_id": t.datasource_id,
                "index_name": t.index_name,
                "expected": t.expected_document_count,
                "observed": observed.document_counts[t.datasource_id],
                "current_state": (
                    "missing_expected" if t.expected_document_count else "absent_empty"
                )
                if t.datasource_id in absent
                else "present",
            }
            for t in targets
        ],
        "steps": [
            "Wartung sperren",
            "Rueckweg sichern",
            "Quellenindizes neu aufbauen",
            "Refresh",
            "Vollstaendiger Text-/Metadaten-/Mengennachweis",
            "Freigabe",
        ],
    }


def inventory(config, targets=None):
    from kiara.retrieval import elasticsearch_service as es

    targets = exact_targets(config) if targets is None else es._validate_targets(config, targets)
    available = es.inspect_datasource_indices(config, targets)
    rows = [
        {
            "datasource_id": t.datasource_id,
            "index_name": t.index_name,
            "expected": t.expected_document_count,
            "observed": available.document_counts[t.datasource_id],
            "state": "absent_empty" if t in available.absent_empty else "present",
        }
        for t in targets
    ]
    return {
        "datasources": rows,
        "doc_count": available.physical_documents,
        "expected_physical_documents": sum(t.expected_document_count for t in targets),
        "counts_match": all(r["expected"] == r["observed"] for r in rows),
        "unit": "physical_primary_documents",
        "embedding_model": get_embedding_model(config),
    }, available


def verify(config, sample_size=100, *, full=False, batch_size=500):
    from kiara.retrieval import elasticsearch_service as es

    blocker = _publication_blocker(config)
    if blocker:
        return blocker
    targets = exact_targets(config)
    result, _ = inventory(config, targets)
    store = _store(config)
    result.update(
        index_binding=_binding(config),
        qdrant_count=store.get_stats()["total_chunks"],
        sampled=0,
        matching=0,
        missing_in_es=0,
        mismatch_count=0,
        errors=0,
        mismatches=[],
        verification="full" if full else "sample",
        release_ready=False,
    )
    counts = Counter()
    logical = 0
    for batch in canonical_batches(config, store, targets, batch_size):
        logical += len(batch)
        # Bound physical actions too, including many owners of one chunk.
        pending = {}

        def check(pending=pending):
            docs = es._mget(config, list(pending))
            for address, expected in pending.items():
                result["sampled"] += 1
                doc = docs[address]
                if not doc["found"]:
                    result["missing_in_es"] += 1
                elif doc.get("_source") == expected:
                    result["matching"] += 1
                else:
                    result["mismatch_count"] += 1
                    if len(result["mismatches"]) < 20:
                        result["mismatches"].append(
                            {"chunk_id": address.chunk_id, "index": address.index_name}
                        )
            pending.clear()

        for cid, text, meta in batch:
            counts.update(meta["ds_ids"])
            for ds in meta["ds_ids"]:
                if full or result["sampled"] + len(pending) < sample_size:
                    pending[document_address(config, ds, cid)] = {"text": text, "metadata": meta}
                    if len(pending) >= batch_size:
                        check()
        if pending:
            check()
        if not full and result["sampled"] >= sample_size:
            break
    canonical_complete = (
        full
        and logical == result["qdrant_count"]
        and all(counts[t.datasource_id] == t.expected_document_count for t in targets)
    )
    passed = result["counts_match"] and result["matching"] == result["sampled"]
    if full:
        passed = (
            passed
            and canonical_complete
            and result["sampled"] == result["expected_physical_documents"]
        )
    result.update(
        status="success" if passed else "error",
        release_ready=bool(full and passed),
        canonical_complete=canonical_complete,
        integrity_pct=round(100 * result["matching"] / result["sampled"], 1)
        if result["sampled"]
        else None,
    )
    if not passed:
        result["error"] = (
            f"Integritaetspruefung fehlgeschlagen: {result['missing_in_es']} fehlend, "
            f"{result['mismatch_count']} mit abweichendem Inhalt; "
            f"Quellenmengen {'identisch' if result['counts_match'] else 'abweichend'}"
        )
        if full and not canonical_complete:
            result["error"] += "; kanonischer Bestand unvollstaendig"
    # Recheck the durable ledger after the scan, including sources absent from it.
    blocker = _publication_blocker(config)
    if blocker:
        result.update(blocker)
    return result


def rebuild(config, progress_callback=None, *, dry_run=False, batch_size=500, cancel=None):
    from elasticsearch.helpers import streaming_bulk

    from kiara.retrieval import elasticsearch_service as es
    from kiara.retrieval.elasticsearch_maintenance import require_session

    if not dry_run:
        require_session(config)
    plan = preview(config, batch_size)
    if dry_run or plan.get("status") == "error":
        return plan
    targets = exact_targets(config)
    per_ds = {t.index_name: {"attempted": 0, "succeeded": 0, "failed": 0} for t in targets}
    result = {
        **plan,
        "actions_by_index": per_ds,
        "status": "error",
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "total": plan["expected_physical_documents"],
        "release_ready": False,
    }

    def progress():
        if progress_callback:
            progress_callback(dict(result))

    def cancelled():
        if cancel and cancel():
            raise RuntimeError("ES-Rebuild kontrolliert abgebrochen")

    try:
        cancelled()
        # Only canonical source indices. The retained old common index is untouched.
        for target in targets:
            es.delete_datasource_index(config, datasource_id=target.datasource_id)
            es.create_index(config, datasource_id=target.datasource_id)
        store = _store(config)

        def actions():
            for batch in canonical_batches(config, store, targets, batch_size):
                cancelled()
                for cid, text, meta in batch:
                    for ds in meta["ds_ids"]:
                        address = document_address(config, ds, cid)
                        result["attempted"] += 1
                        per_ds[address.index_name]["attempted"] += 1
                        yield {
                            "_op_type": "index",
                            "_index": address.index_name,
                            "_id": cid,
                            "_source": {"text": text, "metadata": meta},
                        }

        for ok, item in streaming_bulk(
            es._get_client(config),
            actions(),
            chunk_size=batch_size,
            raise_on_error=False,
            raise_on_exception=True,
        ):
            key = "succeeded" if ok else "failed"
            result[key] += 1
            response = next(iter(item.values()))
            per_ds[response["_index"]][key] += 1
            if (result["succeeded"] + result["failed"]) % batch_size == 0:
                progress()
        if result["failed"] or result["succeeded"] != result["total"]:
            raise RuntimeError("ES-Bulk nicht vollstaendig bestaetigt")
        cancelled()
        es.refresh_index(config, targets=targets, require_index=True)
        verification = verify(config, full=True, batch_size=batch_size)
        result["verification_result"] = verification
        if not verification["release_ready"]:
            raise RuntimeError("ES-Verifikation fehlgeschlagen")
        result.update(status="success", release_ready=True)
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["unconfirmed"] = result["attempted"] - result["succeeded"] - result["failed"]
        for counters in per_ds.values():
            counters["unconfirmed"] = (
                counters["attempted"] - counters["succeeded"] - counters["failed"]
            )
        progress()
    return result
