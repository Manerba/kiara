"""Source-local ColBERT target discovery before destructive cleanup.

Unverified collections may supply IDs only, under the exclusive config fence.
That evidence selects targets; it never authorizes a read/write store. Any
inventory failure propagates before cleanup. All selected stores subsequently
pass the ordinary stored-binding guard.
"""

import json
from contextlib import closing

from kiara.retrieval import index_compatibility as contracts
from kiara.retrieval.colbert import (
    colbert_collection_name,
    colbert_ds_model_key,
    model_name_from_colbert_collection,
    model_name_from_colbert_ds_key,
)
from kiara.retrieval.vectorstore_qdrant import _normalize_point_id, build_qdrant_filter
from kiara.shared.constants import QDRANT_SCROLL_BATCH_SIZE
from kiara.shared.db import db_query


def _source_ids(client, collection, source):
    offset, seen, ids = None, set(), set()
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=build_qdrant_filter({"source": source}),
            limit=QDRANT_SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(_normalize_point_id(p.id) for p in points)
        if next_offset is None:
            return ids
        if not points or next_offset in seen:
            raise RuntimeError("ColBERT-Cleanup-Inventar hat keinen Fortschritt")
        seen.add(next_offset)
        offset = next_offset


def resolve_source_colbert_keys(config, store, chunk_ds, datasource_id, model, sources):
    """Return affected keys from payloads, PG and durable inventories, not DS alone.

    No text/vector transfer or corpus scan: collection names, source-filtered
    ID pages and bounded ID lookups only. Collection presence and global PG
    model keys are candidates, never evidence of source involvement.
    """
    from kiara.ingestion.document_publication import pending_source_cleanups
    from kiara.retrieval import elasticsearch_service as es
    from kiara.retrieval.elasticsearch_indices import (
        DatasourceIndexTarget,
        ElasticsearchDocumentAddress,
    )

    contracts.require_source_cleanup_fence()
    selected = set()
    use_es = es.model_uses_elasticsearch(config, model)
    pg_keys = {key for key in chunk_ds.distinct_models() if key.startswith("colbert::")}
    with closing(contracts.raw_qdrant_client(config)) as client:
        collections = {}
        for item in client.get_collections().collections:
            if not item.name.startswith("colbert_"):
                continue
            # An unresolvable physical name is uncertainty, never absence.
            raw_model = model_name_from_colbert_collection(item.name)
            if colbert_collection_name(raw_model) != item.name:
                raise RuntimeError(f"Nicht eindeutige ColBERT-Collection: {item.name}")
            collections[colbert_ds_model_key(raw_model)] = item.name
        for start in range(0, len(sources), 100):
            batch = sources[start : start + 100]
            pending = pending_source_cleanups(str(store.get_collection_name()), sources=batch)
            hash_owners = {}
            if use_es:
                # invalidate_documents will preserve these owners in the
                # durable ES inventory. Discovery must see the same evidence
                # *before* hash invalidation, including other/shared owners.
                with db_query() as conn:
                    rows = conn.execute(
                        "SELECT file_path, datasource_id FROM file_hashes "
                        "WHERE embedding_model = %s AND file_path = ANY(%s)",
                        (model, batch),
                    ).fetchall()
                for row in rows:
                    hash_owners.setdefault(row["file_path"], set()).add(row["datasource_id"])
            for source in batch:
                saved = pending.get(source, {})
                selected.update(key for key in saved if key.startswith("colbert::"))
                ids = set(store.get_ids_by_sources([source]))
                for key, values in saved.items():
                    if key == model or key.startswith("colbert::"):
                        ids.update(values)
                for key, collection in collections.items():
                    found = _source_ids(client, collection, source)
                    if found:
                        selected.add(key)
                        ids.update(found)
                if use_es:
                    es_key = f"es::{model}"
                    addresses = {
                        ElasticsearchDocumentAddress(*json.loads(value))
                        for value in saved.get(es_key, [])
                    }
                    ids.update(a.chunk_id for a in addresses)
                    owners = set(chunk_ds.get_other_datasources(list(ids), -1, model))
                    owners.update(hash_owners.get(source, ()))
                    owners.update(a.datasource_id for a in addresses)
                    owners.update(int(ds) for ds in saved.get(es_key + "::owners", []))
                    if datasource_id is not None:
                        owners.add(datasource_id)
                    targets = es.targets_for_datasources(config, owners, model)
                    if addresses:
                        targets = tuple(
                            DatasourceIndexTarget(t.datasource_id, t.index_name, 0) for t in targets
                        )
                    ids.update(
                        a.chunk_id for a in es.get_ids_by_source(config, source, targets=targets)
                    )
                ordered_ids = sorted(ids)
                for offset in range(0, len(ordered_ids), QDRANT_SCROLL_BATCH_SIZE):
                    id_batch = ordered_ids[offset : offset + QDRANT_SCROLL_BATCH_SIZE]
                    for key in pg_keys - selected:
                        if chunk_ds.get_other_datasources(id_batch, -1, key):
                            selected.add(key)
                    # IDs can survive under an older payload source or without
                    # PG owners. Actual point existence is also source evidence.
                    for key in collections.keys() - selected:
                        if client.retrieve(
                            collection_name=collections[key],
                            ids=id_batch,
                            with_payload=False,
                            with_vectors=False,
                        ):
                            selected.add(key)
    # Validate durable model keys even when their physical collection is gone;
    # the ordinary guard will then fail closed rather than lose the inventory.
    for key in selected:
        model_name_from_colbert_ds_key(key)
    return frozenset(selected)
