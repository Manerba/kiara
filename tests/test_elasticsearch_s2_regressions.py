"""S2 audit regressions: pending adoption, bounded cleanup and refresh evidence."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_document_publication_memberships import _second_datasource
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_elasticsearch_datasource_lifecycle import (  # noqa: F401
    CONFIG,
    backend,
    targets,
    whole_topology,
    write,
)

from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline, vectordb_ops
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval import elasticsearch_service as es


@pytest.mark.parametrize("failure", ["create", "empty_bulk"])
@pytest.mark.parametrize("retry_owner", ["new", "existing"])
def test_failed_new_owner_publication_can_retry(
    whole_topology, test_db, monkeypatch, failure, retry_owner
):
    case = whole_topology
    ds2 = _second_datasource(case, test_db)
    assert case.run_whole(case.ds)["processed_files"] == 1
    before_embeddings = case.embeddings
    index = f"kiara_ds_{ds2}"
    client = es._get_client(case.store._config)
    original_create = client.indices.create
    original_bulk = case.es_backend.bulk
    with monkeypatch.context() as fault:

        def fail_create(*, index, **kwargs):
            if index == f"kiara_ds_{ds2}":
                raise RuntimeError("injected new-owner create failure")
            return original_create(index=index, **kwargs)

        def fail_bulk(client, actions, **kwargs):
            actions = list(actions)
            if any(a["_index"] == index for a in actions):
                return 0, [{"index": {"status": 503}}]
            return original_bulk(client, actions, **kwargs)

        if failure == "create":
            fault.setattr(client.indices, "create", fail_create)
        else:
            fault.setattr(case.es_backend, "bulk", fail_bulk)
        with pytest.raises(RuntimeError):
            case.run_whole(ds2)
    assert not case.hashes.load(ds2, "test-model")
    assert ChunkDatasources().count_by_datasource(ds2, "test-model") == 3
    assert not case.es_backend.docs.get(index)
    # Failure is removed. Canonical Qdrant text remains; ordinary retry must recover.
    result = case.run_whole(ds2 if retry_owner == "new" else case.ds)
    assert result["errors"] == 0 and result["processed_files"] == 1
    assert len(case.es_backend.docs[index]) == 3
    assert case.embeddings == before_embeddings
    assert case.hashes.load(ds2 if retry_owner == "new" else case.ds, "test-model")
    assert not publication.pending_document_sources("kb_test", [case.path])
    for chunk in case.snapshot():
        doc = case.es_backend.visible[index][chunk["_doc_id"]]
        assert doc["text"] == chunk["text"]
        assert doc["metadata"]["ds_ids"] == sorted([case.ds, ds2])


@pytest.mark.parametrize(
    "reply",
    [
        {"_shards": {"total": 0, "successful": 0, "failed": 0}},
        {"_shards": {"failed": 0}},
        *(
            {"_shards": dict(total=1, successful=1, failed=0) | {key: bad}}
            for key in ("total", "successful", "failed")
            for bad in (None, True, -1, "1", 1.0)
        ),
    ],
)
def test_refresh_requires_positive_complete_shard_ack(backend, reply):
    write(1, ids=("old",))
    es.refresh_index(CONFIG, targets=targets(CONFIG, 1, expected=1))
    write(1, ids=("new",))
    backend.indices.refresh = lambda **kw: deepcopy(reply)
    with pytest.raises(es.ElasticsearchPartialFailureError):
        es.refresh_index(CONFIG, targets=targets(CONFIG, 1, expected=1), require_index=True)
    assert "new" not in backend.visible["s2_ds_1"]


@pytest.mark.parametrize("n", [1, 100, 250])
def test_cleanup_inventory_reads_and_recounts_stay_bounded(monkeypatch, n):
    paths = [f"/audit/file-{i}" for i in range(n)]
    pending = {}
    docs = {p: [p] for p in paths}
    read_sizes, recounts = [], []

    def pending_read(collection, sources=None):
        selected = {p: pending[p] for p in sources if p in pending}
        read_sizes.append(len(selected))
        return deepcopy(selected)

    def invalidate(collection, sources, model, **kwargs):
        for p in sources:
            pending.setdefault(p, {model: []})

    def remember(collection, source, chunks):
        pending[source].update(chunks)

    def finish(collection, sources):
        for source in sources:
            pending.pop(source, None)

    monkeypatch.setattr(publication, "pending_source_cleanups", pending_read)
    monkeypatch.setattr(publication, "invalidate_documents", invalidate)
    monkeypatch.setattr(publication, "remember_source_cleanup", remember)
    monkeypatch.setattr(publication, "finish_source_cleanup", finish)
    monkeypatch.setattr(vectordb_ops, "invalidate_bm25", lambda *a: None)
    monkeypatch.setattr(
        vectordb_ops, "_recount_datasource_embeddings", lambda *a: recounts.append(a)
    )
    monkeypatch.setattr(pipeline, "db_transaction", lambda: nullcontext(object()))
    store = SimpleNamespace(
        get_collection_name=lambda: "audit",
        get_ids_by_sources=lambda sources: [cid for p in sources for cid in docs.get(p, [])],
        delete_by_sources=lambda sources: [docs.pop(p, None) for p in sources],
    )
    chunk_ds = SimpleNamespace(
        distinct_models=lambda: [],
        get_other_datasources=lambda ids, *_: [1] if ids else [],
        acquire_graph_cleanup_lock=lambda conn: None,
        delete_by_chunk_ids=lambda *a, **kw: None,
        filter_unreferenced_chunk_ids=lambda *a, **kw: [],
        get_datasource_ids_for_chunks=lambda *a: {},
    )
    hashes = SimpleNamespace(remove_files=lambda *a, **kw: None)
    cleanup = pipeline._cleanup_stale_chunks.__wrapped__
    cleanup({}, store, hashes, chunk_ds, 1, "test-model", paths)
    print(
        f"cleanup files={n}; inventory reads={len(read_sizes)}; rows materialized={sum(read_sizes)}; datasource-wide recounts={len(recounts)}"
    )
    assert not pending and not docs
    assert sum(read_sizes) == n and max(read_sizes) <= 100, (
        "whole pending inventory is reread for every file"
    )
    assert len(recounts) == (n + 99) // 100, "whole datasource is recounted for every file"


def test_new_owner_without_fault_completes(whole_topology, test_db):
    case = whole_topology
    ds2 = _second_datasource(case, test_db)
    assert case.run_whole(case.ds)["processed_files"] == 1
    before_embeddings = case.embeddings
    result = case.run_whole(ds2)
    assert result["errors"] == 0 and result["processed_files"] == 1
    assert case.hashes.load(ds2, "test-model")
    assert len(case.es_backend.docs[f"kiara_ds_{ds2}"]) == 3
    assert case.embeddings == before_embeddings


def test_positive_refresh_makes_second_write_visible(backend):
    write(1, ids=("old",))
    es.refresh_index(CONFIG, targets=targets(CONFIG, 1, expected=1))
    write(1, ids=("new",))
    assert "new" not in backend.visible["s2_ds_1"]
    es.refresh_index(CONFIG, targets=targets(CONFIG, 1, expected=1), require_index=True)
    assert "new" in backend.visible["s2_ds_1"]


@pytest.mark.parametrize("loss", ["index", "document"])
def test_pending_adoption_does_not_authorize_loss_of_completed_foreign_owner(
    whole_topology, test_db, monkeypatch, loss
):
    case = whole_topology
    other = _second_datasource(case, test_db)
    case.run_whole(case.ds)
    client = es._get_client(case.store._config)
    create = client.indices.create
    with monkeypatch.context() as fault:

        def fail_create(*, index, **kw):
            if index == f"kiara_ds_{other}":
                raise RuntimeError("adoption failure")
            return create(index=index, **kw)

        fault.setattr(client.indices, "create", fail_create)
        with pytest.raises(RuntimeError):
            case.run_whole(other)
    old_index = f"kiara_ds_{case.ds}"
    if loss == "index":
        case.es_backend.delete_index(index=old_index)
    else:
        case.es_backend.docs[old_index].pop(next(iter(case.es_backend.docs[old_index])))
    assert case.hashes.load(case.ds, "test-model")
    result = case.run_whole(other)
    assert result["errors"] == 1 and result["processed_files"] == 0
    assert "Zuordnungs" in str(result["error_details"])
    assert not case.hashes.load(other, "test-model")
    assert not case.es_backend.docs.get(f"kiara_ds_{other}")


def test_preflight_replay_requires_complete_source_evidence(whole_topology, test_db):
    case = whole_topology
    case.run_whole(case.ds)
    publication.invalidate_documents(
        "kb_test", [case.path], "test-model", hash_datasource_id=case.ds
    )
    # Unexplained PG membership is not covered by the pending source snapshot.
    ChunkDatasources().add_associations(["unexplained"], case.ds, "test-model")
    case.es_backend.delete_index(index=f"kiara_ds_{case.ds}")
    with pytest.raises(es.ElasticsearchPartialFailureError, match="Index fehlt"):
        case.run_whole(case.ds)
    assert not case.hashes.load(case.ds, "test-model")


def test_cleanup_batches_preserve_atomic_counters_and_durable_retry(
    document_run, test_db, monkeypatch
):
    case = document_run
    paths = [f"/batch/file-{i}" for i in range(250)]
    docs = {p: [p] for p in paths}
    chunk_ds = ChunkDatasources()
    chunk_ds.add_associations(paths, case.ds, "test-model")
    case.hashes.upsert_many(case.ds, "test-model", {p: ("hash", 1) for p in paths})
    recount = vectordb_ops._recount_datasource_embeddings
    calls = []

    def counted(*args):
        calls.append(args[1:])
        return recount(*args)

    monkeypatch.setattr(vectordb_ops, "_recount_datasource_embeddings", counted)

    def ids(sources):
        return [cid for p in sources for cid in docs.get(p, [])]

    store = SimpleNamespace(
        get_collection_name=lambda: "kb_test",
        get_ids_by_sources=ids,
        delete_by_sources=lambda sources: [docs.pop(p, None) for p in sources],
    )
    cleanup = pipeline._cleanup_stale_chunks.__wrapped__
    with monkeypatch.context() as fault:

        def fail_second_batch(sources):
            if paths[100] in sources:
                raise RuntimeError("second batch failure")
            return ids(sources)

        fault.setattr(store, "get_ids_by_sources", fail_second_batch)
        with pytest.raises(RuntimeError, match="second batch failure"):
            cleanup({}, store, case.hashes, chunk_ds, case.ds, "test-model", paths)
    row = test_db.execute(
        "SELECT chunk_count, file_count FROM datasource_embeddings WHERE datasource_id=%s AND embedding_model=%s",
        (case.ds, "test-model"),
    ).fetchone()
    assert dict(row) == {"chunk_count": 150, "file_count": 0}
    assert len(calls) == 2  # one atomic hash invalidation plus first batch's PG deletion
    assert set(publication.pending_cleanup_sources("kb_test")) == set(paths[100:])
    assert len(docs) == chunk_ds.count_by_datasource(case.ds, "test-model") == 150
    test_db.rollback()
    cleanup({}, store, case.hashes, chunk_ds, case.ds, "test-model", paths[100:])
    assert len(calls) == 4
    assert not docs and not publication.pending_source_cleanups("kb_test")
    assert not test_db.execute(
        "SELECT 1 FROM datasource_embeddings WHERE datasource_id=%s", (case.ds,)
    ).fetchone()
