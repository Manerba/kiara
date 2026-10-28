"""Gemeinsame Quellen: Dateiabschluesse und konkurrierende DS-Loeschungen."""

import threading
from copy import copy, deepcopy
from pathlib import Path

import pytest
from test_document_version_completion import document_run, read  # noqa: F401
from test_index_compatibility_s4 import _config, _persist_contract, _resolved

from kiara.exceptions import RetrievalError, VectorDBConnectionError
from kiara.ingestion import document_publication as publication
from kiara.ingestion import runtime
from kiara.ingestion import vectordb_ops as ops
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.ingestion.pipeline import _complete_file
from kiara.retrieval import index_compatibility as ic

_write_to_elasticsearch = runtime._write_to_elasticsearch


def _second_datasource(case, conn):
    ds = conn.execute(
        "INSERT INTO datasources (name, type, path) VALUES ('Shared', 'local', %s) RETURNING id",
        (str(Path(case.path).parent),),
    ).fetchone()["id"]
    conn.commit()
    return ds


def _run_as(case, ds):
    ctx = copy(case.context)
    ctx.datasource_id = ds
    ctx.stats = {"processed_files": 0, "total_chunks": 0, "errors": 0}
    ctx.pending_hashes = {}
    ctx.file_completion_callback = lambda path: _complete_file(
        case.hashes, ds, "test-model", path, ("source-hash", 300)
    )
    assert runtime._process_file_in_pipeline(ctx, case.path)
    assert ctx.stats["processed_files"] == 1
    assert ctx.stats["dedup_chunks"] == 3


def _assert_completion(case, conn, ds, *, completed):
    assert (case.path in case.hashes.load(ds, "test-model")) == completed
    row = conn.execute(
        "SELECT chunk_count, file_count FROM datasource_embeddings "
        "WHERE datasource_id=%s AND embedding_model='test-model'",
        (ds,),
    ).fetchone()
    assert (row["chunk_count"], row["file_count"]) == (3, int(completed))


@pytest.mark.parametrize("interrupt", [False, True])
def test_shared_full_dedup_preserves_other_datasource_completion(document_run, test_db, interrupt):
    case = document_run
    assert case.run()
    ds2 = _second_datasource(case, test_db)
    before = case.snapshot()
    case.enrich.reset_mock()
    _run_as(case, ds2)
    if interrupt:
        case.fail_es = True
        with pytest.raises(RuntimeError, match="ES publication failed"):
            _run_as(case, ds2)
        _assert_completion(case, test_db, case.ds, completed=True)
        _assert_completion(case, test_db, ds2, completed=False)
        # Der erhaltene Hash einer anderen DS hebt die dokumentweite Sperre nicht auf.
        with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
            case.reader()
        case.fail_es = False
        _run_as(case, ds2)
    for ds in (case.ds, ds2):
        _assert_completion(case, test_db, ds, completed=True)
    _run_as(case, case.ds)
    for ds in (case.ds, ds2):
        _assert_completion(case, test_db, ds, completed=True)
    case.enrich.assert_not_called()
    assert [c["text"] for c in case.snapshot()] == [c["text"] for c in before]
    assert all(c["metadata"]["ds_ids"] == [case.ds, ds2] for c in case.snapshot())
    tool, ref = case.reader()
    assert read(tool, ref, max_chunks=3).success


def test_source_cleanup_still_invalidates_all_datasource_completions(document_run, test_db):
    case = document_run
    assert case.run()
    ds2 = _second_datasource(case, test_db)
    _run_as(case, ds2)
    publication.invalidate_documents("kb_test", [case.path], "test-model")
    for ds in (case.ds, ds2):
        _assert_completion(case, test_db, ds, completed=False)
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader()


def test_file_completion_rejects_different_datasource(document_run, test_db):
    case = document_run
    assert case.run()
    ds2 = _second_datasource(case, test_db)
    with publication.document_publication(
        "kb_test", case.path, "test-model", datasource_id=case.ds
    ) as operation:
        operation.begin()
        publication.stage_document_versions(
            {case.path: {"standard": case.snapshot()[0]["metadata"]["processing_version"]}}
        )
        with pytest.raises(RuntimeError, match="ohne bestaetigte Gesamtversion"):
            _complete_file(case.hashes, ds2, "test-model", case.path, ("source-hash", 300))
    assert case.path not in case.hashes.load(ds2, "test-model")
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader()


def _use_real_guards(case, conn, monkeypatch):
    config = case.store._config
    config.update(_config())
    target = ic.make_index_target(
        config,
        collection_name="kb_test",
        vector_kind=ic.IndexVectorKind.DENSE,
        requested_model="test-model",
    )
    _persist_contract(conn, target, _resolved(target))
    conn.commit()
    case.store._index_target = target
    case.store._require_configured_default = False
    del case.store._index_contract_guard
    case.store._client = ic.GuardedQdrantClient(case.client, config, target)

    class BorrowedClient:
        def __getattr__(self, name):
            return getattr(case.client, name)

        def close(self):
            pass

    def reject_live_lookup(*args, **kwargs):
        raise AssertionError("Unexpected live embedding lookup")

    monkeypatch.setattr(ic, "raw_qdrant_client", lambda _: BorrowedClient())
    monkeypatch.setattr(ic, "stored_primary_index_target", lambda *a, **kw: target)
    monkeypatch.setattr(ic, "resolve_current_index_contract", reject_live_lookup)
    monkeypatch.setattr(ops, "VectorStore", lambda *a, **kw: case.store)
    monkeypatch.setattr(ops, "_resolve_colbert_models_for_ds", lambda *a, **kw: [])


def _use_es_sdk(case, deleted_ds, monkeypatch, request):
    """Real ES bulk serialization against index-aware controlled responses."""
    import json

    from elastic_transport import ApiResponseMeta, NodeConfig
    from elastic_transport._response import ObjectApiResponse
    from elasticsearch import Elasticsearch
    from test_elasticsearch_datasource_lifecycle import Backend

    from kiara.retrieval import elasticsearch_service as service
    from kiara.retrieval.elasticsearch_indices import datasource_index_name

    config = case.store._config
    config.setdefault("retrieval", {})["hybrid_search"] = "elasticsearch"
    monkeypatch.setattr("kiara.config.get_embedding_model", lambda cfg: "test-model")
    backend = Backend()
    for cid, (text, meta) in case.es.items():
        for ds in meta["ds_ids"]:
            name = datasource_index_name(config, ds)
            backend.create(index=name)
            backend.docs[name][cid] = {"text": text, "metadata": deepcopy(meta)}
    case.es_backend = backend
    es = Elasticsearch("http://127.0.0.1:1")
    request.addfinalizer(es.close)
    monkeypatch.setattr(es, "options", lambda **kw: es)
    for name in ("exists", "create", "get_settings", "stats", "refresh", "delete"):
        monkeypatch.setattr(es.indices, name, getattr(backend.indices, name))
    monkeypatch.setattr(es, "mget", backend.mget)
    monkeypatch.setattr(es, "search", backend.search)

    def sync_view():
        case.es.clear()
        for docs in backend.docs.values():
            for cid, doc in docs.items():
                case.es[cid] = (doc["text"], deepcopy(doc["metadata"]))

    def delete_query(**kwargs):
        result = backend.delete_by_query(**kwargs)
        sync_view()
        return result

    monkeypatch.setattr(es, "delete_by_query", delete_query)

    def transport(*args, **kwargs):
        body = kwargs["body"]
        if isinstance(body, (bytes, str)):
            body = body.splitlines()
        items = iter(json.loads(v) if isinstance(v, (bytes, str)) else v for v in body)
        results = []
        for action in items:
            op = next(iter(action))
            command = {"_op_type": op, **action[op]}
            if op == "index":
                command["_source"] = next(items)
            elif op == "update":
                command.update(next(items))
            count, errors = backend.bulk(None, [command])
            result = {
                "_id": command["_id"],
                "_index": command["_index"],
                "status": 200 if count else 409,
            }
            if errors:
                result["error"] = {"type": "version_conflict_engine_exception"}
            results.append({op: result})
        sync_view()
        return ObjectApiResponse(
            {
                "errors": any(next(iter(r.values()))["status"] >= 300 for r in results),
                "items": results,
            },
            ApiResponseMeta(200, "1.1", {}, 0.0, NodeConfig("http", "127.0.0.1", 1)),
        )

    monkeypatch.setattr(es, "perform_request", transport)
    monkeypatch.setattr(service, "_get_client", lambda _: es)
    monkeypatch.setattr(runtime, "_write_to_elasticsearch", _write_to_elasticsearch)


@pytest.mark.parametrize("first", ["publication", "deletion"])
def test_publication_and_shared_ds_deletion_keep_all_indexes_consistent(
    document_run, test_db, monkeypatch, request, first
):
    case = document_run
    assert case.run()
    ds2 = _second_datasource(case, test_db)
    _run_as(case, ds2)
    _use_real_guards(case, test_db, monkeypatch)
    _use_es_sdk(case, ds2, monkeypatch, request)
    delete_started, delete_done = threading.Event(), threading.Event()
    ingest_started, ingest_done = threading.Event(), threading.Event()
    deletion_inside, release_delete = threading.Event(), threading.Event()
    errors = []

    def remove_second():
        delete_started.set()
        try:
            ops.delete_files(
                case.store._config,
                ds2,
                "test-model",
                [case.path],
                chunk_ds=ChunkDatasources(),
                hash_store=case.hashes,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            delete_done.set()

    def ingest():
        ingest_started.set()
        try:
            _run_as(case, case.ds)
        except BaseException as exc:
            errors.append(exc)
        finally:
            ingest_done.set()

    delete_worker = threading.Thread(target=remove_second)
    ingest_worker = threading.Thread(target=ingest)
    if first == "publication":

        def publish(*args, **kwargs):
            if delete_worker.ident is None:
                delete_worker.start()
            assert delete_started.wait(3)
            assert not delete_done.wait(0.2), "deletion overtook the publication snapshot"
            return _write_to_elasticsearch(*args, **kwargs)

        monkeypatch.setattr(runtime, "_write_to_elasticsearch", publish)
    else:
        delete_es = ops.delete_from_elasticsearch

        def pause_delete(*args, **kwargs):
            deletion_inside.set()
            assert release_delete.wait(5)
            return delete_es(*args, **kwargs)

        monkeypatch.setattr(ops, "delete_from_elasticsearch", pause_delete)
    try:
        if first == "deletion":
            delete_worker.start()
            assert deletion_inside.wait(5)
        ingest_worker.start()
        assert ingest_started.wait(3)
        if first == "deletion":
            assert not ingest_done.wait(0.2), "ingestion bypassed the active deletion"
            release_delete.set()
        assert ingest_done.wait(10)
        assert delete_done.wait(10)
    finally:
        release_delete.set()
        for worker in (delete_worker, ingest_worker):
            if worker.ident is not None:
                worker.join(timeout=10)
    assert not errors
    stored = case.snapshot()
    memberships = ChunkDatasources().get_datasource_ids_for_chunks(
        [c["_doc_id"] for c in stored], "test-model"
    )
    assert all(owners == [case.ds] for owners in memberships.values())
    assert all(c["metadata"]["ds_ids"] == [case.ds] for c in stored)
    # DS deletion leaves foreign index documents intact. PG remains membership SoT.
    assert all(case.ds in meta["ds_ids"] for _, meta in case.es.values())
    from kiara.retrieval.elasticsearch_indices import datasource_index_name

    assert case.es_backend.docs[datasource_index_name(case.store._config, ds2)] == {}
    _assert_completion(case, test_db, case.ds, completed=True)
    assert not case.hashes.load(ds2, "test-model")
    assert (
        test_db.execute(
            "SELECT * FROM datasource_embeddings WHERE datasource_id=%s", (ds2,)
        ).fetchone()
        is None
    )
    # Eine frisch aus ES gewonnene Referenz muss in Qdrant lesbar sein.
    from kiara.retrieval.retriever import Retriever

    retriever = Retriever.__new__(Retriever)
    retriever._store = case.store
    hits = retriever.filter_available_results(
        [{"_doc_id": cid, "text": text, "metadata": meta} for cid, (text, meta) in case.es.items()],
        [case.ds],
    )
    tool, ref = case.reader(hits)
    assert read(tool, ref, max_chunks=3).success


def test_publication_rejects_membership_changed_outside_fence(document_run, monkeypatch):
    case = document_run
    assert case.run()
    chunk = case.snapshot()[0]

    def corrupt_membership(*args, **kwargs):
        case.client.set_payload("kb_test", {"ds_ids": [case.ds, 999]}, [chunk["_doc_id"]])

    monkeypatch.setattr(runtime, "_write_to_elasticsearch", corrupt_membership)
    with pytest.raises(VectorDBConnectionError, match="Freigabe nicht bestaetigt"):
        case.run()
    assert case.context.stats["processed_files"] == 0
    assert case.path not in case.hashes.load(case.ds, "test-model")
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader()
