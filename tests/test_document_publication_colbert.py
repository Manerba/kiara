"""ColBERT-Wiederanlauf: echte Multivektoren, Suchtreffer und Dateiabschluss."""

from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, MultiVectorComparator, MultiVectorConfig, VectorParams
from test_document_publication_memberships import _second_datasource
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_index_compatibility_s4 import _persist_contract, _resolved

from kiara.exceptions import RetrievalError, VectorDBConnectionError
from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline
from kiara.ingestion import vectordb_ops as ops
from kiara.ingestion._file_utils import ChunkDatasources, compute_file_hash
from kiara.retrieval import colbert
from kiara.retrieval import index_compatibility as ic
from kiara.retrieval.retriever import HybridRetriever

_real_colbert_store = colbert.ColBERTStore
_resolve_colbert_models = ops._resolve_colbert_models_for_ds


@pytest.fixture
def colbert_pipeline(whole_pipeline, test_db, monkeypatch):
    case = whole_pipeline
    config = case.store._config
    model = "publication-colbert"
    target = ic.colbert_index_target(config, model)
    case.client.create_collection(
        target.collection_name,
        vectors_config=VectorParams(
            size=2,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    )
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    store = _real_colbert_store.__new__(_real_colbert_store)
    store._config, store._model, store._collection = config, model, target.collection_name
    store._intent, store._index_target = "write", target
    store._client = ic.GuardedQdrantClient(case.client, config, target)
    config["retrieval"]["colbert"] = {"enabled": True, "model": model}
    state = SimpleNamespace(
        case=case, store=store, key=colbert.colbert_ds_model_key(model), calls=[]
    )

    def embed(_config, _model, texts, **kwargs):
        state.calls.extend(texts)
        return [[[0.0, 1.0]] if "B" * 100 in text else [[1.0, 0.0]] for text in texts]

    def factory(*args, **kwargs):
        return store

    factory.list_colbert_collections = lambda _: [
        {"model": model, "name": target.collection_name, "kind": "colbert", "chunks": 3}
    ]
    monkeypatch.setattr(colbert, "ColBERTStore", factory)
    monkeypatch.setattr(colbert, "embed_colbert", embed)
    monkeypatch.setattr(colbert, "embed_colbert_query", lambda *a, **kw: [[0.0, 1.0]])
    monkeypatch.setattr(ops, "_resolve_colbert_models_for_ds", _resolve_colbert_models)
    monkeypatch.setattr(
        ic, "resolve_current_index_contract", lambda conn, cfg, t, **kw: _resolved(t)
    )
    process = pipeline._process_file_in_pipeline

    def capture(ctx, path):
        state.context = ctx
        return process(ctx, path)

    monkeypatch.setattr(pipeline, "_process_file_in_pipeline", capture)

    def run(ds=None):
        return pipeline.run_ingestion(
            config,
            incremental=True,
            datasource_id=case.ds if ds is None else ds,
            embedding_model="test-model",
            ds_overrides={
                "chunk_size": 100,
                "chunk_overlap": 0,
                "contextual_retrieval": True,
                "colbert_enabled": True,
            },
        )

    state.run = run
    return state


def _open_retry(state):
    case = state.case
    publication.invalidate_documents(
        "kb_test", [case.path], "test-model", hash_datasource_id=case.ds
    )
    state.calls.clear()
    case.enrich.reset_mock()


def _assert_complete(state, db, ids, owners):
    case = state.case
    points = case.client.retrieve(
        state.store._collection, ids, with_vectors=True, with_payload=True
    )
    assert {p.id.replace("-", "") for p in points} == set(ids)
    assert all(p.vector and p.payload["ds_ids"] == owners for p in points)
    saved = {c["_doc_id"]: c for c in case.snapshot()}
    for p in points:
        assert p.payload["text"] == saved[p.id.replace("-", "")]["text"]
    hits = state.store.search([[0.0, 1.0]], k=10, where={"ds_ids": {"$in": owners}})
    assert {h["_doc_id"] for h in hits} == set(ids)
    assert hits[0]["_doc_id"] == ids[1]
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._config, retriever._store = case.store._config, case.store
    retriever._colbert_enabled, retriever._colbert_model = True, state.store._model
    retriever._colbert_store = state.store
    hydrated = retriever._colbert_retrieve("B", 10, {"ds_ids": {"$in": owners}})
    assert {h["_doc_id"] for h in hydrated} == set(ids)
    for model in ("test-model", state.key):
        assert ChunkDatasources().get_datasource_ids_for_chunks(ids, model) == {
            cid: owners for cid in ids
        }
        for ds in owners:
            row = db.execute(
                "SELECT chunk_count, file_count FROM datasource_embeddings "
                "WHERE datasource_id=%s AND embedding_model=%s",
                (ds, model),
            ).fetchone()
            assert (row["chunk_count"], row["file_count"]) == (3, int(model == "test-model"))
    for ds in owners:
        assert case.hashes.load(ds, "test-model")[case.path] == compute_file_hash(case.path)
        assert (
            db.execute(
                "SELECT file_size FROM file_hashes WHERE datasource_id=%s "
                "AND embedding_model='test-model' AND file_path=%s",
                (ds, case.path),
            ).fetchone()["file_size"]
            == 300
        )
    assert (
        db.execute(
            "SELECT completed_at FROM document_publications WHERE collection_name='kb_test' AND source=%s",
            (case.path,),
        ).fetchone()["completed_at"]
        is not None
    )
    case.reader()


@pytest.mark.parametrize("loss", ["point", "vector"])
@pytest.mark.parametrize("shared", [False, True])
def test_retry_repairs_actual_colbert_vectors(colbert_pipeline, test_db, loss, shared):
    state = colbert_pipeline
    case = state.case
    case.prefix = "Gespeicherter Kontext: "
    assert state.run()["processed_files"] == 1
    owners = [case.ds]
    if shared:
        owners.append(_second_datasource(case, test_db))
        assert state.run(owners[1])["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    _assert_complete(state, test_db, ids, owners)
    if loss == "point":
        state.store.delete_by_ids([ids[1]])
    else:
        case.client.delete_vectors(
            state.store._collection, vectors=[""], points=[ids[1]], wait=True
        )
        assert state.store.existing_ids(ids) == set(ids)
    assert len(state.store.search([[0.0, 1.0]], k=10)) == 2
    _open_retry(state)
    case.prefix = "Darf nicht neu angereichert werden: "
    result = state.run()
    assert result["processed_files"] == 1 and result["errors"] == 0
    _assert_complete(state, test_db, ids, owners)
    assert result["colbert_chunks_written"] == 1
    assert state.calls == ["Gespeicherter Kontext: " + "B" * 100]
    assert result["dedup_chunks"] == 3 and case.embeddings == 3
    case.enrich.assert_not_called()
    assert state.run()["skipped_files"] == 1


def test_backfill_uses_saved_enrichment(colbert_pipeline, test_db):
    state = colbert_pipeline
    case = state.case
    case.prefix = "Gespeicherter Kontext: "
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    state.store.delete_by_ids([ids[1]])
    test_db.execute(
        "DELETE FROM chunk_datasources WHERE chunk_id=%s AND embedding_model=%s",
        (ids[1], state.key),
    )
    test_db.commit()
    _open_retry(state)
    assert state.run()["processed_files"] == 1
    _assert_complete(state, test_db, ids, [case.ds])
    assert state.calls == [case.prefix + "B" * 100]
    case.enrich.assert_not_called()


def test_partial_dense_repair_also_restores_colbert(colbert_pipeline, test_db):
    state = colbert_pipeline
    case = state.case
    case.prefix = "Gespeicherter Kontext: "
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.store.delete_by_ids([ids[0]])
    state.store.delete_by_ids([ids[1]])
    _open_retry(state)
    result = state.run()
    assert result["processed_files"] == 1 and result["errors"] == 0
    assert result["dedup_chunks"] == 2 and case.embeddings == 4
    assert result["colbert_chunks_written"] == 1
    _assert_complete(state, test_db, ids, [case.ds])


@pytest.mark.parametrize(
    "failure",
    [
        "inspect",
        "source_read",
        "stored_incomplete",
        "embed",
        "upsert",
        "membership",
        "payload",
        "verify",
        "lost_write",
        "finalize",
    ],
)
def test_failed_colbert_repair_stays_closed_and_retries(
    colbert_pipeline, test_db, monkeypatch, failure
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    state.store.delete_by_ids(ids)
    _open_retry(state)

    def fail(*args, **kwargs):
        raise RuntimeError("Injected ColBERT repair failure")

    with monkeypatch.context() as fault:
        if failure == "inspect":
            fault.setattr(state.store, "existing_ids", fail)
        elif failure in {"source_read", "stored_incomplete"}:
            documents = case.store.get_documents_by_ids

            def fail_source(*args, **kwargs):
                if kwargs.get("include") == ["documents", "metadatas"]:
                    if failure == "source_read":
                        raise VectorDBConnectionError("Injected stored text read failure")
                    return {"ids": ids, "documents": [], "metadatas": []}
                return documents(*args, **kwargs)

            fault.setattr(case.store, "get_documents_by_ids", fail_source)
        elif failure == "verify":
            existing_ids = state.store.existing_ids
            reads = 0

            def fail_verification(*args, **kwargs):
                nonlocal reads
                reads += 1
                if reads == 2:
                    fail()
                return existing_ids(*args, **kwargs)

            fault.setattr(state.store, "existing_ids", fail_verification)
        elif failure == "embed":
            fault.setattr(colbert, "embed_colbert", fail)
        elif failure == "upsert":
            fault.setattr("kiara.shared.constants.QDRANT_UPSERT_BATCH_SIZE", 1)
            upsert = case.client.upsert
            written = 0

            def partial_upsert(*args, **kwargs):
                nonlocal written
                if kwargs.get("collection_name") == state.store._collection:
                    written += 1
                    if written == 2:
                        fail()
                return upsert(*args, **kwargs)

            fault.setattr(case.client, "upsert", partial_upsert)
        elif failure == "membership":
            add = ChunkDatasources.add_associations

            def fail_membership(self, cids, ds, model, **kwargs):
                if model == state.key:
                    fail()
                return add(self, cids, ds, model, **kwargs)

            fault.setattr(ChunkDatasources, "add_associations", fail_membership)
        elif failure == "payload":
            fault.setattr(state.store, "update_ds_ids", fail)
        elif failure == "lost_write":
            update = state.store.update_ds_ids

            def lose_vector(*args, **kwargs):
                update(*args, **kwargs)
                case.client.delete_vectors(
                    state.store._collection, vectors=[""], points=[ids[1]], wait=True
                )

            fault.setattr(state.store, "update_ds_ids", lose_vector)
        else:
            case.fail_es = True
        if failure == "finalize":
            with pytest.raises(RuntimeError, match="ES publication failed"):
                state.run()
        else:
            result = state.run()
            assert result["errors"] > 0
        assert state.context.stats["processed_files"] == 0
        assert case.path not in case.hashes.load(case.ds, "test-model")
        with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
            case.reader()
        count = test_db.execute(
            "SELECT file_count FROM datasource_embeddings WHERE datasource_id=%s "
            "AND embedding_model='test-model'",
            (case.ds,),
        ).fetchone()
        assert count["file_count"] == 0
    case.fail_es = False
    result = state.run()
    assert result["processed_files"] == 1 and result["errors"] == 0
    _assert_complete(state, test_db, ids, [case.ds])
    assert case.embeddings == 3
    case.enrich.assert_not_called()
