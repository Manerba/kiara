"""P73: real PG/contracts and Qdrant inventories, including unverified models."""

import threading
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_memberships import _second_datasource, _use_es_sdk
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_index_compatibility_s4 import _persist_contract, _resolved

from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.ingestion.cleanup_targets import resolve_source_colbert_keys
from kiara.retrieval import index_compatibility as ic


def _unverify(db, target):
    db.execute(
        "UPDATE index_contracts SET verification_state='unverified', "
        "reason_code='profile_unverified', index_compatibility_id=NULL, "
        "query_compatibility_id=NULL WHERE logical_index_key=%s",
        (target.logical_index_key,),
    )
    db.commit()


def _cleanup(case, datasource_id=None):
    pipeline._cleanup_stale_chunks(
        case.store._config,
        case.store,
        case.hashes,
        ChunkDatasources(),
        case.ds if datasource_id is None else datasource_id,
        "test-model",
        [case.path],
    )


@pytest.mark.parametrize("change", ["replace", "disappear"])
@pytest.mark.parametrize("pg_membership", [False, True])
def test_unrelated_unverified_colbert_does_not_block_dense(
    whole_pipeline, test_db, change, pg_membership
):
    case = whole_pipeline
    assert case.run_whole(case.ds)["processed_files"] == 1
    target = ic.colbert_index_target(case.store._config, "unrelated-historical")
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
    _unverify(test_db, target)
    cid = uuid4().hex
    # Even the current DS is not sufficient evidence: only the source matters.
    case.client.upsert(
        target.collection_name,
        [
            PointStruct(
                id=cid, vector=[[1.0, 0.0]], payload={"source": "/other/file", "ds_ids": [case.ds]}
            )
        ],
    )
    if pg_membership:
        ChunkDatasources().add_associations([cid], case.ds, "colbert::unrelated-historical")
    before = case.client.retrieve(
        target.collection_name, [cid], with_payload=True, with_vectors=True
    )
    if change == "replace":
        Path(case.path).write_text("D" * 300)
    else:
        Path(case.path).unlink()
    result = case.run_whole(case.ds)
    assert result["errors"] == 0
    assert result["processed_files" if change == "replace" else "deleted_files"] == 1
    retry = case.run_whole(case.ds)
    assert retry["errors"] == retry["processed_files"] == retry["deleted_files"] == 0
    assert (
        case.client.retrieve(target.collection_name, [cid], with_payload=True, with_vectors=True)
        == before
    )
    assert (
        test_db.execute(
            "SELECT verification_state FROM index_contracts WHERE logical_index_key=%s",
            (target.logical_index_key,),
        ).fetchone()["verification_state"]
        == "unverified"
    )
    if change == "replace":
        assert case.path in case.hashes.load(case.ds, "test-model")
        assert test_db.execute(
            "SELECT completed_at FROM document_publications WHERE source=%s", (case.path,)
        ).fetchone()["completed_at"]
    else:
        assert not case.hashes.load(case.ds, "test-model")
    assert not publication.pending_source_cleanups("kb_test")


@pytest.mark.parametrize(
    "evidence", ["source", "pg", "point_id", "durable", "es", "es_other_owner"]
)
def test_affected_unverified_colbert_blocks_before_cleanup_mutation(
    colbert_pipeline, test_db, monkeypatch, request, evidence
):
    state, case = colbert_pipeline, colbert_pipeline.case
    cleanup_ds = case.ds
    assert state.run()["processed_files"] == 1
    ids = [chunk["_doc_id"] for chunk in case.snapshot()]
    if evidence == "source":
        case.store.delete_by_ids(ids)
        test_db.execute("DELETE FROM chunk_datasources WHERE embedding_model=%s", (state.key,))
        test_db.commit()
    elif evidence == "pg":
        state.store.delete_by_ids(ids)
    elif evidence == "point_id":
        test_db.execute("DELETE FROM chunk_datasources WHERE embedding_model=%s", (state.key,))
        test_db.commit()
        case.client.set_payload(state.store._collection, {"source": "/historical/path"}, ids)
    elif evidence == "durable":
        publication.invalidate_documents("kb_test", [case.path], "test-model", cleanup=True)
        publication.remember_source_cleanup("kb_test", case.path, {state.key: ids})
        state.store.delete_by_ids(ids)
        case.store.delete_by_ids(ids)
        test_db.execute("DELETE FROM chunk_datasources")
        test_db.commit()
    else:
        _use_es_sdk(case, case.ds, monkeypatch, request)
        state.store.delete_by_ids(ids)
        case.store.delete_by_ids(ids)
        if evidence == "es_other_owner":
            cleanup_ds = _second_datasource(case, test_db)
    _unverify(test_db, state.store._index_target)
    hashes = case.hashes.load(case.ds, "test-model")
    pending = publication.pending_source_cleanups("kb_test")
    owners = test_db.execute("SELECT * FROM chunk_datasources ORDER BY 1,2,3").fetchall()
    before = {
        name: case.client.scroll(name, limit=100, with_payload=True, with_vectors=True)[0]
        for name in ("kb_test", state.store._collection)
    }
    with pytest.raises(ic.IndexCompatibilityError, match="index_contract_unverified"):
        _cleanup(case, cleanup_ds)
    assert case.hashes.load(case.ds, "test-model") == hashes
    assert publication.pending_source_cleanups("kb_test") == pending
    assert test_db.execute("SELECT * FROM chunk_datasources ORDER BY 1,2,3").fetchall() == owners
    assert {
        name: case.client.scroll(name, limit=100, with_payload=True, with_vectors=True)[0]
        for name in before
    } == before


@pytest.mark.parametrize("loss", ["dense", "all_vectors"])
def test_historical_shared_cli_owners_and_durable_retry(
    colbert_pipeline, test_db, monkeypatch, loss
):
    state, case = colbert_pipeline, colbert_pipeline.case
    assert state.run()["processed_files"] == 1
    ids = [chunk["_doc_id"] for chunk in case.snapshot()]
    ds2 = _second_datasource(case, test_db)
    pipeline._ensure_cli_datasource()
    for owner in (0, ds2):
        ChunkDatasources().add_associations(ids, owner, state.key)
        case.hashes.upsert_many(owner, state.key, {case.path: ("old", 300)})
    case.store._config["retrieval"]["colbert"]["enabled"] = False
    case.store.delete_by_ids(ids)
    if loss == "all_vectors":
        publication.invalidate_documents("kb_test", [case.path], "test-model", cleanup=True)
        publication.remember_source_cleanup("kb_test", case.path, {state.key: ids})
        state.store.delete_by_ids(ids)
    delete = case.store.delete_by_sources

    def fail(*args):
        raise RuntimeError("injected last dense cleanup failure")

    with monkeypatch.context() as fault:
        fault.setattr(case.store, "delete_by_sources", fail)
        with pytest.raises(RuntimeError, match="last dense"):
            _cleanup(case)
    assert publication.pending_source_cleanups("kb_test")[case.path][state.key] == sorted(ids)
    assert not state.store.existing_ids(ids)
    assert case.store.delete_by_sources == delete
    _cleanup(case)
    assert not publication.pending_source_cleanups("kb_test")
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not test_db.execute("SELECT * FROM file_hashes").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()


def test_inventory_failure_preserves_hash_anchor_and_can_retry(colbert_pipeline, monkeypatch):
    case, state = colbert_pipeline.case, colbert_pipeline
    assert state.run()["processed_files"] == 1
    hashes = case.hashes.load(case.ds, "test-model")
    snapshot = case.snapshot()
    Path(case.path).unlink()
    with monkeypatch.context() as fault:
        fault.setattr(
            case.client,
            "get_collections",
            lambda: (_ for _ in ()).throw(RuntimeError("inventory unavailable")),
        )
        with pytest.raises(RuntimeError, match="inventory unavailable"):
            state.run()
    assert case.hashes.load(case.ds, "test-model") == hashes
    assert case.snapshot() == snapshot
    assert state.run()["deleted_files"] == 1


def test_inventory_requires_fence_and_never_supplies_semantic_authority(whole_pipeline, test_db):
    case = whole_pipeline
    with pytest.raises(RuntimeError, match="without_fence"):
        resolve_source_colbert_keys(
            case.store._config, case.store, ChunkDatasources(), case.ds, "test-model", [case.path]
        )
    _unverify(test_db, case.store._index_target)
    with ic.source_cleanup_fence():
        assert ic.current_guard_token() is None
        with pytest.raises(ic.IndexCompatibilityError, match="index_contract_unverified"):
            with ic.guard_index(
                case.store._config,
                case.store._index_target,
                operation="test",
                write=True,
                validation_scope="stored_binding",
            ):
                pytest.fail("inventory fence bypassed contract")


@pytest.mark.parametrize("new_model", [False, True])
def test_cleanup_fence_blocks_model_guard_until_scope_is_consumed(
    whole_pipeline, monkeypatch, new_model
):
    case = whole_pipeline
    started, finished = threading.Event(), threading.Event()
    errors = []
    target = case.store._index_target
    if new_model:
        from qdrant_client.http.exceptions import UnexpectedResponse

        target = ic.colbert_index_target(case.store._config, "created-concurrently")
        original = case.client.get_collection

        def collection_or_404(collection_name, **kwargs):
            if not case.client.collection_exists(collection_name):
                raise UnexpectedResponse(404, "Not Found", b"not found", {})
            return original(collection_name, **kwargs)

        monkeypatch.setattr(case.client, "get_collection", collection_or_404)
        monkeypatch.setattr(
            ic, "resolve_current_index_contract", lambda conn, cfg, t, **kw: _resolved(t)
        )

    def writer():
        started.set()
        try:
            with ic.guard_index(
                case.store._config,
                target,
                operation="concurrent_writer",
                write=True,
                allow_initialize=new_model,
                validation_scope="live_profile" if new_model else "stored_binding",
            ):
                if new_model:
                    case.client.create_collection(
                        target.collection_name,
                        vectors_config=VectorParams(
                            size=2,
                            distance=Distance.COSINE,
                            multivector_config=MultiVectorConfig(
                                comparator=MultiVectorComparator.MAX_SIM
                            ),
                        ),
                    )
                finished.set()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=writer)
    with ic.source_cleanup_fence():
        thread.start()
        assert started.wait(5)
        assert not finished.wait(0.2)
        with ic.guard_index(
            case.store._config,
            case.store._index_target,
            operation="cleanup",
            write=True,
            validation_scope="stored_binding",
        ):
            assert not finished.is_set()
    thread.join(5)
    assert not thread.is_alive() and not errors and finished.is_set()


def test_waiting_cleanup_does_not_deadlock_existing_nested_guards(colbert_pipeline, monkeypatch):
    state, case = colbert_pipeline, colbert_pipeline.case
    waiting, finished = threading.Event(), threading.Event()
    errors = []
    sleep = ic.time.sleep

    def signal_wait(seconds):
        waiting.set()
        sleep(seconds)

    monkeypatch.setattr(ic.time, "sleep", signal_wait)

    def cleanup():
        try:
            with ic.source_cleanup_fence():
                finished.set()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=cleanup)
    with ic.guard_index(
        case.store._config,
        case.store._index_target,
        operation="outer",
        write=True,
        validation_scope="stored_binding",
    ):
        thread.start()
        assert waiting.wait(5)
        # The second guard uses another PG connection. It must still obtain
        # its shared config lock while cleanup waits for the first to finish.
        with ic.guard_index(
            case.store._config,
            state.store._index_target,
            operation="inner",
            write=True,
            validation_scope="stored_binding",
        ):
            assert not finished.is_set()
    thread.join(5)
    assert not thread.is_alive() and not errors and finished.is_set()
