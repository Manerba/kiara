"""R1/R2: CLI-Owner und Cleanup nach Verlust des Dense-Wiederanlaufankers."""

import asyncio
from pathlib import Path

import pytest
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401

from kiara.exceptions import RetrievalError
from kiara.graph.store import GraphStore
from kiara.ingestion import document_publication as publication
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval.retriever import HybridRetriever
from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool


def _retriever(case):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._config, retriever._store = case.store._config, case.store
    return retriever


def test_cli_owner_has_readable_chunks_hash_and_exact_statistics(whole_pipeline, test_db):
    case = whole_pipeline
    result = case.run_whole(None)
    assert result["processed_files"] == 1 and result["errors"] == 0
    chunks = case.snapshot()
    ids = [c["_doc_id"] for c in chunks]
    assert ChunkDatasources().get_datasource_ids_for_chunks(ids, "test-model") == {
        cid: [0] for cid in ids
    }
    assert all(c["metadata"]["ds_ids"] == [0] for c in chunks)
    assert case.path in case.hashes.load(None, "test-model")
    row = test_db.execute(
        "SELECT chunk_count,file_count FROM datasource_embeddings "
        "WHERE datasource_id=0 AND embedding_model='test-model'"
    ).fetchone()
    assert (row["chunk_count"], row["file_count"]) == (3, 1)
    retriever = _retriever(case)
    available = retriever.filter_available_results(chunks)
    assert len(available) == 3
    assert retriever.filter_available_results(chunks, [case.ds]) == []
    references = DocumentReferences("kb_test")
    annotated = references.annotate(available)
    reader = ReadDocumentTool(retriever, case.store._config, references)
    ref = annotated[0]["metadata"]["document_ref"]
    page = asyncio.run(reader.execute({"document_id": ref, "max_chunks": 3}))
    assert page.success and [c["text"] for c in page.data["results"]] == [c["text"] for c in chunks]
    assert case.run_whole(None)["skipped_files"] == 1
    # Owner 0 ist keine Ausnahme vom Fail-closed-Vertrag.
    test_db.execute("DELETE FROM chunk_datasources WHERE datasource_id=0")
    test_db.commit()
    assert retriever.filter_available_results(chunks) == []
    with pytest.raises(RetrievalError):
        asyncio.run(reader.execute({"document_id": ref}))


def test_cli_owner_creation_reprocesses_old_ownerless_completion(whole_pipeline, test_db):
    case = whole_pipeline
    assert case.run_whole(case.ds)["processed_files"] == 1
    digest = case.hashes.load(case.ds, "test-model")[case.path]
    case.hashes.upsert_many(None, "test-model", {case.path: (digest, 300)})
    test_db.execute("DELETE FROM chunk_datasources")
    test_db.commit()
    assert test_db.execute("SELECT id FROM datasources WHERE id=0").fetchone() is None
    result = case.run_whole(None)
    assert result["processed_files"] == 1 and result["skipped_files"] == 0
    assert len(_retriever(case).filter_available_results(case.snapshot())) == 3


def test_cli_owner_failure_cannot_confirm_hash_then_retry(whole_pipeline, monkeypatch, test_db):
    case = whole_pipeline
    add = ChunkDatasources.add_associations

    def fail(self, ids, ds_id, model, **kwargs):
        if ds_id == 0:
            raise ValueError("Injected owner write failure")
        return add(self, ids, ds_id, model, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(ChunkDatasources, "add_associations", fail)
        result = case.run_whole(None)
    assert result["errors"] > 0 and result["processed_files"] == 0
    assert case.path not in case.hashes.load(None, "test-model")
    assert not test_db.execute(
        "SELECT * FROM datasource_embeddings WHERE datasource_id=0"
    ).fetchall()
    result = case.run_whole(None)
    assert result["processed_files"] == 1 and result["errors"] == 0
    assert len(_retriever(case).filter_available_results(case.snapshot())) == 3


def test_cli_owner_conflict_fails_before_index_write(whole_pipeline, test_db):
    case = whole_pipeline
    test_db.execute("INSERT INTO datasources(id,name) VALUES(0,'Anderer Zweck')")
    test_db.commit()
    with pytest.raises(ValueError, match="anderweitig"):
        case.run_whole(None)
    assert case.snapshot() == []
    assert case.hashes.load(None, "test-model") == {}


@pytest.mark.parametrize("lost", ["middle", "all"])
@pytest.mark.parametrize("failure", [None, "colbert", "graph", "elasticsearch", "dense"])
def test_cleanup_after_dense_loss_survives_each_boundary(
    colbert_pipeline, test_db, monkeypatch, lost, failure
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.store.delete_by_ids(ids if lost == "all" else ids[1:2])
    Path(case.path).write_text("")

    def fail(*args, **kwargs):
        raise RuntimeError("Injected cleanup interruption")

    if failure:
        with monkeypatch.context() as fault:
            if failure == "colbert":
                delete = state.store.delete_by_ids

                def partially_delete(chunk_ids):
                    delete(ids[1:2])
                    fail()

                fault.setattr(state.store, "delete_by_ids", partially_delete)
            elif failure == "graph":
                fault.setattr(GraphStore, "delete_by_chunk_ids", fail)
            elif failure == "elasticsearch":
                fault.setattr("kiara.ingestion.vectordb_ops.delete_from_elasticsearch", fail)
            else:
                fault.setattr(case.store, "delete_by_sources", fail)
            with pytest.raises(RuntimeError, match="cleanup interruption"):
                case.run_whole(None)
        saved = publication.pending_source_cleanups("kb_test")[case.path]
        assert set(saved["test-model"]) == set(ids)
        assert set(saved[state.key]) == set(ids)
        assert case.path not in case.hashes.load(case.ds, "test-model")
        assert case.path not in case.hashes.load(None, "test-model")
    result = case.run_whole(None)
    assert result["errors"] == 0 and result["processed_files"] == 0
    assert not case.snapshot()
    assert not case.client.retrieve(state.store._collection, ids, with_vectors=True)
    for model in ("test-model", state.key):
        assert not ChunkDatasources().get_datasource_ids_for_chunks(ids, model)
        assert not test_db.execute(
            "SELECT * FROM datasource_embeddings WHERE embedding_model=%s", (model,)
        ).fetchall()
    assert publication.pending_source_cleanups("kb_test") == {}
    assert case.run_whole(None)["errors"] == 0


def test_cleanup_retry_finds_deleted_source_after_all_points_and_hashes_are_gone(
    colbert_pipeline, test_db, monkeypatch
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.store.delete_by_ids(ids[1:2])
    Path(case.path).write_text("")

    def fail(*args, **kwargs):
        raise RuntimeError("After final delete")

    with monkeypatch.context() as fault:
        fault.setattr(publication, "finish_source_cleanup", fail)
        with pytest.raises(RuntimeError, match="After final delete"):
            case.run_whole(None)
    assert not case.snapshot()
    assert not case.client.retrieve(state.store._collection, ids, with_vectors=True)
    Path(case.path).unlink()
    assert case.run_whole(None)["errors"] == 0
    assert publication.pending_source_cleanups("kb_test") == {}
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()


def test_cleanup_inventory_includes_lost_dense_point_beyond_scroll_batch(colbert_pipeline, test_db):
    state = colbert_pipeline
    case = state.case
    case.store._config["ingestion"]["batch_size"] = 512
    Path(case.path).write_text("A" * 100100)
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    assert len(ids) == 1001
    assert set(state.store.get_ids_by_source(case.path)) == set(ids)
    case.store.delete_by_ids(ids[-2:])
    Path(case.path).write_text("")
    result = case.run_whole(None)
    assert result["errors"] == 0 and result["processed_files"] == 0
    assert not case.snapshot()
    assert not case.client.retrieve(state.store._collection, ids, with_vectors=True)
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()


def test_unacknowledged_colbert_cleanup_cannot_clear_inventory(colbert_pipeline, monkeypatch):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.store.delete_by_ids(ids[1:2])
    Path(case.path).write_text("")
    with monkeypatch.context() as fault:
        fault.setattr(state.store, "delete_by_ids", lambda ids: None)
        with pytest.raises(RuntimeError, match="ColBERT-Quellen-Cleanup ist unvollstaendig"):
            case.run_whole(None)
    assert publication.pending_source_cleanups("kb_test")
    assert case.path not in case.hashes.load(case.ds, "test-model")
    assert case.run_whole(None)["errors"] == 0
    assert not state.store.existing_ids(ids)


def test_cleanup_inventory_schema_upgrade_is_idempotent(test_db):
    from kiara.web.core_schema_migration import migrate_core_schema

    test_db.execute("ALTER TABLE document_publications DROP COLUMN cleanup_chunks")
    test_db.execute(
        "INSERT INTO document_publications(collection_name,source,generation) "
        "VALUES('kb_test','/source','original')"
    )
    test_db.commit()
    migrate_core_schema(test_db)
    test_db.commit()
    publication.remember_source_cleanup("kb_test", "/source", {"model": ["id"]})
    migrate_core_schema(test_db)
    test_db.commit()
    assert publication.pending_source_cleanups("kb_test") == {"/source": {"model": ["id"]}}
    row = test_db.execute("SELECT generation,completed_at FROM document_publications").fetchone()
    assert row["generation"] == "original" and row["completed_at"] is None
