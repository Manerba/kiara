"""Cleanup nach Owner-, Collection- und Punktverlust mit echtem PG/Qdrant."""

from copy import copy
from pathlib import Path

import pytest
from qdrant_client.models import Distance, VectorParams
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_memberships import _second_datasource
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_index_compatibility_s4 import _persist_contract, _resolved

from kiara.graph.store import GraphStore
from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval import colbert
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval import index_compatibility as ic


def test_cli_owner_recreation_preserves_deleted_paths_for_each_model(
    whole_pipeline, test_db, monkeypatch
):
    case = whole_pipeline
    config = case.store._config
    assert case.run_whole(None)["processed_files"] == 1
    other = copy(case.store)
    other._collection, other._embed_model = "kb_other_test", "other-model"
    case.client.create_collection(
        other._collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    target = ic.make_index_target(
        config,
        collection_name=other._collection,
        vector_kind=ic.IndexVectorKind.DENSE,
        requested_model=other._embed_model,
    )
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    other._index_target = target
    other._client = ic.GuardedQdrantClient(case.client, config, target)
    other.add = lambda texts, metas, ids: other.add_with_embeddings(
        texts, metas, ids, [[1.0, 0.0]] * len(ids)
    )
    stores = {"test-model": case.store, "other-model": other}
    monkeypatch.setattr(
        pipeline, "VectorStore", lambda cfg, embedding_model, **kw: stores[embedding_model]
    )
    monkeypatch.setattr(
        ic, "stored_primary_index_target", lambda cfg, model: stores[model]._index_target
    )
    monkeypatch.setattr(
        ic, "resolve_current_index_contract", lambda conn, cfg, t, **kw: _resolved(t)
    )

    def run_other():
        return pipeline.run_ingestion(
            config,
            incremental=True,
            datasource_id=None,
            embedding_model="other-model",
            ds_overrides={"chunk_size": 100, "chunk_overlap": 0, "contextual_retrieval": True},
        )

    assert run_other()["processed_files"] == 1
    assert len(other.get_ids_by_sources([case.path])) == 3
    test_db.execute("DELETE FROM datasources WHERE id=0")
    test_db.commit()
    Path(case.path).unlink()
    result = case.run_whole(None)
    assert result["deleted_files"] == 1 and result["errors"] == 0
    assert not case.snapshot()
    assert len(other.get_ids_by_sources([case.path])) == 3
    assert case.path in case.hashes.load(0, "other-model")
    result = run_other()
    assert result["deleted_files"] == 1 and result["errors"] == 0
    assert not other.get_ids_by_sources([case.path])
    assert not test_db.execute("SELECT * FROM file_hashes").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()
    assert run_other()["deleted_files"] == 0


def _interrupt_source_cleanup(state, monkeypatch):
    case = state.case
    Path(case.path).write_text("")
    # QdrantLocal meldet fehlende Collections als ValueError, der HTTP-Server
    # als 404. Nur diese Transportdarstellung fuer den echten Guard angleichen.
    from qdrant_client.http.exceptions import UnexpectedResponse

    get_collection = case.client.get_collection

    def collection_or_404(collection_name, **kwargs):
        try:
            return get_collection(collection_name, **kwargs)
        except ValueError:
            if case.client.collection_exists(collection_name):
                raise
            raise UnexpectedResponse(404, "Not Found", b"not found", {}) from None

    monkeypatch.setattr(case.client, "get_collection", collection_or_404)

    def fail(*args, **kwargs):
        raise RuntimeError("Injected source PG failure")

    with monkeypatch.context() as fault:
        fault.setattr(GraphStore, "delete_by_chunk_ids", fail)
        with pytest.raises(RuntimeError, match="source PG failure"):
            case.run_whole(case.ds)
    assert not state.store.search([[0.0, 1.0]], k=10)
    saved = publication.pending_source_cleanups("kb_test")[case.path]
    assert len(saved["test-model"]) == len(saved[state.key]) == 3
    case.store._config["retrieval"]["colbert"]["enabled"] = False

    def collection_info(config, model):
        assert model == state.store._model
        if not case.client.collection_exists(state.store._collection):
            return None
        return {
            "name": state.store._collection,
            "chunks": case.client.count(state.store._collection, exact=True).count,
        }

    monkeypatch.setattr(
        colbert.ColBERTStore, "get_colbert_collection_info", collection_info, raising=False
    )
    return saved


def _model_cleanup(state, ds=None):
    return pipeline.cleanup_stale_colbert_models(
        state.case.store._config,
        state.case.ds if ds is None else ds,
        ChunkDatasources(),
        [state.store._model],
    )


@pytest.mark.parametrize("failure", ["counter", "memberships", "inventory"])
def test_model_cleanup_rolls_back_pg_and_contract_then_retries(
    colbert_pipeline, test_db, monkeypatch, failure
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    saved = _interrupt_source_cleanup(state, monkeypatch)
    ids = saved[state.key]
    module, name = {
        "counter": (pipeline, "_delete_colbert_datasource_embedding_counter"),
        "memberships": (pipeline, "_delete_colbert_datasource_entries_with_graph_cleanup"),
        "inventory": (publication, "finish_colbert_model_cleanup"),
    }[failure]
    original = getattr(module, name)

    def fail_after(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Injected model PG failure")

    with monkeypatch.context() as fault:
        fault.setattr(module, name, fail_after)
        _model_cleanup(state)
    assert not case.client.collection_exists(state.store._collection)
    assert len(ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key)) == 3
    assert publication.pending_source_cleanups("kb_test")[case.path] == saved
    assert test_db.execute(
        "SELECT 1 FROM index_contracts WHERE logical_index_key=%s",
        (state.store._index_target.logical_index_key,),
    ).fetchone()
    assert (
        test_db.execute(
            "SELECT chunk_count FROM datasource_embeddings WHERE embedding_model=%s", (state.key,)
        ).fetchone()["chunk_count"]
        == 3
    )
    _model_cleanup(state)
    assert publication.pending_source_cleanups("kb_test")[case.path] == {
        "test-model": saved["test-model"]
    }
    assert not ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key)
    assert not test_db.execute(
        "SELECT 1 FROM index_contracts WHERE logical_index_key=%s",
        (state.store._index_target.logical_index_key,),
    ).fetchone()
    result = case.run_whole(case.ds)
    assert result["errors"] == result["processed_files"] == 0
    assert not case.snapshot() and not publication.pending_source_cleanups("kb_test")
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()


def test_model_cleanup_preserves_other_owner_and_primary_inventory(
    colbert_pipeline, test_db, monkeypatch
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ds2 = _second_datasource(case, test_db)
    assert state.run(ds2)["processed_files"] == 1
    saved = _interrupt_source_cleanup(state, monkeypatch)
    _model_cleanup(state)
    assert case.client.collection_exists(state.store._collection)
    assert publication.pending_source_cleanups("kb_test")[case.path] == saved
    assert ChunkDatasources().get_datasource_ids_for_chunks(saved[state.key], state.key) == {
        cid: [ds2] for cid in saved[state.key]
    }
    _model_cleanup(state, ds2)
    assert not case.client.collection_exists(state.store._collection)
    assert publication.pending_source_cleanups("kb_test")[case.path] == {
        "test-model": saved["test-model"]
    }
    assert case.run_whole(case.ds)["errors"] == 0
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()


@pytest.mark.parametrize("fail_after_ack", [False, True])
def test_collection_only_cleanup_acknowledges_inventory(
    colbert_pipeline, test_db, monkeypatch, fail_after_ack
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    saved = _interrupt_source_cleanup(state, monkeypatch)
    test_db.execute("DELETE FROM chunk_datasources WHERE embedding_model=%s", (state.key,))
    test_db.execute("DELETE FROM datasource_embeddings WHERE embedding_model=%s", (state.key,))
    test_db.commit()
    monkeypatch.setattr("qdrant_client.QdrantClient", lambda **kw: case.client)
    selection = [{"model": state.store._model, "name": state.store._collection}]
    if fail_after_ack:
        acknowledge = publication.finish_colbert_model_cleanup

        def fail(*args, **kwargs):
            acknowledge(*args, **kwargs)
            raise RuntimeError("Injected collection inventory failure")

        with monkeypatch.context() as fault:
            fault.setattr(publication, "finish_colbert_model_cleanup", fail)
            failed = pipeline.cleanup_stale_colbert_collections_without_db(
                case.store._config, selection
            )
        assert failed["errors"] and not failed["deleted"]
        assert not case.client.collection_exists(state.store._collection)
        assert publication.pending_source_cleanups("kb_test")[case.path] == saved
        assert test_db.execute(
            "SELECT 1 FROM index_contracts WHERE logical_index_key=%s",
            (state.store._index_target.logical_index_key,),
        ).fetchone()
    result = pipeline.cleanup_stale_colbert_collections_without_db(case.store._config, selection)
    assert not result["errors"] and result["deleted"] == [state.store._collection]
    assert publication.pending_source_cleanups("kb_test")[case.path] == {
        "test-model": saved["test-model"]
    }
    assert case.run_whole(case.ds)["errors"] == 0
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()


@pytest.mark.parametrize("failure", [None, "inventory_read", "pg", "es_delete"])
def test_es_inventory_recovers_pg_after_both_vector_stores_are_lost(
    colbert_pipeline, test_db, monkeypatch, failure, request
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    from test_document_publication_memberships import _use_es_sdk

    _use_es_sdk(case, case.ds, monkeypatch, request)
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.store.delete_by_ids(ids)
    state.store.delete_by_ids(ids)
    Path(case.path).write_text("")
    assert len(case.es) == 3

    def fail(*args, **kwargs):
        raise RuntimeError("Injected ES inventory boundary failure")

    if failure:
        with monkeypatch.context() as fault:
            if failure == "inventory_read":
                fault.setattr(es, "get_ids_by_source", fail)
            elif failure == "pg":
                fault.setattr(GraphStore, "delete_by_chunk_ids", fail)
            else:
                fault.setattr(es, "delete_by_ids", fail)
            with pytest.raises(RuntimeError, match="ES inventory boundary"):
                case.run_whole(case.ds)
        assert case.path not in case.hashes.load(case.ds, "test-model")
        if failure != "inventory_read":
            saved = publication.pending_source_cleanups("kb_test")[case.path]
            assert saved["test-model"] == saved[state.key] == sorted(ids)
            assert len(saved["es::test-model"]) == len(ids)
            # Nach dem Sichern darf auch der letzte externe Quellenbeleg fehlen.
            case.es.clear()
            for docs in case.es_backend.docs.values():
                docs.clear()
    result = case.run_whole(case.ds)
    assert result["errors"] == result["processed_files"] == 0
    assert not case.es and not case.snapshot() and not state.store.existing_ids(ids)
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()
    assert not test_db.execute("SELECT * FROM file_hashes").fetchall()
    assert publication.pending_source_cleanups("kb_test") == {}
