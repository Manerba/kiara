"""N3/B2/B3: dauerhafter Cleanup-Auftrag und atomare aktuelle Modellreparatur."""

from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_memberships import _second_datasource
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_html_cleanup_inventory import html_cleanup  # noqa: F401

from kiara.exceptions import RetrievalError
from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval import colbert
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval import index_compatibility as ic


def _fail(*args, **kwargs):
    raise RuntimeError("Injected inventory read failure")


@pytest.mark.parametrize("failure", ["dense", "elasticsearch", "colbert"])
def test_deleted_sources_survive_first_inventory_failure(
    colbert_pipeline, test_db, monkeypatch, failure, request
):
    state = colbert_pipeline
    case = state.case
    other = Path(case.path).with_name("other.txt")
    other.write_text("D" * 300)
    assert state.run()["processed_files"] == 2
    from test_document_publication_memberships import _use_es_sdk

    _use_es_sdk(case, case.ds, monkeypatch, request)
    ids = case.store.get_ids_by_sources([case.path, str(other)])
    assert len(ids) == 6
    hashes = case.hashes.load(case.ds, "test-model")
    Path(case.path).unlink()
    other.unlink()
    with monkeypatch.context() as fault:
        if failure == "dense":
            fault.setattr(case.store, "get_ids_by_sources", _fail)
        elif failure == "elasticsearch":
            fault.setattr(es, "get_ids_by_source", _fail)
        else:
            fault.setattr(state.store, "get_ids_by_source", _fail)
        with pytest.raises(RuntimeError, match="inventory read failure"):
            state.run()
    saved = publication.pending_source_cleanups("kb_test")
    if failure in ("dense", "elasticsearch"):
        # Target discovery now precedes mutation. A failed preflight preserves
        # the completed hashes as deletion/retry anchors for both missing files.
        assert saved == {}
        assert case.hashes.load(case.ds, "test-model") == hashes
    else:
        assert set(saved) == {case.path, str(other)}
        assert all("test-model" in chunks for chunks in saved.values())
        assert case.hashes.load(case.ds, "test-model") == {}
    for model in ("test-model", state.key):
        assert len(ChunkDatasources().get_datasource_ids_for_chunks(ids, model)) == 6
    # Wiederholter Fehler darf den bereits gesicherten Auftrag nicht entwerten.
    with monkeypatch.context() as fault:
        fault.setattr(es, "get_ids_by_source", _fail)
        with pytest.raises(RuntimeError, match="inventory read failure"):
            state.run()
    assert publication.pending_source_cleanups("kb_test") == saved
    outside = str(Path(case.path).parent.parent / "outside.txt")
    publication.invalidate_documents("kb_test", [outside], "test-model", cleanup=True)
    result = state.run()
    assert result["errors"] == result["processed_files"] == 0
    assert result["deleted_files"] == 2
    assert case.store.get_ids_by_sources([case.path, str(other)]) == []
    assert not state.store.existing_ids(ids)
    assert not test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()
    assert publication.pending_source_cleanups("kb_test") == {outside: {"test-model": []}}
    assert state.run()["deleted_files"] == 0


def test_cleanup_marker_preserves_prior_model_inventory_and_rolls_back_with_hashes(
    colbert_pipeline, test_db
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    publication.remember_source_cleanup("kb_test", case.path, {state.key: ids})
    name = "cleanup_marker_" + uuid4().hex
    test_db.execute(
        f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'Injected cleanup marker failure'; END $$"
    )
    test_db.execute(
        f"CREATE TRIGGER {name} BEFORE UPDATE OF cleanup_chunks ON document_publications "
        "FOR EACH ROW WHEN (NEW.collection_name = 'kb_test') "
        f"EXECUTE FUNCTION {name}()"
    )
    test_db.commit()
    hashes = case.hashes.load(case.ds, "test-model")
    try:
        with pytest.raises(Exception, match="cleanup marker failure"):
            publication.invalidate_documents("kb_test", [case.path], "test-model", cleanup=True)
        assert case.hashes.load(case.ds, "test-model") == hashes
        assert publication.pending_source_cleanups("kb_test") == {
            case.path: {state.key: sorted(ids)}
        }
        assert test_db.execute(
            "SELECT completed_at FROM document_publications WHERE source=%s", (case.path,)
        ).fetchone()["completed_at"]
        assert (
            test_db.execute(
                "SELECT file_count FROM datasource_embeddings WHERE embedding_model='test-model'"
            ).fetchone()["file_count"]
            == 1
        )
    finally:
        test_db.rollback()
        test_db.execute(f"DROP TRIGGER {name} ON document_publications")
        test_db.execute(f"DROP FUNCTION {name}()")
        test_db.commit()
    publication.invalidate_documents("kb_test", [case.path], "test-model", cleanup=True)
    assert publication.pending_source_cleanups("kb_test") == {
        case.path: {"test-model": [], state.key: sorted(ids)}
    }
    assert not case.hashes.load(case.ds, "test-model")
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader()


def test_html_selection_keeps_unselected_empty_cleanup_marker(html_cleanup):
    state = html_cleanup
    other = state.case.path + ".unselected"
    publication.invalidate_documents(state.collection, [other], state.model, cleanup=True)
    assert state.run(migration=True)["processed_files"] == 1
    assert publication.pending_source_cleanups(state.collection) == {other: {state.model: []}}


@pytest.fixture
def current_repair(colbert_pipeline, monkeypatch):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    get_collection = case.client.get_collection

    def collection_or_404(collection_name, **kwargs):
        try:
            return get_collection(collection_name, **kwargs)
        except ValueError:
            if case.client.collection_exists(collection_name):
                raise
            raise UnexpectedResponse(404, "Not Found", b"not found", {}) from None

    # Gleiche fehlende-Collection-Grenze wie HTTP; keine Vertragspruefung ersetzen.
    monkeypatch.setattr(case.client, "get_collection", collection_or_404)
    monkeypatch.setattr(colbert, "QdrantClient", lambda **kw: case.client)
    monkeypatch.setattr(
        colbert.ColBERTStore,
        "get_colbert_collection_info",
        type(state.store).get_colbert_collection_info,
        raising=False,
    )
    return state


def _repair(state, ds=None, *, allow=True):
    return pipeline.cleanup_stale_colbert_models(
        state.case.store._config,
        state.case.ds if ds is None else ds,
        ChunkDatasources(),
        [state.store._model],
        allow_current_models={state.store._model} if allow else None,
    )


def _counter(db, state, ds=None):
    return db.execute(
        "SELECT chunk_count, file_count FROM datasource_embeddings "
        "WHERE datasource_id=%s AND embedding_model=%s",
        (state.case.ds if ds is None else ds, state.key),
    ).fetchone()


def test_missing_current_collection_retires_last_owner_then_can_rebuild(
    current_repair, test_db, monkeypatch
):
    state = current_repair
    case = state.case
    ids = [c["_doc_id"] for c in case.snapshot()]
    ds2 = _second_datasource(case, test_db)
    assert state.run(ds2)["processed_files"] == 1
    dense = case.snapshot()
    publication.remember_source_cleanup("kb_test", case.path, {state.key: ids, "test-model": ids})
    case.client.delete_collection(state.store._collection)
    first = _repair(state)
    assert first["db_only_current_models"] == [state.store._model] and not first["errors"]
    assert not _counter(test_db, state) and _counter(test_db, state, ds2)["chunk_count"] == 3
    assert ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key) == {
        cid: [ds2] for cid in ids
    }
    assert test_db.execute(
        "SELECT 1 FROM index_contracts WHERE logical_index_key=%s",
        (state.store._index_target.logical_index_key,),
    ).fetchone()
    assert state.key in publication.pending_source_cleanups("kb_test")[case.path]
    second = _repair(state, ds2)
    assert second["db_only_current_models"] == [state.store._model] and not second["errors"]
    assert not _counter(test_db, state, ds2)
    assert not test_db.execute(
        "SELECT 1 FROM index_contracts WHERE logical_index_key=%s",
        (state.store._index_target.logical_index_key,),
    ).fetchone()
    assert publication.pending_source_cleanups("kb_test")[case.path] == {"test-model": sorted(ids)}
    assert case.snapshot() == dense
    # Tatsächlicher Neuaufbau mit echtem ColBERT-Konstruktor und Index-Guard.
    # Ein expliziter vollständiger Ingestionlauf folgt der DB-Reparatur.
    monkeypatch.setattr(colbert, "ColBERTStore", type(state.store))
    result = pipeline.run_ingestion(
        case.store._config,
        incremental=False,
        datasource_id=case.ds,
        embedding_model="test-model",
        ds_overrides={
            "chunk_size": 100,
            "chunk_overlap": 0,
            "contextual_retrieval": True,
            "colbert_enabled": True,
        },
    )
    assert result["errors"] == 0 and result["processed_files"] == 1
    assert result["colbert_chunks_written"] == 3
    rebuilt = colbert.ColBERTStore(case.store._config, state.store._model)
    hits = rebuilt.search([[1.0, 0.0]], k=10)
    assert len(hits) == 3 and max(h["score"] for h in hits) == pytest.approx(1.0)
    assert all(p.vector for p in case.client.retrieve(rebuilt._collection, ids, with_vectors=True))
    assert case.hashes.load(case.ds, "test-model")[case.path]
    assert _counter(test_db, state)["chunk_count"] == 3
    assert not publication.pending_source_cleanups("kb_test")
    assert ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key) == {
        cid: [case.ds] for cid in ids
    }


@pytest.mark.parametrize("failure", ["counter", "memberships", "graph", "retire", "inventory"])
def test_current_repair_failure_preserves_counter_memberships_contract_and_inventory(
    current_repair, test_db, monkeypatch, failure
):
    state = current_repair
    case = state.case
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.client.delete_collection(state.store._collection)
    saved = {state.key: sorted(ids), "test-model": sorted(ids)}
    publication.remember_source_cleanup("kb_test", case.path, saved)
    module, name = {
        "counter": (pipeline, "_delete_colbert_datasource_embedding_counter"),
        "memberships": (pipeline, "_delete_colbert_datasource_entries_with_graph_cleanup"),
        "graph": (ChunkDatasources, "filter_unreferenced_chunk_ids"),
        "retire": (ic, "retire_index_contract"),
        "inventory": (publication, "finish_colbert_model_cleanup"),
    }[failure]
    original = getattr(module, name)

    def fail_after(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Injected current repair rollback")

    with monkeypatch.context() as fault:
        fault.setattr(module, name, fail_after)
        result = _repair(state)
    assert result["errors"] and not result["db_only_current_models"]
    assert result["cleanup_invoked"] is False
    assert _counter(test_db, state)["chunk_count"] == 3
    assert ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key) == {
        cid: [case.ds] for cid in ids
    }
    assert publication.pending_source_cleanups("kb_test")[case.path] == saved
    assert test_db.execute(
        "SELECT 1 FROM index_contracts WHERE logical_index_key=%s",
        (state.store._index_target.logical_index_key,),
    ).fetchone()
    assert len(case.snapshot()) == 3
    result = _repair(state)
    assert not result["errors"] and result["db_only_current_models"] == [state.store._model]
    assert not _counter(test_db, state)
    assert not ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key)
    assert publication.pending_source_cleanups("kb_test")[case.path] == {"test-model": sorted(ids)}


@pytest.mark.parametrize("contract", ["missing", "unverified", "wrong_model", "backend_error"])
def test_current_repair_keeps_fail_closed_contract(current_repair, test_db, monkeypatch, contract):
    state = current_repair
    case = state.case
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.client.delete_collection(state.store._collection)
    key = state.store._index_target.logical_index_key
    if contract == "missing":
        test_db.execute("DELETE FROM index_contracts WHERE logical_index_key=%s", (key,))
    elif contract == "unverified":
        test_db.execute(
            "UPDATE index_contracts SET verification_state='unverified', "
            "index_compatibility_id=NULL, query_compatibility_id=NULL WHERE logical_index_key=%s",
            (key,),
        )
    elif contract == "wrong_model":
        test_db.execute(
            "UPDATE index_contracts SET requested_model='other-model' WHERE logical_index_key=%s",
            (key,),
        )
    else:
        monkeypatch.setattr(case.client, "get_collection", _fail)
    test_db.commit()
    if contract == "backend_error":
        outcome = _repair(state)
        assert outcome["errors"] and not outcome["db_only_current_models"]
    else:
        with pytest.raises(ic.IndexCompatibilityError):
            _repair(state)
    assert _counter(test_db, state)["chunk_count"] == 3
    assert len(ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key)) == 3


def test_current_repair_keeps_recreated_vectors_and_requires_explicit_allowance(
    current_repair, test_db
):
    state = current_repair
    case = state.case
    before = case.client.retrieve(
        state.store._collection, [c["_doc_id"] for c in case.snapshot()], with_vectors=True
    )
    result = _repair(state, allow=False)
    assert result["skipped_current_models"] == [state.store._model]
    result = _repair(state)
    assert result["skipped_recreated_current_models"][0]["existing_ids"] == 3
    assert not result["db_only_current_models"] and not result["cleanup_invoked"]
    assert _counter(test_db, state)["chunk_count"] == 3
    assert (
        case.client.retrieve(state.store._collection, [p.id for p in before], with_vectors=True)
        == before
    )
