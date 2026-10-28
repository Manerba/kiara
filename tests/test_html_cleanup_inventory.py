"""HTML-Auswahl darf den Wiederanlauf eines vollstaendigen Cleanups nicht verlieren."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, VectorParams
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_index_compatibility_s4 import _persist_contract, _resolved

from kiara.exceptions import RetrievalError
from kiara.graph.store import GraphStore
from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.ingestion.html_migration import build_selection
from kiara.retrieval import index_compatibility as ic


@pytest.fixture
def html_cleanup(colbert_pipeline, test_db, monkeypatch):
    """Echte Pipeline, PG-Sperren und Qdrant; kontrollierte Modellantworten."""
    state = colbert_pipeline
    case = state.case
    config = case.store._config
    model, collection = "nomic-embed-text", "kb_html_cleanup_test"
    case.client.create_collection(
        collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    target = ic.make_index_target(
        config,
        collection_name=collection,
        vector_kind=ic.IndexVectorKind.DENSE,
        requested_model=model,
    )
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    case.store._collection, case.store._embed_model = collection, model
    case.store._index_target = target
    case.store._client = ic.GuardedQdrantClient(case.client, config, target)
    monkeypatch.setattr(ic, "stored_primary_index_target", lambda *a, **kw: target)

    def run(*, migration=False):
        # Eine .txt-Quelle ist auch fuer die HTML-/Mail-Auswahl zulaessig.
        run_config = dict(config)
        if migration:
            run_config["_html_migration"] = {
                **build_selection(case.ds, {model: [case.path]}),
                "paths": [case.path],
                "errors": {},
            }
        return pipeline.run_ingestion(
            run_config,
            incremental=True,
            datasource_id=case.ds,
            embedding_model=model,
            ds_overrides={
                "chunk_size": 100,
                "chunk_overlap": 0,
                "contextual_retrieval": True,
                "colbert_enabled": True,
            },
        )

    return SimpleNamespace(case=case, colbert=state, model=model, collection=collection, run=run)


def _database_state(db):
    return {
        table: [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()]
        for table in (
            "document_publications",
            "chunk_datasources",
            "file_hashes",
            "datasource_embeddings",
        )
    }


def test_html_migration_preserves_interrupted_colbert_cleanup(html_cleanup, test_db, monkeypatch):
    state = html_cleanup
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    assert len(ids) == 3
    assert len(state.colbert.store.search([[0.0, 1.0]], k=10)) == 3
    case.store.delete_by_ids(ids[1:2])
    Path(case.path).write_text("")

    def fail(*args, **kwargs):
        raise RuntimeError("Injected PG/Graph interruption")

    with monkeypatch.context() as fault:
        fault.setattr(GraphStore, "delete_by_chunk_ids", fail)
        with pytest.raises(RuntimeError, match="PG/Graph interruption"):
            state.run()

    # Die echten externen Deletes sind schon erfolgt, die PG-Transaktion ist zurueckgerollt.
    assert not case.client.retrieve(state.colbert.store._collection, ids, with_vectors=True)
    assert len(case.snapshot()) == 2
    saved = publication.pending_source_cleanups(state.collection)[case.path]
    assert saved == {state.model: sorted(ids), state.colbert.key: sorted(ids)}
    for model in (state.model, state.colbert.key):
        assert ChunkDatasources().get_datasource_ids_for_chunks(ids, model) == {
            cid: [case.ds] for cid in ids
        }
    assert not case.hashes.load(case.ds, state.model)
    before_db, before_dense = _database_state(test_db), case.snapshot()

    migration_error = None
    try:
        state.run(migration=True)
    except RuntimeError as exc:
        migration_error = str(exc)
    after_db, after_dense = _database_state(test_db), case.snapshot()
    after_inventory = publication.pending_source_cleanups(state.collection)
    if migration_error:
        with pytest.raises(RetrievalError):
            publication.assert_documents_published(state.collection, after_dense)

    result = state.run()
    assert result["processed_files"] == 0 and result["errors"] == 0
    assert not case.snapshot()
    assert not case.client.retrieve(state.colbert.store._collection, ids, with_vectors=True)
    assert not state.colbert.store.search([[0.0, 1.0]], k=10)
    assert publication.pending_source_cleanups(state.collection) == {}
    orphaned = test_db.execute("SELECT * FROM chunk_datasources").fetchall()
    assert not orphaned, {"inventory_after_migration": after_inventory, "orphaned": orphaned}
    assert not test_db.execute("SELECT * FROM file_hashes").fetchall()
    assert not test_db.execute("SELECT * FROM datasource_embeddings").fetchall()
    # Nach vollstaendigem Cleanup kann auch die begrenzte Migration wieder abschliessen.
    result = state.run(migration=True)
    assert result["processed_files"] == 0 and result["errors"] == 0
    assert publication.pending_source_cleanups(state.collection) == {}
    assert migration_error and "HTML-Migration" in migration_error
    assert "Cleanup-Inventar" in migration_error
    assert after_db == before_db
    assert after_dense == before_dense


def test_html_migration_can_retry_primary_only_cleanup(html_cleanup, test_db, monkeypatch):
    state = html_cleanup
    case = state.case
    assert state.run(migration=True)["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    assert len(ids) == 3
    assert not state.colbert.store.existing_ids(ids)

    def fail(*args, **kwargs):
        raise RuntimeError("Injected primary cleanup interruption")

    with monkeypatch.context() as fault:
        fault.setattr(case.store, "delete_by_sources", fail)
        with pytest.raises(RuntimeError, match="primary cleanup interruption"):
            state.run(migration=True)
    assert publication.pending_source_cleanups(state.collection)[case.path] == {
        state.model: sorted(ids)
    }
    result = state.run(migration=True)
    assert result["processed_files"] == 1 and result["errors"] == 0
    assert publication.pending_source_cleanups(state.collection) == {}
    assert len(case.snapshot()) == 3 and case.hashes.load(case.ds, state.model)
    row = test_db.execute(
        "SELECT chunk_count, file_count FROM datasource_embeddings "
        "WHERE datasource_id=%s AND embedding_model=%s",
        (case.ds, state.model),
    ).fetchone()
    assert (row["chunk_count"], row["file_count"]) == (3, 1)


def test_html_migration_keeps_unselected_colbert_inventory(html_cleanup):
    state = html_cleanup
    other = state.case.path + ".unselected"
    publication.invalidate_documents(state.collection, [other], state.model)
    publication.remember_source_cleanup(state.collection, other, {state.colbert.key: ["saved-id"]})
    result = state.run(migration=True)
    assert result["processed_files"] == 1 and result["errors"] == 0
    assert publication.pending_source_cleanups(state.collection) == {
        other: {state.colbert.key: ["saved-id"]}
    }
