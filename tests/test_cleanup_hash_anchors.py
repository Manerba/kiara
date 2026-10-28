"""B4/B5: Hashanker bleiben bis zum dauerhaften Quellen-Cleanup erhalten."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run, read  # noqa: F401
from test_html_cleanup_inventory import html_cleanup  # noqa: F401

from kiara.connectors.local_connector import LocalConnector
from kiara.exceptions import RetrievalError
from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import ChunkDatasources, HashStore
from kiara.ingestion.html_reingest import build_request, prepare_reingestion
from kiara.retrieval.retriever import Retriever
from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool


def _database_state(db):
    return {
        table: [dict(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3").fetchall()]
        for table in (
            "document_publications",
            "file_hashes",
            "chunk_datasources",
            "datasource_embeddings",
        )
    }


@contextmanager
def _publication_fault(db, source, boundary):
    name = "hash_anchor_" + uuid4().hex
    db.execute(
        f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'Injected hash anchor transaction failure'; END $$"
    )
    if boundary == "commit":
        trigger = (
            f"CREATE CONSTRAINT TRIGGER {name} AFTER UPDATE ON document_publications "
            "DEFERRABLE INITIALLY DEFERRED"
        )
    else:
        event = "INSERT OR UPDATE" if boundary == "publication" else "UPDATE OF cleanup_chunks"
        trigger = f"CREATE TRIGGER {name} BEFORE {event} ON document_publications"
    db.execute(
        trigger + f" FOR EACH ROW WHEN (NEW.source = %s) EXECUTE FUNCTION {name}()",
        (source,),
    )
    db.commit()
    try:
        yield
    finally:
        db.rollback()
        db.execute(f"DROP TRIGGER {name} ON document_publications")
        db.execute(f"DROP FUNCTION {name}()")
        db.commit()


def _fail(*args, **kwargs):
    raise RuntimeError("Injected interruption before cleanup")


def _assert_removed(case, colbert_store, collection, ids, db):
    assert not case.client.retrieve(collection, ids)
    assert not case.client.retrieve(colbert_store._collection, ids, with_vectors=True)
    assert not colbert_store.search([[0.0, 1.0]], k=10)
    for table in ("file_hashes", "chunk_datasources", "datasource_embeddings"):
        assert not db.execute(f"SELECT 1 FROM {table}").fetchone()
    assert not publication.pending_source_cleanups(collection)


@pytest.fixture
def prepared_html(html_cleanup):
    state = html_cleanup
    case = state.case
    path = Path(case.path).with_suffix(".html")
    path.write_text("<html><body><p>" + "A" * 100 + "B" * 100 + "C" * 100 + "</p></body></html>")
    case.store._config["datasource"]["file_types"] = [".html"]
    assert state.run()["processed_files"] == 1
    data = case.store.get_documents_by_source(str(path))
    assert len(data["ids"]) == 3
    assert all(m["parser"] == "html-to-markdown" for m in data["metadatas"])
    chunks = [
        {"_doc_id": cid, "text": text, "metadata": meta}
        for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"], strict=True)
    ]
    refs = DocumentReferences(state.collection)
    reference = refs.annotate(chunks)[0]["metadata"]["document_ref"]
    retriever = Retriever.__new__(Retriever)
    retriever._store = case.store
    reader = ReadDocumentTool(retriever, case.store._config, refs, [case.ds])
    assert read(reader, reference).success
    config = deepcopy(case.store._config)
    config["_html_reingest"] = build_request("all")
    prepared = prepare_reingestion(
        config,
        LocalConnector,
        {"path": str(path.parent)},
        {
            "id": case.ds,
            "type": "local",
            "path": str(path.parent),
            "chunk_size": 100,
            "chunk_overlap": 0,
        },
        state.model,
        config["_html_reingest"],
        [str(path)],
        case.hashes.load(case.ds, state.model),
        None,
        None,
    )
    assert prepared["paths"] == [str(path)] and not prepared["errors"]
    config["_html_migration"] = {
        **prepared["selection"],
        "paths": prepared["paths"],
        "errors": prepared["errors"],
    }
    config["_html_reingest_prepared_hashes"] = prepared["prepared_hashes"]

    def migrate():
        return pipeline.run_ingestion(
            config,
            incremental=True,
            datasource_id=case.ds,
            embedding_model=state.model,
            ds_overrides={"chunk_size": 100, "chunk_overlap": 0, "contextual_retrieval": True},
        )

    return SimpleNamespace(
        state=state,
        case=case,
        path=path,
        ids=data["ids"],
        reader=reader,
        ref=reference,
        migrate=migrate,
    )


@pytest.mark.parametrize("boundary", ["publication", "inventory", "commit"])
def test_html_precleanup_sql_failure_preserves_hash_and_retries(prepared_html, test_db, boundary):
    h = prepared_html
    original = _database_state(test_db)
    with _publication_fault(test_db, str(h.path), boundary):
        with pytest.raises(Exception, match="hash anchor transaction failure"):
            h.migrate()
        assert _database_state(test_db) == original
        assert len(h.case.client.retrieve(h.state.collection, h.ids)) == 3
        points = h.case.client.retrieve(h.state.colbert.store._collection, h.ids, with_vectors=True)
        assert len(points) == 3 and all(p.vector for p in points)
    h.path.unlink()
    retry, again = h.state.run(), h.state.run()
    assert retry["deleted_files"] == 1 and again["deleted_files"] == 0
    assert retry["errors"] == again["errors"] == retry["processed_files"] == 0
    _assert_removed(h.case, h.state.colbert.store, h.state.collection, h.ids, test_db)
    with pytest.raises(RetrievalError):
        read(h.reader, h.ref)


def test_html_precleanup_committed_intent_blocks_reads_and_retries(
    prepared_html, test_db, monkeypatch
):
    h = prepared_html
    with monkeypatch.context() as fault:
        fault.setattr(pipeline, "_cleanup_source_batch", _fail)
        with pytest.raises(RuntimeError, match="interruption before cleanup"):
            h.migrate()
    assert not h.case.hashes.load(h.case.ds, h.state.model)
    assert publication.pending_source_cleanups(h.state.collection) == {
        str(h.path): {h.state.model: []}
    }
    counter = test_db.execute(
        "SELECT chunk_count, file_count FROM datasource_embeddings WHERE embedding_model=%s",
        (h.state.model,),
    ).fetchone()
    assert (counter["chunk_count"], counter["file_count"]) == (3, 0)
    with pytest.raises(RetrievalError):
        read(h.reader, h.ref)
    h.path.unlink()
    retry, again = h.state.run(), h.state.run()
    assert retry["deleted_files"] == 1 and again["deleted_files"] == 0
    assert retry["errors"] == again["errors"] == 0
    _assert_removed(h.case, h.state.colbert.store, h.state.collection, h.ids, test_db)


@pytest.mark.parametrize("old_mode,new_mode", [("mtime", "content"), ("content", "mtime")])
@pytest.mark.parametrize("boundary", ["migration", "publication", "inventory"])
def test_hash_mode_interruption_keeps_deleted_source_recoverable(
    colbert_pipeline, test_db, monkeypatch, old_mode, new_mode, boundary
):
    state = colbert_pipeline
    case = state.case

    def run(mode):
        return pipeline.run_ingestion(
            case.store._config,
            incremental=True,
            datasource_id=case.ds,
            embedding_model="test-model",
            hash_mode=mode,
            ds_overrides={
                "chunk_size": 100,
                "chunk_overlap": 0,
                "contextual_retrieval": True,
                "colbert_enabled": True,
            },
        )

    assert run(old_mode)["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    original_hashes = case.hashes.load(case.ds, "test-model")
    tool, ref = case.reader()
    assert read(tool, ref).success
    Path(case.path).unlink()
    with monkeypatch.context() as fault:
        if boundary == "migration":
            migrate = HashStore.migrate_hash_mode

            def interrupt(self, *args, **kwargs):
                migrate(self, *args, **kwargs)
                _fail()

            fault.setattr(HashStore, "migrate_hash_mode", interrupt)
        elif boundary == "publication":
            fault.setattr(publication, "invalidate_documents", _fail)
        else:
            fault.setattr(pipeline, "_cleanup_source_batch", _fail)
        with pytest.raises(RuntimeError, match="interruption before cleanup"):
            run(new_mode)
    assert case.hashes.get_hash_mode(case.ds, "test-model") == new_mode
    if boundary == "inventory":
        assert not case.hashes.load(case.ds, "test-model")
        assert publication.pending_source_cleanups("kb_test") == {case.path: {"test-model": []}}
        with pytest.raises(RetrievalError):
            read(tool, ref)
    else:
        assert case.hashes.load(case.ds, "test-model") == original_hashes
        assert not publication.pending_source_cleanups("kb_test")
    for model in ("test-model", state.key):
        assert ChunkDatasources().get_datasource_ids_for_chunks(ids, model) == {
            cid: [case.ds] for cid in ids
        }
    retry, again = run(new_mode), run(new_mode)
    assert retry["deleted_files"] == 1 and again["deleted_files"] == 0
    assert retry["errors"] == again["errors"] == retry["processed_files"] == 0
    _assert_removed(case, state.store, "kb_test", ids, test_db)
    with pytest.raises(RetrievalError):
        read(tool, ref)


@pytest.mark.parametrize("deferred", [False, True])
def test_hash_mode_sql_failure_rolls_back_migration_and_recovers(
    colbert_pipeline, test_db, deferred
):
    state = colbert_pipeline
    case = state.case
    neighbor = Path(case.path).with_name("retained.txt")
    neighbor.write_text("D" * 300)
    assert state.run()["processed_files"] == 2
    missing_ids = case.store.get_ids_by_sources([case.path])
    neighbor_ids = case.store.get_ids_by_sources([str(neighbor)])
    old_hashes = case.hashes.load(case.ds, "test-model")
    original = _database_state(test_db)
    Path(case.path).unlink()
    name = "hash_mode_commit_" + uuid4().hex
    test_db.execute(
        f"CREATE FUNCTION {name}() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RAISE EXCEPTION 'Injected hash mode transaction failure'; END $$"
    )
    trigger = (
        f"CREATE CONSTRAINT TRIGGER {name} AFTER UPDATE ON hash_mode_meta "
        "DEFERRABLE INITIALLY DEFERRED"
        if deferred
        else f"CREATE TRIGGER {name} BEFORE UPDATE ON hash_mode_meta"
    )
    test_db.execute(
        trigger + f" FOR EACH ROW WHEN (NEW.datasource_id = %s) EXECUTE FUNCTION {name}()",
        (case.ds,),
    )
    test_db.commit()

    def run():
        return pipeline.run_ingestion(
            case.store._config,
            incremental=True,
            datasource_id=case.ds,
            embedding_model="test-model",
            hash_mode="content",
            ds_overrides={
                "chunk_size": 100,
                "chunk_overlap": 0,
                "contextual_retrieval": True,
                "colbert_enabled": True,
            },
        )

    try:
        with pytest.raises(Exception, match="hash mode transaction failure"):
            run()
        assert case.hashes.load(case.ds, "test-model") == old_hashes
        assert case.hashes.get_hash_mode(case.ds, "test-model") == "mtime"
        assert _database_state(test_db) == original
        assert (
            len(case.client.retrieve(state.store._collection, missing_ids, with_vectors=True)) == 3
        )
    finally:
        test_db.rollback()
        test_db.execute(f"DROP TRIGGER {name} ON hash_mode_meta")
        test_db.execute(f"DROP FUNCTION {name}()")
        test_db.commit()
    retry, again = run(), run()
    assert retry["errors"] == again["errors"] == 0
    assert retry["deleted_files"] == 1 and again["deleted_files"] == 0
    assert again["processed_files"] == 0 and again["skipped_files"] == 1
    assert not case.client.retrieve("kb_test", missing_ids)
    assert not case.client.retrieve(state.store._collection, missing_ids, with_vectors=True)
    assert len(case.client.retrieve("kb_test", neighbor_ids)) == 3
    assert len(case.client.retrieve(state.store._collection, neighbor_ids, with_vectors=True)) == 3
    assert {hit["_doc_id"] for hit in state.store.search([[1.0, 0.0]], k=10)} == set(neighbor_ids)
    for model in ("test-model", state.key):
        assert not ChunkDatasources().get_datasource_ids_for_chunks(missing_ids, model)
        assert ChunkDatasources().get_datasource_ids_for_chunks(neighbor_ids, model) == {
            cid: [case.ds] for cid in neighbor_ids
        }
        row = test_db.execute(
            "SELECT chunk_count, file_count FROM datasource_embeddings WHERE embedding_model=%s",
            (model,),
        ).fetchone()
        assert (row["chunk_count"], row["file_count"]) == (3, int(model == "test-model"))
    assert set(case.hashes.load(case.ds, "test-model")) == {str(neighbor)}
    assert not publication.pending_source_cleanups("kb_test")
