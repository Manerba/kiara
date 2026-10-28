"""Ein Hashmoduswechsel darf erst nach Verarbeitung neue Abschluesse schreiben."""

import os
from pathlib import Path

import pytest
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run, read  # noqa: F401

from kiara.exceptions import RetrievalError
from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import ChunkDatasources, HashStore, compute_file_hash


def _run(state, mode):
    case = state.case
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


def _change_file(case):
    stat = os.stat(case.path)
    Path(case.path).write_text("B" * 300)
    # Der Moduswechsel muss auch geaenderte Bytes bei gleicher Groesse/mtime finden.
    os.utime(case.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    case.prefix = "Neu: "


def _assert_completed(state, test_db, mode):
    case = state.case
    chunks = case.snapshot()
    ids = [c["_doc_id"] for c in chunks]
    assert [c["text"] for c in chunks] == ["Neu: " + "B" * 100] * 3
    assert [case.es[cid][0] for cid in ids] == [c["text"] for c in chunks]
    points = case.client.retrieve(state.store._collection, ids, with_vectors=True)
    assert len(points) == 3 and all(p.vector == [[0.0, 1.0]] for p in points)
    hits = state.store.search([[0.0, 1.0]], k=10, where={"ds_ids": {"$in": [case.ds]}})
    assert {h["_doc_id"] for h in hits} == set(ids)
    assert all(h["score"] == pytest.approx(1.0) for h in hits)
    assert case.hashes.load(case.ds, "test-model") == {
        case.path: compute_file_hash(case.path, mode)
    }
    assert case.hashes.get_hash_mode(case.ds, "test-model") == mode
    for model in ("test-model", state.key):
        assert ChunkDatasources().get_datasource_ids_for_chunks(ids, model) == {
            cid: [case.ds] for cid in ids
        }
        row = test_db.execute(
            "SELECT chunk_count, file_count FROM datasource_embeddings "
            "WHERE datasource_id=%s AND embedding_model=%s",
            (case.ds, model),
        ).fetchone()
        assert (row["chunk_count"], row["file_count"]) == (3, int(model == "test-model"))
    reader, ref = case.reader()
    assert read(reader, ref, max_chunks=3).success


@pytest.mark.parametrize("old_mode,new_mode", [("mtime", "content"), ("content", "mtime")])
def test_mode_change_processes_changed_bytes(colbert_pipeline, test_db, old_mode, new_mode):
    state = colbert_pipeline
    case = state.case
    assert _run(state, old_mode)["processed_files"] == 1
    reader, ref = case.reader()
    assert read(reader, ref, max_chunks=1).success
    _change_file(case)
    result = _run(state, new_mode)
    assert result["processed_files"] == 1 and result["errors"] == 0
    _assert_completed(state, test_db, new_mode)
    with pytest.raises(RetrievalError):
        read(reader, ref, start_chunk=1)
    again = _run(state, new_mode)
    assert again["skipped_files"] == 1 and again["processed_files"] == again["errors"] == 0


@pytest.mark.parametrize("old_mode,new_mode", [("mtime", "content"), ("content", "mtime")])
@pytest.mark.parametrize("failure", ["mode_commit", "publication"])
def test_mode_change_failure_keeps_retry_possible(
    colbert_pipeline, test_db, monkeypatch, old_mode, new_mode, failure
):
    state = colbert_pipeline
    case = state.case
    assert _run(state, old_mode)["processed_files"] == 1
    old_hashes = case.hashes.load(case.ds, "test-model")
    _change_file(case)
    migrate = HashStore.migrate_hash_mode

    def interrupt(self, *args, **kwargs):
        migrate(self, *args, **kwargs)
        raise RuntimeError("Interrupted after mode commit")

    with monkeypatch.context() as fault:
        if failure == "mode_commit":
            fault.setattr(HashStore, "migrate_hash_mode", interrupt)
        else:
            case.fail_es = True
        with pytest.raises(RuntimeError, match="mode commit|publication failed"):
            _run(state, new_mode)
    if failure == "mode_commit":
        assert case.hashes.load(case.ds, "test-model") == old_hashes
    else:
        assert case.path not in case.hashes.load(case.ds, "test-model")
        assert state.context.stats["processed_files"] == 0
        with pytest.raises(RetrievalError):
            case.reader()
    case.fail_es = False
    retry = _run(state, new_mode)
    assert retry["processed_files"] == 1 and retry["errors"] == 0
    _assert_completed(state, test_db, new_mode)
    assert _run(state, new_mode)["skipped_files"] == 1


def test_unchanged_file_is_verified_without_new_embeddings(colbert_pipeline):
    state = colbert_pipeline
    case = state.case
    assert _run(state, "mtime")["processed_files"] == 1
    chunks = case.snapshot()
    case.enrich.reset_mock()
    state.calls.clear()
    result = _run(state, "content")
    assert result["processed_files"] == 1 and result["dedup_chunks"] == 3
    assert case.embeddings == 3 and state.calls == []
    case.enrich.assert_not_called()
    assert case.snapshot() == chunks
    assert case.hashes.load(case.ds, "test-model")[case.path] == compute_file_hash(
        case.path, "content"
    )
    assert _run(state, "content")["skipped_files"] == 1
