"""P64: a file becomes complete only after every required write and recount."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kiara_ingestion_core.models import ProcessingContext

from kiara.ingestion import runtime


class _HardStop(BaseException):
    """Simulate process loss without running Exception cleanup handlers."""


@pytest.fixture
def completion_run(monkeypatch, tmp_path):
    path = str(tmp_path / "document.html")
    events = []
    persisted = {}
    dense = set()
    colbert = set()
    associations = set()
    hash_store = Mock()
    hash_store.upsert_many.side_effect = lambda _ds, _model, hashes: persisted.update(hashes)
    hash_store.remove_files.side_effect = lambda _ds, _model, paths: [
        persisted.pop(item, None) for item in paths
    ]
    chunk_ds = Mock()
    chunk_ds.chunks_exist.side_effect = lambda ids, _model: set(ids) & associations
    store = Mock()
    store.finalize_document_versions.side_effect = lambda ids, *, publish_search, on_complete: (
        on_complete({})
    )

    def stored_documents(ids, *, include):
        found = [cid for cid in ids if cid in dense]
        result = {"ids": found}
        if "documents" in include:
            result["documents"] = [{"a": "first", "b": "second"}[cid] for cid in found]
        if "metadatas" in include:
            result["metadatas"] = [{"source": path} for _ in found]
        return result

    store.get_documents_by_ids.side_effect = stored_documents
    ictx = runtime._IngestionContext(
        config={},
        ctx=ProcessingContext(),
        store=store,
        chunk_ds=chunk_ds,
        hash_store=hash_store,
        batch_size=10,
        datasource_id=1,
        embedding_model="p64-model",
        hash_mode="mtime",
        ds_overrides=None,
        file_hashes={path: ("hash-v1", 100)},
        stats={"processed_files": 0, "total_chunks": 0, "errors": 0},
        pending_hashes={},
        file_completion_callback=Mock(),
        total=1,
        file_log_callback=lambda log: events.append(("file_log", log.status)),
        progress_callback=lambda *args: events.append(("progress", args)),
    )

    def load(*args, **kwargs):
        events.append(("load",))
        return (
            ["first", "second"],
            [{"source": path}] * 2,
            ["a", "b"],
            "first second",
            [(0, 5), (6, 12)],
        )

    def write_dense(texts, metas, ids, *args, **kwargs):
        events.append(("dense", tuple(ids)))
        dense.update(ids)
        return list(ids)

    def write_graph(context, texts, ids):
        events.append(("graph", tuple(ids)))
        associations.update(ids)

    def write_colbert(context, ids, texts, metas):
        events.append(("colbert", tuple(ids)))
        colbert.update(ids)
        return len(ids), "colbert_p64"

    def complete(file_path):
        events.append(("recount",))
        assert dense == colbert == associations == {"a", "b"}
        persisted[file_path] = ictx.file_hashes[file_path]

    # The callback is the pipeline's atomic hash + statistics completion boundary.
    ictx.file_completion_callback = Mock(side_effect=complete)
    monkeypatch.setattr(runtime, "HASH_CHECKPOINT_INTERVAL", 1)
    monkeypatch.setattr(runtime, "_process_single_file", Mock(side_effect=load))
    monkeypatch.setattr(runtime, "_write_embeddings", Mock(side_effect=write_dense))
    monkeypatch.setattr(runtime, "_persist_graph_and_associations", write_graph)
    monkeypatch.setattr(runtime, "_write_to_elasticsearch", lambda *a: events.append(("es",)))
    monkeypatch.setattr(runtime, "_sync_ds_ids", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "_colbert_enabled_for_run", lambda *a: True)
    monkeypatch.setattr(runtime, "_sync_colbert_chunks", Mock(side_effect=write_colbert))
    return SimpleNamespace(
        ictx=ictx,
        path=path,
        events=events,
        persisted=persisted,
        dense=dense,
        colbert=colbert,
        associations=associations,
    )


def _assert_unfinished(run):
    assert run.ictx.stats["processed_files"] == 0
    assert not run.persisted
    assert not run.ictx.pending_hashes
    assert not any(event[0] in {"file_log", "progress"} for event in run.events)


def test_hard_stop_after_dense_before_colbert_cannot_checkpoint_success(
    completion_run, monkeypatch
):
    run = completion_run
    monkeypatch.setattr(
        runtime, "_sync_colbert_chunks", Mock(side_effect=_HardStop("before ColBERT"))
    )
    with pytest.raises(_HardStop, match="before ColBERT"):
        runtime._process_file_in_pipeline(run.ictx, run.path)
    assert run.dense == run.associations == {"a", "b"}
    assert not run.colbert
    _assert_unfinished(run)


@pytest.mark.parametrize("full_dedup", [False, True])
def test_hard_stop_after_writes_before_recount_cannot_report_success(completion_run, full_dedup):
    run = completion_run
    if full_dedup:
        run.dense.update({"a", "b"})
        run.associations.update({"a", "b"})
    run.ictx.file_completion_callback.side_effect = _HardStop("before recount")
    with pytest.raises(_HardStop, match="before recount"):
        runtime._process_file_in_pipeline(run.ictx, run.path)
    assert run.dense == run.colbert == run.associations == {"a", "b"}
    _assert_unfinished(run)


@pytest.mark.parametrize("full_dedup", [False, True])
def test_success_runs_recount_before_hash_and_file_finished(completion_run, full_dedup):
    run = completion_run
    if full_dedup:
        run.dense.update({"a", "b"})
        run.associations.update({"a", "b"})
    assert runtime._process_file_in_pipeline(run.ictx, run.path) is True
    assert run.ictx.stats["processed_files"] == 1
    assert run.persisted == run.ictx.file_hashes
    assert not run.ictx.pending_hashes
    run.ictx.file_completion_callback.assert_called_once_with(run.path)
    kinds = [event[0] for event in run.events]
    assert kinds.index("colbert") < kinds.index("recount") < kinds.index("file_log")
    assert kinds.index("file_log") < kinds.index("progress")


@pytest.mark.parametrize("existing_ids", [set(), {"a"}, {"a", "b"}])
def test_colbert_retry_preserves_all_chunks_until_completion(
    completion_run, monkeypatch, existing_ids
):
    run = completion_run
    run.dense.update(existing_ids)
    run.associations.update(existing_ids)
    sync = runtime._sync_colbert_chunks
    attempts = []

    def retry_then_write(context, ids, texts, metas):
        attempts.append(set(ids))
        _assert_unfinished(run)
        if len(attempts) == 1:
            raise runtime.VectorDBConnectionError("temporary ColBERT outage")
        return sync(context, ids, texts, metas)

    monkeypatch.setattr(runtime, "_sync_colbert_chunks", retry_then_write)

    def retry(context, path, file_log, **kwargs):
        file_log.status = "pending"
        return runtime._CircuitBreakerResult.RETRY, 1

    monkeypatch.setattr(
        runtime,
        "_handle_circuit_breaker",
        retry,
    )
    assert runtime._process_file_in_pipeline(run.ictx, run.path) is True
    assert attempts == [{"a", "b"}, {"a", "b"}]
    run.ictx.file_completion_callback.assert_called_once_with(run.path)
    assert run.ictx.stats["processed_files"] == 1
    assert run.persisted == run.ictx.file_hashes
    assert ("file_log", "success") in run.events


def test_partial_dedup_dense_retry_includes_existing_chunks_in_colbert(completion_run, monkeypatch):
    run = completion_run
    run.dense.add("a")
    run.associations.add("a")
    embed = runtime._handle_embedding_phase
    attempts = []

    def retry_then_embed(context, path, texts, metas, ids, log, retries, started):
        attempts.append(list(ids))
        if len(attempts) == 1:
            return runtime._FileAction.RETRY, retries + 1
        return embed(context, path, texts, metas, ids, log, retries, started)

    monkeypatch.setattr(runtime, "_handle_embedding_phase", retry_then_embed)
    assert runtime._process_file_in_pipeline(run.ictx, run.path) is True
    assert attempts == [["b"], ["b"]]
    assert run.dense == run.colbert == run.associations == {"a", "b"}
    assert [event for event in run.events if event[0] == "colbert"] == [("colbert", ("b", "a"))]
    run.ictx.file_completion_callback.assert_called_once_with(run.path)
    assert run.ictx.stats["processed_files"] == 1
