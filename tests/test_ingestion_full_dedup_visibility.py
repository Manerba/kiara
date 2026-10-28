"""P50 FULL-dedup completion: real PG, scan and hashes; controlled search backends."""

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kiara_ingestion_core.index_state import ChunkDatasources, HashStore
from test_ingestion_membership_visibility import MemoryStore
from test_ingestion_review_regressions import backend  # noqa: F401 -- shared stateful fixture

from kiara.ingestion import pipeline, runtime
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval.index_compatibility import IndexCompatibilityError, dense_index_target


@pytest.fixture(params=["mtime", "content"])
def dedup_run(request, test_db, backend, tmp_path, monkeypatch, s4_guard_passthrough):
    source = tmp_path / "source"
    source.mkdir()
    paths = [str(source / name) for name in ("one.txt", "two.txt")]
    for path in paths:
        Path(path).write_text("A document written before an interrupted hash checkpoint.")
    test_db.execute("INSERT INTO datasources(id,name,type) VALUES (1,'P50-full-dedup','local')")
    test_db.commit()
    config = {
        "datasource": {"path": str(source), "file_types": [".txt"]},
        "vectordb": {"path": str(tmp_path / "vectors")},
        "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
        "retrieval": {"hybrid_search": "elasticsearch"},
        "graph": {"enabled": False},
    }
    store = MemoryStore({path: [1] for path in paths})
    store._index_contract_guard = lambda **kwargs: nullcontext()
    store.add = Mock(side_effect=AssertionError("Existing chunks must not be re-embedded"))
    monkeypatch.setattr(pipeline, "VectorStore", lambda *a, **k: store)
    load = Mock(
        side_effect=lambda path, *a, **kw: (
            ["document text"],
            [{"source": path}],
            [path],
            "document text",
            [(0, 13)],
        )
    )
    monkeypatch.setattr(runtime, "_process_single_file", load)

    # All realtime memberships already agree. ES writes happened outside the
    # new run, so its dirty flag cannot provide the missing completion barrier.
    ChunkDatasources().add_associations(paths, 1, "p50")
    HashStore().set_hash_mode(1, "p50", request.param)
    assert HashStore().load(1, "p50") == {}
    assert (
        es.bulk_index_chunks(
            config, ["document text"] * 2, [{"source": p, "ds_ids": [1]} for p in paths], paths
        )
        == 2
    )
    original = copy.deepcopy(backend.docs)
    assert original and not backend.visible
    backend.actions.clear()
    backend.indices.refresh = Mock(wraps=backend.refresh)
    backend.count = Mock(wraps=backend.count)
    create = Mock(side_effect=AssertionError("Completion must not create a repair index"))
    monkeypatch.setattr(es, "create_index", create)

    def run(**kwargs):
        return pipeline.run_ingestion(
            config,
            incremental=True,
            datasource_id=1,
            embedding_model="p50",
            hash_mode=request.param,
            **kwargs,
        )

    yield SimpleNamespace(
        run=run,
        config=config,
        paths=paths,
        source=source,
        store=store,
        load=load,
        backend=backend,
        original=original,
    )
    assert not backend.actions, "Matching ES memberships must not be rewritten"
    assert not store.writes, "Matching Qdrant memberships must not be rewritten"
    store.add.assert_not_called()
    create.assert_not_called()


CONDITIONS = ["unrefreshed", "missing_index", "empty_replacement", "refresh_error"]
MESSAGES = {
    "missing_index": "Index .* fehlt",
    "empty_replacement": "ohne nachgewiesene Suchdaten",
    "refresh_error": "refresh unavailable",
}


def _inject_condition(run, condition):
    if condition in ("missing_index", "empty_replacement"):
        run.backend.docs.clear()
        run.backend.visible.clear()
        run.backend.indices.exists = Mock(return_value=condition == "empty_replacement")
    elif condition == "refresh_error":
        run.backend.indices.refresh.side_effect = es.ElasticsearchPartialFailureError(
            "refresh unavailable"
        )


def _assert_barriers(run, condition, count=1):
    expected_refreshes = 0 if condition == "missing_index" else count
    assert run.backend.indices.refresh.call_count == expected_refreshes
    expected_counts = count if condition in ("unrefreshed", "empty_replacement") else 0
    assert run.backend.count.call_count == expected_counts
    for call in run.backend.count.call_args_list:
        assert call.kwargs["terminate_after"] == 1
    if condition == "unrefreshed" and count:
        assert run.backend.visible == run.backend.docs == run.original


@pytest.mark.parametrize("condition", CONDITIONS)
def test_full_dedup_completion_and_unchanged_hash_retry(dedup_run, monkeypatch, condition):
    run = dedup_run
    observed = {}

    def at_completion(*args, stats, **kwargs):
        assert set(HashStore().load(1, "p50")) == set(run.paths)
        assert run.backend.docs == run.original
        assert not run.backend.actions and not run.store.writes
        observed.update(stats)
        _inject_condition(run, condition)
        return True

    monkeypatch.setattr(pipeline, "_run_colbert_model_switch_cleanup", at_completion)
    if condition == "unrefreshed":
        run.run()
    else:
        with pytest.raises(es.ElasticsearchPartialFailureError, match=MESSAGES[condition]):
            run.run()
    assert {
        key: observed.get(key)
        for key in ("processed_files", "dedup_chunks", "skipped_files", "total_chunks", "errors")
    } == {
        "processed_files": 2,
        "dedup_chunks": 2,
        "skipped_files": 0,
        "total_chunks": 0,
        "errors": 0,
    }
    hashes = HashStore().load(1, "p50")
    assert run.load.call_count == 2
    _assert_barriers(run, condition)

    # Both persisted hashes bypass loading on retry; failure cannot turn into
    # success just because the preceding FULL-dedup run checkpointed them.
    monkeypatch.setattr(pipeline, "_run_colbert_model_switch_cleanup", lambda *a, **k: True)
    if condition == "unrefreshed":
        stats = run.run()
        assert stats["processed_files"] == stats["errors"] == 0
        assert stats["skipped_files"] == 2
    else:
        with pytest.raises(es.ElasticsearchPartialFailureError, match=MESSAGES[condition]):
            run.run()
    assert HashStore().load(1, "p50") == hashes
    assert run.load.call_count == 2
    _assert_barriers(run, condition, count=2)

    if condition == "refresh_error":
        run.backend.indices.refresh.side_effect = None
        stats = run.run()
        assert stats["skipped_files"] == 2
        assert stats["processed_files"] == stats["errors"] == 0
        assert run.backend.visible == run.backend.docs == run.original
        assert HashStore().load(1, "p50") == hashes
        assert run.load.call_count == 2
        assert run.backend.indices.refresh.call_count == 3


def _guard_error(run):
    return IndexCompatibilityError(
        reason_code="contract_mismatch",
        operation="dedup",
        target=dense_index_target(run.config, "p50"),
        detail="injected guard diagnosis",
    )


@pytest.mark.parametrize("accepted_files", [0, 1])
@pytest.mark.parametrize("condition", CONDITIONS)
def test_guard_abort_refreshes_only_after_actual_adoption(
    dedup_run, monkeypatch, accepted_files, condition
):
    run = dedup_run
    primary = _guard_error(run)
    sync = runtime._sync_ds_ids
    accepted = []

    def guarded_sync(ictx, ids, **kwargs):
        if len(accepted) == accepted_files:
            _inject_condition(run, condition)
            raise primary
        sync(ictx, ids, **kwargs)
        accepted.extend(ids)

    monkeypatch.setattr(runtime, "_sync_ds_ids", guarded_sync)
    stats = run.run()
    assert stats["index_compatibility_error"] == primary.to_dict()
    assert stats["errors"] == 1
    assert stats["abort_reason"] == (
        f"Index-Kompatibilitaetsguard: {primary.reason_code} ({primary.detail})"
    )
    assert stats["processed_files"] == stats.get("dedup_chunks", 0) == accepted_files
    assert stats["skipped_files"] == stats["total_chunks"] == 0
    assert set(HashStore().load(1, "p50")) == set(accepted)
    assert run.load.call_count == accepted_files + 1
    details = stats["error_details"]
    assert details[0]["phase"] == "index_compatibility"
    assert details[0]["diagnostic"] == primary.to_dict()
    if accepted_files and condition != "unrefreshed":
        assert len(details) == 2
        assert details[1]["phase"] == "elasticsearch_refresh"
        assert details[1]["error_type"] == "ElasticsearchPartialFailureError"
    else:
        assert len(details) == 1
    _assert_barriers(run, condition, count=accepted_files)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_exception_after_adoption_preserves_primary_and_completes_visibility(
    dedup_run, monkeypatch, condition
):
    run = dedup_run
    primary = _guard_error(run)

    def fail_after_checkpoint(*args, **kwargs):
        assert set(HashStore().load(1, "p50")) == set(run.paths)
        _inject_condition(run, condition)
        raise primary

    monkeypatch.setattr(pipeline, "_run_colbert_model_switch_cleanup", fail_after_checkpoint)
    with pytest.raises(IndexCompatibilityError) as raised:
        run.run()
    assert raised.value is primary
    assert run.load.call_count == 2
    if condition != "unrefreshed":
        assert len(primary.ingestion_secondary_errors) == 1
        assert primary.ingestion_secondary_errors[0]["phase"] == "elasticsearch_refresh"
        assert (
            primary.ingestion_secondary_errors[0]["error_type"]
            == "ElasticsearchPartialFailureError"
        )
        assert len(primary.__notes__) == 1
    else:
        assert not getattr(primary, "ingestion_secondary_errors", [])
    _assert_barriers(run, condition)


@pytest.mark.parametrize("accepted_files", [0, 1])
def test_cancellation_refreshes_only_after_adoption(dedup_run, accepted_files):
    run = dedup_run
    stats = run.run(
        control_callback=lambda: "cancel" if run.load.call_count == accepted_files else None
    )
    assert stats["processed_files"] == stats.get("dedup_chunks", 0) == accepted_files
    assert run.load.call_count == accepted_files
    assert set(HashStore().load(1, "p50")) == {call.args[0] for call in run.load.call_args_list}
    _assert_barriers(run, "unrefreshed", count=accepted_files)


def test_partial_dedup_retains_barrier_if_embedding_phase_raises(dedup_run, monkeypatch):
    run = dedup_run
    run.load.side_effect = lambda path, *a, **kw: (
        ["document text", "new chunk"],
        [{"source": path}] * 2,
        [path, "new"],
        "document text and new chunk",
        [(0, 13), (14, 26)],
    )
    primary = RuntimeError("embedding phase failed before new writes")
    monkeypatch.setattr(runtime, "_handle_embedding_phase", Mock(side_effect=primary))
    with pytest.raises(RuntimeError) as raised:
        run.run()
    assert raised.value is primary
    assert run.load.call_count == 1
    assert HashStore().load(1, "p50") == {}
    _assert_barriers(run, "unrefreshed")


def test_preflight_guard_does_not_contact_es(dedup_run, monkeypatch):
    run = dedup_run
    primary = _guard_error(run)
    monkeypatch.setattr(pipeline, "VectorStore", Mock(side_effect=primary))
    client = Mock(side_effect=AssertionError("Preflight must not contact ES"))
    monkeypatch.setattr(es, "_get_client", client)
    with pytest.raises(IndexCompatibilityError) as raised:
        run.run()
    assert raised.value is primary
    run.load.assert_not_called()
    client.assert_not_called()
    assert HashStore().load(1, "p50") == {}
    _assert_barriers(run, "unrefreshed", count=0)


def test_full_dedup_and_retry_with_disabled_es(dedup_run, monkeypatch):
    run = dedup_run
    run.config["retrieval"]["hybrid_search"] = "none"
    client = Mock(side_effect=AssertionError("Disabled ES must not contact a backend"))
    monkeypatch.setattr(es, "_get_client", client)
    stats = run.run()
    assert stats["processed_files"] == stats["dedup_chunks"] == 2
    assert stats["errors"] == stats["skipped_files"] == stats["total_chunks"] == 0
    hashes = HashStore().load(1, "p50")
    assert set(hashes) == set(run.paths)
    stats = run.run()
    assert stats["skipped_files"] == 2
    assert stats["processed_files"] == stats["errors"] == 0
    assert run.load.call_count == 2
    assert HashStore().load(1, "p50") == hashes
    client.assert_not_called()
    _assert_barriers(run, "unrefreshed", count=0)


def test_empty_first_run_does_not_contact_missing_es_index(dedup_run, monkeypatch):
    run = dedup_run
    for path in run.source.iterdir():
        path.unlink()
    ChunkDatasources().remove_datasource(1, "p50")
    run.store.mapping.clear()
    _inject_condition(run, "missing_index")
    client = Mock(side_effect=AssertionError("An empty first run needs no ES barrier"))
    monkeypatch.setattr(es, "_get_client", client)
    stats = run.run()
    assert stats["processed_files"] == stats["skipped_files"] == stats["errors"] == 0
    assert HashStore().load(1, "p50") == {}
    run.load.assert_not_called()
    client.assert_not_called()
    _assert_barriers(run, "missing_index", count=0)
