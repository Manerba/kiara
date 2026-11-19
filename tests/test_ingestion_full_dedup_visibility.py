"""P50 FULL-dedup completion: real PG, Qdrant, scan and hashes; index-aware ES."""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_document_version_completion import document_run  # noqa: F401
from test_elasticsearch_datasource_lifecycle import whole_pipeline, whole_topology  # noqa: F401
from test_ingestion_review_regressions import MESSAGES, es_pipeline  # noqa: F401
from test_ingestion_review_regressions import inject_visibility_failure

from kiara.ingestion import pipeline, runtime
from kiara.ingestion._file_utils import ChunkDatasources, HashStore
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval.index_compatibility import IndexCompatibilityError, dense_index_target

MODEL = "test-model"


@pytest.fixture(params=["mtime", "content"])
def dedup_run(request, es_pipeline, monkeypatch):
    case = es_pipeline
    hash_mode = request.param

    def run(**kwargs):
        return case.run_whole(case.ds, hash_mode=hash_mode, **kwargs)

    first = run()
    assert first["processed_files"] == 2 and first["errors"] == 0
    backend = case.es_backend
    # Interrupted hash checkpoint: realtime chunks and memberships agree in PG,
    # Qdrant and ES, but no hash exists and the ES writes never became visible.
    # The writes happened outside the new run, so its dirty flag cannot provide
    # the missing completion barrier.
    HashStore().remove_files(case.ds, MODEL, case.paths)
    assert HashStore().load(case.ds, MODEL) == {}
    backend.visible[case.index] = {}
    original = copy.deepcopy(backend.docs)
    assert original[case.index]
    backend.actions.clear()
    case.refresh.reset_mock()
    case.inventory.reset_mock()
    case.load.reset_mock()
    embeddings = case.embeddings
    client = es._get_client(case.store._config)
    create = Mock(side_effect=AssertionError("Completion must not create a repair index"))
    monkeypatch.setattr(client.indices, "create", create)
    qdrant_writes = Mock(wraps=case.store.update_ds_ids)
    monkeypatch.setattr(case.store, "update_ds_ids", qdrant_writes)

    yield SimpleNamespace(
        run=run,
        case=case,
        config=case.store._config,
        paths=case.paths,
        load=case.load,
        backend=backend,
        original=original,
    )
    # Matching memberships are not rewritten. Completion republishes only the
    # unchanged canonical snapshot of the existing documents.
    assert not any(a["_op_type"] in ("update", "delete") for a in backend.actions)
    assert all(
        a["_index"] == case.index and a["_id"] in original[case.index] for a in backend.actions
    )
    assert backend.docs.get(case.index) in (None, {}, original[case.index])
    qdrant_writes.assert_not_called()
    assert case.embeddings == embeddings, "Existing chunks must not be re-embedded"
    create.assert_not_called()


CONDITIONS = ["unrefreshed", "missing_index", "empty_replacement", "refresh_error"]
CHUNKS = {"document.txt": 3, "second.txt": 2}


def _chunks(paths):
    return sum(CHUNKS[path.rsplit("/", 1)[1]] for path in paths)


def _inject_condition(run, condition):
    if condition != "unrefreshed":
        inject_visibility_failure(run.case, condition)


def _completion(condition):
    """(refresh, inventory) calls of one completion barrier under condition."""
    return {
        "unrefreshed": (1, 1),
        "missing_index": (0, 0),  # Rejected before refresh: no repair index.
        "empty_replacement": (1, 1),
        "refresh_error": (1, 0),
    }[condition]


def _assert_barriers(run, refreshes, inventories):
    assert run.case.refresh.call_count == refreshes
    assert run.case.inventory.call_count == inventories


@pytest.mark.parametrize("condition", CONDITIONS)
def test_full_dedup_completion_and_unchanged_hash_retry(dedup_run, monkeypatch, condition):
    run = dedup_run
    observed = {}

    def at_completion(*args, stats, **kwargs):
        assert set(HashStore().load(run.case.ds, MODEL)) == set(run.paths)
        assert run.backend.docs == run.original
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
        "dedup_chunks": 5,
        "skipped_files": 0,
        "total_chunks": 0,
        "errors": 0,
    }
    hashes = HashStore().load(run.case.ds, MODEL)
    assert set(hashes) == set(run.paths)
    assert run.load.call_count == 2
    # Preflight barrier (index and data present) plus the completion barrier.
    refreshes, inventories = (1 + n for n in _completion(condition))
    _assert_barriers(run, refreshes, inventories)
    if condition == "unrefreshed":
        assert run.backend.visible == run.backend.docs == run.original

    # Both persisted hashes bypass loading on retry; failure cannot turn into
    # success just because the preceding FULL-dedup run checkpointed them.
    monkeypatch.setattr(pipeline, "_run_colbert_model_switch_cleanup", lambda *a, **k: True)
    if condition == "unrefreshed":
        stats = run.run()
        assert stats["processed_files"] == stats["errors"] == 0
        assert stats["skipped_files"] == 2
        refreshes, inventories = refreshes + 2, inventories + 2
    else:
        # The retry already fails at its preflight barrier.
        with pytest.raises(es.ElasticsearchPartialFailureError, match=MESSAGES[condition]):
            run.run()
        refreshes += _completion(condition)[0]
        inventories += _completion(condition)[1]
    assert HashStore().load(run.case.ds, MODEL) == hashes
    assert run.load.call_count == 2
    _assert_barriers(run, refreshes, inventories)

    if condition == "refresh_error":
        run.case.refresh.side_effect = None
        stats = run.run()
        assert stats["skipped_files"] == 2
        assert stats["processed_files"] == stats["errors"] == 0
        assert run.backend.visible == run.backend.docs == run.original
        assert HashStore().load(run.case.ds, MODEL) == hashes
        assert run.load.call_count == 2
        _assert_barriers(run, refreshes + 2, inventories + 2)


def _guard_error(run):
    return IndexCompatibilityError(
        reason_code="contract_mismatch",
        operation="dedup",
        target=dense_index_target(run.config, MODEL),
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
        accepted.append(run.load.call_args.args[0])

    monkeypatch.setattr(runtime, "_sync_ds_ids", guarded_sync)
    stats = run.run()
    assert stats["index_compatibility_error"] == primary.to_dict()
    assert stats["errors"] == 1
    assert stats["abort_reason"] == (
        f"Index-Kompatibilitaetsguard: {primary.reason_code} ({primary.detail})"
    )
    assert stats["processed_files"] == accepted_files
    assert stats.get("dedup_chunks", 0) == _chunks(accepted)
    assert stats["skipped_files"] == stats["total_chunks"] == 0
    assert set(HashStore().load(run.case.ds, MODEL)) == set(accepted)
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
    # Preflight always; a completion barrier only after actual adoption.
    completion = _completion(condition) if accepted_files else (0, 0)
    _assert_barriers(run, 1 + completion[0], 1 + completion[1])


@pytest.mark.parametrize("condition", CONDITIONS)
def test_exception_after_adoption_preserves_primary_and_completes_visibility(
    dedup_run, monkeypatch, condition
):
    run = dedup_run
    primary = _guard_error(run)

    def fail_after_checkpoint(*args, **kwargs):
        assert set(HashStore().load(run.case.ds, MODEL)) == set(run.paths)
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
        assert run.backend.visible == run.backend.docs == run.original
    refreshes, inventories = (1 + n for n in _completion(condition))
    _assert_barriers(run, refreshes, inventories)


@pytest.mark.parametrize("accepted_files", [0, 1])
def test_cancellation_refreshes_only_after_adoption(dedup_run, accepted_files):
    run = dedup_run
    stats = run.run(
        control_callback=lambda: "cancel" if run.load.call_count == accepted_files else None
    )
    loaded = [call.args[0] for call in run.load.call_args_list]
    assert stats["processed_files"] == accepted_files
    assert stats.get("dedup_chunks", 0) == _chunks(loaded)
    assert run.load.call_count == accepted_files
    assert set(HashStore().load(run.case.ds, MODEL)) == set(loaded)
    completion = _completion("unrefreshed") if accepted_files else (0, 0)
    _assert_barriers(run, 1 + completion[0], 1 + completion[1])
    if accepted_files:
        assert run.backend.visible == run.backend.docs


def test_partial_dedup_retains_barrier_if_embedding_phase_raises(dedup_run, monkeypatch):
    run = dedup_run
    # The interrupted run had stored only two of the first file's three vectors:
    # two chunks are adopted, the third one needs a new embedding.
    stored = run.case.snapshot()
    assert len(stored) == 3
    run.case.store.delete_by_ids([stored[-1]["_doc_id"]])
    primary = RuntimeError("embedding phase failed before new writes")
    monkeypatch.setattr(runtime, "_handle_embedding_phase", Mock(side_effect=primary))
    with pytest.raises(RuntimeError) as raised:
        run.run()
    assert raised.value is primary
    assert run.load.call_count == 1
    assert HashStore().load(run.case.ds, MODEL) == {}
    _assert_barriers(run, 2, 2)
    assert run.backend.visible == run.backend.docs


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
    assert HashStore().load(run.case.ds, MODEL) == {}
    _assert_barriers(run, 0, 0)


def test_full_dedup_and_retry_with_disabled_es(dedup_run, monkeypatch):
    run = dedup_run
    run.config["retrieval"]["hybrid_search"] = "none"
    client = Mock(side_effect=AssertionError("Disabled ES must not contact a backend"))
    monkeypatch.setattr(es, "_get_client", client)
    stats = run.run()
    assert stats["processed_files"] == 2
    assert stats["dedup_chunks"] == 5
    assert stats["errors"] == stats["skipped_files"] == stats["total_chunks"] == 0
    hashes = HashStore().load(run.case.ds, MODEL)
    assert set(hashes) == set(run.paths)
    stats = run.run()
    assert stats["skipped_files"] == 2
    assert stats["processed_files"] == stats["errors"] == 0
    assert run.load.call_count == 2
    assert HashStore().load(run.case.ds, MODEL) == hashes
    client.assert_not_called()
    _assert_barriers(run, 0, 0)


def test_empty_first_run_does_not_contact_missing_es_index(dedup_run, monkeypatch):
    run = dedup_run
    case = run.case
    ids = case.store.get_ids_by_sources(run.paths)
    for path in run.paths:
        HashStore().remove_files(case.ds, MODEL, [path])
        Path(path).unlink()
    ChunkDatasources().remove_datasource(case.ds, MODEL)
    case.store.delete_by_ids(ids)
    _inject_condition(run, "missing_index")
    stats = run.run()
    assert stats["processed_files"] == stats["skipped_files"] == stats["errors"] == 0
    assert HashStore().load(case.ds, MODEL) == {}
    run.load.assert_not_called()
    # Only the read-only inventory: the missing index of an empty source is
    # neither required, refreshed nor recreated.
    assert case.index not in run.backend.docs
    _assert_barriers(run, 0, 0)
