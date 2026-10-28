"""Concurrency and completion contracts for coalesced ingestion caches."""

import threading
from unittest.mock import Mock

import pytest

from kiara.web import cache_refresh
from kiara.web.ingestion_worker import IngestionWorker


def test_progress_coalesces_while_refresh_is_blocked(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def refresh(ds_id):
        calls.append(ds_id)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(cache_refresh, "refresh_datasource_cache", refresh)
    monkeypatch.setattr(cache_refresh, "refresh_jobs_cache", Mock())
    cache = cache_refresh.IngestionCacheRefresh(7, 9, interval=3)
    try:
        cache.request()
        assert entered.wait(2)
        thread = cache._thread
        for _ in range(10_000):
            cache.request()
        assert cache._thread is thread
        assert calls == [7]
        # Completion must drain the in-flight publication before final state.
        closed = threading.Event()
        closer = threading.Thread(target=lambda: (cache.close(), closed.set()))
        closer.start()
        assert not closed.wait(0.02)
        release.set()
        closer.join(2)
        assert closed.is_set()
        assert not thread.is_alive()
        assert calls == [7]  # pending intermediate refresh was coalesced away
    finally:
        release.set()
        cache.close()


def test_refresh_failure_retries_without_new_progress(monkeypatch):
    retried = threading.Event()
    attempts = []

    def refresh(_):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("transient cache failure")
        retried.set()

    monkeypatch.setattr(cache_refresh, "refresh_datasource_cache", refresh)
    monkeypatch.setattr(cache_refresh, "refresh_jobs_cache", Mock())
    cache = cache_refresh.IngestionCacheRefresh(7, 9, interval=0.01)
    try:
        cache.request()
        assert retried.wait(2)
    finally:
        cache.close()
    assert len(attempts) == 2


def test_cache_contract_failure_is_propagated_on_completion(monkeypatch):
    from kiara.retrieval.index_compatibility import IndexCompatibilityError, dense_index_target

    error = IndexCompatibilityError(
        "index_contract_missing", "cache", dense_index_target({}, "test"), "injected"
    )
    monkeypatch.setattr(cache_refresh, "refresh_jobs_cache", Mock(side_effect=error))
    cache = cache_refresh.IngestionCacheRefresh(7, 9, interval=3)
    cache.request()
    cache._thread.join(2)
    with pytest.raises(IndexCompatibilityError) as caught:
        cache.close()
    assert caught.value is error


@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
def test_finalizers_drain_cache_before_status_write(monkeypatch, outcome):
    worker = IngestionWorker()
    events = []
    worker._progress_cache = Mock()
    worker._progress_cache.close.side_effect = lambda: events.append("drained")

    def stop_at_database(*args, **kwargs):
        events.append("finalize")
        raise RuntimeError("stop before DB mutation")

    monkeypatch.setattr("kiara.web.ingestion_worker.db_transaction", stop_at_database)
    monkeypatch.setattr(worker, "_aggregate_final_stats", stop_at_database)
    with pytest.raises(RuntimeError, match="stop before DB mutation"):
        if outcome == "success":
            worker._finalize_success(9, 7, {}, {}, "test")
        elif outcome == "error":
            worker._finalize_error(9, 7, "failed")
        else:
            worker._finalize_cancelled(9, 7, {})
    assert events == ["drained", "finalize"]
    assert worker._progress_cache is None


def test_worker_progress_keeps_immediate_db_and_activity(monkeypatch):
    from contextlib import contextmanager

    db = Mock()

    @contextmanager
    def transaction():
        yield db

    monkeypatch.setattr("kiara.web.ingestion_worker.db_transaction", transaction)
    expensive = Mock(side_effect=AssertionError("aggregation on file thread"))
    monkeypatch.setattr(cache_refresh, "refresh_datasource_cache", expensive)
    monkeypatch.setattr(cache_refresh, "refresh_jobs_cache", expensive)
    worker = IngestionWorker()
    worker._activity_tracker = Mock()
    worker._progress_cache = Mock()
    worker._build_callbacks(9, 7)["on_progress"](17, 20)
    assert db.execute.call_count == 2
    assert db.execute.call_args_list[0].args[1] == (17, 20, 9)
    worker._activity_tracker.record.assert_called_once_with(9, "file_progress")
    worker._progress_cache.request.assert_called_once_with()
