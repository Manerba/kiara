"""GPU-Admission beim parallelen Docling-/Embedding-Warmup sicher abwarten."""

import threading
import time

import httpx
import pytest

from kiara import backend_warmup as warmup


def _conflict(**body):
    request = httpx.Request("POST", "http://gpu.test/api/embed")
    response = httpx.Response(
        409,
        request=request,
        json={
            "error": "GPU-Operation blockiert",
            "reason": "resource_conflict",
            **body,
        },
    )
    return httpx.HTTPStatusError("GPU belegt", request=request, response=response)


def _task(category="embedding"):
    return warmup.WarmupTask(category, "test", "http://gpu.test", category)


def test_parallel_ddu_can_hold_gpu_longer_than_normal_retry_budget(monkeypatch):
    warmer = warmup.BackendWarmer()
    third_attempt, released = threading.Event(), threading.Event()
    attempts = []

    def embedding(task, timeout):
        attempts.append(timeout)
        if released.is_set():
            return
        if len(attempts) == 3:
            third_attempt.set()
        raise _conflict()

    def ddu(task, timeout):
        assert third_attempt.wait(2), "Embedding gave up while Docling still owned the GPU"
        released.set()

    monkeypatch.setattr(warmup, "WARMUP_RETRY_DELAY", 0.01)
    monkeypatch.setattr(warmer, "_warm_embedding", embedding)
    monkeypatch.setattr(warmer, "_warm_ddu", ddu)
    result = warmer.warm([_task(), _task("ddu")], timeout=2)
    assert result.success and all(detail["ok"] for detail in result.details)
    assert len(attempts) == 4
    assert all(0 < value <= 2 for value in attempts)


def test_busy_gpu_deadline_bounds_requests_and_never_reports_success(monkeypatch):
    now = [0.0]
    timeouts = []
    failure = _conflict()
    monkeypatch.setattr(warmup.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        warmup.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration)
    )

    def busy(task, timeout):
        timeouts.append(timeout)
        raise failure

    with pytest.raises(httpx.HTTPStatusError) as caught:
        warmup.BackendWarmer()._attempt(busy, _task(), 5)
    assert caught.value is failure
    assert now[0] == 5
    assert timeouts == [5, 3, 1]


def test_cancel_stops_gpu_wait_without_further_requests(monkeypatch):
    warmer = warmup.BackendWarmer()
    started = threading.Event()
    attempts = []

    def busy(task, timeout):
        attempts.append(timeout)
        started.set()
        raise _conflict()

    monkeypatch.setattr(warmer, "_warm_embedding", busy)
    monkeypatch.setattr(warmer, "_CONTROL_POLL_INTERVAL", 0.01)
    before = time.monotonic()
    result = warmer.warm(
        [_task()], timeout=30, check_control=lambda: "cancel" if started.is_set() else None
    )
    assert not result.success and time.monotonic() - before < 1
    time.sleep(0.05)
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "body,expected_attempts",
    [
        ({"reason": "other_conflict"}, 2),
        ({"error": "CUDA out of memory"}, 1),
    ],
)
def test_other_conflicts_and_oom_keep_existing_failure_policy(monkeypatch, body, expected_attempts):
    attempts = []
    monkeypatch.setattr(warmup.time, "sleep", lambda _: None)

    def fail(task, timeout):
        attempts.append(timeout)
        raise _conflict(**body)

    with pytest.raises(httpx.HTTPStatusError):
        warmup.BackendWarmer()._attempt(fail, _task(), 60, retries=1)
    assert len(attempts) == expected_attempts
