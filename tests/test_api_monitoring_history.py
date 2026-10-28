"""Verlaufs-API: gemeinsame Cache-Nutzung, Fehler und authentifizierter Stream."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from kiara.monitoring import history
from kiara.monitoring.high_res_history import HighResHistoryService
from kiara.web.api_monitoring import live_history, router
from kiara.web.auth import get_admin_user


@pytest.fixture(autouse=True)
def fresh_history_cache():
    with history._cache_lock:
        history._cache.clear()
    yield
    with history._cache_lock:
        history._cache.clear()


def test_concurrent_browsers_share_one_query_and_receive_copies(monkeypatch):
    barrier = threading.Barrier(6)
    query = Mock(return_value={"series": {"cpu_usage": [42.0]}})
    monkeypatch.setattr(history, "_query_history", query)

    def fetch(_):
        barrier.wait(timeout=5)
        return history.get_history("1h")

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fetch, range(6)))
    assert query.call_count == 1
    results[0]["series"]["cpu_usage"][0] = 99
    assert results[1]["series"]["cpu_usage"] == [42]
    assert history.get_history("1h")["series"]["cpu_usage"] == [42]


def test_expired_cache_refreshes_and_failed_reads_are_not_cached(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(history.time, "monotonic", lambda: now[0])
    query = Mock(side_effect=[{"count": 1}, RuntimeError("DB unavailable"), {"count": 2}])
    monkeypatch.setattr(history, "_query_history", query)
    assert history.get_history("1h") == {"count": 1}
    now[0] += 11
    with pytest.raises(RuntimeError):
        history.get_history("1h")
    assert history.get_history("1h") == {"count": 2}
    assert query.call_count == 3


def test_cache_is_bounded_across_point_limits(monkeypatch):
    monkeypatch.setattr(history, "_query_history", lambda *_: {"count": 0})
    for limit in range(10, 90):
        history.get_history("1h", limit)
    assert len(history._cache) == history._CACHE_CAPACITY


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_admin_user] = lambda: {"is_admin": True}
    with TestClient(app) as client:
        yield client


def test_history_returns_all_curves_in_one_request(client, monkeypatch):
    data = {"timestamps": [100], "series": {key: [None] for key in history._METRIC_SQL}}
    query = Mock(return_value=data)
    monkeypatch.setattr(history, "get_history", query)
    for time_range in history.HISTORY_RANGES:
        response = client.get(f"/api/monitoring/history?range={time_range}")
        assert response.status_code == 200
        assert response.json() == data
        query.assert_called_with(time_range, 500)


@pytest.mark.parametrize(
    "path",
    [
        "/api/monitoring/history?range=1m",
        "/api/monitoring/history?range=1h&max_points=2001",
        "/api/monitoring/live?range=24h",
    ],
)
def test_invalid_history_requests_rejected(client, path):
    assert client.get(path).status_code == 422


def test_both_history_endpoints_require_admin(client):
    def forbidden():
        raise HTTPException(status_code=403)

    client.app.dependency_overrides[get_admin_user] = forbidden
    assert client.get("/api/monitoring/history").status_code == 403
    assert client.get("/api/monitoring/live").status_code == 403


def test_live_sampler_missing_returns_unavailable(client):
    assert client.get("/api/monitoring/live").status_code == 503


@pytest.mark.asyncio
async def test_live_stream_starts_with_buffer_and_resnapshots_after_reconnect(monkeypatch):
    service = HighResHistoryService()
    service._running = True
    for index in range(2500):
        timestamp = index / 4
        service.buffer.append(
            {"timestamp": timestamp, "values": {"net_sent_mb_s": 1}}, monotonic_now=timestamp
        )
    app = FastAPI()
    app.state.monitoring_history = service
    request = Request({"type": "http", "app": app, "headers": []})
    monkeypatch.setattr(history, "get_history", Mock(side_effect=AssertionError("Live used DB")))
    for range_key, count in [("1m", 240), ("10m", 2400), ("1m", 240)]:
        response = await live_history(request, range_key, {})
        events = response.body_iterator
        assert await anext(events) == "retry: 1000\n\n"
        snapshot = json.loads((await anext(events)).split("data: ", 1)[1])
        assert len(snapshot["samples"]) == count
        assert snapshot["interval_ms"] == 250
        assert snapshot["samples"][-1]["sequence"] == 2500
        await events.aclose()


@pytest.mark.asyncio
async def test_shutdown_wakes_waiting_stream_consumers():
    service = HighResHistoryService()
    service._closed = False
    waiting = asyncio.create_task(service.wait_for_next(0))
    await asyncio.sleep(0)
    await service.stop()
    from kiara.monitoring.high_res_history import HighResHistoryClosed

    with pytest.raises(HighResHistoryClosed):
        await asyncio.wait_for(waiting, timeout=1)
