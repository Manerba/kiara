"""Tests fuer kiara.backend_selector — Failover/Round-Robin + Cache."""

import threading
from unittest.mock import patch

import pytest

from kiara.backend_selector import (
    _cache,
    get_all_active,
    get_next_backend,
    invalidate_cache,
)


def _make_backends(category, count=2):
    """Erzeugt Mock-Backend-Rows als Dicts."""
    return [
        {
            "id": i + 1,
            "category": category,
            "provider": "ollama",
            "name": f"Backend {i + 1}",
            "url": f"http://server{i + 1}:11434",
            "base_url": "",
            "model": "nomic-embed-text" if category == "embedding" else f"model-{i + 1}",
            "api_key": "",
            "is_active": True,
            "priority": (i + 1) * 10,
            "extra_config": "{}",
        }
        for i in range(count)
    ]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Cache vor jedem Test leeren."""
    invalidate_cache()
    yield
    invalidate_cache()


class TestGetAllActive:
    def test_returns_cached_data(self):
        """Zweiter Aufruf nutzt Cache statt DB."""
        backends = _make_backends("embedding")
        with patch("kiara.backend_selector._fetch_backends", return_value=backends) as mock_fetch:
            result1 = get_all_active("embedding")
            result2 = get_all_active("embedding")
            assert result1 == result2
            assert mock_fetch.call_count == 1

    def test_invalidate_clears_cache(self):
        """invalidate_cache() erzwingt neuen DB-Zugriff."""
        backends = _make_backends("llm")
        with patch("kiara.backend_selector._fetch_backends", return_value=backends) as mock_fetch:
            get_all_active("llm")
            invalidate_cache("llm")
            get_all_active("llm")
            assert mock_fetch.call_count == 2

    def test_invalidate_all(self):
        """invalidate_cache(None) leert gesamten Cache."""
        with patch("kiara.backend_selector._fetch_backends", return_value=[]):
            get_all_active("embedding")
            get_all_active("llm")
        invalidate_cache()
        assert "embedding" not in _cache
        assert "llm" not in _cache

    def test_embedding_invalidation_clears_embedding_client_cache(self):
        """Embedding-Backend-CRUD invalidiert auch SDK-/HTTP-Client-Pools."""
        from kiara.retrieval import embedding

        with patch.object(embedding, "invalidate_clients") as mock_invalidate:
            invalidate_cache("embedding")

        mock_invalidate.assert_called_once_with("embedding")

    def test_llm_invalidation_does_not_clear_embedding_client_cache(self):
        from kiara.retrieval import embedding

        with patch.object(embedding, "invalidate_clients") as mock_invalidate:
            invalidate_cache("llm")

        mock_invalidate.assert_not_called()


class TestGetNextBackend:
    def test_embedding_failover(self):
        """Embedding gibt immer erstes Backend (Failover)."""
        backends = _make_backends("embedding", 3)
        with (
            patch("kiara.backend_selector._fetch_backends", return_value=backends),
            patch("kiara.backend_selector._is_healthy_cached", return_value=True),
        ):
            for _ in range(5):
                result = get_next_backend("embedding")
                assert result["id"] == 1

    def test_vlm_failover(self):
        """VLM gibt immer erstes Backend (Failover)."""
        backends = _make_backends("vlm", 2)
        with (
            patch("kiara.backend_selector._fetch_backends", return_value=backends),
            patch("kiara.backend_selector._is_healthy_cached", return_value=True),
        ):
            r1 = get_next_backend("vlm")
            r2 = get_next_backend("vlm")
            assert r1["id"] == r2["id"] == 1

    def test_llm_round_robin(self):
        """LLM rotiert durch Backends (Round-Robin)."""
        backends = _make_backends("llm", 3)
        with patch("kiara.backend_selector._fetch_backends", return_value=backends):
            ids = [get_next_backend("llm")["id"] for _ in range(6)]
            assert ids == [1, 2, 3, 1, 2, 3]

    def test_cloud_round_robin(self):
        """Cloud rotiert durch Backends (Round-Robin)."""
        backends = _make_backends("cloud", 2)
        with patch("kiara.backend_selector._fetch_backends", return_value=backends):
            ids = [get_next_backend("cloud")["id"] for _ in range(4)]
            assert ids == [1, 2, 1, 2]

    def test_empty_returns_none(self):
        """Keine Backends = None."""
        with patch("kiara.backend_selector._fetch_backends", return_value=[]):
            assert get_next_backend("llm") is None

    def test_single_backend_rr(self):
        """Round-Robin mit nur einem Backend = immer dasselbe."""
        backends = _make_backends("llm", 1)
        with patch("kiara.backend_selector._fetch_backends", return_value=backends):
            for _ in range(3):
                assert get_next_backend("llm")["id"] == 1


class TestThreadSafety:
    def test_concurrent_access(self):
        """Paralleler Zugriff loest keine Race-Conditions aus."""
        backends = _make_backends("llm", 3)
        results = []
        errors = []

        with patch("kiara.backend_selector._fetch_backends", return_value=backends):

            def worker():
                try:
                    for _ in range(50):
                        b = get_next_backend("llm")
                        if b:
                            results.append(b["id"])
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        assert len(results) == 200
        assert all(r in (1, 2, 3) for r in results)
