"""Tests fuer Sprint 08: Cache Admin-Funktionen.

Testet cache.stats, .clear — 3 Unit-Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# cache.stats
# ---------------------------------------------------------------------------


class TestCacheStats:
    @patch("kiara.shared.cache.get_query_cache")
    def test_cache_stats(self, mock_cache_fn):
        """Test 19: Cache-Statistiken abrufbar."""
        mock_cache = MagicMock()
        mock_cache.stats.return_value = {"size": 42, "hits": 100, "misses": 10}
        mock_cache_fn.return_value = mock_cache

        from kiara.admin.functions.cache import cache_stats

        result = _run(cache_stats(MOCK_USER))

        assert result.success is True
        assert result.data["size"] == 42
        assert result.data["hits"] == 100


# ---------------------------------------------------------------------------
# cache.clear
# ---------------------------------------------------------------------------


class TestCacheClear:
    @patch("kiara.shared.cache.get_query_cache")
    def test_cache_clear(self, mock_cache_fn):
        """Test 20: Cache leeren mit Confirmation."""
        mock_cache = MagicMock()
        mock_cache_fn.return_value = mock_cache

        from kiara.admin.functions.cache import cache_clear

        result = _run(cache_clear(MOCK_USER))

        assert result.success is True
        assert "geleert" in result.data["message"]
        mock_cache.invalidate.assert_called_once()

    @patch("kiara.shared.cache.get_query_cache")
    def test_cache_clear_idempotent(self, mock_cache_fn):
        """Test 21: Doppeltes Leeren kein Fehler."""
        mock_cache = MagicMock()
        mock_cache_fn.return_value = mock_cache

        from kiara.admin.functions.cache import cache_clear

        result1 = _run(cache_clear(MOCK_USER))
        result2 = _run(cache_clear(MOCK_USER))

        assert result1.success is True
        assert result2.success is True
        assert mock_cache.invalidate.call_count == 2
