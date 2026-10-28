"""Tests fuer Multi-Embedding-Funktionen im Backend-Selector.

Testet:
- normalize_model_name: Normalisierung, Edge Cases
- _is_healthy_cached: TTL, Thundering-Herd, non-Ollama
- get_embedding_backend_for_model: Modell-Matching, Health-Filtering
- invalidate_cache: Gezielte Health-Cache-Bereinigung fuer Embedding-URLs
"""

import time
from unittest.mock import patch

import pytest

from kiara.backend_selector import (
    _CHECKING,
    _cache,
    _cache_ts,
    _health_cache,
    _health_last_known,
    _is_healthy_cached,
    _lock,
    _rr_index,
    get_embedding_backend_for_model,
    invalidate_cache,
    normalize_model_name,
)
from kiara.shared.constants import BACKEND_HEALTH_TTL_SECS as _HEALTH_TTL

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_caches():
    """Alle Caches vor/nach jedem Test zuruecksetzen."""
    with _lock:
        _cache.clear()
        _cache_ts.clear()
        _rr_index.clear()
        _health_cache.clear()
        _health_last_known.clear()
    yield
    with _lock:
        _cache.clear()
        _cache_ts.clear()
        _rr_index.clear()
        _health_cache.clear()
        _health_last_known.clear()


# ---------------------------------------------------------------------------
# normalize_model_name
# ---------------------------------------------------------------------------


class TestNormalizeModelName:
    def test_none_returns_empty(self):
        assert normalize_model_name(None) == ""

    def test_empty_string_returns_empty(self):
        assert normalize_model_name("") == ""

    def test_strips_latest_tag(self):
        assert normalize_model_name("nomic-embed-text:latest") == "nomic-embed-text"

    def test_preserves_non_latest_tag(self):
        assert normalize_model_name("nomic-embed-text:v2") == "nomic-embed-text:v2"

    def test_lowercases(self):
        assert normalize_model_name("Nomic-Embed-Text") == "nomic-embed-text"

    def test_lowercase_with_tag(self):
        assert normalize_model_name("Nomic-Embed-Text:V2") == "nomic-embed-text:v2"

    def test_strips_whitespace(self):
        assert normalize_model_name("  nomic-embed-text  ") == "nomic-embed-text"

    def test_strips_whitespace_and_latest(self):
        assert normalize_model_name(" nomic-embed-text:latest ") == "nomic-embed-text"

    def test_only_latest(self):
        """Modellname ':latest' allein -> leerer String nach Entfernung."""
        assert normalize_model_name(":latest") == ""

    def test_plain_model_unchanged(self):
        assert normalize_model_name("nomic-embed-text") == "nomic-embed-text"

    def test_model_with_slashes(self):
        assert normalize_model_name("library/nomic-embed-text:latest") == "library/nomic-embed-text"

    def test_model_with_numbers(self):
        assert normalize_model_name("mxbai-embed-large:335m") == "mxbai-embed-large:335m"

    def test_idempotent(self):
        """Zweifache Normalisierung aendert nichts."""
        name = "Nomic-Embed-Text:latest"
        once = normalize_model_name(name)
        twice = normalize_model_name(once)
        assert once == twice == "nomic-embed-text"

    def test_case_insensitive_latest(self):
        """':LATEST' in Grossbuchstaben wird auch entfernt (nach lowercase)."""
        assert normalize_model_name("model:LATEST") == "model"

    def test_latest_in_middle_not_stripped(self):
        """':latest' nur am Ende entfernt, nicht im Modellnamen."""
        assert normalize_model_name("latest-model:v1") == "latest-model:v1"


# ---------------------------------------------------------------------------
# _is_healthy_cached
# ---------------------------------------------------------------------------


class TestIsHealthyCached:
    def test_non_ollama_always_healthy(self):
        """Nicht-Ollama-Provider gelten immer als erreichbar."""
        assert _is_healthy_cached("http://any-url", provider="openai") is True
        assert _is_healthy_cached("http://any-url", provider="anthropic") is True

    @patch("kiara.backend_selector._do_health_check", return_value=True)
    def test_cache_miss_triggers_check(self, mock_check):
        result = _is_healthy_cached("http://localhost:11434")
        assert result is True
        mock_check.assert_called_once_with("http://localhost:11434", 5.0)

    @patch("kiara.backend_selector._do_health_check", return_value=True)
    def test_cache_hit_no_recheck(self, mock_check):
        """Zweiter Aufruf innerhalb TTL nutzt Cache."""
        _is_healthy_cached("http://localhost:11434")
        _is_healthy_cached("http://localhost:11434")
        assert mock_check.call_count == 1

    @patch("kiara.backend_selector._do_health_check", return_value=False)
    def test_unhealthy_cached(self, mock_check):
        result = _is_healthy_cached("http://localhost:11434")
        assert result is False

    @patch("kiara.backend_selector._do_health_check", return_value=True)
    def test_ttl_expiry_triggers_recheck(self, mock_check):
        """Nach TTL-Ablauf wird erneut gecheckt."""
        _is_healthy_cached("http://localhost:11434")
        # TTL kuenstlich ablaufen lassen
        with _lock:
            _health_cache["http://localhost:11434"] = (True, time.time() - _HEALTH_TTL - 1)
        _is_healthy_cached("http://localhost:11434")
        assert mock_check.call_count == 2

    def test_checking_sentinel_returns_last_known(self):
        """Bei laufendem Check wird der letzte bekannte Wert zurueckgegeben."""
        url = "http://localhost:11434"
        with _lock:
            _health_cache[url] = (_CHECKING, time.time())
            _health_last_known[url] = False
        # Sollte den last_known-Wert (False) zurueckgeben, NICHT True
        result = _is_healthy_cached(url)
        assert result is False

    def test_checking_sentinel_first_time_returns_true(self):
        """Beim allerersten Check (kein last_known) wird True angenommen."""
        url = "http://brand-new-url:11434"
        with _lock:
            _health_cache[url] = (_CHECKING, time.time())
            # Kein Eintrag in _health_last_known
        result = _is_healthy_cached(url)
        assert result is True

    @patch("kiara.backend_selector._do_health_check", return_value=True)
    def test_stale_sentinel_triggers_recheck(self, mock_check):
        """Veralteter Sentinel (Thread abgestuerzt) loest neuen Check aus."""
        url = "http://localhost:11434"
        with _lock:
            # Sentinel aelter als 2x TTL -> "stale"
            _health_cache[url] = (_CHECKING, time.time() - _HEALTH_TTL * 3)
        result = _is_healthy_cached(url)
        assert result is True
        mock_check.assert_called_once()

    @patch("kiara.backend_selector._do_health_check", return_value=False)
    def test_failed_check_marks_unhealthy(self, mock_check):
        """Fehlgeschlagener Health-Check -> False im Cache gespeichert."""
        result = _is_healthy_cached("http://localhost:11434")
        assert result is False
        # Verifizieren dass False im Cache steht
        with _lock:
            cached = _health_cache.get("http://localhost:11434")
            assert cached is not None
            assert cached[0] is False


# ---------------------------------------------------------------------------
# get_embedding_backend_for_model
# ---------------------------------------------------------------------------

_BACKEND_A = {
    "id": 1,
    "name": "Server-A",
    "category": "embedding",
    "provider": "ollama",
    "model": "nomic-embed-text:latest",
    "url": "http://server-a:11434",
    "priority": 10,
}
_BACKEND_B = {
    "id": 2,
    "name": "Server-B",
    "category": "embedding",
    "provider": "ollama",
    "model": "nomic-embed-text:latest",
    "url": "http://server-b:11434",
    "priority": 20,
}
_BACKEND_OTHER = {
    "id": 3,
    "name": "Server-C",
    "category": "embedding",
    "provider": "ollama",
    "model": "mxbai-embed-large",
    "url": "http://server-c:11434",
    "priority": 10,
}


class TestGetEmbeddingBackendForModel:
    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_returns_matching_backend(self, mock_active, mock_health):
        mock_active.return_value = [_BACKEND_A]
        result = get_embedding_backend_for_model("nomic-embed-text")
        assert result == _BACKEND_A

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_model_tag_normalization(self, mock_active, mock_health):
        """'nomic-embed-text:latest' matcht Backend mit 'nomic-embed-text:latest'."""
        mock_active.return_value = [_BACKEND_A]
        result = get_embedding_backend_for_model("nomic-embed-text:latest")
        assert result == _BACKEND_A

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_model_without_tag_matches_latest(self, mock_active, mock_health):
        """'nomic-embed-text' (ohne Tag) matcht Backend 'nomic-embed-text:latest'."""
        mock_active.return_value = [_BACKEND_A]
        result = get_embedding_backend_for_model("nomic-embed-text")
        assert result is not None
        assert result["name"] == "Server-A"

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_no_match_returns_none(self, mock_active, mock_health):
        """Kein Backend fuer unbekanntes Modell -> None."""
        mock_active.return_value = [_BACKEND_A]
        result = get_embedding_backend_for_model("totally-unknown-model")
        assert result is None

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_filters_by_model(self, mock_active, mock_health):
        """Nur Backends mit passendem Modell werden zurueckgegeben."""
        mock_active.return_value = [_BACKEND_A, _BACKEND_OTHER]
        result = get_embedding_backend_for_model("mxbai-embed-large")
        assert result == _BACKEND_OTHER

    @patch("kiara.backend_selector._is_healthy_cached")
    @patch("kiara.backend_selector.get_all_active")
    def test_failover_to_second_backend(self, mock_active, mock_health):
        """Erstes Backend unhealthy -> Fallback auf zweites."""
        mock_active.return_value = [_BACKEND_A, _BACKEND_B]
        mock_health.side_effect = lambda url, **kw: url != "http://server-a:11434"
        result = get_embedding_backend_for_model("nomic-embed-text")
        assert result == _BACKEND_B

    @patch("kiara.backend_selector._is_healthy_cached", return_value=False)
    @patch("kiara.backend_selector.get_all_active")
    def test_all_unhealthy_returns_none(self, mock_active, mock_health):
        """Alle Backends unhealthy -> None."""
        mock_active.return_value = [_BACKEND_A, _BACKEND_B]
        result = get_embedding_backend_for_model("nomic-embed-text")
        assert result is None

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_empty_backends_returns_none(self, mock_active, mock_health):
        mock_active.return_value = []
        result = get_embedding_backend_for_model("nomic-embed-text")
        assert result is None

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_case_insensitive_match(self, mock_active, mock_health):
        """Modellname case-insensitive matchen."""
        mock_active.return_value = [_BACKEND_A]
        result = get_embedding_backend_for_model("Nomic-Embed-Text")
        assert result == _BACKEND_A


# ---------------------------------------------------------------------------
# invalidate_cache — Embedding-URL-Cleanup
# ---------------------------------------------------------------------------


class TestInvalidateCacheEmbedding:
    def test_embedding_invalidation_clears_health_for_urls(self):
        """invalidate_cache('embedding') bereinigt Health-Cache fuer Embedding-URLs."""
        url = "http://server-a:11434"
        with _lock:
            _cache["embedding"] = [{"url": url, "model": "nomic"}]
            _cache_ts["embedding"] = time.time()
            _health_cache[url] = (True, time.time())
            _health_last_known[url] = True

        invalidate_cache("embedding")

        assert url not in _health_cache
        assert url not in _health_last_known

    def test_embedding_invalidation_preserves_other_health(self):
        """invalidate_cache('embedding') laesst Health-Eintraege anderer URLs intakt."""
        emb_url = "http://embed:11434"
        other_url = "http://other:11434"
        with _lock:
            _cache["embedding"] = [{"url": emb_url, "model": "nomic"}]
            _cache_ts["embedding"] = time.time()
            _health_cache[emb_url] = (True, time.time())
            _health_cache[other_url] = (True, time.time())
            _health_last_known[emb_url] = True
            _health_last_known[other_url] = True

        invalidate_cache("embedding")

        assert emb_url not in _health_cache
        assert other_url in _health_cache

    def test_full_invalidation_clears_all_health(self):
        """invalidate_cache(None) bereinigt alle Caches komplett."""
        with _lock:
            _health_cache["url1"] = (True, time.time())
            _health_cache["url2"] = (False, time.time())
            _health_last_known["url1"] = True
            _cache["llm"] = [{"url": "url3"}]

        invalidate_cache(None)

        assert len(_health_cache) == 0
        assert len(_health_last_known) == 0
        assert len(_cache) == 0

    def test_llm_invalidation_does_not_touch_health(self):
        """invalidate_cache('llm') laesst Health-Cache komplett unangetastet."""
        url = "http://llm:11434"
        with _lock:
            _cache["llm"] = [{"url": url}]
            _cache_ts["llm"] = time.time()
            _health_cache[url] = (True, time.time())

        invalidate_cache("llm")

        assert "llm" not in _cache
        assert url in _health_cache
