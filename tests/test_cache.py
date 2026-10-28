"""Tests fuer kiara.cache.query_cache.QueryCache."""

import json
import threading
import time
from unittest.mock import patch

import pytest

from kiara.cache.query_cache import QueryCache

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")


def _make_config(**overrides):
    """Erzeugt eine minimale Config mit optionalen Cache-Overrides."""
    cache_defaults = {"enabled": True, "max_size": 1000, "ttl_seconds": 3600}
    cache_defaults.update(overrides)
    return {"cache": cache_defaults}


def _query_profile(**overrides):
    profile = {
        "canonical_model_id": "registry.example/dense@sha256:canonical",
        "profile_id": "dense-v1",
        "profile_kind": "dense",
        "endpoint": "/api/embed",
        "input_type": "search_query",
        "query_compatibility_id": "sha256:" + "2" * 64,
        "logical_index_key": "qdrant:http://qdrant.test:6333:kb_dense:dense:model",
        "fence_version": 7,
    }
    profile.update(overrides)
    return (profile,)


class TestQueryCache:
    def test_v2_key_preimage_binds_query_contract_and_normalized_input(self):
        cache = QueryCache(_make_config())

        payload = json.loads(
            cache._key_preimage(
                "  Cafe\u0301\r\nFirewall  Regeln  ",
                _query_profile(),
                datasource_ids={2, 1},
            )
        )

        assert payload == {
            "schema": "kiara.query_embedding_cache_key",
            "schema_version": 2,
            "operation": "search_query",
            "profiles": [
                {
                    "canonical_model_id": "registry.example/dense@sha256:canonical",
                    "endpoint": "/api/embed",
                    "fence_version": 7,
                    "input_type": "search_query",
                    "logical_index_key": ("qdrant:http://qdrant.test:6333:kb_dense:dense:model"),
                    "profile_id": "dense-v1",
                    "profile_kind": "dense",
                    "query_compatibility_id": "sha256:" + "2" * 64,
                }
            ],
            "input": {
                "query": "Caf\u00e9\nFirewall  Regeln",
                "filters": {"datasource_ids": [1, 2]},
            },
        }

    @pytest.mark.parametrize(
        ("changed_field", "changed_value"),
        [
            ("canonical_model_id", "registry.example/dense@sha256:other"),
            ("profile_id", "dense-v2"),
            ("query_compatibility_id", "sha256:" + "9" * 64),
            ("input_type", "search_document"),
        ],
    )
    def test_v2_key_changes_for_each_required_contract_identity(
        self,
        changed_field,
        changed_value,
    ):
        cache = QueryCache(_make_config())
        baseline = cache._make_key("Suche", _query_profile())

        changed = cache._make_key(
            "Suche",
            _query_profile(**{changed_field: changed_value}),
        )

        assert changed != baseline

    def test_v2_key_query_normalization_matches_embedding_input(self):
        cache = QueryCache(_make_config())

        composed = cache._make_key("  Caf\u00e9\r\nRegel  ", _query_profile())
        decomposed = cache._make_key("Cafe\u0301\nRegel", _query_profile())
        meaningfully_different = cache._make_key("Caf\u00e9\nRegel  zwei", _query_profile())

        assert composed == decomposed
        assert meaningfully_different != composed

    def test_legacy_keyspace_is_cleared_before_lookup(self):
        cache = QueryCache(_make_config())
        legacy_key = "legacy-sha256-query-plus-filters"
        cache._cache[legacy_key] = (time.monotonic(), [{"legacy": True}])

        with patch.object(
            cache,
            "_get_by_key_locked",
            wraps=cache._get_by_key_locked,
        ) as lookup:
            assert cache.get("Suche") is None

        looked_up_key = lookup.call_args.args[0]
        assert looked_up_key != legacy_key
        assert legacy_key not in cache._cache
        assert cache.stats()["legacy_entries_invalidated"] == 1

    def test_cache_put_and_get(self):
        """put() speichert Ergebnis, get() liefert es zurueck."""
        cache = QueryCache(_make_config())
        results = [{"doc": "hello", "score": 0.9}]
        cache.put("test query", results)

        got = cache.get("test query")
        assert got is not None
        assert got == results

    def test_cache_miss(self):
        """get() liefert None bei nicht vorhandenem Key."""
        cache = QueryCache(_make_config())

        got = cache.get("nonexistent query")
        assert got is None

    def test_cache_disabled(self):
        """Bei enabled=False wird nie gecached."""
        cache = QueryCache(_make_config(enabled=False))
        results = [{"doc": "hello"}]
        cache.put("test query", results)

        got = cache.get("test query")
        assert got is None

    def test_cache_ttl_expiry(self):
        """Eintraege werden nach TTL-Ablauf als Miss behandelt."""
        cache = QueryCache(_make_config(ttl_seconds=0))
        results = [{"doc": "hello"}]
        cache.put("test query", results)

        # TTL=0 bedeutet sofort abgelaufen; warte kurz fuer time.time()-Differenz
        time.sleep(0.01)

        got = cache.get("test query")
        assert got is None

    def test_cache_lru_eviction(self):
        """Bei max_size=2 wird der aelteste Eintrag verdraengt."""
        cache = QueryCache(_make_config(max_size=2))
        cache.put("query1", [{"doc": "a"}])
        cache.put("query2", [{"doc": "b"}])
        cache.put("query3", [{"doc": "c"}])

        # query1 sollte verdraengt sein
        assert cache.get("query1") is None
        assert cache.get("query2") is not None
        assert cache.get("query3") is not None

    def test_cache_invalidate(self):
        """invalidate() leert den gesamten Cache."""
        cache = QueryCache(_make_config())
        cache.put("query1", [{"doc": "a"}])
        cache.put("query2", [{"doc": "b"}])

        cache.invalidate()

        assert cache.get("query1") is None
        assert cache.get("query2") is None

    def test_cache_stats(self):
        """stats() liefert korrekte Hit/Miss-Zaehler und Hit-Rate."""
        cache = QueryCache(_make_config())
        cache.put("query1", [{"doc": "a"}])

        # 1 hit
        cache.get("query1")
        # 1 miss
        cache.get("nonexistent")

        stats = cache.stats()
        assert stats["enabled"] is True
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_key_includes_filters(self):
        """Gleiche Query mit unterschiedlichen Filtern ergibt unterschiedliche Keys."""
        cache = QueryCache(_make_config())
        results_a = [{"doc": "filtered_a"}]
        results_b = [{"doc": "filtered_b"}]

        cache.put("test query", results_a, directory="/docs")
        cache.put("test query", results_b, directory="/code")

        got_a = cache.get("test query", directory="/docs")
        got_b = cache.get("test query", directory="/code")

        assert got_a == results_a
        assert got_b == results_b

    def test_cache_stats_zero_requests(self):
        """stats() liefert hit_rate=0.0 wenn keine Requests."""
        cache = QueryCache(_make_config())
        stats = cache.stats()
        assert stats["hit_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_lru_access_refreshes(self):
        """Zugriff auf Eintrag verschiebt ihn ans Ende (LRU-Refresh)."""
        cache = QueryCache(_make_config(max_size=2))
        cache.put("query1", [{"doc": "a"}])
        cache.put("query2", [{"doc": "b"}])

        # Access query1 to refresh it
        cache.get("query1")

        # Add query3 -> query2 should be evicted (oldest unused)
        cache.put("query3", [{"doc": "c"}])

        assert cache.get("query1") is not None
        assert cache.get("query2") is None
        assert cache.get("query3") is not None

    def test_cache_invalidate_with_datasource_id(self):
        """invalidate(datasource_id=X) leert den gesamten Cache."""
        cache = QueryCache(_make_config())
        cache.put("query1", [{"doc": "a"}])

        cache.invalidate(datasource_id=42)

        assert cache.get("query1") is None
        assert cache.stats()["size"] == 0

    def test_cache_filters_none_values_ignored(self):
        """Filter mit None-Wert werden ignoriert beim Key-Building."""
        cache = QueryCache(_make_config())
        results = [{"doc": "test"}]

        cache.put("query", results, directory=None)
        got = cache.get("query")

        assert got == results

    def test_get_or_compute_singleflight_parallel_miss(self):
        """Parallele identische Misses fuehren compute_fn nur einmal aus."""
        cache = QueryCache(_make_config())
        calls = 0
        calls_lock = threading.Lock()
        barrier = threading.Barrier(5)
        results = []

        def compute():
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return [{"doc": "computed"}]

        def worker():
            barrier.wait()
            results.append(cache.get_or_compute("query", compute, directory="/docs"))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert calls == 1
        assert results == [[{"doc": "computed"}]] * 5
        assert cache.get("query", directory="/docs") == [{"doc": "computed"}]

    def test_get_or_compute_propagates_exception_and_allows_retry(self):
        """Fehlgeschlagene Computes bleiben nicht als inflight blockiert."""
        cache = QueryCache(_make_config())
        calls = 0

        def failing_compute():
            nonlocal calls
            calls += 1
            raise RuntimeError("backend down")

        with pytest.raises(RuntimeError, match="backend down"):
            cache.get_or_compute("query", failing_compute)

        assert cache.stats()["inflight"] == 0
        assert cache.get_or_compute("query", lambda: [{"doc": "ok"}]) == [{"doc": "ok"}]
        assert calls == 1

    def test_get_or_compute_disabled_does_not_dedupe_or_cache(self):
        """Bei disabled=False-Cache bleibt get_or_compute ein direkter Aufruf."""
        cache = QueryCache(_make_config(enabled=False))
        calls = 0

        def compute():
            nonlocal calls
            calls += 1
            return [{"doc": calls}]

        assert cache.get_or_compute("query", compute) == [{"doc": 1}]
        assert cache.get_or_compute("query", compute) == [{"doc": 2}]
        assert cache.get("query") is None
