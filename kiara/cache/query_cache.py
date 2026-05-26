"""LRU-Cache fuer haeufige Retrieval-Anfragen."""

import copy
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Callable

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL = 100  # Cleanup alle N put()-Operationen
_CACHE_MISS = object()


class QueryCache:
    """LRU-Cache fuer Retrieval-Ergebnisse (thread-safe).

    Cached: Retrieval-Ergebnisse (embedding-basiert)
    NICHT gecached: LLM-Antworten

    Cache-Key: SHA256(query + sorted filter_params)
    TTL: Konfigurierbar (Default: 1h)
    Max-Size: Konfigurierbar (Default: 1000)
    Invalidierung: Bei Ingestion
    """

    def __init__(self, config: dict):
        self._enabled = config.get("cache", {}).get("enabled", True)
        self._max_size = config.get("cache", {}).get("max_size", 1000)
        self._ttl_seconds = config.get("cache", {}).get("ttl_seconds", 3600)
        self._cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._expirations = 0
        self._lock = threading.Lock()
        self._inflight: dict[str, dict] = {}
        self._generation = 0
        self._put_count = 0

    def _make_key(self, query: str, **filters) -> str:
        """Erzeugt Cache-Key aus Query + Filtern."""
        active_filters = {k: filters[k] for k in sorted(filters.keys()) if filters[k] is not None}
        payload = json.dumps({"q": query, "f": active_filters}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, query: str, **filters) -> list[dict] | None:
        """Cache-Lookup. Returns None bei Miss oder wenn disabled."""
        if not self._enabled:
            return None

        key = self._make_key(query, **filters)
        with self._lock:
            result = self._get_by_key_locked(key)
            return None if result is _CACHE_MISS else result

    def get_or_compute(
        self,
        query: str,
        compute_fn: Callable[[], list[dict]],
        **filters,
    ) -> list[dict]:
        """Liefert Cachewert oder berechnet ihn pro Key nur einmal parallel.

        Bei parallelen identischen Cache-Misses fuehrt genau ein Thread
        ``compute_fn`` aus. Die anderen warten auf dasselbe Ergebnis.
        """
        if not self._enabled:
            return compute_fn()

        key = self._make_key(query, **filters)
        owner = False

        with self._lock:
            result = self._get_by_key_locked(key)
            if result is not _CACHE_MISS:
                return result

            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = {
                    "event": threading.Event(),
                    "result": None,
                    "error": None,
                }
                self._inflight[key] = inflight
                generation = self._generation
                owner = True

        if not owner:
            inflight["event"].wait()
            if inflight["error"] is not None:
                raise inflight["error"]
            return copy.deepcopy(inflight["result"])

        try:
            computed = compute_fn()
        except BaseException as exc:
            with self._lock:
                inflight["error"] = exc
                self._inflight.pop(key, None)
                inflight["event"].set()
            raise

        with self._lock:
            result_for_waiters = copy.deepcopy(computed)
            if generation == self._generation:
                self._store_by_key_locked(key, computed)
            inflight["result"] = result_for_waiters
            self._inflight.pop(key, None)
            inflight["event"].set()
        return computed

    def _get_by_key_locked(self, key: str):
        """Cache-Lookup unter self._lock."""
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return _CACHE_MISS

        timestamp, results = entry

        # TTL pruefen
        if time.monotonic() - timestamp > self._ttl_seconds:
            del self._cache[key]
            self._expirations += 1
            return _CACHE_MISS

        # Move to end (LRU)
        self._cache.move_to_end(key)
        self._hits += 1
        return copy.deepcopy(results)

    def _store_by_key_locked(self, key: str, results: list[dict]) -> None:
        """Cache-Eintrag unter self._lock setzen."""
        self._cache[key] = (time.monotonic(), copy.deepcopy(results))
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        # Periodischer Cleanup abgelaufener Eintraege
        self._put_count += 1
        if self._put_count >= _CLEANUP_INTERVAL:
            self._put_count = 0
            self._evict_expired()

    def _evict_expired(self):
        """Entfernt TTL-abgelaufene Eintraege. Muss unter self._lock aufgerufen werden."""
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._cache.items() if now - ts > self._ttl_seconds]
        for k in expired:
            del self._cache[k]
        self._expirations += len(expired)

    def put(self, query: str, results: list[dict], **filters):
        """Cache-Eintrag setzen. Evicted aelteste bei Ueberlauf."""
        if not self._enabled:
            return

        key = self._make_key(query, **filters)
        with self._lock:
            self._store_by_key_locked(key, results)

    def invalidate(self, datasource_id: int = None):
        """Cache leeren. Bei datasource_id: alle leeren (einfache Impl)."""
        with self._lock:
            self._cache.clear()
            self._generation += 1
        logger.info("QueryCache invalidiert (datasource_id=%s)", datasource_id)

    def stats(self) -> dict:
        """Cache-Statistiken."""
        with self._lock:
            total = self._hits + self._misses + self._expirations
            return {
                "enabled": self._enabled,
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "expirations": self._expirations,
                "inflight": len(self._inflight),
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            }
