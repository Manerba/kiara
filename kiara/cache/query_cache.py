"""LRU-Cache fuer haeufige Retrieval-Anfragen."""

import copy
import hashlib
import json
import logging
import math
import threading
import time
import unicodedata
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from typing import Callable

from kiara.retrieval.embedding_contract import ProfileKind
from kiara.retrieval.embedding_roles import (
    SEARCH_QUERY,
    normalize_semantic_query_input,
    request_profile_from_guard,
)

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL = 100  # Cleanup alle N put()-Operationen
_CACHE_MISS = object()
QUERY_CACHE_KEY_SCHEMA_VERSION = 2


class QueryCache:
    """LRU-Cache fuer Retrieval-Ergebnisse (thread-safe).

    Cached: Retrieval-Ergebnisse (embedding-basiert)
    NICHT gecached: LLM-Antworten

    Cache-Key: versionierter SHA-256 ueber Query-/Profilvertrag + Fachinput
    TTL: Konfigurierbar (Default: 1h)
    Max-Size: Konfigurierbar (Default: 1000)
    Invalidierung: Bei Ingestion
    """

    def __init__(self, config: dict):
        self._config = config
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
        self._keyspace_active = False
        self._legacy_entries_invalidated = 0

    @contextmanager
    def _contract_guard(self, operation: str):
        """Guard the selected primary profile and ColBERT before cache delivery."""
        from kiara.config import get_embedding_model
        from kiara.retrieval.index_compatibility import (
            colbert_index_target,
            guard_index,
            primary_index_target,
        )
        from kiara.shared.constants import COLBERT_DEFAULT_MODEL

        with ExitStack() as stack:
            dense_model = get_embedding_model(self._config)
            dense_token = stack.enter_context(
                guard_index(
                    self._config,
                    primary_index_target(self._config, dense_model),
                    operation=operation,
                    write=False,
                    require_configured_default=True,
                )
            )
            dense_profile = request_profile_from_guard(
                dense_token,
                role=SEARCH_QUERY,
                expected_kind=ProfileKind(dense_token.target.vector_kind.value),
                expected_endpoint=(
                    "/api/embed_late"
                    if dense_token.target.vector_kind.value == "late_chunking"
                    else "/api/embed"
                ),
            )
            identities = [self._cache_profile_identity(dense_token, dense_profile)]
            colbert_cfg = self._config.get("retrieval", {}).get("colbert", {})
            if colbert_cfg.get("enabled", False):
                colbert_model = colbert_cfg.get("model", COLBERT_DEFAULT_MODEL)
                colbert_token = stack.enter_context(
                    guard_index(
                        self._config,
                        colbert_index_target(self._config, colbert_model),
                        operation=operation,
                        write=False,
                    )
                )
                colbert_profile = request_profile_from_guard(
                    colbert_token,
                    role=SEARCH_QUERY,
                    expected_kind=ProfileKind.MULTI_VECTOR,
                    expected_endpoint="/api/embed_colbert",
                )
                identities.append(self._cache_profile_identity(colbert_token, colbert_profile))
            yield tuple(identities)

    @staticmethod
    def _cache_profile_identity(token, profile) -> dict[str, str | int]:
        """Minimale dauerhafte Query-Key-Identitaet eines Vektorpfads."""

        return {
            "canonical_model_id": profile.canonical_model_id,
            "profile_id": profile.profile_id,
            "profile_kind": profile.kind,
            "endpoint": profile.endpoint,
            "input_type": SEARCH_QUERY,
            "query_compatibility_id": profile.query_compatibility_id,
            "logical_index_key": token.target.logical_index_key,
            "fence_version": int(token.fence_version),
        }

    @staticmethod
    def _normalize_cache_value(value):
        """Kanonisiert Filterwerte ohne fachliche Typen zusammenzulegen."""

        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Nicht-finite Filterwerte sind im Query-Cache unzulaessig")
            return value
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Query-Cache-Filterkeys muessen Strings sein")
            return {key: QueryCache._normalize_cache_value(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [QueryCache._normalize_cache_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [QueryCache._normalize_cache_value(item) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        raise TypeError(f"Nicht unterstuetzter Query-Cache-Filtertyp: {type(value).__name__}")

    def _activate_current_keyspace_locked(self) -> None:
        """Invalidiert vor dem ersten V2-Zugriff kontrolliert alle Altkeys."""

        if self._keyspace_active:
            return
        self._legacy_entries_invalidated += len(self._cache)
        self._cache.clear()
        self._generation += 1
        self._keyspace_active = True
        logger.info(
            "QueryCache-Keyspace aktiviert schema_version=%d legacy_entries_invalidated=%d",
            QUERY_CACHE_KEY_SCHEMA_VERSION,
            self._legacy_entries_invalidated,
        )

    def _key_preimage(
        self,
        query: str,
        profile_identities: tuple[dict, ...],
        **filters,
    ) -> str:
        """Erzeugt die kanonische, auditierbare V2-Key-Praeimage."""
        if not profile_identities:
            raise ValueError("Query-Cache-Key erfordert mindestens eine Profilidentitaet")
        normalized_profiles = sorted(
            (self._normalize_cache_value(item) for item in profile_identities),
            key=lambda item: (item["profile_kind"], item["profile_id"]),
        )
        active_filters = {
            key: self._normalize_cache_value(filters[key])
            for key in sorted(filters)
            if filters[key] is not None
        }
        return json.dumps(
            {
                "schema": "kiara.query_embedding_cache_key",
                "schema_version": QUERY_CACHE_KEY_SCHEMA_VERSION,
                "operation": SEARCH_QUERY,
                "profiles": normalized_profiles,
                "input": {
                    "query": normalize_semantic_query_input(query),
                    "filters": active_filters,
                },
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _make_key(self, query: str, profile_identities: tuple[dict, ...], **filters) -> str:
        """Hasht die vollstaendige, versionierte Vertrags-Key-Praeimage."""

        payload = self._key_preimage(query, profile_identities, **filters)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, query: str, **filters) -> list[dict] | None:
        """Cache-Lookup. Returns None bei Miss oder wenn disabled."""
        if not self._enabled:
            return None

        with self._contract_guard("cache_read") as profiles:
            key = self._make_key(query, profiles, **filters)
            with self._lock:
                self._activate_current_keyspace_locked()
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

        with self._contract_guard("cache_get_or_compute") as profiles:
            return self._get_or_compute_guarded(query, compute_fn, profiles, **filters)

    def _get_or_compute_guarded(
        self,
        query: str,
        compute_fn: Callable[[], list[dict]],
        profile_identities: tuple[dict, ...],
        **filters,
    ) -> list[dict]:
        key = self._make_key(query, profile_identities, **filters)
        owner = False

        with self._lock:
            self._activate_current_keyspace_locked()
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

        with self._contract_guard("cache_write") as profiles:
            key = self._make_key(query, profiles, **filters)
            with self._lock:
                self._activate_current_keyspace_locked()
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
                "key_schema_version": QUERY_CACHE_KEY_SCHEMA_VERSION,
                "keyspace_active": self._keyspace_active,
                "legacy_entries_invalidated": self._legacy_entries_invalidated,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
            }
