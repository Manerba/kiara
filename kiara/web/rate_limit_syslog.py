"""Sliding-Window-Rate-Limiter fuer System-Log-API (60 req/min pro Admin).

Thread-safe (Lock pro Worker-Prozess). Worker-lokaler State — bei N uvicorn-
Workern gilt effektiv 60/min * N (akzeptiertes V1-Risiko, dokumentiert in
TODO_FEATURE_ADMIN_SYSTEM_LOG_V18.md Abschnitt 4.5).
"""

from __future__ import annotations

import threading
import time
from collections import deque

_WINDOW_SECONDS = 60.0
_MAX_REQUESTS = 60
_CLEANUP_THRESHOLD = 1000

_lock = threading.Lock()
_buckets: dict[str, deque] = {}


def check_rate_limit(user_id: str | int) -> tuple[bool, float | None]:
    """Prueft und protokolliert einen Request.

    Returns:
        (allowed, retry_after): bei allowed=True ist retry_after None; sonst
        die Wartezeit in Sekunden bis der aelteste Eintrag aus dem Fenster
        faellt.
    """
    key = str(user_id)
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS

    with _lock:
        bucket = _buckets.get(key)
        if bucket is None:
            bucket = deque()
            _buckets[key] = bucket

        while bucket and bucket[0] < cutoff:
            bucket.popleft()

        if len(bucket) >= _MAX_REQUESTS:
            retry_after = max(0.0, _WINDOW_SECONDS - (now - bucket[0]))
            return False, retry_after

        bucket.append(now)

        if len(_buckets) > _CLEANUP_THRESHOLD:
            _cleanup_stale(cutoff)

        return True, None


def _cleanup_stale(cutoff: float) -> None:
    """Entfernt Buckets ohne aktive Eintraege (caller haelt bereits den Lock)."""
    stale = [key for key, bucket in _buckets.items() if not bucket or bucket[-1] < cutoff]
    for key in stale:
        _buckets.pop(key, None)


def reset_for_tests() -> None:
    """Nur fuer Tests: Buckets komplett leeren."""
    with _lock:
        _buckets.clear()
