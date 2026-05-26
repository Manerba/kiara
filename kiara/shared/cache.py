"""Cache-Singleton — extrahiert aus api_cache.py.

Inline-Singleton statt lazy_singleton-Import um Dependency-Richtung
shared/ → web/ zu vermeiden.
"""

import threading

_lock = threading.Lock()
_instance = None


def get_query_cache():
    """Gibt die QueryCache-Singleton-Instanz zurück (thread-safe, lazy)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from kiara.cache.query_cache import QueryCache
                from kiara.config import load_config

                _instance = QueryCache(load_config())
    return _instance


def _reset_query_cache():
    """Nur für Tests — setzt die Singleton-Instanz zurück."""
    global _instance
    with _lock:
        _instance = None
