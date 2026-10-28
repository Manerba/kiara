"""Backend-Selector fuer Multi-Backend-Verwaltung (Phase 13).

Stellt Failover (embedding/vlm) und Round-Robin (llm/cloud)
Verteilung ueber mehrere Backends pro Kategorie bereit.

Caching-Architektur:
    Zwei unabhaengige Caches mit unterschiedlichen TTLs:

    1. Backend-Cache (_cache / _cache_ts): TTL = BACKEND_CACHE_TTL_SECS (30s).
       Cached die DB-Abfrage ``ki_backends WHERE category = ? AND is_active = true``.
       Invalidierung explizit per invalidate_cache() nach CRUD-Operationen.

    2. Health-Check-Cache (_health_cache / _health_last_known):
       TTL = BACKEND_HEALTH_TTL_SECS (30s). Nur fuer Ollama-Provider relevant
       (Cloud-Provider werden als erreichbar angenommen — Fehler werden
       reaktiv beim API-Call behandelt).

Health-Check mit Thundering-Herd-Prevention:
    Wenn mehrere Threads gleichzeitig einen Health-Check fuer dieselbe URL
    anfordern, koennte ohne Schutz jeder Thread einen eigenen HTTP-Request
    senden. Der _CHECKING-Sentinel verhindert das:

    1. Erster Thread setzt _health_cache[url] = (_CHECKING, timestamp)
       und fuehrt den HTTP-Check ausserhalb des Locks durch.
    2. Nachfolgende Threads sehen den _CHECKING-Sentinel und geben
       den letzten bekannten Wert (_health_last_known) zurueck, statt
       einen eigenen Check zu starten.
    3. Nach Abschluss ersetzt der erste Thread den Sentinel durch
       das echte Ergebnis und aktualisiert _health_last_known.
    4. Stale-Sentinel-Schutz: Wenn ein _CHECKING-Eintrag aelter als
       2x TTL ist (Thread vermutlich abgestuerzt), wird ein neuer
       Check gestartet.

    Beim allerersten Check (kein _health_last_known-Eintrag vorhanden)
    wird optimistisch True angenommen.

Beziehungen:
    - kiara.config          -> get_*_backend() nutzt diesen Selector als primaere Quelle
    - kiara.web.api_ki_backends -> invalidate_cache() nach CRUD
    - kiara.repositories.backend -> KiBackendRepository fuer DB-Zugriff
    - kiara.shared.crypto   -> decrypt() fuer API-Keys
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any

from kiara.shared.constants import (
    BACKEND_CACHE_TTL_SECS,
    BACKEND_HEALTH_TTL_SECS,
    BACKEND_LOG_RATE_LIMIT_SECS,
)

logger = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """Kategorialer Fehler: ein benoetigtes Backend ist nicht erreichbar.

    Wird von _get_url_for_model() in retrieval/embedding.py geworfen
    und in ingestion/pipeline.py zur Consecutive-Error-Zaehlung
    herangezogen. Definiert hier (nicht pipeline.py), um zirkulaere
    Imports zu vermeiden.
    """

    pass


_lock = threading.Lock()

# Backend-Cache: {category: list[dict]} mit TTL-basierter Invalidierung.
_cache: dict[str, list[dict]] = {}
_cache_ts: dict[str, float] = {}
# Round-Robin-Index pro Kategorie (fuer LLM/Cloud Load-Balancing).
_rr_index: dict[str, int] = {}

# Health-Check-Cache: {url: (is_healthy_or_sentinel, timestamp)}.
# Sentinel _CHECKING verhindert parallele Checks derselben URL (Thundering-Herd).
_health_cache: dict[str, tuple[bool | str, float]] = {}
# Last-known-good-State als Fallback bei Timeout.
_health_last_known: dict[str, bool] = {}
_CHECKING = "_checking"  # Sentinel-Wert

# Rate-Limit-Cache fuer Unreachable-WARNINGs: {url: last_logged_timestamp}.
# Verhindert Log-Spam bei Cluster-Ausfall (siehe BACKEND_LOG_RATE_LIMIT_SECS).
_unreachable_log_ts: dict[str, float] = {}


def _should_log_unreachable(url: str) -> bool:
    """True, wenn fuer diese URL seit BACKEND_LOG_RATE_LIMIT_SECS nichts mehr geloggt wurde.

    Markiert die URL als geloggt sobald True zurueckgegeben wird, damit
    parallele Aufrufer ohne weitere Synchronisation auskommen.
    """
    now = time.time()
    with _lock:
        last = _unreachable_log_ts.get(url, 0.0)
        if (now - last) < BACKEND_LOG_RATE_LIMIT_SECS:
            return False
        _unreachable_log_ts[url] = now
        return True


def _fetch_backends(category: str) -> list[dict] | None:
    """Laedt aktive Backends einer Kategorie aus der DB, sortiert nach Prioritaet.

    Returns:
        Liste (ggf. leer) bei erfolgreicher DB-Abfrage; None bei DB-Fehler,
        damit get_all_active() auf den Stale-Cache-Fallback zurueckgreifen kann
        statt fuer 30s eine leere Liste zu cachen.
    """
    try:
        from kiara.repositories.backend import KiBackendRepository

        _backend_repo = KiBackendRepository()
        rows = _backend_repo.list_by_category(category, active_only=True)
        return [_decrypt_backend(r) for r in rows]
    except Exception as exc:
        logger.debug("Backend-Abfrage fehlgeschlagen fuer '%s': %s", category, exc)
        return None


def _decrypt_backend(backend: dict) -> dict:
    """Entschluesselt den API-Key falls verschluesselt."""
    api_key = backend.get("api_key", "")
    if api_key:
        try:
            from kiara.shared.crypto import decrypt, is_encrypted

            if is_encrypted(api_key):
                backend["api_key"] = decrypt(api_key)
        except Exception as exc:
            logger.warning(
                "API-Key-Entschluesselung fehlgeschlagen fuer Backend '%s': %s",
                backend.get("name", "?"),
                exc,
            )
            backend["api_key"] = ""
    return backend


def get_all_active(category: str) -> list[dict]:
    """Gibt alle aktiven Backends einer Kategorie zurueck (mit TTL-Cache).

    Args:
        category: 'embedding', 'llm', 'cloud', 'vlm' oder 'ddu'.

    Returns:
        Liste von Backend-Dicts, sortiert nach Prioritaet.
    """
    with _lock:
        return _get_all_active_locked(category, time.time())


def _get_all_active_locked(category: str, now: float) -> list[dict]:
    """Gibt aktive Backends zurueck. Muss unter _lock aufgerufen werden."""
    if category in _cache and (now - _cache_ts.get(category, 0)) < BACKEND_CACHE_TTL_SECS:
        return _cache[category]
    backends = _fetch_backends(category)
    if backends is None:
        return _cache.get(category, [])
    _cache[category] = backends
    _cache_ts[category] = now
    return backends


def get_next_backend(category: str) -> dict[str, Any] | None:
    """Gibt das naechste Backend zurueck.

    Failover fuer embedding/vlm/ddu (immer erstes aktives),
    Round-Robin fuer llm/cloud (rotiert durch aktive Backends).

    Returns:
        Backend-Dict oder None wenn keines verfuegbar.
    """
    if category in ("llm", "cloud"):
        # Snapshot und Round-Robin-Index gehoeren in denselben kritischen
        # Abschnitt: sonst kann parallele Invalidierung/Reload den Index auf
        # eine andere Backend-Liste anwenden als die, aus der ausgewaehlt wird.
        with _lock:
            backends = _get_all_active_locked(category, time.time())
            if not backends:
                return None
            idx = _rr_index.get(category, 0)
            if idx >= len(backends):
                idx = 0
            backend = backends[idx]
            _rr_index[category] = (idx + 1) % len(backends)
            return backend

    backends = get_all_active(category)
    if not backends:
        return None

    if category in ("embedding", "vlm", "ddu"):
        # Failover: hoechste Prioritaet mit Health-Check
        for b in backends:
            if _is_healthy_cached(b["url"], provider=b.get("provider", "ollama")):
                return b
            if _should_log_unreachable(b["url"]):
                logger.warning(
                    "%s-Backend '%s' (%s) nicht erreichbar, versuche naechstes",
                    category.upper(),
                    b.get("name", "?"),
                    b["url"],
                )
        return None

    return backends[0]


def get_backend_by_id(backend_id: int) -> dict[str, Any] | None:
    """Gibt ein bestimmtes aktives Backend per ID zurueck.

    Args:
        backend_id: Die Datenbank-ID des Backends.

    Returns:
        Backend-Dict mit entschluesseltem API-Key, oder None wenn
        nicht gefunden oder inaktiv.
    """
    try:
        from kiara.repositories.backend import KiBackendRepository

        _backend_repo = KiBackendRepository()
        row = _backend_repo.get_active_by_id(backend_id)
        if row:
            return _decrypt_backend(row)
        return None
    except Exception as exc:
        logger.debug("Backend-Lookup fehlgeschlagen fuer ID %d: %s", backend_id, exc)
        return None


def normalize_model_name(model: str | None) -> str:
    """Normalisiert Modellnamen: entfernt ':latest' Tag, lowercase.

    None / ''                 -> ''
    'nomic-embed-text:latest' -> 'nomic-embed-text'
    'nomic-embed-text'        -> 'nomic-embed-text'
    'Nomic-Embed-Text:v2'    -> 'nomic-embed-text:v2'  (nicht-latest Tags bleiben)
    """
    if not model:
        return ""
    name = model.strip().lower()
    if name.endswith(":latest"):
        name = name[:-7]
    return name


_HEALTH_CACHE_GC_FACTOR = 10
"""Cutoff-Faktor fuer Health-Cache-GC: Eintraege > FACTOR * TTL alt gelten als stale."""


def _gc_health_cache_locked(now: float) -> None:
    """Entfernt Health-Eintraege, deren Timestamp deutlich aelter als TTL ist.

    Verhindert unbounded Wachstum bei dynamischer Backend-Konfiguration:
    Aktive Backends frischen ihren Timestamp alle TTL Sekunden auf und
    bleiben unberuehrt; URLs entfernter/umkonfigurierter Backends fallen
    nach FACTOR * TTL raus. Muss innerhalb _lock aufgerufen werden.
    """
    cutoff = now - BACKEND_HEALTH_TTL_SECS * _HEALTH_CACHE_GC_FACTOR
    stale = [url for url, (_, ts) in _health_cache.items() if ts < cutoff]
    for url in stale:
        _health_cache.pop(url, None)
        _health_last_known.pop(url, None)


def _is_healthy_cached(url: str, provider: str = "ollama", timeout: float = 5.0) -> bool:
    """Health-Check mit TTL-Cache (thread-safe, thundering-herd-geschuetzt).

    Gibt gecachten Wert zurueck wenn TTL nicht abgelaufen.
    Bei Cache-Miss oder abgelaufenem TTL: neuer Health-Check.
    Nicht-Ollama-Provider werden als erreichbar angenommen.

    Bei laufendem Check (_CHECKING-Sentinel) wird der letzte bekannte
    Wert zurueckgegeben statt blindem True. Nur beim allerersten Check
    wird True angenommen.
    """
    if provider != "ollama":
        return True

    now = time.time()
    with _lock:
        cached = _health_cache.get(url)
        if cached:
            if cached[0] == _CHECKING:
                if abs(now - cached[1]) < BACKEND_HEALTH_TTL_SECS * 2:
                    return _health_last_known.get(url, True)
                # Staler Sentinel — Thread vermutlich abgestuerzt, neu pruefen
            elif abs(now - cached[1]) < BACKEND_HEALTH_TTL_SECS:
                return cached[0]
        else:
            _gc_health_cache_locked(now)
        _health_cache[url] = (_CHECKING, now)

    # Health-Check ausserhalb Lock — blockiert nicht andere Threads
    healthy = False
    try:
        healthy = _do_health_check(url, timeout)
    finally:
        checked_at = time.time()
        with _lock:
            _health_cache[url] = (healthy, checked_at)
            _health_last_known[url] = healthy
    return healthy


def _do_health_check(url: str, timeout: float = 5.0) -> bool:
    """GET auf Ollama-Root-Endpoint (synchron, fuer Worker-Thread).

    Prueft nur Server-Erreichbarkeit, nicht Modell-Verfuegbarkeit.
    Fehlende Modelle werden reaktiv ueber ollama.ResponseError erkannt.

    Verwendet httpx statt urllib.request.urlopen — urlopen laeuft bei
    bestimmten Server-Konfigurationen (Ollama hinter uvicorn-Proxy)
    systematisch in Timeouts, waehrend httpx zuverlaessig funktioniert.
    """
    try:
        import httpx

        resp = httpx.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception as exc:
        logger.debug("Health-Check fehlgeschlagen fuer '%s': %s", url, exc)
        return False


def get_embedding_backend_for_model(model: str) -> dict | None:
    """Gibt ein aktives, erreichbares Embedding-Backend fuer ein bestimmtes Modell zurueck.

    Modellnamen werden tag-normalisiert verglichen (':latest' wird ignoriert).
    Iteriert ueber alle aktiven Backends mit passendem Modell (nach Prioritaet sortiert).
    Prueft Erreichbarkeit per TTL-gecachtem Health-Check (nur Ollama-Provider).
    Returns None wenn kein Backend fuer dieses Modell erreichbar ist.
    """
    backends = get_all_active("embedding")
    norm_model = normalize_model_name(model)
    matching = [b for b in backends if normalize_model_name(b["model"]) == norm_model]

    for b in matching:
        if _is_healthy_cached(b["url"], provider=b.get("provider", "ollama")):
            return b
        if _should_log_unreachable(b["url"]):
            logger.warning(
                "Embedding-Backend '%s' (%s) nicht erreichbar, versuche naechstes",
                b.get("name", "?"),
                b["url"],
            )

    return None


_HEALTH_CHECKED_CATEGORIES = ("embedding", "vlm", "ddu")


def _invalidate_embedding_clients_if_loaded(category: str | None) -> None:
    """Invalidiert Embedding-Client-Pools ohne Importzyklus zu erzwingen."""
    if category not in (None, "embedding"):
        return
    embedding_module = sys.modules.get("kiara.retrieval.embedding")
    if embedding_module is None:
        return
    invalidate = getattr(embedding_module, "invalidate_clients", None)
    if not callable(invalidate):
        return
    try:
        invalidate(category)
    except Exception:
        logger.warning("Embedding-Client-Cache-Invalidierung fehlgeschlagen", exc_info=True)


def _invalidate_embedding_contracts_if_loaded(category: str | None) -> None:
    """Invalidiert den read-only Contract-Discovery-Cache ohne Importzyklus."""
    if category not in (None, "embedding"):
        return
    discovery_module = sys.modules.get("kiara.retrieval.embedding_discovery")
    if discovery_module is None:
        return
    invalidate = getattr(discovery_module, "invalidate_embedding_contract_cache", None)
    if not callable(invalidate):
        return
    try:
        invalidate()
    except Exception:
        logger.warning("Embedding-Contract-Cache-Invalidierung fehlgeschlagen", exc_info=True)


def _invalidate_query_cache_if_loaded(category: str | None) -> None:
    """Invalidate, but never instantiate, the retrieval-result cache."""
    if category not in (None, "embedding"):
        return
    shared_cache = sys.modules.get("kiara.shared.cache")
    if shared_cache is None:
        return
    instance = getattr(shared_cache, "_instance", None)
    if instance is None:
        return
    try:
        instance.invalidate()
    except Exception:
        logger.warning("Query-Cache-Invalidierung fehlgeschlagen", exc_info=True)


def invalidate_cache(category: str | None = None) -> None:
    """Invalidiert den Backend-Cache nach CRUD-Operationen.

    Bei category in _HEALTH_CHECKED_CATEGORIES werden nur Health-Eintraege
    fuer URLs geloescht, die zur jeweiligen Kategorie gehoeren. Bei
    category=None wird alles geloescht.
    """
    with _lock:
        # URLs VOR Cache-Clear sammeln (danach nicht mehr verfuegbar)
        old_health_urls: set[str] = set()
        if category in _HEALTH_CHECKED_CATEGORIES:
            for b in _cache.get(category, []):
                old_health_urls.add(b["url"])

        if category:
            _cache.pop(category, None)
            _cache_ts.pop(category, None)
            _rr_index.pop(category, None)
        else:
            _cache.clear()
            _cache_ts.clear()
            _rr_index.clear()

        # Health-Cache gezielt invalidieren
        if category is None:
            _health_cache.clear()
            _health_last_known.clear()
            _unreachable_log_ts.clear()
        elif category in _HEALTH_CHECKED_CATEGORIES:
            for url in old_health_urls:
                _health_cache.pop(url, None)
                _health_last_known.pop(url, None)
                _unreachable_log_ts.pop(url, None)

    _invalidate_embedding_clients_if_loaded(category)
    _invalidate_embedding_contracts_if_loaded(category)
    _invalidate_query_cache_if_loaded(category)


def mark_healthy(url: str) -> None:
    """Markiert eine URL im Health-Cache als erreichbar.

    Wird von BackendWarmer nach erfolgreichem Warmup aufgerufen,
    um den Health-Cache mit dem Warmup-Ergebnis zu synchronisieren.
    """
    now = time.time()
    with _lock:
        if url not in _health_cache:
            _gc_health_cache_locked(now)
        _health_cache[url] = (True, now)
        _health_last_known[url] = True
