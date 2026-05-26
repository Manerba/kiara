"""System-Metrik-Sammler mit SSE-Broadcast.

Laeuft als Daemon-Thread im kiara-admin Prozess (1s Takt).
- Jede Sekunde: psutil-Snapshot → In-Memory + SSE-Push
- Jede 60. Sekunde: PostgreSQL INSERT (1 Zeile/min)

SSE-Subscriber registrieren sich ueber subscribe()/unsubscribe().
"""

import asyncio
import logging
import threading
import time

try:
    import psutil

    # Validierung: Ghost-Package erkennen (leeres Namespace-Verzeichnis ohne Module)
    if not hasattr(psutil, "cpu_percent"):
        raise ImportError(
            "psutil ist installiert aber defekt (keine Attribute). "
            "Bitte neu installieren: pip install --force-reinstall psutil"
        )
    _PSUTIL_OK = True
except ImportError as _exc:
    _PSUTIL_OK = False
    _PSUTIL_ERR = str(_exc)

from kiara.shared.constants import (
    METRICS_COLLECT_INTERVAL_SECS,
    METRICS_PERSIST_INTERVAL_SECS,
)
from kiara.shared.db import db_transaction

log = logging.getLogger(__name__)

# --- Shared State ---
_lock = threading.Lock()
_latest: dict = {}  # Letzter Snapshot (Lock-geschuetzt)
_prev_net: dict | None = None  # Netzwerk-Delta-Basis
_prev_ts: float | None = None

# --- SSE-Subscriber ---
# WICHTIG: asyncio.Queue ist NICHT thread-safe! Der Collector-Thread darf
# NICHT direkt put_nowait() aufrufen. Stattdessen verwenden wir
# loop.call_soon_threadsafe() um die Queue-Operationen im Event-Loop
# Thread auszufuehren. Jeder Subscriber speichert sein Queue + Loop-Tuple.
_subscribers: set[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = set()
_sub_lock = threading.Lock()


# ──────────────────── Sampling ────────────────────


def _sample() -> dict:
    """Sammelt einmalig alle Basis-Metriken. Non-blocking.

    cpu_percent(interval=None) gibt den %-Wert seit dem letzten Aufruf
    zurueck. Da wir jede Sekunde aufrufen, ist das der 1s-Durchschnitt.
    Beim allerersten Aufruf (kein Vorwert) liefert psutil 0.0.
    """
    global _prev_net, _prev_ts

    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    load = psutil.getloadavg()
    disk = psutil.disk_usage("/")
    net = psutil.net_io_counters()

    # Netzwerk-Rate (bytes/sec) via Delta
    now = time.monotonic()
    sent_bps = 0.0
    recv_bps = 0.0
    if _prev_net is not None and _prev_ts is not None:
        dt = now - _prev_ts
        if dt > 0:
            sent_bps = max(0.0, (net.bytes_sent - _prev_net["sent"]) / dt)
            recv_bps = max(0.0, (net.bytes_recv - _prev_net["recv"]) / dt)
    _prev_net = {"sent": net.bytes_sent, "recv": net.bytes_recv}
    _prev_ts = now

    return {
        "cpu_percent": round(cpu, 1),
        "ram_used_bytes": mem.used,
        "ram_total_bytes": mem.total,
        "swap_used_bytes": swap.used,
        "swap_total_bytes": swap.total,
        "load_1m": round(load[0], 2),
        "load_5m": round(load[1], 2),
        "load_15m": round(load[2], 2),
        "net_sent_bps": round(sent_bps, 1),
        "net_recv_bps": round(recv_bps, 1),
        "disk_used_bytes": disk.used,
        "disk_total_bytes": disk.total,
    }


def _enrich(metrics: dict) -> dict:
    """Ergaenzt Basis-Metriken um Live-Only-Daten (nicht persistiert).

    Wird nur fuer den SSE-Stream aufgerufen, nicht fuer DB-Writes.
    Teure Operationen (DB-Groessen) werden nur alle 60s berechnet.
    """
    # Top-5 Prozesse nach RSS
    top_procs = []
    try:
        for p in sorted(
            psutil.process_iter(["pid", "name", "memory_info"]),
            key=lambda p: getattr(p.info.get("memory_info"), "rss", 0),
            reverse=True,
        )[:5]:
            mi = p.info.get("memory_info")
            top_procs.append(
                {
                    "pid": p.info["pid"],
                    "name": p.info["name"],
                    "rss_bytes": mi.rss if mi else 0,
                }
            )
    except Exception:
        pass
    metrics["top_processes"] = top_procs

    # Uptime + OS-Info (jeweils defensiv — darf Collector nicht crashen)
    try:
        metrics["boot_time"] = psutil.boot_time()
        metrics["uptime_seconds"] = round(time.time() - psutil.boot_time())
    except Exception:
        metrics["boot_time"] = 0
        metrics["uptime_seconds"] = 0

    try:
        import platform

        metrics["os_version"] = platform.platform()
        metrics["kernel"] = platform.release()
    except Exception:
        metrics["os_version"] = "unbekannt"
        metrics["kernel"] = "unbekannt"

    # Timestamp fuer Frontend
    from datetime import datetime, timezone

    metrics["ts"] = datetime.now(timezone.utc).isoformat()

    return metrics


# --- DB-Groessen (gecached, 60s TTL) ---
_db_sizes_cache: dict = {}
_db_sizes_ts: float = 0.0


def _get_db_sizes() -> dict:
    """Ermittelt Groesse der PostgreSQL-DB und des Qdrant-Datenverzeichnisses.

    Ergebnis wird 60s gecached (rglob auf Qdrant-Verzeichnis ist teuer).
    """
    global _db_sizes_cache, _db_sizes_ts
    if time.time() - _db_sizes_ts < 60 and _db_sizes_cache:
        return _db_sizes_cache

    sizes = {}

    # PostgreSQL
    try:
        from kiara.shared.db import db_query

        with db_query() as conn:
            row = conn.execute(
                "SELECT pg_database_size(current_database()) AS size_bytes"
            ).fetchone()
            sizes["postgresql_bytes"] = row["size_bytes"] if row else 0
    except Exception:
        sizes["postgresql_bytes"] = 0

    # Qdrant — Pfad aus Config oder Auto-Detect
    try:
        from pathlib import Path

        from kiara.config import load_config

        cfg = load_config()
        custom = cfg.get("monitoring", {}).get("qdrant_data_path", "")
        if custom:
            qdrant_path = Path(custom)
        else:
            # Auto-Detect: Docker-Volume > nativer Pfad
            candidates = [
                Path("/var/lib/docker/volumes/qdrant_qdrant_data/_data"),
                Path("/opt/qdrant/storage"),
            ]
            qdrant_path = next((p for p in candidates if p.exists()), None)

        if qdrant_path and qdrant_path.exists():
            # Schutz vor konfigurierten Pfaden wie "/" (rglob wuerde
            # das gesamte Dateisystem traversieren → DoS)
            resolved = qdrant_path.resolve()
            if len(resolved.parts) < 3:
                sizes["qdrant_bytes"] = 0
            else:
                total = sum(f.stat().st_size for f in resolved.rglob("*") if f.is_file())
                sizes["qdrant_bytes"] = total
        else:
            sizes["qdrant_bytes"] = 0
    except Exception:
        sizes["qdrant_bytes"] = 0

    _db_sizes_cache = sizes
    _db_sizes_ts = time.time()
    return sizes


# ──────────────────── SSE-Pub/Sub ────────────────────


def subscribe() -> asyncio.Queue:
    """Registriert einen neuen SSE-Subscriber. Gibt Queue zurueck.

    Muss aus einem async-Context aufgerufen werden (FastAPI-Endpoint),
    damit asyncio.get_running_loop() den korrekten Event-Loop liefert.
    """
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=5)
    with _sub_lock:
        _subscribers.add((q, loop))
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Entfernt einen SSE-Subscriber."""
    with _sub_lock:
        entry = next(((qq, ll) for qq, ll in _subscribers if qq is q), None)
        if entry is not None:
            _subscribers.discard(entry)


def _enqueue_threadsafe(q: asyncio.Queue, data: dict) -> None:
    """Thread-safe Einfuegen in eine asyncio.Queue.

    Wird innerhalb von loop.call_soon_threadsafe() ausgefuehrt,
    laeuft also im Event-Loop-Thread — dort ist put_nowait() sicher.
    """
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(data)
    except asyncio.QueueFull:
        pass


def _broadcast(data: dict) -> None:
    """Pusht Metriken an alle SSE-Subscriber (thread-safe).

    Verwendet loop.call_soon_threadsafe() um Queue-Operationen
    im Event-Loop-Thread auszufuehren. asyncio.Queue ist NICHT
    thread-safe — direktes put_nowait() aus dem Collector-Thread
    wuerde Race Conditions verursachen.

    Defensives dict-Kopieren: call_soon_threadsafe plant die
    Ausfuehrung ASYNCHRON — der Collector-Thread laeuft weiter
    und koennte das dict theoretisch mutieren bevor der Callback
    ausgefuehrt wird. Daher: Kopie erstellen.
    """
    snapshot = dict(data)  # Defensive Kopie (verhindert Race bei dict-Mutation)
    with _sub_lock:
        dead: list[tuple] = []
        for q, loop in _subscribers:
            try:
                loop.call_soon_threadsafe(_enqueue_threadsafe, q, snapshot)
            except RuntimeError:
                # Event-Loop wurde geschlossen (Server-Shutdown)
                dead.append((q, loop))
        for entry in dead:
            _subscribers.discard(entry)


# ──────────────────── Persistenz ────────────────────


def _persist(metrics: dict) -> None:
    """Schreibt eine Metrik-Zeile in die DB (1x/min)."""
    with db_transaction() as conn:
        conn.execute(
            """INSERT INTO system_metrics
               (resolution, cpu_percent, ram_used_bytes, ram_total_bytes,
                swap_used_bytes, swap_total_bytes,
                load_1m, load_5m, load_15m,
                net_sent_bps, net_recv_bps,
                disk_used_bytes, disk_total_bytes)
               VALUES ('raw', %(cpu_percent)s, %(ram_used_bytes)s, %(ram_total_bytes)s,
                        %(swap_used_bytes)s, %(swap_total_bytes)s,
                        %(load_1m)s, %(load_5m)s, %(load_15m)s,
                        %(net_sent_bps)s, %(net_recv_bps)s,
                        %(disk_used_bytes)s, %(disk_total_bytes)s)""",
            metrics,
        )


def get_latest() -> dict:
    """Gibt den letzten Snapshot zurueck (fuer Nicht-SSE-Zugriffe)."""
    with _lock:
        return dict(_latest)


# ──────────────────── Collector-Thread ────────────────────


def start_collector(stop_event: threading.Event) -> threading.Thread | None:
    """Startet den Collector als Daemon-Thread (1s Takt).

    - Jede Sekunde: _sample() → _latest + _broadcast()
    - Jede 60. Sekunde: _persist() → PostgreSQL
    - Alle 60s: _get_db_sizes() → wird im SSE-Event mitgeliefert

    Gibt None zurueck wenn psutil nicht verfuegbar ist.
    """
    if not _PSUTIL_OK:
        log.error("Metrik-Collector NICHT gestartet: %s", _PSUTIL_ERR)
        return None

    def _loop():
        global _latest, _prev_net, _prev_ts
        log.info(
            "Metrik-Collector gestartet (1s Sampling, %ds DB-Persist)",
            METRICS_PERSIST_INTERVAL_SECS,
        )

        # Netzwerk-Delta-Basis fuer Stop+Start im selben Prozess zuruecksetzen,
        # sonst wuerden die ersten Samples ueber die Stop-Phase mitteln.
        _prev_net = None
        _prev_ts = None

        # Erster Aufruf mit interval=1 um psutil-Counter zu initialisieren
        psutil.cpu_percent(interval=1)

        last_persist = time.monotonic()
        while not stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                metrics = _sample()

                # DB-Persist zeitbasiert, unabhaengig vom Sampling-Intervall.
                if cycle_start - last_persist >= METRICS_PERSIST_INTERVAL_SECS:
                    last_persist = cycle_start
                    try:
                        _persist(metrics)
                    except Exception as exc:
                        log.warning("Collector: DB-Write fehlgeschlagen: %s", exc)

                # Anreichern fuer SSE (Top-Procs, Uptime, DB-Groessen)
                enriched = _enrich(dict(metrics))
                enriched["db_sizes"] = _get_db_sizes()

                # Shared State aktualisieren
                with _lock:
                    _latest = enriched

                # An alle SSE-Clients pushen
                _broadcast(enriched)

            except Exception as exc:
                log.warning("Collector: Messung fehlgeschlagen: %s", exc)

            elapsed = time.monotonic() - cycle_start
            stop_event.wait(max(0.0, METRICS_COLLECT_INTERVAL_SECS - elapsed))

    t = threading.Thread(target=_loop, daemon=True, name="metrics-collector")
    t.start()
    return t
