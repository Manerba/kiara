"""Read-only storage measurements shared by monitoring and VectorDB management.

Filesystem walks and remote requests run in a background thread. The one-second
metrics stream and HTTP requests only read the most recent snapshot.
"""

import copy
import logging
import os
import socket
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from kiara.shared.constants import DEFAULT_QDRANT_URL
from kiara.shared.db import db_query

logger = logging.getLogger(__name__)

_REFRESH_SECONDS = 60
_IO_TIMEOUT_SECONDS = 5
_QDRANT_PATHS = (
    "/var/lib/qdrant/storage",
    "/var/lib/docker/volumes/qdrant_qdrant_data/_data",
    "/opt/qdrant/storage",
)
_lock = threading.Lock()
_snapshot: dict = {}
_refreshing = False
_refreshed_at = 0.0


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "size_bytes": None, "reason": reason}


def _byte_count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Ungültige Speichergröße.")
    return value


def _postgresql_sizes() -> dict:
    result = _unavailable("PostgreSQL-Größe konnte nicht ermittelt werden.")
    try:
        with db_query() as conn:
            row = conn.execute(
                "SELECT pg_database_size(current_database()) AS size_bytes"
            ).fetchone()
            result = {"status": "ok", "size_bytes": _byte_count(row["size_bytes"])}
            rows = conn.execute(
                "SELECT schemaname AS schema, relname AS name, "
                "pg_total_relation_size(relid) AS size_bytes, "
                "pg_table_size(relid) AS table_bytes, "
                "pg_indexes_size(relid) AS index_bytes "
                "FROM pg_stat_user_tables "
                "ORDER BY pg_total_relation_size(relid) DESC, schemaname, relname LIMIT 10"
            ).fetchall()
            result["tables"] = [dict(row) for row in rows]
    except Exception:
        # A failed detail query must not discard a successfully measured total.
        logger.warning("PostgreSQL-Speichermessung fehlgeschlagen", exc_info=True)
    return result


def _qdrant_path(config: dict) -> Path:
    custom = config.get("monitoring", {}).get("qdrant_data_path", "")
    if custom:
        path = Path(custom).resolve(strict=True)
    else:
        host = urlsplit(config.get("vectordb", {}).get("qdrant_url", DEFAULT_QDRANT_URL)).hostname
        local_hosts = {"localhost", "127.0.0.1", "::1", socket.gethostname(), socket.getfqdn()}
        if host not in local_hosts:
            raise ValueError("Für externes Qdrant ist kein lokaler Datenpfad konfiguriert.")
        paths = [Path(candidate) for candidate in _QDRANT_PATHS]
        matches = []
        for candidate in paths:
            try:
                if (candidate / "collections").is_dir():
                    matches.append(candidate)
            except PermissionError:
                # Docker's volume parent is commonly root-only, even when
                # Qdrant actually uses the accessible /var/lib/qdrant bind mount.
                continue
        if len(matches) != 1:
            raise ValueError("Qdrant-Datenpfad fehlt oder ist nicht eindeutig.")
        path = matches[0].resolve(strict=True)
    if len(path.parts) < 3 or not (path / "collections").is_dir():
        raise ValueError("Kein gültiges Qdrant-Datenverzeichnis.")
    return path


def _directory_sizes(root: Path) -> tuple[int, dict[str, int]]:
    """Count allocated bytes, including directories, once per inode.

    Permission errors, disappearing files and symlinks invalidate the scan;
    returning a partial count as the full disk usage would be misleading.
    """
    total = 0
    collections: dict[str, int] = {}
    seen = set()
    pending = [(root, None)]
    deadline = time.monotonic() + 15
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError("Qdrant-Speichermessung dauert zu lange.")
        path, collection = pending.pop()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("Qdrant-Datenverzeichnis enthält symbolische Links.")
        inode = (info.st_dev, info.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        size = info.st_blocks * 512
        total += size
        if collection is not None:
            collections[collection] = collections.get(collection, 0) + size
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                for entry in entries:
                    child_collection = collection
                    if path == root / "collections" and entry.is_dir(follow_symlinks=False):
                        child_collection = entry.name
                    pending.append((Path(entry.path), child_collection))
    return total, collections


def _qdrant_sizes(config: dict) -> dict:
    result = _unavailable("Qdrant-Datenpfad nicht verfügbar oder nicht vollständig lesbar.")
    try:
        path = _qdrant_path(config)
        # Segment swaps during ingestion can briefly remove a scanned file.
        try:
            total, by_collection = _directory_sizes(path)
        except FileNotFoundError:
            total, by_collection = _directory_sizes(path)
        result = {"status": "ok", "size_bytes": total, "collections": []}
    except Exception:
        logger.warning("Qdrant-Speichermessung fehlgeschlagen", exc_info=True)
        by_collection = {}

    # Always enumerate every model, including ColBERT. Never open a default
    # collection: monitoring must not create or mutate an index.
    metadata = {}
    try:
        from kiara.retrieval.colbert import ColBERTStore
        from kiara.retrieval.vectorstore import VectorStore

        bounded_config = {
            **config,
            "vectordb": {**config.get("vectordb", {}), "timeout": _IO_TIMEOUT_SECONDS},
        }
        for item in VectorStore.list_collections(bounded_config):
            metadata[item["name"]] = {**item, "kind": "kb"}
        for item in ColBERTStore.list_colbert_collections(bounded_config):
            metadata[item["name"]] = item
    except Exception:
        logger.warning("Qdrant-Collection-Metadaten nicht verfügbar", exc_info=True)

    result["collections"] = [
        {
            "name": name,
            "model": metadata.get(name, {}).get("model"),
            "kind": metadata.get(name, {}).get("kind"),
            "size_bytes": by_collection.get(name),
        }
        for name in sorted(set(by_collection) | set(metadata))
    ]
    return result


def _elasticsearch_sizes(config: dict) -> dict:
    retrieval = config.get("retrieval", {})
    if not retrieval.get("elasticsearch_url") and retrieval.get("hybrid_search") != "elasticsearch":
        return {"status": "not_configured", "size_bytes": None}
    try:
        from kiara.retrieval.elasticsearch_service import storage_stats

        return storage_stats(config)
    except Exception:
        logger.warning("Elasticsearch-Speichermessung fehlgeschlagen", exc_info=True)
        return _unavailable("Elasticsearch-Indexgröße konnte nicht ermittelt werden.")


def _collect_sizes(config: dict) -> dict:
    result = {
        "postgresql": _postgresql_sizes(),
        "qdrant": _qdrant_sizes(config),
        "elasticsearch": _elasticsearch_sizes(config),
        "keyword_mode": config.get("retrieval", {}).get("hybrid_search", "none"),
    }
    measured = [result[key] for key in ("postgresql", "qdrant", "elasticsearch")]
    complete = all(item["status"] in {"ok", "not_configured"} for item in measured)
    result["total_bytes"] = sum(item["size_bytes"] or 0 for item in measured) if complete else None
    result["measured_at"] = datetime.now(timezone.utc).isoformat()
    return result


def _refresh() -> None:
    global _snapshot, _refreshing, _refreshed_at
    try:
        from kiara.config import load_config

        snapshot = _collect_sizes(load_config())
    except Exception:
        logger.warning("Speichermessung fehlgeschlagen", exc_info=True)
        snapshot = {
            key: _unavailable("Speichermessung nicht verfügbar.")
            for key in ("postgresql", "qdrant", "elasticsearch")
        }
        snapshot.update(total_bytes=None, measured_at=datetime.now(timezone.utc).isoformat())
    with _lock:
        _snapshot = snapshot
        _refreshed_at = time.monotonic()
        _refreshing = False


def get_storage_sizes() -> dict:
    """Return cached measurements immediately; refresh at most once per minute."""
    global _refreshing
    with _lock:
        if not _refreshing and (
            not _snapshot or time.monotonic() - _refreshed_at >= _REFRESH_SECONDS
        ):
            _refreshing = True
            threading.Thread(target=_refresh, daemon=True, name="storage-metrics").start()
        return copy.deepcopy(_snapshot) if _snapshot else {"loading": True, "total_bytes": None}
