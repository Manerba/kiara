"""
Elasticsearch-Service — extrahiert aus api_elasticsearch.py.
Verwaltet Background-Rebuild, Verify und Status-Tracking.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_es_lock = threading.Lock()

# Shared state fuer Background-ES-Tasks
# progress: None (idle/verify) oder {"indexed": int, "total": int} (rebuild)
# result: {"status": "success"|"error", ...} — identisch zu api_elasticsearch.py
_es_task: dict = {
    "running": False,
    "action": None,
    "progress": None,
    "result": None,
}


def is_running() -> bool:
    """Prüft ob ein ES-Task läuft (thread-safe)."""
    with _es_lock:
        return _es_task["running"]


def get_task_status() -> dict:
    """Gibt den aktuellen Task-Status zurück (thread-safe)."""
    with _es_lock:
        return dict(_es_task)


def get_es_status() -> dict:
    """Gibt ES-Verbindungsstatus, Index-Info und Chunk-Counts zurück.

    Logik extrahiert aus api_elasticsearch.py get_es_status (Zeile 21-41).
    Response-Format identisch zum Original (Frontend-Kompatibilität).
    """
    from kiara.config import load_config
    from kiara.retrieval.elasticsearch_service import get_status
    from kiara.retrieval.vectorstore import VectorStore

    config = load_config()
    status = get_status(config)

    # Qdrant-Chunk-Anzahl fuer Vergleich (wie api_elasticsearch.py Zeile 31-35)
    try:
        store = VectorStore(config)
        status["qdrant_count"] = store.get_stats().get("total_chunks", 0)
    except Exception:
        status["qdrant_count"] = 0

    # Task-Status aus Service-State (wie api_elasticsearch.py Zeile 37-39)
    with _es_lock:
        status["task_running"] = _es_task["running"]
        status["task_action"] = _es_task["action"]
        status["task_progress"] = _es_task["progress"]

    return status


def start_rebuild() -> None:
    """Startet einen Index-Rebuild in einem Background-Thread.

    Raises: RuntimeError wenn bereits ein Task läuft oder ES nicht konfiguriert.
    Logik extrahiert aus api_elasticsearch.py rebuild_index (Route Zeile 44, Logik Zeile 56-81).
    """
    from kiara.config import load_config
    from kiara.retrieval.elasticsearch_service import is_configured

    config = load_config()
    if not is_configured(config):
        raise RuntimeError("Elasticsearch ist nicht als Hybrid-Suche konfiguriert")

    with _es_lock:
        if _es_task["running"]:
            raise RuntimeError("Ein ES-Task läuft bereits")
        _es_task.update(
            {
                "running": True,
                "action": "rebuild",
                "progress": {"indexed": 0, "total": 0},
                "result": None,
            }
        )

    thread = threading.Thread(target=_run_rebuild, daemon=True)
    try:
        thread.start()
    except RuntimeError:
        with _es_lock:
            _es_task.update({"running": False, "action": None, "progress": None})
        raise


def _run_rebuild() -> None:
    """Background-Thread: Baut den ES-Index komplett neu auf.

    Ruft rebuild_from_vectorstore() aus kiara.retrieval.elasticsearch_service auf.
    Progress-Callback aktualisiert _es_task["progress"].
    """
    try:
        from kiara.config import load_config
        from kiara.retrieval.elasticsearch_service import rebuild_from_vectorstore

        config = load_config()

        def _progress(indexed: int, total: int) -> None:
            # progress_callback wird mit (indexed + errors, total) aufgerufen
            # Key "indexed" — identisch zum Original (api_elasticsearch.py Zeile 65)
            with _es_lock:
                _es_task["progress"] = {"indexed": indexed, "total": total}

        result = rebuild_from_vectorstore(config, progress_callback=_progress)
        # Key "status" — identisch zum Original (api_elasticsearch.py Zeile 69)
        with _es_lock:
            _es_task["result"] = {"status": "success", **result}
    except Exception as exc:
        logger.exception("ES-Rebuild fehlgeschlagen")
        with _es_lock:
            _es_task["result"] = {"status": "error", "error": str(exc)}
    finally:
        with _es_lock:
            _es_task["running"] = False
            _es_task["action"] = None
            _es_task["progress"] = None


def delete_index() -> dict:
    """Löscht den Elasticsearch-Index.

    Raises: RuntimeError wenn ein ES-Task läuft.
    Logik extrahiert aus api_elasticsearch.py delete_index (Zeile 84-97).
    Ruft kiara.retrieval.elasticsearch_service.delete_index(config) auf.
    """
    with _es_lock:
        if _es_task["running"]:
            raise RuntimeError("Ein ES-Task läuft bereits")

    from kiara.config import load_config
    from kiara.retrieval.elasticsearch_service import delete_index as es_delete

    config = load_config()
    if not es_delete(config):
        raise RuntimeError("Elasticsearch-Index konnte nicht geloescht werden")
    return {"message": "Index geloescht"}


def start_verify() -> None:
    """Startet eine Integritätsprüfung in einem Background-Thread.

    Raises: RuntimeError wenn bereits ein Task läuft.
    Logik extrahiert aus api_elasticsearch.py verify_integrity (Route Zeile 100, Logik Zeile 108-129).
    """
    with _es_lock:
        if _es_task["running"]:
            raise RuntimeError("Ein ES-Task läuft bereits")
        _es_task.update({"running": True, "action": "verify", "progress": None, "result": None})

    thread = threading.Thread(target=_run_verify, daemon=True)
    try:
        thread.start()
    except RuntimeError:
        with _es_lock:
            _es_task.update({"running": False, "action": None, "progress": None})
        raise


def _run_verify() -> None:
    """Background-Thread: Prüft Index-Integrität (ES vs Qdrant).

    Ruft verify_integrity() aus kiara.retrieval.elasticsearch_service auf.
    """
    try:
        from kiara.config import load_config
        from kiara.retrieval.elasticsearch_service import verify_integrity

        config = load_config()
        result = verify_integrity(config)
        # Key "status" — identisch zum Original (api_elasticsearch.py Zeile 117)
        with _es_lock:
            _es_task["result"] = {"status": "success", **result}
    except Exception as exc:
        logger.exception("ES-Verify fehlgeschlagen")
        with _es_lock:
            _es_task["result"] = {"status": "error", "error": str(exc)}
    finally:
        with _es_lock:
            _es_task["running"] = False
            _es_task["action"] = None
