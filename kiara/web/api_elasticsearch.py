"""KIara Admin - Elasticsearch API.

Endpoints fuer Index-Management: Status, Rebuild, Loeschen, Integritaetspruefung.
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/elasticsearch", tags=["elasticsearch"])

_es_task = {"running": False, "action": None, "progress": None, "result": None}

# --- Migriert zu Admin-Registry: elasticsearch.status (Sprint 08) ---
# @router.get("/status")
# async def get_es_status(user=Depends(get_admin_user)):
#     """Elasticsearch-Status: Erreichbarkeit, Index-Existenz, Chunk-Anzahl."""
#     from kiara.retrieval.elasticsearch_service import get_status
#     from kiara.retrieval.vectorstore import VectorStore
#     config = load_config()
#     status = get_status(config)
#     try:
#         store = VectorStore(config)
#         status["qdrant_count"] = store.get_stats().get("total_chunks", 0)
#     except Exception as exc:
#         status["qdrant_count"] = 0
#     status["task_running"] = _es_task["running"]
#     status["task_action"] = _es_task["action"]
#     status["task_progress"] = _es_task["progress"]
#     return status

# --- Migriert zu Admin-Registry: elasticsearch.rebuild (Sprint 08) ---
# @router.post("/rebuild")
# async def rebuild_index(user=Depends(get_admin_user)):
#     """Index komplett neu aufbauen (loescht bestehenden Index)."""
#     from kiara.retrieval.elasticsearch_service import is_configured
#     config = load_config()
#     if not is_configured(config):
#         raise HTTPException(400, "Elasticsearch ist nicht als Hybrid-Suche konfiguriert")
#     if _es_task["running"]:
#         raise HTTPException(409, "ES-Aufgabe laeuft bereits")
#     def _run():
#         try:
#             from kiara.retrieval.elasticsearch_service import rebuild_from_vectorstore
#             _es_task["running"] = True
#             _es_task["action"] = "rebuild"
#             _es_task["progress"] = {"indexed": 0, "total": 0}
#             _es_task["result"] = None
#             def on_progress(indexed, total):
#                 _es_task["progress"] = {"indexed": indexed, "total": total}
#             config = load_config()
#             result = rebuild_from_vectorstore(config, progress_callback=on_progress)
#             _es_task["result"] = {"status": "success", **result}
#             logger.info("ES-Rebuild abgeschlossen: %s", result)
#         except Exception as exc:
#             logger.error("ES-Rebuild fehlgeschlagen: %s", exc)
#             _es_task["result"] = {"status": "error", "error": str(exc)}
#         finally:
#             _es_task["running"] = False
#             _es_task["action"] = None
#     thread = threading.Thread(target=_run, daemon=True)
#     thread.start()
#     return {"message": "Index-Rebuild gestartet"}

# --- Migriert zu Admin-Registry: elasticsearch.delete (Sprint 08) ---
# @router.delete("/index")
# async def delete_index(user=Depends(get_admin_user)):
#     """Elasticsearch-Index loeschen."""
#     from kiara.retrieval.elasticsearch_service import delete_index as es_delete
#     if _es_task["running"]:
#         raise HTTPException(409, "ES-Aufgabe laeuft bereits")
#     config = load_config()
#     try:
#         es_delete(config)
#         return {"message": "Index geloescht"}
#     except Exception as exc:
#         raise HTTPException(500, f"Loeschen fehlgeschlagen: {exc}")

# --- Migriert zu Admin-Registry: elasticsearch.verify (Sprint 08) ---
# @router.post("/verify")
# async def verify_integrity(user=Depends(get_admin_user)):
#     """Integritaetspruefung: Stichprobenvergleich Qdrant vs. ES."""
#     from kiara.retrieval.elasticsearch_service import verify_integrity as es_verify
#     if _es_task["running"]:
#         raise HTTPException(409, "ES-Aufgabe laeuft bereits")
#     def _run():
#         try:
#             _es_task["running"] = True
#             _es_task["action"] = "verify"
#             _es_task["progress"] = None
#             _es_task["result"] = None
#             config = load_config()
#             result = es_verify(config)
#             _es_task["result"] = {"status": "success", **result}
#             logger.info("ES-Integritaetspruefung: %s", result)
#         except Exception as exc:
#             logger.error("ES-Integritaetspruefung fehlgeschlagen: %s", exc)
#             _es_task["result"] = {"status": "error", "error": str(exc)}
#         finally:
#             _es_task["running"] = False
#             _es_task["action"] = None
#     thread = threading.Thread(target=_run, daemon=True)
#     thread.start()
#     return {"message": "Integritaetspruefung gestartet"}

# --- Migriert zu Admin-Registry: elasticsearch.task (Sprint 08) ---
# @router.get("/task")
# async def get_task_status(user=Depends(get_admin_user)):
#     """Status der laufenden oder letzten ES-Aufgabe."""
#     return {
#         "running": _es_task["running"],
#         "action": _es_task["action"],
#         "progress": _es_task["progress"],
#         "result": _es_task["result"],
#     }
