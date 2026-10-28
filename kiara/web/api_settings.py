"""
KIara Admin - Settings API (Allgemein-Tab)
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/admin", tags=["settings"])

# --- Migriert zu Admin-Registry: system.settings.read (Sprint 08) ---
# @router.get("/settings")
# async def get_settings(user=Depends(get_admin_user)):
#     """Get current kiara.yaml configuration."""
#     config = load_config()
#     return {"config": config}

# --- Migriert zu Admin-Registry: system.settings.update (Sprint 08) ---
# @router.post("/settings")
# async def update_settings(data: ConfigUpdateRequest, user=Depends(get_admin_user)):
#     """Update kiara.yaml configuration."""
#     try:
#         save_config(data.config)
#         logger.info("Config updated by %s", user["username"])
#         return {"message": "Konfiguration gespeichert"}
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc))
#     except Exception as exc:
#         logger.error("Config save failed: %s", exc)
#         raise HTTPException(status_code=500, detail="Speichern fehlgeschlagen")

# --- Migriert zu Admin-Registry: system.vectordb.test (Sprint 08) ---
# class VectorDBTestRequest(BaseModel):
#     """Request zum Testen der VectorDB-Verbindung."""
#     backend: str = "qdrant"
#     path: str = ""
#     qdrant_url: str = ""
#     qdrant_api_key: str = ""
#     vector_size: int = DEFAULT_VECTOR_SIZE
#
# @router.post("/vectordb/test-connection")
# async def test_vectordb_connection(data: VectorDBTestRequest, user=Depends(get_admin_user)):
#     """Testet die Verbindung zum konfigurierten VectorDB-Backend."""
#     try:
#         if data.backend == "qdrant":
#             from qdrant_client import QdrantClient
#             client = QdrantClient(
#                 url=data.qdrant_url or DEFAULT_QDRANT_URL,
#                 api_key=data.qdrant_api_key or None,
#                 timeout=DEFAULT_TIMEOUTS["db_connect"],
#             )
#             colls = client.get_collections()
#             return {
#                 "success": True,
#                 "message": f"Qdrant erreichbar, {len(colls.collections)} Collection(s)",
#             }
#         else:
#             raise HTTPException(status_code=400, detail=f"Unbekanntes Backend: {data.backend}")
#     except HTTPException:
#         raise
#     except Exception as exc:
#         raise HTTPException(status_code=502, detail=f"Verbindung fehlgeschlagen: {exc}")
