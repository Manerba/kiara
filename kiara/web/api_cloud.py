"""
KIara Admin - Cloud-KI API-Router
Konfiguration, Audit-Log, Statistiken und Verbindungstest.
"""

import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from kiara.shared.constants import (
    CLOUD_BUDGET_ALERT_THRESHOLD_PCT,
    CLOUD_MONTHLY_BUDGET_EUR,
    CLOUD_ROUTING_CONFIDENCE_THRESHOLD,
    CLOUD_ROUTING_MIN_CHUNKS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/cloud", tags=["cloud"])


class CloudConfigRequest(BaseModel):
    """Request zum Speichern der Cloud-KI-Konfiguration (Provider, Routing, Budget)."""

    enabled: bool = False
    provider: str = "openai"
    api_key: str = ""
    model: str = DEFAULT_OPENAI_MODEL
    base_url: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_LLM_TEMPERATURE
    routing_mode: Literal["auto", "local", "cloud"] = "auto"
    confidence_threshold: float = CLOUD_ROUTING_CONFIDENCE_THRESHOLD
    min_relevant_chunks: int = CLOUD_ROUTING_MIN_CHUNKS
    cloud_keywords: list[str] = []
    monthly_limit_eur: float = CLOUD_MONTHLY_BUDGET_EUR
    alert_threshold_pct: int = CLOUD_BUDGET_ALERT_THRESHOLD_PCT


class CloudTestRequest(BaseModel):
    """Request zum Testen einer Cloud-API-Verbindung."""

    provider: str = "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


# --- Migriert zu Admin-Registry: cloud.config.read (Sprint 09) ---
# @router.get("/config")
# async def get_cloud_config(user=Depends(get_admin_user)):
#     """Cloud-Konfiguration lesen."""
#     config = load_config()
#     cloud = config.get("cloud", {})
#     routing = cloud.get("routing", {})
#     budget = cloud.get("budget", {})
#
#     return {
#         "config": {
#             "enabled": cloud.get("enabled", False),
#             "provider": cloud.get("provider", "openai"),
#             "api_key": PASSWORD_MASK if cloud.get("api_key") else "",
#             "model": cloud.get("model", DEFAULT_OPENAI_MODEL),
#             "base_url": cloud.get("base_url", ""),
#             "max_tokens": cloud.get("max_tokens", DEFAULT_MAX_TOKENS),
#             "temperature": cloud.get("temperature", DEFAULT_LLM_TEMPERATURE),
#             "routing_mode": routing.get("mode", cloud.get("mode", "auto")),
#             "confidence_threshold": routing.get("confidence_threshold", CLOUD_ROUTING_CONFIDENCE_THRESHOLD),
#             "min_relevant_chunks": routing.get("min_relevant_chunks", CLOUD_ROUTING_MIN_CHUNKS),
#             "cloud_keywords": routing.get("cloud_keywords", []),
#             "monthly_limit_eur": budget.get("monthly_limit_eur", CLOUD_MONTHLY_BUDGET_EUR),
#             "alert_threshold_pct": budget.get("alert_threshold_pct", CLOUD_BUDGET_ALERT_THRESHOLD_PCT),
#         }
#     }

# --- Migriert zu Admin-Registry: cloud.config.update (Sprint 09) ---
# @router.post("/config")
# async def save_cloud_config(data: CloudConfigRequest, user=Depends(get_admin_user)):
#     """Cloud-Konfiguration speichern."""
#     config = load_config()
#
#     # F-72: Cloud-API-Key verschluesseln
#     if data.api_key == PASSWORD_MASK:
#         resolved_key = config.get("cloud", {}).get("api_key", "")
#         # Alten Klartext-Key opportunistisch verschluesseln (Migration bei Speicherung)
#         if resolved_key and not is_encrypted(resolved_key):
#             resolved_key = encrypt(resolved_key)
#     else:
#         resolved_key = encrypt(data.api_key) if data.api_key else ""
#
#     config["cloud"] = {
#         "enabled": data.enabled,
#         "provider": data.provider,
#         "api_key": resolved_key,
#         "model": data.model,
#         "base_url": data.base_url,
#         "max_tokens": data.max_tokens,
#         "temperature": data.temperature,
#         "routing": {
#             "mode": data.routing_mode,
#             "confidence_threshold": data.confidence_threshold,
#             "min_relevant_chunks": data.min_relevant_chunks,
#             "cloud_keywords": data.cloud_keywords,
#         },
#         "budget": {
#             "monthly_limit_eur": data.monthly_limit_eur,
#             "alert_threshold_pct": data.alert_threshold_pct,
#         },
#     }
#
#     save_config(config)
#     logger.info("Cloud-Konfiguration gespeichert von %s", user["username"])
#     return {"message": "Cloud-Konfiguration gespeichert"}

# --- Migriert zu Admin-Registry: cloud.audit (Sprint 09) ---
# @router.get("/audit")
# async def get_audit_log(
#     user=Depends(get_admin_user),
#     page: int = Query(1, ge=1),
#     per_page: int = Query(20, ge=1, le=100),
#     user_id: int = Query(None),
#     date_from: str = Query(None),
# ):
#     """Audit-Log mit Filterung/Paginierung."""
#     config = load_config()
#     from kiara.cloud.audit import AuditLogger
#     audit = AuditLogger(config)
#
#     offset = (page - 1) * per_page
#     entries = audit.get_log(
#         limit=per_page, offset=offset,
#         user_id=user_id, date_from=date_from,
#     )
#
#     # Total Count
#     try:
#         with db_query() as db:
#             total = db.execute(
#                 "SELECT COUNT(*) AS count FROM cloud_audit_log"
#             ).fetchone()["count"]
#     except Exception as exc:
#         total = len(entries)
#
#     return {
#         "data": entries,
#         "total": total,
#         "page": page,
#     }

# --- Migriert zu Admin-Registry: cloud.stats (Sprint 09) ---
# @router.get("/stats")
# async def get_cloud_stats(
#     user=Depends(get_admin_user),
#     period: str = Query("month"),
# ):
#     """Aggregierte Statistiken: Requests/Tag, Token-Verbrauch, Kosten-Schaetzung."""
#     config = load_config()
#     from kiara.cloud.audit import AuditLogger
#     audit = AuditLogger(config)
#     return audit.get_stats(period)

# --- Migriert zu Admin-Registry: cloud.test (Sprint 09) ---
# @router.post("/test")
# async def test_cloud_connection(data: CloudTestRequest, user=Depends(get_admin_user)):
#     """Test-Anfrage an Cloud-API (Verbindungstest)."""
#     if not data.api_key:
#         raise HTTPException(status_code=422, detail="API-Key ist erforderlich")
#
#     from kiara.cloud.client import CloudKIClient
#
#     client = CloudKIClient(
#         provider=data.provider,
#         api_key=data.api_key,
#         model=data.model or (DEFAULT_OPENAI_MODEL if data.provider == "openai" else DEFAULT_CLOUD_MODEL),
#         base_url=data.base_url,
#         max_tokens=10,
#     )
#
#     try:
#         response = await client.generate(
#             prompt="Antworte mit einem einzigen Wort: OK",
#             temperature=0.0,
#         )
#         return {
#             "success": True,
#             "response": response.strip(),
#             "model": client.model,
#             "provider": client.provider,
#         }
#     except Exception as exc:
#         raise HTTPException(
#             status_code=502,
#             detail=f"Cloud-Verbindung fehlgeschlagen: {exc}",
#         )
