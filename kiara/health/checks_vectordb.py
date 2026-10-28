"""VectorDB Health-Checks (Qdrant)."""

import logging
import time

import httpx

from kiara.config import DEFAULT_TIMEOUTS
from kiara.health.models import ComponentHealth
from kiara.shared.constants import DEFAULT_QDRANT_URL
from kiara.shared.enums import HealthStatus

logger = logging.getLogger(__name__)


async def check_vectordb(config: dict) -> ComponentHealth:
    """VectorDB-Backend pruefen."""
    start = time.time()

    try:
        return await _check_qdrant(config, start)
    except Exception as exc:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="vectordb",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 1),
            message=f"VectorDB (qdrant) nicht erreichbar: {exc}",
        )


async def _check_qdrant(config: dict, start: float) -> ComponentHealth:
    """Qdrant-spezifischer Health-Check: GET {qdrant_url}/collections."""
    vectordb_cfg = config.get("vectordb", {})
    qdrant_url = vectordb_cfg.get("qdrant_url", DEFAULT_QDRANT_URL)
    qdrant_api_key = vectordb_cfg.get("qdrant_api_key", "")
    headers = {"api-key": qdrant_api_key} if qdrant_api_key else {}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUTS["http_health"]) as client:
            resp = await client.get(f"{qdrant_url}/collections", headers=headers)
            latency = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            collections = (data.get("result") or {}).get("collections", [])
            kb_colls = [c for c in collections if c.get("name", "").startswith("kb_")]
            return ComponentHealth(
                name="vectordb",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 1),
                message=f"Qdrant OK, {len(kb_colls)} kb_*-Collection(s)",
                details={
                    "backend": "qdrant",
                    "url": qdrant_url,
                    "collections": len(collections),
                },
            )
        else:
            return ComponentHealth(
                name="vectordb",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency, 1),
                message=f"Qdrant HTTP {resp.status_code}",
                details={"backend": "qdrant", "url": qdrant_url},
            )

    except Exception as exc:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="vectordb",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 1),
            message=f"Qdrant nicht erreichbar: {exc}",
            details={"backend": "qdrant", "url": qdrant_url},
        )
