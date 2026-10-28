"""GraphRAG Admin API."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from kiara.config import load_config
from kiara.graph.store import GraphStore
from kiara.web.auth import get_admin_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin/graph", tags=["graph"])


def _get_graph_store() -> GraphStore:
    load_config()
    return GraphStore()


@router.get("/stats")
def graph_stats(user=Depends(get_admin_user)):
    """Gibt Graph-Statistiken zurueck."""
    store = _get_graph_store()
    return store.get_stats()


@router.post("/sync-erp")
async def sync_erp(user=Depends(get_admin_user)):
    """Importiert ERP-Stammdaten in den Graphen."""
    config = load_config()
    if not config.get("graph", {}).get("erp_sync_enabled", True):
        logger.info("ERP-Sync abgelehnt: graph.erp_sync_enabled=False")
        raise HTTPException(
            status_code=503,
            detail="ERP-Graph-Sync ist per Konfiguration deaktiviert (graph.erp_sync_enabled).",
        )
    store = _get_graph_store()
    from kiara.graph.erp_sync import (
        ERPGraphSync,
        ERPSyncAlreadyRunningError,
        ERPSyncConfigurationError,
    )
    from kiara.tools.erp_backend import ERPUpstreamError
    from kiara.tools.erp_connectors.registry import ERPAssignmentBrokenError

    try:
        sync = ERPGraphSync(config, store)
        result = await sync.sync_all()
    except ERPSyncAlreadyRunningError as exc:
        logger.info("ERP-Sync abgelehnt: %s", exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ERPSyncConfigurationError as exc:
        logger.warning("ERP-Sync-Konfigurationsfehler: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=str(exc) or "ERP-Graph-Sync ist nicht korrekt konfiguriert.",
        ) from exc
    except ERPAssignmentBrokenError as exc:
        logger.warning("ERP-Sync-Konfigurationsfehler: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="ERP-Graph-Sync ist nicht korrekt konfiguriert.",
        ) from exc
    except NotImplementedError as exc:
        logger.warning("ERP-Sync nicht unterstuetzt: %s", exc)
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except ERPUpstreamError as exc:
        logger.warning("ERP-Sync-Upstream-Fehler: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="ERP-Daten konnten nicht geladen werden.",
        ) from exc
    except Exception as exc:
        logger.exception("Unerwarteter Fehler beim ERP-Sync")
        raise HTTPException(
            status_code=500,
            detail="ERP-Sync ist unerwartet fehlgeschlagen.",
        ) from exc
    return {"message": "ERP-Sync abgeschlossen", **result}


@router.post("/clear")
def clear_graph(confirm: str = Query(...), user=Depends(get_admin_user)):
    """Loescht den gesamten Graphen."""
    if confirm != "clear_graph":
        raise HTTPException(status_code=400, detail="Bestaetigung erforderlich")
    store = _get_graph_store()
    deleted = store.clear()
    return {"message": f"{deleted} Knoten geloescht"}


@router.get("/search")
def search_nodes(
    q: str = Query(..., min_length=1, max_length=200),
    entity_type: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    user=Depends(get_admin_user),
):
    """Sucht Knoten im Graphen."""
    store = _get_graph_store()
    return {"nodes": store.find_nodes(q, entity_type, limit)}
