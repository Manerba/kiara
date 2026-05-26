"""
KIara Admin - Dashboard API
KPI-Daten fuer das Admin-Dashboard.
"""

import logging

from fastapi import APIRouter

from kiara.repositories.backend import KiBackendRepository
from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
from kiara.shared.db import db_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["dashboard"])

ds_repo = DatasourceRepository()
de_repo = DatasourceEmbeddingRepository()
backend_repo = KiBackendRepository()


def get_dashboard_kpis(config: dict) -> dict:
    """Sammelt Dashboard-KPIs. Genutzt von /api/admin/dashboard und /api/admin-chat/greeting."""
    return {
        "backends": _get_backends_stats(),
        "datasources": _get_datasources_stats(),
        "vectordb": _get_vectordb_stats(config),
        "jobs": _get_jobs_stats(),
    }


# --- Migriert zu Admin-Registry: dashboard.status (Sprint 04) ---
# @router.get("/dashboard")
# async def get_dashboard(user=Depends(get_admin_user)):
#     """Get all KPI data for the admin dashboard."""
#     config = load_config()
#     return get_dashboard_kpis(config)


def _get_backends_stats() -> dict:
    """KI-Backends: Gesamt/aktiv, gruppiert nach Kategorie."""
    try:
        with db_query() as db:
            total = db.execute("SELECT COUNT(*) AS cnt FROM ki_backends").fetchone()["cnt"]
            active = db.execute(
                "SELECT COUNT(*) AS cnt FROM ki_backends WHERE is_active = true"
            ).fetchone()["cnt"]
            by_category = db.execute(
                "SELECT category, "
                "COUNT(*) AS total, "
                "SUM(CASE WHEN is_active = true THEN 1 ELSE 0 END) AS active "
                "FROM ki_backends GROUP BY category ORDER BY category"
            ).fetchall()
        return {
            "status": "ok",
            "error": None,
            "total": total,
            "active": active,
            "by_category": [
                {"category": r["category"], "total": r["total"], "active": r["active"]}
                for r in by_category
            ],
        }
    except Exception as exc:
        logger.warning("Backend-Stats nicht verfuegbar: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "total": None,
            "active": None,
            "by_category": [],
        }


def _get_datasources_stats() -> dict:
    """Datenquellen: Gesamt/aktiv, gruppiert nach Typ."""
    try:
        total = ds_repo.count()
        active = ds_repo.count({"is_active": True})
        with db_query() as db:
            by_type = db.execute(
                "SELECT type, "
                "COUNT(*) AS total, "
                "SUM(CASE WHEN is_active = true THEN 1 ELSE 0 END) AS active "
                "FROM datasources GROUP BY type ORDER BY type"
            ).fetchall()
        return {
            "status": "ok",
            "error": None,
            "total": total,
            "active": active,
            "by_type": [
                {"type": r["type"], "total": r["total"], "active": r["active"]} for r in by_type
            ],
        }
    except Exception as exc:
        logger.warning("Datasource-Stats nicht verfuegbar: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "total": None,
            "active": None,
            "by_type": [],
        }


def _get_vectordb_stats(config: dict) -> dict:
    """VektorDB: Groesse, Chunks nach Embedding-Modell."""
    result = {
        "status": "ok",
        "size_bytes": 0,
        "chunks_by_model": [],
        "total_unique_chunks": 0,
        "error": None,
    }

    # DB-Groesse aus Qdrant-Datenverzeichnis
    try:
        import os

        vdb_path = config.get("vectordb", {}).get("path", "")
        if vdb_path and os.path.isdir(vdb_path):
            total_size = 0
            for dirpath, _dirnames, filenames in os.walk(vdb_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass
            result["size_bytes"] = total_size
    except Exception as exc:
        logger.warning("VektorDB Groesse konnte nicht ermittelt werden: %s", exc)

    # Chunks nach Embedding-Modell
    try:
        with db_query() as db:
            rows = db.execute(
                "SELECT embedding_model, "
                "SUM(chunk_count) AS chunks "
                "FROM datasource_embeddings "
                "WHERE embedding_model != '' "
                "GROUP BY embedding_model ORDER BY embedding_model"
            ).fetchall()
        result["chunks_by_model"] = [
            {"model": r["embedding_model"], "chunks": r["chunks"] or 0, "kind": "kb"} for r in rows
        ]

        # ColBERT-Collections ergaenzen
        try:
            from kiara.retrieval.colbert import ColBERTStore

            for coll in ColBERTStore.list_colbert_collections(config):
                result["chunks_by_model"].append(
                    {
                        "model": coll["model"],
                        "chunks": coll["chunks"],
                        "kind": "colbert",
                    }
                )
        except Exception:
            logger.debug("ColBERT collection listing failed", exc_info=True)

        # total_unique_chunks ohne Dense/ColBERT-Doppelzaehlung
        try:
            with db_query() as db2:
                _unique = db2.execute(
                    "SELECT COUNT(DISTINCT chunk_id) AS total FROM chunk_datasources"
                ).fetchone()
            result["total_unique_chunks"] = _unique["total"] if _unique else 0
        except Exception as exc:
            logger.warning(
                "Unique-Chunk-Count fehlgeschlagen, Fallback auf Summe pro Modell "
                "(Doppelzaehlung moeglich): %s",
                exc,
            )
            result["total_unique_chunks"] = sum(
                m["chunks"] for m in result["chunks_by_model"] if m.get("kind") == "kb"
            )

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        logger.warning("VektorDB Chunk-Stats fehlgeschlagen: %s", exc)

    return result


def _get_jobs_stats() -> dict:
    """Jobs: Laufender Job + Warteschlange."""
    try:
        with db_query() as db:
            running = db.execute(
                "SELECT j.id, j.datasource_id, j.status, j.progress_processed, j.progress_total, "
                "d.name AS datasource_name "
                "FROM ingestion_jobs j "
                "LEFT JOIN datasources d ON j.datasource_id = d.id "
                "WHERE j.status IN ('running', 'paused') "
                "ORDER BY j.id DESC LIMIT 1"
            ).fetchone()
            pending_count = db.execute(
                "SELECT COUNT(*) AS cnt FROM ingestion_jobs WHERE status = 'pending'"
            ).fetchone()["cnt"]
        running_info = None
        if running:
            running_info = {
                "id": running["id"],
                "status": running["status"],
                "datasource": running["datasource_name"] or f"DS #{running['datasource_id']}",
                "processed": running["progress_processed"] or 0,
                "total": running["progress_total"] or 0,
            }
        return {
            "status": "ok",
            "error": None,
            "running": running_info,
            "pending": pending_count,
        }
    except Exception as exc:
        logger.warning("Job-Stats nicht verfuegbar: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "running": None,
            "pending": None,
        }
