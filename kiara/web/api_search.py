"""
KIara - Search API-Router
Textsuche (Elasticsearch) und Semantische Suche (Qdrant).
"""

import logging
import re
import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from kiara.config import load_config
from kiara.shared.constants import (
    ES_MAX_RESULT_WINDOW,
    SEARCH_MAX_PAGE_SIZE,
)
from kiara.web.api_chat import _build_datasource_filter
from kiara.web.auth import get_chat_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DIRECTORY_WILDCARD_RE = re.compile(r"[*?]")


class SearchRequest(BaseModel):
    """Request fuer die Dokumentensuche."""

    query: str = Field(..., max_length=10_000)
    mode: str = "text"
    page: int = 1
    page_size: int = 10
    embedding_model: str | None = Field(None, max_length=255)
    filter_datasource_ids: list[int] | None = None
    filter_datasource_exclude: bool = False
    filter_type: str | None = Field(None, max_length=255)
    filter_dir: str | None = Field(None, max_length=2048)
    filter_date_from: str | None = None
    filter_date_to: str | None = None


def _normalize_filter_date_to(value: str | None) -> str | None:
    """Normalisiert date-only Enddaten auf das Ende des ausgewaehlten Tages."""
    if value and _DATE_ONLY_RE.match(value):
        end_of_day = datetime.fromisoformat(value) + timedelta(days=1) - timedelta(microseconds=1)
        return end_of_day.isoformat()
    return value


def _validate_directory_filter(value: str | None) -> None:
    """Directory-Filter ist ein literal path fragment, keine Wildcard-Sprache."""
    if value and _DIRECTORY_WILDCARD_RE.search(value):
        raise HTTPException(
            status_code=400,
            detail="filter_dir akzeptiert keine Wildcards (* oder ?)",
        )


def _directory_filter_clauses(value: str) -> list[dict]:
    """Erzeugt ES-Filter fuer ganze Pfadsegmente statt beliebige Substrings."""
    normalized = value.replace("\\", "/").strip().rstrip("/")
    if not normalized or normalized == "/":
        return []

    if normalized.startswith("/"):
        return [
            {"term": {"metadata.directory": normalized}},
            {"wildcard": {"metadata.directory": f"{normalized}/*"}},
        ]

    segment = normalized.strip("/")
    return [
        {"term": {"metadata.directory": segment}},
        {"wildcard": {"metadata.directory": f"{segment}/*"}},
        {"wildcard": {"metadata.directory": f"*/{segment}"}},
        {"wildcard": {"metadata.directory": f"*/{segment}/*"}},
    ]


def _stable_score(score: float | int | None) -> float:
    """Mappt unbegrenzte ES-Scores monoton auf 0..1 ohne Page-lokalen Max-Wert."""
    try:
        raw = float(score or 0)
    except (TypeError, ValueError):
        return 0.0
    if raw <= 0:
        return 0.0
    return round(raw / (raw + 1.0), 4)


@router.post("/search")
async def search(data: SearchRequest, user=Depends(get_chat_user)):
    """Dokumentensuche mit Text- oder Semantischem Modus."""
    # Validierung
    if not data.query or not data.query.strip():
        raise HTTPException(status_code=400, detail="Suchbegriff erforderlich")
    if data.page < 1:
        raise HTTPException(status_code=400, detail="page muss >= 1 sein")
    if data.page_size < 1 or data.page_size > SEARCH_MAX_PAGE_SIZE:
        raise HTTPException(
            status_code=400, detail=f"page_size muss zwischen 1 und {SEARCH_MAX_PAGE_SIZE} liegen"
        )
    if data.page * data.page_size > ES_MAX_RESULT_WINDOW:
        max_page = ES_MAX_RESULT_WINDOW // data.page_size
        raise HTTPException(
            status_code=400,
            detail=f"Pagination-Limit erreicht (maximal Seite {max_page} bei page_size={data.page_size})",
        )
    if data.mode not in ("text", "semantic"):
        raise HTTPException(status_code=400, detail="mode muss 'text' oder 'semantic' sein")
    _validate_directory_filter(data.filter_dir)
    for _field_name, _field_val in [
        ("filter_date_from", data.filter_date_from),
        ("filter_date_to", data.filter_date_to),
    ]:
        if _field_val is not None:
            try:
                datetime.fromisoformat(_field_val)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail=f"{_field_name} muss ein gueltiges ISO-8601-Datum sein (z.B. 2025-01-15)",
                )

    config = load_config()

    # Frueh-Validierung: Embedding-Modell bei semantischer Suche
    if data.mode == "semantic" and not data.embedding_model:
        from kiara.config import get_embedding_model

        try:
            _emb_model = get_embedding_model(config)
        except RuntimeError:
            raise HTTPException(status_code=503, detail="Kein Embedding-Backend verfuegbar")
        if not _emb_model:
            raise HTTPException(
                status_code=400, detail="Kein Embedding-Modell angegeben oder konfiguriert"
            )

    t_start = time.monotonic()

    # DS-Berechtigungen (identische Logik wie Chat-Endpoint)
    allowed_ds_ids = _build_datasource_filter(
        user, data.filter_datasource_ids, data.filter_datasource_exclude
    )

    if allowed_ds_ids == []:
        results, total = [], 0
    elif data.mode == "text":
        results, total = _search_text(config, data, allowed_ds_ids)
    else:
        results, total = _search_semantic(config, data, allowed_ds_ids)

    query_time_ms = int((time.monotonic() - t_start) * 1000)

    return {
        "results": results,
        "total": total,
        "page": data.page,
        "page_size": data.page_size,
        "mode": data.mode,
        "query_time_ms": query_time_ms,
    }


def _search_text(
    config: dict, data: SearchRequest, allowed_ds_ids: list[int] | None
) -> tuple[list[dict], int]:
    """Textsuche via Elasticsearch."""
    from kiara.retrieval import elasticsearch_service as es_svc

    _validate_directory_filter(data.filter_dir)

    if not es_svc.is_configured(config):
        raise HTTPException(status_code=503, detail="Elasticsearch ist nicht konfiguriert")

    try:
        es = es_svc._get_client(config)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Elasticsearch-Paket nicht installiert. Bitte 'pip install elasticsearch' ausfuehren.",
        )
    except Exception as exc:
        logger.error("Elasticsearch nicht erreichbar: %s", exc)
        raise HTTPException(status_code=503, detail="Elasticsearch nicht erreichbar")

    index_name = config.get("retrieval", {}).get("elasticsearch_index", "kiara")
    raw_fuzziness = config.get("retrieval", {}).get("elasticsearch_fuzziness", "none")
    fuzziness = None if raw_fuzziness in ("none", None, "") else raw_fuzziness

    # ES-Query aufbauen
    match_params = {
        "query": data.query,
        "fields": ["text", "metadata.source^2"],
    }
    if fuzziness is not None:
        match_params["fuzziness"] = fuzziness
    must = [{"multi_match": match_params}]
    es_filter = []

    # DS-Filter direkt in ES-Query (ds_ids ist im ES-Index als metadata.ds_ids)
    if allowed_ds_ids is not None:
        es_filter.append({"terms": {"metadata.ds_ids": allowed_ds_ids}})

    # Dateityp
    if data.filter_type:
        es_filter.append({"term": {"metadata.file_type": data.filter_type}})

    # Verzeichnis: Pfadsegment-/Prefix-Match. Wildcard-Zeichen werden auf
    # Route-Ebene abgelehnt, damit User-Eingaben keine ES-Wildcards steuern.
    if data.filter_dir:
        clauses = _directory_filter_clauses(data.filter_dir)
        if clauses:
            es_filter.append(
                {
                    "bool": {
                        "should": clauses,
                        "minimum_should_match": 1,
                    }
                }
            )

    # Datumsfilter
    date_range = {}
    if data.filter_date_from:
        date_range["gte"] = data.filter_date_from
    if data.filter_date_to:
        date_range["lte"] = _normalize_filter_date_to(data.filter_date_to)
    if date_range:
        es_filter.append({"range": {"metadata.modified": date_range}})

    fetch_size = data.page_size
    body = {
        "query": {"bool": {"must": must, "filter": es_filter}}
        if es_filter
        else {"bool": {"must": must}},
        "from": (data.page - 1) * data.page_size,
        "size": fetch_size,
    }

    try:
        resp = es.search(index=index_name, body=body)
    except Exception as exc:
        logger.error("Elasticsearch-Fehler: %s", exc)
        raise HTTPException(status_code=503, detail="Elasticsearch-Fehler")

    hits = resp.get("hits", {})
    total = hits.get("total", {}).get("value", 0)
    es_hits = hits.get("hits", [])
    results = []
    for hit in es_hits:
        src = hit.get("_source", {})
        meta = src.get("metadata", {})
        score = hit.get("_score", 0)
        results.append(
            {
                "id": hit.get("_id", ""),
                "text": src.get("text", ""),
                "score": _stable_score(score),
                "metadata": {
                    "source": meta.get("source", ""),
                    "directory": meta.get("directory", ""),
                    "file_type": meta.get("file_type", ""),
                    "modified": meta.get("modified", ""),
                },
            }
        )

    return results, total


def _search_semantic(
    config: dict, data: SearchRequest, allowed_ds_ids: list[int] | None
) -> tuple[list[dict], int]:
    """Semantische Suche via Qdrant."""
    from kiara.backend_selector import BackendUnavailableError
    from kiara.config import get_embedding_model

    _validate_directory_filter(data.filter_dir)

    from kiara.backend_selector import normalize_model_name

    model = normalize_model_name(data.embedding_model or get_embedding_model(config))
    if not model:
        raise HTTPException(
            status_code=400, detail="Kein Embedding-Modell angegeben oder konfiguriert"
        )

    from kiara.retrieval.vectorstore import VectorStore

    try:
        available_models = {c["model"] for c in VectorStore.list_collections(config)}
    except Exception as exc:
        logger.error("Vector-Store nicht erreichbar: %s", exc)
        raise HTTPException(status_code=503, detail="Vector-Store nicht erreichbar")
    if model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Embedding-Modell '{model}' hat keine indexierten Daten",
        )

    # Qdrant-Filter aufbauen
    conditions = []
    if allowed_ds_ids is not None:
        conditions.append({"ds_ids": {"$in": allowed_ds_ids}})
    if data.filter_type:
        conditions.append({"file_type": data.filter_type})
    if data.filter_dir:
        conditions.append({"directory": {"$contains": data.filter_dir}})
    date_cond = {}
    if data.filter_date_from:
        date_cond["$gte"] = data.filter_date_from
    if data.filter_date_to:
        date_cond["$lte"] = _normalize_filter_date_to(data.filter_date_to)
    if date_cond:
        conditions.append({"modified": date_cond})

    where = {"$and": conditions} if conditions else None

    try:
        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(config, embedding_model=model)
        offset = (data.page - 1) * data.page_size
        raw_results = store.search(
            query_text=data.query,
            k=data.page_size,
            offset=offset,
            where=where,
        )
        if len(raw_results) < data.page_size:
            total = offset + len(raw_results)
        else:
            total = store.count_filtered(where)
    except BackendUnavailableError:
        raise HTTPException(status_code=503, detail="Embedding-Backend nicht erreichbar")
    except Exception as exc:
        logger.error("Semantische Suche fehlgeschlagen: %s", exc)
        raise HTTPException(status_code=503, detail="Semantische Suche fehlgeschlagen")

    results = []
    for r in raw_results:
        meta = r.get("metadata", {})
        results.append(
            {
                "id": r["_doc_id"],
                "text": r.get("text", ""),
                "score": round(r.get("score", 0), 4),
                "metadata": {
                    "source": meta.get("source", ""),
                    "directory": meta.get("directory", ""),
                    "file_type": meta.get("file_type", ""),
                    "modified": meta.get("modified", ""),
                },
            }
        )

    return results, total
