"""Admin API for benchmark result artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException

from kiara.benchmarks.beir import (
    DEFAULT_DATASET_DIR,
    DEFAULT_K_VALUES,
    evaluate_beir_retrieval,
    parse_k_values,
)
from kiara.shared.db import db_query
from kiara.web.auth import get_admin_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/benchmarks", tags=["benchmarks"])

BENCHMARK_BASE_DIR = Path("/var/lib/kiara/benchmarks")
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
MAX_DETAILS_DERIVE_BYTES = 50 * 1024 * 1024
_beir_task_lock = threading.RLock()
_beir_task: dict[str, Any] = {
    "running": False,
    "status": "idle",
    "task_id": None,
    "started_at": None,
    "finished_at": None,
    "progress": 0,
    "total": 0,
    "current_query_id": None,
    "last_latency_ms": None,
    "last_rerank_latency_ms": None,
    "last_first_relevant_rank": None,
    "last_retrieval_failed": None,
    "last_colbert_failed": None,
    "last_colbert_result_count": None,
    "settings": None,
    "result": None,
    "error": None,
}


def _artifact_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]


def _relative_path(path: Path, base_dir: Path) -> str:
    return path.resolve().relative_to(base_dir.resolve()).as_posix()


def _dataset_name(relative_path: str) -> str:
    parts = [part for part in relative_path.split("/") if part]
    return parts[0] if parts else ""


def _modified_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _artifact_timestamp(path: Path, payload: dict[str, Any]) -> str:
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    for value in (
        payload.get("created_at"),
        run.get("finished_at"),
        run.get("started_at"),
    ):
        if value:
            return str(value)
    return _modified_iso(path)


def _read_json_artifact(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Artefakt ist zu gross ({size} Bytes)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Artefakt enthaelt kein JSON-Objekt")
    return payload


def _metric_value(payload: dict[str, Any], key: str) -> float:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    value = overall.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _average_numeric(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile_numeric(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _numeric_field(row: dict[str, Any], key: str) -> float | None:
    candidate_stats = row.get("candidate_stats")
    if isinstance(candidate_stats, dict) and candidate_stats.get(key) is not None:
        value = candidate_stats.get(key)
    else:
        value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _details_path_for_summary(summary_path: Path, payload: dict[str, Any]) -> Path | None:
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    raw_path = artifacts.get("details")
    candidates: list[Path] = []
    if raw_path:
        candidates.append(Path(str(raw_path)))
    if summary_path.name.endswith("_summary.json"):
        candidates.append(
            summary_path.with_name(summary_path.name.replace("_summary.json", "_details.jsonl"))
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _derive_beir_ranking_diagnostics(
    summary_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort Diagnose fuer BEIR-Runs vor ranking_diagnostics."""

    details_path = _details_path_for_summary(summary_path, payload)
    if details_path is None or details_path.stat().st_size > MAX_DETAILS_DERIVE_BYTES:
        return {}

    rows = 0
    first_relevant_ranks: list[float] = []
    retrieved_chunks: list[float] = []
    mapped_chunks: list[float] = []
    unmapped_chunks: list[float] = []
    duplicate_doc_chunks: list[float] = []
    unique_docs_seen: list[float] = []
    top_docs_returned: list[float] = []

    with details_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            rows += 1

            ranking = row.get("ranking") if isinstance(row.get("ranking"), dict) else {}
            first_rank = ranking.get("first_relevant_rank")
            if first_rank is None:
                relevant = set(row.get("relevant_doc_ids") or [])
                top_docs = row.get("top_docs") if isinstance(row.get("top_docs"), list) else []
                for rank, doc in enumerate(top_docs, 1):
                    if isinstance(doc, dict) and doc.get("doc_id") in relevant:
                        first_rank = rank
                        break
            if first_rank is not None:
                try:
                    first_relevant_ranks.append(float(first_rank))
                except (TypeError, ValueError):
                    pass

            for key, target in (
                ("retrieved_chunks", retrieved_chunks),
                ("mapped_chunks", mapped_chunks),
                ("unmapped_chunks", unmapped_chunks),
                ("duplicate_doc_chunks", duplicate_doc_chunks),
                ("unique_docs_seen", unique_docs_seen),
            ):
                value = _numeric_field(row, key)
                if value is not None:
                    target.append(value)

            top_docs_returned_value = _numeric_field(row, "top_docs_returned")
            if top_docs_returned_value is not None:
                top_docs_returned.append(top_docs_returned_value)
            else:
                top_docs = row.get("top_docs") if isinstance(row.get("top_docs"), list) else []
                top_docs_returned.append(float(len(top_docs)))

    if rows == 0:
        return {}

    return {
        "legacy_derived": True,
        "first_relevant_rank_min": min(first_relevant_ranks) if first_relevant_ranks else None,
        "first_relevant_rank_median": _percentile_numeric(first_relevant_ranks, 50),
        "first_relevant_rank_p90": _percentile_numeric(first_relevant_ranks, 90),
        "queries_without_relevant_at_max_k": rows - len(first_relevant_ranks),
        "avg_retrieved_chunks": _average_numeric(retrieved_chunks),
        "avg_mapped_chunks": _average_numeric(mapped_chunks),
        "avg_unmapped_chunks": _average_numeric(unmapped_chunks),
        "avg_duplicate_doc_chunks": _average_numeric(duplicate_doc_chunks),
        "duplicate_doc_chunks_available": bool(duplicate_doc_chunks),
        "avg_unique_docs_seen": _average_numeric(unique_docs_seen),
        "avg_top_docs_returned": _average_numeric(top_docs_returned),
    }


def _summary_record(path: Path, base_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    relative = _relative_path(path, base_dir)
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    label = str(run.get("label") or path.stem.removesuffix("_summary"))
    return {
        "id": _artifact_id(relative),
        "type": "summary",
        "label": label,
        "dataset": _dataset_name(relative),
        "path": str(path),
        "relative_path": relative,
        "modified": _modified_iso(path),
        "created_at": payload.get("created_at"),
        "timestamp": _artifact_timestamp(path, payload),
        "run": run,
        "overall": overall,
        "breakdowns": payload.get("breakdowns")
        if isinstance(payload.get("breakdowns"), dict)
        else {},
    }


def _beir_summary_record(path: Path, base_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    relative = _relative_path(path, base_dir)
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    ranking_diagnostics = (
        payload.get("ranking_diagnostics")
        if isinstance(payload.get("ranking_diagnostics"), dict)
        else {}
    )
    retrieval_failures = (
        payload.get("retrieval_failures")
        if isinstance(payload.get("retrieval_failures"), dict)
        else {}
    )
    colbert_diagnostics = (
        payload.get("colbert_diagnostics")
        if isinstance(payload.get("colbert_diagnostics"), dict)
        else {}
    )
    if not ranking_diagnostics:
        ranking_diagnostics = _derive_beir_ranking_diagnostics(path, payload)
    label = str(run.get("label") or path.stem.removesuffix("_summary"))
    return {
        "id": _artifact_id(relative),
        "type": "beir-summary",
        "label": label,
        "dataset": payload.get("dataset") or _dataset_name(relative),
        "path": str(path),
        "relative_path": relative,
        "modified": _modified_iso(path),
        "created_at": payload.get("created_at"),
        "timestamp": _artifact_timestamp(path, payload),
        "split": payload.get("split"),
        "run": run,
        "settings": settings,
        "overall": overall,
        "status": run.get("status", "success"),
        "valid": run.get("valid", True),
        "ranking_diagnostics": ranking_diagnostics,
        "retrieval_failures": retrieval_failures,
        "colbert_diagnostics": colbert_diagnostics,
        "key_metrics": {
            "recall@10": _metric_value(payload, "recall@10"),
            "ndcg@10": _metric_value(payload, "ndcg@10"),
            "mrr@10": _metric_value(payload, "mrr@10"),
            "hitrate@10": _metric_value(payload, "hitrate@10"),
            "recall@100": _metric_value(payload, "recall@100"),
        },
        "artifacts": payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {},
    }


def _comparison_record(path: Path, base_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    relative = _relative_path(path, base_dir)
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    return {
        "id": _artifact_id(relative),
        "type": "comparison",
        "label": path.stem.removesuffix("_comparison"),
        "dataset": _dataset_name(relative),
        "path": str(path),
        "relative_path": relative,
        "modified": _modified_iso(path),
        "baseline_index": payload.get("baseline_index", 0),
        "baseline_label": payload.get("baseline_label", ""),
        "runs": runs,
    }


def _collect_crag_artifacts(base_dir: Path = BENCHMARK_BASE_DIR) -> dict[str, Any]:
    """Collect CRAG benchmark summaries without requiring DB state."""

    base = Path(base_dir)
    if not base.exists():
        return {
            "base_dir": str(base),
            "summaries": [],
            "comparisons": [],
            "errors": [],
        }

    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in sorted(base.rglob("*_summary.json")):
        try:
            payload = _read_json_artifact(path)
            if payload.get("schema") != "kiara-crag-summary":
                continue
            summaries.append(_summary_record(path, base, payload))
        except Exception as exc:
            log.warning("CRAG-Summary konnte nicht gelesen werden (%s): %s", path, exc)
            errors.append({"path": str(path), "error": str(exc)})

    for path in sorted(base.rglob("*_comparison.json")):
        try:
            payload = _read_json_artifact(path)
            if payload.get("schema") != "kiara-crag-comparison":
                continue
            comparisons.append(_comparison_record(path, base, payload))
        except Exception as exc:
            log.warning("CRAG-Vergleich konnte nicht gelesen werden (%s): %s", path, exc)
            errors.append({"path": str(path), "error": str(exc)})

    summaries.sort(key=lambda item: item.get("timestamp") or item["modified"], reverse=True)
    comparisons.sort(key=lambda item: item["modified"], reverse=True)
    return {
        "base_dir": str(base),
        "summaries": summaries,
        "comparisons": comparisons,
        "errors": errors,
    }


def _beir_base_dir(base_dir: Path = BENCHMARK_BASE_DIR) -> Path:
    return Path(base_dir) / "beir"


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _collect_beir_datasets(base_dir: Path = BENCHMARK_BASE_DIR) -> list[dict[str, Any]]:
    base = _beir_base_dir(base_dir)
    if not base.exists():
        return []
    datasets: list[dict[str, Any]] = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
        try:
            manifest = _read_json_artifact(manifest_path)
            dataset_dir = manifest_path.parent
            documents_dir = Path(manifest.get("documents_dir") or dataset_dir / "documents")
            datasets.append(
                {
                    "dataset": manifest.get("dataset") or dataset_dir.name,
                    "split": manifest.get("split") or "",
                    "dataset_dir": str(dataset_dir),
                    "documents_dir": str(documents_dir),
                    "documents_exists": documents_dir.exists(),
                    "queries": _count_jsonl_rows(dataset_dir / "queries.jsonl"),
                    "qrels": _count_jsonl_rows(dataset_dir / "qrels.jsonl"),
                    "manifest_path": str(manifest_path),
                    "modified": _modified_iso(manifest_path),
                }
            )
        except Exception as exc:
            log.warning("BEIR-Manifest konnte nicht gelesen werden (%s): %s", manifest_path, exc)
    return datasets


def _collect_beir_datasources() -> list[dict[str, Any]]:
    with db_query() as db:
        rows = db.execute(
            """
            SELECT d.id, d.name, d.type, d.path, d.is_active,
                   de.embedding_model, de.chunk_count, de.file_count, de.last_ingested_at
            FROM datasources d
            LEFT JOIN datasource_embeddings de ON de.datasource_id = d.id
            WHERE d.name ILIKE %s OR d.path LIKE %s
            ORDER BY d.id, de.embedding_model
            """,
            ("%BEIR%", f"{_beir_base_dir()}%"),
        ).fetchall()
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        ds_id = int(row["id"])
        item = grouped.setdefault(
            ds_id,
            {
                "id": ds_id,
                "name": row.get("name"),
                "type": row.get("type"),
                "path": row.get("path"),
                "is_active": row.get("is_active"),
                "embeddings": [],
            },
        )
        if row.get("embedding_model"):
            last_ingested_at = row.get("last_ingested_at")
            item["embeddings"].append(
                {
                    "embedding_model": row.get("embedding_model"),
                    "chunk_count": row.get("chunk_count") or 0,
                    "file_count": row.get("file_count") or 0,
                    "last_ingested_at": last_ingested_at.isoformat()
                    if hasattr(last_ingested_at, "isoformat")
                    else str(last_ingested_at)
                    if last_ingested_at is not None
                    else None,
                }
            )
    return list(grouped.values())


def _collect_beir_artifacts(base_dir: Path = BENCHMARK_BASE_DIR) -> dict[str, Any]:
    base = _beir_base_dir(base_dir)
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if base.exists():
        for path in sorted(base.rglob("*_summary.json")):
            try:
                payload = _read_json_artifact(path)
                if payload.get("schema") != "kiara-beir-retrieval-summary":
                    continue
                summaries.append(_beir_summary_record(path, base, payload))
            except Exception as exc:
                log.warning("BEIR-Summary konnte nicht gelesen werden (%s): %s", path, exc)
                errors.append({"path": str(path), "error": str(exc)})
    summaries.sort(key=lambda item: item.get("timestamp") or item["modified"], reverse=True)
    return {
        "base_dir": str(base),
        "datasets": _collect_beir_datasets(base_dir),
        "datasources": _collect_beir_datasources_safe(errors),
        "summaries": summaries,
        "errors": errors,
        "task": _beir_task_snapshot(),
    }


def _collect_beir_datasources_safe(errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    try:
        return _collect_beir_datasources()
    except Exception as exc:
        log.warning("BEIR-Datenquellen konnten nicht gelesen werden: %s", exc)
        errors.append({"path": "datasources", "error": str(exc)})
        return []


def _beir_task_snapshot() -> dict[str, Any]:
    with _beir_task_lock:
        return json.loads(json.dumps(_beir_task, default=str))


def _parse_optional_int(value: Any, field: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field} muss eine Zahl sein") from exc
    if parsed <= 0:
        raise HTTPException(status_code=400, detail=f"{field} muss groesser 0 sein")
    return parsed


def _parse_optional_float(value: Any, field: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field} muss eine Zahl sein") from exc
    if parsed < 0:
        raise HTTPException(status_code=400, detail=f"{field} darf nicht negativ sein")
    return parsed


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "ja"}
    return bool(value)


def _parse_beir_run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(str(payload.get("dataset_dir") or DEFAULT_DATASET_DIR)).resolve()
    if not (dataset_dir / "manifest.json").exists():
        raise HTTPException(status_code=400, detail="BEIR-Dataset ist nicht vorbereitet")
    datasource_id = _parse_optional_int(payload.get("datasource_id"), "datasource_id")
    if datasource_id is None:
        raise HTTPException(
            status_code=400, detail="Datasource-ID ist fuer BEIR-Laeufe erforderlich"
        )
    retriever = str(payload.get("retriever") or "hybrid")
    if retriever not in {"dense", "hybrid"}:
        raise HTTPException(status_code=400, detail="retriever muss 'dense' oder 'hybrid' sein")
    try:
        k_values = parse_k_values(
            str(payload.get("k_values") or ",".join(map(str, DEFAULT_K_VALUES)))
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    chunk_overfetch_factor = _parse_optional_int(
        payload.get("chunk_overfetch_factor") or 3,
        "chunk_overfetch_factor",
    )
    rrf_k = _parse_optional_int(payload.get("rrf_k"), "rrf_k")
    rerank_batch_size = _parse_optional_int(
        payload.get("rerank_batch_size"),
        "rerank_batch_size",
    )
    rerank_candidate_limit = _parse_optional_int(
        payload.get("rerank_candidate_limit"),
        "rerank_candidate_limit",
    )
    doc_aggregation = str(payload.get("doc_aggregation") or "first")
    if doc_aggregation not in {"first", "rank_rrf", "best_plus_bonus"}:
        raise HTTPException(
            status_code=400,
            detail="doc_aggregation muss 'first', 'rank_rrf' oder 'best_plus_bonus' sein",
        )
    doc_aggregation_rrf_k = _parse_optional_int(
        payload.get("doc_aggregation_rrf_k") or 60,
        "doc_aggregation_rrf_k",
    )
    doc_aggregation_bonus = _parse_optional_float(
        payload.get("doc_aggregation_bonus") or 0.01,
        "doc_aggregation_bonus",
    )
    doc_aggregation_max_bonus_chunks = _parse_optional_int(
        payload.get("doc_aggregation_max_bonus_chunks") or 3,
        "doc_aggregation_max_bonus_chunks",
    )
    return {
        "dataset_dir": dataset_dir,
        "datasource_id": datasource_id,
        "retriever_name": retriever,
        "rrf_k": rrf_k,
        "elasticsearch_fuzziness": str(payload.get("elasticsearch_fuzziness")).strip()
        if payload.get("elasticsearch_fuzziness") not in (None, "")
        else None,
        "k_values": k_values,
        "max_queries": _parse_optional_int(payload.get("max_queries"), "max_queries"),
        "embedding_model": str(payload.get("embedding_model")).strip()
        if payload.get("embedding_model") not in (None, "")
        else None,
        "run_label": str(payload.get("run_label")).strip()
        if payload.get("run_label") not in (None, "")
        else None,
        "use_filter_dir": _parse_bool(payload.get("use_filter_dir"), default=True),
        "chunk_overfetch_factor": chunk_overfetch_factor or 3,
        "reranking_enabled": _parse_bool(payload.get("reranking_enabled"), default=False),
        "rerank_model": str(payload.get("rerank_model")).strip()
        if payload.get("rerank_model") not in (None, "")
        else None,
        "rerank_batch_size": rerank_batch_size,
        "rerank_candidate_limit": rerank_candidate_limit,
        "doc_aggregation": doc_aggregation,
        "doc_aggregation_rrf_k": doc_aggregation_rrf_k or 60,
        "doc_aggregation_bonus": doc_aggregation_bonus
        if doc_aggregation_bonus is not None
        else 0.01,
        "doc_aggregation_max_bonus_chunks": doc_aggregation_max_bonus_chunks or 3,
    }


def _beir_progress_callback(update: dict[str, Any]) -> None:
    with _beir_task_lock:
        _beir_task["progress"] = update.get("current", _beir_task.get("progress", 0))
        _beir_task["total"] = update.get("total", _beir_task.get("total", 0))
        _beir_task["current_query_id"] = update.get("query_id")
        _beir_task["last_latency_ms"] = update.get("latency_ms")
        _beir_task["last_rerank_latency_ms"] = update.get("rerank_latency_ms")
        _beir_task["last_first_relevant_rank"] = update.get("first_relevant_rank")
        _beir_task["last_retrieval_failed"] = update.get("retrieval_failed")
        _beir_task["last_colbert_failed"] = update.get("colbert_failed")
        _beir_task["last_colbert_result_count"] = update.get("colbert_result_count")


def _run_beir_background(settings: dict[str, Any]) -> None:
    try:
        stats = evaluate_beir_retrieval(
            dataset_dir=settings["dataset_dir"],
            datasource_id=settings["datasource_id"],
            embedding_model=settings["embedding_model"],
            retriever_name=settings["retriever_name"],
            rrf_k=settings["rrf_k"],
            elasticsearch_fuzziness=settings["elasticsearch_fuzziness"],
            k_values=settings["k_values"],
            max_queries=settings["max_queries"],
            run_label=settings["run_label"],
            use_filter_dir=settings["use_filter_dir"],
            chunk_overfetch_factor=settings["chunk_overfetch_factor"],
            reranking_enabled=settings["reranking_enabled"],
            rerank_model=settings["rerank_model"],
            rerank_batch_size=settings["rerank_batch_size"],
            rerank_candidate_limit=settings["rerank_candidate_limit"],
            doc_aggregation=settings["doc_aggregation"],
            doc_aggregation_rrf_k=settings["doc_aggregation_rrf_k"],
            doc_aggregation_bonus=settings["doc_aggregation_bonus"],
            doc_aggregation_max_bonus_chunks=settings["doc_aggregation_max_bonus_chunks"],
            progress_callback=_beir_progress_callback,
        )
        summary_path = Path(stats.summary_path)
        payload = _read_json_artifact(summary_path)
        result = _beir_summary_record(summary_path, _beir_base_dir(), payload)
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        run_status = str(run.get("status") or "success")
        with _beir_task_lock:
            _beir_task.update(
                {
                    "running": False,
                    "status": run_status,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "progress": stats.queries_evaluated,
                    "total": stats.queries_evaluated,
                    "result": result,
                    "error": None,
                }
            )
    except Exception as exc:
        log.exception("BEIR-Benchmark-Lauf fehlgeschlagen")
        error_fields: dict[str, Any] = {"error": str(exc)}
        from kiara.retrieval.index_compatibility import IndexCompatibilityError

        if isinstance(exc, IndexCompatibilityError):
            error_fields.update(
                {
                    "reason_code": exc.reason_code,
                    "operation": exc.operation,
                    "consumer": "beir_background",
                    "index_compatibility_error": exc.to_dict(),
                }
            )
        with _beir_task_lock:
            _beir_task.update(
                {
                    "running": False,
                    "status": "error",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "result": None,
                    **error_fields,
                }
            )


@router.get("/crag")
async def get_crag_benchmarks(user: dict = Depends(get_admin_user)) -> dict[str, Any]:
    try:
        return _collect_crag_artifacts()
    except Exception as exc:
        log.exception("CRAG-Benchmark-Artefakte konnten nicht geladen werden")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/beir")
async def get_beir_benchmarks(user: dict = Depends(get_admin_user)) -> dict[str, Any]:
    try:
        return _collect_beir_artifacts()
    except Exception as exc:
        log.exception("BEIR-Benchmark-Artefakte konnten nicht geladen werden")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/beir/status")
async def get_beir_status(user: dict = Depends(get_admin_user)) -> dict[str, Any]:
    return _beir_task_snapshot()


@router.post("/beir/run")
async def start_beir_run(
    payload: dict[str, Any] | None = Body(default=None),
    user: dict = Depends(get_admin_user),
) -> dict[str, Any]:
    payload = payload or {}
    settings = _parse_beir_run_payload(payload)
    with _beir_task_lock:
        if _beir_task.get("running"):
            raise HTTPException(status_code=409, detail="BEIR-Benchmark laeuft bereits")
        task_id = uuid4().hex
        _beir_task.update(
            {
                "running": True,
                "status": "running",
                "task_id": task_id,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "progress": 0,
                "total": 0,
                "current_query_id": None,
                "last_latency_ms": None,
                "last_rerank_latency_ms": None,
                "last_first_relevant_rank": None,
                "last_retrieval_failed": None,
                "last_colbert_failed": None,
                "last_colbert_result_count": None,
                "settings": {
                    "dataset_dir": str(settings["dataset_dir"]),
                    "datasource_id": settings["datasource_id"],
                    "retriever": settings["retriever_name"],
                    "rrf_k": settings["rrf_k"],
                    "elasticsearch_fuzziness": settings["elasticsearch_fuzziness"],
                    "k_values": list(settings["k_values"]),
                    "max_queries": settings["max_queries"],
                    "embedding_model": settings["embedding_model"],
                    "run_label": settings["run_label"],
                    "use_filter_dir": settings["use_filter_dir"],
                    "chunk_overfetch_factor": settings["chunk_overfetch_factor"],
                    "reranking_enabled": settings["reranking_enabled"],
                    "rerank_model": settings["rerank_model"],
                    "rerank_batch_size": settings["rerank_batch_size"],
                    "rerank_candidate_limit": settings["rerank_candidate_limit"],
                    "doc_aggregation": settings["doc_aggregation"],
                    "doc_aggregation_rrf_k": settings["doc_aggregation_rrf_k"],
                    "doc_aggregation_bonus": settings["doc_aggregation_bonus"],
                    "doc_aggregation_max_bonus_chunks": settings[
                        "doc_aggregation_max_bonus_chunks"
                    ],
                },
                "result": None,
                "error": None,
            }
        )
    thread = threading.Thread(
        target=_run_beir_background,
        args=(settings,),
        daemon=True,
        name=f"beir-benchmark-{task_id[:8]}",
    )
    thread.start()
    return _beir_task_snapshot()
