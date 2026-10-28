"""Evaluate KIara retrieval on a prepared BEIR dataset."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import socket
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from kiara.exceptions import ColBERTInitializationError

DEFAULT_DATASET_DIR = Path("/var/lib/kiara/benchmarks/beir/scifact")
DEFAULT_K_VALUES = (1, 3, 5, 10, 100)
DEFAULT_DOC_AGGREGATION_RRF_K = 60
DEFAULT_DOC_AGGREGATION_BONUS = 0.01
DEFAULT_DOC_AGGREGATION_MAX_BONUS_CHUNKS = 3
_RETRIEVAL_DIAGNOSTIC_PATHS = ("semantic", "keyword", "colbert", "graph")
_DIAGNOSTIC_EXCERPT_CHARS = 1000


@dataclass(frozen=True)
class BeirEvalStats:
    dataset: str
    split: str
    run_label: str
    dataset_dir: str
    summary_path: str
    details_path: str
    queries_evaluated: int
    datasource_id: int | None
    retriever: str
    max_k: int


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "run"


def parse_k_values(raw: str) -> tuple[int, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("k values must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one k value is required")
    return tuple(sorted(set(values)))


def load_queries(dataset_dir: Path) -> dict[str, str]:
    return {
        str(row["query_id"]): str(row.get("text") or "")
        for row in iter_jsonl(dataset_dir / "queries.jsonl")
    }


def load_qrels(dataset_dir: Path) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in iter_jsonl(dataset_dir / "qrels.jsonl"):
        score = int(row.get("score") or 0)
        if score <= 0:
            continue
        grouped.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = score
    return grouped


class SourceDocMapper:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir.resolve()
        self._by_absolute: dict[str, str] = {}
        self._by_relative: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        self._ambiguous_names: set[str] = set()
        self._load(dataset_dir / "doc_map.jsonl")

    def _load(self, path: Path) -> None:
        for row in iter_jsonl(path):
            doc_id = str(row["doc_id"])
            relative = str(row["relative_path"])
            absolute = str((self.dataset_dir / relative).resolve())
            self._by_absolute[absolute] = doc_id
            self._by_relative[relative] = doc_id
            self._by_relative[relative.replace("\\", "/")] = doc_id
            name = Path(relative).name
            if name in self._by_name and self._by_name[name] != doc_id:
                self._ambiguous_names.add(name)
            else:
                self._by_name[name] = doc_id

    def doc_id_for_source(self, source: str | None) -> str | None:
        if not source:
            return None
        raw = str(source)
        try:
            path = Path(raw)
        except ValueError:
            return None
        if path.is_absolute():
            resolved = str(path.resolve())
            if resolved in self._by_absolute:
                return self._by_absolute[resolved]
            try:
                relative = str(path.resolve().relative_to(self.dataset_dir)).replace("\\", "/")
            except ValueError:
                relative = ""
            if relative and relative in self._by_relative:
                return self._by_relative[relative]
        normalized = raw.replace("\\", "/")
        if normalized in self._by_relative:
            return self._by_relative[normalized]
        name = Path(normalized).name
        if name and name not in self._ambiguous_names:
            return self._by_name.get(name)
        return None


def _dcg(grades: list[int]) -> float:
    return sum(((2**grade) - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))


def _metrics_for_query(
    ranked_doc_ids: list[str],
    relevant: dict[str, int],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    relevant_ids = set(relevant)
    ideal_grades = sorted(relevant.values(), reverse=True)
    relevant_total = max(len(relevant_ids), 1)
    for k in k_values:
        top = ranked_doc_ids[:k]
        hits = [doc_id for doc_id in top if doc_id in relevant_ids]
        first_rank = next((idx + 1 for idx, doc_id in enumerate(top) if doc_id in relevant_ids), 0)
        grades = [relevant.get(doc_id, 0) for doc_id in top]
        ideal = ideal_grades[:k]
        ideal_dcg = _dcg(ideal)
        metrics[f"recall@{k}"] = len(hits) / relevant_total
        metrics[f"precision@{k}"] = len(hits) / k
        metrics[f"hitrate@{k}"] = 1.0 if hits else 0.0
        metrics[f"mrr@{k}"] = (1.0 / first_rank) if first_rank else 0.0
        metrics[f"ndcg@{k}"] = (_dcg(grades) / ideal_dcg) if ideal_dcg > 0 else 0.0
    return metrics


def _average_metric(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _config_with_retrieval_overrides(
    config: dict[str, Any],
    *,
    rrf_k: int | None,
    elasticsearch_fuzziness: str | None,
) -> dict[str, Any]:
    if rrf_k is None and elasticsearch_fuzziness is None:
        return config
    updated = deepcopy(config)
    retrieval = updated.setdefault("retrieval", {})
    if rrf_k is not None:
        retrieval["rrf_k"] = rrf_k
    if elasticsearch_fuzziness is not None:
        retrieval["elasticsearch_fuzziness"] = elasticsearch_fuzziness
    return updated


def _load_config_with_retrieval_overrides(
    config_path: str | None,
    *,
    rrf_k: int | None,
    elasticsearch_fuzziness: str | None,
) -> dict[str, Any]:
    from kiara.config import load_config

    return _config_with_retrieval_overrides(
        load_config(config_path),
        rrf_k=rrf_k,
        elasticsearch_fuzziness=elasticsearch_fuzziness,
    )


def _instantiate_retriever(
    retriever_name: str,
    config: dict[str, Any],
    *,
    embedding_model: str | None,
):
    from kiara.retrieval.retriever import HybridRetriever, Retriever

    if retriever_name == "dense":
        return Retriever(config, embedding_model=embedding_model)
    if retriever_name == "hybrid":
        return HybridRetriever(config, embedding_model=embedding_model)
    raise ValueError("retriever must be 'dense' or 'hybrid'")


def _build_retriever(
    retriever_name: str,
    *,
    config_path: str | None,
    embedding_model: str | None,
    rrf_k: int | None,
    elasticsearch_fuzziness: str | None,
):
    config = _load_config_with_retrieval_overrides(
        config_path,
        rrf_k=rrf_k,
        elasticsearch_fuzziness=elasticsearch_fuzziness,
    )
    return _instantiate_retriever(
        retriever_name,
        config,
        embedding_model=embedding_model,
    ), config


def _effective_embedding_model(config: dict[str, Any], requested: str | None) -> str:
    return str(
        requested
        or config.get("embedding", {}).get("model")
        or config.get("ollama", {}).get("embedding_model")
        or ""
    )


def _datasource_snapshot(datasource_id: int | None) -> dict[str, Any] | None:
    if datasource_id is None:
        return None
    try:
        from kiara.shared.db import db_query

        with db_query() as db:
            rows = db.execute(
                """
                SELECT id, name, type, path, is_active
                FROM datasources
                WHERE id = %s
                """,
                (datasource_id,),
            ).fetchall()
        if not rows:
            return {"id": datasource_id, "missing": True}
        row = dict(rows[0])
        return {
            "id": row.get("id"),
            "name": row.get("name"),
            "type": row.get("type"),
            "path": row.get("path"),
            "is_active": row.get("is_active"),
        }
    except Exception as exc:
        return {"id": datasource_id, "error": str(exc)}


def _retrieval_settings_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    keys = (
        "hybrid_search",
        "rrf_k",
        "elasticsearch_url",
        "elasticsearch_index",
        "elasticsearch_fuzziness",
        "reranking",
        "rerank_model",
        "rerank_top_k",
        "rerank_fetch_multiplier",
        "rerank_batch_size",
        "query_expansion",
        "max_context_chunks",
        "agent_dedup",
        "top_k",
    )
    snapshot = {key: retrieval.get(key) for key in keys if key in retrieval}
    if "elasticsearch_index" in snapshot:
        snapshot["elasticsearch_index_prefix"] = snapshot.pop("elasticsearch_index")
    colbert = retrieval.get("colbert")
    if isinstance(colbert, dict):
        snapshot["colbert"] = {
            key: colbert.get(key) for key in ("enabled", "model", "url") if key in colbert
        }
    return snapshot


def _embedding_settings_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    embedding = config.get("embedding") if isinstance(config.get("embedding"), dict) else {}
    ollama = config.get("ollama") if isinstance(config.get("ollama"), dict) else {}
    snapshot = {
        "embedding_model": embedding.get("model"),
        "ollama_embedding_model": ollama.get("embedding_model"),
        "ollama_base_url": ollama.get("base_url"),
    }
    return {key: value for key, value in snapshot.items() if value not in (None, "")}


def _retriever_runtime_snapshot(retriever: Any) -> dict[str, Any]:
    attrs = {
        "class": f"{type(retriever).__module__}.{type(retriever).__name__}",
        "embedding_model": getattr(retriever, "_embed_model_name", None),
        "hybrid_mode": getattr(retriever, "_hybrid_mode", None),
        "rrf_k": getattr(retriever, "_rrf_k", None),
        "bm25_enabled": getattr(retriever, "_bm25_enabled", None),
        "bm25_ttl": getattr(retriever, "_bm25_ttl", None),
        "bm25_max_corpus": getattr(retriever, "_bm25_max_corpus", None),
        "elasticsearch_enabled": getattr(retriever, "_es_enabled", None),
        "elasticsearch_url": getattr(retriever, "_es_url", None),
        "elasticsearch_index_prefix": getattr(retriever, "_es_index_prefix", None),
        "elasticsearch_fuzziness": getattr(retriever, "_es_fuzziness", None),
        "graph_enabled": getattr(retriever, "_graph_enabled", None),
        "colbert_enabled": getattr(retriever, "_colbert_enabled", None),
        "colbert_model": getattr(retriever, "_colbert_model", None),
    }
    return {key: value for key, value in attrs.items() if value is not None}


def _retriever_colbert_diagnostics(retriever: Any) -> dict[str, Any] | None:
    getter = getattr(retriever, "get_last_colbert_diagnostics", None)
    if not callable(getter):
        return None
    try:
        diagnostics = getter()
    except Exception as exc:
        return {
            "diagnostics_failed": True,
            "attempted": True,
            "failed": True,
            "result_count": 0,
            "latency_ms": 0,
            "error_type": "DiagnosticsError",
            "error": str(exc),
        }
    if not isinstance(diagnostics, dict):
        return None
    return dict(diagnostics)


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        name = str(value)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _colbert_diagnostics_summary(
    retriever: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    attempted = [row for row in rows if row.get("attempted")]
    failed = [row for row in rows if row.get("failed")]
    successful = [row for row in attempted if not row.get("failed")]
    diagnostic_failures = [row for row in rows if row.get("diagnostics_failed")]
    latencies = [int(row.get("latency_ms") or 0) for row in attempted]
    result_counts = [int(row.get("result_count") or 0) for row in attempted]
    enabled_effective = bool(getattr(retriever, "_colbert_enabled", False))
    if not enabled_effective:
        status = "disabled"
    elif not attempted:
        status = "unattempted"
    elif failed and len(failed) == len(attempted):
        status = "invalid"
    elif failed:
        status = "partial"
    else:
        status = "success"
    return {
        "enabled_effective": enabled_effective,
        "model": getattr(retriever, "_colbert_model", None),
        "status": status,
        "queries_with_diagnostics": len(rows),
        "attempted_queries": len(attempted),
        "successful_queries": len(successful),
        "failed_queries": len(failed),
        "diagnostics_failed_queries": len(diagnostic_failures),
        "zero_result_queries": sum(
            1 for row in successful if int(row.get("result_count") or 0) == 0
        ),
        "total_result_count": sum(result_counts),
        "avg_result_count": (sum(result_counts) / len(result_counts)) if result_counts else 0.0,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "error_types": _count_values(failed, "error_type"),
        "partial": status == "partial",
    }


def _beir_run_status(
    *,
    evaluated_queries: int,
    retrieval_failed_queries: int,
    colbert_failed_queries: int,
) -> str:
    if evaluated_queries <= 0:
        return "invalid"
    if retrieval_failed_queries == 0 and colbert_failed_queries == 0:
        return "success"
    if retrieval_failed_queries >= evaluated_queries:
        return "invalid"
    if colbert_failed_queries >= evaluated_queries:
        return "invalid"
    return "partial"


def _metric_zeroes(k_values: tuple[int, ...]) -> dict[str, float]:
    return {
        f"{name}@{k}": 0.0
        for k in k_values
        for name in ("recall", "precision", "hitrate", "mrr", "ndcg")
    }


def _configured_colbert_model(config: dict[str, Any]) -> str | None:
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    colbert = retrieval.get("colbert") if isinstance(retrieval.get("colbert"), dict) else {}
    model = colbert.get("model")
    return str(model) if model not in (None, "") else None


def _exception_chain_contains_type(
    exc: Exception,
    error_type: type[BaseException],
) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, error_type):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_colbert_startup_failure(exc: Exception, config: dict[str, Any]) -> bool:
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    colbert_cfg = retrieval.get("colbert") if isinstance(retrieval.get("colbert"), dict) else {}
    if not bool(colbert_cfg.get("enabled", False)):
        return False
    return _exception_chain_contains_type(exc, ColBERTInitializationError)


def _write_beir_startup_failure_artifacts(
    *,
    exc: Exception,
    dataset: str,
    split: str,
    dataset_dir: Path,
    documents_dir: Path,
    config: dict[str, Any],
    config_path: str | None,
    datasource_id: int | None,
    embedding_model: str | None,
    effective_embedding_model: str,
    retriever_name: str,
    rrf_k: int | None,
    elasticsearch_fuzziness: str | None,
    k_values: tuple[int, ...],
    max_queries: int | None,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    run_label: str,
    details_path: Path,
    summary_path: Path,
    use_filter_dir: bool,
    chunk_overfetch_factor: int,
    reranking_enabled: bool,
    rerank_model: str | None,
    rerank_candidate_limit: int | None,
    doc_aggregation: str,
    doc_aggregation_rrf_k: int,
    doc_aggregation_bonus: float,
    doc_aggregation_max_bonus_chunks: int,
    started_at: str,
    started_monotonic: float,
) -> BeirEvalStats:
    finished_at = _utc_timestamp()
    max_k = max(k_values)
    retrieve_k = max_k * max(chunk_overfetch_factor, 1)
    error = {
        "phase": "retriever_init",
        "type": exc.__class__.__name__,
        "message": str(exc),
    }
    planned_queries = len(query_ids)
    retrieval_failures = {
        "startup_failed": True,
        "failed_queries": planned_queries,
        "error_types": {exc.__class__.__name__: 1},
        "error": error,
    }
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    colbert_cfg = retrieval.get("colbert") if isinstance(retrieval.get("colbert"), dict) else {}
    colbert_requested = bool(colbert_cfg.get("enabled", False))
    colbert_startup_failure = _is_colbert_startup_failure(exc, config)
    colbert_status = "disabled"
    if colbert_requested:
        colbert_status = "invalid" if colbert_startup_failure else "unattempted"
    colbert_diagnostics = {
        "enabled_requested": colbert_requested,
        "enabled_effective": False,
        "model": _configured_colbert_model(config),
        "status": colbert_status,
        "initialization_failed": colbert_startup_failure,
        "blocked_by_retriever_startup_failure": colbert_requested and not colbert_startup_failure,
        "failure_phase": "retriever_init" if colbert_startup_failure else None,
        "attempted_queries": 0,
        "successful_queries": 0,
        "failed_queries": planned_queries if colbert_startup_failure else 0,
        "zero_result_queries": 0,
        "total_result_count": 0,
        "avg_result_count": 0.0,
        "avg_latency_ms": 0.0,
        "error_types": {exc.__class__.__name__: 1} if colbert_startup_failure else {},
        "error": str(exc) if colbert_startup_failure else None,
        "partial": False,
    }
    detail_row = {
        "query_index": None,
        "query_id": None,
        "query": None,
        "relevant_doc_ids": [],
        "metrics": {},
        "latency_ms": 0,
        "retrieval_latency_ms": 0,
        "rerank_latency_ms": 0,
        "retrieval_error": error,
        "colbert": colbert_diagnostics,
        "candidate_stats": {
            "retrieved_chunks": 0,
            "mapped_chunks": 0,
            "unmapped_chunks": 0,
            "duplicate_doc_chunks": 0,
            "unique_docs_seen": 0,
            "top_docs_returned": 0,
        },
        "ranking": {
            "first_relevant_rank": None,
            "relevant_doc_ranks": {},
            "relevant_found_at_max_k": False,
            "missing_relevant_doc_ids": [],
        },
        "top_docs": [],
    }
    summary = {
        "schema": "kiara-beir-retrieval-summary",
        "created_at": finished_at,
        "dataset": dataset,
        "split": split,
        "run": {
            "label": run_label,
            "retriever": retriever_name,
            "datasource_id": datasource_id,
            "embedding_model": embedding_model,
            "effective_embedding_model": effective_embedding_model,
            "k_values": list(k_values),
            "retrieve_k": retrieve_k,
            "filter_dir": str(documents_dir) if use_filter_dir else None,
            "rrf_k_override": rrf_k,
            "elasticsearch_fuzziness_override": elasticsearch_fuzziness,
            "reranking_enabled": reranking_enabled,
            "rerank_model": rerank_model,
            "rerank_candidate_limit": rerank_candidate_limit,
            "doc_aggregation": doc_aggregation,
            "doc_aggregation_rrf_k": doc_aggregation_rrf_k,
            "doc_aggregation_bonus": doc_aggregation_bonus,
            "doc_aggregation_max_bonus_chunks": doc_aggregation_max_bonus_chunks,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": int((time.monotonic() - started_monotonic) * 1000),
            "status": "invalid",
            "valid": False,
        },
        "settings": {
            "dataset_dir": str(dataset_dir),
            "documents_dir": str(documents_dir),
            "config_path": config_path,
            "datasource": _datasource_snapshot(datasource_id),
            "datasource_id": datasource_id,
            "environment": _runtime_snapshot(),
            "embedding_config": _embedding_settings_snapshot(config),
            "embedding_model_requested": embedding_model,
            "embedding_model_effective": effective_embedding_model,
            "qdrant_collection": _collection_for_model(effective_embedding_model),
            "retriever": retriever_name,
            "retriever_runtime": {
                "class": None,
                "init_failed": True,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
            "retrieval_overrides": {
                "rrf_k": rrf_k,
                "elasticsearch_fuzziness": elasticsearch_fuzziness,
            },
            "k_values": list(k_values),
            "max_k": max_k,
            "retrieve_k": retrieve_k,
            "max_queries": max_queries,
            "queries_planned": planned_queries,
            "queries_evaluated": 0,
            "use_filter_dir": use_filter_dir,
            "filter_dir": str(documents_dir) if use_filter_dir else None,
            "chunk_overfetch_factor": chunk_overfetch_factor,
            "ranking": {
                "doc_aggregation": doc_aggregation,
                "doc_aggregation_rrf_k": doc_aggregation_rrf_k,
                "doc_aggregation_bonus": doc_aggregation_bonus,
                "doc_aggregation_max_bonus_chunks": doc_aggregation_max_bonus_chunks,
            },
            "retrieval_config": _retrieval_settings_snapshot(config),
            "elasticsearch": _elasticsearch_snapshot(config),
        },
        "overall": {
            "queries_total": len(qrels),
            "queries_evaluated": 0,
            "avg_latency_ms": 0.0,
            "avg_retrieval_latency_ms": 0.0,
            "avg_rerank_latency_ms": 0.0,
            **_metric_zeroes(k_values),
        },
        "ranking_diagnostics": {
            "first_relevant_rank_min": None,
            "first_relevant_rank_median": None,
            "first_relevant_rank_p90": None,
            "queries_without_relevant_at_max_k": planned_queries,
            "startup_failed": True,
        },
        "retrieval_failures": retrieval_failures,
        "colbert_diagnostics": colbert_diagnostics,
        "artifacts": {
            "dataset_dir": str(dataset_dir),
            "summary": str(summary_path),
            "details": str(details_path),
        },
    }
    write_jsonl(details_path, [detail_row])
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BeirEvalStats(
        dataset=dataset,
        split=split,
        run_label=run_label,
        dataset_dir=str(dataset_dir),
        summary_path=str(summary_path),
        details_path=str(details_path),
        queries_evaluated=0,
        datasource_id=datasource_id,
        retriever=retriever_name,
        max_k=max_k,
    )


def _reranking_config(
    config: dict[str, Any],
    *,
    enabled: bool,
    model_name: str | None,
    batch_size: int | None,
) -> dict[str, Any]:
    rerank_config = deepcopy(config) if config else {}
    retrieval = rerank_config.setdefault("retrieval", {})
    retrieval["reranking"] = enabled
    if model_name:
        retrieval["rerank_model"] = model_name
    if batch_size is not None:
        retrieval["rerank_batch_size"] = batch_size
    return rerank_config


def _build_reranker(
    config: dict[str, Any],
    *,
    enabled: bool,
    model_name: str | None,
    batch_size: int | None,
):
    if not enabled:
        return None, _reranking_config(
            config,
            enabled=False,
            model_name=model_name,
            batch_size=batch_size,
        )
    from kiara.retrieval.reranker import create_reranker

    rerank_config = _reranking_config(
        config,
        enabled=True,
        model_name=model_name,
        batch_size=batch_size,
    )
    return create_reranker(rerank_config), rerank_config


def _reranker_runtime_snapshot(reranker: Any) -> dict[str, Any]:
    if reranker is None:
        return {"enabled": False}
    attrs = {
        "class": f"{type(reranker).__module__}.{type(reranker).__name__}",
        "enabled": getattr(reranker, "enabled", None),
        "model": getattr(reranker, "_model_name", None),
        "top_k": getattr(reranker, "_top_k", None),
        "batch_size": getattr(reranker, "_batch_size", None),
    }
    return {key: value for key, value in attrs.items() if value is not None}


def _runtime_snapshot() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def _apply_reranking(
    query_text: str,
    chunk_results: list[dict[str, Any]],
    reranker: Any,
    candidate_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reranker is None:
        return chunk_results, {
            "enabled": False,
            "applied": False,
            "candidate_limit": candidate_limit,
            "input_chunks": len(chunk_results),
            "reranked_chunks": 0,
            "rerank_score_count": 0,
            "latency_ms": 0,
        }

    candidates = (
        chunk_results[:candidate_limit]
        if candidate_limit is not None and candidate_limit > 0
        else chunk_results
    )
    tail = chunk_results[len(candidates) :]
    started = time.monotonic()
    reranked = reranker.rerank(query_text, candidates, top_k=len(candidates))
    latency_ms = int((time.monotonic() - started) * 1000)
    rerank_score_count = sum(1 for item in reranked if item.get("rerank_score") is not None)
    return reranked + tail, {
        "enabled": True,
        "effective_enabled": bool(getattr(reranker, "enabled", True)),
        "applied": bool(candidates),
        "candidate_limit": candidate_limit,
        "input_chunks": len(chunk_results),
        "reranked_chunks": len(reranked),
        "tail_chunks": len(tail),
        "rerank_score_count": rerank_score_count,
        "latency_ms": latency_ms,
        "runtime": _reranker_runtime_snapshot(reranker),
    }


def _elasticsearch_snapshot(config: dict[str, Any]) -> dict[str, Any] | None:
    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    if retrieval.get("hybrid_search") != "elasticsearch":
        return None
    try:
        from kiara.retrieval.elasticsearch_service import get_status

        status = get_status(config)
        return {
            "available": status.get("available"),
            "index_exists": status.get("index_exists"),
            "doc_count": status.get("doc_count"),
            "health": status.get("health"),
            "index_prefix": status.get("index_prefix"),
            "datasources": status.get("datasources"),
            "unit": status.get("unit"),
            "embedding_model": status.get("embedding_model"),
            "expected_physical_documents": status.get("expected_physical_documents"),
            "counts_match": status.get("counts_match"),
            "error": status.get("error"),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _collection_for_model(model_name: str) -> str | None:
    if not model_name:
        return None
    try:
        from kiara.retrieval.embedding import collection_name_for_model

        return collection_name_for_model(model_name)
    except Exception:
        return None


def _score_for_result(result: dict[str, Any]) -> float | None:
    value = result.get("rerank_score")
    if value is None:
        value = result.get("rrf_score", result.get("score"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chunk_id_for_result(result: dict[str, Any]) -> str:
    for key in ("chunk_id", "point_id", "_doc_id"):
        value = result.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def _compact_chunk_result(result: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    chunk_id = _chunk_id_for_result(result)
    text = str(result.get("text") or "")
    item = {
        "rank": rank,
        "final_context_rank": result.get("final_context_rank", rank),
        "chunk_id": chunk_id,
        "point_id": str(result.get("point_id") or chunk_id),
        "source": metadata.get("source", ""),
        "metadata": dict(metadata),
        "chunk_index": metadata.get("chunk_index"),
        "score": result.get("rrf_score", result.get("rerank_score", result.get("score"))),
        "semantic_rank": result.get("semantic_rank"),
        "semantic_score": result.get("semantic_score"),
        "keyword_rank": result.get("keyword_rank"),
        "keyword_score": result.get("keyword_score"),
        "colbert_rank": result.get("colbert_rank"),
        "colbert_score": result.get("colbert_score"),
        "graph_rank": result.get("graph_rank"),
        "graph_score": result.get("graph_score"),
        "rrf_rank": result.get("rrf_rank"),
        "rrf_score": result.get("rrf_score"),
        "rrf_contributions": dict(result.get("rrf_contributions") or {}),
        "retriever_paths": list(result.get("retriever_paths") or []),
        "rerank_score": result.get("rerank_score"),
        "excerpt": text[:_DIAGNOSTIC_EXCERPT_CHARS],
        "text_chars": len(text),
    }
    return item


def _compact_chunk_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_compact_chunk_result(result, rank) for rank, result in enumerate(results, 1)]


def _retriever_path_diagnostics(retriever: Any) -> dict[str, Any] | None:
    getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
    if not callable(getter):
        return None
    diagnostics = getter()
    return dict(diagnostics) if isinstance(diagnostics, dict) else None


def _with_final_context_ranks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, result in enumerate(results, 1):
        item = result.copy()
        item["final_context_rank"] = rank
        ranked.append(item)
    return ranked


def _rank_docs_by_first_chunk(
    chunk_results: list[dict[str, Any]],
    mapper: SourceDocMapper,
    max_docs: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    unmapped = 0
    mapped_chunks = 0
    duplicate_doc_chunks = 0
    for chunk_rank, result in enumerate(chunk_results, 1):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        doc_id = mapper.doc_id_for_source(metadata.get("source"))
        if doc_id is None:
            unmapped += 1
            continue
        mapped_chunks += 1
        if doc_id in seen:
            duplicate_doc_chunks += 1
            continue
        seen.add(doc_id)
        if len(docs) < max_docs:
            docs.append(
                {
                    "rank": len(docs) + 1,
                    "doc_id": doc_id,
                    "chunk_rank": chunk_rank,
                    "source": metadata.get("source", ""),
                    "score": result.get("rrf_score", result.get("score")),
                    "rerank_score": result.get("rerank_score"),
                }
            )
    return docs, {
        "retrieved_chunks": len(chunk_results),
        "mapped_chunks": mapped_chunks,
        "unmapped_chunks": unmapped,
        "duplicate_doc_chunks": duplicate_doc_chunks,
        "unique_docs_seen": len(seen),
        "top_docs_returned": len(docs),
    }


def _rank_docs_by_rank_rrf(
    chunk_results: list[dict[str, Any]],
    mapper: SourceDocMapper,
    max_docs: int,
    doc_aggregation_rrf_k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, dict[str, Any]] = {}
    unmapped = 0
    mapped_chunks = 0
    duplicate_doc_chunks = 0

    for chunk_rank, result in enumerate(chunk_results, 1):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        doc_id = mapper.doc_id_for_source(metadata.get("source"))
        if doc_id is None:
            unmapped += 1
            continue
        mapped_chunks += 1
        score = _score_for_result(result)
        contribution = 1.0 / (doc_aggregation_rrf_k + chunk_rank)
        item = grouped.get(doc_id)
        if item is None:
            grouped[doc_id] = {
                "doc_id": doc_id,
                "best_chunk_rank": chunk_rank,
                "source": metadata.get("source", ""),
                "score": score,
                "rerank_score": result.get("rerank_score"),
                "aggregation_score": contribution,
                "chunk_hits": 1,
            }
        else:
            duplicate_doc_chunks += 1
            item["aggregation_score"] += contribution
            item["chunk_hits"] += 1
            if chunk_rank < item["best_chunk_rank"]:
                item["best_chunk_rank"] = chunk_rank
                item["source"] = metadata.get("source", "")
                item["score"] = score
                item["rerank_score"] = result.get("rerank_score")

    ordered = sorted(
        grouped.values(),
        key=lambda item: (-item["aggregation_score"], item["best_chunk_rank"], item["doc_id"]),
    )
    docs = [
        {
            "rank": rank,
            "doc_id": item["doc_id"],
            "chunk_rank": item["best_chunk_rank"],
            "source": item["source"],
            "score": item["score"],
            "rerank_score": item["rerank_score"],
            "aggregation_score": item["aggregation_score"],
            "chunk_hits": item["chunk_hits"],
        }
        for rank, item in enumerate(ordered[:max_docs], 1)
    ]
    return docs, {
        "retrieved_chunks": len(chunk_results),
        "mapped_chunks": mapped_chunks,
        "unmapped_chunks": unmapped,
        "duplicate_doc_chunks": duplicate_doc_chunks,
        "unique_docs_seen": len(grouped),
        "top_docs_returned": len(docs),
    }


def _rank_docs_by_best_plus_bonus(
    chunk_results: list[dict[str, Any]],
    mapper: SourceDocMapper,
    max_docs: int,
    doc_aggregation_rrf_k: int,
    doc_aggregation_bonus: float,
    doc_aggregation_max_bonus_chunks: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, dict[str, Any]] = {}
    unmapped = 0
    mapped_chunks = 0
    duplicate_doc_chunks = 0

    for chunk_rank, result in enumerate(chunk_results, 1):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        doc_id = mapper.doc_id_for_source(metadata.get("source"))
        if doc_id is None:
            unmapped += 1
            continue
        mapped_chunks += 1
        score = _score_for_result(result)
        item = grouped.get(doc_id)
        if item is None:
            grouped[doc_id] = {
                "doc_id": doc_id,
                "best_chunk_rank": chunk_rank,
                "source": metadata.get("source", ""),
                "score": score,
                "rerank_score": result.get("rerank_score"),
                "chunk_hits": 1,
            }
        else:
            duplicate_doc_chunks += 1
            item["chunk_hits"] += 1
            if chunk_rank < item["best_chunk_rank"]:
                item["best_chunk_rank"] = chunk_rank
                item["source"] = metadata.get("source", "")
                item["score"] = score
                item["rerank_score"] = result.get("rerank_score")

    for item in grouped.values():
        base_score = 1.0 / (doc_aggregation_rrf_k + item["best_chunk_rank"])
        bonus_chunks = min(
            max(0, item["chunk_hits"] - 1),
            doc_aggregation_max_bonus_chunks,
        )
        bonus_multiplier = 1.0 + (bonus_chunks * doc_aggregation_bonus)
        item["aggregation_score"] = base_score * bonus_multiplier
        item["bonus_chunks"] = bonus_chunks
        item["bonus_multiplier"] = bonus_multiplier

    ordered = sorted(
        grouped.values(),
        key=lambda item: (-item["aggregation_score"], item["best_chunk_rank"], item["doc_id"]),
    )
    docs = [
        {
            "rank": rank,
            "doc_id": item["doc_id"],
            "chunk_rank": item["best_chunk_rank"],
            "source": item["source"],
            "score": item["score"],
            "rerank_score": item["rerank_score"],
            "aggregation_score": item["aggregation_score"],
            "chunk_hits": item["chunk_hits"],
            "bonus_chunks": item["bonus_chunks"],
            "bonus_multiplier": item["bonus_multiplier"],
        }
        for rank, item in enumerate(ordered[:max_docs], 1)
    ]
    return docs, {
        "retrieved_chunks": len(chunk_results),
        "mapped_chunks": mapped_chunks,
        "unmapped_chunks": unmapped,
        "duplicate_doc_chunks": duplicate_doc_chunks,
        "unique_docs_seen": len(grouped),
        "top_docs_returned": len(docs),
    }


def _rank_chunk_results_as_docs(
    chunk_results: list[dict[str, Any]],
    mapper: SourceDocMapper,
    max_docs: int,
    *,
    doc_aggregation: str,
    doc_aggregation_rrf_k: int,
    doc_aggregation_bonus: float,
    doc_aggregation_max_bonus_chunks: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if doc_aggregation == "first":
        return _rank_docs_by_first_chunk(chunk_results, mapper, max_docs)
    if doc_aggregation == "rank_rrf":
        return _rank_docs_by_rank_rrf(
            chunk_results,
            mapper,
            max_docs,
            doc_aggregation_rrf_k,
        )
    if doc_aggregation == "best_plus_bonus":
        return _rank_docs_by_best_plus_bonus(
            chunk_results,
            mapper,
            max_docs,
            doc_aggregation_rrf_k,
            doc_aggregation_bonus,
            doc_aggregation_max_bonus_chunks,
        )
    raise ValueError("doc_aggregation must be 'first', 'rank_rrf' or 'best_plus_bonus'")


def _relevant_rank_info(
    ranked_doc_ids: list[str],
    relevant: dict[str, int],
) -> dict[str, Any]:
    ranks = {doc_id: rank for rank, doc_id in enumerate(ranked_doc_ids, 1) if doc_id in relevant}
    first_rank = min(ranks.values()) if ranks else None
    return {
        "first_relevant_rank": first_rank,
        "relevant_doc_ranks": ranks,
        "relevant_found_at_max_k": first_rank is not None,
        "missing_relevant_doc_ids": sorted(set(relevant) - set(ranks)),
    }


def evaluate_beir_retrieval(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    config_path: str | None = None,
    datasource_id: int | None = None,
    embedding_model: str | None = None,
    retriever_name: str = "hybrid",
    rrf_k: int | None = None,
    elasticsearch_fuzziness: str | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    max_queries: int | None = None,
    run_label: str | None = None,
    output_dir: Path | None = None,
    use_filter_dir: bool = True,
    chunk_overfetch_factor: int = 3,
    reranking_enabled: bool = False,
    rerank_model: str | None = None,
    rerank_batch_size: int | None = None,
    rerank_candidate_limit: int | None = None,
    doc_aggregation: str = "first",
    doc_aggregation_rrf_k: int = DEFAULT_DOC_AGGREGATION_RRF_K,
    doc_aggregation_bonus: float = DEFAULT_DOC_AGGREGATION_BONUS,
    doc_aggregation_max_bonus_chunks: int = DEFAULT_DOC_AGGREGATION_MAX_BONUS_CHUNKS,
    retriever=None,
    reranker=None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BeirEvalStats:
    started_at = _utc_timestamp()
    started_monotonic = time.monotonic()
    if doc_aggregation not in {"first", "rank_rrf", "best_plus_bonus"}:
        raise ValueError("doc_aggregation must be 'first', 'rank_rrf' or 'best_plus_bonus'")
    doc_aggregation_rrf_k = max(1, int(doc_aggregation_rrf_k))
    doc_aggregation_bonus = max(0.0, float(doc_aggregation_bonus))
    doc_aggregation_max_bonus_chunks = max(0, int(doc_aggregation_max_bonus_chunks))
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = str(manifest.get("dataset") or dataset_dir.name)
    split = str(manifest.get("split") or "")
    documents_dir = Path(manifest.get("documents_dir") or dataset_dir / "documents").resolve()

    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir)
    mapper = SourceDocMapper(dataset_dir)

    if retriever is None:
        config = _load_config_with_retrieval_overrides(
            config_path=config_path,
            rrf_k=rrf_k,
            elasticsearch_fuzziness=elasticsearch_fuzziness,
        )
    else:
        config = {}

    effective_embedding_model = _effective_embedding_model(config, embedding_model)

    if run_label is None:
        rerank_label = "_rerank" if reranking_enabled else ""
        rrf_label = f"_rrf{rrf_k}" if rrf_k is not None else ""
        fuzz_label = f"_fuzz{_slug(elasticsearch_fuzziness)}" if elasticsearch_fuzziness else ""
        model_label = embedding_model or str(
            config.get("embedding", {}).get("model")
            or config.get("ollama", {}).get("embedding_model")
            or "configured"
        )
        run_label = _slug(
            f"{dataset}_{retriever_name}{rerank_label}{rrf_label}{fuzz_label}_{model_label}_k{max(k_values)}_{_timestamp_for_filename()}"
        )
    else:
        run_label = _slug(run_label)

    if output_dir is None:
        output_dir = dataset_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"{run_label}_details.jsonl"
    summary_path = output_dir / f"{run_label}_summary.json"

    query_ids = [query_id for query_id in sorted(qrels) if query_id in queries]
    if max_queries is not None:
        query_ids = query_ids[:max_queries]

    max_k = max(k_values)
    retrieve_k = max_k * max(chunk_overfetch_factor, 1)

    if retriever is None:
        try:
            retriever = _instantiate_retriever(
                retriever_name,
                config,
                embedding_model=embedding_model,
            )
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            return _write_beir_startup_failure_artifacts(
                exc=exc,
                dataset=dataset,
                split=split,
                dataset_dir=dataset_dir,
                documents_dir=documents_dir,
                config=config,
                config_path=config_path,
                datasource_id=datasource_id,
                embedding_model=embedding_model,
                effective_embedding_model=effective_embedding_model,
                retriever_name=retriever_name,
                rrf_k=rrf_k,
                elasticsearch_fuzziness=elasticsearch_fuzziness,
                k_values=k_values,
                max_queries=max_queries,
                query_ids=query_ids,
                qrels=qrels,
                run_label=run_label,
                details_path=details_path,
                summary_path=summary_path,
                use_filter_dir=use_filter_dir,
                chunk_overfetch_factor=chunk_overfetch_factor,
                reranking_enabled=reranking_enabled,
                rerank_model=rerank_model,
                rerank_candidate_limit=rerank_candidate_limit,
                doc_aggregation=doc_aggregation,
                doc_aggregation_rrf_k=doc_aggregation_rrf_k,
                doc_aggregation_bonus=doc_aggregation_bonus,
                doc_aggregation_max_bonus_chunks=doc_aggregation_max_bonus_chunks,
                started_at=started_at,
                started_monotonic=started_monotonic,
            )

    if not reranking_enabled:
        reranker = None
        rerank_config = _reranking_config(
            config,
            enabled=False,
            model_name=rerank_model,
            batch_size=rerank_batch_size,
        )
    elif reranker is None:
        reranker, rerank_config = _build_reranker(
            config,
            enabled=reranking_enabled,
            model_name=rerank_model,
            batch_size=rerank_batch_size,
        )
    else:
        rerank_config = _reranking_config(
            config,
            enabled=reranking_enabled,
            model_name=rerank_model,
            batch_size=rerank_batch_size,
        )

    details: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    latencies: list[int] = []
    candidate_stat_rows: list[dict[str, int]] = []
    first_relevant_ranks: list[int] = []
    retrieval_latencies: list[int] = []
    rerank_latencies: list[int] = []
    rerank_score_counts: list[int] = []
    colbert_diagnostic_rows: list[dict[str, Any]] = []
    retrieval_error_rows: list[dict[str, Any]] = []

    for index, query_id in enumerate(query_ids, 1):
        query_text = queries[query_id]
        retrieval_started = time.monotonic()
        retrieval_error: dict[str, Any] | None = None
        try:
            chunk_results = retriever.retrieve(
                query_text,
                k=retrieve_k,
                filter_dir=str(documents_dir) if use_filter_dir else None,
                filter_datasource_ids=[datasource_id] if datasource_id is not None else None,
            )
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            retrieval_error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            retrieval_error_rows.append(retrieval_error)
            chunk_results = []
        retrieval_latency_ms = int((time.monotonic() - retrieval_started) * 1000)
        colbert_diagnostics = _retriever_colbert_diagnostics(retriever)
        retrieval_path_diagnostics = _retriever_path_diagnostics(retriever)
        if colbert_diagnostics is not None:
            colbert_diagnostic_rows.append(colbert_diagnostics)
        chunk_results, rerank_stats = _apply_reranking(
            query_text,
            chunk_results,
            reranker,
            rerank_candidate_limit,
        )
        chunk_results = _with_final_context_ranks(chunk_results)
        rerank_latency_ms = int(rerank_stats.get("latency_ms") or 0)
        latency_ms = retrieval_latency_ms + rerank_latency_ms
        latencies.append(latency_ms)
        retrieval_latencies.append(retrieval_latency_ms)
        rerank_latencies.append(rerank_latency_ms)
        rerank_score_counts.append(int(rerank_stats.get("rerank_score_count") or 0))
        top_docs, candidate_stats = _rank_chunk_results_as_docs(
            chunk_results,
            mapper,
            max_k,
            doc_aggregation=doc_aggregation,
            doc_aggregation_rrf_k=doc_aggregation_rrf_k,
            doc_aggregation_bonus=doc_aggregation_bonus,
            doc_aggregation_max_bonus_chunks=doc_aggregation_max_bonus_chunks,
        )
        candidate_stat_rows.append(candidate_stats)
        ranked_doc_ids = [row["doc_id"] for row in top_docs]
        rank_info = _relevant_rank_info(ranked_doc_ids, qrels[query_id])
        if rank_info["first_relevant_rank"] is not None:
            first_relevant_ranks.append(int(rank_info["first_relevant_rank"]))
        query_metrics = _metrics_for_query(ranked_doc_ids, qrels[query_id], k_values)
        metric_rows.append(query_metrics)
        detail_row = {
            "query_index": index,
            "query_id": query_id,
            "query": query_text,
            "relevant_doc_ids": sorted(qrels[query_id]),
            "metrics": query_metrics,
            "latency_ms": latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
            "reranking": rerank_stats,
            "candidate_stats": candidate_stats,
            "ranking": rank_info,
            "top_docs": top_docs,
            "retrieved_chunks": _compact_chunk_results(chunk_results),
            "retrieval_diagnostics": retrieval_path_diagnostics,
        }
        if retrieval_error is not None:
            detail_row["retrieval_error"] = retrieval_error
        if colbert_diagnostics is not None:
            detail_row["colbert"] = colbert_diagnostics
        details.append(detail_row)
        if progress_callback is not None:
            progress_callback(
                {
                    "current": index,
                    "total": len(query_ids),
                    "query_id": query_id,
                    "latency_ms": latency_ms,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "rerank_latency_ms": rerank_latency_ms,
                    "retrieved_chunks": candidate_stats["retrieved_chunks"],
                    "unmapped_chunks": candidate_stats["unmapped_chunks"],
                    "first_relevant_rank": rank_info["first_relevant_rank"],
                    "rerank_score_count": rerank_stats.get("rerank_score_count"),
                    "retrieval_failed": retrieval_error is not None,
                    "retrieval_error_type": retrieval_error.get("type")
                    if retrieval_error is not None
                    else None,
                    "colbert_attempted": colbert_diagnostics.get("attempted")
                    if colbert_diagnostics is not None
                    else None,
                    "colbert_failed": colbert_diagnostics.get("failed")
                    if colbert_diagnostics is not None
                    else None,
                    "colbert_result_count": colbert_diagnostics.get("result_count")
                    if colbert_diagnostics is not None
                    else None,
                }
            )

    summary_metrics = {
        metric_name: _average_metric(metric_rows, metric_name)
        for metric_name in [
            f"{name}@{k}"
            for k in k_values
            for name in ("recall", "precision", "hitrate", "mrr", "ndcg")
        ]
    }
    finished_at = _utc_timestamp()
    ranking_diagnostics = {
        "first_relevant_rank_min": min(first_relevant_ranks) if first_relevant_ranks else None,
        "first_relevant_rank_median": _percentile(first_relevant_ranks, 50),
        "first_relevant_rank_p90": _percentile(first_relevant_ranks, 90),
        "queries_without_relevant_at_max_k": len(query_ids) - len(first_relevant_ranks),
        "avg_retrieved_chunks": _average_metric(candidate_stat_rows, "retrieved_chunks"),
        "avg_mapped_chunks": _average_metric(candidate_stat_rows, "mapped_chunks"),
        "avg_unmapped_chunks": _average_metric(candidate_stat_rows, "unmapped_chunks"),
        "avg_duplicate_doc_chunks": _average_metric(candidate_stat_rows, "duplicate_doc_chunks"),
        "avg_unique_docs_seen": _average_metric(candidate_stat_rows, "unique_docs_seen"),
        "avg_top_docs_returned": _average_metric(candidate_stat_rows, "top_docs_returned"),
        "avg_retrieval_latency_ms": (sum(retrieval_latencies) / len(retrieval_latencies))
        if retrieval_latencies
        else 0.0,
        "avg_rerank_latency_ms": (sum(rerank_latencies) / len(rerank_latencies))
        if rerank_latencies
        else 0.0,
        "avg_rerank_score_count": (sum(rerank_score_counts) / len(rerank_score_counts))
        if rerank_score_counts
        else 0.0,
    }
    retrieval_config = _retrieval_settings_snapshot(config)
    rerank_retrieval_config = _retrieval_settings_snapshot(rerank_config)
    reranker_runtime = _reranker_runtime_snapshot(reranker)
    colbert_diagnostics = _colbert_diagnostics_summary(retriever, colbert_diagnostic_rows)
    retrieval_failures = {
        "failed_queries": len(retrieval_error_rows),
        "error_types": _count_values(retrieval_error_rows, "type"),
    }
    run_status = _beir_run_status(
        evaluated_queries=len(query_ids),
        retrieval_failed_queries=retrieval_failures["failed_queries"],
        colbert_failed_queries=colbert_diagnostics["failed_queries"],
    )
    run_is_valid = run_status == "success"
    summary = {
        "schema": "kiara-beir-retrieval-summary",
        "created_at": finished_at,
        "dataset": dataset,
        "split": split,
        "run": {
            "label": run_label,
            "retriever": retriever_name,
            "datasource_id": datasource_id,
            "embedding_model": embedding_model,
            "effective_embedding_model": effective_embedding_model,
            "k_values": list(k_values),
            "retrieve_k": retrieve_k,
            "filter_dir": str(documents_dir) if use_filter_dir else None,
            "rrf_k_override": rrf_k,
            "elasticsearch_fuzziness_override": elasticsearch_fuzziness,
            "reranking_enabled": reranking_enabled,
            "rerank_model": rerank_retrieval_config.get("rerank_model"),
            "rerank_candidate_limit": rerank_candidate_limit,
            "doc_aggregation": doc_aggregation,
            "doc_aggregation_rrf_k": doc_aggregation_rrf_k,
            "doc_aggregation_bonus": doc_aggregation_bonus,
            "doc_aggregation_max_bonus_chunks": doc_aggregation_max_bonus_chunks,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": int((time.monotonic() - started_monotonic) * 1000),
            "status": run_status,
            "valid": run_is_valid,
        },
        "settings": {
            "dataset_dir": str(dataset_dir),
            "documents_dir": str(documents_dir),
            "config_path": config_path,
            "datasource": _datasource_snapshot(datasource_id),
            "datasource_id": datasource_id,
            "environment": _runtime_snapshot(),
            "embedding_config": _embedding_settings_snapshot(config),
            "embedding_model_requested": embedding_model,
            "embedding_model_effective": effective_embedding_model,
            "qdrant_collection": _collection_for_model(effective_embedding_model),
            "retriever": retriever_name,
            "retriever_runtime": _retriever_runtime_snapshot(retriever),
            "retrieval_overrides": {
                "rrf_k": rrf_k,
                "elasticsearch_fuzziness": elasticsearch_fuzziness,
            },
            "doc_aggregation": {
                "strategy": doc_aggregation,
                "rrf_k": doc_aggregation_rrf_k,
                "bonus": doc_aggregation_bonus,
                "max_bonus_chunks": doc_aggregation_max_bonus_chunks,
            },
            "reranking": {
                "enabled_requested": reranking_enabled,
                "enabled_effective": bool(reranker_runtime.get("enabled"))
                if reranker is not None
                else False,
                "model_requested": rerank_model,
                "model_effective": rerank_retrieval_config.get("rerank_model"),
                "batch_size": rerank_retrieval_config.get("rerank_batch_size"),
                "candidate_limit": rerank_candidate_limit,
                "runtime": reranker_runtime,
                "retrieval_config": rerank_retrieval_config,
            },
            "k_values": list(k_values),
            "max_k": max_k,
            "retrieve_k": retrieve_k,
            "max_queries": max_queries,
            "queries_evaluated": len(query_ids),
            "use_filter_dir": use_filter_dir,
            "filter_dir": str(documents_dir) if use_filter_dir else None,
            "chunk_overfetch_factor": chunk_overfetch_factor,
            "ranking": {
                "dedupe_strategy": "first_chunk_rank_per_document"
                if doc_aggregation == "first"
                else "document_rank_rrf"
                if doc_aggregation == "rank_rrf"
                else "best_chunk_plus_duplicate_bonus",
                "doc_aggregation": doc_aggregation,
                "doc_aggregation_rrf_k": doc_aggregation_rrf_k,
                "doc_aggregation_bonus": doc_aggregation_bonus,
                "doc_aggregation_max_bonus_chunks": doc_aggregation_max_bonus_chunks,
                "candidate_source": "reranked_chunks" if reranking_enabled else "retriever_chunks",
                "max_k": max_k,
                "retrieve_k": retrieve_k,
                "chunk_overfetch_factor": chunk_overfetch_factor,
                "reranking_enabled": reranking_enabled,
                "rerank_candidate_limit": rerank_candidate_limit,
                "retrieval_config": retrieval_config,
            },
            "diagnostic_fields": {
                "schema": "kiara-retrieval-path-diagnostics",
                "paths": list(_RETRIEVAL_DIAGNOSTIC_PATHS),
                "chunk_fields": [
                    "chunk_id",
                    "point_id",
                    "semantic_rank",
                    "semantic_score",
                    "keyword_rank",
                    "keyword_score",
                    "colbert_rank",
                    "colbert_score",
                    "graph_rank",
                    "graph_score",
                    "rrf_rank",
                    "rrf_score",
                    "rrf_contributions",
                    "final_context_rank",
                ],
            },
            "retrieval_config": retrieval_config,
            "elasticsearch": _elasticsearch_snapshot(config),
        },
        "overall": {
            "queries_total": len(qrels),
            "queries_evaluated": len(query_ids),
            "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "avg_retrieval_latency_ms": (sum(retrieval_latencies) / len(retrieval_latencies))
            if retrieval_latencies
            else 0.0,
            "avg_rerank_latency_ms": (sum(rerank_latencies) / len(rerank_latencies))
            if rerank_latencies
            else 0.0,
            **summary_metrics,
        },
        "ranking_diagnostics": ranking_diagnostics,
        "retrieval_failures": retrieval_failures,
        "colbert_diagnostics": colbert_diagnostics,
        "artifacts": {
            "dataset_dir": str(dataset_dir),
            "summary": str(summary_path),
            "details": str(details_path),
        },
    }

    write_jsonl(details_path, details)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return BeirEvalStats(
        dataset=dataset,
        split=split,
        run_label=run_label,
        dataset_dir=str(dataset_dir),
        summary_path=str(summary_path),
        details_path=str(details_path),
        queries_evaluated=len(query_ids),
        datasource_id=datasource_id,
        retriever=retriever_name,
        max_k=max_k,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--config-path")
    parser.add_argument("--datasource-id", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--retriever", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--rrf-k", type=int)
    parser.add_argument("--elasticsearch-fuzziness")
    parser.add_argument("--k-values", default="1,3,5,10,100")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--run-label")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-filter-dir", action="store_true")
    parser.add_argument("--chunk-overfetch-factor", type=int, default=3)
    parser.add_argument("--reranking", action="store_true")
    parser.add_argument("--rerank-model")
    parser.add_argument("--rerank-batch-size", type=int)
    parser.add_argument("--rerank-candidate-limit", type=int)
    parser.add_argument(
        "--doc-aggregation",
        choices=["first", "rank_rrf", "best_plus_bonus"],
        default="first",
    )
    parser.add_argument("--doc-aggregation-rrf-k", type=int, default=DEFAULT_DOC_AGGREGATION_RRF_K)
    parser.add_argument(
        "--doc-aggregation-bonus", type=float, default=DEFAULT_DOC_AGGREGATION_BONUS
    )
    parser.add_argument(
        "--doc-aggregation-max-bonus-chunks",
        type=int,
        default=DEFAULT_DOC_AGGREGATION_MAX_BONUS_CHUNKS,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = evaluate_beir_retrieval(
        dataset_dir=args.dataset_dir,
        config_path=args.config_path,
        datasource_id=args.datasource_id,
        embedding_model=args.embedding_model,
        retriever_name=args.retriever,
        rrf_k=args.rrf_k,
        elasticsearch_fuzziness=args.elasticsearch_fuzziness,
        k_values=parse_k_values(args.k_values),
        max_queries=args.max_queries,
        run_label=args.run_label,
        output_dir=args.output_dir,
        use_filter_dir=not args.no_filter_dir,
        chunk_overfetch_factor=args.chunk_overfetch_factor,
        reranking_enabled=args.reranking,
        rerank_model=args.rerank_model,
        rerank_batch_size=args.rerank_batch_size,
        rerank_candidate_limit=args.rerank_candidate_limit,
        doc_aggregation=args.doc_aggregation,
        doc_aggregation_rrf_k=args.doc_aggregation_rrf_k,
        doc_aggregation_bonus=args.doc_aggregation_bonus,
        doc_aggregation_max_bonus_chunks=args.doc_aggregation_max_bonus_chunks,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
