"""Sprint 4 retrieval-policy ablation runner.

The runner keeps production config immutable. Each variant is materialized as a
temporary YAML config below the output directory, then evaluated against the
KIara Golden Set and CRAG Full136 with retrieval-only calls.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from kiara.benchmarks import crag, golden
from kiara.benchmarks.artifacts import (
    ensure_private_dir,
    write_private_json,
    write_private_jsonl,
    write_private_text,
)

DEFAULT_OUTPUT_ROOT = Path("/var/lib/kiara/benchmarks/retrieval_policy")
DEFAULT_CRAG_MANIFEST = Path("/var/lib/kiara/benchmarks/crag_full136/manifests/task1.jsonl")
DEFAULT_GOLDEN_SET = Path("/var/lib/kiara/benchmarks/kiara_golden/golden_set_v1.jsonl")
DEFAULT_BEIR_DATASET = Path("/var/lib/kiara/benchmarks/beir/scifact")
DEFAULT_K_VALUES = (1, 3, 5, 10, 20)
PRIORITY_GOLDEN_CASE_IDS = (
    "KGS-001",
    "KGS-002",
    "KGS-003",
    "KGS-005",
    "KGS-009",
    "KGS-011",
    "KGS-020",
    "KGS-028",
    "KGS-029",
    "KGS-030",
)


@dataclass(frozen=True)
class RetrievalPolicyVariant:
    label: str
    description: str
    overrides: dict[str, Any]
    retrieve_k: int = 20
    reranking_enabled: bool = False
    rerank_candidate_limit: int | None = None
    include_in_beir_smoke: bool = False


@dataclass(frozen=True)
class RetrievalPolicyRunStats:
    label: str
    output_dir: str
    summary_path: str
    report_path: str
    variants: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "run"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_private_json(path, payload, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_private_jsonl(path, rows, sort_keys=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_default_variants(*, include_bm25: bool = True) -> list[RetrievalPolicyVariant]:
    variants = [
        RetrievalPolicyVariant(
            label="dense_only_k20",
            description="Semantic vector retrieval only; no keyword, no ColBERT, no reranking.",
            overrides={
                "retrieval": {
                    "hybrid_search": "none",
                    "reranking": False,
                    "colbert": {"enabled": False},
                },
                "graph": {"enabled": False},
            },
            retrieve_k=20,
            include_in_beir_smoke=True,
        ),
        RetrievalPolicyVariant(
            label="es_rrf60_colbert_off",
            description="Semantic + Elasticsearch keyword via RRF k=60; ColBERT off.",
            overrides={
                "retrieval": {
                    "hybrid_search": "elasticsearch",
                    "rrf_k": 60,
                    "reranking": False,
                    "colbert": {"enabled": False},
                },
                "graph": {"enabled": False},
            },
            retrieve_k=20,
            include_in_beir_smoke=True,
        ),
        RetrievalPolicyVariant(
            label="live_es_rrf60_colbert_on",
            description="Current live retrieval policy: Elasticsearch + ColBERT + RRF k=60, reranking off.",
            overrides={
                "retrieval": {
                    "hybrid_search": "elasticsearch",
                    "rrf_k": 60,
                    "reranking": False,
                    "colbert": {"enabled": True},
                },
                "graph": {"enabled": False},
            },
            retrieve_k=20,
            include_in_beir_smoke=True,
        ),
        RetrievalPolicyVariant(
            label="es_rrf30_colbert_on",
            description="Elasticsearch + ColBERT + RRF k=30.",
            overrides={
                "retrieval": {
                    "hybrid_search": "elasticsearch",
                    "rrf_k": 30,
                    "reranking": False,
                    "colbert": {"enabled": True},
                },
                "graph": {"enabled": False},
            },
            retrieve_k=20,
        ),
        RetrievalPolicyVariant(
            label="es_rrf10_colbert_on",
            description="Elasticsearch + ColBERT + RRF k=10.",
            overrides={
                "retrieval": {
                    "hybrid_search": "elasticsearch",
                    "rrf_k": 10,
                    "reranking": False,
                    "colbert": {"enabled": True},
                },
                "graph": {"enabled": False},
            },
            retrieve_k=20,
        ),
        RetrievalPolicyVariant(
            label="es_rrf60_colbert_on_rerank20",
            description="Live retrieval candidates, then CrossEncoder reranking over top 20.",
            overrides={
                "retrieval": {
                    "hybrid_search": "elasticsearch",
                    "rrf_k": 60,
                    "reranking": True,
                    "colbert": {"enabled": True},
                },
                "graph": {"enabled": False},
            },
            retrieve_k=50,
            reranking_enabled=True,
            rerank_candidate_limit=20,
        ),
        RetrievalPolicyVariant(
            label="es_rrf60_colbert_on_rerank50",
            description="Live retrieval candidates, then CrossEncoder reranking over top 50.",
            overrides={
                "retrieval": {
                    "hybrid_search": "elasticsearch",
                    "rrf_k": 60,
                    "reranking": True,
                    "colbert": {"enabled": True},
                },
                "graph": {"enabled": False},
            },
            retrieve_k=50,
            reranking_enabled=True,
            rerank_candidate_limit=50,
            include_in_beir_smoke=True,
        ),
    ]
    if include_bm25:
        variants.insert(
            2,
            RetrievalPolicyVariant(
                label="bm25_rrf60_colbert_off",
                description="Semantic + in-process BM25 via RRF k=60; ColBERT off.",
                overrides={
                    "retrieval": {
                        "hybrid_search": "bm25",
                        "rrf_k": 60,
                        "reranking": False,
                        "colbert": {"enabled": False},
                    },
                    "graph": {"enabled": False},
                },
                retrieve_k=20,
            ),
        )
    return variants


def _write_variant_config(
    base_config: dict[str, Any],
    variant: RetrievalPolicyVariant,
    config_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    config = deep_merge(base_config, variant.overrides)
    config_path = config_dir / f"{variant.label}.yaml"
    write_private_text(
        config_path,
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    )
    return config_path, config


def _optional_dependency_snapshot() -> dict[str, Any]:
    snapshot = {}
    for module_name in ("elasticsearch", "rank_bm25", "qdrant_client", "sentence_transformers"):
        try:
            module = __import__(module_name)
            snapshot[module_name] = {
                "installed": True,
                "version": str(getattr(module, "__version__", "")),
            }
        except Exception as exc:
            snapshot[module_name] = {
                "installed": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
    return snapshot


def environment_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "optional_dependencies": _optional_dependency_snapshot(),
    }
    try:
        from kiara.retrieval.elasticsearch_service import get_status

        snapshot["elasticsearch"] = get_status(config)
    except Exception as exc:
        snapshot["elasticsearch"] = {
            "available": False,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    try:
        from kiara.retrieval.vectorstore import VectorStore

        snapshot["qdrant_collections"] = VectorStore.list_collections(config)
    except Exception as exc:
        snapshot["qdrant_collections_error"] = {
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    return snapshot


def _build_retriever(config: dict[str, Any], embedding_model: str | None = None):
    from kiara.retrieval.retriever import HybridRetriever

    return HybridRetriever(config, embedding_model=embedding_model)


def _build_reranker(config: dict[str, Any], enabled: bool):
    if not enabled:
        return None
    from kiara.retrieval.reranker import create_reranker

    return create_reranker(config)


def _reranker_snapshot(reranker: Any) -> dict[str, Any]:
    if reranker is None:
        return {"requested": False, "effective_enabled": False}
    return {
        "requested": True,
        "effective_enabled": bool(getattr(reranker, "enabled", False)),
        "model": getattr(reranker, "_model_name", None),
        "top_k": getattr(reranker, "_top_k", None),
        "batch_size": getattr(reranker, "_batch_size", None),
    }


def _apply_reranking(
    query: str,
    results: list[dict[str, Any]],
    reranker: Any,
    candidate_limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reranker is None:
        return results, {
            "requested": False,
            "effective_enabled": False,
            "applied": False,
            "candidate_limit": candidate_limit,
            "input_chunks": len(results),
            "reranked_chunks": 0,
            "latency_ms": 0,
        }

    candidates = (
        results[:candidate_limit]
        if candidate_limit is not None and candidate_limit > 0
        else results
    )
    tail = results[len(candidates) :]
    started = time.monotonic()
    reranked = reranker.rerank(query, candidates, top_k=len(candidates))
    latency_ms = int((time.monotonic() - started) * 1000)
    return reranked + tail, {
        "requested": True,
        "effective_enabled": bool(getattr(reranker, "enabled", False)),
        "applied": bool(candidates),
        "candidate_limit": candidate_limit,
        "input_chunks": len(results),
        "reranked_chunks": len(reranked),
        "tail_chunks": len(tail),
        "rerank_score_count": sum(1 for item in reranked if item.get("rerank_score") is not None),
        "latency_ms": latency_ms,
        "runtime": _reranker_snapshot(reranker),
    }


def _with_final_context_ranks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, result in enumerate(results, 1):
        item = result.copy()
        item["final_context_rank"] = rank
        ranked.append(item)
    return ranked


def _average_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["metrics"].get(key) for row in rows if row["metrics"].get(key) is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _count_path_states(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = {}
    for row in rows:
        for path, state in (row.get("retriever_path_states") or {}).items():
            counts.setdefault(path, Counter())[state] += 1
    return {path: dict(counter) for path, counter in sorted(counts.items())}


def _path_state_warnings(path_state_counts: dict[str, dict[str, int]]) -> list[str]:
    warnings = []
    for path, counts in sorted(path_state_counts.items()):
        for state in ("disabled", "empty", "error"):
            if counts.get(state):
                warnings.append(f"{path}_path_{state}")
    return warnings


def _filter_datasources_for_golden_case(
    case: dict[str, Any],
    *,
    enabled: bool,
) -> list[int] | None:
    if not enabled:
        return None
    values = [int(value) for value in case.get("required_datasources") or []]
    return values or None


def run_golden_variant(
    *,
    variant: RetrievalPolicyVariant,
    config: dict[str, Any],
    golden_set_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...],
    filter_required_datasources: bool = False,
    embedding_model: str | None = None,
    retriever_factory: Callable[[dict[str, Any], str | None], Any] | None = None,
    reranker_factory: Callable[[dict[str, Any], bool], Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cases = golden.load_golden_set(golden_set_path)
    ensure_private_dir(output_dir)
    details_path = output_dir / f"{variant.label}_golden_details.jsonl"
    summary_path = output_dir / f"{variant.label}_golden_summary.json"

    retriever = (retriever_factory or _build_retriever)(config, embedding_model)
    reranker = (reranker_factory or _build_reranker)(config, variant.reranking_enabled)
    started_at = _utc_timestamp()
    started = time.monotonic()
    details: list[dict[str, Any]] = []
    latencies: list[int] = []

    for index, case in enumerate(cases, 1):
        case_started = time.monotonic()
        query = str(case["question"])
        error = ""
        diagnostics = None
        try:
            results = retriever.retrieve(
                query=query,
                k=variant.retrieve_k,
                filter_datasource_ids=_filter_datasources_for_golden_case(
                    case,
                    enabled=filter_required_datasources,
                ),
            )
            getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
            diagnostics = getter() if callable(getter) else None
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            results = []
            getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
            diagnostics = getter() if callable(getter) else None
            error = f"{exc.__class__.__name__}: {exc}"

        reranked, rerank_stats = _apply_reranking(
            query,
            results,
            reranker,
            variant.rerank_candidate_limit,
        )
        ranked = _with_final_context_ranks(reranked)
        chunks = [golden._compact_chunk(chunk, rank) for rank, chunk in enumerate(ranked, 1)]
        latency_ms = int((time.monotonic() - case_started) * 1000)
        latencies.append(latency_ms)
        metrics = golden._metrics_for_case(case, chunks, k_values)
        row = {
            "case_index": index,
            "id": case["id"],
            "question": case["question"],
            "tags": list(case.get("tags") or []),
            "expected_answer_type": case["expected_answer_type"],
            "expected_failure_class": case["expected_failure_class"],
            "expected_behavior": case["expected_behavior"],
            "required_datasources": list(case.get("required_datasources") or []),
            "short_or_numeric_answer": bool(case["short_or_numeric_answer"]),
            "sensitivity": case["sensitivity"],
            "evaluation_rule": case["evaluation_rule"],
            "expected_sources": case["expected_sources"],
            "answer": "",
            "answer_status": "not_generated" if not error else "error",
            "error": error,
            "latency_ms": latency_ms,
            "retrieved_chunks": chunks,
            "metrics": metrics,
            "retriever_path_states": golden._path_states(diagnostics),
            "retriever_path_latencies_ms": golden._path_latencies_ms(diagnostics),
            "retrieval_diagnostics": diagnostics,
            "reranking": rerank_stats,
        }
        details.append(row)
        if progress_callback:
            progress_callback(
                {
                    "dataset": "golden",
                    "variant": variant.label,
                    "current": index,
                    "total": len(cases),
                    "id": case["id"],
                    "latency_ms": latency_ms,
                    "error": error,
                }
            )

    path_state_counts = _count_path_states(details)
    overall: dict[str, Any] = {
        "cases_total": len(cases),
        "cases_evaluated": len(details),
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "answer_status_counts": dict(Counter(row["answer_status"] for row in details)),
        "expected_behavior_counts": dict(Counter(str(row["expected_behavior"]) for row in details)),
        "expected_failure_class_counts": dict(
            Counter(str(row["expected_failure_class"]) for row in details)
        ),
        "short_or_numeric_answer_count": sum(
            1 for row in details if row["short_or_numeric_answer"]
        ),
    }
    for k in k_values:
        overall[f"recall@{k}"] = _average_metric(details, f"recall@{k}")
        overall[f"gold_ctx@{k}"] = _average_metric(details, f"gold_ctx@{k}")

    priority_cases = {
        row["id"]: {
            "recall@20": row["metrics"].get("recall@20"),
            "gold_ctx@20": row["metrics"].get("gold_ctx@20"),
            "answer_status": row["answer_status"],
            "first_hit": (
                row["metrics"]
                .get("source_hits_by_k", {})
                .get("20", {})
                .get("hits", [{}])[0]
                .get("first_hit")
                if row["metrics"].get("source_hits_by_k", {}).get("20", {}).get("hits")
                else None
            ),
            "top_sources": [chunk.get("source") for chunk in row["retrieved_chunks"][:5]],
        }
        for row in details
        if row["id"] in PRIORITY_GOLDEN_CASE_IDS
    }
    summary = {
        "schema": "kiara-retrieval-policy-golden-summary",
        "schema_version": 1,
        "created_at": _utc_timestamp(),
        "run": {
            "variant": variant.label,
            "started_at": started_at,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "k_values": list(k_values),
            "retrieve_k": variant.retrieve_k,
            "filter_required_datasources": filter_required_datasources,
            "reranking_enabled": variant.reranking_enabled,
            "rerank_candidate_limit": variant.rerank_candidate_limit,
        },
        "overall": overall,
        "path_state_counts": path_state_counts,
        "partial_state_warnings": _path_state_warnings(path_state_counts),
        "reranker": _reranker_snapshot(reranker),
        "priority_cases": priority_cases,
        "artifacts": {
            "summary": str(summary_path),
            "details": str(details_path),
            "golden_set": str(golden_set_path),
        },
    }
    _write_jsonl(details_path, details)
    _write_json(summary_path, summary)
    return summary


def _crag_metrics_for_row(
    answers: list[str],
    chunks: list[dict[str, Any]],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for k in k_values:
        hit = crag._context_contains_answer(chunks, answers, top_k=k)
        metrics[f"gold_answer_in_context@{k}"] = 1.0 if hit else 0.0
        metrics[f"gold_ctx@{k}"] = 1.0 if hit else 0.0
        metrics[f"recall_proxy@{k}"] = 1.0 if hit else 0.0
    return metrics


def _load_sprint2_case_context(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    context = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        interaction_id = str(row.get("interaction_id") or "")
        if not interaction_id:
            continue
        context[interaction_id] = {
            "failure_class": row.get("failure_class"),
            "loss_stage": row.get("loss_stage"),
            "answer_label": row.get("answer_label"),
            "confidence": row.get("confidence"),
            "recommended_next_step": row.get("recommended_next_step"),
            "warnings": list(row.get("warnings") or []),
        }
    return context


def run_crag_variant(
    *,
    variant: RetrievalPolicyVariant,
    config: dict[str, Any],
    manifest_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...],
    sprint2_audit_path: Path | None = None,
    embedding_model: str | None = None,
    context_excerpt_chars: int = 2000,
    retriever_factory: Callable[[dict[str, Any], str | None], Any] | None = None,
    reranker_factory: Callable[[dict[str, Any], bool], Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    manifest_rows = list(crag.iter_manifest(manifest_path))
    sprint2_context = _load_sprint2_case_context(sprint2_audit_path)
    ensure_private_dir(output_dir)
    details_path = output_dir / f"{variant.label}_crag_full136_details.jsonl"
    summary_path = output_dir / f"{variant.label}_crag_full136_summary.json"

    retriever = (retriever_factory or _build_retriever)(config, embedding_model)
    reranker = (reranker_factory or _build_reranker)(config, variant.reranking_enabled)
    started_at = _utc_timestamp()
    started = time.monotonic()
    details: list[dict[str, Any]] = []
    latencies: list[int] = []

    for index, entry in enumerate(manifest_rows, 1):
        case_started = time.monotonic()
        interaction_id = str(entry.get("interaction_id") or "")
        query = str(entry.get("query") or "")
        answers = [str(value) for value in entry.get("answers") or []]
        diagnostics = None
        error = ""
        try:
            results = retriever.retrieve(
                query=query,
                k=variant.retrieve_k,
                filter_dir=str(entry.get("doc_dir") or ""),
            )
            getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
            diagnostics = getter() if callable(getter) else None
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            results = []
            getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
            diagnostics = getter() if callable(getter) else None
            error = f"{exc.__class__.__name__}: {exc}"

        reranked, rerank_stats = _apply_reranking(
            query,
            results,
            reranker,
            variant.rerank_candidate_limit,
        )
        ranked = _with_final_context_ranks(reranked)
        chunks = crag._compact_context_chunks(ranked, excerpt_chars=context_excerpt_chars)
        latency_ms = int((time.monotonic() - case_started) * 1000)
        latencies.append(latency_ms)
        path_diagnostics = crag._retrieval_path_gold_diagnostics(
            {"retrieval_diagnostics": diagnostics},
            chunks,
            answers,
        )
        metrics = _crag_metrics_for_row(answers, chunks, k_values)
        doc_dir = str(entry.get("doc_dir") or "")
        sources = [str(chunk.get("source") or "") for chunk in chunks if chunk.get("source")]
        source_hits = [crag._source_under_doc_dir(source, doc_dir) for source in sources]
        row = {
            "case_index": index,
            "interaction_id": interaction_id,
            "query": query,
            "answers": answers,
            "domain": entry.get("domain", ""),
            "question_type": entry.get("question_type", ""),
            "static_or_dynamic": entry.get("static_or_dynamic", ""),
            "popularity": entry.get("popularity", ""),
            "doc_dir": doc_dir,
            "file_count": entry.get("file_count", 0),
            "short_or_numeric_answer": crag._has_low_confidence_literal_answer(answers),
            "error": error,
            "latency_ms": latency_ms,
            "context_chunks": len(chunks),
            "context_chunk_details": chunks,
            "metrics": metrics,
            "gold_answer_in_context": bool(metrics.get(f"gold_answer_in_context@{max(k_values)}")),
            "any_source_under_doc_dir": bool(source_hits and any(source_hits)),
            "all_sources_under_doc_dir": bool(source_hits and all(source_hits)),
            "path_gold_ranks": path_diagnostics["ranks"],
            "path_gold_scores": path_diagnostics["scores"],
            "path_gold_hits": path_diagnostics["hits"],
            "retriever_path_states": path_diagnostics["path_states"],
            "retriever_path_latencies_ms": path_diagnostics["path_latencies_ms"],
            "retrieval_diagnostics": diagnostics,
            "reranking": rerank_stats,
            "sprint2_context": sprint2_context.get(interaction_id, {}),
        }
        details.append(row)
        if progress_callback:
            progress_callback(
                {
                    "dataset": "crag_full136",
                    "variant": variant.label,
                    "current": index,
                    "total": len(manifest_rows),
                    "id": interaction_id,
                    "latency_ms": latency_ms,
                    "error": error,
                }
            )

    path_state_counts = _count_path_states(details)
    overall: dict[str, Any] = {
        "records_total": len(manifest_rows),
        "records_evaluated": len(details),
        "records_failed": sum(1 for row in details if row["error"]),
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "avg_context_chunks": (
            sum(int(row["context_chunks"]) for row in details) / len(details) if details else 0.0
        ),
        "any_source_under_doc_dir": sum(1 for row in details if row["any_source_under_doc_dir"]),
        "all_sources_under_doc_dir": sum(1 for row in details if row["all_sources_under_doc_dir"]),
        "short_or_numeric_answer_count": sum(
            1 for row in details if row["short_or_numeric_answer"]
        ),
    }
    for k in k_values:
        overall[f"gold_answer_in_context@{k}"] = sum(
            float(row["metrics"][f"gold_answer_in_context@{k}"]) for row in details
        )
        overall[f"gold_answer_in_context@{k}_rate"] = (
            overall[f"gold_answer_in_context@{k}"] / len(details) if details else 0.0
        )
        overall[f"gold_ctx@{k}"] = overall[f"gold_answer_in_context@{k}_rate"]
        overall[f"recall_proxy@{k}"] = overall[f"gold_answer_in_context@{k}_rate"]

    failure_class_counts = Counter(
        str((row.get("sprint2_context") or {}).get("failure_class") or "unknown") for row in details
    )
    summary = {
        "schema": "kiara-retrieval-policy-crag-full136-summary",
        "schema_version": 1,
        "created_at": _utc_timestamp(),
        "run": {
            "variant": variant.label,
            "started_at": started_at,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "k_values": list(k_values),
            "retrieve_k": variant.retrieve_k,
            "reranking_enabled": variant.reranking_enabled,
            "rerank_candidate_limit": variant.rerank_candidate_limit,
            "scope": "per-question",
            "manifest_path": str(manifest_path),
        },
        "overall": overall,
        "path_state_counts": path_state_counts,
        "partial_state_warnings": _path_state_warnings(path_state_counts),
        "reranker": _reranker_snapshot(reranker),
        "sprint2_failure_class_counts": dict(failure_class_counts),
        "artifacts": {
            "summary": str(summary_path),
            "details": str(details_path),
            "manifest": str(manifest_path),
            "sprint2_audit": str(sprint2_audit_path) if sprint2_audit_path else None,
        },
    }
    _write_jsonl(details_path, details)
    _write_json(summary_path, summary)
    return summary


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _case_metric_map(details_path: str, id_key: str, metric_key: str) -> dict[str, float | None]:
    rows = _load_jsonl(details_path)
    return {str(row.get(id_key)): row.get("metrics", {}).get(metric_key) for row in rows}


def _compare_case_metric(
    baseline_details: str,
    candidate_details: str,
    *,
    id_key: str,
    metric_key: str,
) -> dict[str, list[str]]:
    baseline = _case_metric_map(baseline_details, id_key, metric_key)
    candidate = _case_metric_map(candidate_details, id_key, metric_key)
    improvements = []
    regressions = []
    for case_id, candidate_value in candidate.items():
        base_value = baseline.get(case_id)
        if base_value is None or candidate_value is None:
            continue
        if float(candidate_value) > float(base_value):
            improvements.append(case_id)
        elif float(candidate_value) < float(base_value):
            regressions.append(case_id)
    return {"improvements": sorted(improvements), "regressions": sorted(regressions)}


def _variant_metric_table_row(variant_result: dict[str, Any], k: int) -> dict[str, Any]:
    golden_summary = variant_result.get("golden", {})
    crag_summary = variant_result.get("crag_full136", {})
    golden_overall = golden_summary.get("overall", {})
    crag_overall = crag_summary.get("overall", {})
    return {
        "variant": variant_result["variant"]["label"],
        "golden_recall": golden_overall.get(f"recall@{k}"),
        "golden_gold_ctx": golden_overall.get(f"gold_ctx@{k}"),
        "golden_latency_ms": golden_overall.get("avg_latency_ms"),
        "crag_recall_proxy": crag_overall.get(f"recall_proxy@{k}"),
        "crag_gold_ctx": crag_overall.get(f"gold_ctx@{k}"),
        "crag_latency_ms": crag_overall.get("avg_latency_ms"),
        "golden_paths": golden_summary.get("path_state_counts", {}),
        "crag_paths": crag_summary.get("path_state_counts", {}),
    }


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def _short_path_state(path_counts: dict[str, dict[str, int]]) -> str:
    parts = []
    for path, counts in sorted(path_counts.items()):
        state_bits = ",".join(f"{state}:{count}" for state, count in sorted(counts.items()))
        parts.append(f"{path}={state_bits}")
    return "; ".join(parts)


def _decision_template(summary: dict[str, Any], *, baseline_label: str, k: int) -> dict[str, Any]:
    variants = {row["variant"]["label"]: row for row in summary["variants"]}
    baseline = variants.get(baseline_label, {})
    colbert_off = variants.get("es_rrf60_colbert_off", {})
    dense = variants.get("dense_only_k20", {})
    rerank20 = variants.get("es_rrf60_colbert_on_rerank20", {})
    rerank50 = variants.get("es_rrf60_colbert_on_rerank50", {})

    def golden_metric(item: dict[str, Any], key: str) -> float | None:
        value = item.get("golden", {}).get("overall", {}).get(key)
        return float(value) if value is not None else None

    def crag_metric(item: dict[str, Any], key: str) -> float | None:
        value = item.get("crag_full136", {}).get("overall", {}).get(key)
        return float(value) if value is not None else None

    base_golden = golden_metric(baseline, f"gold_ctx@{k}")
    off_golden = golden_metric(colbert_off, f"gold_ctx@{k}")
    base_crag = crag_metric(baseline, f"gold_ctx@{k}")
    off_crag = crag_metric(colbert_off, f"gold_ctx@{k}")
    rerank20_golden = golden_metric(rerank20, f"gold_ctx@{k}")
    rerank50_golden = golden_metric(rerank50, f"gold_ctx@{k}")
    rerank20_crag = crag_metric(rerank20, f"gold_ctx@{k}")
    rerank50_crag = crag_metric(rerank50, f"gold_ctx@{k}")

    def avg_latency_ms(item: dict[str, Any]) -> float | None:
        values = []
        for dataset_key in ("golden", "crag_full136"):
            value = item.get(dataset_key, {}).get("overall", {}).get("avg_latency_ms")
            if value is not None:
                values.append(float(value))
        return sum(values) / len(values) if values else None

    def combined_gold_ctx_score(item: dict[str, Any]) -> float:
        return (golden_metric(item, f"gold_ctx@{k}") or 0.0) + (
            crag_metric(item, f"gold_ctx@{k}") or 0.0
        )

    def policy_key(item: dict[str, Any]) -> tuple[float, float]:
        latency = avg_latency_ms(item)
        return (combined_gold_ctx_score(item), -(latency if latency is not None else 0.0))

    colbert_decision = "keep_default"
    if (
        base_golden is not None
        and off_golden is not None
        and base_crag is not None
        and off_crag is not None
    ):
        if off_golden >= base_golden and off_crag >= base_crag:
            colbert_decision = "disable_or_opt_in"
        elif off_golden < base_golden or off_crag < base_crag:
            colbert_decision = "keep_default"

    reranking_decision = "keep_disabled"
    candidate_limit = None
    rerank_candidates = [
        item
        for item in (rerank20, rerank50)
        if item and item.get("golden", {}).get("reranker", {}).get("effective_enabled")
    ]
    non_rerank_candidates = [
        item
        for item in variants.values()
        if item and not item.get("variant", {}).get("reranking_enabled")
    ]
    best_rerank = max(rerank_candidates, key=policy_key) if rerank_candidates else {}
    best_non_rerank = (
        max(non_rerank_candidates, key=policy_key) if non_rerank_candidates else baseline
    )
    best_rerank_score = combined_gold_ctx_score(best_rerank) if best_rerank else None
    best_non_rerank_score = combined_gold_ctx_score(best_non_rerank) if best_non_rerank else None
    if (
        best_rerank
        and best_non_rerank_score is not None
        and best_rerank_score is not None
        and best_rerank_score > best_non_rerank_score
    ):
        reranking_decision = "enable"
        best_rerank_label = best_rerank["variant"]["label"]
        candidate_limit = 20 if "rerank20" in best_rerank_label else 50

    return {
        "live_default": baseline_label,
        "colbert": {
            "decision": colbert_decision,
            "basis": {
                "baseline_golden_gold_ctx": base_golden,
                "colbert_off_golden_gold_ctx": off_golden,
                "baseline_crag_gold_ctx": base_crag,
                "colbert_off_crag_gold_ctx": off_crag,
            },
        },
        "reranking": {
            "decision": reranking_decision,
            "candidate_limit": candidate_limit,
            "basis": {
                "comparison": "best_rerank_variant_vs_best_non_rerank_variant",
                "best_non_rerank_variant": best_non_rerank.get("variant", {}).get("label"),
                "best_non_rerank_combined_gold_ctx": best_non_rerank_score,
                "best_non_rerank_avg_latency_ms": avg_latency_ms(best_non_rerank)
                if best_non_rerank
                else None,
                "best_rerank_variant": best_rerank.get("variant", {}).get("label"),
                "best_rerank_combined_gold_ctx": best_rerank_score,
                "best_rerank_avg_latency_ms": avg_latency_ms(best_rerank) if best_rerank else None,
                "baseline_combined_gold_ctx": (base_golden or 0.0) + (base_crag or 0.0),
                "rerank20_golden_gold_ctx": rerank20_golden,
                "rerank50_golden_gold_ctx": rerank50_golden,
                "rerank20_crag_gold_ctx": rerank20_crag,
                "rerank50_crag_gold_ctx": rerank50_crag,
            },
        },
        "dense_only": {
            "golden_gold_ctx": golden_metric(dense, f"gold_ctx@{k}"),
            "crag_gold_ctx": crag_metric(dense, f"gold_ctx@{k}"),
        },
        "rollback": "Keep previous /etc/kiara/kiara.yaml and revert retrieval.* keys, then run ./deploy.sh and service checks after explicit approval.",
        "note": "Automated suggestion; final production config change still requires explicit user approval.",
    }


def _write_report(summary: dict[str, Any], report_path: Path, *, k: int) -> None:
    lines = [
        "# Sprint 4 Retrieval Policy Ablation",
        "",
        f"Run label: `{summary['run']['label']}`",
        f"Created: `{summary['created_at']}`",
        "",
        "## Environment",
        "",
    ]
    env = summary.get("environment", {})
    deps = env.get("optional_dependencies", {})
    lines.append(f"- Python: `{env.get('python_executable')}` ({env.get('python')})")
    lines.append(f"- Elasticsearch: `{env.get('elasticsearch')}`")
    lines.append(f"- Optional deps: `{deps}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| Variant | Golden Recall@20 | Golden GoldCtx@20 | Golden ms | CRAG RecallProxy@20 | CRAG GoldCtx@20 | CRAG ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary["tables"][f"at_{k}"]:
        lines.append(
            "| {variant} | {gr} | {gg} | {gl} | {cr} | {cg} | {cl} |".format(
                variant=row["variant"],
                gr=_format_float(row["golden_recall"]),
                gg=_format_float(row["golden_gold_ctx"]),
                gl=_format_ms(row["golden_latency_ms"]),
                cr=_format_float(row["crag_recall_proxy"]),
                cg=_format_float(row["crag_gold_ctx"]),
                cl=_format_ms(row["crag_latency_ms"]),
            )
        )
    lines.append("")
    lines.append("## Path States")
    lines.append("")
    for variant in summary["variants"]:
        label = variant["variant"]["label"]
        lines.append(
            f"- `{label}` Golden: {_short_path_state(variant['golden']['path_state_counts'])}"
        )
        lines.append(
            f"- `{label}` CRAG: {_short_path_state(variant['crag_full136']['path_state_counts'])}"
        )
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(summary["decision_template"], ensure_ascii=False, indent=2, sort_keys=True)
    )
    lines.append("```")
    lines.append("")
    lines.append("## Case Deltas vs Baseline")
    lines.append("")
    for label, deltas in summary["comparisons"]["golden"].items():
        if label == summary["comparisons"]["baseline_label"]:
            continue
        lines.append(f"- Golden `{label}`: +{deltas['improvements']} / -{deltas['regressions']}")
    for label, deltas in summary["comparisons"]["crag_full136"].items():
        if label == summary["comparisons"]["baseline_label"]:
            continue
        lines.append(
            f"- CRAG `{label}`: +{len(deltas['improvements'])} cases / -{len(deltas['regressions'])} cases"
        )
    lines.append("")
    write_private_text(report_path, "\n".join(lines) + "\n")


def _run_beir_smoke(
    *,
    variant: RetrievalPolicyVariant,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    k_values: tuple[int, ...],
    max_queries: int,
) -> dict[str, Any] | None:
    if not dataset_dir.exists():
        return None
    from kiara.benchmarks.beir import evaluate_beir_retrieval

    retriever_name = (
        "dense"
        if variant.overrides.get("retrieval", {}).get("hybrid_search") == "none"
        else "hybrid"
    )
    stats = evaluate_beir_retrieval(
        dataset_dir,
        config_path=str(config_path),
        retriever_name=retriever_name,
        k_values=k_values,
        max_queries=max_queries,
        run_label=f"{variant.label}_beir_smoke{max_queries}",
        output_dir=output_dir,
        reranking_enabled=variant.reranking_enabled,
        rerank_candidate_limit=variant.rerank_candidate_limit,
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    return {
        "summary_path": stats.summary_path,
        "details_path": stats.details_path,
        "overall": summary.get("overall", {}),
        "colbert_diagnostics": summary.get("colbert_diagnostics", {}),
        "retrieval_failures": summary.get("retrieval_failures", {}),
    }


def run_retrieval_policy_matrix(
    *,
    config_path: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_label: str | None = None,
    golden_set_path: Path = DEFAULT_GOLDEN_SET,
    crag_manifest_path: Path = DEFAULT_CRAG_MANIFEST,
    sprint2_audit_path: Path | None = Path(
        "/var/lib/kiara/benchmarks/crag_full136/results/task1_full136_vertical_audit_diag_full.json"
    ),
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    include_bm25: bool = True,
    run_beir_smoke: bool = False,
    beir_dataset_dir: Path = DEFAULT_BEIR_DATASET,
    beir_max_queries: int = 20,
    filter_required_datasources: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RetrievalPolicyRunStats:
    from kiara.config import load_config

    label = _slug(run_label or f"sprint4_retrieval_policy_{_timestamp_for_filename()}")
    output_dir = output_root / label
    configs_dir = output_dir / "configs"
    golden_dir = output_dir / "golden"
    crag_dir = output_dir / "crag_full136"
    beir_dir = output_dir / "beir_smoke"
    ensure_private_dir(output_dir)

    base_config = load_config(config_path)
    env = environment_snapshot(base_config)
    variants = build_default_variants(include_bm25=include_bm25)
    variant_results = []
    for variant in variants:
        config_file, variant_config = _write_variant_config(base_config, variant, configs_dir)
        golden_summary = run_golden_variant(
            variant=variant,
            config=variant_config,
            golden_set_path=golden_set_path,
            output_dir=golden_dir,
            k_values=k_values,
            filter_required_datasources=filter_required_datasources,
            progress_callback=progress_callback,
        )
        crag_summary = run_crag_variant(
            variant=variant,
            config=variant_config,
            manifest_path=crag_manifest_path,
            output_dir=crag_dir,
            k_values=k_values,
            sprint2_audit_path=sprint2_audit_path,
            progress_callback=progress_callback,
        )
        beir_summary = None
        if run_beir_smoke and variant.include_in_beir_smoke:
            beir_summary = _run_beir_smoke(
                variant=variant,
                config_path=config_file,
                dataset_dir=beir_dataset_dir,
                output_dir=beir_dir,
                k_values=k_values,
                max_queries=beir_max_queries,
            )
        variant_results.append(
            {
                "variant": asdict(variant),
                "config_path": str(config_file),
                "golden": golden_summary,
                "crag_full136": crag_summary,
                "beir_smoke": beir_summary,
            }
        )

    baseline_label = "live_es_rrf60_colbert_on"
    baseline = next(
        (item for item in variant_results if item["variant"]["label"] == baseline_label),
        variant_results[0],
    )
    comparisons = {
        "baseline_label": baseline["variant"]["label"],
        "golden": {},
        "crag_full136": {},
    }
    for item in variant_results:
        label_item = item["variant"]["label"]
        comparisons["golden"][label_item] = _compare_case_metric(
            baseline["golden"]["artifacts"]["details"],
            item["golden"]["artifacts"]["details"],
            id_key="id",
            metric_key=f"gold_ctx@{max(k_values)}",
        )
        comparisons["crag_full136"][label_item] = _compare_case_metric(
            baseline["crag_full136"]["artifacts"]["details"],
            item["crag_full136"]["artifacts"]["details"],
            id_key="interaction_id",
            metric_key=f"gold_ctx@{max(k_values)}",
        )

    table_at_max = [_variant_metric_table_row(item, max(k_values)) for item in variant_results]
    summary = {
        "schema": "kiara-retrieval-policy-ablation-matrix",
        "schema_version": 1,
        "created_at": _utc_timestamp(),
        "run": {
            "label": label,
            "config_path": config_path,
            "output_dir": str(output_dir),
            "k_values": list(k_values),
            "golden_set_path": str(golden_set_path),
            "crag_manifest_path": str(crag_manifest_path),
            "sprint2_audit_path": str(sprint2_audit_path) if sprint2_audit_path else None,
            "run_beir_smoke": run_beir_smoke,
            "beir_dataset_dir": str(beir_dataset_dir),
            "beir_max_queries": beir_max_queries if run_beir_smoke else None,
            "productive_config_changed": False,
        },
        "environment": env,
        "variants": variant_results,
        "tables": {f"at_{max(k_values)}": table_at_max},
        "comparisons": comparisons,
    }
    summary["decision_template"] = _decision_template(
        summary,
        baseline_label=baseline["variant"]["label"],
        k=max(k_values),
    )
    summary_path = output_dir / "retrieval_policy_ablation_summary.json"
    report_path = output_dir / "retrieval_policy_decision_report.md"
    _write_json(summary_path, summary)
    _write_report(summary, report_path, k=max(k_values))
    return RetrievalPolicyRunStats(
        label=label,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        report_path=str(report_path),
        variants=len(variants),
    )


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--crag-manifest", type=Path, default=DEFAULT_CRAG_MANIFEST)
    parser.add_argument(
        "--sprint2-audit",
        type=Path,
        default=Path(
            "/var/lib/kiara/benchmarks/crag_full136/results/task1_full136_vertical_audit_diag_full.json"
        ),
    )
    parser.add_argument("--k-values", default="1,3,5,10,20")
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--beir-smoke", action="store_true")
    parser.add_argument("--beir-dataset-dir", type=Path, default=DEFAULT_BEIR_DATASET)
    parser.add_argument("--beir-max-queries", type=int, default=20)
    parser.add_argument("--filter-required-datasources", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    def progress(event: dict[str, Any]) -> None:
        if args.quiet:
            return
        print(
            "{dataset} {variant} {current}/{total} {id} {latency_ms}ms".format(
                dataset=event["dataset"],
                variant=event["variant"],
                current=event["current"],
                total=event["total"],
                id=event["id"],
                latency_ms=event["latency_ms"],
            ),
            flush=True,
        )

    stats = run_retrieval_policy_matrix(
        config_path=args.config_path,
        output_root=args.output_root,
        run_label=args.run_label,
        golden_set_path=args.golden_set,
        crag_manifest_path=args.crag_manifest,
        sprint2_audit_path=args.sprint2_audit,
        k_values=parse_k_values(args.k_values),
        include_bm25=not args.no_bm25,
        run_beir_smoke=args.beir_smoke,
        beir_dataset_dir=args.beir_dataset_dir,
        beir_max_queries=args.beir_max_queries,
        filter_required_datasources=args.filter_required_datasources,
        progress_callback=progress,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
