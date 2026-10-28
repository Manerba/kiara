"""KIara-specific Golden Set evaluation.

This module evaluates curated company-style questions against the current
KIara retrieval/RAG pipeline. The curated data can live outside the repository
when it contains sensitive source hints; only schema, runner and sanitized
examples are versioned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from kiara.benchmarks.artifacts import write_private_json, write_private_jsonl

DEFAULT_GOLDEN_SET = Path("/var/lib/kiara/benchmarks/kiara_golden/golden_set_v1.jsonl")
DEFAULT_OUTPUT_DIR = Path("/var/lib/kiara/benchmarks/kiara_golden/results")
DEFAULT_K_VALUES = (1, 3, 5, 10, 20)
DETAIL_EXCERPT_CHARS = 1000

REQUIRED_CASE_FIELDS = {
    "id",
    "question",
    "expected_sources",
    "expected_answer_type",
    "evaluation_rule",
    "required_datasources",
    "expected_failure_class",
    "expected_behavior",
    "sensitivity",
    "short_or_numeric_answer",
}

_MISSING_PATTERNS = (
    "ich weiss es nicht",
    "ich weiß es nicht",
    "keine relevanten dokumente gefunden",
    "keine information",
    "nicht gefunden",
    "nicht beantwortet",
    "nicht direkt beantwortet",
    "kann nicht beantwortet werden",
    "not enough information",
    "cannot find",
    "can't find",
)

ALLOWED_EVALUATION_RULES = {
    "contains_any",
    "contains_all",
    "regex_any",
    "source_only",
}
ALLOWED_EXPECTED_BEHAVIORS = {"rag", "tool_api", "hybrid"}
ALLOWED_SOURCE_TYPES = {
    "document",
    "docx",
    "email",
    "invoice_pdf",
    "pdf",
    "spreadsheet",
    "sql_object",
    "technical_doc",
}
ALLOWED_EXPECTED_ANSWER_TYPES = {
    "amount",
    "amount_sum",
    "concept_summary",
    "count",
    "erp_sql_count",
    "erp_sql_fact",
    "erp_sql_status",
    "exclusion_reason",
    "identifier",
    "invoice_lookup",
    "list",
    "multi_source_comparison",
    "multi_source_summary",
    "path_or_filename",
    "procedure_summary",
    "routing_decision",
    "status",
    "structure_summary",
    "summary",
    "technical_fact",
    "technical_summary",
}
ALLOWED_EXPECTED_FAILURE_CLASSES = {
    "book_docx_pdf_semantic",
    "book_docx_semantic",
    "book_pdf_semantic",
    "dense_rank_miss_large_pdf",
    "dense_rank_miss_technical_synonym",
    "dense_rank_miss_temporal_invoice",
    "email_invoice_rank_miss",
    "generator_only",
    "generator_only_noise_confusion",
    "generator_only_short_numeric_multi_doc",
    "lexical_id_email_invoice",
    "lexical_id_exact_match",
    "loader_sensitive_large_pdf",
    "loader_sensitive_pdf_identifier",
    "loader_sensitive_school_docs",
    "loader_sensitive_spreadsheet_metadata",
    "loader_sensitive_structure",
    "metadata_path_question",
    "multi_hop_book_pdf",
    "short_numeric_multi_doc_count",
    "short_or_numeric_answer",
    "tool_api_expected_not_rag",
}


@dataclass(frozen=True)
class GoldenEvalStats:
    """Summary returned by a KIara Golden Set run."""

    run_label: str
    mode: str
    golden_set_path: str
    summary_path: str
    details_path: str
    cases_evaluated: int
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
    write_private_jsonl(path, rows, sort_keys=True)


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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "run"


def _sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _effective_config_path(config_path: str | None) -> Path:
    from kiara.config import DEFAULT_CONFIG_PATH

    return Path(config_path or os.environ.get("KIARA_CONFIG") or DEFAULT_CONFIG_PATH)


def _git_output(args: list[str], repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _git_snapshot(repo_root: Path | None = None) -> dict[str, Any]:
    repo = repo_root or Path(__file__).resolve().parents[2]
    try:
        top_level = Path(_git_output(["rev-parse", "--show-toplevel"], repo))
        status = _git_output(["status", "--porcelain"], top_level)
        dirty_paths = [line for line in status.splitlines() if line.strip()]
        return {
            "available": True,
            "repo_root": str(top_level),
            "commit": _git_output(["rev-parse", "HEAD"], top_level),
            "branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"], top_level),
            "dirty": bool(dirty_paths),
            "dirty_count": len(dirty_paths),
            "status_porcelain": dirty_paths[:200],
            "status_truncated": len(dirty_paths) > 200,
        }
    except Exception as exc:
        return {
            "available": False,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def _reject_existing_outputs(paths: list[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Golden-Set-Output existiert bereits; --overwrite ist erforderlich: "
            + ", ".join(existing)
        )


def validate_golden_case(case: dict[str, Any], *, line_no: int | None = None) -> None:
    prefix = f"case line {line_no}" if line_no is not None else "case"
    missing = sorted(REQUIRED_CASE_FIELDS - set(case))
    if missing:
        raise ValueError(f"{prefix} missing required fields: {', '.join(missing)}")

    for key in ("id", "question", "expected_answer_type", "evaluation_rule"):
        if not str(case.get(key) or "").strip():
            raise ValueError(f"{prefix} field {key!r} must be non-empty")

    behavior = str(case.get("expected_behavior") or "").strip()
    if behavior not in ALLOWED_EXPECTED_BEHAVIORS:
        raise ValueError(f"{prefix} unknown expected_behavior: {behavior!r}")
    answer_type = str(case.get("expected_answer_type") or "").strip()
    if answer_type not in ALLOWED_EXPECTED_ANSWER_TYPES:
        raise ValueError(f"{prefix} unknown expected_answer_type: {answer_type!r}")
    failure_class = str(case.get("expected_failure_class") or "").strip()
    if failure_class not in ALLOWED_EXPECTED_FAILURE_CLASSES:
        raise ValueError(f"{prefix} unknown expected_failure_class: {failure_class!r}")
    evaluation_rule = str(case.get("evaluation_rule") or "").strip()
    if evaluation_rule not in ALLOWED_EVALUATION_RULES:
        raise ValueError(f"{prefix} unknown evaluation_rule: {evaluation_rule!r}")

    abstain_expected = case.get("abstain_expected")
    if abstain_expected is not None and not isinstance(abstain_expected, bool):
        raise ValueError(f"{prefix} abstain_expected must be a boolean: {abstain_expected!r}")

    expected_sources = case.get("expected_sources")
    if not isinstance(expected_sources, list) or not expected_sources:
        raise ValueError(f"{prefix} expected_sources must be a non-empty list")
    for idx, source in enumerate(expected_sources, 1):
        if not isinstance(source, dict):
            raise ValueError(f"{prefix} expected_sources[{idx}] must be an object")
        has_retrievable_matcher = any(
            str(source.get(key) or "").strip()
            for key in ("source", "source_contains", "source_regex")
        )
        source_type = str(source.get("source_type") or "").strip()
        if source_type and source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(
                f"{prefix} expected_sources[{idx}] unknown source_type: {source_type!r}"
            )
        if not has_retrievable_matcher and not source_type:
            raise ValueError(
                f"{prefix} expected_sources[{idx}] needs source/source_contains/"
                "source_regex or a source_type"
            )
        if not has_retrievable_matcher and (
            source_type != "sql_object" or behavior not in {"tool_api", "hybrid"}
        ):
            raise ValueError(
                f"{prefix} expected_sources[{idx}] without source matcher is only "
                "valid for source_type='sql_object' with expected_behavior "
                "'tool_api' or 'hybrid'"
            )

    required_datasources = case.get("required_datasources")
    if not isinstance(required_datasources, list):
        raise ValueError(f"{prefix} required_datasources must be a list")
    for ds_id in required_datasources:
        if not isinstance(ds_id, int):
            raise ValueError(f"{prefix} required_datasources values must be integers")

    if not isinstance(case.get("short_or_numeric_answer"), bool):
        raise ValueError(f"{prefix} short_or_numeric_answer must be boolean")

    variants = case.get("allowed_answer_variants", [])
    if variants is not None and not isinstance(variants, list):
        raise ValueError(f"{prefix} allowed_answer_variants must be a list when present")


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            validate_golden_case(row, line_no=line_no)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no Golden Set cases")
    return rows


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().replace("ß", "ss").split())


def _abstain_is_expected(case: dict[str, Any]) -> bool:
    """Ob Abstain fuer diesen Fall die policy-konforme RAG-Antwort ist.

    Produktentscheidung 2026-07-08 (streng quellengebunden): KIara raet nicht
    aus Modellwissen. Fuer tool_api-Faelle ist die richtige Loesung das
    Tool-Routing — eine RAG-Abstinenz ist dort korrekt, ein Weltwissens-Treffer
    waere ein Policy-Verstoss. `abstain_expected: true` markiert zusaetzlich
    Faelle, deren Antwort bewusst in keiner indexierten Quelle steht.
    """
    if case.get("abstain_expected") is True:
        return True
    return str(case.get("expected_behavior") or "") == "tool_api"


def _answer_status(case: dict[str, Any], answer: str, error: str, *, generated: bool) -> str:
    if error:
        return "error"
    if not generated:
        return "not_generated"

    normalized_answer = _normalize_text(answer)
    abstained = not normalized_answer or any(
        pattern in normalized_answer for pattern in _MISSING_PATTERNS
    )
    if abstained:
        return "correct" if _abstain_is_expected(case) else "missing"

    rule = str(case.get("evaluation_rule") or "contains_any")
    variants = [str(v) for v in case.get("allowed_answer_variants") or [] if str(v).strip()]
    if rule == "source_only":
        return "not_evaluated"
    if not variants:
        return "not_evaluated"
    if rule == "contains_all":
        return (
            "correct"
            if all(_normalize_text(v) in normalized_answer for v in variants)
            else "incorrect"
        )
    if rule == "regex_any":
        return (
            "correct" if any(re.search(v, answer, re.IGNORECASE) for v in variants) else "incorrect"
        )
    return (
        "correct" if any(_normalize_text(v) in normalized_answer for v in variants) else "incorrect"
    )


def _source_matches(spec: dict[str, Any], source: str) -> bool:
    if not source:
        return False
    if spec.get("source") and source == str(spec["source"]):
        return True
    if spec.get("source_contains"):
        return str(spec["source_contains"]).casefold() in source.casefold()
    if spec.get("source_regex"):
        return re.search(str(spec["source_regex"]), source) is not None
    return False


def _source_for_chunk(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return str(chunk.get("source") or metadata.get("source") or "")


def _text_for_chunk(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text") or chunk.get("excerpt") or "")


def _chunk_id_for_chunk(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return str(
        chunk.get("chunk_id")
        or chunk.get("point_id")
        or chunk.get("_doc_id")
        or metadata.get("chunk_id")
        or ""
    )


def _compact_chunk(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    text = str(chunk.get("text") or chunk.get("excerpt") or "")
    return {
        "rank": int(chunk.get("rank") or chunk.get("final_context_rank") or rank),
        "final_context_rank": chunk.get("final_context_rank", rank),
        "chunk_id": _chunk_id_for_chunk(chunk),
        "point_id": str(chunk.get("point_id") or _chunk_id_for_chunk(chunk)),
        "source": _source_for_chunk(chunk),
        "file_type": metadata.get("file_type") or chunk.get("file_type"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_total": metadata.get("chunk_total"),
        "score": chunk.get("rrf_score", chunk.get("rerank_score", chunk.get("score"))),
        "semantic_rank": chunk.get("semantic_rank"),
        "semantic_score": chunk.get("semantic_score"),
        "keyword_rank": chunk.get("keyword_rank"),
        "keyword_score": chunk.get("keyword_score"),
        "colbert_rank": chunk.get("colbert_rank"),
        "colbert_score": chunk.get("colbert_score"),
        "graph_rank": chunk.get("graph_rank"),
        "graph_score": chunk.get("graph_score"),
        "rrf_rank": chunk.get("rrf_rank"),
        "rrf_score": chunk.get("rrf_score"),
        "rrf_contributions": dict(chunk.get("rrf_contributions") or {}),
        "retriever_paths": list(chunk.get("retriever_paths") or []),
        "excerpt": text[:DETAIL_EXCERPT_CHARS],
        "text_chars": len(text),
    }


def _with_final_context_ranks(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, result in enumerate(results, 1):
        item = result.copy()
        item["final_context_rank"] = rank
        ranked.append(item)
    return ranked


def _source_metrics_for_k(
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    top_chunks = chunks[:k]
    required_specs = [
        spec
        for spec in case.get("expected_sources", [])
        if bool(spec.get("required", True))
        and any(spec.get(key) for key in ("source", "source_contains", "source_regex"))
    ]
    optional_specs = [
        spec
        for spec in case.get("expected_sources", [])
        if not bool(spec.get("required", True))
        and any(spec.get(key) for key in ("source", "source_contains", "source_regex"))
    ]
    hits = []
    first_required_source_rank: int | None = None
    first_required_evidence_rank: int | None = None
    required_evidence_ranks: list[int] = []
    for spec_index, spec in enumerate(required_specs + optional_specs):
        first_source_hit = None
        first_evidence_hit = None
        for rank, chunk in enumerate(top_chunks, 1):
            source = _source_for_chunk(chunk)
            if _source_matches(spec, source):
                hit = {
                    "rank": int(chunk.get("rank") or chunk.get("final_context_rank") or rank),
                    "source": source,
                    "chunk_id": _chunk_id_for_chunk(chunk),
                }
                if first_source_hit is None:
                    first_source_hit = hit
                    if bool(spec.get("required", True)):
                        first_required_source_rank = _min_int(
                            first_required_source_rank,
                            int(hit["rank"]),
                        )
                if _chunk_matches_evidence(case, spec, chunk):
                    first_evidence_hit = hit
                    if bool(spec.get("required", True)):
                        evidence_rank = int(hit["rank"])
                        first_required_evidence_rank = _min_int(
                            first_required_evidence_rank,
                            evidence_rank,
                        )
                        required_evidence_ranks.append(evidence_rank)
                    break
                continue
            if first_source_hit is not None and first_evidence_hit is not None:
                break
        hits.append(
            {
                "label": spec.get("label") or spec.get("source_contains") or spec.get("source"),
                "spec_index": spec_index,
                "required": bool(spec.get("required", True)),
                "hit": first_source_hit is not None,
                "source_hit": first_source_hit is not None,
                "evidence_hit": first_evidence_hit is not None,
                "first_hit": first_source_hit,
                "first_source_hit": first_source_hit,
                "first_evidence_hit": first_evidence_hit,
            }
        )

    required_total = len(required_specs)
    required_hit = sum(1 for hit in hits if hit["required"] and hit["hit"])
    required_evidence_hit = sum(1 for hit in hits if hit["required"] and hit["evidence_hit"])
    source_recall = (required_hit / required_total) if required_total else None
    evidence_recall = (required_evidence_hit / required_total) if required_total else None
    source_gold_ctx = (required_hit == required_total) if required_total else None
    gold_ctx = (required_evidence_hit == required_total) if required_total else None
    if required_total:
        mrr = (1.0 / first_required_evidence_rank) if first_required_evidence_rank else 0.0
    else:
        mrr = None
    ndcg = _ndcg_for_required_ranks(required_evidence_ranks, required_total, k)
    return {
        "required_total": required_total,
        "required_hit": required_hit,
        "required_evidence_hit": required_evidence_hit,
        "recall": source_recall,
        "source_recall": source_recall,
        "evidence_recall": evidence_recall,
        "source_gold_ctx": source_gold_ctx,
        "gold_ctx": gold_ctx,
        "first_required_rank": first_required_evidence_rank,
        "first_required_source_rank": first_required_source_rank,
        "first_required_evidence_rank": first_required_evidence_rank,
        "mrr": mrr,
        "ndcg": ndcg,
        "hits": hits,
    }


def _min_int(current: int | None, value: int) -> int:
    return value if current is None else min(current, value)


def _metadata_evidence_allowed(case: dict[str, Any]) -> bool:
    return (
        str(case.get("evaluation_rule") or "") == "source_only"
        or str(case.get("expected_answer_type") or "") == "path_or_filename"
    )


def _chunk_matches_evidence(
    case: dict[str, Any], spec: dict[str, Any], chunk: dict[str, Any]
) -> bool:
    text = _text_for_chunk(chunk)
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    if spec.get("chunk_index") is not None:
        try:
            if int(metadata.get("chunk_index")) != int(spec["chunk_index"]):
                return False
        except (TypeError, ValueError):
            return False
    if spec.get("text_contains"):
        return _normalize_text(str(spec["text_contains"])) in _normalize_text(text)
    if spec.get("text_regex"):
        return re.search(str(spec["text_regex"]), text, re.IGNORECASE) is not None

    rule = str(case.get("evaluation_rule") or "contains_any")
    if rule == "source_only":
        return True

    variants = [str(v) for v in case.get("allowed_answer_variants") or [] if str(v).strip()]
    if not variants:
        return False

    evidence_text = text
    if _metadata_evidence_allowed(case):
        metadata_parts = [
            _source_for_chunk(chunk),
            str(metadata.get("directory") or ""),
            str(metadata.get("title") or ""),
            str(metadata.get("filename") or ""),
        ]
        evidence_text = "\n".join([text, *metadata_parts])

    normalized = _normalize_text(evidence_text)
    if rule == "contains_all":
        return all(_normalize_text(v) in normalized for v in variants)
    if rule == "regex_any":
        return any(re.search(v, evidence_text, re.IGNORECASE) for v in variants)
    return any(_normalize_text(v) in normalized for v in variants)


def _ndcg_for_required_ranks(
    ranks: list[int],
    required_total: int,
    k: int,
) -> float | None:
    if required_total == 0:
        return None
    if not ranks:
        return 0.0
    unique_ranks = sorted(set(rank for rank in ranks if 1 <= rank <= k))
    dcg = sum(1.0 / math.log2(rank + 1) for rank in unique_ranks)
    ideal_hits = min(required_total, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else None


def _metrics_for_case(
    case: dict[str, Any],
    chunks: list[dict[str, Any]],
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    source_hits_by_k = {}
    for k in k_values:
        k_metrics = _source_metrics_for_k(case, chunks, k)
        metrics[f"recall@{k}"] = k_metrics["recall"]
        metrics[f"source_recall@{k}"] = k_metrics["source_recall"]
        metrics[f"evidence_recall@{k}"] = k_metrics["evidence_recall"]
        metrics[f"source_gold_ctx@{k}"] = (
            None
            if k_metrics["source_gold_ctx"] is None
            else 1.0
            if k_metrics["source_gold_ctx"]
            else 0.0
        )
        metrics[f"gold_ctx@{k}"] = (
            None if k_metrics["gold_ctx"] is None else 1.0 if k_metrics["gold_ctx"] else 0.0
        )
        metrics[f"mrr@{k}"] = k_metrics["mrr"]
        metrics[f"ndcg@{k}"] = k_metrics["ndcg"]
        metrics[f"first_required_rank@{k}"] = k_metrics["first_required_rank"]
        metrics[f"first_required_source_rank@{k}"] = k_metrics["first_required_source_rank"]
        metrics[f"first_required_evidence_rank@{k}"] = k_metrics["first_required_evidence_rank"]
        source_hits_by_k[str(k)] = k_metrics
    metrics["source_hits_by_k"] = source_hits_by_k
    return metrics


def _diagnostic_variants(diagnostics: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(diagnostics, dict):
        return []
    variants = diagnostics.get("variants")
    if isinstance(variants, list):
        return [dict(v) for v in variants if isinstance(v, dict)]
    return [dict(diagnostics)]


def _path_states(diagnostics: dict[str, Any] | None) -> dict[str, str]:
    states_by_path: dict[str, set[str]] = {}
    for variant in _diagnostic_variants(diagnostics):
        paths = variant.get("paths") if isinstance(variant.get("paths"), dict) else {}
        for path, payload in paths.items():
            if isinstance(payload, dict):
                states_by_path.setdefault(str(path), set()).add(
                    str(payload.get("status") or "unknown")
                )
        rrf = variant.get("rrf")
        if isinstance(rrf, dict):
            states_by_path.setdefault("rrf", set()).add(str(rrf.get("status") or "unknown"))
    states = {}
    for path, values in states_by_path.items():
        states[path] = next(iter(values)) if len(values) == 1 else "mixed"
    return states


def _path_latencies_ms(diagnostics: dict[str, Any] | None) -> dict[str, int]:
    latencies: dict[str, int] = {}
    for variant in _diagnostic_variants(diagnostics):
        paths = variant.get("paths") if isinstance(variant.get("paths"), dict) else {}
        for path, payload in paths.items():
            if isinstance(payload, dict) and isinstance(payload.get("latency_ms"), int | float):
                latencies.setdefault(str(path), int(payload["latency_ms"]))
        rrf = variant.get("rrf")
        if isinstance(rrf, dict) and isinstance(rrf.get("latency_ms"), int | float):
            latencies.setdefault("rrf", int(rrf["latency_ms"]))
    return latencies


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


def _runtime_snapshot() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "optional_dependencies": _optional_dependency_snapshot(),
    }


def _retrieval_settings_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    retrieval = config.get("retrieval", {}) if isinstance(config.get("retrieval"), dict) else {}
    colbert = retrieval.get("colbert", {}) if isinstance(retrieval.get("colbert"), dict) else {}
    return {
        "hybrid_search": retrieval.get("hybrid_search"),
        "rrf_k": retrieval.get("rrf_k"),
        "reranking": retrieval.get("reranking", retrieval.get("rerank_enabled")),
        "rerank_model": retrieval.get("rerank_model"),
        "max_context_chunks": retrieval.get("max_context_chunks"),
        "max_sibling_chunks": retrieval.get("max_sibling_chunks"),
        "colbert": {
            "enabled": colbert.get("enabled"),
            "model": colbert.get("model"),
        },
        "elasticsearch_url_configured": bool(retrieval.get("elasticsearch_url")),
        "elasticsearch_index_prefix": retrieval.get("elasticsearch_index"),
    }


def _embedding_model_from_config(config: dict[str, Any]) -> str:
    return str(
        config.get("embedding", {}).get("model")
        or config.get("ollama", {}).get("embedding_model")
        or ""
    )


def _build_chain(config: dict[str, Any], backend_id: int | None, embedding_model: str | None):
    from kiara.generation.backends.factory import create_backend
    from kiara.generation.chain import RAGChain

    return RAGChain(
        config,
        backend=create_backend(config, backend_id=backend_id),
        embedding_model=embedding_model,
    )


def _build_retriever(config: dict[str, Any], embedding_model: str | None):
    from kiara.retrieval.retriever import HybridRetriever

    return HybridRetriever(config, embedding_model=embedding_model)


def _run_rag_case(
    chain: Any,
    case: dict[str, Any],
    *,
    retrieve_k: int,
    filter_datasource_ids: list[int] | None,
    llm_timeout: float | None,
    think: bool | None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None, str]:
    question = str(case["question"])

    if llm_timeout is not None:
        import signal
        import threading

        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.getsignal(signal.SIGALRM)

            def _raise_timeout(_signum, _frame):
                raise TimeoutError(f"Golden Set case exceeded {llm_timeout:g}s")

            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, llm_timeout)
        else:
            previous_handler = None
    else:
        previous_handler = None

    try:
        result = chain.query(
            question=question,
            k=retrieve_k,
            filter_datasource_ids=filter_datasource_ids,
            think=think,
        )
        answer = str(result.get("answer") or "")
        chunks = [
            _compact_chunk(chunk, rank)
            for rank, chunk in enumerate(result.get("context_chunks") or [], 1)
        ]
        diagnostics = result.get("retrieval_diagnostics") or getattr(
            chain,
            "last_retrieval_diagnostics",
            None,
        )
        return answer, chunks, diagnostics, ""
    except Exception as exc:
        from kiara.retrieval.index_compatibility import IndexCompatibilityError

        if isinstance(exc, IndexCompatibilityError):
            raise
        diagnostics = getattr(chain, "last_retrieval_diagnostics", None)
        return "", [], diagnostics, f"{exc.__class__.__name__}: {exc}"
    finally:
        if llm_timeout is not None and previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


def _run_retrieval_case(
    retriever: Any,
    case: dict[str, Any],
    *,
    retrieve_k: int,
    filter_datasource_ids: list[int] | None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None, str]:
    try:
        results = retriever.retrieve(
            query=str(case["question"]),
            k=retrieve_k,
            filter_datasource_ids=filter_datasource_ids,
        )
        ranked = _with_final_context_ranks(results)
        chunks = [_compact_chunk(chunk, rank) for rank, chunk in enumerate(ranked, 1)]
        getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
        diagnostics = getter() if callable(getter) else None
        return "", chunks, diagnostics, ""
    except Exception as exc:
        from kiara.retrieval.index_compatibility import IndexCompatibilityError

        if isinstance(exc, IndexCompatibilityError):
            raise
        getter = getattr(retriever, "get_last_retrieval_diagnostics", None)
        diagnostics = getter() if callable(getter) else None
        return "", [], diagnostics, f"{exc.__class__.__name__}: {exc}"


def _datasource_filter_for_case(
    case: dict[str, Any],
    *,
    filter_required_datasources: bool,
) -> list[int] | None:
    if not filter_required_datasources:
        return None
    values = [int(value) for value in case.get("required_datasources") or []]
    return values or None


def _average_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["metrics"].get(key) for row in rows if row["metrics"].get(key) is not None]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _overall_for_rows(rows: list[dict[str, Any]], k_values: tuple[int, ...]) -> dict[str, Any]:
    overall = {
        "cases_evaluated": len(rows),
        "answer_status_counts": dict(Counter(row["answer_status"] for row in rows)),
    }
    for k in k_values:
        for prefix in (
            "recall",
            "source_recall",
            "evidence_recall",
            "source_gold_ctx",
            "gold_ctx",
            "mrr",
            "ndcg",
        ):
            overall[f"{prefix}@{k}"] = _average_metric(rows, f"{prefix}@{k}")
    return overall


def _overall_by_field(
    rows: list[dict[str, Any]],
    field: str,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    values = sorted({str(row.get(field) or "") for row in rows})
    return {
        value: _overall_for_rows(
            [row for row in rows if str(row.get(field) or "") == value], k_values
        )
        for value in values
        if value
    }


def _count_path_states(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = {}
    for row in rows:
        for path, state in (row.get("retriever_path_states") or {}).items():
            counts.setdefault(path, Counter())[state] += 1
    return {path: dict(counter) for path, counter in sorted(counts.items())}


def _partial_state_warnings(
    *,
    config: dict[str, Any],
    path_state_counts: dict[str, dict[str, int]],
    runtime: dict[str, Any],
) -> list[str]:
    warnings = []
    retrieval = config.get("retrieval", {}) if isinstance(config.get("retrieval"), dict) else {}
    optional = runtime.get("optional_dependencies") if isinstance(runtime, dict) else {}
    es_snapshot = optional.get("elasticsearch") if isinstance(optional, dict) else {}
    if retrieval.get("hybrid_search") == "elasticsearch" and not es_snapshot.get("installed"):
        warnings.append(
            "keyword_elasticsearch_partial_state: retrieval.hybrid_search=elasticsearch "
            "but Python package 'elasticsearch' is missing; keyword path must not be "
            "interpreted as a successful ES ablation."
        )
    keyword_counts = path_state_counts.get("keyword") or {}
    if keyword_counts.get("empty") and retrieval.get("hybrid_search") == "elasticsearch":
        warnings.append("keyword_path_empty")
    if keyword_counts.get("error"):
        warnings.append("keyword_path_error")
    colbert_counts = path_state_counts.get("colbert") or {}
    if colbert_counts.get("empty"):
        warnings.append("colbert_path_empty")
    if colbert_counts.get("error"):
        warnings.append("colbert_path_error")
    if (path_state_counts.get("graph") or {}).get("disabled"):
        warnings.append("graph_path_disabled")
    return warnings


def evaluate_golden_set(
    golden_set_path: Path = DEFAULT_GOLDEN_SET,
    *,
    output_dir: Path | None = None,
    config_path: str | None = None,
    backend_id: int | None = None,
    embedding_model: str | None = None,
    mode: str = "rag",
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    retrieve_k: int | None = None,
    max_cases: int | None = None,
    run_label: str | None = None,
    filter_required_datasources: bool = False,
    llm_timeout: float | None = None,
    think: bool | None = None,
    chain: Any | None = None,
    retriever: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    overwrite: bool = False,
) -> GoldenEvalStats:
    if mode not in {"rag", "retrieval"}:
        raise ValueError("mode must be 'rag' or 'retrieval'")
    if not k_values:
        raise ValueError("k_values must not be empty")
    if llm_timeout is not None and llm_timeout <= 0:
        raise ValueError("llm_timeout must be positive")

    max_k = max(k_values)
    effective_retrieve_k = retrieve_k or max_k
    if effective_retrieve_k < max_k:
        raise ValueError("retrieve_k must be >= max(k_values)")

    output_root = output_dir or DEFAULT_OUTPUT_DIR
    label = _slug(run_label or f"kiara_golden_{mode}_{_timestamp_for_filename()}")
    summary_path = output_root / f"{label}_summary.json"
    details_path = output_root / f"{label}_details.jsonl"
    _reject_existing_outputs([summary_path, details_path], overwrite=overwrite)

    from kiara.config import load_config

    cases = load_golden_set(golden_set_path)
    cases_total = len(cases)
    if max_cases is not None:
        cases = cases[:max_cases]

    config = load_config(config_path)
    effective_config_path = _effective_config_path(config_path)
    effective_embedding_model = embedding_model or _embedding_model_from_config(config)
    started_at = _utc_timestamp()
    started_monotonic = time.monotonic()

    if mode == "rag" and chain is None:
        chain = _build_chain(config, backend_id, embedding_model)
    if mode == "retrieval" and retriever is None:
        retriever = _build_retriever(config, embedding_model)

    details: list[dict[str, Any]] = []
    latencies = []
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        filter_datasource_ids = _datasource_filter_for_case(
            case,
            filter_required_datasources=filter_required_datasources,
        )
        if mode == "rag":
            answer, chunks, diagnostics, error = _run_rag_case(
                chain,
                case,
                retrieve_k=effective_retrieve_k,
                filter_datasource_ids=filter_datasource_ids,
                llm_timeout=llm_timeout,
                think=think,
            )
            generated = not error
        else:
            answer, chunks, diagnostics, error = _run_retrieval_case(
                retriever,
                case,
                retrieve_k=effective_retrieve_k,
                filter_datasource_ids=filter_datasource_ids,
            )
            generated = False

        latency_ms = int((time.monotonic() - started) * 1000)
        latencies.append(latency_ms)
        metrics = _metrics_for_case(case, chunks, k_values)
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
            "answer": answer,
            "answer_status": _answer_status(case, answer, error, generated=generated),
            "error": error,
            "latency_ms": latency_ms,
            "retrieved_chunks": chunks,
            "metrics": metrics,
            "retriever_path_states": _path_states(diagnostics),
            "retriever_path_latencies_ms": _path_latencies_ms(diagnostics),
            "retrieval_diagnostics": diagnostics,
        }
        details.append(row)
        if progress_callback is not None:
            progress_callback(
                {
                    "current": index,
                    "total": len(cases),
                    "id": case["id"],
                    "answer_status": row["answer_status"],
                    "latency_ms": latency_ms,
                    "error": error,
                }
            )

    finished_at = _utc_timestamp()
    path_state_counts = _count_path_states(details)
    runtime = _runtime_snapshot()
    overall = {
        "cases_total": cases_total,
        "cases_evaluated": len(cases),
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "answer_status_counts": dict(Counter(row["answer_status"] for row in details)),
        "expected_failure_class_counts": dict(
            Counter(str(row["expected_failure_class"]) for row in details)
        ),
        "expected_behavior_counts": dict(Counter(str(row["expected_behavior"]) for row in details)),
        "short_or_numeric_answer_count": sum(
            1 for row in details if row["short_or_numeric_answer"]
        ),
    }
    for k in k_values:
        for prefix in (
            "recall",
            "source_recall",
            "evidence_recall",
            "source_gold_ctx",
            "gold_ctx",
            "mrr",
            "ndcg",
        ):
            overall[f"{prefix}@{k}"] = _average_metric(details, f"{prefix}@{k}")
    overall["overall_by_behavior"] = _overall_by_field(details, "expected_behavior", k_values)
    overall["overall_by_failure_class"] = _overall_by_field(
        details,
        "expected_failure_class",
        k_values,
    )
    reproducibility = {
        "golden_set_sha256": _sha256_file(golden_set_path),
        "config_sha256": _sha256_json(config),
        "config_file_sha256": _sha256_file(effective_config_path),
        "config_effective_path": str(effective_config_path),
        "git": _git_snapshot(),
    }

    summary = {
        "schema": "kiara-golden-set-summary",
        "schema_version": 1,
        "created_at": finished_at,
        "run": {
            "label": label,
            "mode": mode,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": int((time.monotonic() - started_monotonic) * 1000),
            "k_values": list(k_values),
            "retrieve_k": effective_retrieve_k,
            "backend_id": backend_id,
            "embedding_model_requested": embedding_model,
            "embedding_model_effective": effective_embedding_model,
            "filter_required_datasources": filter_required_datasources,
            "llm_timeout": llm_timeout,
            "think": think,
            "overwrite": overwrite,
        },
        "settings": {
            "golden_set_path": str(golden_set_path),
            "output_dir": str(output_root),
            "config_path": config_path,
            "config_effective_path": str(effective_config_path),
            "retrieval_config": _retrieval_settings_snapshot(config),
            "environment": runtime,
            "diagnostic_fields": {
                "schema": "kiara-retrieval-path-diagnostics",
                "paths": ["semantic", "keyword", "colbert", "graph"],
                "metrics": {
                    "recall@k": "required source recall",
                    "source_recall@k": "required source recall",
                    "source_gold_ctx@k": "all required sources reached top-k",
                    "evidence_recall@k": "required source plus answer evidence recall",
                    "gold_ctx@k": "all required evidence-bearing chunks reached top-k",
                    "mrr@k": "reciprocal rank of first required evidence hit",
                    "ndcg@k": "nDCG over required evidence hits",
                },
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
        },
        "overall": overall,
        "path_state_counts": path_state_counts,
        "partial_state_warnings": _partial_state_warnings(
            config=config,
            path_state_counts=path_state_counts,
            runtime=runtime,
        ),
        "artifacts": {
            "summary": str(summary_path),
            "details": str(details_path),
            "golden_set": str(golden_set_path),
        },
        "reproducibility": reproducibility,
    }

    write_jsonl(details_path, details)
    summary["artifacts"]["details_sha256"] = _sha256_file(details_path)
    summary["artifacts"]["golden_set_sha256"] = reproducibility["golden_set_sha256"]
    summary["artifacts"]["summary_payload_sha256"] = _sha256_json(summary)
    summary["artifacts"]["summary_payload_sha256_scope"] = (
        "canonical summary payload before adding summary file hash; "
        "self-referential final file hash is intentionally not embedded"
    )
    write_private_json(summary_path, summary, indent=2, sort_keys=True)

    return GoldenEvalStats(
        run_label=label,
        mode=mode,
        golden_set_path=str(golden_set_path),
        summary_path=str(summary_path),
        details_path=str(details_path),
        cases_evaluated=len(cases),
        max_k=max_k,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-path")
    parser.add_argument("--backend-id", type=int)
    parser.add_argument("--embedding-model")
    parser.add_argument("--mode", choices=["rag", "retrieval"], default="rag")
    parser.add_argument("--k-values", default="1,3,5,10,20")
    parser.add_argument("--retrieve-k", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--run-label")
    parser.add_argument("--filter-required-datasources", action="store_true")
    parser.add_argument("--llm-timeout", type=float)
    parser.add_argument("--think", dest="think", action="store_true", default=None)
    parser.add_argument("--no-think", dest="think", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = evaluate_golden_set(
        golden_set_path=args.golden_set,
        output_dir=args.output_dir,
        config_path=args.config_path,
        backend_id=args.backend_id,
        embedding_model=args.embedding_model,
        mode=args.mode,
        k_values=parse_k_values(args.k_values),
        retrieve_k=args.retrieve_k,
        max_cases=args.max_cases,
        run_label=args.run_label,
        filter_required_datasources=args.filter_required_datasources,
        llm_timeout=args.llm_timeout,
        think=args.think,
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
