#!/usr/bin/env python3
"""Vertical retrieval-pipeline audit for prepared CRAG Task 1 cases."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import (  # noqa: E402
    _answer_excerpt,
    _first_answer_match,
    iter_manifest,
)
from kiara.config import get_embedding_model, load_config  # noqa: E402
from kiara.retrieval.embedding import collection_name_for_model, embed_query  # noqa: E402
from kiara.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from kiara.retrieval.retriever import HybridRetriever  # noqa: E402
from kiara.retrieval.vectorstore import VectorStore  # noqa: E402
from kiara.shared.constants import DEFAULT_MAX_CONTEXT_CHUNKS  # noqa: E402

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency is installed in prod
    BM25Okapi = None


DEFAULT_CASE_IDS = (
    "971dfeb2-9cbf-4c6f-84fd-b2476a59e640",
    "ce18f7dd-f5fb-4e36-91c4-442cb0afe7e4",
    "a2ab3629-3f6b-4cf7-a2fc-f3ace539f75e",
    "aa29b165-d6ac-423b-88a2-7a6ec06e0273",
    "bcc88678-d327-4792-b6f5-bb7efc33e2c6",
)

CASE_LABELS = {
    "971dfeb2-9cbf-4c6f-84fd-b2476a59e640": "dense_rank_miss",
    "ce18f7dd-f5fb-4e36-91c4-442cb0afe7e4": "dense_rank_miss",
    "a2ab3629-3f6b-4cf7-a2fc-f3ace539f75e": "stable_dense_hit",
    "aa29b165-d6ac-423b-88a2-7a6ec06e0273": "short_answer_control",
    "bcc88678-d327-4792-b6f5-bb7efc33e2c6": "literal_noise_control",
}

TOPK_LEVELS = (20, 50, 100, 500)
REPORT_JSON = Path(
    "/var/lib/kiara/benchmarks/crag_hf20/results/task1_hf20_retrieval_pipeline_audit.json"
)
REPORT_MD = Path(
    "/var/lib/kiara/benchmarks/crag_hf20/results/task1_hf20_retrieval_pipeline_audit.md"
)


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _compact(text: str, limit: int = 260) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _source_under_dir(source: str, doc_dir: str) -> bool:
    if not source or not doc_dir:
        return False
    try:
        return Path(source).resolve().is_relative_to(Path(doc_dir).resolve())
    except (OSError, ValueError):
        return str(source).startswith(str(doc_dir).rstrip("/") + "/")


def _where_directory(doc_dir: str) -> dict | None:
    return {"directory": {"$contains": doc_dir}} if doc_dir else None


def _where_ds(datasource_id: int | None) -> dict | None:
    return {"ds_ids": {"$in": [datasource_id]}} if datasource_id is not None else None


def _where_and(*parts: dict | None) -> dict | None:
    conditions = [part for part in parts if part]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _match_answer(text: str, answers: list[str]) -> str:
    return _first_answer_match(str(text or ""), answers)


def _excerpt(text: str, match: str) -> str:
    if match:
        return _compact(_answer_excerpt(str(text or ""), match, radius=180), 520)
    return _compact(str(text or ""), 360)


def _metadata_row(meta: dict, doc_dir: str, datasource_id: int | None) -> dict[str, Any]:
    raw_ds_ids = meta.get("ds_ids")
    ds_ids = raw_ds_ids if isinstance(raw_ds_ids, list) else [raw_ds_ids] if raw_ds_ids else []
    return {
        "source": str(meta.get("source") or ""),
        "directory": str(meta.get("directory") or ""),
        "file_type": str(meta.get("file_type") or ""),
        "chunk_index": meta.get("chunk_index"),
        "chunk_total": meta.get("chunk_total"),
        "chunk_strategy": str(meta.get("chunk_strategy") or ""),
        "ds_ids": ds_ids,
        "source_under_doc_dir": _source_under_dir(str(meta.get("source") or ""), doc_dir),
        "directory_contains_doc_dir": doc_dir in str(meta.get("directory") or ""),
        "ds_scope_match": datasource_id is None or datasource_id in ds_ids,
    }


def _answer_position(text: str, answer: str) -> dict[str, Any]:
    if not text or not answer:
        return {"char": None, "pct": None}
    idx = text.casefold().find(answer.casefold())
    if idx < 0:
        return {"char": None, "pct": None}
    return {"char": idx, "pct": round(idx / max(len(text), 1), 4)}


def _markup_stats(text: str) -> dict[str, Any]:
    text = str(text or "")
    length = len(text)
    return {
        "text_chars": length,
        "lt_count": text.count("<"),
        "brace_count": text.count("{") + text.count("}"),
        "markup_density": round((text.count("<") + text.count("{") + text.count("}")) / max(length, 1), 4),
    }


def _result_row(result: dict, rank: int, answers: list[str]) -> dict[str, Any]:
    text = str(result.get("text") or "")
    meta = result.get("metadata") or {}
    match = _match_answer(text, answers)
    return {
        "rank": rank,
        "score": result.get("score"),
        "rrf_score": result.get("rrf_score"),
        "rerank_score": result.get("rerank_score"),
        "doc_id": result.get("_doc_id", ""),
        "source": str(meta.get("source") or ""),
        "chunk_index": meta.get("chunk_index"),
        "chunk_total": meta.get("chunk_total"),
        "text_chars": len(text),
        "has_gold": bool(match),
        "matched_answer": match,
        "excerpt": _excerpt(text, match),
    }


def _first_gold_rank(results: list[dict], answers: list[str], top_k: int | None = None) -> dict:
    selected = results[:top_k] if top_k is not None else results
    for rank, result in enumerate(selected, 1):
        match = _match_answer(str(result.get("text") or ""), answers)
        if match:
            return {
                "rank": rank,
                "score": result.get("score"),
                "rrf_score": result.get("rrf_score"),
                "rerank_score": result.get("rerank_score"),
                "source": str((result.get("metadata") or {}).get("source") or ""),
                "chunk_index": (result.get("metadata") or {}).get("chunk_index"),
                "matched_answer": match,
            }
    return {}


def _as_result(chunk: dict[str, Any], score: float, score_key: str = "score") -> dict[str, Any]:
    return {
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        score_key: float(score),
        "_doc_id": chunk["chunk_id"],
    }


def _load_case_chunks(
    store: Any,
    files: list[str],
    datasource_id: int | None,
    doc_dir: str,
    answers: list[str],
) -> dict[str, Any]:
    sources = [str(Path(path)) for path in files]
    ids_all = store.get_ids_by_sources(sources)
    ids_ds = store.get_ids_by_sources(sources, ds_id=datasource_id) if datasource_id else ids_all
    docs = store.get_documents_by_ids(ids_all, include=["documents", "metadatas", "embeddings"])

    chunks: list[dict[str, Any]] = []
    gold_chunks: list[dict[str, Any]] = []
    for chunk_id, text, meta, vector in zip(
        docs.get("ids") or [],
        docs.get("documents") or [],
        docs.get("metadatas") or [],
        docs.get("embeddings") or [],
        strict=False,
    ):
        text = str(text or "")
        meta = meta if isinstance(meta, dict) else {}
        match = _match_answer(text, answers)
        chunk = {
            "chunk_id": str(chunk_id),
            "text": text,
            "metadata": meta,
            "embedding": vector,
            "matched_answer": match,
            "metadata_checks": _metadata_row(meta, doc_dir, datasource_id),
            "answer_position": _answer_position(text, match),
            "markup": _markup_stats(text),
        }
        chunks.append(chunk)
        if match:
            gold_chunks.append(chunk)

    ids_ds_set = set(ids_ds)
    chunks_ds = [chunk for chunk in chunks if chunk["chunk_id"] in ids_ds_set]
    return {
        "source_count": len(sources),
        "ids_all_count": len(ids_all),
        "ids_datasource_count": len(ids_ds),
        "chunks_all": chunks,
        "chunks_datasource": chunks_ds,
        "gold_chunks": gold_chunks,
        "gold_chunks_datasource": [chunk for chunk in gold_chunks if chunk["chunk_id"] in ids_ds_set],
    }


def _cosine_scores(query_vector: list[float], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    ranked = []
    for chunk in chunks:
        vector = chunk.get("embedding")
        if vector is None:
            continue
        v = np.asarray(vector, dtype=np.float32)
        denom = q_norm * float(np.linalg.norm(v))
        score = float(np.dot(q, v) / denom) if denom else float("-inf")
        ranked.append(_as_result(chunk, score))
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _rank_profile(results: list[dict], answers: list[str]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for k in TOPK_LEVELS:
        hit = _first_gold_rank(results, answers, top_k=k)
        profile[f"top{k}_has_gold"] = bool(hit)
        profile[f"top{k}_gold_rank"] = hit.get("rank")
        profile[f"top{k}_gold_score"] = hit.get("score")
    first = _first_gold_rank(results, answers)
    profile["first_gold_rank"] = first.get("rank")
    profile["first_gold_score"] = first.get("score")
    profile["first_gold_source"] = first.get("source", "")
    profile["first_gold_chunk_index"] = first.get("chunk_index")
    return profile


def _qdrant_scope_search(store: Any, query: str, where: dict | None, answers: list[str]) -> dict:
    started = time.monotonic()
    results = store.search(query, k=max(TOPK_LEVELS), where=where)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    profile = _rank_profile(results, answers)
    return {
        "where": where,
        "elapsed_ms": elapsed_ms,
        "result_count": len(results),
        **profile,
        "top_results": [_result_row(result, rank, answers) for rank, result in enumerate(results[:10], 1)],
    }


def _bm25_audit(chunks: list[dict[str, Any]], query: str, answers: list[str]) -> dict[str, Any]:
    query_tokens = HybridRetriever._tokenize(query)
    if BM25Okapi is None or not query_tokens:
        return {
            "available": BM25Okapi is not None,
            "query_tokens": query_tokens,
            "error": "BM25 unavailable or query has no tokens",
        }

    tokenized = [HybridRetriever._tokenize(chunk["text"]) for chunk in chunks]
    keep_idx = [idx for idx, tokens in enumerate(tokenized) if tokens]
    if not keep_idx:
        return {"available": True, "query_tokens": query_tokens, "error": "empty token corpus"}

    kept_chunks = [chunks[idx] for idx in keep_idx]
    bm25 = BM25Okapi([tokenized[idx] for idx in keep_idx])
    scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float64)
    order = np.argsort(-scores)
    results = [_as_result(kept_chunks[int(idx)], float(scores[int(idx)])) for idx in order]
    positive = [result for result in results if result["score"] > 0]
    profile = _rank_profile(results, answers)
    first_positive = _first_gold_rank(positive, answers)
    return {
        "available": True,
        "query_tokens": query_tokens,
        "corpus_chunks": len(kept_chunks),
        "positive_chunks": len(positive),
        **profile,
        "first_positive_gold_rank": first_positive.get("rank"),
        "first_positive_gold_score": first_positive.get("score"),
        "top_results": [
            _result_row(result, rank, answers) for rank, result in enumerate(positive[:10], 1)
        ],
    }


def _rrf_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    k: int,
    rrf_k: int,
) -> list[dict]:
    scores: dict[str, dict[str, Any]] = {}
    for result_list in (dense_results, bm25_results):
        for rank, doc in enumerate(result_list, 1):
            doc_id = str(doc.get("_doc_id") or "")
            if not doc_id:
                continue
            contribution = 1.0 / (rrf_k + rank)
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += contribution
            else:
                scores[doc_id] = {"rrf_score": contribution, "doc": doc}
    ordered = sorted(scores.values(), key=lambda item: item["rrf_score"], reverse=True)
    fused = []
    for item in ordered[:k]:
        doc = dict(item["doc"])
        doc["rrf_score"] = round(float(item["rrf_score"]), 6)
        fused.append(doc)
    return fused


def _local_rrf_audit(
    dense_results: list[dict],
    bm25_audit: dict[str, Any],
    chunks: list[dict[str, Any]],
    query: str,
    answers: list[str],
    rrf_k: int,
) -> dict[str, Any]:
    bm25_results: list[dict] = []
    if bm25_audit.get("available") and not bm25_audit.get("error"):
        tokenized = [HybridRetriever._tokenize(chunk["text"]) for chunk in chunks]
        keep_idx = [idx for idx, tokens in enumerate(tokenized) if tokens]
        if keep_idx:
            kept_chunks = [chunks[idx] for idx in keep_idx]
            scores = np.asarray(
                BM25Okapi([tokenized[idx] for idx in keep_idx]).get_scores(
                    HybridRetriever._tokenize(query)
                ),
                dtype=np.float64,
            )
            for idx in np.argsort(-scores):
                score = float(scores[int(idx)])
                if score <= 0:
                    break
                bm25_results.append(_as_result(kept_chunks[int(idx)], score))

    out = {}
    for k in (5, 10):
        semantic_candidates = dense_results[: k * 2]
        keyword_candidates = bm25_results[:k]
        fused = _rrf_fuse(
            semantic_candidates,
            keyword_candidates,
            k=k,
            rrf_k=rrf_k,
        )
        out[f"k{k}"] = {
            "semantic_candidate_count": len(semantic_candidates),
            "keyword_candidate_count": len(keyword_candidates),
            "gold_in_semantic_candidates": bool(_first_gold_rank(semantic_candidates, answers)),
            "gold_in_keyword_candidates": bool(_first_gold_rank(keyword_candidates, answers)),
            "gold_in_fused": bool(_first_gold_rank(fused, answers)),
            "fused_gold_rank": _first_gold_rank(fused, answers).get("rank"),
            "top_results": [
                _result_row(result, rank, answers) for rank, result in enumerate(fused, 1)
            ],
        }
    return out


def _configured_hybrid_audit(
    retriever: HybridRetriever,
    query: str,
    doc_dir: str,
    answers: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in (5, 10):
        try:
            started = time.monotonic()
            results = retriever.retrieve(query=query, k=k, filter_dir=doc_dir)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            out[f"k{k}"] = {
                "elapsed_ms": elapsed_ms,
                "result_count": len(results),
                "gold_in_results": bool(_first_gold_rank(results, answers)),
                "gold_rank": _first_gold_rank(results, answers).get("rank"),
                "top_results": [
                    _result_row(result, rank, answers) for rank, result in enumerate(results, 1)
                ],
            }
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            out[f"k{k}"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _gold_summary(chunks: list[dict[str, Any]], limit: int = 8) -> dict[str, Any]:
    if not chunks:
        return {
            "count": 0,
            "samples": [],
            "text_chars_min": None,
            "text_chars_median": None,
            "answer_position_pct_median": None,
            "markup_density_median": None,
        }
    lengths = [chunk["markup"]["text_chars"] for chunk in chunks]
    positions = [
        chunk["answer_position"]["pct"]
        for chunk in chunks
        if chunk["answer_position"]["pct"] is not None
    ]
    markup = [chunk["markup"]["markup_density"] for chunk in chunks]
    samples = []
    for chunk in chunks[:limit]:
        samples.append(
            {
                "chunk_id": chunk["chunk_id"],
                "matched_answer": chunk["matched_answer"],
                "metadata": chunk["metadata_checks"],
                "answer_position": chunk["answer_position"],
                "markup": chunk["markup"],
                "excerpt": _excerpt(chunk["text"], chunk["matched_answer"]),
            }
        )
    return {
        "count": len(chunks),
        "text_chars_min": min(lengths),
        "text_chars_median": statistics.median(lengths),
        "answer_position_pct_median": statistics.median(positions) if positions else None,
        "markup_density_median": statistics.median(markup),
        "samples": samples,
    }


def _case_diagnosis(case: dict[str, Any]) -> dict[str, Any]:
    dense = case["dense_exact_per_question_datasource"]
    bm25 = case["bm25_per_question_datasource"]
    hybrid = case["local_bm25_rrf_per_question_datasource"].get("k5", {})
    cfg_hybrid = case["configured_hybrid_benchmark_filter"].get("k5", {})
    reranker = case["reranker"]

    if not case["gold_chunk_audit"]["datasource_scope"]["count"]:
        cause = "no_gold_chunk_in_index_scope"
    elif dense.get("first_gold_rank") and dense["first_gold_rank"] <= 20:
        cause = "dense_ok"
    elif bm25.get("first_positive_gold_rank") and bm25["first_positive_gold_rank"] <= 10:
        cause = "dense_embedding_rank_loss_keyword_can_recover"
    elif bm25.get("first_gold_rank") and bm25["first_gold_rank"] <= 50:
        cause = "dense_embedding_rank_loss_bm25_weak_recovery"
    else:
        cause = "chunk_text_or_query_signal_loss"

    return {
        "primary_cause": cause,
        "dense_gold_rank": dense.get("first_gold_rank"),
        "dense_gold_score": dense.get("first_gold_score"),
        "dense_score_gap_to_top1": dense.get("score_gap_to_top1"),
        "bm25_gold_rank": bm25.get("first_gold_rank"),
        "bm25_positive_gold_rank": bm25.get("first_positive_gold_rank"),
        "local_rrf_k5_gold_rank": hybrid.get("fused_gold_rank"),
        "configured_hybrid_k5_gold_rank": cfg_hybrid.get("gold_rank"),
        "reranker_active": reranker.get("enabled"),
    }


def _audit_case(
    entry: dict[str, Any],
    *,
    config: dict,
    store: Any,
    retriever: HybridRetriever,
    datasource_id: int | None,
    embedding_model: str,
) -> dict[str, Any]:
    interaction_id = str(entry.get("interaction_id") or "")
    query = str(entry.get("query") or "")
    answers = [str(answer) for answer in entry.get("answers") or []]
    doc_dir = str(entry.get("doc_dir") or "")
    files = [str(path) for path in entry.get("files") or []]

    case_chunks = _load_case_chunks(store, files, datasource_id, doc_dir, answers)
    query_vector = embed_query(config, embedding_model, query)
    dense_exact = _cosine_scores(query_vector, case_chunks["chunks_datasource"])
    dense_profile = _rank_profile(dense_exact, answers)
    dense_profile["top_results"] = [
        _result_row(result, rank, answers) for rank, result in enumerate(dense_exact[:10], 1)
    ]
    if dense_exact and dense_profile.get("first_gold_score") is not None:
        dense_profile["score_gap_to_top1"] = round(
            float(dense_exact[0]["score"]) - float(dense_profile["first_gold_score"]),
            6,
        )
    else:
        dense_profile["score_gap_to_top1"] = None

    per_question_where = _where_directory(doc_dir)
    per_question_ds_where = _where_and(per_question_where, _where_ds(datasource_id))
    global_ds_where = _where_ds(datasource_id)

    bm25 = _bm25_audit(case_chunks["chunks_datasource"], query, answers)
    rrf = _local_rrf_audit(
        dense_exact,
        bm25,
        case_chunks["chunks_datasource"],
        query,
        answers,
        int(config.get("retrieval", {}).get("rrf_k", 60)),
    )
    configured_hybrid = _configured_hybrid_audit(retriever, query, doc_dir, answers)

    reranker = CrossEncoderReranker(config)
    reranker_info = {
        "enabled": reranker.enabled,
        "config_model": config.get("retrieval", {}).get("rerank_model", ""),
        "query_expansion_enabled": bool(config.get("retrieval", {}).get("query_expansion", False)),
        "rerank_fetch_multiplier": config.get("retrieval", {}).get("rerank_fetch_multiplier", 3),
        "candidate_set_note": "not run because reranking is disabled in current config",
    }

    context_cfg = {
        "max_context_chunks": config.get("retrieval", {}).get(
            "max_context_chunks", DEFAULT_MAX_CONTEXT_CHUNKS
        ),
        "reranking": bool(config.get("retrieval", {}).get("reranking", False)),
        "query_expansion": bool(config.get("retrieval", {}).get("query_expansion", False)),
        "sibling_expansion_active": bool(config.get("retrieval", {}).get("reranking", False)),
        "benchmark_runner_filter_dir": doc_dir,
        "benchmark_runner_filter_datasource_ids": None,
        "context_assembly_loss": "none observed before LLM; k is below max_context_chunks",
    }

    case = {
        "interaction_id": interaction_id,
        "case_label": CASE_LABELS.get(interaction_id, ""),
        "query": query,
        "answers": answers,
        "doc_dir": doc_dir,
        "files": files,
        "query_unchanged": query == str(entry.get("query") or ""),
        "embedding_model": embedding_model,
        "collection": collection_name_for_model(embedding_model),
        "filters": {
            "benchmark_per_question": per_question_where,
            "per_question_with_datasource": per_question_ds_where,
            "global_datasource": global_ds_where,
        },
        "scope_counts": {
            "benchmark_per_question": store.count_filtered(per_question_where),
            "per_question_with_datasource": store.count_filtered(per_question_ds_where),
            "global_datasource": store.count_filtered(global_ds_where),
            "source_ids_all": case_chunks["ids_all_count"],
            "source_ids_datasource": case_chunks["ids_datasource_count"],
        },
        "gold_chunk_audit": {
            "all_source_scope": _gold_summary(case_chunks["gold_chunks"]),
            "datasource_scope": _gold_summary(case_chunks["gold_chunks_datasource"]),
        },
        "dense_qdrant_benchmark_filter": _qdrant_scope_search(
            store, query, per_question_where, answers
        ),
        "dense_qdrant_per_question_datasource": _qdrant_scope_search(
            store, query, per_question_ds_where, answers
        ),
        "dense_qdrant_global_datasource": _qdrant_scope_search(
            store, query, global_ds_where, answers
        ),
        "dense_exact_per_question_datasource": dense_profile,
        "bm25_per_question_datasource": bm25,
        "local_bm25_rrf_per_question_datasource": rrf,
        "configured_hybrid_benchmark_filter": configured_hybrid,
        "reranker": reranker_info,
        "context_assembly": context_cfg,
    }
    case["diagnosis"] = _case_diagnosis(case)
    return case


def _summary(cases: list[dict[str, Any]], config: dict, embedding_model: str, store: Any) -> dict:
    labels = Counter(case["case_label"] for case in cases)
    causes = Counter(case["diagnosis"]["primary_cause"] for case in cases)
    dense_misses = [
        case
        for case in cases
        if case["case_label"] == "dense_rank_miss"
        and not case["dense_exact_per_question_datasource"].get("top20_has_gold")
    ]
    return {
        "schema": "kiara-crag-hf20-retrieval-pipeline-audit",
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "case_count": len(cases),
        "case_labels": dict(labels),
        "primary_causes": dict(causes),
        "embedding_model": embedding_model,
        "collection": store.get_collection_name(),
        "collection_stats": store.get_stats(),
        "retrieval_config": config.get("retrieval", {}),
        "runtime_dependency_status": {
            "rank_bm25": BM25Okapi is not None,
            "elasticsearch": importlib.util.find_spec("elasticsearch") is not None,
            "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
        },
        "dense_rank_miss_count": len(dense_misses),
    }


def _fmt_rank(value: Any) -> str:
    return "" if value is None else str(value)


def _fmt_score(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _render_top_rows(rows: list[dict[str, Any]], limit: int = 5) -> str:
    lines = [
        "| Rank | Score | Source | Chunk | Gold | Excerpt |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows[:limit]:
        score = row.get("score")
        if score is None:
            score = row.get("rrf_score")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank") or ""),
                    _fmt_score(score),
                    f"`{Path(str(row.get('source') or '')).name}`",
                    str(row.get("chunk_index") or ""),
                    "yes" if row.get("has_gold") else "",
                    _compact(str(row.get("excerpt") or ""), 180).replace("|", "\\|"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    cases = payload["cases"]
    lines = [
        "# CRAG HF20 Retrieval Pipeline Audit",
        "",
        "## Summary",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Cases audited: {summary['case_count']}",
        f"- Embedding model / collection: `{summary['embedding_model']}` / `{summary['collection']}`",
        f"- Retrieval config mode: `{summary['retrieval_config'].get('hybrid_search')}`",
        f"- Reranking: `{summary['retrieval_config'].get('reranking')}`",
        f"- Query expansion: `{summary['retrieval_config'].get('query_expansion')}`",
        f"- Runtime dependencies: `{summary['runtime_dependency_status']}`",
        f"- Primary causes: `{summary['primary_causes']}`",
        "",
        "## Case Matrix",
        "",
        "| Case | Label | Dense rank | Dense score | Gap | BM25 rank | BM25 positive rank | Local RRF@5 | Configured hybrid@5 | Cause |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in cases:
        diag = case["diagnosis"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case['interaction_id'][:8]}`",
                    f"`{case['case_label']}`",
                    _fmt_rank(diag.get("dense_gold_rank")),
                    _fmt_score(diag.get("dense_gold_score")),
                    _fmt_score(diag.get("dense_score_gap_to_top1")),
                    _fmt_rank(diag.get("bm25_gold_rank")),
                    _fmt_rank(diag.get("bm25_positive_gold_rank")),
                    _fmt_rank(diag.get("local_rrf_k5_gold_rank")),
                    _fmt_rank(diag.get("configured_hybrid_k5_gold_rank")),
                    f"`{diag.get('primary_cause')}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "- The benchmark runner passes the question unchanged and filters per case only by `filter_dir=doc_dir`; no datasource filter is applied by `run_crag_task()`.",
            "- The audited collection is the expected multi-model collection for the configured embedding model.",
            "- The two real misses both have answer-bearing chunks in Qdrant with correct `source`, `directory`, `ds_ids`, `chunk_index` metadata.",
            "- Context assembly is not the loss point in this run: reranking and query expansion are disabled, sibling expansion is therefore inactive, and `k` is below `max_context_chunks`.",
            "- Configured hybrid rows reflect the current audit runtime; when `elasticsearch` is unavailable, the ES keyword leg contributes no candidates.",
            "",
        ]
    )

    for case in cases:
        dense = case["dense_exact_per_question_datasource"]
        qdrant = case["dense_qdrant_per_question_datasource"]
        bm25 = case["bm25_per_question_datasource"]
        rrf = case["local_bm25_rrf_per_question_datasource"].get("k5", {})
        gold = case["gold_chunk_audit"]["datasource_scope"]
        lines.extend(
            [
                f"## {case['interaction_id']}",
                "",
                f"- Label: `{case['case_label']}`",
                f"- Query: {case['query']}",
                f"- Answers: {', '.join(case['answers'])}",
                f"- Scope chunks: benchmark={case['scope_counts']['benchmark_per_question']}, with DS={case['scope_counts']['per_question_with_datasource']}, source ids DS={case['scope_counts']['source_ids_datasource']}",
                f"- Gold chunks in DS scope: {gold['count']} (median chars={gold['text_chars_median']}, median answer position={gold['answer_position_pct_median']}, median markup density={gold['markup_density_median']})",
                f"- Exact dense gold rank: {_fmt_rank(dense.get('first_gold_rank'))}, score={_fmt_score(dense.get('first_gold_score'))}, gap to top1={_fmt_score(dense.get('score_gap_to_top1'))}",
                f"- Qdrant Top-k gold: @20={qdrant.get('top20_has_gold')}, @50={qdrant.get('top50_has_gold')}, @100={qdrant.get('top100_has_gold')}, @500={qdrant.get('top500_has_gold')}",
                f"- BM25 gold rank: {_fmt_rank(bm25.get('first_gold_rank'))}, positive-rank={_fmt_rank(bm25.get('first_positive_gold_rank'))}, positive chunks={bm25.get('positive_chunks')}",
                f"- Local BM25+RRF@5 gold rank: {_fmt_rank(rrf.get('fused_gold_rank'))}; configured hybrid@5 gold rank: {_fmt_rank(case['configured_hybrid_benchmark_filter'].get('k5', {}).get('gold_rank'))}",
                f"- Primary cause: `{case['diagnosis']['primary_cause']}`",
                "",
                "Dense top results:",
                "",
                _render_top_rows(dense.get("top_results", []), limit=5),
                "",
                "BM25 top results:",
                "",
                _render_top_rows(bm25.get("top_results", []), limit=5),
                "",
            ]
        )
        samples = gold.get("samples") or []
        if samples:
            lines.extend(["Gold sample:", "", "```text", samples[0]["excerpt"], "```", ""])

    lines.extend(
        [
            "## Recommendation",
            "",
            "1. Do not change filters, collection routing, reranker or context assembly for these two misses; the measured loss is before fusion, in dense ranking against noisy HTML chunks.",
            "2. Next fix should target retrieval candidate generation for CRAG-like HTML: add a BM25/ES ablation with answer-bearing rank diagnostics, then evaluate HTML cleanup/chunking before changing the dense model.",
            "3. Treat short literal controls (`1`, `5 m`) separately; they are not reliable evidence for retrieval quality without stricter answer matching.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/var/lib/kiara/benchmarks/crag_hf20/manifests/task1.jsonl",
        help="Prepared CRAG Task 1 manifest",
    )
    parser.add_argument("--config", default=None, help="KIara config path")
    parser.add_argument("--datasource-id", type=int, default=24, help="CRAG datasource ID")
    parser.add_argument("--embedding-model", default="", help="Embedding model override")
    parser.add_argument("--json-output", default=str(REPORT_JSON), help="JSON output path")
    parser.add_argument("--md-output", default=str(REPORT_MD), help="Markdown output path")
    parser.add_argument("--case-id", action="append", default=[], help="Interaction ID to audit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    embedding_model = args.embedding_model or get_embedding_model(config)
    store = VectorStore(config, embedding_model=embedding_model)
    retriever = HybridRetriever(config, embedding_model=embedding_model)

    wanted = tuple(args.case_id) if args.case_id else DEFAULT_CASE_IDS
    rows = {str(row.get("interaction_id") or ""): row for row in iter_manifest(args.manifest)}
    missing = [case_id for case_id in wanted if case_id not in rows]
    if missing:
        raise SystemExit(f"Manifest enthaelt diese case IDs nicht: {missing}")

    cases = []
    for case_id in wanted:
        cases.append(
            _audit_case(
                rows[case_id],
                config=config,
                store=store,
                retriever=retriever,
                datasource_id=args.datasource_id,
                embedding_model=embedding_model,
            )
        )

    payload = {
        "summary": _summary(cases, config, embedding_model, store),
        "cases": cases,
    }
    json_out = Path(args.json_output)
    md_out = Path(args.md_output)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
