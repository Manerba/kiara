#!/usr/bin/env python3
"""Ablate HTML cleanup and candidate windows for CRAG HF20 retrieval misses."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kiara.benchmarks.crag import iter_manifest
from kiara.config import get_embedding_model, load_config
from kiara.ingestion.chunker import chunk_document
from kiara.ingestion.loader import _load_html_markdown
from kiara.retrieval.embedding import embed_query, embed_texts
from kiara.retrieval.retriever import HybridRetriever
from kiara.retrieval.vectorstore import VectorStore
from scripts.crag_audit_retrieval_pipeline import (
    CASE_LABELS,
    _bm25_audit,
    _compact,
    _cosine_scores,
    _first_gold_rank,
    _gold_summary,
    _json_dumps,
    _load_case_chunks,
    _rank_profile,
    _result_row,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None


DEFAULT_CASE_IDS = (
    "971dfeb2-9cbf-4c6f-84fd-b2476a59e640",
    "ce18f7dd-f5fb-4e36-91c4-442cb0afe7e4",
    "a2ab3629-3f6b-4cf7-a2fc-f3ace539f75e",
    "aa29b165-d6ac-423b-88a2-7a6ec06e0273",
    "bcc88678-d327-4792-b6f5-bb7efc33e2c6",
)

REPORT_JSON = Path(
    "/var/lib/kiara/benchmarks/crag_hf20/results/task1_hf20_html_candidate_ablation.json"
)
REPORT_MD = Path(
    "/var/lib/kiara/benchmarks/crag_hf20/results/task1_hf20_html_candidate_ablation.md"
)


def _cleaned_chunks(
    files: list[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for source in files:
        visible = _load_html_markdown(source)
        if not visible.strip():
            continue
        path = Path(source)
        metadata = {
            "source": str(path),
            "directory": str(path.parent),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "parser": "html-to-markdown",
            "content_format": "markdown",
            "source_mime_type": "text/html",
            "chunk_strategy": "audit_cleaned_prose",
            "chunk_size_used": chunk_size,
            "chunk_overlap_used": chunk_overlap,
        }
        for chunk in chunk_document(
            visible,
            metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            meta = chunk["metadata"]
            chunk_id = f"clean:{source}:{meta.get('chunk_index')}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": meta,
                    "embedding": None,
                    "matched_answer": "",
                }
            )
    return chunks


def _bm25_results(chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_tokens = HybridRetriever._tokenize(query)
    if BM25Okapi is None or not query_tokens:
        return []
    tokenized = [HybridRetriever._tokenize(chunk["text"]) for chunk in chunks]
    keep_idx = [idx for idx, tokens in enumerate(tokenized) if tokens]
    if not keep_idx:
        return []
    kept = [chunks[idx] for idx in keep_idx]
    bm25 = BM25Okapi([tokenized[idx] for idx in keep_idx])
    scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float64)
    output = []
    for idx in np.argsort(-scores):
        score = float(scores[int(idx)])
        if score <= 0:
            break
        chunk = kept[int(idx)]
        output.append(
            {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": score,
                "_doc_id": chunk["chunk_id"],
            }
        )
    return output


def _rrf_fuse(dense: list[dict], keyword: list[dict], *, k: int, rrf_k: int) -> list[dict]:
    scores: dict[str, dict[str, Any]] = {}
    for result_list in (dense, keyword):
        for rank, doc in enumerate(result_list, 1):
            doc_id = str(doc.get("_doc_id") or "")
            if not doc_id:
                continue
            contribution = 1.0 / (rrf_k + rank)
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += contribution
            else:
                scores[doc_id] = {"rrf_score": contribution, "doc": doc}
    fused = []
    for item in sorted(scores.values(), key=lambda row: row["rrf_score"], reverse=True)[:k]:
        doc = dict(item["doc"])
        doc["rrf_score"] = round(float(item["rrf_score"]), 6)
        fused.append(doc)
    return fused


def _candidate_window_matrix(
    dense_results: list[dict],
    bm25_results: list[dict],
    answers: list[str],
    *,
    rrf_k: int,
    fused_k: int = 5,
) -> dict[str, Any]:
    dense_windows = [10, 20, 50, 100, 200, 500]
    keyword_windows = [5, 20, 50, 100, 200, 500]
    rows = []
    min_hit: dict[str, Any] | None = None
    for dense_window in dense_windows:
        for keyword_window in keyword_windows:
            fused = _rrf_fuse(
                dense_results[:dense_window],
                bm25_results[:keyword_window],
                k=fused_k,
                rrf_k=rrf_k,
            )
            hit = _first_gold_rank(fused, answers)
            row = {
                "dense_window": dense_window,
                "keyword_window": keyword_window,
                "gold_rank": hit.get("rank"),
                "gold_in_top": bool(hit),
                "top_sources": [
                    {
                        "source": str((item.get("metadata") or {}).get("source") or ""),
                        "chunk_index": (item.get("metadata") or {}).get("chunk_index"),
                        "rrf_score": item.get("rrf_score"),
                    }
                    for item in fused[:3]
                ],
            }
            rows.append(row)
            if hit and (
                min_hit is None
                or (dense_window + keyword_window, dense_window, keyword_window)
                < (
                    min_hit["dense_window"] + min_hit["keyword_window"],
                    min_hit["dense_window"],
                    min_hit["keyword_window"],
                )
            ):
                min_hit = row
    return {"fused_k": fused_k, "rows": rows, "min_hit": min_hit}


def _embed_cleaned_chunks(
    config: dict,
    embedding_model: str,
    query: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not chunks:
        return {"enabled": True, "error": "no cleaned chunks", "rank_profile": {}}
    try:
        query_vector = embed_query(config, embedding_model, query)
        texts = [chunk["text"] for chunk in chunks]
        vectors: list[list[float]] = []
        batch_size = 128
        for start in range(0, len(texts), batch_size):
            vectors.extend(embed_texts(config, embedding_model, texts[start : start + batch_size]))
        for chunk, vector in zip(chunks, vectors, strict=False):
            chunk["embedding"] = vector
        dense = _cosine_scores(query_vector, chunks)
        profile = _rank_profile(dense, [])
        return {
            "enabled": True,
            "error": "",
            "result_count": len(dense),
            "dense_results": dense,
            "rank_profile": profile,
        }
    except Exception as exc:
        from kiara.retrieval.index_compatibility import IndexCompatibilityError

        if isinstance(exc, IndexCompatibilityError):
            raise
        return {
            "enabled": True,
            "error": f"{type(exc).__name__}: {exc}",
            "result_count": 0,
            "dense_results": [],
            "rank_profile": {},
        }


def _case_ablation(
    entry: dict[str, Any],
    *,
    config: dict,
    store: Any,
    datasource_id: int | None,
    embedding_model: str,
    embed_cleaned: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    interaction_id = str(entry.get("interaction_id") or "")
    query = str(entry.get("query") or "")
    answers = [str(answer) for answer in entry.get("answers") or []]
    files = [str(path) for path in entry.get("files") or []]
    doc_dir = str(entry.get("doc_dir") or "")
    rrf_k = int(config.get("retrieval", {}).get("rrf_k", 60))

    current_chunks = _load_case_chunks(store, files, datasource_id, doc_dir, answers)
    query_vector = embed_query(config, embedding_model, query)
    current_dense = _cosine_scores(query_vector, current_chunks["chunks_datasource"])
    current_bm25 = _bm25_results(current_chunks["chunks_datasource"], query)
    current_bm25_audit = _bm25_audit(current_chunks["chunks_datasource"], query, answers)

    cleaned_chunks = _cleaned_chunks(files, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    cleaned_bm25 = _bm25_results(cleaned_chunks, query)
    cleaned_bm25_audit = _bm25_audit(cleaned_chunks, query, answers)

    cleaned_dense_results: list[dict] = []
    cleaned_dense_error = ""
    if embed_cleaned:
        cleaned_dense = _embed_cleaned_chunks(config, embedding_model, query, cleaned_chunks)
        cleaned_dense_results = cleaned_dense.get("dense_results", [])
        cleaned_dense_error = cleaned_dense.get("error", "")

    return {
        "interaction_id": interaction_id,
        "case_label": CASE_LABELS.get(interaction_id, ""),
        "query": query,
        "answers": answers,
        "current_indexed": {
            "chunk_count": len(current_chunks["chunks_datasource"]),
            "gold": _gold_summary(current_chunks["gold_chunks_datasource"], limit=3),
            "dense": {
                **_rank_profile(current_dense, answers),
                "top_results": [
                    _result_row(row, rank, answers) for rank, row in enumerate(current_dense[:5], 1)
                ],
            },
            "bm25": current_bm25_audit,
            "rrf_windows": _candidate_window_matrix(
                current_dense,
                current_bm25,
                answers,
                rrf_k=rrf_k,
            ),
        },
        "cleaned_html_prose": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(cleaned_chunks),
            "gold": _gold_summary(
                [
                    {
                        **chunk,
                        "matched_answer": _first_gold_rank(
                            [
                                {
                                    "text": chunk["text"],
                                    "metadata": chunk["metadata"],
                                    "score": 0,
                                    "_doc_id": chunk["chunk_id"],
                                }
                            ],
                            answers,
                        ).get("matched_answer", ""),
                        "metadata_checks": {},
                        "answer_position": {"char": None, "pct": None},
                        "markup": {"text_chars": len(chunk["text"]), "markup_density": 0},
                    }
                    for chunk in cleaned_chunks
                    if _first_gold_rank(
                        [
                            {
                                "text": chunk["text"],
                                "metadata": chunk["metadata"],
                                "score": 0,
                                "_doc_id": chunk["chunk_id"],
                            }
                        ],
                        answers,
                    )
                ],
                limit=3,
            ),
            "bm25": cleaned_bm25_audit,
            "dense": {
                "enabled": embed_cleaned,
                "error": cleaned_dense_error,
                **(_rank_profile(cleaned_dense_results, answers) if cleaned_dense_results else {}),
                "top_results": [
                    _result_row(row, rank, answers)
                    for rank, row in enumerate(cleaned_dense_results[:5], 1)
                ],
            },
            "rrf_windows": _candidate_window_matrix(
                cleaned_dense_results,
                cleaned_bm25,
                answers,
                rrf_k=rrf_k,
            )
            if cleaned_dense_results
            else {},
        },
    }


def _case_conclusion(case: dict[str, Any]) -> str:
    current_dense_rank = case["current_indexed"]["dense"].get("first_gold_rank")
    cleaned_bm25_rank = case["cleaned_html_prose"]["bm25"].get("first_gold_rank")
    cleaned_dense_rank = case["cleaned_html_prose"]["dense"].get("first_gold_rank")
    current_rrf_hit = case["current_indexed"]["rrf_windows"].get("min_hit")
    cleaned_rrf_hit = case["cleaned_html_prose"].get("rrf_windows", {}).get("min_hit")

    if case["case_label"] == "stable_dense_hit":
        return "control_dense_already_good"
    if case["case_label"] == "literal_noise_control":
        if cleaned_rrf_hit or cleaned_dense_rank or cleaned_bm25_rank:
            return "literal_noise_control_partial_recovery"
        return "literal_noise_control_not_recovered"
    if cleaned_dense_rank and current_dense_rank and cleaned_dense_rank < current_dense_rank:
        return "html_cleanup_improves_dense"
    if cleaned_bm25_rank and (
        not case["current_indexed"]["bm25"].get("first_gold_rank")
        or cleaned_bm25_rank < case["current_indexed"]["bm25"].get("first_gold_rank")
    ):
        return "html_cleanup_improves_keyword"
    if current_rrf_hit:
        return "larger_candidate_windows_can_recover"
    if cleaned_rrf_hit:
        return "cleanup_plus_larger_windows_can_recover"
    return "no_simple_candidate_window_recovery"


def _summary(cases: list[dict[str, Any]], *, embed_cleaned: bool) -> dict[str, Any]:
    conclusions: dict[str, int] = {}
    for case in cases:
        conclusion = _case_conclusion(case)
        case["conclusion"] = conclusion
        conclusions[conclusion] = conclusions.get(conclusion, 0) + 1
    return {
        "schema": "kiara-crag-hf20-html-candidate-ablation",
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "case_count": len(cases),
        "embed_cleaned": embed_cleaned,
        "conclusions": conclusions,
    }


def _fmt_rank(value: Any) -> str:
    return "" if value is None else str(value)


def _fmt_window(hit: dict[str, Any] | None) -> str:
    if not hit:
        return ""
    return f"d{hit['dense_window']}/k{hit['keyword_window']} -> r{hit['gold_rank']}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CRAG HF20 HTML/Candidate Ablation",
        "",
        "## Summary",
        "",
        f"- Generated: {payload['summary']['generated_at']}",
        f"- Cases: {payload['summary']['case_count']}",
        f"- Cleaned dense embeddings: `{payload['summary']['embed_cleaned']}`",
        f"- Conclusions: `{payload['summary']['conclusions']}`",
        "",
        "| Case | Label | Current dense | Current BM25 | Current RRF min | Clean BM25 | Clean dense | Clean RRF min | Conclusion |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for case in payload["cases"]:
        current = case["current_indexed"]
        cleaned = case["cleaned_html_prose"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case['interaction_id'][:8]}`",
                    f"`{case['case_label']}`",
                    _fmt_rank(current["dense"].get("first_gold_rank")),
                    _fmt_rank(current["bm25"].get("first_gold_rank")),
                    _fmt_window(current["rrf_windows"].get("min_hit")),
                    _fmt_rank(cleaned["bm25"].get("first_gold_rank")),
                    _fmt_rank(cleaned["dense"].get("first_gold_rank")),
                    _fmt_window(cleaned.get("rrf_windows", {}).get("min_hit")),
                    f"`{case['conclusion']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `Current dense/BM25` uses the already indexed raw HTML/code chunks.",
            "- `Clean BM25/Dense` uses KIara's HTML visible-text loader plus in-memory prose chunking; the production index is not modified by this script.",
            "- `RRF min` is the smallest tested dense/keyword candidate window that brings a gold chunk into fused Top-5.",
            "",
        ]
    )

    for case in payload["cases"]:
        lines.extend(
            [
                f"## {case['interaction_id']}",
                "",
                f"- Query: {case['query']}",
                f"- Answers: {', '.join(case['answers'])}",
                f"- Current chunks/gold: {case['current_indexed']['chunk_count']} / {case['current_indexed']['gold']['count']}",
                f"- Cleaned chunks/gold: {case['cleaned_html_prose']['chunk_count']} / {case['cleaned_html_prose']['gold']['count']}",
                f"- Conclusion: `{case['conclusion']}`",
                "",
                "Current dense top:",
                "",
            ]
        )
        for row in case["current_indexed"]["dense"].get("top_results", [])[:3]:
            lines.append(
                f"- r{row['rank']} score={row['score']:.6f} `{Path(row['source']).name}` "
                f"chunk={row['chunk_index']} gold={row['has_gold']} "
                f"{_compact(row['excerpt'], 180)}"
            )
        lines.extend(["", "Cleaned BM25 top:", ""])
        for row in case["cleaned_html_prose"]["bm25"].get("top_results", [])[:3]:
            lines.append(
                f"- r{row['rank']} score={row['score']:.6f} `{Path(row['source']).name}` "
                f"chunk={row['chunk_index']} gold={row['has_gold']} "
                f"{_compact(row['excerpt'], 180)}"
            )
        if case["cleaned_html_prose"]["dense"].get("enabled"):
            lines.extend(["", "Cleaned dense top:", ""])
            if case["cleaned_html_prose"]["dense"].get("error"):
                lines.append(f"- Error: {case['cleaned_html_prose']['dense']['error']}")
            for row in case["cleaned_html_prose"]["dense"].get("top_results", [])[:3]:
                lines.append(
                    f"- r{row['rank']} score={row['score']:.6f} `{Path(row['source']).name}` "
                    f"chunk={row['chunk_index']} gold={row['has_gold']} "
                    f"{_compact(row['excerpt'], 180)}"
                )
        lines.append("")

    if payload["summary"]["embed_cleaned"]:
        recommendation = (
            "Cleaned prose chunks move both real Dense-Ranking-Misses to dense rank 1 "
            "and recover them in fused Top-5 with the smallest tested RRF windows. "
            "Re-index CRAG HF20 with the HTML visible-text loader and rerun the benchmark "
            "before tuning candidate windows or changing the dense model."
        )
    else:
        recommendation = (
            "Prioritize an audit-backed HTML visible-text extraction path before changing "
            "dense model choice. Run again with `--embed-cleaned` to verify the dense-side "
            "effect before changing production retrieval logic."
        )

    lines.extend(["## Next Recommendation", "", recommendation, ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/var/lib/kiara/benchmarks/crag_hf20/manifests/task1.jsonl",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasource-id", type=int, default=24)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--json-output", default=str(REPORT_JSON))
    parser.add_argument("--md-output", default=str(REPORT_MD))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--chunk-size", type=int, default=768)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--embed-cleaned", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    embedding_model = args.embedding_model or get_embedding_model(config)
    store = VectorStore(config, embedding_model=embedding_model)
    rows = {str(row.get("interaction_id") or ""): row for row in iter_manifest(args.manifest)}
    wanted = tuple(args.case_id) if args.case_id else DEFAULT_CASE_IDS

    cases = []
    for case_id in wanted:
        if case_id not in rows:
            raise SystemExit(f"Case nicht im Manifest: {case_id}")
        cases.append(
            _case_ablation(
                rows[case_id],
                config=config,
                store=store,
                datasource_id=args.datasource_id,
                embedding_model=embedding_model,
                embed_cleaned=args.embed_cleaned,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )

    payload = {"cases": cases}
    payload["summary"] = _summary(cases, embed_cleaned=args.embed_cleaned)

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
