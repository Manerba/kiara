import json

import pytest

from kiara.benchmarks.crag import (
    analyze_crag_audit_cases,
    audit_crag_ingestion_embeddings,
    audit_crag_preparation_coverage,
    build_crag_vertical_audit_report,
    compare_summaries,
    comparison_to_markdown,
    crag_audit_case_analysis_to_markdown,
    estimate_official_judge_tokens,
    evaluate_predictions,
    evaluate_predictions_with_official_judge,
    evaluate_retrieval,
    export_official_eval_payload,
    iter_crag_records,
    parse_official_judge_response,
    prepare_crag_task,
    summarize_results,
)
from kiara.web.api_benchmarks import _collect_crag_artifacts


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_prepare_task1_exports_web_docs_without_gold_answer(tmp_path):
    input_path = tmp_path / "crag.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "interaction_id": "case-1",
                "query": "What is the total?",
                "answer": "42 EUR",
                "alt_ans": "[]",
                "alternative_answers": ["42 Euro"],
                "search_results": [
                    {
                        "page_name": "Invoice",
                        "page_url": "https://example.invalid/invoice",
                        "page_result": "<html><body>Total is visible here.</body></html>",
                    }
                ],
            }
        ],
    )

    stats = prepare_crag_task(input_path, tmp_path / "out", 1)

    assert stats.records_written == 1
    assert stats.files_written == 1
    manifest = (tmp_path / "out" / "manifests" / "task1.jsonl").read_text(encoding="utf-8")
    assert "42 EUR" in manifest
    assert "42 Euro" in manifest
    assert '"[]"' not in manifest
    document = next((tmp_path / "out" / "documents" / "task1" / "case-1").iterdir())
    assert "42 EUR" not in document.read_text(encoding="utf-8")


def test_prepare_task2_exports_structured_retrieval_payload(tmp_path):
    input_path = tmp_path / "crag.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "interaction_id": "case-2",
                "query": "What is the ticker?",
                "answer": "META",
                "mock_api_result": {"company": "Meta Platforms", "ticker": "META"},
            }
        ],
    )

    stats = prepare_crag_task(input_path, tmp_path / "out", 2)

    assert stats.files_written == 1
    document = next((tmp_path / "out" / "documents" / "task2" / "case-2").iterdir())
    payload = json.loads(document.read_text(encoding="utf-8"))
    assert payload["field"] == "mock_api_result"
    assert payload["value"]["ticker"] == "META"


def test_iter_crag_records_reads_pretty_printed_jsonl_stream(tmp_path):
    input_path = tmp_path / "pretty.jsonl"
    input_path.write_text(
        '{\n  "interaction_id": "one",\n  "query": "Q1"\n}\n'
        '{\n  "interaction_id": "two",\n  "query": "Q2"\n}\n',
        encoding="utf-8",
    )

    records = list(iter_crag_records(input_path))

    assert [record["interaction_id"] for record in records] == ["one", "two"]


def test_lightweight_evaluator_scores_correct_missing_incorrect(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {"interaction_id": "a", "prediction": "The value is 42 EUR.", "answers": ["42 EUR"]},
            {
                "interaction_id": "b",
                "prediction": "Keine relevanten Dokumente gefunden.",
                "answers": ["Paris"],
            },
            {"interaction_id": "c", "prediction": "Berlin", "answers": ["Paris"]},
        ],
    )

    stats = evaluate_predictions(predictions)

    assert stats.total == 3
    assert stats.correct == 1
    assert stats.missing == 1
    assert stats.incorrect == 1
    assert stats.score == 0


def test_lightweight_evaluator_does_not_substring_match_short_answers(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {
                "interaction_id": "short",
                "prediction": "Die Frage enthält keine direkte Antwort.",
                "answers": ["en"],
            }
        ],
    )

    stats = evaluate_predictions(predictions)

    assert stats.correct == 0
    assert stats.incorrect == 1


def test_export_official_eval_payload_writes_crag_sequences(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "official.json"
    _write_jsonl(
        predictions,
        [
            {
                "interaction_id": "case-1",
                "query": "What is the total?",
                "answers": ["42 EUR", "42 Euro"],
                "prediction": "42 EUR",
                "domain": "finance",
            }
        ],
    )

    stats = export_official_eval_payload(predictions, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert stats.total == 1
    assert payload["queries"] == ["What is the total?"]
    assert payload["ground_truths"] == ["42 EUR"]
    assert payload["ground_truths_list"] == [["42 EUR", "42 Euro"]]
    assert payload["predictions"] == ["42 EUR"]
    assert payload["interaction_ids"] == ["case-1"]
    assert payload["metadata"][0]["domain"] == "finance"


def test_parse_official_judge_response_accepts_json_and_fences():
    assert parse_official_judge_response('{"score": 1, "explanation": "ok"}') == (
        1,
        "ok",
        "",
    )
    assert parse_official_judge_response('```json\n{"score": 0, "explanation": "no"}\n```') == (
        0,
        "no",
        "",
    )


def test_official_judge_token_estimate_skips_local_scores(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        predictions,
        [
            {"interaction_id": "a", "query": "Q1", "prediction": "42 EUR", "answers": ["42 EUR"]},
            {
                "interaction_id": "b",
                "query": "Q2",
                "prediction": "Keine relevanten Dokumente gefunden.",
                "answers": ["Paris"],
            },
            {"interaction_id": "c", "query": "Q3", "prediction": "Berlin", "answers": ["Paris"]},
        ],
    )

    estimate = estimate_official_judge_tokens(predictions, max_completion_tokens=12)

    assert estimate["total"] == 3
    assert estimate["local_exact"] == 1
    assert estimate["local_missing"] == 1
    assert estimate["llm_judged"] == 1
    assert estimate["estimated_completion_tokens"] == 12
    assert estimate["estimated_prompt_tokens"] > 0


def test_official_judge_estimate_only_writes_pending_details(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "judge.json"
    _write_jsonl(
        predictions,
        [
            {"interaction_id": "a", "query": "Q1", "prediction": "42 EUR", "answers": ["42 EUR"]},
            {"interaction_id": "b", "query": "Q2", "prediction": "Berlin", "answers": ["Paris"]},
        ],
    )

    stats = evaluate_predictions_with_official_judge(
        predictions,
        output,
        estimate_only=True,
        max_completion_tokens=12,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert stats.estimate_only is True
    assert stats.total == 2
    assert stats.correct == 1
    assert stats.llm_judged == 1
    assert stats.estimated_total_tokens > 0
    assert payload["details"][1]["label"] == "pending_judge"


def test_retrieval_evaluator_scores_answer_context_and_source_scope(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    doc_dir = tmp_path / "docs" / "case-1"
    source = doc_dir / "web_001.html"
    _write_jsonl(
        predictions,
        [
            {
                "interaction_id": "case-1",
                "query": "What is the total?",
                "answers": ["42 EUR"],
                "doc_dir": str(doc_dir),
                "latency_ms": 120,
                "context_chunks": [
                    {"source": str(source), "excerpt": "No value here."},
                    {"source": str(source), "excerpt": "The total is 42 EUR."},
                ],
            },
            {
                "interaction_id": "case-2",
                "query": "What is the city?",
                "answers": ["Paris"],
                "doc_dir": str(tmp_path / "docs" / "case-2"),
                "latency_ms": 80,
                "context_chunks": [{"source": str(tmp_path / "other.txt"), "excerpt": "Berlin"}],
            },
        ],
    )

    stats = evaluate_retrieval(predictions)

    assert stats.total == 2
    assert stats.with_context == 2
    assert stats.gold_answer_in_context == 1
    assert stats.gold_answer_in_context_at_1 == 0
    assert stats.gold_answer_in_context_at_3 == 1
    assert stats.any_source_under_doc_dir == 1
    assert stats.all_sources_under_doc_dir == 1
    assert stats.avg_context_chunks == 1.5
    assert stats.avg_latency_ms == 100


def test_retrieval_evaluator_writes_path_diagnostic_schema(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "retrieval_eval.json"
    source = tmp_path / "docs" / "case-1" / "web_001.html"
    source.parent.mkdir(parents=True)
    _write_jsonl(
        predictions,
        [
            {
                "interaction_id": "case-1",
                "query": "What is the total?",
                "answers": ["42 EUR"],
                "doc_dir": str(source.parent),
                "short_or_numeric_answer": True,
                "context_chunks": [
                    {
                        "rank": 1,
                        "final_context_rank": 1,
                        "chunk_id": "chunk-answer",
                        "point_id": "chunk-answer",
                        "source": str(source),
                        "semantic_rank": 2,
                        "semantic_score": 0.8,
                        "keyword_rank": 1,
                        "keyword_score": 12.0,
                        "rrf_rank": 1,
                        "rrf_score": 0.0325,
                        "rrf_contributions": {"semantic": 0.0161, "keyword": 0.0164},
                        "retriever_paths": ["semantic", "keyword"],
                        "excerpt": "The total is 42 EUR.",
                    }
                ],
                "retrieval_diagnostics": {
                    "schema": "kiara-rag-retrieval-diagnostics",
                    "schema_version": 1,
                    "variants": [
                        {
                            "schema": "kiara-retrieval-path-diagnostics",
                            "paths": {
                                "semantic": {
                                    "status": "success",
                                    "latency_ms": 7,
                                    "hits": [
                                        {
                                            "rank": 2,
                                            "chunk_id": "chunk-answer",
                                            "point_id": "chunk-answer",
                                            "source": str(source),
                                            "score": 0.8,
                                            "excerpt": "The total is 42 EUR.",
                                        }
                                    ],
                                },
                                "keyword": {
                                    "status": "success",
                                    "latency_ms": 3,
                                    "hits": [
                                        {
                                            "rank": 1,
                                            "chunk_id": "chunk-answer",
                                            "point_id": "chunk-answer",
                                            "source": str(source),
                                            "score": 12.0,
                                            "excerpt": "The total is 42 EUR.",
                                        }
                                    ],
                                },
                            },
                            "rrf": {"status": "success", "latency_ms": 1},
                            "final_results": [
                                {
                                    "rrf_rank": 1,
                                    "chunk_id": "chunk-answer",
                                    "point_id": "chunk-answer",
                                    "source": str(source),
                                    "rrf_score": 0.0325,
                                    "excerpt": "The total is 42 EUR.",
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    )

    evaluate_retrieval(predictions, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    detail = payload["details"][0]

    assert payload["summary"]["diagnostic_fields"]["schema"] == "kiara-retrieval-path-diagnostics"
    assert detail["short_or_numeric_answer"] is True
    assert detail["path_gold_ranks"]["semantic_rank"] == 2
    assert detail["path_gold_ranks"]["keyword_rank"] == 1
    assert detail["path_gold_ranks"]["rrf_rank"] == 1
    assert detail["path_gold_ranks"]["final_context_rank"] == 1
    assert detail["retriever_path_states"] == {
        "semantic": "success",
        "keyword": "success",
        "rrf": "success",
    }
    assert detail["context_chunk_details"][0]["chunk_id"] == "chunk-answer"


def test_summarize_results_builds_metadata_breakdowns(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    doc_dir = tmp_path / "docs" / "case-1"
    source = doc_dir / "web_001.html"
    _write_jsonl(
        predictions,
        [
            {
                "interaction_id": "case-1",
                "task": 1,
                "domain": "finance",
                "question_type": "comparison",
                "static_or_dynamic": "dynamic",
                "query": "What is the total?",
                "answers": ["42 EUR"],
                "prediction": "The value is 42 EUR.",
                "doc_dir": str(doc_dir),
                "latency_ms": 100,
                "context_chunks": [{"source": str(source), "excerpt": "The value is 42 EUR."}],
            },
            {
                "interaction_id": "case-2",
                "task": 1,
                "domain": "finance",
                "question_type": "simple",
                "static_or_dynamic": "static",
                "query": "What is the city?",
                "answers": ["Paris"],
                "prediction": "Berlin",
                "doc_dir": str(tmp_path / "docs" / "case-2"),
                "latency_ms": 300,
                "context_chunks": [],
            },
        ],
    )

    summary = summarize_results(
        predictions,
        breakdown_fields=("domain", "question_type", "scope", "embedding_model", "k"),
        run_metadata={
            "scope": "per-question",
            "retrieval_mode": "none",
            "embedding_model": "nomic-embed-text",
            "k": 5,
        },
    )

    assert summary["overall"]["total"] == 2
    assert summary["overall"]["accuracy"] == 0.5
    assert summary["overall"]["gold_answer_in_context_rate"] == 0.5
    assert summary["overall"]["avg_latency_ms"] == 200
    assert summary["overall"]["p50_latency_ms"] == 200
    assert summary["breakdowns"]["domain"]["finance"]["total"] == 2
    assert summary["breakdowns"]["question_type"]["comparison"]["accuracy"] == 1
    assert summary["breakdowns"]["question_type"]["simple"]["hallucination_rate"] == 1
    assert summary["breakdowns"]["scope"]["per-question"]["total"] == 2
    assert summary["breakdowns"]["embedding_model"]["nomic-embed-text"]["total"] == 2
    assert summary["breakdowns"]["k"]["5"]["total"] == 2
    assert summary["run"]["retrieval_mode"] == "none"


def test_compare_summaries_reports_deltas(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "run": {"label": "baseline", "retrieval_mode": "elasticsearch", "k": 5},
                "overall": {
                    "total": 10,
                    "score": -0.5,
                    "accuracy": 0.1,
                    "missing_rate": 0.2,
                    "hallucination_rate": 0.7,
                    "gold_answer_in_context_rate": 0.1,
                    "gold_answer_in_context_at_1_rate": 0.1,
                    "avg_latency_ms": 100,
                    "p50_latency_ms": 90,
                    "p95_latency_ms": 150,
                },
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "run": {"label": "dense", "retrieval_mode": "none", "k": 5},
                "overall": {
                    "total": 10,
                    "score": -0.3,
                    "accuracy": 0.2,
                    "missing_rate": 0.1,
                    "hallucination_rate": 0.7,
                    "gold_answer_in_context_rate": 0.2,
                    "gold_answer_in_context_at_1_rate": 0.1,
                    "avg_latency_ms": 80,
                    "p50_latency_ms": 70,
                    "p95_latency_ms": 120,
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_summaries([first, second])
    markdown = comparison_to_markdown(comparison)

    assert comparison["runs"][1]["delta_vs_baseline"]["score"] == 0.2
    assert comparison["runs"][1]["metrics"]["accuracy"] == 0.2
    assert "| dense | none | 5 | -0.300 |" in markdown


def test_collect_crag_artifacts_reads_summaries_and_comparisons(tmp_path):
    results = tmp_path / "dataset" / "results"
    results.mkdir(parents=True)
    summary_path = results / "run_summary.json"
    comparison_path = results / "run_comparison.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema": "kiara-crag-summary",
                "run": {"label": "hybrid_k5", "retrieval_mode": "elasticsearch", "k": 5},
                "overall": {"total": 20, "score": -0.75},
                "breakdowns": {"domain": {"finance": {"total": 4, "score": -1}}},
            }
        ),
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(
            {
                "schema": "kiara-crag-comparison",
                "baseline_index": 0,
                "baseline_label": "hybrid_k5",
                "runs": [{"label": "hybrid_k5", "metrics": {"score": -0.75}}],
            }
        ),
        encoding="utf-8",
    )

    artifacts = _collect_crag_artifacts(tmp_path)

    assert artifacts["summaries"][0]["label"] == "hybrid_k5"
    assert artifacts["summaries"][0]["dataset"] == "dataset"
    assert artifacts["summaries"][0]["overall"]["score"] == -0.75
    assert artifacts["comparisons"][0]["baseline_label"] == "hybrid_k5"


def test_audit_crag_ingestion_embeddings_separates_index_and_vector_hits(tmp_path):
    doc_dir = tmp_path / "docs" / "case-1"
    doc_dir.mkdir(parents=True)
    source = doc_dir / "web_001.txt"
    source.write_text("The quarterly total is 42 EUR.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "audit.jsonl"
    summary_output = tmp_path / "audit_summary.json"
    _write_jsonl(
        manifest,
        [
            {
                "interaction_id": "case-1",
                "task": 1,
                "query": "What is the total?",
                "answers": ["42 EUR"],
                "doc_dir": str(doc_dir),
                "files": [str(source)],
                "file_count": 1,
            }
        ],
    )

    class FakeStore:
        def get_ids_by_sources(self, sources, *, ds_id=None):
            assert sources == [str(source)]
            assert ds_id == 24
            return ["chunk-answer"]

        def get_documents_by_ids(self, ids, include=None):
            assert ids == ["chunk-answer"]
            return {
                "ids": ["chunk-answer"],
                "documents": ["The indexed chunk still says 42 EUR."],
                "metadatas": [{"source": str(source), "chunk_index": 0}],
            }

        def search(self, query, k=10, where=None):
            assert query == "What is the total?"
            assert k == 3
            assert where == {
                "$and": [
                    {"directory": {"$contains": str(doc_dir)}},
                    {"ds_ids": {"$in": [24]}},
                ]
            }
            return [
                {
                    "text": "No value here.",
                    "metadata": {"source": str(source), "chunk_index": 1},
                    "score": 0.9,
                },
                {
                    "text": "The embedded hit contains 42 EUR.",
                    "metadata": {"source": str(source), "chunk_index": 0},
                    "score": 0.8,
                },
            ]

    summary = audit_crag_ingestion_embeddings(
        manifest,
        output,
        config={"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}},
        store=FakeStore(),
        datasource_id=24,
        k=3,
        summary_output=summary_output,
    )
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])

    assert summary["total"] == 1
    assert summary["raw_has_gold"] == 1
    assert summary["chunk_has_gold"] == 1
    assert summary["indexed_has_gold"] == 1
    assert summary["vector_topk_has_gold"] == 1
    assert summary["diagnosis_counts"] == {"embedding_topk_hit": 1}
    assert row["vector_gold_rank"] == 2
    assert row["vector_gold_source"] == str(source)
    assert row["diagnosis"] == "embedding_topk_hit"
    assert json.loads(summary_output.read_text(encoding="utf-8"))["total"] == 1


def test_audit_crag_preparation_coverage_reports_page_limit_and_snippet_loss(tmp_path):
    raw = tmp_path / "raw.jsonl"
    docs = tmp_path / "docs"
    first_dir = docs / "case-1"
    second_dir = docs / "case-2"
    third_dir = docs / "case-3"
    for directory in (first_dir, second_dir, third_dir):
        directory.mkdir(parents=True)

    (first_dir / "web_001.html").write_text("The invoice says 42 EUR.", encoding="utf-8")
    for rank in range(1, 4):
        (second_dir / f"web_{rank:03d}.html").write_text("No answer here.", encoding="utf-8")
    (third_dir / "web_001.html").write_text("HTML body without answer.", encoding="utf-8")

    _write_jsonl(
        raw,
        [
            {
                "interaction_id": "case-1",
                "query": "What is the total?",
                "answer": "42 EUR",
                "search_results": [{"page_result": "The invoice says 42 EUR."}],
            },
            {
                "interaction_id": "case-2",
                "query": "What is the city?",
                "answer": "Paris",
                "search_results": [
                    {"page_result": "No answer here."},
                    {"page_result": "No answer here."},
                    {"page_result": "No answer here."},
                    {"page_result": "The city is Paris."},
                ],
            },
            {
                "interaction_id": "case-3",
                "query": "What is the code?",
                "answer": "ZX-9",
                "search_results": [
                    {
                        "page_result": "HTML body without answer.",
                        "page_snippet": "The code is ZX-9.",
                    }
                ],
            },
        ],
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "interaction_id": "case-1",
                "task": 1,
                "query": "What is the total?",
                "answers": ["42 EUR"],
                "doc_dir": str(first_dir),
                "files": [str(first_dir / "web_001.html")],
            },
            {
                "interaction_id": "case-2",
                "task": 1,
                "query": "What is the city?",
                "answers": ["Paris"],
                "doc_dir": str(second_dir),
                "files": [str(second_dir / f"web_{rank:03d}.html") for rank in range(1, 4)],
            },
            {
                "interaction_id": "case-3",
                "task": 1,
                "query": "What is the code?",
                "answers": ["ZX-9"],
                "doc_dir": str(third_dir),
                "files": [str(third_dir / "web_001.html")],
            },
        ],
    )
    output = tmp_path / "prep_audit.jsonl"
    summary = audit_crag_preparation_coverage(
        raw,
        manifest,
        output,
        page_limits=(1, 3, 5),
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    by_id = {row["interaction_id"]: row for row in rows}

    assert summary["total"] == 3
    assert summary["prepared_has_gold"] == 1
    assert summary["top_limit_counts"] == {"top_1": 1, "top_3": 1, "top_5": 2}
    assert summary["diagnosis_counts"] == {
        "page_limit_miss": 1,
        "prepared_hit": 1,
        "snippet_shadowed_by_html": 1,
    }
    assert by_id["case-2"]["first_preferred_gold_rank"] == 4
    assert by_id["case-2"]["diagnosis"] == "page_limit_miss"
    assert by_id["case-3"]["source_any_within_prepared_has_gold"] is True
    assert by_id["case-3"]["source_preferred_within_prepared_has_gold"] is False


def test_analyze_crag_audit_cases_classifies_actionable_labels(tmp_path):
    preparation = tmp_path / "prep.jsonl"
    ingestion = tmp_path / "ingestion.jsonl"
    output = tmp_path / "analysis.json"
    markdown_output = tmp_path / "analysis.md"
    _write_jsonl(
        preparation,
        [
            {
                "interaction_id": "dense-miss",
                "query": "Who won?",
                "answers": ["Tom Hanks"],
                "diagnosis": "prepared_hit",
                "prepared_has_gold": True,
            },
            {
                "interaction_id": "dense-hit-short",
                "query": "How many?",
                "answers": ["1"],
                "diagnosis": "prepared_hit",
                "prepared_has_gold": True,
            },
            {
                "interaction_id": "noise",
                "query": "How high?",
                "answers": ["5 m"],
                "diagnosis": "source_missing",
                "prepared_has_gold": False,
                "source_any_has_gold": False,
            },
            {
                "interaction_id": "page-limit",
                "query": "How many films?",
                "answers": ["109"],
                "diagnosis": "page_limit_miss",
                "first_preferred_gold_rank": 4,
                "prepared_max_web_rank": 3,
            },
            {
                "interaction_id": "missing-source",
                "query": "What is missing?",
                "answers": ["Paris"],
                "diagnosis": "source_missing",
            },
        ],
    )
    _write_jsonl(
        ingestion,
        [
            {
                "interaction_id": "dense-miss",
                "answers": ["Tom Hanks"],
                "diagnosis": "embedding_topk_miss",
                "indexed_has_gold": True,
                "vector_topk_has_gold": False,
                "indexed_answer_chunks": [{"source": "doc.html", "excerpt": "Winner: Tom Hanks"}],
            },
            {
                "interaction_id": "dense-hit-short",
                "answers": ["1"],
                "diagnosis": "embedding_topk_hit",
                "indexed_has_gold": True,
                "vector_topk_has_gold": True,
                "vector_gold_rank": 3,
                "vector_gold_hit": {"source": "doc.html", "rank": 3, "excerpt": "Score: 1"},
            },
            {
                "interaction_id": "noise",
                "answers": ["5 m"],
                "diagnosis": "embedding_topk_miss",
                "chunk_has_gold": True,
                "indexed_has_gold": True,
                "vector_topk_has_gold": False,
            },
            {
                "interaction_id": "page-limit",
                "answers": ["109"],
                "diagnosis": "raw_missing",
                "indexed_has_gold": False,
                "vector_topk_has_gold": False,
            },
            {
                "interaction_id": "missing-source",
                "answers": ["Paris"],
                "diagnosis": "raw_missing",
                "indexed_has_gold": False,
                "vector_topk_has_gold": False,
            },
        ],
    )

    analysis = analyze_crag_audit_cases(
        preparation,
        ingestion,
        output,
        markdown_output=markdown_output,
    )
    by_id = {case["interaction_id"]: case for case in analysis["cases"]}
    markdown = crag_audit_case_analysis_to_markdown(analysis)

    assert analysis["summary"]["label_counts"] == {
        "audit_match_noise": 1,
        "dense_hit_low_confidence": 1,
        "dense_rank_miss": 1,
        "page_limit_candidate": 1,
        "source_missing": 1,
    }
    assert analysis["summary"]["actionable_counts"]["retrieval_or_ranking"] == 1
    assert by_id["dense-miss"]["recommended_next_step"].startswith("Audit retrieval")
    assert by_id["noise"]["warnings"] == [
        "short_or_numeric_answer",
        "preparation_source_missing_but_ingestion_match",
    ]
    assert by_id["page-limit"]["warnings"] == ["short_or_numeric_answer"]
    assert "dense_rank_miss" in output.read_text(encoding="utf-8")
    assert "| dense_rank_miss |" in markdown
    assert markdown_output.read_text(encoding="utf-8").startswith("# CRAG Audit Case Analysis")


def test_build_crag_vertical_audit_report_classifies_pipeline_losses(tmp_path):
    preparation = tmp_path / "prep.jsonl"
    ingestion = tmp_path / "ingestion.jsonl"
    retrieval = tmp_path / "retrieval.json"
    judge = tmp_path / "judge.json"
    output = tmp_path / "vertical.json"
    markdown = tmp_path / "vertical.md"

    _write_jsonl(
        preparation,
        [
            {"interaction_id": "ok", "answers": ["42"], "diagnosis": "prepared_hit"},
            {"interaction_id": "generator", "answers": ["Paris"], "diagnosis": "prepared_hit"},
            {"interaction_id": "source", "answers": ["Rome"], "diagnosis": "source_missing"},
            {
                "interaction_id": "prep",
                "answers": ["Madrid"],
                "diagnosis": "page_limit_miss",
            },
            {"interaction_id": "loader", "answers": ["A"], "diagnosis": "prepared_hit"},
            {"interaction_id": "chunk", "answers": ["B"], "diagnosis": "prepared_hit"},
            {"interaction_id": "dense", "answers": ["C"], "diagnosis": "prepared_hit"},
            {"interaction_id": "rrf", "answers": ["D"], "diagnosis": "prepared_hit"},
            {
                "interaction_id": "index",
                "answers": ["E"],
                "diagnosis": "prepared_hit",
                "prepared_has_gold": True,
            },
        ],
    )
    _write_jsonl(
        ingestion,
        [
            {"interaction_id": "ok", "answers": ["42"], "indexed_has_gold": True},
            {"interaction_id": "generator", "answers": ["Paris"], "indexed_has_gold": True},
            {"interaction_id": "loader", "answers": ["A"], "raw_has_gold": True},
            {
                "interaction_id": "chunk",
                "answers": ["B"],
                "raw_has_gold": True,
                "loaded_has_gold": True,
            },
            {
                "interaction_id": "dense",
                "answers": ["C"],
                "indexed_has_gold": True,
                "vector_topk_has_gold": True,
                "vector_gold_rank": 25,
            },
            {
                "interaction_id": "rrf",
                "answers": ["D"],
                "indexed_has_gold": True,
                "vector_topk_has_gold": True,
                "vector_gold_rank": 4,
            },
            {
                "interaction_id": "index",
                "answers": ["E"],
                "chunk_has_gold": True,
                "indexed_has_gold": False,
            },
        ],
    )
    retrieval_details = [
        {"interaction_id": "ok", "answers": ["42"], "gold_answer_in_context": True},
        {
            "interaction_id": "generator",
            "answers": ["Paris"],
            "gold_answer_in_context": True,
        },
    ]
    retrieval_details.extend(
        {
            "interaction_id": case_id,
            "answers": [answer],
            "gold_answer_in_context": False,
        }
        for case_id, answer in (
            ("source", "Rome"),
            ("prep", "Madrid"),
            ("loader", "A"),
            ("chunk", "B"),
            ("dense", "C"),
            ("rrf", "D"),
            ("index", "E"),
        )
    )
    retrieval.write_text(
        json.dumps(
            {
                "details": retrieval_details,
            }
        ),
        encoding="utf-8",
    )
    judge.write_text(
        json.dumps(
            {
                "details": [
                    {"interaction_id": "ok", "label": "correct"},
                    {"interaction_id": "generator", "label": "incorrect"},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_crag_vertical_audit_report(
        preparation,
        ingestion,
        retrieval,
        judge_eval_path=judge,
        output_path=output,
        markdown_output=markdown,
        dense_candidate_k=10,
    )

    by_id = {case["interaction_id"]: case for case in report["cases"]}
    assert by_id["ok"]["failure_class"] == "no_loss_observed"
    assert by_id["generator"]["failure_class"] == "generator_only"
    assert by_id["source"]["failure_class"] == "source_missing"
    assert by_id["prep"]["failure_class"] == "preparation_loss"
    assert by_id["loader"]["failure_class"] == "loader_loss"
    assert by_id["chunk"]["failure_class"] == "chunking_loss"
    assert by_id["dense"]["failure_class"] == "dense_rank_miss"
    assert by_id["rrf"]["failure_class"] == "rrf_loss"
    assert by_id["index"]["failure_class"] == "unknown"
    assert by_id["index"]["unknown_reason"] == "qdrant_index_loss_not_in_requested_taxonomy"
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["total"] == 9
    assert markdown.read_text(encoding="utf-8").startswith("# CRAG Full136 Vertical Audit")


def test_build_crag_vertical_audit_report_requires_existing_required_artifacts(tmp_path):
    existing = tmp_path / "rows.jsonl"
    _write_jsonl(existing, [{"interaction_id": "case-1", "answers": ["42"]}])
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(
        json.dumps({"details": [{"interaction_id": "case-1", "answers": ["42"]}]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="preparation_audit"):
        build_crag_vertical_audit_report(
            tmp_path / "missing-prep.jsonl",
            existing,
            retrieval,
        )


def test_build_crag_vertical_audit_report_rejects_empty_required_artifacts(tmp_path):
    empty_preparation = tmp_path / "empty-prep.jsonl"
    empty_preparation.write_text("", encoding="utf-8")
    ingestion = tmp_path / "ingestion.jsonl"
    _write_jsonl(ingestion, [{"interaction_id": "case-1", "answers": ["42"]}])
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(
        json.dumps({"details": [{"interaction_id": "case-1", "answers": ["42"]}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preparation_audit.*interaction_id"):
        build_crag_vertical_audit_report(empty_preparation, ingestion, retrieval)
