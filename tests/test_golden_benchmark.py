import json
from pathlib import Path

import pytest

from kiara.benchmarks import golden


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def _case(case_id: str = "KGS-T-001") -> dict:
    return {
        "id": case_id,
        "question": "Wie hoch ist die Rechnung 123?",
        "expected_sources": [
            {
                "label": "invoice",
                "source_contains": "/mail/invoice-123.txt",
                "source_type": "email",
                "required": True,
            }
        ],
        "expected_answer_type": "amount",
        "allowed_answer_variants": ["42,00 EUR", "42,00 €"],
        "evaluation_rule": "contains_any",
        "required_datasources": [12],
        "expected_failure_class": "short_or_numeric_answer",
        "expected_behavior": "rag",
        "sensitivity": "test-data",
        "short_or_numeric_answer": True,
        "tags": ["invoice"],
    }


def test_load_golden_set_validates_required_fields(tmp_path):
    path = tmp_path / "golden.jsonl"
    record = _case()
    _write_jsonl(path, [record])

    assert golden.load_golden_set(path) == [record]

    broken = dict(record)
    broken.pop("expected_behavior")
    broken_path = tmp_path / "broken.jsonl"
    _write_jsonl(broken_path, [broken])

    with pytest.raises(ValueError, match="expected_behavior"):
        golden.load_golden_set(broken_path)


def test_load_golden_set_rejects_unknown_taxonomies_and_empty_source_specs(tmp_path):
    base = _case()

    empty_source = dict(base)
    empty_source["expected_sources"] = [{}]
    empty_source_path = tmp_path / "empty-source.jsonl"
    _write_jsonl(empty_source_path, [empty_source])
    with pytest.raises(ValueError, match="expected_sources\\[1\\] needs"):
        golden.load_golden_set(empty_source_path)

    unknown_behavior = dict(base)
    unknown_behavior["expected_behavior"] = "rag_typo"
    unknown_behavior_path = tmp_path / "unknown-behavior.jsonl"
    _write_jsonl(unknown_behavior_path, [unknown_behavior])
    with pytest.raises(ValueError, match="unknown expected_behavior"):
        golden.load_golden_set(unknown_behavior_path)

    unknown_answer_type = dict(base)
    unknown_answer_type["expected_answer_type"] = "answer_typo"
    unknown_answer_type_path = tmp_path / "unknown-answer-type.jsonl"
    _write_jsonl(unknown_answer_type_path, [unknown_answer_type])
    with pytest.raises(ValueError, match="unknown expected_answer_type"):
        golden.load_golden_set(unknown_answer_type_path)

    unknown_failure = dict(base)
    unknown_failure["expected_failure_class"] = "new_unvalidated_bucket"
    unknown_failure_path = tmp_path / "unknown-failure.jsonl"
    _write_jsonl(unknown_failure_path, [unknown_failure])
    with pytest.raises(ValueError, match="unknown expected_failure_class"):
        golden.load_golden_set(unknown_failure_path)


def test_load_golden_set_allows_matcherless_sql_object_only_for_tool_behaviors(tmp_path):
    tool_case = _case("KGS-T-SQL")
    tool_case["expected_behavior"] = "tool_api"
    tool_case["expected_answer_type"] = "erp_sql_count"
    tool_case["expected_failure_class"] = "tool_api_expected_not_rag"
    tool_case["expected_sources"] = [
        {"label": "order-count", "source_type": "sql_object", "required": True}
    ]
    tool_path = tmp_path / "tool.jsonl"
    _write_jsonl(tool_path, [tool_case])
    assert golden.load_golden_set(tool_path) == [tool_case]

    rag_case = dict(tool_case)
    rag_case["expected_behavior"] = "rag"
    rag_path = tmp_path / "rag-sql-object.jsonl"
    _write_jsonl(rag_path, [rag_case])
    with pytest.raises(ValueError, match="source_type='sql_object'"):
        golden.load_golden_set(rag_path)


def test_gold_ctx_requires_answer_evidence_not_only_matching_source():
    case = _case()
    chunks = [
        {
            "rank": 1,
            "text": "Diese Passage nennt nur die Rechnungsnummer, nicht den Betrag.",
            "metadata": {"source": "/mail/invoice-123.txt"},
        },
        {
            "rank": 2,
            "text": "Der Rechnungsbetrag lautet 42,00 EUR.",
            "metadata": {"source": "/mail/invoice-123.txt"},
        },
    ]

    metrics = golden._metrics_for_case(case, chunks, (1, 2))

    assert metrics["recall@1"] == 1.0
    assert metrics["source_gold_ctx@1"] == 1.0
    assert metrics["evidence_recall@1"] == 0.0
    assert metrics["gold_ctx@1"] == 0.0
    assert metrics["mrr@1"] == 0.0
    assert metrics["ndcg@1"] == 0.0
    assert metrics["source_hits_by_k"]["1"]["hits"][0]["source_hit"] is True
    assert metrics["source_hits_by_k"]["1"]["hits"][0]["evidence_hit"] is False

    assert metrics["gold_ctx@2"] == 1.0
    assert metrics["evidence_recall@2"] == 1.0
    assert metrics["mrr@2"] == 0.5
    assert metrics["first_required_source_rank@2"] == 1
    assert metrics["first_required_evidence_rank@2"] == 2


def test_path_answer_type_can_use_source_metadata_as_evidence():
    case = _case("KGS-009")
    case["question"] = "Welche Datei enthaelt die Testfrage zu KIARA-CANARY-742?"
    case["expected_answer_type"] = "path_or_filename"
    case["expected_failure_class"] = "metadata_path_question"
    case["short_or_numeric_answer"] = False
    case["allowed_answer_variants"] = ["KIARA-CANARY-742-Q.txt"]
    case["expected_sources"] = [
        {
            "label": "canary",
            "source_contains": "KIARA-CANARY-742-Q.txt",
            "source_type": "document",
            "required": True,
        }
    ]
    chunks = [
        {
            "rank": 1,
            "text": "Der Chunktext nennt den Dateinamen nicht.",
            "metadata": {"source": "/kb/KIARA-CANARY-742-Q.txt"},
        }
    ]

    metrics = golden._metrics_for_case(case, chunks, (1,))

    assert metrics["gold_ctx@1"] == 1.0
    assert metrics["source_hits_by_k"]["1"]["hits"][0]["first_evidence_hit"]["source"].endswith(
        "KIARA-CANARY-742-Q.txt"
    )


def test_retrieval_mode_writes_recall_goldctx_and_path_states(tmp_path, monkeypatch):
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, [_case()])

    def fake_load_config(_config_path=None):
        return {
            "embedding": {"model": "test-embedding"},
            "retrieval": {
                "hybrid_search": "elasticsearch",
                "colbert": {"enabled": True, "model": "test-colbert"},
            },
        }

    monkeypatch.setattr("kiara.config.load_config", fake_load_config)

    class FakeRetriever:
        def retrieve(self, query, **kwargs):
            assert query == "Wie hoch ist die Rechnung 123?"
            assert kwargs["filter_datasource_ids"] is None
            return [
                {
                    "text": "Rechnungsbetrag 42,00 EUR",
                    "metadata": {
                        "source": "/mail/invoice-123.txt",
                        "chunk_index": 0,
                        "chunk_total": 1,
                        "file_type": ".txt",
                    },
                    "score": 0.9,
                    "semantic_rank": 1,
                    "rrf_rank": 1,
                    "retriever_paths": ["semantic"],
                }
            ]

        def get_last_retrieval_diagnostics(self):
            return {
                "schema": "kiara-retrieval-path-diagnostics",
                "paths": {
                    "semantic": {"status": "success", "latency_ms": 5},
                    "keyword": {"status": "empty", "latency_ms": 1},
                    "colbert": {"status": "empty", "latency_ms": 0},
                    "graph": {"status": "disabled", "latency_ms": 0},
                },
                "rrf": {"status": "success", "latency_ms": 1},
                "final_results": [],
            }

    stats = golden.evaluate_golden_set(
        path,
        output_dir=tmp_path / "results",
        mode="retrieval",
        k_values=(1, 3),
        run_label="unit",
        retriever=FakeRetriever(),
    )

    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["overall"]["recall@1"] == 1.0
    assert summary["overall"]["source_recall@1"] == 1.0
    assert summary["overall"]["evidence_recall@1"] == 1.0
    assert summary["overall"]["source_gold_ctx@1"] == 1.0
    assert summary["overall"]["gold_ctx@1"] == 1.0
    assert summary["overall"]["mrr@1"] == 1.0
    assert summary["overall"]["ndcg@1"] == 1.0
    assert summary["overall"]["overall_by_behavior"]["rag"]["gold_ctx@1"] == 1.0
    assert (
        summary["overall"]["overall_by_failure_class"]["short_or_numeric_answer"]["gold_ctx@1"]
        == 1.0
    )
    assert summary["overall"]["answer_status_counts"] == {"not_generated": 1}
    assert summary["path_state_counts"]["keyword"]["empty"] == 1
    assert "keyword_path_empty" in summary["partial_state_warnings"]
    assert len(summary["reproducibility"]["golden_set_sha256"]) == 64
    assert len(summary["reproducibility"]["config_sha256"]) == 64
    assert "git" in summary["reproducibility"]
    assert len(summary["artifacts"]["details_sha256"]) == 64
    assert len(summary["artifacts"]["summary_payload_sha256"]) == 64
    assert summary["run"]["overwrite"] is False
    assert details[0]["metrics"]["source_hits_by_k"]["1"]["required_hit"] == 1
    assert details[0]["metrics"]["source_hits_by_k"]["1"]["required_evidence_hit"] == 1
    assert details[0]["metrics"]["source_hits_by_k"]["1"]["hits"][0]["evidence_hit"] is True
    assert details[0]["retrieved_chunks"][0]["source"] == "/mail/invoice-123.txt"


def test_golden_run_label_requires_overwrite_for_existing_artifacts(tmp_path, monkeypatch):
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, [_case()])
    monkeypatch.setattr(
        "kiara.config.load_config",
        lambda _config_path=None: {"embedding": {"model": "test-embedding"}},
    )

    class CountingRetriever:
        def __init__(self):
            self.calls = 0

        def retrieve(self, **kwargs):
            self.calls += 1
            return [
                {
                    "text": "Rechnungsbetrag 42,00 EUR",
                    "metadata": {"source": "/mail/invoice-123.txt"},
                    "score": 0.9,
                }
            ]

    output_dir = tmp_path / "results"
    first = CountingRetriever()
    golden.evaluate_golden_set(
        path,
        output_dir=output_dir,
        mode="retrieval",
        k_values=(1,),
        run_label="duplicate",
        retriever=first,
    )
    assert first.calls == 1

    second = CountingRetriever()
    with pytest.raises(FileExistsError, match="--overwrite"):
        golden.evaluate_golden_set(
            path,
            output_dir=output_dir,
            mode="retrieval",
            k_values=(1,),
            run_label="duplicate",
            retriever=second,
        )
    assert second.calls == 0

    third = CountingRetriever()
    stats = golden.evaluate_golden_set(
        path,
        output_dir=output_dir,
        mode="retrieval",
        k_values=(1,),
        run_label="duplicate",
        retriever=third,
        overwrite=True,
    )
    assert third.calls == 1
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    assert summary["run"]["overwrite"] is True


def test_rag_mode_evaluates_answer_status(tmp_path, monkeypatch):
    path = tmp_path / "golden.jsonl"
    _write_jsonl(path, [_case()])
    monkeypatch.setattr(
        "kiara.config.load_config",
        lambda _config_path=None: {"embedding": {"model": "test-embedding"}},
    )

    class FakeChain:
        last_retrieval_diagnostics = {
            "schema": "kiara-rag-retrieval-diagnostics",
            "variants": [
                {
                    "schema": "kiara-retrieval-path-diagnostics",
                    "paths": {"semantic": {"status": "success", "latency_ms": 2}},
                    "rrf": {"status": "success", "latency_ms": 1},
                }
            ],
        }

        def query(self, **kwargs):
            return {
                "answer": "Die Rechnung betraegt 42,00 EUR.",
                "context_chunks": [
                    {
                        "text": "Rechnungsbetrag 42,00 EUR",
                        "metadata": {"source": "/mail/invoice-123.txt"},
                        "final_context_rank": 1,
                    }
                ],
                "retrieval_diagnostics": self.last_retrieval_diagnostics,
            }

    stats = golden.evaluate_golden_set(
        path,
        output_dir=tmp_path / "results",
        mode="rag",
        k_values=(1,),
        run_label="rag-unit",
        chain=FakeChain(),
    )

    detail = json.loads(Path(stats.details_path).read_text(encoding="utf-8").splitlines()[0])

    assert detail["answer_status"] == "correct"
    assert detail["metrics"]["recall@1"] == 1.0
    assert detail["retriever_path_states"]["semantic"] == "success"


class TestAbstainExpectedPolicy:
    """Policy 'streng quellengebunden' (2026-07-08): Abstain-Bewertung."""

    def _case(self, **over):
        case = {
            "id": "KGS-T1",
            "question": "Testfrage?",
            "expected_sources": [{"source": "/x.txt"}],
            "expected_answer_type": "summary",
            "allowed_answer_variants": ["antwortwert"],
            "evaluation_rule": "contains_any",
            "required_datasources": [1],
            "expected_failure_class": "generator_only",
            "expected_behavior": "rag",
            "sensitivity": "none",
            "short_or_numeric_answer": False,
        }
        case.update(over)
        return case

    def test_tool_api_abstain_counts_correct(self):
        from kiara.benchmarks.golden import _answer_status

        case = self._case(expected_behavior="tool_api")
        status = _answer_status(
            case, "Dazu finde ich keine Informationen in der Wissensbasis.", "", generated=True
        )
        assert status == "correct"

    def test_abstain_expected_flag_counts_correct(self):
        from kiara.benchmarks.golden import _answer_status

        case = self._case(abstain_expected=True)
        status = _answer_status(case, "Ich weiss es nicht.", "", generated=True)
        assert status == "correct"

    def test_rag_abstain_still_missing(self):
        from kiara.benchmarks.golden import _answer_status

        case = self._case()
        status = _answer_status(
            case, "Dazu finde ich keine Informationen in der Wissensbasis.", "", generated=True
        )
        assert status == "missing"

    def test_tool_api_real_answer_still_evaluated(self):
        from kiara.benchmarks.golden import _answer_status

        case = self._case(expected_behavior="tool_api")
        assert _answer_status(case, "Der antwortwert ist X.", "", generated=True) == "correct"
        assert _answer_status(case, "Etwas voellig anderes.", "", generated=True) == "incorrect"

    def test_abstain_expected_must_be_bool(self):
        import pytest

        from kiara.benchmarks.golden import validate_golden_case

        with pytest.raises(ValueError, match="abstain_expected"):
            validate_golden_case(self._case(abstain_expected="ja"))
