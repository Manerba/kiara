import json
from pathlib import Path

from kiara.benchmarks import generator_prompt


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _golden_case() -> dict:
    return {
        "id": "KGS-009",
        "question": "Welche Datei enthaelt die Testfrage zu KIARA-CANARY-742?",
        "expected_sources": [
            {
                "label": "canary",
                "source_contains": "KIARA-CANARY-742-Q.txt",
                "source_type": "document",
                "required": True,
            }
        ],
        "expected_answer_type": "path_or_filename",
        "allowed_answer_variants": ["KIARA-CANARY-742-Q.txt"],
        "evaluation_rule": "contains_any",
        "required_datasources": [19],
        "expected_failure_class": "metadata_path_question",
        "expected_behavior": "rag",
        "sensitivity": "test",
        "short_or_numeric_answer": False,
        "tags": ["path"],
    }


def test_collects_golden_and_crag_goldctx_bad_cases(tmp_path):
    golden_set = tmp_path / "golden.jsonl"
    _write_jsonl(golden_set, [_golden_case()])
    golden_details = tmp_path / "golden_details.jsonl"
    _write_jsonl(
        golden_details,
        [
            {
                "id": "KGS-009",
                "question": "Welche Datei enthaelt die Testfrage zu KIARA-CANARY-742?",
                "answer": "KIARA-CANARY-742.txt",
                "answer_status": "incorrect",
                "metrics": {
                    "gold_ctx@20": 1.0,
                    "source_hits_by_k": {
                        "20": {
                            "hits": [
                                {
                                    "hit": True,
                                    "first_hit": {
                                        "rank": 1,
                                        "source": "/docs/KIARA-CANARY-742-Q.txt",
                                    },
                                }
                            ]
                        }
                    },
                },
                "retrieved_chunks": [
                    {
                        "rank": 1,
                        "source": "/docs/KIARA-CANARY-742-Q.txt",
                        "excerpt": "Welchen Wartungsstatus hat KIARA-CANARY-742?",
                    }
                ],
            }
        ],
    )
    crag_audit = tmp_path / "crag_audit.json"
    crag_audit.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "interaction_id": "crag-1",
                        "failure_class": "generator_only",
                        "answer_label": "missing",
                        "answers": ["42"],
                        "query": "what is the number?",
                        "warnings": ["short_or_numeric_answer"],
                        "stage_statuses": {
                            "prompt_context_has_gold": True,
                            "final_context_rank": 2,
                        },
                    }
                ],
                "summary": {},
            }
        ),
        encoding="utf-8",
    )
    crag_predictions = tmp_path / "crag_predictions.jsonl"
    _write_jsonl(
        crag_predictions,
        [
            {
                "interaction_id": "crag-1",
                "query": "what is the number?",
                "prediction": "Keine Information.",
                "latency_ms": 123,
                "context_chunks": [
                    {"rank": 1, "source": "/case/doc.txt", "excerpt": "The number is 42."}
                ],
            }
        ],
    )

    cases = generator_prompt.collect_goldctx_bad_cases(
        golden_set_path=golden_set,
        golden_baseline_details_path=golden_details,
        crag_audit_path=crag_audit,
        crag_baseline_predictions_path=crag_predictions,
    )

    assert [case["case_id"] for case in cases] == ["crag-1", "KGS-009"]
    assert cases[0]["dataset"] == "crag_full136"
    assert "missing_policy_falsch_angewendet" in cases[0]["error_clusters"]
    assert cases[1]["answers"] == ["KIARA-CANARY-742-Q.txt"]
    assert cases[1]["gold_context_rank"] == 1


def test_run_audit_uses_frozen_context_and_scores_variants(tmp_path, monkeypatch):
    golden_set = tmp_path / "golden.jsonl"
    _write_jsonl(golden_set, [_golden_case()])
    golden_details = tmp_path / "golden_details.jsonl"
    _write_jsonl(
        golden_details,
        [
            {
                "id": "KGS-009",
                "question": "Welche Datei enthaelt die Testfrage zu KIARA-CANARY-742?",
                "answer": "KIARA-CANARY-742.txt",
                "answer_status": "incorrect",
                "metrics": {
                    "gold_ctx@20": 1.0,
                    "source_hits_by_k": {"20": {"hits": [{"hit": True, "first_hit": {"rank": 1}}]}},
                },
                "retrieved_chunks": [
                    {
                        "rank": 1,
                        "source": "/docs/KIARA-CANARY-742-Q.txt",
                        "excerpt": "Welchen Wartungsstatus hat KIARA-CANARY-742?",
                    }
                ],
            }
        ],
    )
    crag_audit = tmp_path / "crag_audit.json"
    crag_audit.write_text(json.dumps({"cases": [], "summary": {}}), encoding="utf-8")
    crag_predictions = tmp_path / "crag_predictions.jsonl"
    _write_jsonl(crag_predictions, [])

    monkeypatch.setattr(
        "kiara.config.load_config", lambda _path=None: {"ollama": {"num_ctx": 4096}}
    )

    class FakeBackend:
        provider = "ollama"
        model = "fake"
        last_usage = {"prompt_tokens": 10, "response_tokens": 2}

        def generate(self, _system_prompt, user_message, **kwargs):
            assert "KIARA-CANARY-742-Q.txt" in user_message
            assert kwargs["temperature"] == 0.0
            return "KIARA-CANARY-742-Q.txt"

    stats = generator_prompt.run_generator_prompt_audit(
        output_root=tmp_path / "out",
        run_label="unit",
        golden_set_path=golden_set,
        golden_baseline_details_path=golden_details,
        crag_audit_path=crag_audit,
        crag_baseline_predictions_path=crag_predictions,
        variant_labels={"short_extractive_answer"},
        backend_factory=lambda _config, _backend_id: FakeBackend(),
    )

    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    detail = json.loads(Path(stats.details_path).read_text(encoding="utf-8").splitlines()[0])

    assert stats.cases == 1
    assert detail["answer_status"] == "correct"
    assert summary["case_selection"]["baseline"]["incorrect"] == 1
    assert summary["metrics_by_variant"]["short_extractive_answer"]["goldctx_bad_reduction"] == 1
    assert summary["recommendation"]["hallucination_not_increased"] is True
    assert summary["metrics_by_variant"]["short_extractive_answer"]["incorrect_rate"] == 0.0
    assert "hallucination_rate" not in summary["metrics_by_variant"]["short_extractive_answer"]


def test_same_run_current_prompt_is_primary_control(tmp_path, monkeypatch):
    golden_set = tmp_path / "golden.jsonl"
    _write_jsonl(golden_set, [_golden_case()])
    golden_details = tmp_path / "golden_details.jsonl"
    _write_jsonl(
        golden_details,
        [
            {
                "id": "KGS-009",
                "question": "Welche Datei enthaelt die Testfrage zu KIARA-CANARY-742?",
                "answer": "KIARA-CANARY-742.txt",
                "answer_status": "incorrect",
                "metrics": {
                    "gold_ctx@20": 1.0,
                    "source_hits_by_k": {"20": {"hits": [{"hit": True, "first_hit": {"rank": 1}}]}},
                },
                "retrieved_chunks": [
                    {
                        "rank": 1,
                        "source": "/docs/KIARA-CANARY-742-Q.txt",
                        "excerpt": "Welchen Wartungsstatus hat KIARA-CANARY-742?",
                    }
                ],
            }
        ],
    )
    crag_audit = tmp_path / "crag_audit.json"
    crag_audit.write_text(json.dumps({"cases": [], "summary": {}}), encoding="utf-8")
    crag_predictions = tmp_path / "crag_predictions.jsonl"
    _write_jsonl(crag_predictions, [])
    monkeypatch.setattr("kiara.config.load_config", lambda _path=None: {})

    class FakeBackend:
        provider = "fake"
        model = "fake"
        last_usage = {}

        def generate(self, _system_prompt, _user_message, **_kwargs):
            return "KIARA-CANARY-742-Q.txt"

    stats = generator_prompt.run_generator_prompt_audit(
        output_root=tmp_path / "out",
        run_label="same-run",
        golden_set_path=golden_set,
        golden_baseline_details_path=golden_details,
        crag_audit_path=crag_audit,
        crag_baseline_predictions_path=crag_predictions,
        variant_labels={"current_prompt_replay", "metadata_source_answer"},
        backend_factory=lambda _config, _backend_id: FakeBackend(),
    )

    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    metadata_metrics = summary["metrics_by_variant"]["metadata_source_answer"]

    assert summary["evaluation"]["primary_control_label"] == "current_prompt_replay"
    assert summary["evaluation"]["primary_control_source"] == "same_run_variant"
    assert metadata_metrics["artifact_baseline_goldctx_bad_reduction"] == 1
    assert metadata_metrics["goldctx_bad_reduction_vs_control"] == 0
    assert metadata_metrics["correct_delta_vs_control"] == 0
    assert summary["recommendation"]["best_variant"] is None
    assert summary["recommendation"]["measurable_reduction"] is False


def test_abstain_only_variant_is_not_safe_recommendation(tmp_path, monkeypatch):
    golden_set = tmp_path / "golden.jsonl"
    _write_jsonl(golden_set, [_golden_case()])
    golden_details = tmp_path / "golden_details.jsonl"
    _write_jsonl(
        golden_details,
        [
            {
                "id": "KGS-009",
                "question": "Welche Datei enthaelt die Testfrage zu KIARA-CANARY-742?",
                "answer": "Falsche Datei.txt",
                "answer_status": "incorrect",
                "metrics": {
                    "gold_ctx@20": 1.0,
                    "source_hits_by_k": {"20": {"hits": [{"hit": True, "first_hit": {"rank": 1}}]}},
                },
                "retrieved_chunks": [
                    {
                        "rank": 1,
                        "source": "/docs/KIARA-CANARY-742-Q.txt",
                        "excerpt": "Welchen Wartungsstatus hat KIARA-CANARY-742?",
                    }
                ],
            }
        ],
    )
    crag_audit = tmp_path / "crag_audit.json"
    crag_audit.write_text(json.dumps({"cases": [], "summary": {}}), encoding="utf-8")
    crag_predictions = tmp_path / "crag_predictions.jsonl"
    _write_jsonl(crag_predictions, [])
    monkeypatch.setattr("kiara.config.load_config", lambda _path=None: {})

    class FakeBackend:
        provider = "fake"
        model = "fake"
        last_usage = {}

        def generate(self, _system_prompt, user_message, **_kwargs):
            if "Ich weiss es nicht" in user_message:
                return "Ich weiss es nicht."
            return "Falsche Datei.txt"

    stats = generator_prompt.run_generator_prompt_audit(
        output_root=tmp_path / "out",
        run_label="abstain",
        golden_set_path=golden_set,
        golden_baseline_details_path=golden_details,
        crag_audit_path=crag_audit,
        crag_baseline_predictions_path=crag_predictions,
        variant_labels={"current_prompt_replay", "explicit_abstain_if_insufficient"},
        backend_factory=lambda _config, _backend_id: FakeBackend(),
    )

    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    metrics = summary["metrics_by_variant"]["explicit_abstain_if_insufficient"]

    assert metrics["incorrect_delta_vs_control"] == -1
    assert metrics["missing_delta_vs_control"] == 1
    assert metrics["correct_delta_vs_control"] == 0
    assert summary["recommendation"]["best_variant"] is None
    assert summary["recommendation"]["missing_not_increased"] is False


def test_metadata_source_answer_system_variant_and_prompt_dump(tmp_path, monkeypatch):
    golden_set = tmp_path / "golden.jsonl"
    _write_jsonl(golden_set, [_golden_case()])
    golden_details = tmp_path / "golden_details.jsonl"
    _write_jsonl(
        golden_details,
        [
            {
                "id": "KGS-009",
                "question": "Welche Datei enthaelt die Testfrage zu KIARA-CANARY-742?",
                "answer": "KIARA-CANARY-742.txt",
                "answer_status": "incorrect",
                "metrics": {
                    "gold_ctx@20": 1.0,
                    "source_hits_by_k": {"20": {"hits": [{"hit": True, "first_hit": {"rank": 1}}]}},
                },
                "retrieved_chunks": [
                    {
                        "rank": 1,
                        "source": "/docs/KIARA-CANARY-742-Q.txt",
                        "excerpt": "Welchen Wartungsstatus hat KIARA-CANARY-742?",
                    }
                ],
            }
        ],
    )
    crag_audit = tmp_path / "crag_audit.json"
    crag_audit.write_text(json.dumps({"cases": [], "summary": {}}), encoding="utf-8")
    crag_predictions = tmp_path / "crag_predictions.jsonl"
    _write_jsonl(crag_predictions, [])
    monkeypatch.setattr("kiara.config.load_config", lambda _path=None: {})

    class FakeBackend:
        provider = "fake"
        model = "fake"
        last_usage = {}

        def generate(self, system_prompt, user_message, **_kwargs):
            assert "Source-/Pfad-Header" in system_prompt
            assert "Source-/Pfad-Header" not in user_message
            return "KIARA-CANARY-742-Q.txt"

    stats = generator_prompt.run_generator_prompt_audit(
        output_root=tmp_path / "out",
        run_label="metadata-system",
        golden_set_path=golden_set,
        golden_baseline_details_path=golden_details,
        crag_audit_path=crag_audit,
        crag_baseline_predictions_path=crag_predictions,
        variant_labels={"metadata_source_answer_system"},
        backend_factory=lambda _config, _backend_id: FakeBackend(),
    )

    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    detail = json.loads(Path(stats.details_path).read_text(encoding="utf-8").splitlines()[0])
    prompt_dump = json.loads(
        Path(summary["artifacts"]["prompts"]).read_text(encoding="utf-8").splitlines()[0]
    )

    assert detail["answer_status"] == "correct"
    assert detail["evaluator_version"] == generator_prompt.EVALUATOR_VERSION
    assert len(detail["prompt_hashes"]["system_prompt_sha256"]) == 64
    assert prompt_dump["variant_label"] == "metadata_source_answer_system"
    assert "Source-/Pfad-Header" in prompt_dump["system_prompt"]
