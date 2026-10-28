"""Generator/prompt replay audit for frozen retrieval contexts.

This Sprint-5 harness intentionally does not call retrieval. It collects
GoldCtx-but-bad cases from existing Golden Set and CRAG artefacts, replays the
stored context chunks, and varies only the generator prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import socket
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from kiara.benchmarks import crag, golden
from kiara.benchmarks.artifacts import write_private_json, write_private_jsonl, write_private_text

DEFAULT_OUTPUT_ROOT = Path("/var/lib/kiara/benchmarks/generator_prompt")
DEFAULT_GOLDEN_SET = Path("/var/lib/kiara/benchmarks/kiara_golden/golden_set_v1.jsonl")
DEFAULT_GOLDEN_BASELINE_DETAILS = Path(
    "/var/lib/kiara/benchmarks/kiara_golden/results/golden_v1_live_rag_full_retry2_details.jsonl"
)
DEFAULT_CRAG_AUDIT = Path(
    "/var/lib/kiara/benchmarks/crag_full136/results/task1_full136_vertical_audit_diag_full.json"
)
DEFAULT_CRAG_BASELINE_PREDICTIONS = Path(
    "/var/lib/kiara/benchmarks/crag_full136/results/"
    "task1_full136_html_loader_hybrid_k5_predictions.jsonl"
)
DEFAULT_K_VALUES = (1, 3, 5, 10, 20)
DEFAULT_CONTEXT_EXCERPT_CHARS = 2000
CONTROL_VARIANT_LABEL = "current_prompt_replay"
EVALUATOR_VERSION = "generator_prompt_replay_v2_same_run_control_incorrect_rate"

BAD_STATUSES = {"incorrect", "missing"}
_REPLAY_MISSING_PATTERNS = (
    "ich weiss es nicht",
    "ich weiß es nicht",
    "keine information",
    "keine informationen",
    "keine daten",
    "keine angabe",
    "nicht im kontext",
    "nicht direkt im kontext",
    "nicht verfügbar",
    "nicht vorhanden",
    "nicht angegeben",
    "nicht beantwortet",
    "nicht aus dem gegebenen kontext",
    "kann nicht beantwortet",
    "does not include",
    "does not provide",
    "not available",
    "not provided",
    "cannot be answered",
    "cannot answer",
    "insufficient",
)


@dataclass(frozen=True)
class GeneratorPromptVariant:
    """Prompt variant evaluated against the same frozen evidence."""

    label: str
    description: str
    system_prompt: str = ""
    user_instruction: str = ""


@dataclass(frozen=True)
class GeneratorPromptAuditStats:
    """Summary returned by a Sprint-5 generator/prompt audit run."""

    label: str
    output_dir: str
    summary_path: str
    details_path: str
    report_path: str
    cases: int
    variants: int


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "run"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_private_json(path, payload, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_private_jsonl(path, rows, sort_keys=True)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _normalize_answer_text(text: str) -> str:
    return " ".join(str(text or "").casefold().replace("ß", "ss").split())


def _short_answer_values(values: list[str]) -> bool:
    return crag._has_low_confidence_literal_answer(values)


def _golden_cases_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in golden.load_golden_set(path)}


def _compact_replay_chunk(raw: dict[str, Any], rank: int, excerpt_chars: int) -> dict[str, Any]:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    source = str(raw.get("source") or metadata.get("source") or "")
    excerpt = str(raw.get("text") or raw.get("excerpt") or "")[:excerpt_chars]
    chunk_rank = int(raw.get("rank") or raw.get("final_context_rank") or rank)
    replay_metadata = dict(metadata)
    replay_metadata.setdefault("source", source)
    if raw.get("chunk_index") is not None:
        replay_metadata.setdefault("chunk_index", raw.get("chunk_index"))
    return {
        "rank": chunk_rank,
        "final_context_rank": raw.get("final_context_rank", chunk_rank),
        "chunk_id": str(raw.get("chunk_id") or raw.get("point_id") or ""),
        "point_id": str(raw.get("point_id") or raw.get("chunk_id") or ""),
        "source": source,
        "metadata": replay_metadata,
        "excerpt": excerpt,
        "text": excerpt,
        "text_chars": int(raw.get("text_chars") or len(excerpt)),
        "score": raw.get("score"),
        "semantic_rank": raw.get("semantic_rank"),
        "keyword_rank": raw.get("keyword_rank"),
        "colbert_rank": raw.get("colbert_rank"),
        "rrf_rank": raw.get("rrf_rank"),
        "retriever_paths": list(raw.get("retriever_paths") or []),
    }


def _context_chunks_from_row(row: dict[str, Any], excerpt_chars: int) -> list[dict[str, Any]]:
    raw_chunks = (
        row.get("retrieved_chunks")
        or row.get("context_chunks")
        or row.get("context_chunk_details")
        or []
    )
    chunks = []
    for rank, raw in enumerate(raw_chunks, 1):
        if isinstance(raw, dict):
            chunks.append(_compact_replay_chunk(raw, rank, excerpt_chars))
    return chunks


def _golden_gold_ctx(row: dict[str, Any], k: int) -> bool:
    value = row.get("metrics", {}).get(f"gold_ctx@{k}")
    try:
        return float(value) >= 1.0
    except (TypeError, ValueError):
        return False


def _first_gold_rank_from_golden(row: dict[str, Any], k: int) -> int | None:
    hits = row.get("metrics", {}).get("source_hits_by_k", {}).get(str(k), {}).get("hits", [])
    ranks = []
    for hit in hits:
        first_hit = hit.get("first_hit") if isinstance(hit, dict) else None
        if isinstance(first_hit, dict) and first_hit.get("rank") is not None:
            ranks.append(int(first_hit["rank"]))
    return min(ranks) if ranks else None


def _cluster_case(
    *,
    dataset: str,
    baseline_status: str,
    answers: list[str],
    question: str,
    prediction: str,
    gold_rank: int | None,
    warnings: list[str],
    expected_failure_class: str = "",
) -> list[str]:
    clusters: set[str] = set()
    if baseline_status == "missing":
        clusters.add("antwort_nicht_extrahiert")
        clusters.add("missing_policy_falsch_angewendet")
    if baseline_status == "incorrect":
        if _short_answer_values(answers) or "short_or_numeric_answer" in warnings:
            clusters.add("falsche_zahl_oder_entitaet")
        else:
            clusters.add("falsche_entitaet_oder_antwort")
    if gold_rank is not None and gold_rank > 1:
        clusters.add("zu_viel_oder_konkurrierender_kontext")
    if "source_conflict" in expected_failure_class:
        clusters.add("quellenkonflikt")
    lowered = _normalize_answer_text(prediction)
    if any(token in lowered for token in ("heute", "yesterday", "current", "weltwissen")):
        clusters.add("prompt_verletzt_quellenbindung")
    if dataset == "golden" and "metadata_path_question" in expected_failure_class:
        clusters.add("prompt_verletzt_quellenbindung")
    if "generator_only" in expected_failure_class:
        clusters.add("modellschwaeche")
    if not clusters:
        clusters.add("bewertungsheuristik_zweifelhaft")
    return sorted(clusters)


def collect_goldctx_bad_cases(
    *,
    golden_set_path: Path = DEFAULT_GOLDEN_SET,
    golden_baseline_details_path: Path = DEFAULT_GOLDEN_BASELINE_DETAILS,
    crag_audit_path: Path = DEFAULT_CRAG_AUDIT,
    crag_baseline_predictions_path: Path = DEFAULT_CRAG_BASELINE_PREDICTIONS,
    k: int = 20,
    context_excerpt_chars: int = DEFAULT_CONTEXT_EXCERPT_CHARS,
) -> list[dict[str, Any]]:
    """Collect measurable GoldCtx-but-bad cases from existing artefacts."""

    cases: list[dict[str, Any]] = []

    golden_by_id = _golden_cases_by_id(golden_set_path)
    if golden_baseline_details_path.exists():
        for row in _load_jsonl(golden_baseline_details_path):
            status = str(row.get("answer_status") or "")
            if status not in BAD_STATUSES or not _golden_gold_ctx(row, k):
                continue
            case = dict(golden_by_id.get(str(row.get("id")), {}))
            case.update(row)
            answers = [str(value) for value in case.get("allowed_answer_variants") or []]
            prediction = str(row.get("answer") or "")
            gold_rank = _first_gold_rank_from_golden(row, k)
            expected_failure_class = str(case.get("expected_failure_class") or "")
            cases.append(
                {
                    "dataset": "golden",
                    "case_id": str(row.get("id") or ""),
                    "question": str(row.get("question") or case.get("question") or ""),
                    "answers": answers,
                    "baseline_answer": prediction,
                    "baseline_status": status,
                    "baseline_latency_ms": row.get("latency_ms"),
                    "expected_answer_type": case.get("expected_answer_type"),
                    "expected_failure_class": expected_failure_class,
                    "expected_behavior": case.get("expected_behavior"),
                    "evaluation_rule": case.get("evaluation_rule", row.get("evaluation_rule")),
                    "short_or_numeric_answer": bool(case.get("short_or_numeric_answer", False)),
                    "gold_context_metric": f"gold_ctx@{k}",
                    "gold_context_rank": gold_rank,
                    "source_gold_ctx": True,
                    "warnings": [],
                    "error_clusters": _cluster_case(
                        dataset="golden",
                        baseline_status=status,
                        answers=answers,
                        question=str(row.get("question") or ""),
                        prediction=prediction,
                        gold_rank=gold_rank,
                        warnings=[],
                        expected_failure_class=expected_failure_class,
                    ),
                    "context_chunks": _context_chunks_from_row(row, context_excerpt_chars),
                    "source_artifacts": {
                        "golden_set": str(golden_set_path),
                        "baseline_details": str(golden_baseline_details_path),
                    },
                }
            )

    if crag_audit_path.exists() and crag_baseline_predictions_path.exists():
        audit = json.loads(crag_audit_path.read_text(encoding="utf-8"))
        predictions = {
            str(row.get("interaction_id") or ""): row
            for row in _load_jsonl(crag_baseline_predictions_path)
        }
        for row in audit.get("cases") or []:
            if row.get("failure_class") != "generator_only":
                continue
            statuses = (
                row.get("stage_statuses") if isinstance(row.get("stage_statuses"), dict) else {}
            )
            if not statuses.get("prompt_context_has_gold"):
                continue
            baseline_status = str(row.get("answer_label") or "")
            if baseline_status not in BAD_STATUSES:
                continue
            interaction_id = str(row.get("interaction_id") or "")
            prediction_row = predictions.get(interaction_id, {})
            answers = [str(value) for value in row.get("answers") or []]
            prediction = str(prediction_row.get("prediction") or "")
            gold_rank = statuses.get("final_context_rank")
            gold_rank_int = int(gold_rank) if gold_rank is not None else None
            warnings = [str(value) for value in row.get("warnings") or []]
            cases.append(
                {
                    "dataset": "crag_full136",
                    "case_id": interaction_id,
                    "question": str(row.get("query") or prediction_row.get("query") or ""),
                    "answers": answers,
                    "baseline_answer": prediction,
                    "baseline_status": baseline_status,
                    "baseline_latency_ms": prediction_row.get("latency_ms"),
                    "expected_answer_type": "crag_answer",
                    "expected_failure_class": "generator_only",
                    "expected_behavior": "rag",
                    "evaluation_rule": "crag_contains_or_missing",
                    "short_or_numeric_answer": _short_answer_values(answers)
                    or "short_or_numeric_answer" in warnings,
                    "gold_context_metric": "prompt_context_has_gold",
                    "gold_context_rank": gold_rank_int,
                    "source_gold_ctx": True,
                    "warnings": warnings,
                    "error_clusters": _cluster_case(
                        dataset="crag_full136",
                        baseline_status=baseline_status,
                        answers=answers,
                        question=str(row.get("query") or ""),
                        prediction=prediction,
                        gold_rank=gold_rank_int,
                        warnings=warnings,
                        expected_failure_class="generator_only",
                    ),
                    "context_chunks": _context_chunks_from_row(
                        prediction_row,
                        context_excerpt_chars,
                    ),
                    "source_artifacts": {
                        "vertical_audit": str(crag_audit_path),
                        "baseline_predictions": str(crag_baseline_predictions_path),
                    },
                }
            )

    cases.sort(key=lambda item: (item["dataset"], item["case_id"]))
    return cases


def build_default_variants() -> list[GeneratorPromptVariant]:
    """Return the Sprint-5 required prompt/generator variants."""

    strict_source_binding = (
        "Verwende ausschliesslich Informationen aus dem markierten Kontext. "
        "Uebernimm keine Fakten, Daten, Namen, Pfade oder Zahlen, die nicht "
        "im Kontext stehen. Wenn mehrere Kontextstellen konkurrieren, bevorzuge "
        "die Stelle, die die Frage am direktesten beantwortet."
    )
    numeric_entity_from_context = (
        "Zahlen-, Entitaets-, Datums-, Datei- und Pfadantworten sind nur erlaubt, "
        "wenn der exakte Wert im zitierten Kontext vorkommt. Gib die Antwort "
        "kurz aus und fuege die Quellenmarke hinzu, z.B. `Quelle 2`."
    )
    metadata_source_answer = (
        "Bei Fragen nach Datei, Pfad, Quelle, Dokumentname oder Metadaten muss "
        "der Source-/Pfad-Header des passenden Kontextchunks als Antwort-Evidenz "
        "gelten. Wenn nach einem Dateinamen gefragt ist, extrahiere den Basename "
        "aus dem Source-Pfad, statt ihn aus dem Chunktext zu erraten."
    )

    return [
        GeneratorPromptVariant(
            label=CONTROL_VARIANT_LABEL,
            description="Current KIara RAG user template replayed on frozen contexts.",
        ),
        GeneratorPromptVariant(
            label="strict_source_binding",
            description="Answer only from the supplied context; no unstated facts.",
            user_instruction=strict_source_binding,
        ),
        GeneratorPromptVariant(
            label="strict_source_binding_system",
            description="System-level source binding rule.",
            system_prompt=strict_source_binding,
        ),
        GeneratorPromptVariant(
            label="short_extractive_answer",
            description="Prefer a short extracted answer over explanation.",
            user_instruction=(
                "Gib eine kurze extraktive Antwort. Bei einer Zahl, Entitaet, Datei "
                "oder einem Pfad gib nur den gesuchten Wert plus eine knappe Quellenangabe "
                "aus. Keine Herleitung, keine Einleitung."
            ),
        ),
        GeneratorPromptVariant(
            label="explicit_abstain_if_insufficient",
            description="Abstain only when the evidence is truly insufficient.",
            user_instruction=(
                "Antworte genau `Ich weiss es nicht.`, wenn der Kontext die Antwort "
                "nicht explizit hergibt. Wenn der Kontext die Antwort enthaelt, darfst "
                "du nicht ausweichen und musst die Antwort extrahieren."
            ),
        ),
        GeneratorPromptVariant(
            label="no_world_knowledge",
            description="Do not use world knowledge or the current system date.",
            user_instruction=(
                "Nutze kein Weltwissen, keine Annahmen und nicht das heutige Systemdatum. "
                "Relative Zeitangaben wie yesterday, previous week oder most recent "
                "duerfen nur aus Kontext oder Fall-Metadaten interpretiert werden."
            ),
        ),
        GeneratorPromptVariant(
            label="numeric_entity_from_cited_context",
            description="Numbers and entities must be copied from cited context.",
            user_instruction=numeric_entity_from_context,
        ),
        GeneratorPromptVariant(
            label="numeric_entity_from_cited_context_system",
            description="System-level numeric/entity extraction rule.",
            system_prompt=numeric_entity_from_context,
        ),
        GeneratorPromptVariant(
            label="metadata_source_answer",
            description="Treat source/path metadata as answer evidence for metadata questions.",
            user_instruction=metadata_source_answer,
        ),
        GeneratorPromptVariant(
            label="metadata_source_answer_system",
            description="System-level source/path metadata answer rule.",
            system_prompt=metadata_source_answer,
        ),
        GeneratorPromptVariant(
            label="combined_strict_extractive",
            description=(
                "Combined recommendation candidate: strict source binding, no world "
                "knowledge, short extraction, and abstain only if evidence is absent."
            ),
            user_instruction=(
                "Beantworte RAG-Fragen strikt aus dem Kontext. Nutze kein Weltwissen "
                "und nicht das heutige Systemdatum. Extrahiere die kuerzeste passende "
                "Antwort aus der relevantesten Quelle. Bei Zahlen, Entitaeten, Dateien "
                "und Pfaden muss der Wert wortgleich im Kontext stehen. Antworte nur "
                "`Ich weiss es nicht.`, wenn der Kontext die Antwort nicht explizit "
                "enthaelt."
            ),
        ),
    ]


def _case_metadata_block(case: dict[str, Any]) -> str:
    lines = [
        f"Dataset: {case['dataset']}",
        f"Case-ID: {case['case_id']}",
    ]
    if case.get("gold_context_rank") is not None:
        lines.append(f"Gold-Kontext-Rang: {case['gold_context_rank']}")
    if case.get("short_or_numeric_answer"):
        lines.append("Antworttyp: kurz/numerisch")
    if case.get("expected_answer_type"):
        lines.append(f"Erwarteter Antworttyp: {case['expected_answer_type']}")
    if case.get("expected_failure_class"):
        lines.append(f"Failure-Class: {case['expected_failure_class']}")
    sources = []
    for chunk in case.get("context_chunks") or []:
        source = str(chunk.get("source") or "")
        if source and source not in sources:
            sources.append(source)
    if sources:
        lines.append("Quellenpfade: " + " | ".join(sources[:5]))
    return "\n".join(lines)


def _replay_context_text(chunks: list[dict[str, Any]]) -> str:
    from kiara.generation.chain import format_context

    ranked = sorted(chunks, key=lambda item: int(item.get("rank") or 0))
    return format_context(ranked)


def _build_user_message(case: dict[str, Any], variant: GeneratorPromptVariant) -> str:
    from kiara.generation.chain import _DEFAULT_RAG_TEMPLATE
    from kiara.shared.date_utils import get_german_date_string

    context = _replay_context_text(case["context_chunks"])
    question = str(case["question"])
    instruction = variant.user_instruction.strip()
    if instruction:
        question = f"{instruction}\n\nFrage: {question}"

    filled = re.sub(
        r"\{(context|question)\}",
        lambda match: {"context": context, "question": question}[match.group(1)],
        _DEFAULT_RAG_TEMPLATE,
    )
    metadata = _case_metadata_block(case)
    return f"{get_german_date_string()}\n\nAudit-Fall:\n{metadata}\n\n{filled}"


def _build_system_prompt(variant: GeneratorPromptVariant) -> str:
    from kiara.shared.prompt_defense import build_hardened_system_prompt

    return build_hardened_system_prompt(variant.system_prompt)


def _is_missing_answer(answer: str) -> bool:
    normalized = _normalize_answer_text(answer)
    return (
        crag._is_missing(answer)
        or any(pattern in normalized for pattern in _REPLAY_MISSING_PATTERNS)
        or golden._answer_status(
            {"evaluation_rule": "contains_any", "allowed_answer_variants": ["__never__"]},
            answer,
            "",
            generated=True,
        )
        == "missing"
    )


def _evaluate_case_answer(case: dict[str, Any], answer: str, error: str) -> str:
    if error:
        return "error"
    if case["dataset"] == "golden":
        golden_case = {
            "evaluation_rule": case.get("evaluation_rule") or "contains_any",
            "allowed_answer_variants": case.get("answers") or [],
        }
        return golden._answer_status(golden_case, answer, "", generated=True)

    answers = [str(value) for value in case.get("answers") or []]
    if _is_missing_answer(answer):
        return "missing"
    if crag._matches_gold(answer, answers):
        return "correct"
    return "incorrect"


def _normalize_case_baseline(case: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(case)
    source_status = str(case.get("baseline_status") or "")
    replay_status = _evaluate_case_answer(case, str(case.get("baseline_answer") or ""), "")
    normalized["baseline_status_source"] = source_status
    normalized["baseline_status"] = replay_status
    normalized["baseline_evaluator_version"] = EVALUATOR_VERSION
    normalized["baseline_status_changed_by_replay_evaluator"] = replay_status != source_status
    normalized["error_clusters"] = _cluster_case(
        dataset=str(normalized.get("dataset") or ""),
        baseline_status=replay_status,
        answers=[str(value) for value in normalized.get("answers") or []],
        question=str(normalized.get("question") or ""),
        prediction=str(normalized.get("baseline_answer") or ""),
        gold_rank=normalized.get("gold_context_rank"),
        warnings=[str(value) for value in normalized.get("warnings") or []],
        expected_failure_class=str(normalized.get("expected_failure_class") or ""),
    )
    return normalized


def _status_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("answer_status") or "") for row in rows)
    total = len(rows)
    correct = counts.get("correct", 0)
    missing = counts.get("missing", 0)
    incorrect = counts.get("incorrect", 0)
    return {
        "total": total,
        "correct": correct,
        "missing": missing,
        "incorrect": incorrect,
        "error": counts.get("error", 0),
        "correct_rate": correct / total if total else 0.0,
        "missing_rate": missing / total if total else 0.0,
        "incorrect_rate": incorrect / total if total else 0.0,
        "goldctx_bad_remaining": total - correct,
    }


def _baseline_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [{"answer_status": case["baseline_status"]} for case in cases]
    return _status_metrics(rows)


def _generation_kwargs(
    config: dict[str, Any], backend: Any, timeout: float | None, think: bool | None
):
    from kiara.shared.constants import DEFAULT_LLM_NUM_CTX

    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "think": think,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if getattr(backend, "provider", "") == "ollama":
        kwargs["num_ctx"] = int(
            config.get("ollama", {}).get("num_ctx")
            or config.get("llm", {}).get("num_ctx")
            or DEFAULT_LLM_NUM_CTX
        )
    return kwargs


def run_generator_prompt_audit(
    *,
    config_path: str | None = None,
    backend_id: int | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_label: str | None = None,
    golden_set_path: Path = DEFAULT_GOLDEN_SET,
    golden_baseline_details_path: Path = DEFAULT_GOLDEN_BASELINE_DETAILS,
    crag_audit_path: Path = DEFAULT_CRAG_AUDIT,
    crag_baseline_predictions_path: Path = DEFAULT_CRAG_BASELINE_PREDICTIONS,
    k: int = 20,
    context_excerpt_chars: int = DEFAULT_CONTEXT_EXCERPT_CHARS,
    max_cases: int | None = None,
    variant_labels: set[str] | None = None,
    timeout: float | None = 45.0,
    think: bool | None = False,
    dry_run: bool = False,
    backend_factory: Callable[[dict[str, Any], int | None], Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> GeneratorPromptAuditStats:
    """Run prompt variants against frozen GoldCtx-but-bad contexts."""

    from kiara.config import load_config
    from kiara.generation.backends.factory import create_backend

    label = _slug(run_label or f"sprint5_generator_prompt_audit_{_timestamp_for_filename()}")
    output_dir = output_root / label
    details_path = output_dir / "generator_prompt_audit_details.jsonl"
    summary_path = output_dir / "generator_prompt_audit_summary.json"
    cases_path = output_dir / "goldctx_bad_cases.jsonl"
    prompts_path = output_dir / "generator_prompt_audit_prompts.jsonl"
    report_path = output_dir / "generator_prompt_audit_report.md"

    config = load_config(config_path)
    cases = collect_goldctx_bad_cases(
        golden_set_path=golden_set_path,
        golden_baseline_details_path=golden_baseline_details_path,
        crag_audit_path=crag_audit_path,
        crag_baseline_predictions_path=crag_baseline_predictions_path,
        k=k,
        context_excerpt_chars=context_excerpt_chars,
    )
    cases = [_normalize_case_baseline(case) for case in cases]
    if max_cases is not None:
        cases = cases[:max_cases]
    variants = build_default_variants()
    if variant_labels:
        variants = [variant for variant in variants if variant.label in variant_labels]
    if not variants:
        raise ValueError("No prompt variants selected")

    backend = None
    if not dry_run:
        backend = (backend_factory or create_backend)(config, backend_id)

    started_at = _utc_timestamp()
    started = time.monotonic()
    details: list[dict[str, Any]] = []
    prompt_dumps: list[dict[str, Any]] = []
    for variant in variants:
        for index, case in enumerate(cases, 1):
            case_started = time.monotonic()
            error = ""
            answer = ""
            usage: dict[str, Any] = {}
            system_prompt = _build_system_prompt(variant)
            user_message = _build_user_message(case, variant)
            system_prompt_sha256 = _sha256_text(system_prompt)
            user_message_sha256 = _sha256_text(user_message)
            if dry_run:
                answer = ""
            else:
                try:
                    answer = str(
                        backend.generate(
                            system_prompt,
                            user_message,
                            **_generation_kwargs(config, backend, timeout, think),
                        )
                        or ""
                    )
                    usage = dict(getattr(backend, "last_usage", {}) or {})
                except Exception as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
            latency_ms = int((time.monotonic() - case_started) * 1000)
            status = "not_generated" if dry_run else _evaluate_case_answer(case, answer, error)
            row = {
                "run_label": label,
                "variant": asdict(variant),
                "case_index": index,
                "dataset": case["dataset"],
                "case_id": case["case_id"],
                "question": case["question"],
                "answers": case["answers"],
                "baseline_answer": case["baseline_answer"],
                "baseline_status": case["baseline_status"],
                "baseline_status_source": case.get("baseline_status_source"),
                "baseline_evaluator_version": case.get("baseline_evaluator_version"),
                "baseline_status_changed_by_replay_evaluator": bool(
                    case.get("baseline_status_changed_by_replay_evaluator")
                ),
                "baseline_latency_ms": case["baseline_latency_ms"],
                "answer": answer,
                "answer_status": status,
                "incorrect_indicator": status == "incorrect",
                "error": error,
                "evaluator_version": EVALUATOR_VERSION,
                "latency_ms": latency_ms,
                "usage": usage,
                "gold_context_metric": case["gold_context_metric"],
                "gold_context_rank": case["gold_context_rank"],
                "error_clusters": case["error_clusters"],
                "short_or_numeric_answer": case["short_or_numeric_answer"],
                "context_chunk_count": len(case["context_chunks"]),
                "top_sources": [chunk.get("source") for chunk in case["context_chunks"][:5]],
                "prompt_chars": len(system_prompt) + len(user_message),
                "prompt_hashes": {
                    "system_prompt_sha256": system_prompt_sha256,
                    "user_message_sha256": user_message_sha256,
                },
            }
            details.append(row)
            prompt_dumps.append(
                {
                    "run_label": label,
                    "variant_label": variant.label,
                    "dataset": case["dataset"],
                    "case_id": case["case_id"],
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "system_prompt_sha256": system_prompt_sha256,
                    "user_message_sha256": user_message_sha256,
                    "evaluator_version": EVALUATOR_VERSION,
                }
            )
            if progress_callback:
                progress_callback(
                    {
                        "variant": variant.label,
                        "current": index,
                        "total": len(cases),
                        "dataset": case["dataset"],
                        "case_id": case["case_id"],
                        "answer_status": status,
                        "latency_ms": latency_ms,
                        "error": error,
                    }
                )

    baseline = _baseline_metrics(cases)
    by_variant = {}
    for variant in variants:
        rows = [row for row in details if row["variant"]["label"] == variant.label]
        metrics = _status_metrics(rows)
        metrics["avg_latency_ms"] = (
            sum(float(row["latency_ms"]) for row in rows) / len(rows) if rows else 0.0
        )
        metrics["correct_case_ids"] = sorted(
            f"{row['dataset']}:{row['case_id']}"
            for row in rows
            if row["answer_status"] == "correct"
        )
        metrics["incorrect_case_ids"] = sorted(
            f"{row['dataset']}:{row['case_id']}"
            for row in rows
            if row["answer_status"] == "incorrect"
        )
        by_variant[variant.label] = metrics

    control_source = "artifact_baseline"
    control_label = "artifact_baseline"
    control_metrics = baseline
    if CONTROL_VARIANT_LABEL in by_variant:
        control_source = "same_run_variant"
        control_label = CONTROL_VARIANT_LABEL
        control_metrics = by_variant[CONTROL_VARIANT_LABEL]

    for label_item, metrics in by_variant.items():
        metrics["artifact_baseline_goldctx_bad_reduction"] = (
            baseline["goldctx_bad_remaining"] - metrics["goldctx_bad_remaining"]
        )
        metrics["artifact_baseline_correct_delta"] = metrics["correct"] - baseline["correct"]
        metrics["artifact_baseline_missing_delta"] = metrics["missing"] - baseline["missing"]
        metrics["artifact_baseline_incorrect_delta"] = metrics["incorrect"] - baseline["incorrect"]
        metrics["incorrect_rate_delta_vs_artifact_baseline"] = (
            metrics["incorrect_rate"] - baseline["incorrect_rate"]
        )
        metrics["goldctx_bad_reduction_vs_control"] = (
            control_metrics["goldctx_bad_remaining"] - metrics["goldctx_bad_remaining"]
        )
        metrics["correct_delta_vs_control"] = metrics["correct"] - control_metrics["correct"]
        metrics["missing_delta_vs_control"] = metrics["missing"] - control_metrics["missing"]
        metrics["incorrect_delta_vs_control"] = metrics["incorrect"] - control_metrics["incorrect"]
        metrics["incorrect_rate_delta_vs_control"] = (
            metrics["incorrect_rate"] - control_metrics["incorrect_rate"]
        )
        metrics["missing_rate_delta_vs_control"] = (
            metrics["missing_rate"] - control_metrics["missing_rate"]
        )
        metrics["control_label"] = control_label
        metrics["control_source"] = control_source
        metrics["goldctx_bad_reduction"] = metrics["goldctx_bad_reduction_vs_control"]
        metrics["is_primary_control"] = label_item == control_label

    cluster_counts = Counter()
    for case in cases:
        cluster_counts.update(case["error_clusters"])

    def _rank_variant(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float, float]:
        _label, metrics = item
        return (
            int(metrics["goldctx_bad_reduction_vs_control"]),
            float(metrics["correct_delta_vs_control"]),
            -float(metrics["goldctx_bad_remaining"]),
            -float(metrics["avg_latency_ms"]),
        )

    candidate_items = [
        (label_item, metrics)
        for label_item, metrics in by_variant.items()
        if label_item != control_label
    ]
    best_reduction_label = max(candidate_items, key=_rank_variant)[0] if candidate_items else None
    safe_candidates = {
        label_item: metrics
        for label_item, metrics in candidate_items
        if metrics["goldctx_bad_reduction_vs_control"] > 0
        and metrics["correct_delta_vs_control"] > 0
        and metrics["incorrect_rate_delta_vs_control"] <= 0
        and metrics["missing_delta_vs_control"] <= 0
    }
    best_label = max(safe_candidates.items(), key=_rank_variant)[0] if safe_candidates else None

    summary = {
        "schema": "kiara-generator-prompt-audit",
        "schema_version": 2,
        "created_at": _utc_timestamp(),
        "run": {
            "label": label,
            "started_at": started_at,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "config_path": config_path,
            "backend_id": backend_id,
            "backend_provider": getattr(backend, "provider", None) if backend is not None else None,
            "backend_model": getattr(backend, "model", None) if backend is not None else None,
            "dry_run": dry_run,
            "timeout": timeout,
            "think": think,
            "k": k,
            "context_excerpt_chars": context_excerpt_chars,
            "retrieval_changed": False,
            "productive_config_changed": False,
        },
        "evaluation": {
            "evaluator_version": EVALUATOR_VERSION,
            "artifact_baseline_statuses_replayed": True,
            "primary_control_label": control_label,
            "primary_control_source": control_source,
            "selection_policy": (
                "Recommend only variants with paired same-run correct gain, "
                "GoldCtx-bad reduction, no incorrect-rate increase, and no "
                "missing-regression versus the primary control."
            ),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
        },
        "case_selection": {
            "total": len(cases),
            "by_dataset": dict(Counter(case["dataset"] for case in cases)),
            "baseline": baseline,
            "baseline_status_changed_by_replay_evaluator": sum(
                1 for case in cases if case.get("baseline_status_changed_by_replay_evaluator")
            ),
            "error_cluster_counts": dict(sorted(cluster_counts.items())),
            "artifacts": {
                "golden_set": str(golden_set_path),
                "golden_baseline_details": str(golden_baseline_details_path),
                "crag_audit": str(crag_audit_path),
                "crag_baseline_predictions": str(crag_baseline_predictions_path),
            },
        },
        "variants": [asdict(variant) for variant in variants],
        "metrics_by_variant": by_variant,
        "recommendation": {
            "best_variant": best_label,
            "best_reduction_variant": best_reduction_label,
            "primary_control_label": control_label,
            "primary_control_source": control_source,
            "measurable_reduction": bool(
                best_label and by_variant[best_label]["goldctx_bad_reduction_vs_control"] > 0
            ),
            "correct_gain": bool(
                best_label and by_variant[best_label]["correct_delta_vs_control"] > 0
            ),
            "incorrect_not_increased": bool(
                best_label and by_variant[best_label]["incorrect_rate_delta_vs_control"] <= 0
            ),
            "missing_not_increased": bool(
                best_label and by_variant[best_label]["missing_delta_vs_control"] <= 0
            ),
            "hallucination_not_increased": bool(
                best_label and by_variant[best_label]["incorrect_rate_delta_vs_control"] <= 0
            ),
            "retrieval_followups": [
                "KGS-001/002/003/020 bleiben Retrieval/source/query-shape-Faelle.",
                "KGS-029/KGS-030 bleiben Tool/API-Routing-Faelle.",
                "CRAG-ColBERT erst neu bewerten, wenn der CRAG-ColBERT-Path nicht mehr empty ist.",
            ],
        },
        "artifacts": {
            "summary": str(summary_path),
            "details": str(details_path),
            "cases": str(cases_path),
            "prompts": str(prompts_path),
            "report": str(report_path),
        },
    }

    _write_jsonl(cases_path, cases)
    _write_jsonl(details_path, details)
    _write_jsonl(prompts_path, prompt_dumps)
    _write_json(summary_path, summary)
    _write_report(summary, cases, details, report_path)
    return GeneratorPromptAuditStats(
        label=label,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        details_path=str(details_path),
        report_path=str(report_path),
        cases=len(cases),
        variants=len(variants),
    )


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _write_report(
    summary: dict[str, Any],
    cases: list[dict[str, Any]],
    details: list[dict[str, Any]],
    report_path: Path,
) -> None:
    lines = [
        "# Sprint 5 Generator-/Prompt-Audit",
        "",
        f"Run: `{summary['run']['label']}`",
        f"Created: `{summary['created_at']}`",
        "",
        "## Scope",
        "",
        "- Retrieval was not called or changed; all cases use frozen context chunks from existing artefacts.",
        "- Golden basis: GoldCtx-but-bad rows from the Golden Set RAG baseline.",
        "- CRAG basis: Sprint-1/2 `generator_only` rows with `prompt_context_has_gold=true`.",
        "",
        "## Case Selection",
        "",
        f"- Total cases: `{summary['case_selection']['total']}`",
        f"- By dataset: `{summary['case_selection']['by_dataset']}`",
        f"- Baseline: `{summary['case_selection']['baseline']}`",
        f"- Error clusters: `{summary['case_selection']['error_cluster_counts']}`",
        "",
        "## Variant Metrics",
        "",
        "| Variant | Correct | Missing | Incorrect | Bad Remaining | Paired Reduction | Correct Delta | Missing Delta | Incorrect Rate | Delta | Avg ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in summary["metrics_by_variant"].items():
        lines.append(
            "| {label} | {correct} | {missing} | {incorrect} | {bad} | {reduction} | {correct_delta} | {missing_delta} | {ir} | {delta} | {ms:.1f} |".format(
                label=label,
                correct=metrics["correct"],
                missing=metrics["missing"],
                incorrect=metrics["incorrect"],
                bad=metrics["goldctx_bad_remaining"],
                reduction=metrics["goldctx_bad_reduction_vs_control"],
                correct_delta=metrics["correct_delta_vs_control"],
                missing_delta=metrics["missing_delta_vs_control"],
                ir=_format_rate(metrics["incorrect_rate"]),
                delta=_format_rate(metrics["incorrect_rate_delta_vs_control"]),
                ms=float(metrics["avg_latency_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "```json",
            json.dumps(summary["recommendation"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Cases",
            "",
        ]
    )
    for case in cases:
        lines.append(
            "- `{dataset}:{case_id}` baseline=`{status}` gold_rank=`{rank}` clusters=`{clusters}` question={question!r}".format(
                dataset=case["dataset"],
                case_id=case["case_id"],
                status=case["baseline_status"],
                rank=case.get("gold_context_rank"),
                clusters=",".join(case["error_clusters"]),
                question=case["question"][:180],
            )
        )
    lines.extend(["", "## Corrected Cases By Variant", ""])
    for label, metrics in summary["metrics_by_variant"].items():
        lines.append(f"- `{label}`: {metrics['correct_case_ids']}")
    lines.extend(["", "## Artefacts", ""])
    for key, value in summary["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")

    write_private_text(report_path, "\n".join(lines) + "\n")


def parse_variant_labels(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    labels = {item.strip() for item in raw.split(",") if item.strip()}
    return labels or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path")
    parser.add_argument("--backend-id", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-label")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument(
        "--golden-baseline-details", type=Path, default=DEFAULT_GOLDEN_BASELINE_DETAILS
    )
    parser.add_argument("--crag-audit", type=Path, default=DEFAULT_CRAG_AUDIT)
    parser.add_argument(
        "--crag-baseline-predictions", type=Path, default=DEFAULT_CRAG_BASELINE_PREDICTIONS
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--context-excerpt-chars", type=int, default=DEFAULT_CONTEXT_EXCERPT_CHARS)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--variants", help="Comma-separated variant labels to run")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--think", dest="think", action="store_true", default=False)
    parser.add_argument("--no-think", dest="think", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    def progress(event: dict[str, Any]) -> None:
        print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)

    stats = run_generator_prompt_audit(
        config_path=args.config_path,
        backend_id=args.backend_id,
        output_root=args.output_root,
        run_label=args.run_label,
        golden_set_path=args.golden_set,
        golden_baseline_details_path=args.golden_baseline_details,
        crag_audit_path=args.crag_audit,
        crag_baseline_predictions_path=args.crag_baseline_predictions,
        k=args.k,
        context_excerpt_chars=args.context_excerpt_chars,
        max_cases=args.max_cases,
        variant_labels=parse_variant_labels(args.variants),
        timeout=args.timeout,
        think=args.think,
        dry_run=args.dry_run,
        progress_callback=progress,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
