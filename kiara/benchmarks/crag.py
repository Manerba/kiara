"""CRAG benchmark harness for KIara.

The code in this module is intentionally independent from the Admin UI.  It
prepares CRAG task data as normal KIara documents, runs KIara's existing RAG
chain against a prepared manifest, and provides a lightweight local evaluator.
The official CRAG evaluator can still be used separately.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import math
import os
import re
import tarfile
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kiara.benchmarks.artifacts import write_private_text

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_GOLD_AND_METADATA_KEYS = {
    "interaction_id",
    "query_time",
    "domain",
    "question_type",
    "static_or_dynamic",
    "query",
    "answer",
    "alt_ans",
    "alternative_answers",
    "split",
    "popularity",
}

_STRUCTURED_HINTS = (
    "api",
    "kg",
    "knowledge",
    "mock",
    "entity",
    "entities",
    "graph",
    "table",
    "retrieval",
)

_MISSING_PATTERNS = (
    "i don't know",
    "i do not know",
    "ich weiss es nicht",
    "ich weiß es nicht",
    "keine relevanten dokumente gefunden",
    "keine information",
    "nicht gefunden",
    "nicht beantwortet",
    "nicht direkt beantwortet",
    "kann nicht beantwortet werden",
    "kann aus dem gegebenen kontext nicht beantwortet werden",
    "not enough information",
    "cannot find",
    "can't find",
)

_RETRIEVAL_DIAGNOSTIC_PATHS = ("semantic", "keyword", "colbert", "graph")


def _retrieval_diagnostic_field_schema() -> dict[str, Any]:
    return {
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
    }


@dataclass(frozen=True)
class PrepareStats:
    """Summary returned by CRAG preparation."""

    task: int
    records_seen: int
    records_written: int
    files_written: int
    records_without_documents: int
    manifest_path: str
    documents_dir: str


@dataclass(frozen=True)
class RunStats:
    """Summary returned by a KIara CRAG run."""

    task: int
    records_seen: int
    records_answered: int
    records_failed: int
    records_skipped: int
    output_path: str


class BenchmarkTimeoutError(TimeoutError):
    """Wall-clock timeout for a single benchmark record."""


def _run_with_wall_timeout(seconds: float | None, callback):
    """Run a callback with a POSIX wall-clock timeout in the main thread.

    Cloud SDKs may do internal retries and not honor our call-specific timeout
    kwargs. The benchmark harness needs a hard per-record guard so one slow
    model response cannot block a full run.
    """

    if seconds is None:
        return callback()

    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return callback()

    def _raise_timeout(_signum, _frame):
        raise BenchmarkTimeoutError(f"Benchmark record exceeded {seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return callback()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@dataclass(frozen=True)
class EvalStats:
    """Lightweight local evaluation summary."""

    total: int
    correct: int
    missing: int
    incorrect: int
    score: float
    accuracy: float
    missing_rate: float
    hallucination_rate: float


@dataclass(frozen=True)
class RetrievalEvalStats:
    """Retrieval-focused evaluation summary."""

    total: int
    with_context: int
    gold_answer_in_context: int
    gold_answer_in_context_at_1: int
    gold_answer_in_context_at_3: int
    gold_answer_in_context_at_5: int
    any_source_under_doc_dir: int
    all_sources_under_doc_dir: int
    gold_answer_in_context_rate: float
    gold_answer_in_context_at_1_rate: float
    gold_answer_in_context_at_3_rate: float
    gold_answer_in_context_at_5_rate: float
    any_source_under_doc_dir_rate: float
    all_sources_under_doc_dir_rate: float
    avg_context_chunks: float
    avg_latency_ms: float


@dataclass(frozen=True)
class OfficialEvalExportStats:
    """Summary for CRAG official-evaluator input export."""

    total: int
    output_path: str


@dataclass(frozen=True)
class OfficialJudgeEvalStats:
    """Summary for CRAG rubric LLM-judge evaluation."""

    total: int
    estimate_only: bool
    llm_judged: int
    correct: int
    missing: int
    incorrect: int
    parse_errors: int
    api_errors: int
    score: float
    accuracy: float
    missing_rate: float
    hallucination_rate: float
    estimated_prompt_tokens: int
    estimated_completion_tokens: int
    estimated_total_tokens: int
    actual_prompt_tokens: int
    actual_completion_tokens: int
    actual_total_tokens: int
    judge_backend_id: int | None
    judge_model: str
    output_path: str


DEFAULT_BREAKDOWN_FIELDS = (
    "task",
    "domain",
    "question_type",
    "static_or_dynamic",
    "popularity",
    "scope",
    "retrieval_mode",
    "embedding_model",
    "k",
)

DEFAULT_AUDIT_TOPK = 20
DEFAULT_PREPARATION_PAGE_LIMITS = (1, 3, 5, 10)
DEFAULT_JUDGE_MAX_TOKENS = 64
OFFICIAL_JUDGE_SYSTEM_PROMPT = """Assume you are a human expert grading CRAG benchmark predictions.
You are given a question, one or more accepted ground truth answers, and one model prediction.
Return JSON only with keys "score" and "explanation".

Rules:
1. The ground truth is always correct.
2. Score 1 only if the prediction fully answers the question and semantically matches at least one ground truth answer.
3. Score 1 for exact matches, accepted abbreviations, equivalent names, concise correct summaries, and near-identical rounded numeric values.
4. Score 0 if the prediction is unsure, refuses, does not answer the question, is self-contradictory, is incomplete, contains extra wrong items, or gives a different number/date/entity.
5. If the ground truth is a set of items, the prediction must contain the same items and no extra wrong items.

Output example: {"score": 1, "explanation": "The prediction matches the ground truth."}
"""


def _safe_name(value: str, fallback: str) -> str:
    raw = (value or "").strip() or fallback
    safe = _SAFE_NAME_RE.sub("_", raw).strip("._")
    return safe[:120] or fallback


def _stable_id(record: dict[str, Any], index: int) -> str:
    raw = str(record.get("interaction_id") or "").strip()
    if raw:
        return _safe_name(raw, f"record_{index:06d}")
    digest = hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"record_{index:06d}_{digest}"


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _iter_json_records_from_text(text: str, source_name: str) -> Iterator[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return

    if source_name.endswith(".jsonl") or source_name.endswith(".jsonl.bz2"):
        try:
            for line_no, line in enumerate(stripped.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
                else:
                    raise ValueError(f"{source_name}:{line_no} enthaelt kein JSON-Objekt")
        except json.JSONDecodeError:
            yield from _iter_json_object_stream(stripped, source_name)
        return

    loaded = json.loads(stripped)
    if isinstance(loaded, list):
        for item in loaded:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(loaded, dict):
        for key in ("data", "examples", "records", "items"):
            maybe_records = loaded.get(key)
            if isinstance(maybe_records, list):
                for item in maybe_records:
                    if isinstance(item, dict):
                        yield item
                return
        yield loaded
        return
    raise ValueError(f"{source_name} enthaelt kein unterstuetztes JSON-Format")


def _iter_json_object_stream(text: str, source_name: str) -> Iterator[dict[str, Any]]:
    """Yield whitespace-separated JSON objects from pretty-printed JSONL files."""

    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        item, index = decoder.raw_decode(text, index)
        if isinstance(item, dict):
            yield item
        else:
            raise ValueError(f"{source_name} enthaelt kein JSON-Objekt")


def _read_text_file(path: Path) -> str:
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".bz2"):
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    if suffixes.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def iter_crag_records(input_path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield CRAG records from a file, directory or tar archive.

    Supported files: ``.json``, ``.jsonl``, ``.json.bz2``, ``.jsonl.bz2``,
    ``.json.gz`` and ``.jsonl.gz``.  Tar archives are scanned for JSON/JSONL
    members.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        patterns = ("*.json", "*.jsonl", "*.json.bz2", "*.jsonl.bz2", "*.json.gz", "*.jsonl.gz")
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend(path.rglob(pattern))
        for candidate in sorted(set(candidates)):
            yield from iter_crag_records(candidate)
        return

    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if not (
                    name.endswith(".json")
                    or name.endswith(".jsonl")
                    or name.endswith(".json.bz2")
                    or name.endswith(".jsonl.bz2")
                    or name.endswith(".json.gz")
                    or name.endswith(".jsonl.gz")
                ):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                raw = handle.read()
                if name.endswith(".bz2"):
                    text = bz2.decompress(raw).decode("utf-8")
                elif name.endswith(".gz"):
                    text = gzip.decompress(raw).decode("utf-8")
                else:
                    text = raw.decode("utf-8")
                yield from _iter_json_records_from_text(text, name)
        return

    yield from _iter_json_records_from_text(_read_text_file(path), path.name)


def _answer_values(record: dict[str, Any]) -> list[str]:
    def append_value(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                append_value(item)
            return
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                append_value(decoded)
                return
        answers.append(text)

    answers: list[str] = []
    raw_answer = record.get("answer")
    append_value(raw_answer)
    for alt_key in ("alt_ans", "alternative_answers"):
        append_value(record.get(alt_key))
    seen: set[str] = set()
    unique = []
    for answer in answers:
        key = answer.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(answer)
    return unique


def _manifest_entry(
    record: dict[str, Any],
    task: int,
    interaction_id: str,
    doc_dir: Path,
    files: list[Path],
) -> dict[str, Any]:
    return {
        "interaction_id": interaction_id,
        "task": task,
        "query": record.get("query", ""),
        "answers": _answer_values(record),
        "domain": record.get("domain", ""),
        "question_type": record.get("question_type", ""),
        "static_or_dynamic": record.get("static_or_dynamic", ""),
        "query_time": record.get("query_time", ""),
        "popularity": record.get("popularity", ""),
        "split": record.get("split"),
        "doc_dir": str(doc_dir),
        "files": [str(path) for path in files],
        "file_count": len(files),
    }


def _write_web_documents(
    record: dict[str, Any],
    record_dir: Path,
    max_pages: int | None,
) -> list[Path]:
    files: list[Path] = []
    search_results = record.get("search_results") or []
    if not isinstance(search_results, list):
        return files
    for idx, page in enumerate(search_results[:max_pages] if max_pages else search_results, 1):
        if not isinstance(page, dict):
            continue
        html = page.get("page_result") or page.get("html") or ""
        snippet = page.get("page_snippet") or page.get("snippet") or ""
        if html and str(html).strip():
            suffix = ".html"
            body = str(html)
        elif snippet and str(snippet).strip():
            suffix = ".md"
            body = "\n".join(
                part
                for part in (
                    f"# {page.get('page_name', f'Page {idx}')}",
                    f"URL: {page.get('page_url', '')}",
                    "",
                    str(snippet),
                )
                if part is not None
            )
        else:
            continue
        out = record_dir / f"web_{idx:03d}{suffix}"
        out.write_text(body, encoding="utf-8")
        files.append(out)
    return files


def _structured_payloads(record: dict[str, Any]) -> list[tuple[str, Any]]:
    payloads: list[tuple[str, Any]] = []
    for key, value in sorted(record.items()):
        if key in _GOLD_AND_METADATA_KEYS or key == "search_results":
            continue
        if value in (None, "", [], {}):
            continue
        lower_key = key.lower()
        if any(hint in lower_key for hint in _STRUCTURED_HINTS):
            payloads.append((key, value))
    return payloads


def _write_structured_documents(record: dict[str, Any], record_dir: Path) -> list[Path]:
    files: list[Path] = []
    for idx, (key, value) in enumerate(_structured_payloads(record), 1):
        out = record_dir / f"structured_{idx:03d}_{_safe_name(key, 'data')}.json"
        out.write_text(_json_dumps({"field": key, "value": value}), encoding="utf-8")
        files.append(out)
    return files


def prepare_crag_task(
    input_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    task: int,
    *,
    max_records: int | None = None,
    max_pages: int | None = None,
    overwrite: bool = False,
) -> PrepareStats:
    """Prepare a CRAG task as KIara-indexable documents plus a manifest."""

    if task not in (1, 2, 3):
        raise ValueError("task muss 1, 2 oder 3 sein")

    base = Path(output_dir)
    documents_dir = base / "documents" / f"task{task}"
    manifest_dir = base / "manifests"
    manifest_path = manifest_dir / f"task{task}.jsonl"

    if overwrite and documents_dir.exists():
        import shutil

        shutil.rmtree(documents_dir)
    if overwrite and manifest_path.exists():
        manifest_path.unlink()

    documents_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    records_seen = 0
    records_written = 0
    files_written = 0
    records_without_documents = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, record in enumerate(iter_crag_records(input_path), 1):
            if max_records is not None and records_seen >= max_records:
                break
            records_seen += 1
            interaction_id = _stable_id(record, index)
            record_dir = documents_dir / interaction_id
            record_dir.mkdir(parents=True, exist_ok=True)

            files: list[Path] = []
            if task in (1, 3):
                files.extend(_write_web_documents(record, record_dir, max_pages=max_pages))
            if task in (2, 3):
                files.extend(_write_structured_documents(record, record_dir))

            if not files:
                records_without_documents += 1

            entry = _manifest_entry(record, task, interaction_id, record_dir, files)
            manifest.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            records_written += 1
            files_written += len(files)

    return PrepareStats(
        task=task,
        records_seen=records_seen,
        records_written=records_written,
        files_written=files_written,
        records_without_documents=records_without_documents,
        manifest_path=str(manifest_path),
        documents_dir=str(documents_dir),
    )


def iter_manifest(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} enthaelt kein JSON-Objekt")
            yield item


def run_crag_task(
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    task: int,
    *,
    config_path: str | None = None,
    backend_id: int | None = None,
    embedding_model: str | None = None,
    k: int = 5,
    max_records: int | None = None,
    scope: str = "per-question",
    context_excerpt_chars: int = 2000,
    resume: bool = False,
    llm_timeout: float | None = None,
    think: bool | None = None,
) -> RunStats:
    """Run KIara RAG on a prepared CRAG manifest."""

    if task not in (1, 2, 3):
        raise ValueError("task muss 1, 2 oder 3 sein")
    if scope not in ("per-question", "global"):
        raise ValueError("scope muss 'per-question' oder 'global' sein")
    if llm_timeout is not None and llm_timeout <= 0:
        raise ValueError("llm_timeout muss groesser 0 sein")

    from kiara.config import load_config
    from kiara.generation.backends.factory import create_backend
    from kiara.generation.chain import RAGChain

    config = load_config(config_path)
    effective_embedding_model = embedding_model or str(
        config.get("embedding", {}).get("model")
        or config.get("ollama", {}).get("embedding_model")
        or ""
    )
    retrieval_mode = str(config.get("retrieval", {}).get("hybrid_search", "none"))

    def _build_chain() -> RAGChain:
        backend = create_backend(config, backend_id=backend_id)
        if llm_timeout is not None:
            raw_generate = backend.generate

            def _generate_with_timeout(system_prompt: str, user_message: str, **kwargs) -> str:
                kwargs.setdefault("timeout", llm_timeout)
                return raw_generate(system_prompt, user_message, **kwargs)

            backend.generate = _generate_with_timeout
        return RAGChain(config, backend=backend, embedding_model=embedding_model)

    chain = _build_chain()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    if resume and out.exists():
        completed_ids = {
            str(row.get("interaction_id") or "")
            for row in iter_manifest(out)
            if row.get("interaction_id")
        }

    records_seen = 0
    records_answered = 0
    records_failed = 0
    records_skipped = 0

    write_mode = "a" if resume else "w"
    with out.open(write_mode, encoding="utf-8") as handle:
        for entry in iter_manifest(manifest_path):
            interaction_id = str(entry.get("interaction_id") or "")
            if resume and interaction_id in completed_ids:
                records_skipped += 1
                continue
            if max_records is not None and records_seen >= max_records:
                break
            records_seen += 1
            started = time.monotonic()
            query = str(entry.get("query") or "")
            row = {
                "interaction_id": interaction_id,
                "task": task,
                "query": query,
                "answers": entry.get("answers") or [],
                "domain": entry.get("domain", ""),
                "question_type": entry.get("question_type", ""),
                "static_or_dynamic": entry.get("static_or_dynamic", ""),
                "query_time": entry.get("query_time", ""),
                "popularity": entry.get("popularity", ""),
                "split": entry.get("split"),
                "file_count": entry.get("file_count", 0),
                "doc_dir": entry.get("doc_dir", ""),
                "short_or_numeric_answer": _has_low_confidence_literal_answer(
                    [str(v) for v in entry.get("answers") or []]
                ),
                "run_scope": scope,
                "retrieval_mode": retrieval_mode,
                "embedding_model": effective_embedding_model,
                "retrieval_k": k,
                "backend_id": backend_id,
                "context_excerpt_chars": context_excerpt_chars,
                "llm_timeout": llm_timeout,
                "think": think,
            }
            try:
                filter_dir = entry.get("doc_dir") if scope == "per-question" else None

                def _query_current_chain(
                    current_chain=chain,
                    current_query=query,
                    current_filter_dir=filter_dir,
                ):
                    return current_chain.query(
                        question=current_query,
                        k=k,
                        filter_dir=current_filter_dir,
                        think=think,
                    )

                result = _run_with_wall_timeout(
                    llm_timeout,
                    _query_current_chain,
                )
                row.update(
                    {
                        "prediction": result.get("answer", ""),
                        "sources": result.get("sources", []),
                        "context_chunks": _compact_context_chunks(
                            result.get("context_chunks", []),
                            excerpt_chars=context_excerpt_chars,
                        ),
                        "retrieval_diagnostics": result.get("retrieval_diagnostics")
                        or getattr(chain, "last_retrieval_diagnostics", None),
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "error": "",
                    }
                )
                records_answered += 1
            except Exception as exc:
                from kiara.retrieval.index_compatibility import IndexCompatibilityError

                if isinstance(exc, IndexCompatibilityError):
                    raise
                if isinstance(exc, BenchmarkTimeoutError):
                    chain = _build_chain()
                row.update(
                    {
                        "prediction": "",
                        "sources": [],
                        "context_chunks": [],
                        "retrieval_diagnostics": getattr(chain, "last_retrieval_diagnostics", None),
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                records_failed += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    return RunStats(
        task=task,
        records_seen=records_seen,
        records_answered=records_answered,
        records_failed=records_failed,
        records_skipped=records_skipped,
        output_path=str(out),
    )


def _compact_context_chunks(
    chunks: Iterable[dict[str, Any]], excerpt_chars: int = 2000
) -> list[dict]:
    compact = []
    for rank, chunk in enumerate(chunks, 1):
        text = str(chunk.get("text") or "")
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        chunk_id = str(
            chunk.get("chunk_id")
            or chunk.get("point_id")
            or chunk.get("_doc_id")
            or metadata.get("chunk_id")
            or ""
        )
        item = {
            "rank": rank,
            "final_context_rank": chunk.get("final_context_rank", rank),
            "chunk_id": chunk_id,
            "point_id": str(chunk.get("point_id") or chunk_id),
            "source": metadata.get("source", ""),
            "metadata": dict(metadata),
            "chunk_index": metadata.get("chunk_index"),
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
            "rerank_score": chunk.get("rerank_score"),
            "excerpt": text[:excerpt_chars],
            "text_chars": len(text),
        }
        compact.append(item)
    return compact


def _normalize_answer(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _answer_in_text(text: str, answer: str) -> bool:
    normalized_text = _normalize_answer(text)
    normalized_answer = _normalize_answer(answer)
    if not normalized_text or not normalized_answer:
        return False
    if normalized_text == normalized_answer:
        return True
    if len(normalized_answer) <= 3:
        pattern = rf"(?<!\w){re.escape(normalized_answer)}(?!\w)"
        return re.search(pattern, normalized_text) is not None
    return normalized_answer in normalized_text


def _is_missing(prediction: str) -> bool:
    lowered = prediction.casefold()
    return not lowered.strip() or any(pattern in lowered for pattern in _MISSING_PATTERNS)


def _matches_gold(prediction: str, answers: list[str]) -> bool:
    normalized_prediction = _normalize_answer(prediction)
    if not normalized_prediction:
        return False
    return any(_answer_in_text(normalized_prediction, answer) for answer in answers)


def evaluate_predictions(
    predictions_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> EvalStats:
    """Lightweight local evaluator.

    This is not the official CRAG LLM evaluator.  It is a fast smoke metric:
    exact/contains match against gold answers, plus missing-answer detection.
    """

    total = correct = missing = incorrect = 0
    details: list[dict[str, Any]] = []
    for row in iter_manifest(predictions_path):
        total += 1
        prediction = str(row.get("prediction") or "")
        answers = [str(v) for v in row.get("answers") or []]
        if _matches_gold(prediction, answers):
            label = "correct"
            correct += 1
        elif _is_missing(prediction):
            label = "missing"
            missing += 1
        else:
            label = "incorrect"
            incorrect += 1
        details.append(
            {
                "interaction_id": row.get("interaction_id", ""),
                "label": label,
                "prediction": prediction,
                "answers": answers,
            }
        )

    if total:
        score = (2 * correct + missing) / total - 1
        accuracy = correct / total
        missing_rate = missing / total
        hallucination_rate = incorrect / total
    else:
        score = accuracy = missing_rate = hallucination_rate = 0.0

    stats = EvalStats(
        total=total,
        correct=correct,
        missing=missing,
        incorrect=incorrect,
        score=score,
        accuracy=accuracy,
        missing_rate=missing_rate,
        hallucination_rate=hallucination_rate,
    )

    if output_path:
        payload = {"summary": asdict(stats), "details": details}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")

    return stats


def export_official_eval_payload(
    predictions_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> OfficialEvalExportStats:
    """Export KIara predictions as CRAG local-evaluation inputs.

    The upstream CRAG local evaluator passes three sequences to
    ``evaluate_predictions``: ``queries``, ``ground_truths`` and
    ``predictions``.  We also include ``ground_truths_list`` and metadata so
    adapters can preserve alternate answers and trace records back to KIara.
    """

    queries: list[str] = []
    ground_truths: list[str] = []
    ground_truths_list: list[list[str]] = []
    predictions: list[str] = []
    interaction_ids: list[str] = []
    metadata: list[dict[str, Any]] = []

    for row in iter_manifest(predictions_path):
        answers = [str(v) for v in row.get("answers") or []]
        queries.append(str(row.get("query") or ""))
        ground_truths.append(answers[0] if answers else "")
        ground_truths_list.append(answers)
        predictions.append(str(row.get("prediction") or ""))
        interaction_ids.append(str(row.get("interaction_id") or ""))
        metadata.append(
            {
                "interaction_id": row.get("interaction_id", ""),
                "task": row.get("task"),
                "domain": row.get("domain", ""),
                "question_type": row.get("question_type", ""),
                "static_or_dynamic": row.get("static_or_dynamic", ""),
                "query_time": row.get("query_time", ""),
                "popularity": row.get("popularity", ""),
                "split": row.get("split"),
                "latency_ms": row.get("latency_ms"),
                "error": row.get("error", ""),
            }
        )

    payload = {
        "schema": "kiara-crag-official-local-evaluation-inputs",
        "schema_version": 1,
        "queries": queries,
        "ground_truths": ground_truths,
        "ground_truths_list": ground_truths_list,
        "predictions": predictions,
        "interaction_ids": interaction_ids,
        "metadata": metadata,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")
    return OfficialEvalExportStats(total=len(predictions), output_path=str(out))


def _estimate_tokens(text: str) -> int:
    from kiara.shared.constants import CHARS_PER_TOKEN

    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def _load_official_judge_prompt(prompt_path: str | os.PathLike[str] | None) -> str:
    if not prompt_path:
        return OFFICIAL_JUDGE_SYSTEM_PROMPT
    prompt = Path(prompt_path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Judge-Prompt-Datei ist leer")
    return prompt


def _trim_prediction_for_judge(prediction: str, token_limit: int | None) -> str:
    text = " ".join(str(prediction or "").split())
    if token_limit is None or token_limit <= 0:
        return text
    words = text.split()
    if len(words) <= token_limit:
        return text
    return " ".join(words[:token_limit])


def _official_exact_or_invalid_match(prediction: str, answers: list[str]) -> bool:
    normalized_prediction = _normalize_answer(prediction)
    if not normalized_prediction:
        return False
    for answer in answers:
        normalized_answer = _normalize_answer(answer)
        if normalized_prediction == normalized_answer:
            return True
        if "invalid" in normalized_prediction and "invalid" in normalized_answer:
            return True
    return False


def _official_judge_user_message(query: str, answers: list[str], prediction: str) -> str:
    payload = {
        "question": query,
        "ground_truth_answers": answers,
        "prediction": prediction,
    }
    return _json_dumps(payload)


def estimate_official_judge_tokens(
    predictions_path: str | os.PathLike[str],
    *,
    max_records: int | None = None,
    prompt_path: str | os.PathLike[str] | None = None,
    prediction_token_limit: int | None = 75,
    max_completion_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
) -> dict[str, int]:
    """Estimate token usage for CRAG LLM-judge evaluation.

    The estimate mirrors the records that would actually call a judge.  Exact
    matches and missing/abstained predictions are scored locally like the CRAG
    evaluator and therefore do not consume judge tokens.
    """

    system_prompt = _load_official_judge_prompt(prompt_path)
    total = judged = exact = missing = 0
    prompt_tokens = completion_tokens = 0

    for row in iter_manifest(predictions_path):
        if max_records is not None and total >= max_records:
            break
        total += 1
        answers = [str(v) for v in row.get("answers") or []]
        prediction = _trim_prediction_for_judge(
            str(row.get("prediction") or ""),
            prediction_token_limit,
        )
        if _official_exact_or_invalid_match(prediction, answers):
            exact += 1
            continue
        if _is_missing(prediction):
            missing += 1
            continue

        user_message = _official_judge_user_message(
            str(row.get("query") or ""),
            answers,
            prediction,
        )
        judged += 1
        prompt_tokens += _estimate_tokens(system_prompt) + _estimate_tokens(user_message)
        completion_tokens += max_completion_tokens

    return {
        "total": total,
        "llm_judged": judged,
        "local_exact": exact,
        "local_missing": missing,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimated_total_tokens": prompt_tokens + completion_tokens,
    }


def _strip_json_fence(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def parse_official_judge_response(response: str) -> tuple[int | None, str, str]:
    """Parse CRAG judge JSON into ``(score, explanation, error)``."""

    text = _strip_json_fence(response)
    try:
        parsed = json.loads(text)
        raw_score = parsed.get("score")
        score = int(raw_score)
        if score not in (0, 1):
            return None, str(parsed.get("explanation") or ""), f"bad score: {raw_score!r}"
        return score, str(parsed.get("explanation") or ""), ""
    except Exception:
        pass

    matches = re.findall(r"{([^}]*)}", text, flags=re.DOTALL)
    candidate = "{" + matches[-1] + "}" if matches else text
    score_match = re.search(r'"score"\s*:\s*"?([01])"?', candidate)
    if not score_match:
        return None, "", "score not found"

    explanation_match = re.search(
        r'"explanation"\s*:\s*"(.+?)"\s*(?:,|})',
        candidate,
        flags=re.DOTALL,
    )
    explanation = explanation_match.group(1) if explanation_match else candidate
    return int(score_match.group(1)), explanation, ""


def evaluate_predictions_with_official_judge(
    predictions_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    config_path: str | None = None,
    backend_id: int | None = None,
    max_records: int | None = None,
    prompt_path: str | os.PathLike[str] | None = None,
    prediction_token_limit: int | None = 75,
    max_completion_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    temperature: float = 0.0,
    judge_timeout: float | None = 120.0,
    reasoning_effort: str | None = "minimal",
    estimate_only: bool = False,
) -> OfficialJudgeEvalStats:
    """Evaluate CRAG predictions with a configurable LLM judge.

    This uses the CRAG score formula and a CRAG-rubric judge prompt, but routes
    calls through KIara's backend registry instead of hardcoding OpenAI or an
    obsolete upstream evaluation model.
    """

    from kiara.config import load_config

    if judge_timeout is not None and judge_timeout <= 0:
        raise ValueError("judge_timeout muss groesser 0 sein")

    system_prompt = _load_official_judge_prompt(prompt_path)
    estimate = estimate_official_judge_tokens(
        predictions_path,
        max_records=max_records,
        prompt_path=prompt_path,
        prediction_token_limit=prediction_token_limit,
        max_completion_tokens=max_completion_tokens,
    )

    backend = None
    judge_model = ""
    if not estimate_only:
        from kiara.generation.backends.factory import create_backend

        config = load_config(config_path)
        backend = create_backend(config, backend_id=backend_id)
        judge_model = backend.model

    total = correct = missing = incorrect = parse_errors = api_errors = llm_judged = 0
    actual_prompt_tokens = actual_completion_tokens = 0
    details: list[dict[str, Any]] = []

    for row in iter_manifest(predictions_path):
        if max_records is not None and total >= max_records:
            break
        total += 1
        interaction_id = str(row.get("interaction_id") or "")
        query = str(row.get("query") or "")
        answers = [str(v) for v in row.get("answers") or []]
        original_prediction = str(row.get("prediction") or "")
        prediction = _trim_prediction_for_judge(original_prediction, prediction_token_limit)

        label = "incorrect"
        raw_response = ""
        explanation = ""
        parse_error = ""
        api_error = ""
        used_llm = False

        if _official_exact_or_invalid_match(prediction, answers):
            label = "correct"
            correct += 1
            explanation = "Exact or invalid-marker match."
        elif _is_missing(prediction):
            label = "missing"
            missing += 1
            explanation = "Prediction is empty or abstains."
        elif estimate_only:
            used_llm = True
            label = "pending_judge"
            explanation = "Estimate-only run; no judge call made."
        else:
            used_llm = True
            llm_judged += 1
            user_message = _official_judge_user_message(query, answers, prediction)
            started = time.monotonic()
            try:
                assert backend is not None

                def _call_judge(current_user_message=user_message) -> str:
                    return backend.generate(
                        system_prompt,
                        current_user_message,
                        temperature=temperature,
                        max_tokens=max_completion_tokens,
                        think=False,
                        reasoning_effort=reasoning_effort,
                    )

                raw_response = _run_with_wall_timeout(
                    judge_timeout,
                    _call_judge,
                )
                usage = backend.last_usage or {}
                actual_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                actual_completion_tokens += int(usage.get("response_tokens") or 0)
                score, explanation, parse_error = parse_official_judge_response(raw_response)
                if parse_error or score is None:
                    parse_errors += 1
                    incorrect += 1
                    label = "incorrect"
                elif score == 1:
                    correct += 1
                    label = "correct"
                else:
                    incorrect += 1
                    label = "incorrect"
            except Exception as exc:
                api_errors += 1
                incorrect += 1
                api_error = f"{type(exc).__name__}: {exc}"
                explanation = api_error
            latency_ms = int((time.monotonic() - started) * 1000)
            details.append(
                {
                    "interaction_id": interaction_id,
                    "query": query,
                    "answers": answers,
                    "prediction": original_prediction,
                    "prediction_judged": prediction,
                    "label": label,
                    "explanation": explanation,
                    "raw_judge_response": raw_response,
                    "parse_error": parse_error,
                    "api_error": api_error,
                    "llm_judged": used_llm,
                    "latency_ms": latency_ms,
                }
            )
            continue

        details.append(
            {
                "interaction_id": interaction_id,
                "query": query,
                "answers": answers,
                "prediction": original_prediction,
                "prediction_judged": prediction,
                "label": label,
                "explanation": explanation,
                "raw_judge_response": raw_response,
                "parse_error": parse_error,
                "api_error": api_error,
                "llm_judged": used_llm,
                "latency_ms": 0,
            }
        )

    if estimate_only:
        llm_judged = estimate["llm_judged"]

    if estimate_only:
        score = accuracy = missing_rate = hallucination_rate = 0.0
    elif total:
        score = (2 * correct + missing) / total - 1
        accuracy = correct / total
        missing_rate = missing / total
        hallucination_rate = incorrect / total
    else:
        score = accuracy = missing_rate = hallucination_rate = 0.0

    stats = OfficialJudgeEvalStats(
        total=total,
        estimate_only=estimate_only,
        llm_judged=llm_judged,
        correct=correct,
        missing=missing,
        incorrect=incorrect,
        parse_errors=parse_errors,
        api_errors=api_errors,
        score=score,
        accuracy=accuracy,
        missing_rate=missing_rate,
        hallucination_rate=hallucination_rate,
        estimated_prompt_tokens=estimate["estimated_prompt_tokens"],
        estimated_completion_tokens=estimate["estimated_completion_tokens"],
        estimated_total_tokens=estimate["estimated_total_tokens"],
        actual_prompt_tokens=actual_prompt_tokens,
        actual_completion_tokens=actual_completion_tokens,
        actual_total_tokens=actual_prompt_tokens + actual_completion_tokens,
        judge_backend_id=backend_id,
        judge_model=judge_model,
        output_path=str(output_path),
    )

    payload = {
        "schema": "kiara-crag-official-judge-results",
        "schema_version": 1,
        "summary": asdict(stats),
        "details": details,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")
    return stats


def _context_contains_answer(
    chunks: list[dict[str, Any]],
    answers: list[str],
    *,
    top_k: int | None = None,
) -> bool:
    selected = chunks[:top_k] if top_k is not None else chunks
    for chunk in selected:
        text = str(chunk.get("excerpt") or "")
        if any(_answer_in_text(text, answer) for answer in answers):
            return True
    return False


def _chunk_text_for_gold_match(chunk: dict[str, Any]) -> str:
    return str(chunk.get("excerpt") or chunk.get("text") or "")


def _first_gold_hit(
    hits: list[dict[str, Any]],
    answers: list[str],
    *,
    rank_key: str = "rank",
) -> dict[str, Any] | None:
    for index, hit in enumerate(hits, 1):
        text = _chunk_text_for_gold_match(hit)
        if any(_answer_in_text(text, answer) for answer in answers):
            rank = hit.get(rank_key)
            if rank is None:
                rank = index
            return {
                "rank": rank,
                "chunk_id": hit.get("chunk_id") or hit.get("point_id") or "",
                "point_id": hit.get("point_id") or hit.get("chunk_id") or "",
                "source": hit.get("source") or (hit.get("metadata") or {}).get("source", ""),
                "score": hit.get("score"),
                "rrf_rank": hit.get("rrf_rank"),
                "rrf_score": hit.get("rrf_score"),
                "rrf_contributions": dict(hit.get("rrf_contributions") or {}),
                "retriever_paths": list(hit.get("retriever_paths") or []),
            }
    return None


def _iter_retrieval_diagnostic_variants(row: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = row.get("retrieval_diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    variants = diagnostics.get("variants")
    if isinstance(variants, list):
        return [item for item in variants if isinstance(item, dict)]
    if diagnostics.get("schema") == "kiara-retrieval-path-diagnostics":
        return [diagnostics]
    return []


def _min_rank(current: int | None, candidate: Any) -> int | None:
    try:
        value = int(candidate)
    except (TypeError, ValueError):
        return current
    if current is None or value < current:
        return value
    return current


def _retrieval_path_gold_diagnostics(
    row: dict[str, Any],
    chunks: list[dict[str, Any]],
    answers: list[str],
) -> dict[str, Any]:
    ranks: dict[str, int | None] = {
        "semantic_rank": None,
        "keyword_rank": None,
        "colbert_rank": None,
        "graph_rank": None,
        "rrf_rank": None,
        "final_context_rank": None,
    }
    scores: dict[str, Any] = {
        "semantic_score": None,
        "keyword_score": None,
        "colbert_score": None,
        "graph_score": None,
        "rrf_score": None,
    }
    states: dict[str, str] = {}
    latencies: dict[str, int | None] = {}
    hits: dict[str, dict[str, Any] | None] = {
        "semantic": None,
        "keyword": None,
        "colbert": None,
        "graph": None,
        "rrf": None,
        "final_context": None,
    }

    context_hit = _first_gold_hit(chunks, answers, rank_key="final_context_rank")
    if context_hit is not None:
        ranks["final_context_rank"] = _min_rank(
            ranks["final_context_rank"], context_hit.get("rank")
        )
        ranks["rrf_rank"] = _min_rank(ranks["rrf_rank"], context_hit.get("rrf_rank"))
        scores["rrf_score"] = context_hit.get("rrf_score")
        hits["final_context"] = context_hit
        hits["rrf"] = context_hit
        for path in _RETRIEVAL_DIAGNOSTIC_PATHS:
            path_rank = next(
                (
                    chunk.get(f"{path}_rank")
                    for chunk in chunks
                    if any(_answer_in_text(_chunk_text_for_gold_match(chunk), a) for a in answers)
                ),
                None,
            )
            ranks[f"{path}_rank"] = _min_rank(ranks[f"{path}_rank"], path_rank)
            if scores[f"{path}_score"] is None:
                scores[f"{path}_score"] = next(
                    (
                        chunk.get(f"{path}_score")
                        for chunk in chunks
                        if any(
                            _answer_in_text(_chunk_text_for_gold_match(chunk), a) for a in answers
                        )
                    ),
                    None,
                )

    for variant in _iter_retrieval_diagnostic_variants(row):
        paths = variant.get("paths") if isinstance(variant.get("paths"), dict) else {}
        for path in _RETRIEVAL_DIAGNOSTIC_PATHS:
            path_info = paths.get(path) if isinstance(paths.get(path), dict) else {}
            if path_info:
                states.setdefault(path, str(path_info.get("status") or ""))
                latency = path_info.get("latency_ms")
                if isinstance(latency, int | float):
                    latencies.setdefault(path, int(latency))
            path_hits = (
                [hit for hit in path_info.get("hits", []) if isinstance(hit, dict)]
                if path_info
                else []
            )
            path_hit = _first_gold_hit(path_hits, answers)
            if path_hit is None:
                continue
            ranks[f"{path}_rank"] = _min_rank(ranks[f"{path}_rank"], path_hit["rank"])
            if scores[f"{path}_score"] is None:
                scores[f"{path}_score"] = path_hit.get("score")
            hits[path] = path_hit

        rrf_info = variant.get("rrf") if isinstance(variant.get("rrf"), dict) else {}
        if rrf_info:
            states.setdefault("rrf", str(rrf_info.get("status") or ""))
            latency = rrf_info.get("latency_ms")
            if isinstance(latency, int | float):
                latencies.setdefault("rrf", int(latency))
        final_hits = [hit for hit in variant.get("final_results", []) if isinstance(hit, dict)]
        rrf_hit = _first_gold_hit(final_hits, answers, rank_key="rrf_rank")
        if rrf_hit is not None:
            ranks["rrf_rank"] = _min_rank(ranks["rrf_rank"], rrf_hit.get("rank"))
            scores["rrf_score"] = rrf_hit.get("rrf_score")
            hits["rrf"] = rrf_hit

    return {
        "ranks": ranks,
        "scores": scores,
        "hits": hits,
        "path_states": states,
        "path_latencies_ms": latencies,
    }


def _source_under_doc_dir(source: str, doc_dir: str) -> bool:
    if not source or not doc_dir:
        return False
    try:
        return Path(source).resolve().is_relative_to(Path(doc_dir).resolve())
    except (OSError, ValueError):
        normalized_dir = str(doc_dir).rstrip("/") + "/"
        return str(source).startswith(normalized_dir)


def evaluate_retrieval(
    predictions_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> RetrievalEvalStats:
    """Evaluate retrieval evidence stored in a CRAG prediction JSONL file."""

    total = 0
    with_context = 0
    gold_context = 0
    gold_context_at_1 = 0
    gold_context_at_3 = 0
    gold_context_at_5 = 0
    any_source_doc_dir = 0
    all_sources_doc_dir = 0
    total_context_chunks = 0
    latency_sum = 0
    latency_count = 0
    details: list[dict[str, Any]] = []

    for row in iter_manifest(predictions_path):
        total += 1
        answers = [str(v) for v in row.get("answers") or []]
        chunks = [chunk for chunk in row.get("context_chunks") or [] if isinstance(chunk, dict)]
        doc_dir = str(row.get("doc_dir") or "")
        sources = [str(chunk.get("source") or "") for chunk in chunks if chunk.get("source")]
        source_hits = [_source_under_doc_dir(source, doc_dir) for source in sources]
        path_diagnostics = _retrieval_path_gold_diagnostics(row, chunks, answers)
        short_or_numeric = bool(row.get("short_or_numeric_answer")) or (
            _has_low_confidence_literal_answer(answers)
        )

        context_count = len(chunks)
        total_context_chunks += context_count
        if context_count:
            with_context += 1

        row_gold_context = _context_contains_answer(chunks, answers)
        row_gold_context_at_1 = _context_contains_answer(chunks, answers, top_k=1)
        row_gold_context_at_3 = _context_contains_answer(chunks, answers, top_k=3)
        row_gold_context_at_5 = _context_contains_answer(chunks, answers, top_k=5)
        if row_gold_context:
            gold_context += 1
        if row_gold_context_at_1:
            gold_context_at_1 += 1
        if row_gold_context_at_3:
            gold_context_at_3 += 1
        if row_gold_context_at_5:
            gold_context_at_5 += 1
        if source_hits and any(source_hits):
            any_source_doc_dir += 1
        if source_hits and all(source_hits):
            all_sources_doc_dir += 1

        latency = row.get("latency_ms")
        if isinstance(latency, int | float):
            latency_sum += int(latency)
            latency_count += 1

        details.append(
            {
                "interaction_id": row.get("interaction_id", ""),
                "query": row.get("query", ""),
                "answers": answers,
                "context_chunks": context_count,
                "gold_answer_in_context": row_gold_context,
                "gold_answer_in_context_at_1": row_gold_context_at_1,
                "gold_answer_in_context_at_3": row_gold_context_at_3,
                "gold_answer_in_context_at_5": row_gold_context_at_5,
                "any_source_under_doc_dir": bool(source_hits and any(source_hits)),
                "all_sources_under_doc_dir": bool(source_hits and all(source_hits)),
                "short_or_numeric_answer": short_or_numeric,
                "path_gold_ranks": path_diagnostics["ranks"],
                "path_gold_scores": path_diagnostics["scores"],
                "path_gold_hits": path_diagnostics["hits"],
                "retriever_path_states": path_diagnostics["path_states"],
                "retriever_path_latencies_ms": path_diagnostics["path_latencies_ms"],
                "context_chunk_details": chunks,
                "retrieval_diagnostics": row.get("retrieval_diagnostics"),
                "latency_ms": latency,
            }
        )

    denominator = total or 1
    stats = RetrievalEvalStats(
        total=total,
        with_context=with_context,
        gold_answer_in_context=gold_context,
        gold_answer_in_context_at_1=gold_context_at_1,
        gold_answer_in_context_at_3=gold_context_at_3,
        gold_answer_in_context_at_5=gold_context_at_5,
        any_source_under_doc_dir=any_source_doc_dir,
        all_sources_under_doc_dir=all_sources_doc_dir,
        gold_answer_in_context_rate=gold_context / denominator,
        gold_answer_in_context_at_1_rate=gold_context_at_1 / denominator,
        gold_answer_in_context_at_3_rate=gold_context_at_3 / denominator,
        gold_answer_in_context_at_5_rate=gold_context_at_5 / denominator,
        any_source_under_doc_dir_rate=any_source_doc_dir / denominator,
        all_sources_under_doc_dir_rate=all_sources_doc_dir / denominator,
        avg_context_chunks=total_context_chunks / denominator,
        avg_latency_ms=(latency_sum / latency_count) if latency_count else 0.0,
    )

    if output_path:
        summary = asdict(stats)
        summary["diagnostic_fields"] = _retrieval_diagnostic_field_schema()
        payload = {"summary": summary, "details": details}
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")

    return stats


def _prediction_label(row: dict[str, Any]) -> str:
    prediction = str(row.get("prediction") or "")
    answers = [str(v) for v in row.get("answers") or []]
    if _matches_gold(prediction, answers):
        return "correct"
    if _is_missing(prediction):
        return "missing"
    return "incorrect"


def _retrieval_flags(row: dict[str, Any]) -> dict[str, bool | int]:
    answers = [str(v) for v in row.get("answers") or []]
    chunks = [chunk for chunk in row.get("context_chunks") or [] if isinstance(chunk, dict)]
    doc_dir = str(row.get("doc_dir") or "")
    sources = [str(chunk.get("source") or "") for chunk in chunks if chunk.get("source")]
    source_hits = [_source_under_doc_dir(source, doc_dir) for source in sources]
    return {
        "context_chunks": len(chunks),
        "gold_answer_in_context": _context_contains_answer(chunks, answers),
        "gold_answer_in_context_at_1": _context_contains_answer(chunks, answers, top_k=1),
        "gold_answer_in_context_at_3": _context_contains_answer(chunks, answers, top_k=3),
        "gold_answer_in_context_at_5": _context_contains_answer(chunks, answers, top_k=5),
        "any_source_under_doc_dir": bool(source_hits and any(source_hits)),
        "all_sources_under_doc_dir": bool(source_hits and all(source_hits)),
    }


def _new_summary_bucket() -> dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "missing": 0,
        "incorrect": 0,
        "gold_answer_in_context": 0,
        "gold_answer_in_context_at_1": 0,
        "gold_answer_in_context_at_3": 0,
        "gold_answer_in_context_at_5": 0,
        "any_source_under_doc_dir": 0,
        "all_sources_under_doc_dir": 0,
        "context_chunks_total": 0,
        "latencies_ms": [],
    }


def _accumulate_summary_bucket(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    label = _prediction_label(row)
    flags = _retrieval_flags(row)
    bucket["total"] += 1
    bucket[label] += 1
    for key in (
        "gold_answer_in_context",
        "gold_answer_in_context_at_1",
        "gold_answer_in_context_at_3",
        "gold_answer_in_context_at_5",
        "any_source_under_doc_dir",
        "all_sources_under_doc_dir",
    ):
        if flags[key]:
            bucket[key] += 1
    bucket["context_chunks_total"] += flags["context_chunks"]
    latency = row.get("latency_ms")
    if isinstance(latency, int | float):
        bucket["latencies_ms"].append(int(latency))


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _finalize_summary_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    total = bucket["total"]
    denominator = total or 1
    latencies = bucket["latencies_ms"]
    correct = bucket["correct"]
    missing = bucket["missing"]
    incorrect = bucket["incorrect"]
    return {
        "total": total,
        "correct": correct,
        "missing": missing,
        "incorrect": incorrect,
        "score": ((2 * correct + missing) / denominator - 1) if total else 0.0,
        "accuracy": correct / denominator,
        "missing_rate": missing / denominator,
        "hallucination_rate": incorrect / denominator,
        "gold_answer_in_context": bucket["gold_answer_in_context"],
        "gold_answer_in_context_rate": bucket["gold_answer_in_context"] / denominator,
        "gold_answer_in_context_at_1": bucket["gold_answer_in_context_at_1"],
        "gold_answer_in_context_at_1_rate": bucket["gold_answer_in_context_at_1"] / denominator,
        "gold_answer_in_context_at_3": bucket["gold_answer_in_context_at_3"],
        "gold_answer_in_context_at_3_rate": bucket["gold_answer_in_context_at_3"] / denominator,
        "gold_answer_in_context_at_5": bucket["gold_answer_in_context_at_5"],
        "gold_answer_in_context_at_5_rate": bucket["gold_answer_in_context_at_5"] / denominator,
        "any_source_under_doc_dir": bucket["any_source_under_doc_dir"],
        "any_source_under_doc_dir_rate": bucket["any_source_under_doc_dir"] / denominator,
        "all_sources_under_doc_dir": bucket["all_sources_under_doc_dir"],
        "all_sources_under_doc_dir_rate": bucket["all_sources_under_doc_dir"] / denominator,
        "avg_context_chunks": bucket["context_chunks_total"] / denominator,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "min_latency_ms": float(min(latencies)) if latencies else 0.0,
        "max_latency_ms": float(max(latencies)) if latencies else 0.0,
    }


def _breakdown_value(row: dict[str, Any], field: str, run_metadata: dict[str, Any]) -> str:
    if field == "scope":
        value = row.get("run_scope") or row.get("scope") or run_metadata.get("scope")
    elif field == "k":
        value = row.get("retrieval_k") or row.get("k") or run_metadata.get("k")
    elif field == "retrieval_mode":
        value = (
            row.get("retrieval_mode")
            or row.get("hybrid_search")
            or run_metadata.get("retrieval_mode")
        )
    else:
        value = row.get(field)
        if value in (None, ""):
            value = run_metadata.get(field)
    if value in (None, ""):
        return "unknown"
    return str(value)


def summarize_results(
    predictions_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    breakdown_fields: Iterable[str] = DEFAULT_BREAKDOWN_FIELDS,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize CRAG predictions with answer, retrieval and metadata breakdowns."""

    metadata = dict(run_metadata or {})
    rows = list(iter_manifest(predictions_path))
    overall = _new_summary_bucket()
    breakdown_buckets: dict[str, dict[str, dict[str, Any]]] = {
        field: {} for field in breakdown_fields
    }

    for row in rows:
        _accumulate_summary_bucket(overall, row)
        for field in breakdown_buckets:
            value = _breakdown_value(row, field, metadata)
            bucket = breakdown_buckets[field].setdefault(value, _new_summary_bucket())
            _accumulate_summary_bucket(bucket, row)

    breakdowns = {
        field: {
            value: _finalize_summary_bucket(bucket)
            for value, bucket in sorted(values.items(), key=lambda item: item[0])
        }
        for field, values in breakdown_buckets.items()
    }
    run = {
        "label": metadata.get("label", ""),
        "scope": metadata.get("scope", _breakdown_value(rows[0], "scope", {}) if rows else ""),
        "retrieval_mode": metadata.get(
            "retrieval_mode",
            _breakdown_value(rows[0], "retrieval_mode", {}) if rows else "",
        ),
        "embedding_model": metadata.get(
            "embedding_model",
            _breakdown_value(rows[0], "embedding_model", {}) if rows else "",
        ),
        "k": metadata.get("k", _breakdown_value(rows[0], "k", {}) if rows else ""),
        "predictions_path": str(predictions_path),
    }
    payload = {
        "schema": "kiara-crag-summary",
        "schema_version": 1,
        "run": run,
        "overall": _finalize_summary_bucket(overall),
        "breakdowns": breakdowns,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")

    return payload


_COMPARISON_METRICS = (
    "score",
    "accuracy",
    "missing_rate",
    "hallucination_rate",
    "gold_answer_in_context_rate",
    "gold_answer_in_context_at_1_rate",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
)


def _load_summary(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "overall" not in payload:
        raise ValueError(f"{path} ist keine CRAG-Summary-Datei")
    return payload


def compare_summaries(
    summary_paths: Iterable[str | os.PathLike[str]],
    output_path: str | os.PathLike[str] | None = None,
    *,
    baseline_index: int = 0,
) -> dict[str, Any]:
    """Compare multiple CRAG summary JSON files."""

    paths = [str(path) for path in summary_paths]
    if not paths:
        raise ValueError("Mindestens eine Summary-Datei ist erforderlich")
    if baseline_index < 0 or baseline_index >= len(paths):
        raise ValueError("baseline_index ausserhalb der Summary-Liste")

    loaded = [_load_summary(path) for path in paths]
    baseline = loaded[baseline_index]
    baseline_metrics = baseline["overall"]
    runs: list[dict[str, Any]] = []
    for index, (path, summary) in enumerate(zip(paths, loaded, strict=True)):
        overall = summary["overall"]
        metrics = {metric: overall.get(metric, 0.0) for metric in _COMPARISON_METRICS}
        deltas = {
            metric: metrics[metric] - baseline_metrics.get(metric, 0.0)
            for metric in _COMPARISON_METRICS
        }
        run = dict(summary.get("run") or {})
        run.update(
            {
                "index": index,
                "summary_path": path,
                "is_baseline": index == baseline_index,
                "total": overall.get("total", 0),
                "metrics": metrics,
                "delta_vs_baseline": deltas,
            }
        )
        runs.append(run)

    payload = {
        "schema": "kiara-crag-comparison",
        "schema_version": 1,
        "baseline_index": baseline_index,
        "baseline_label": runs[baseline_index].get("label", ""),
        "runs": runs,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")

    return payload


def comparison_to_markdown(comparison: dict[str, Any]) -> str:
    headers = [
        "Run",
        "Mode",
        "k",
        "Score",
        "Accuracy",
        "Missing",
        "Hallucination",
        "GoldCtx",
        "Avg ms",
        "Delta Score",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for run in comparison.get("runs") or []:
        metrics = run.get("metrics") or {}
        deltas = run.get("delta_vs_baseline") or {}
        label = str(run.get("label") or run.get("summary_path") or f"run-{run.get('index')}")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(run.get("retrieval_mode") or ""),
                    str(run.get("k") or ""),
                    f"{float(metrics.get('score', 0.0)):.3f}",
                    f"{float(metrics.get('accuracy', 0.0)):.3f}",
                    f"{float(metrics.get('missing_rate', 0.0)):.3f}",
                    f"{float(metrics.get('hallucination_rate', 0.0)):.3f}",
                    f"{float(metrics.get('gold_answer_in_context_rate', 0.0)):.3f}",
                    f"{float(metrics.get('avg_latency_ms', 0.0)):.1f}",
                    f"{float(deltas.get('score', 0.0)):+.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _first_answer_match(text: str, answers: list[str]) -> str:
    for answer in answers:
        if _answer_in_text(text, answer):
            return answer
    return ""


def _has_answer(text: str, answers: list[str]) -> bool:
    return bool(_first_answer_match(text, answers))


def _answer_excerpt(text: str, answer: str, radius: int = 160) -> str:
    if not text:
        return ""
    if not answer:
        return text[: radius * 2].strip()
    lower_text = text.casefold()
    lower_answer = answer.casefold()
    index = lower_text.find(lower_answer)
    if index < 0:
        return text[: radius * 2].strip()
    start = max(0, index - radius)
    end = min(len(text), index + len(answer) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def _read_audit_file_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _audit_raw_files(files: list[str], answers: list[str]) -> dict[str, Any]:
    hits = []
    readable = 0
    for raw_path in files:
        path = Path(raw_path)
        text = _read_audit_file_text(path)
        if text:
            readable += 1
        match = _first_answer_match(text, answers)
        if match:
            hits.append(
                {
                    "source": str(path),
                    "matched_answer": match,
                    "excerpt": _answer_excerpt(text, match),
                }
            )
    return {
        "raw_file_count": len(files),
        "raw_readable_files": readable,
        "raw_has_gold": bool(hits),
        "raw_answer_files": hits[:5],
    }


def _chunk_loaded_file(path: Path, config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    from kiara.ingestion.chunk_strategy import detect_chunk_params
    from kiara.ingestion.chunker import chunk_document
    from kiara.ingestion.loader import load_file

    ingestion_cfg = config.get("ingestion") or {}
    default_chunk_size = int(ingestion_cfg.get("chunk_size") or 1000)
    default_chunk_overlap = int(ingestion_cfg.get("chunk_overlap") or 200)
    doc = load_file(str(path), ddu_enabled=False)
    if not doc or doc.get("metadata", {}).get("no_content"):
        return "", []
    text = str(doc.get("text") or "")
    metadata = dict(doc.get("metadata") or {})
    try:
        chunk_size, chunk_overlap = detect_chunk_params(text, metadata)
    except Exception:
        chunk_size, chunk_overlap = default_chunk_size, default_chunk_overlap
        metadata["chunk_strategy"] = "default"
    metadata["chunk_size_used"] = chunk_size
    chunks = chunk_document(
        text,
        metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text, chunks


def _audit_loaded_chunks(
    files: list[str],
    answers: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    loaded_hits = []
    chunk_hits = []
    loaded_count = 0
    chunk_count = 0
    for raw_path in files:
        path = Path(raw_path)
        text, chunks = _chunk_loaded_file(path, config)
        if text:
            loaded_count += 1
        match = _first_answer_match(text, answers)
        if match:
            loaded_hits.append(
                {
                    "source": str(path),
                    "matched_answer": match,
                    "excerpt": _answer_excerpt(text, match),
                }
            )
        chunk_count += len(chunks)
        for chunk in chunks:
            chunk_text = str(chunk.get("text") or "")
            chunk_match = _first_answer_match(chunk_text, answers)
            if not chunk_match:
                continue
            meta = chunk.get("metadata") or {}
            chunk_hits.append(
                {
                    "source": str(meta.get("source") or path),
                    "chunk_index": meta.get("chunk_index"),
                    "chunk_strategy": meta.get("chunk_strategy", ""),
                    "matched_answer": chunk_match,
                    "excerpt": _answer_excerpt(chunk_text, chunk_match),
                }
            )
    return {
        "loaded_file_count": loaded_count,
        "loaded_has_gold": bool(loaded_hits),
        "loaded_answer_files": loaded_hits[:5],
        "local_chunk_count": chunk_count,
        "chunk_has_gold": bool(chunk_hits),
        "chunk_answer_count": len(chunk_hits),
        "chunk_answer_samples": chunk_hits[:5],
    }


def _audit_indexed_chunks(
    store: Any,
    files: list[str],
    answers: list[str],
    datasource_id: int | None,
) -> dict[str, Any]:
    sources = [str(Path(path)) for path in files]
    if datasource_id is None:
        chunk_ids = store.get_ids_by_sources(sources)
    else:
        chunk_ids = store.get_ids_by_sources(sources, ds_id=datasource_id)
    if not chunk_ids:
        return {
            "indexed_chunk_count": 0,
            "indexed_has_gold": False,
            "indexed_answer_count": 0,
            "indexed_answer_chunks": [],
        }
    docs = store.get_documents_by_ids(chunk_ids, include=["documents", "metadatas"])
    hits = []
    for chunk_id, text, meta in zip(
        docs.get("ids") or [],
        docs.get("documents") or [],
        docs.get("metadatas") or [],
        strict=False,
    ):
        match = _first_answer_match(str(text or ""), answers)
        if not match:
            continue
        hits.append(
            {
                "chunk_id": chunk_id,
                "source": str(meta.get("source") or ""),
                "chunk_index": meta.get("chunk_index"),
                "score": None,
                "matched_answer": match,
                "excerpt": _answer_excerpt(str(text or ""), match),
            }
        )
    return {
        "indexed_chunk_count": len(chunk_ids),
        "indexed_has_gold": bool(hits),
        "indexed_answer_count": len(hits),
        "indexed_answer_chunks": hits[:5],
    }


def _audit_vector_where(
    scope: str,
    doc_dir: str,
    datasource_id: int | None,
) -> dict[str, Any] | None:
    conditions = []
    if scope == "per-question" and doc_dir:
        conditions.append({"directory": {"$contains": doc_dir}})
    if datasource_id is not None:
        conditions.append({"ds_ids": {"$in": [datasource_id]}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _audit_vector_search(
    store: Any,
    query: str,
    answers: list[str],
    *,
    k: int,
    scope: str,
    doc_dir: str,
    datasource_id: int | None,
) -> dict[str, Any]:
    where = _audit_vector_where(scope, doc_dir, datasource_id)
    results = store.search(query, k=k, where=where)
    top = []
    first_hit: dict[str, Any] | None = None
    for rank, result in enumerate(results, 1):
        text = str(result.get("text") or "")
        meta = result.get("metadata") or {}
        match = _first_answer_match(text, answers)
        row = {
            "rank": rank,
            "source": str(meta.get("source") or ""),
            "chunk_index": meta.get("chunk_index"),
            "score": result.get("score"),
            "has_gold": bool(match),
            "matched_answer": match,
        }
        if match and first_hit is None:
            first_hit = {
                **row,
                "excerpt": _answer_excerpt(text, match),
            }
        top.append(row)
    return {
        "vector_scope": scope,
        "vector_where": where,
        "vector_topk": k,
        "vector_result_count": len(results),
        "vector_topk_has_gold": first_hit is not None,
        "vector_gold_rank": first_hit.get("rank") if first_hit else None,
        "vector_gold_score": first_hit.get("score") if first_hit else None,
        "vector_gold_source": first_hit.get("source") if first_hit else "",
        "vector_gold_hit": first_hit,
        "vector_top_results": top[:10],
    }


def _audit_diagnosis(row: dict[str, Any]) -> str:
    if not row.get("answers"):
        return "no_gold_answers"
    if row.get("vector_topk_has_gold"):
        return "embedding_topk_hit"
    if row.get("indexed_has_gold"):
        return "embedding_topk_miss"
    if row.get("chunk_has_gold"):
        return "indexing_missing"
    if row.get("loaded_has_gold"):
        return "chunking_missing"
    if row.get("raw_has_gold"):
        return "loader_missing"
    return "raw_missing"


def _audit_summary(
    rows: list[dict[str, Any]], output_path: str | os.PathLike[str]
) -> dict[str, Any]:
    total = len(rows)
    denominator = total or 1
    keys = (
        "raw_has_gold",
        "loaded_has_gold",
        "chunk_has_gold",
        "indexed_has_gold",
        "vector_topk_has_gold",
    )
    counts = {key: sum(1 for row in rows if row.get(key)) for key in keys}
    diagnosis_counts: dict[str, int] = {}
    for row in rows:
        diagnosis = str(row.get("diagnosis") or "unknown")
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
    return {
        "schema": "kiara-crag-ingestion-embedding-audit-summary",
        "schema_version": 1,
        "total": total,
        "output_path": str(output_path),
        **counts,
        **{f"{key}_rate": counts[key] / denominator for key in keys},
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
    }


def audit_crag_ingestion_embeddings(
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    config_path: str | None = None,
    config: dict[str, Any] | None = None,
    embedding_model: str | None = None,
    k: int = DEFAULT_AUDIT_TOPK,
    max_records: int | None = None,
    scope: str = "per-question",
    datasource_id: int | None = None,
    summary_output: str | os.PathLike[str] | None = None,
    store: Any | None = None,
) -> dict[str, Any]:
    """Audit whether CRAG gold answers survive ingestion and dense retrieval."""

    if scope not in ("per-question", "global"):
        raise ValueError("scope muss 'per-question' oder 'global' sein")

    if config is None:
        from kiara.config import load_config

        config = load_config(config_path)
    if store is None:
        from kiara.retrieval.vectorstore import VectorStore

        store = VectorStore(config, embedding_model=embedding_model)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with out.open("w", encoding="utf-8") as handle:
        for index, entry in enumerate(iter_manifest(manifest_path), 1):
            if max_records is not None and index > max_records:
                break
            answers = [str(answer) for answer in entry.get("answers") or [] if str(answer).strip()]
            files = [str(path) for path in entry.get("files") or []]
            row: dict[str, Any] = {
                "interaction_id": entry.get("interaction_id", ""),
                "task": entry.get("task"),
                "query": str(entry.get("query") or ""),
                "answers": answers,
                "domain": entry.get("domain", ""),
                "question_type": entry.get("question_type", ""),
                "static_or_dynamic": entry.get("static_or_dynamic", ""),
                "popularity": entry.get("popularity", ""),
                "doc_dir": str(entry.get("doc_dir") or ""),
                "file_count": len(files),
                "audit_scope": scope,
                "audit_k": k,
                "audit_datasource_id": datasource_id,
            }
            row.update(_audit_raw_files(files, answers))
            row.update(_audit_loaded_chunks(files, answers, config))
            row.update(_audit_indexed_chunks(store, files, answers, datasource_id))
            row.update(
                _audit_vector_search(
                    store,
                    row["query"],
                    answers,
                    k=k,
                    scope=scope,
                    doc_dir=row["doc_dir"],
                    datasource_id=datasource_id,
                )
            )
            row["diagnosis"] = _audit_diagnosis(row)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = _audit_summary(rows, out)
    if summary_output:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_json_dumps(summary) + "\n", encoding="utf-8")
        summary["summary_path"] = str(summary_path)
    return summary


def _page_content_variants(page: dict[str, Any]) -> dict[str, str]:
    html = str(page.get("page_result") or page.get("html") or "")
    snippet = str(page.get("page_snippet") or page.get("snippet") or "")
    prepared_body = html if html.strip() else snippet
    return {
        "html": html,
        "snippet": snippet,
        "prepared_body": prepared_body,
        "any_content": "\n".join(part for part in (html, snippet) if part.strip()),
    }


def _web_page_hits(
    record: dict[str, Any],
    answers: list[str],
    *,
    max_rank: int | None = None,
) -> list[dict[str, Any]]:
    search_results = record.get("search_results") or []
    if not isinstance(search_results, list):
        return []
    hits = []
    for rank, page in enumerate(search_results, 1):
        if max_rank is not None and rank > max_rank:
            break
        if not isinstance(page, dict):
            continue
        variants = _page_content_variants(page)
        html_match = _first_answer_match(variants["html"], answers)
        snippet_match = _first_answer_match(variants["snippet"], answers)
        prepared_match = _first_answer_match(variants["prepared_body"], answers)
        any_match = _first_answer_match(variants["any_content"], answers)
        if not any_match:
            continue
        hits.append(
            {
                "rank": rank,
                "page_name": str(page.get("page_name") or ""),
                "page_url": str(page.get("page_url") or ""),
                "html_has_gold": bool(html_match),
                "snippet_has_gold": bool(snippet_match),
                "prepared_body_has_gold": bool(prepared_match),
                "matched_answer": prepared_match or any_match,
                "excerpt": _answer_excerpt(
                    variants["prepared_body"] if prepared_match else variants["any_content"],
                    prepared_match or any_match,
                ),
            }
        )
    return hits


def _structured_hits(record: dict[str, Any], answers: list[str]) -> list[dict[str, Any]]:
    hits = []
    for key, value in _structured_payloads(record):
        text = _json_dumps({"field": key, "value": value})
        match = _first_answer_match(text, answers)
        if match:
            hits.append(
                {
                    "field": key,
                    "matched_answer": match,
                    "excerpt": _answer_excerpt(text, match),
                }
            )
    return hits


def _prepared_file_hits(files: list[str], answers: list[str]) -> list[dict[str, Any]]:
    hits = []
    for raw_path in files:
        path = Path(raw_path)
        text = _read_audit_file_text(path)
        match = _first_answer_match(text, answers)
        if match:
            hits.append(
                {
                    "source": str(path),
                    "matched_answer": match,
                    "excerpt": _answer_excerpt(text, match),
                }
            )
    return hits


def _prepared_max_web_rank(files: list[str]) -> int:
    max_rank = 0
    pattern = re.compile(r"web_(\d{3})\.")
    for raw_path in files:
        match = pattern.search(Path(raw_path).name)
        if match:
            max_rank = max(max_rank, int(match.group(1)))
    return max_rank


def _prep_diagnosis(row: dict[str, Any]) -> str:
    if not row.get("answers"):
        return "no_gold_answers"
    if row.get("prepared_has_gold"):
        return "prepared_hit"
    if row.get("source_preferred_within_prepared_has_gold"):
        return "preparation_loss"
    if row.get("source_any_within_prepared_has_gold"):
        return "snippet_shadowed_by_html"
    first_preferred = row.get("first_preferred_gold_rank")
    prepared_rank = row.get("prepared_max_web_rank") or 0
    if first_preferred and prepared_rank and first_preferred > prepared_rank:
        return "page_limit_miss"
    if row.get("source_preferred_has_gold") or row.get("source_structured_has_gold"):
        return "content_not_exported"
    if row.get("source_any_has_gold"):
        return "content_not_exported"
    return "source_missing"


def _find_source_records_for_manifest(
    raw_input_path: str | os.PathLike[str],
    manifest_entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pending = {str(entry.get("interaction_id") or "") for entry in manifest_entries}
    pending.discard("")
    found: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(iter_crag_records(raw_input_path), 1):
        interaction_id = _stable_id(record, index)
        if interaction_id not in pending:
            continue
        found[interaction_id] = record
        pending.remove(interaction_id)
        if not pending:
            break
    return found


def _preparation_summary(rows: list[dict[str, Any]], output_path: str | os.PathLike[str]) -> dict:
    total = len(rows)
    denominator = total or 1
    keys = (
        "source_any_has_gold",
        "source_preferred_has_gold",
        "source_preferred_within_prepared_has_gold",
        "prepared_has_gold",
    )
    counts = {key: sum(1 for row in rows if row.get(key)) for key in keys}
    top_limit_counts: dict[str, int] = {}
    for row in rows:
        for key, value in (row.get("top_limit_hits") or {}).items():
            if value:
                top_limit_counts[key] = top_limit_counts.get(key, 0) + 1
            else:
                top_limit_counts.setdefault(key, 0)
    diagnosis_counts: dict[str, int] = {}
    for row in rows:
        diagnosis = str(row.get("diagnosis") or "unknown")
        diagnosis_counts[diagnosis] = diagnosis_counts.get(diagnosis, 0) + 1
    return {
        "schema": "kiara-crag-preparation-coverage-summary",
        "schema_version": 1,
        "total": total,
        "output_path": str(output_path),
        **counts,
        **{f"{key}_rate": counts[key] / denominator for key in keys},
        "top_limit_counts": dict(sorted(top_limit_counts.items())),
        "top_limit_rates": {
            key: value / denominator for key, value in sorted(top_limit_counts.items())
        },
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
    }


def audit_crag_preparation_coverage(
    raw_input_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    page_limits: Iterable[int] = DEFAULT_PREPARATION_PAGE_LIMITS,
    summary_output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Audit whether CRAG source retrieval content survived preparation."""

    manifest_entries = list(iter_manifest(manifest_path))
    source_records = _find_source_records_for_manifest(raw_input_path, manifest_entries)
    limits = tuple(sorted({int(limit) for limit in page_limits if int(limit) > 0}))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with out.open("w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            interaction_id = str(entry.get("interaction_id") or "")
            record = source_records.get(interaction_id)
            answers = [str(answer) for answer in entry.get("answers") or [] if str(answer).strip()]
            files = [str(path) for path in entry.get("files") or []]
            prepared_max_rank = _prepared_max_web_rank(files)
            row: dict[str, Any] = {
                "interaction_id": interaction_id,
                "task": entry.get("task"),
                "query": str(entry.get("query") or ""),
                "answers": answers,
                "domain": entry.get("domain", ""),
                "question_type": entry.get("question_type", ""),
                "static_or_dynamic": entry.get("static_or_dynamic", ""),
                "popularity": entry.get("popularity", ""),
                "doc_dir": str(entry.get("doc_dir") or ""),
                "prepared_file_count": len(files),
                "prepared_max_web_rank": prepared_max_rank,
                "source_record_found": record is not None,
            }
            if record is None:
                row.update(
                    {
                        "source_search_result_count": 0,
                        "source_any_has_gold": False,
                        "source_preferred_has_gold": False,
                        "source_any_within_prepared_has_gold": False,
                        "source_preferred_within_prepared_has_gold": False,
                        "source_structured_has_gold": False,
                        "prepared_has_gold": bool(_prepared_file_hits(files, answers)),
                        "top_limit_hits": {f"top_{limit}": False for limit in limits},
                        "diagnosis": "source_record_not_found",
                    }
                )
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                continue

            search_results = record.get("search_results") or []
            all_web_hits = _web_page_hits(record, answers)
            prepared_window_hits = _web_page_hits(
                record,
                answers,
                max_rank=prepared_max_rank or None,
            )
            structured_hits = _structured_hits(record, answers)
            prepared_hits = _prepared_file_hits(files, answers)
            preferred_hits = [hit for hit in all_web_hits if hit["prepared_body_has_gold"]]
            any_within_prepared = bool(prepared_window_hits)
            preferred_within_prepared = any(
                hit["prepared_body_has_gold"] for hit in prepared_window_hits
            )
            top_limit_hits = {
                f"top_{limit}": any(
                    hit["rank"] <= limit and hit["prepared_body_has_gold"] for hit in all_web_hits
                )
                for limit in limits
            }
            row.update(
                {
                    "source_search_result_count": (
                        len(search_results) if isinstance(search_results, list) else 0
                    ),
                    "source_any_has_gold": bool(all_web_hits or structured_hits),
                    "source_preferred_has_gold": bool(preferred_hits or structured_hits),
                    "source_any_within_prepared_has_gold": any_within_prepared,
                    "source_preferred_within_prepared_has_gold": preferred_within_prepared,
                    "source_structured_has_gold": bool(structured_hits),
                    "prepared_has_gold": bool(prepared_hits),
                    "first_any_gold_rank": all_web_hits[0]["rank"] if all_web_hits else None,
                    "first_preferred_gold_rank": (
                        preferred_hits[0]["rank"] if preferred_hits else None
                    ),
                    "source_web_gold_hits": all_web_hits[:5],
                    "source_structured_gold_hits": structured_hits[:5],
                    "prepared_gold_files": prepared_hits[:5],
                    "top_limit_hits": top_limit_hits,
                }
            )
            row["diagnosis"] = _prep_diagnosis(row)
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = _preparation_summary(rows, out)
    if summary_output:
        summary_path = Path(summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_json_dumps(summary) + "\n", encoding="utf-8")
        summary["summary_path"] = str(summary_path)
    return summary


def _rows_by_interaction_id(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_manifest(path):
        interaction_id = str(row.get("interaction_id") or "")
        if interaction_id:
            rows[interaction_id] = row
    return rows


def _combined_answers(*rows: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for answer in row.get("answers") or []:
            text = str(answer).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            answers.append(text)
    return answers


def _has_invalid_question_answer(answers: list[str]) -> bool:
    return any(_normalize_answer(answer) == "invalid question" for answer in answers)


def _has_low_confidence_literal_answer(answers: list[str]) -> bool:
    for answer in answers:
        normalized = _normalize_answer(answer)
        if not normalized or normalized == "invalid question":
            continue
        compact = normalized.replace(" ", "")
        if len(normalized) <= 3 or len(compact) <= 3:
            return True
        if re.fullmatch(r"[\d.,:+/%$ -]+[a-z]?", normalized):
            return True
    return False


def _analysis_label(
    prep_row: dict[str, Any],
    audit_row: dict[str, Any],
    answers: list[str],
) -> str:
    prep_diagnosis = str(prep_row.get("diagnosis") or "")
    low_confidence = _has_low_confidence_literal_answer(answers)
    audit_has_gold = any(
        bool(audit_row.get(key))
        for key in (
            "raw_has_gold",
            "loaded_has_gold",
            "chunk_has_gold",
            "indexed_has_gold",
            "vector_topk_has_gold",
        )
    )

    if not answers:
        return "no_gold_answers"
    if _has_invalid_question_answer(answers):
        return "invalid_question_label"
    if not prep_row:
        return "missing_preparation_audit"
    if not audit_row:
        return "missing_ingestion_audit"
    if prep_diagnosis == "page_limit_miss":
        return "page_limit_candidate"
    if prep_diagnosis == "source_missing" and audit_has_gold:
        return "audit_match_noise"
    if prep_diagnosis == "source_missing":
        return "source_missing"
    if prep_diagnosis in {
        "preparation_loss",
        "snippet_shadowed_by_html",
        "content_not_exported",
    }:
        return "preparation_loss"
    if audit_row.get("vector_topk_has_gold"):
        return "dense_hit_low_confidence" if low_confidence else "dense_hit"
    if audit_row.get("indexed_has_gold"):
        return "dense_rank_miss_low_confidence" if low_confidence else "dense_rank_miss"
    if audit_row.get("chunk_has_gold"):
        return "indexing_gap"
    if audit_row.get("loaded_has_gold"):
        return "chunking_gap"
    if audit_row.get("raw_has_gold"):
        return "loader_gap"
    if prep_row.get("prepared_has_gold"):
        return "ingestion_audit_gap"
    return prep_diagnosis or str(audit_row.get("diagnosis") or "unknown")


def _analysis_warnings(
    label: str,
    prep_row: dict[str, Any],
    audit_row: dict[str, Any],
    answers: list[str],
) -> list[str]:
    warnings = []
    if _has_low_confidence_literal_answer(answers):
        warnings.append("short_or_numeric_answer")
    if str(prep_row.get("diagnosis") or "") == "source_missing" and any(
        bool(audit_row.get(key))
        for key in ("chunk_has_gold", "indexed_has_gold", "vector_topk_has_gold")
    ):
        warnings.append("preparation_source_missing_but_ingestion_match")
    if label.endswith("_low_confidence"):
        warnings.append("literal_match_needs_manual_check")
    return warnings


def _analysis_notes(
    label: str,
    prep_row: dict[str, Any],
    audit_row: dict[str, Any],
    answers: list[str],
) -> list[str]:
    notes = []
    if label == "dense_rank_miss":
        notes.append(
            "Gold chunk is indexed, but dense Top-k did not retrieve it. "
            "Inspect vector ranking, reranking, query expansion and fusion for this case."
        )
    elif label == "dense_hit":
        notes.append("Gold chunk is indexed and appears in dense Top-k.")
    elif label == "dense_hit_low_confidence":
        notes.append(
            "Dense Top-k contains a literal answer match, but the answer is too short "
            "or numeric to treat as strong evidence without manual review."
        )
    elif label == "dense_rank_miss_low_confidence":
        notes.append(
            "Indexed chunks contain a literal answer match, but the answer is too short "
            "or numeric to prove a retrieval miss without manual review."
        )
    elif label == "page_limit_candidate":
        first_rank = prep_row.get("first_preferred_gold_rank")
        prepared_rank = prep_row.get("prepared_max_web_rank")
        notes.append(
            f"Gold evidence is outside the prepared page window "
            f"(first rank {first_rank}, prepared through rank {prepared_rank})."
        )
    elif label == "audit_match_noise":
        notes.append(
            "Preparation audit cannot find source evidence, but ingestion/index audit "
            "matched an answer string. Treat this as likely audit noise until reviewed."
        )
    elif label == "source_missing":
        notes.append(
            "No gold answer was found in the CRAG source retrieval material by the "
            "literal coverage audit."
        )
    elif label == "invalid_question_label":
        notes.append(
            "Gold label is 'invalid question'; this should evaluate refusal behavior, "
            "not document retrieval."
        )
    elif label == "preparation_loss":
        notes.append(
            "Gold evidence exists in CRAG source material but was not exported into "
            "the prepared KIara documents."
        )
    elif label == "indexing_gap":
        notes.append("Local chunks contain the answer, but indexed chunks do not.")
    elif label == "chunking_gap":
        notes.append("Loaded text contains the answer, but reproduced chunks do not.")
    elif label == "loader_gap":
        notes.append("Raw files contain the answer, but the loader output does not.")

    if _has_low_confidence_literal_answer(answers) and "short_or_numeric_answer" not in notes:
        notes.append("Answer is short or numeric; literal contains-match can overcount.")
    if audit_row.get("vector_gold_rank"):
        notes.append(f"Dense gold rank: {audit_row.get('vector_gold_rank')}.")
    return notes


def _analysis_next_step(label: str) -> str:
    steps = {
        "dense_rank_miss": "Audit retrieval/ranking for this case before changing ingestion.",
        "dense_rank_miss_low_confidence": "Manually verify the matched chunk before tuning retrieval.",
        "dense_hit": "No ingestion or dense-retrieval fix indicated for this case.",
        "dense_hit_low_confidence": "Manually verify whether the short match is real evidence.",
        "page_limit_candidate": "Re-run preparation with a larger page limit or tune the page cutoff.",
        "audit_match_noise": "Tighten the audit or manually inspect the matched chunk.",
        "source_missing": "Exclude from retrieval-quality diagnosis or use official answer eval only.",
        "invalid_question_label": "Evaluate refusal behavior separately from retrieval quality.",
        "preparation_loss": "Fix CRAG preparation/export before retrieval tuning.",
        "indexing_gap": "Inspect datasource indexing and Qdrant payload/source mapping.",
        "chunking_gap": "Inspect chunk strategy and loader-to-chunker handoff.",
        "loader_gap": "Inspect file loader behavior for this source type.",
        "ingestion_audit_gap": "Inspect the ingestion audit setup and source filters.",
    }
    return steps.get(label, "Inspect this case manually.")


def _first_analysis_sample(prep_row: dict[str, Any], audit_row: dict[str, Any]) -> dict[str, Any]:
    vector_hit = audit_row.get("vector_gold_hit")
    if isinstance(vector_hit, dict) and vector_hit.get("excerpt"):
        return {
            "source": vector_hit.get("source", ""),
            "excerpt": vector_hit.get("excerpt", ""),
            "rank": vector_hit.get("rank"),
        }

    for key in (
        "indexed_answer_chunks",
        "chunk_answer_samples",
        "prepared_gold_files",
        "source_web_gold_hits",
    ):
        source_row = audit_row if key in audit_row else prep_row
        samples = source_row.get(key) or []
        if not isinstance(samples, list) or not samples:
            continue
        sample = samples[0]
        if not isinstance(sample, dict):
            continue
        return {
            "source": sample.get("source") or sample.get("page_url") or sample.get("field", ""),
            "excerpt": sample.get("excerpt", ""),
            "rank": sample.get("rank"),
        }
    return {}


def _increment_count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def analyze_crag_audit_cases(
    preparation_audit_path: str | os.PathLike[str],
    ingestion_audit_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    markdown_output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Join CRAG preparation and ingestion audits into actionable case labels."""

    prep_rows = _rows_by_interaction_id(preparation_audit_path)
    audit_rows = _rows_by_interaction_id(ingestion_audit_path)
    interaction_ids = sorted(set(prep_rows) | set(audit_rows))
    cases: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}

    for interaction_id in interaction_ids:
        prep_row = prep_rows.get(interaction_id, {})
        audit_row = audit_rows.get(interaction_id, {})
        answers = _combined_answers(prep_row, audit_row)
        label = _analysis_label(prep_row, audit_row, answers)
        warnings = _analysis_warnings(label, prep_row, audit_row, answers)
        notes = _analysis_notes(label, prep_row, audit_row, answers)
        sample = _first_analysis_sample(prep_row, audit_row)

        for warning in warnings:
            _increment_count(warning_counts, warning)
        _increment_count(label_counts, label)

        case = {
            "interaction_id": interaction_id,
            "label": label,
            "query": str(prep_row.get("query") or audit_row.get("query") or ""),
            "answers": answers,
            "preparation_diagnosis": str(prep_row.get("diagnosis") or ""),
            "ingestion_diagnosis": str(audit_row.get("diagnosis") or ""),
            "prepared_has_gold": bool(prep_row.get("prepared_has_gold")),
            "source_any_has_gold": bool(prep_row.get("source_any_has_gold")),
            "indexed_has_gold": bool(audit_row.get("indexed_has_gold")),
            "vector_topk_has_gold": bool(audit_row.get("vector_topk_has_gold")),
            "vector_gold_rank": audit_row.get("vector_gold_rank"),
            "first_preferred_gold_rank": prep_row.get("first_preferred_gold_rank"),
            "prepared_max_web_rank": prep_row.get("prepared_max_web_rank"),
            "warnings": warnings,
            "notes": notes,
            "recommended_next_step": _analysis_next_step(label),
        }
        if sample:
            case["sample"] = sample
        cases.append(case)

    summary = {
        "schema": "kiara-crag-audit-case-analysis",
        "schema_version": 1,
        "total": len(cases),
        "preparation_audit_path": str(preparation_audit_path),
        "ingestion_audit_path": str(ingestion_audit_path),
        "label_counts": dict(sorted(label_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "actionable_counts": {
            "retrieval_or_ranking": (
                label_counts.get("dense_rank_miss", 0)
                + label_counts.get("dense_rank_miss_low_confidence", 0)
            ),
            "preparation_page_limit": label_counts.get("page_limit_candidate", 0),
            "preparation_loss": label_counts.get("preparation_loss", 0),
            "audit_noise_or_manual_review": (
                label_counts.get("audit_match_noise", 0)
                + label_counts.get("dense_hit_low_confidence", 0)
                + label_counts.get("dense_rank_miss_low_confidence", 0)
            ),
            "source_material_missing": label_counts.get("source_missing", 0),
            "invalid_question_labels": label_counts.get("invalid_question_label", 0),
        },
    }
    payload = {"summary": summary, "cases": cases}

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json_dumps(payload) + "\n", encoding="utf-8")
    if markdown_output:
        out = Path(markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(crag_audit_case_analysis_to_markdown(payload), encoding="utf-8")
    return payload


def _markdown_cell(value: Any, max_chars: int = 180) -> str:
    text = str(value if value is not None else "")
    text = " ".join(text.replace("|", "\\|").split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def crag_audit_case_analysis_to_markdown(analysis: dict[str, Any]) -> str:
    """Render CRAG audit case analysis as a compact markdown report."""

    summary = analysis.get("summary") or {}
    cases = [case for case in analysis.get("cases") or [] if isinstance(case, dict)]
    lines = [
        "# CRAG Audit Case Analysis",
        "",
        "## Summary",
        "",
        f"- Total cases: {summary.get('total', len(cases))}",
    ]
    for label, count in (summary.get("label_counts") or {}).items():
        lines.append(f"- {label}: {count}")
    warning_counts = summary.get("warning_counts") or {}
    if warning_counts:
        lines.extend(["", "## Warnings", ""])
        for warning, count in warning_counts.items():
            lines.append(f"- {warning}: {count}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `dense_rank_miss`: answer-bearing chunks are indexed, but dense retrieval did not surface them.",
            "- `page_limit_candidate`: source evidence exists only outside the prepared page window.",
            "- `audit_match_noise`: literal answer matching likely overcounted weak evidence.",
            "- `source_missing`: the sample is not useful for retrieval-quality diagnosis without additional source evidence.",
            "",
            "## Cases",
            "",
            "| Label | ID | Query | Answers | Prep | Ingestion | Rank | Next step |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(case.get("label"), 80),
                    _markdown_cell(case.get("interaction_id"), 48),
                    _markdown_cell(case.get("query"), 160),
                    _markdown_cell(", ".join(str(a) for a in case.get("answers") or []), 80),
                    _markdown_cell(case.get("preparation_diagnosis"), 80),
                    _markdown_cell(case.get("ingestion_diagnosis"), 80),
                    _markdown_cell(case.get("vector_gold_rank"), 20),
                    _markdown_cell(case.get("recommended_next_step"), 180),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Evidence Samples", ""])
    for case in cases:
        sample = case.get("sample") or {}
        if not sample:
            continue
        notes = " ".join(str(note) for note in case.get("notes") or [])
        lines.extend(
            [
                f"### {_markdown_cell(case.get('interaction_id'), 80)}",
                "",
                f"- Label: `{_markdown_cell(case.get('label'), 80)}`",
                f"- Source: `{_markdown_cell(sample.get('source'), 180)}`",
                f"- Notes: {_markdown_cell(notes, 500)}",
                "",
                "```text",
                _markdown_cell(sample.get("excerpt"), 700),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _artifact_rows_by_interaction_id(
    path: str | os.PathLike[str] | None,
    *,
    row_keys: tuple[str, ...] = ("details", "cases"),
) -> dict[str, dict[str, Any]]:
    """Load JSON or JSONL benchmark artifacts keyed by interaction_id."""

    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}

    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    rows: list[dict[str, Any]]
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        rows = list(iter_manifest(source))
    else:
        if isinstance(loaded, list):
            rows = [row for row in loaded if isinstance(row, dict)]
        elif isinstance(loaded, dict):
            rows = []
            for key in row_keys:
                candidate = loaded.get(key)
                if isinstance(candidate, list):
                    rows = [row for row in candidate if isinstance(row, dict)]
                    break
            if not rows and loaded.get("interaction_id"):
                rows = [loaded]
        else:
            rows = []

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        interaction_id = str(row.get("interaction_id") or "")
        if interaction_id:
            out[interaction_id] = row
    return out


def _required_artifact_rows_by_interaction_id(
    path: str | os.PathLike[str] | None,
    label: str,
    *,
    row_keys: tuple[str, ...] = ("details", "cases"),
) -> dict[str, dict[str, Any]]:
    """Load a mandatory benchmark artifact and fail on missing/empty joins."""

    if not path:
        raise ValueError(f"{label}-Artefaktpfad fehlt")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"{label}-Artefakt nicht gefunden: {source}")
    if not source.is_file():
        raise ValueError(f"{label}-Artefakt ist keine Datei: {source}")
    rows = _artifact_rows_by_interaction_id(source, row_keys=row_keys)
    if not rows:
        raise ValueError(
            f"{label}-Artefakt enthaelt keine verwertbaren interaction_id-Zeilen: {source}"
        )
    return rows


def _judge_label(judge_row: dict[str, Any]) -> str:
    label = str(judge_row.get("label") or "").strip()
    if label:
        return label
    score = judge_row.get("score")
    if score == 1:
        return "correct"
    if score == 0:
        prediction = str(judge_row.get("prediction") or judge_row.get("prediction_judged") or "")
        return "missing" if _is_missing(prediction) else "incorrect"
    return "unknown"


def _bool_hit(row: dict[str, Any], key: str) -> bool | None:
    if key not in row:
        return None
    return bool(row.get(key))


def _rank_value(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _path_gold_rank(row: dict[str, Any], key: str) -> int | None:
    ranks = row.get("path_gold_ranks")
    if not isinstance(ranks, dict):
        return None
    value = ranks.get(key)
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _vertical_recommendation(failure_class: str) -> str:
    recommendations = {
        "source_missing": "Do not tune retrieval for this case until source evidence exists.",
        "preparation_loss": "Fix or rerun CRAG preparation/export before retrieval tuning.",
        "loader_loss": "Inspect loader output for the prepared source file.",
        "chunking_loss": "Inspect chunk strategy and loader-to-chunker handoff.",
        "dense_rank_miss": "Measure dense candidate generation and ranking variants before changing live retrieval.",
        "keyword_rank_miss": "Measure keyword/ES rank diagnostics before tuning keyword retrieval.",
        "colbert_rank_miss": "Measure ColBERT rank diagnostics before tuning ColBERT.",
        "rrf_loss": "Persist per-path ranks and RRF inputs to split fusion loss from missing keyword/ColBERT recovery.",
        "context_assembly_loss": "Trace final context assembly with chunk IDs and ranks.",
        "generator_only": "Audit prompt/generator behavior on the same retrieved evidence.",
        "no_loss_observed": "No Sprint-1 action; keep as positive control.",
        "unknown": "Add the missing diagnostic artifact or manually review the case.",
    }
    return recommendations.get(failure_class, recommendations["unknown"])


def _classify_vertical_case(
    *,
    prep_row: dict[str, Any],
    ingestion_row: dict[str, Any],
    retrieval_row: dict[str, Any],
    judge_row: dict[str, Any],
    answers: list[str],
    dense_candidate_k: int,
) -> dict[str, Any]:
    """Classify one CRAG case using only measured artifacts."""

    prep_diagnosis = str(prep_row.get("diagnosis") or "")
    ingestion_diagnosis = str(ingestion_row.get("diagnosis") or "")
    gold_in_context = bool(retrieval_row.get("gold_answer_in_context"))
    answer_label = _judge_label(judge_row)
    dense_rank = _rank_value(ingestion_row, "vector_gold_rank")
    semantic_rank = _path_gold_rank(retrieval_row, "semantic_rank")
    keyword_rank = _path_gold_rank(retrieval_row, "keyword_rank")
    colbert_rank = _path_gold_rank(retrieval_row, "colbert_rank")
    graph_rank = _path_gold_rank(retrieval_row, "graph_rank")
    rrf_rank = _path_gold_rank(retrieval_row, "rrf_rank")
    final_context_rank = _path_gold_rank(retrieval_row, "final_context_rank")
    any_path_rank = any(
        rank is not None for rank in (semantic_rank, keyword_rank, colbert_rank, graph_rank)
    )

    def result(
        failure_class: str,
        stage: str,
        rationale: str,
        *,
        confidence: str = "measured",
        unknown_reason: str = "",
    ) -> dict[str, Any]:
        return {
            "failure_class": failure_class,
            "loss_stage": stage,
            "rationale": rationale,
            "confidence": confidence,
            "unknown_reason": unknown_reason,
            "recommended_next_step": _vertical_recommendation(failure_class),
        }

    if not answers:
        return result(
            "unknown",
            "gold_label",
            "No gold answers are present in the artifacts.",
            unknown_reason="no_gold_answers",
        )
    if _has_invalid_question_answer(answers):
        return result(
            "unknown",
            "gold_label",
            "Gold label is 'invalid question'; this is not a document retrieval-quality case.",
            unknown_reason="invalid_question_label",
        )

    if gold_in_context:
        if answer_label == "correct":
            return result(
                "no_loss_observed",
                "none",
                "Gold evidence reached the prompt context and the judged answer is correct.",
            )
        if answer_label in {"missing", "incorrect"}:
            return result(
                "generator_only",
                "generator",
                "Gold evidence reached the prompt context, but the judged answer is not correct.",
            )
        return result(
            "unknown",
            "generator",
            "Gold evidence reached the prompt context, but no usable answer judgement is available.",
            unknown_reason="missing_answer_judgement",
        )

    if not prep_row:
        return result(
            "unknown",
            "source",
            "No preparation audit row is available for this case.",
            unknown_reason="missing_preparation_audit",
        )
    if prep_diagnosis == "source_missing":
        return result(
            "source_missing",
            "original_crag_source",
            "The preparation audit found no literal gold answer in CRAG source retrieval material.",
        )
    if prep_diagnosis == "source_record_not_found":
        return result(
            "unknown",
            "original_crag_source",
            "The preparation audit could not match this manifest case to the raw CRAG record.",
            unknown_reason="source_record_not_found",
        )
    if prep_diagnosis in {
        "page_limit_miss",
        "preparation_loss",
        "snippet_shadowed_by_html",
        "content_not_exported",
    }:
        return result(
            "preparation_loss",
            "prepared_file",
            f"Preparation diagnosis is {prep_diagnosis}.",
        )

    if not ingestion_row:
        return result(
            "unknown",
            "loader_text",
            "No ingestion/embedding audit row is available for this case.",
            unknown_reason="missing_ingestion_audit",
        )

    raw_hit = _bool_hit(ingestion_row, "raw_has_gold")
    loaded_hit = _bool_hit(ingestion_row, "loaded_has_gold")
    chunk_hit = _bool_hit(ingestion_row, "chunk_has_gold")
    indexed_hit = _bool_hit(ingestion_row, "indexed_has_gold")
    dense_hit = _bool_hit(ingestion_row, "vector_topk_has_gold")

    if raw_hit and not loaded_hit:
        return result(
            "loader_loss",
            "loader_text",
            "Prepared file contains the gold answer, but loader output does not.",
        )
    if loaded_hit and not chunk_hit:
        return result(
            "chunking_loss",
            "chunks",
            "Loader output contains the gold answer, but reproduced chunks do not.",
        )
    if chunk_hit and not indexed_hit:
        return result(
            "unknown",
            "qdrant_index",
            "Local chunks contain the gold answer, but indexed Qdrant chunks do not.",
            unknown_reason="qdrant_index_loss_not_in_requested_taxonomy",
        )
    if indexed_hit:
        if not dense_hit and semantic_rank is None:
            return result(
                "dense_rank_miss",
                "qdrant_dense",
                "Indexed chunks contain the gold answer, but dense search did not retrieve it in the audited top-k.",
            )
        measured_dense_rank = semantic_rank if semantic_rank is not None else dense_rank
        if measured_dense_rank is not None and measured_dense_rank > dense_candidate_k:
            return result(
                "dense_rank_miss",
                "qdrant_dense",
                (
                    "Dense search can find a gold chunk, but only outside the "
                    f"baseline dense candidate window ({measured_dense_rank} > {dense_candidate_k})."
                ),
            )
        if rrf_rank is not None and final_context_rank is None:
            return result(
                "context_assembly_loss",
                "rerank_or_context_assembly",
                (
                    "Gold evidence reached the fused retrieval result list "
                    f"(RRF rank {rrf_rank}) but did not reach the final prompt context."
                ),
            )
        if any_path_rank and rrf_rank is None:
            path_summary = {
                "semantic": semantic_rank,
                "keyword": keyword_rank,
                "colbert": colbert_rank,
                "graph": graph_rank,
            }
            present = ", ".join(
                f"{path}={rank}" for path, rank in path_summary.items() if rank is not None
            )
            return result(
                "rrf_loss",
                "rrf_top_k",
                (
                    "A retriever path surfaced gold evidence, but the fused RRF result list "
                    f"did not retain it. Path ranks: {present}."
                ),
            )
        return result(
            "rrf_loss",
            "rrf_top_k",
            (
                "Dense candidate generation contains a gold chunk inside the baseline "
                "candidate window, but the final prompt context does not contain gold evidence."
            ),
            confidence="measured_dense_then_inferred_fusion_loss",
        )

    if prep_row.get("prepared_has_gold"):
        return result(
            "unknown",
            "ingestion_audit",
            "Preparation sees a prepared hit, but ingestion audit did not localize the loss.",
            unknown_reason="ingestion_audit_gap",
        )

    return result(
        "unknown",
        "pipeline",
        (
            "Available artifacts do not prove whether evidence was lost in preparation, "
            "loading, chunking, dense retrieval, fusion, or context assembly."
        ),
        unknown_reason=ingestion_diagnosis or prep_diagnosis or "unclassified",
    )


def _stage_statuses(
    prep_row: dict[str, Any],
    ingestion_row: dict[str, Any],
    retrieval_row: dict[str, Any],
) -> dict[str, Any]:
    path_ranks = retrieval_row.get("path_gold_ranks")
    path_scores = retrieval_row.get("path_gold_scores")
    path_states = retrieval_row.get("retriever_path_states")
    path_latencies = retrieval_row.get("retriever_path_latencies_ms")
    return {
        "source_any_has_gold": prep_row.get("source_any_has_gold"),
        "source_preferred_has_gold": prep_row.get("source_preferred_has_gold"),
        "prepared_has_gold": prep_row.get("prepared_has_gold"),
        "loader_raw_has_gold": ingestion_row.get("raw_has_gold"),
        "loaded_has_gold": ingestion_row.get("loaded_has_gold"),
        "chunk_has_gold": ingestion_row.get("chunk_has_gold"),
        "indexed_has_gold": ingestion_row.get("indexed_has_gold"),
        "dense_topk_has_gold": ingestion_row.get("vector_topk_has_gold"),
        "dense_gold_rank": ingestion_row.get("vector_gold_rank"),
        "semantic_rank": (path_ranks or {}).get("semantic_rank"),
        "semantic_score": (path_scores or {}).get("semantic_score"),
        "keyword_rank": (path_ranks or {}).get("keyword_rank"),
        "keyword_score": (path_scores or {}).get("keyword_score"),
        "colbert_rank": (path_ranks or {}).get("colbert_rank"),
        "colbert_score": (path_scores or {}).get("colbert_score"),
        "graph_rank": (path_ranks or {}).get("graph_rank"),
        "graph_score": (path_scores or {}).get("graph_score"),
        "rrf_rank": (path_ranks or {}).get("rrf_rank"),
        "rrf_score": (path_scores or {}).get("rrf_score"),
        "final_context_rank": (path_ranks or {}).get("final_context_rank"),
        "retriever_path_states": path_states or {},
        "retriever_path_latencies_ms": path_latencies or {},
        "prompt_context_has_gold": retrieval_row.get("gold_answer_in_context"),
    }


def build_crag_vertical_audit_report(
    preparation_audit_path: str | os.PathLike[str],
    ingestion_audit_path: str | os.PathLike[str],
    retrieval_eval_path: str | os.PathLike[str],
    *,
    judge_eval_path: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    markdown_output: str | os.PathLike[str] | None = None,
    run_label: str = "",
    dense_candidate_k: int = 10,
    keyword_candidate_k: int = 20,
    colbert_candidate_k: int = 10,
    final_context_k: int = 5,
) -> dict[str, Any]:
    """Build a vertical CRAG audit report from measured artifacts."""

    prep_rows = _required_artifact_rows_by_interaction_id(
        preparation_audit_path,
        "preparation_audit",
    )
    ingestion_rows = _required_artifact_rows_by_interaction_id(
        ingestion_audit_path,
        "ingestion_audit",
    )
    retrieval_rows = _required_artifact_rows_by_interaction_id(
        retrieval_eval_path,
        "retrieval_eval",
    )
    judge_rows = _artifact_rows_by_interaction_id(judge_eval_path)
    interaction_ids = sorted(set(prep_rows) | set(ingestion_rows) | set(retrieval_rows))
    if not interaction_ids:
        raise ValueError("CRAG Vertical Audit hat keine joinbaren interaction_id-Zeilen")
    path_diagnostic_rows = sum(
        1 for row in retrieval_rows.values() if isinstance(row.get("path_gold_ranks"), dict)
    )
    path_diagnostics_available = path_diagnostic_rows > 0
    path_diagnostics_complete = path_diagnostic_rows >= len(interaction_ids)

    cases: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    loss_stage_counts: dict[str, int] = {}
    unknown_reason_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}

    for interaction_id in interaction_ids:
        prep_row = prep_rows.get(interaction_id, {})
        ingestion_row = ingestion_rows.get(interaction_id, {})
        retrieval_row = retrieval_rows.get(interaction_id, {})
        judge_row = judge_rows.get(interaction_id, {})
        answers = _combined_answers(prep_row, ingestion_row, retrieval_row, judge_row)
        classification = _classify_vertical_case(
            prep_row=prep_row,
            ingestion_row=ingestion_row,
            retrieval_row=retrieval_row,
            judge_row=judge_row,
            answers=answers,
            dense_candidate_k=dense_candidate_k,
        )
        failure_class = classification["failure_class"]
        loss_stage = classification["loss_stage"]
        _increment_count(failure_counts, failure_class)
        _increment_count(loss_stage_counts, loss_stage)
        if classification.get("unknown_reason"):
            _increment_count(unknown_reason_counts, str(classification["unknown_reason"]))

        warnings = []
        if _has_low_confidence_literal_answer(answers):
            warnings.append("short_or_numeric_answer")
        unknown_reason = str(classification.get("unknown_reason") or "")
        if not isinstance(retrieval_row.get("path_gold_ranks"), dict) and (
            failure_class == "rrf_loss"
            or (
                failure_class == "unknown"
                and unknown_reason not in {"invalid_question_label", "no_gold_answers"}
            )
        ):
            warnings.append("missing_keyword_colbert_rrf_path_diagnostics")
        for warning in warnings:
            _increment_count(warning_counts, warning)

        case = {
            "interaction_id": interaction_id,
            "query": str(
                prep_row.get("query")
                or ingestion_row.get("query")
                or retrieval_row.get("query")
                or judge_row.get("query")
                or ""
            ),
            "answers": answers,
            "failure_class": failure_class,
            "loss_stage": loss_stage,
            "preparation_diagnosis": str(prep_row.get("diagnosis") or ""),
            "ingestion_diagnosis": str(ingestion_row.get("diagnosis") or ""),
            "answer_label": _judge_label(judge_row),
            "rationale": classification["rationale"],
            "confidence": classification["confidence"],
            "unknown_reason": classification.get("unknown_reason", ""),
            "warnings": warnings,
            "stage_statuses": _stage_statuses(prep_row, ingestion_row, retrieval_row),
            "recommended_next_step": classification["recommended_next_step"],
        }
        cases.append(case)

    ranked_loss_points = [
        {"failure_class": key, "count": value}
        for key, value in sorted(
            failure_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if key not in {"no_loss_observed", "unknown"}
    ]
    if path_diagnostics_complete:
        diagnostic_gaps = {}
    else:
        diagnostic_gaps = {
            "path_diagnostics_coverage": (
                f"path diagnostics available for {path_diagnostic_rows}/{len(interaction_ids)} "
                "cases in the joined report"
            )
        }
        if not path_diagnostics_available:
            diagnostic_gaps.update(
                {
                    "keyword_bm25_rank": (
                        "not present in the current Full136 artifacts; cannot distinguish "
                        "keyword_rank_miss from no keyword recovery"
                    ),
                    "colbert_rank": (
                        "not present in the current Full136 artifacts; cannot distinguish "
                        "colbert_rank_miss from disabled/empty/failed ColBERT recovery"
                    ),
                    "rrf_inputs": (
                        "not present in the current Full136 artifacts; rrf_loss means dense "
                        "candidate evidence existed but final context lacked it"
                    ),
                    "final_context_chunk_ids": (
                        "retrieval eval stores only compact context evidence; exact chunk IDs "
                        "must be persisted for context assembly debugging"
                    ),
                }
            )
    sprint2_plan = [
        (
            "Persist per-result semantic_rank/score, keyword_rank/score, "
            "colbert_rank/score, rrf_rank/score and final_context_rank for Full136."
        ),
        (
            "Re-run Full136 after adding path diagnostics to split current rrf_loss "
            "and non-label unknown buckets into keyword_rank_miss, colbert_rank_miss, "
            "rrf_loss or context_assembly_loss."
        ),
        (
            "Review short_or_numeric_answer cases separately because literal contains "
            "matching can overcount evidence for answers such as numbers or units."
        ),
        (
            "Keep production retrieval config unchanged until the path-diagnostic "
            "rerun identifies a measured retrieval policy change."
        ),
    ]

    summary = {
        "schema": "kiara-crag-vertical-audit-report",
        "schema_version": 1,
        "run_label": run_label,
        "total": len(cases),
        "preparation_audit_path": str(preparation_audit_path),
        "ingestion_audit_path": str(ingestion_audit_path),
        "retrieval_eval_path": str(retrieval_eval_path),
        "judge_eval_path": str(judge_eval_path or ""),
        "candidate_windows": {
            "dense_candidate_k": dense_candidate_k,
            "keyword_candidate_k": keyword_candidate_k,
            "colbert_candidate_k": colbert_candidate_k,
            "final_context_k": final_context_k,
        },
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "loss_stage_counts": dict(sorted(loss_stage_counts.items())),
        "unknown_reason_counts": dict(sorted(unknown_reason_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "largest_loss_points": ranked_loss_points,
        "diagnostic_fields": {
            "path_diagnostics_available": path_diagnostics_available,
            "path_diagnostics_complete": path_diagnostics_complete,
            "path_diagnostic_rows": path_diagnostic_rows,
            "joined_case_count": len(interaction_ids),
            "paths": list(_RETRIEVAL_DIAGNOSTIC_PATHS),
            "ranks": [
                "semantic_rank",
                "keyword_rank",
                "colbert_rank",
                "graph_rank",
                "rrf_rank",
                "final_context_rank",
            ],
        },
        "diagnostic_gaps": diagnostic_gaps,
        "sprint2_plan": sprint2_plan,
    }
    payload = {"summary": summary, "cases": cases}

    if output_path:
        out = Path(output_path)
        write_private_text(out, _json_dumps(payload) + "\n")
    if markdown_output:
        out = Path(markdown_output)
        write_private_text(out, crag_vertical_audit_report_to_markdown(payload))
    return payload


def crag_vertical_audit_report_to_markdown(report: dict[str, Any]) -> str:
    """Render a CRAG vertical audit report as markdown."""

    summary = report.get("summary") or {}
    cases = [case for case in report.get("cases") or [] if isinstance(case, dict)]
    lines = [
        "# CRAG Full136 Vertical Audit",
        "",
        "## Summary",
        "",
        f"- Run label: `{summary.get('run_label', '')}`",
        f"- Total cases: {summary.get('total', len(cases))}",
        f"- Preparation audit: `{summary.get('preparation_audit_path', '')}`",
        f"- Ingestion audit: `{summary.get('ingestion_audit_path', '')}`",
        f"- Retrieval eval: `{summary.get('retrieval_eval_path', '')}`",
        f"- Judge eval: `{summary.get('judge_eval_path', '')}`",
        f"- Candidate windows: `{summary.get('candidate_windows', {})}`",
        f"- Path diagnostics available: `{(summary.get('diagnostic_fields') or {}).get('path_diagnostics_available')}`",
        "",
        "## Failure Classes",
        "",
    ]
    for key, value in (summary.get("failure_class_counts") or {}).items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Largest Loss Points", ""])
    for item in summary.get("largest_loss_points") or []:
        lines.append(f"- `{item.get('failure_class')}`: {item.get('count')}")

    unknown_counts = summary.get("unknown_reason_counts") or {}
    if unknown_counts:
        lines.extend(["", "## Unknown Reasons", ""])
        for key, value in unknown_counts.items():
            lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Diagnostic Gaps", ""])
    for key, value in (summary.get("diagnostic_gaps") or {}).items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Sprint 2 Plan", ""])
    for item in summary.get("sprint2_plan") or []:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Case Matrix",
            "",
            "| Class | Stage | ID | Judge | Dense rank | Prep | Ingestion | Confidence | Rationale |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for case in cases:
        statuses = case.get("stage_statuses") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(case.get("failure_class"), 80),
                    _markdown_cell(case.get("loss_stage"), 80),
                    _markdown_cell(case.get("interaction_id"), 48),
                    _markdown_cell(case.get("answer_label"), 40),
                    _markdown_cell(statuses.get("dense_gold_rank"), 20),
                    _markdown_cell(case.get("preparation_diagnosis"), 80),
                    _markdown_cell(case.get("ingestion_diagnosis"), 80),
                    _markdown_cell(case.get("confidence"), 80),
                    _markdown_cell(case.get("rationale"), 220),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _print_json(data: Any) -> None:
    print(_json_dumps(data))


def main_vertical_audit_report(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join CRAG audit artifacts into a vertical per-case report."
    )
    parser.add_argument("preparation_audit_path", help="Preparation audit JSONL")
    parser.add_argument("ingestion_audit_path", help="Ingestion/embedding audit JSONL")
    parser.add_argument("retrieval_eval_path", help="Retrieval evaluation JSON")
    parser.add_argument("--judge-eval", default=None, help="Optional judge evaluation JSON")
    parser.add_argument("--output", default=None, help="Optional JSON report output")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown report output")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--dense-candidate-k", type=int, default=10)
    parser.add_argument("--keyword-candidate-k", type=int, default=20)
    parser.add_argument("--colbert-candidate-k", type=int, default=10)
    parser.add_argument("--final-context-k", type=int, default=5)
    args = parser.parse_args(argv)
    report = build_crag_vertical_audit_report(
        args.preparation_audit_path,
        args.ingestion_audit_path,
        args.retrieval_eval_path,
        judge_eval_path=args.judge_eval,
        output_path=args.output,
        markdown_output=args.markdown_output,
        run_label=args.run_label,
        dense_candidate_k=args.dense_candidate_k,
        keyword_candidate_k=args.keyword_candidate_k,
        colbert_candidate_k=args.colbert_candidate_k,
        final_context_k=args.final_context_k,
    )
    _print_json(report["summary"])
    return 0


def main_prepare(task: int, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Prepare CRAG task {task} for KIara.")
    parser.add_argument("input_path", help="CRAG JSON/JSONL/BZ2/TAR file or directory")
    parser.add_argument("output_dir", help="Output directory for documents and manifest")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    stats = prepare_crag_task(
        args.input_path,
        args.output_dir,
        task,
        max_records=args.max_records,
        max_pages=args.max_pages,
        overwrite=args.overwrite,
    )
    _print_json(asdict(stats))
    return 0


def main_run(task: int, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Run KIara on prepared CRAG task {task}.")
    parser.add_argument("manifest_path", help="Prepared task manifest JSONL")
    parser.add_argument("output_path", help="Prediction JSONL output path")
    parser.add_argument("--config", default=None, help="KIara config path")
    parser.add_argument("--backend-id", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing prediction JSONL and skip completed interaction IDs",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="Optional per-question LLM HTTP timeout in seconds",
    )
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--think", dest="think", action="store_true")
    think_group.add_argument("--no-think", dest="think", action="store_false")
    parser.set_defaults(think=None)
    parser.add_argument(
        "--context-excerpt-chars",
        type=int,
        default=2000,
        help="Characters stored per retrieved context chunk for retrieval metrics",
    )
    parser.add_argument(
        "--scope",
        choices=["per-question", "global"],
        default="per-question",
        help="per-question filters retrieval to the CRAG documents of the current record",
    )
    args = parser.parse_args(argv)
    stats = run_crag_task(
        args.manifest_path,
        args.output_path,
        task,
        config_path=args.config,
        backend_id=args.backend_id,
        embedding_model=args.embedding_model,
        k=args.k,
        max_records=args.max_records,
        scope=args.scope,
        context_excerpt_chars=args.context_excerpt_chars,
        resume=args.resume,
        llm_timeout=args.llm_timeout,
        think=args.think,
    )
    _print_json(asdict(stats))
    return 0


def main_eval(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight CRAG prediction evaluator.")
    parser.add_argument("predictions_path", help="Prediction JSONL from crag_run_task*.py")
    parser.add_argument("--output", default=None, help="Optional detailed JSON output")
    args = parser.parse_args(argv)
    stats = evaluate_predictions(args.predictions_path, args.output)
    _print_json(asdict(stats))
    return 0


def main_eval_retrieval(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CRAG retrieval evidence.")
    parser.add_argument("predictions_path", help="Prediction JSONL from crag_run_task*.py")
    parser.add_argument("--output", default=None, help="Optional detailed JSON output")
    args = parser.parse_args(argv)
    stats = evaluate_retrieval(args.predictions_path, args.output)
    _print_json(asdict(stats))
    return 0


def main_export_official(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export CRAG predictions for official local eval.")
    parser.add_argument("predictions_path", help="Prediction JSONL from crag_run_task*.py")
    parser.add_argument("output_path", help="JSON output consumed by official-eval adapters")
    args = parser.parse_args(argv)
    stats = export_official_eval_payload(args.predictions_path, args.output_path)
    _print_json(asdict(stats))
    return 0


def main_eval_official_judge(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CRAG predictions with a configurable CRAG-rubric LLM judge."
    )
    parser.add_argument("predictions_path", help="Prediction JSONL from crag_run_task*.py")
    parser.add_argument("output_path", help="Detailed JSON judge result output")
    parser.add_argument("--config", default=None, help="KIara config path")
    parser.add_argument("--backend-id", type=int, default=None, help="KIara LLM/cloud backend ID")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional system prompt file; omit to use KIara's compact CRAG rubric prompt",
    )
    parser.add_argument(
        "--prediction-token-limit",
        type=int,
        default=75,
        help="Approximate word-token limit for predictions before judging; <=0 disables trimming",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Maximum completion tokens per judge call",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reasoning-effort",
        default="minimal",
        help="Optional OpenAI reasoning_effort for GPT/o-series judges; empty disables it",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=120.0,
        help="Wall-clock timeout per judge call in seconds; <=0 disables timeout",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate judge token usage; do not call the LLM backend",
    )
    args = parser.parse_args(argv)
    prediction_token_limit = (
        None
        if args.prediction_token_limit is not None and args.prediction_token_limit <= 0
        else args.prediction_token_limit
    )
    stats = evaluate_predictions_with_official_judge(
        args.predictions_path,
        args.output_path,
        config_path=args.config,
        backend_id=args.backend_id,
        max_records=args.max_records,
        prompt_path=args.prompt_file,
        prediction_token_limit=prediction_token_limit,
        max_completion_tokens=args.judge_max_tokens,
        temperature=args.temperature,
        judge_timeout=None if args.judge_timeout <= 0 else args.judge_timeout,
        reasoning_effort=args.reasoning_effort or None,
        estimate_only=args.estimate_only,
    )
    _print_json(asdict(stats))
    return 0


def main_summarize(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize CRAG benchmark results.")
    parser.add_argument("predictions_path", help="Prediction JSONL from crag_run_task*.py")
    parser.add_argument("--output", default=None, help="Optional JSON summary output")
    parser.add_argument("--run-label", default="", help="Human-readable run label")
    parser.add_argument("--scope", default=None, help="Run scope fallback for older predictions")
    parser.add_argument("--retrieval-mode", default=None, help="Retrieval mode fallback")
    parser.add_argument("--embedding-model", default=None, help="Embedding model fallback")
    parser.add_argument("-k", type=int, default=None, help="Retrieval top-k fallback")
    parser.add_argument(
        "--breakdown-field",
        action="append",
        default=None,
        help="Additional/explicit breakdown field. Can be repeated.",
    )
    args = parser.parse_args(argv)
    run_metadata = {
        key: value
        for key, value in {
            "label": args.run_label,
            "scope": args.scope,
            "retrieval_mode": args.retrieval_mode,
            "embedding_model": args.embedding_model,
            "k": args.k,
        }.items()
        if value not in (None, "")
    }
    payload = summarize_results(
        args.predictions_path,
        args.output,
        breakdown_fields=args.breakdown_field or DEFAULT_BREAKDOWN_FIELDS,
        run_metadata=run_metadata,
    )
    _print_json(payload["overall"])
    return 0


def main_compare(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CRAG summary JSON files.")
    parser.add_argument("summary_paths", nargs="+", help="Summary JSON files")
    parser.add_argument("--output", default=None, help="Optional comparison JSON output")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown table output")
    parser.add_argument("--baseline-index", type=int, default=0)
    args = parser.parse_args(argv)
    comparison = compare_summaries(
        args.summary_paths,
        args.output,
        baseline_index=args.baseline_index,
    )
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(comparison_to_markdown(comparison), encoding="utf-8")
    _print_json(
        {
            "runs": len(comparison["runs"]),
            "baseline_index": comparison["baseline_index"],
            "baseline_label": comparison["baseline_label"],
        }
    )
    return 0


def main_audit_ingestion_embeddings(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit CRAG ingestion, indexed chunks and dense embedding retrieval."
    )
    parser.add_argument("manifest_path", help="Prepared CRAG manifest JSONL")
    parser.add_argument("output_path", help="Per-record audit JSONL output path")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary output")
    parser.add_argument("--config", default=None, help="KIara config path")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("-k", type=int, default=DEFAULT_AUDIT_TOPK, help="Dense vector top-k")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--datasource-id", type=int, default=None)
    parser.add_argument(
        "--scope",
        choices=["per-question", "global"],
        default="per-question",
        help="per-question filters dense search to the CRAG doc_dir of each record",
    )
    args = parser.parse_args(argv)
    summary = audit_crag_ingestion_embeddings(
        args.manifest_path,
        args.output_path,
        config_path=args.config,
        embedding_model=args.embedding_model,
        k=args.k,
        max_records=args.max_records,
        scope=args.scope,
        datasource_id=args.datasource_id,
        summary_output=args.summary_output,
    )
    _print_json(summary)
    return 0


def main_audit_preparation(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether CRAG source retrieval content survived preparation."
    )
    parser.add_argument("raw_input_path", help="Original CRAG JSON/JSONL/BZ2/TAR file")
    parser.add_argument("manifest_path", help="Prepared CRAG manifest JSONL")
    parser.add_argument("output_path", help="Per-record preparation audit JSONL output path")
    parser.add_argument("--summary-output", default=None, help="Optional JSON summary output")
    parser.add_argument(
        "--page-limit",
        action="append",
        type=int,
        default=None,
        help="Top-N page coverage to report. Can be repeated. Default: 1,3,5,10.",
    )
    args = parser.parse_args(argv)
    summary = audit_crag_preparation_coverage(
        args.raw_input_path,
        args.manifest_path,
        args.output_path,
        page_limits=args.page_limit or DEFAULT_PREPARATION_PAGE_LIMITS,
        summary_output=args.summary_output,
    )
    _print_json(summary)
    return 0


def main_analyze_audits(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Join CRAG preparation and ingestion audits into case diagnostics."
    )
    parser.add_argument("preparation_audit_path", help="Preparation audit JSONL")
    parser.add_argument("ingestion_audit_path", help="Ingestion/embedding audit JSONL")
    parser.add_argument("--output", default=None, help="Optional JSON analysis output")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown report output")
    args = parser.parse_args(argv)
    analysis = analyze_crag_audit_cases(
        args.preparation_audit_path,
        args.ingestion_audit_path,
        args.output,
        markdown_output=args.markdown_output,
    )
    _print_json(analysis["summary"])
    return 0
