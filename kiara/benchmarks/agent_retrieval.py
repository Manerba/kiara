"""Agent-Evaluation: Suchtreffer, vorbereitete Belege und Modellantwort getrennt."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from kiara.benchmarks.agent_dataset import (
    digest,
    freeze_suite,
    load_suite,
    normalized,
    sample_documents,
)
from kiara.benchmarks.artifacts import write_private_json
from kiara.generation.agent_chain import AgentChain
from kiara.generation.context_selection import (
    _chunk_join,
    _join_adjacent,
    document_key,
    prepare_context,
)
from kiara.shared.processing_warnings import ProcessingWarnings, processing_warning_scope
from kiara.shared.prompt_defense import sanitize_with_report


class _DeadlineExceeded(BaseException):
    """Die Agent-Retries duerfen die Deadline des Benchmarks nicht verschlucken."""


@contextmanager
def deadline(seconds: float):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(_signum, _frame):
        raise _DeadlineExceeded(f"Fall nach {seconds:g} Sekunden abgebrochen")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.001, previous_timer[0] - (time.monotonic() - started)),
                previous_timer[1],
            )


def _chunk_position(chunk: dict) -> tuple[int, int]:
    meta = chunk.get("metadata", {})
    reading = meta.get("document_read")
    if reading:
        return reading["start_chunk"], reading["start_offset"]
    index = meta.get("chunk_index")
    return (index if type(index) is int else -1), 0


def _join_read_pages(chunks: list[dict]) -> str:
    """Vereinigt Leseseiten in den Koordinaten vollstaendig bereinigter Chunks."""
    by_index = defaultdict(list)
    lengths = {}
    joins = {}
    for chunk in chunks:
        text = chunk["text"]
        spans = chunk["metadata"].get("document_read_spans")
        if not spans:
            raise ValueError("Leseseite ohne Zuordnung zu bereinigten Chunkpositionen")
        for span in spans:
            index = span["chunk_index"]
            start, end = span["clean_start"], span["clean_end"]
            low, high = span["text_start"], span["text_end"]
            length = span["clean_length"]
            join = span.get("join_before", {"overlap": 0, "separator": "\n"})
            if not (
                0 <= start <= end <= length
                and 0 <= low <= high <= len(text)
                and high - low == end - start
                and lengths.setdefault(index, length) == length
                and joins.setdefault(index, join) == join
            ):
                raise ValueError("Widerspruechliche Positionsmetadaten einer Leseseite")
            by_index[index].append((start, end, text[low:high]))

    fragments = []
    for index, spans in sorted(by_index.items()):
        merged = []
        for start, end, text in sorted(spans, key=lambda span: (span[0], -span[1])):
            if not merged or start > merged[-1][1]:
                merged.append((start, end, text))
                continue
            old_start, old_end, old_text = merged[-1]
            shared = min(old_end, end) - start
            if old_text[start - old_start : start - old_start + shared] != text[:shared]:
                raise ValueError("Widerspruechlicher Text an derselben Dokumentposition")
            if end > old_end:
                merged[-1] = old_start, end, old_text + text[shared:]
        fragments.extend((index, start, end, text) for start, end, text in merged)

    parts, previous = [], None
    for index, start, end, text in fragments:
        original = text
        if previous is not None:
            old_index, old_end, old_text = previous
            adjacent = index == old_index + 1 and old_end == lengths[old_index] and start == 0
            if adjacent:
                overlap, separator = _join_adjacent(old_text, text, joins[index])
                text = text[overlap:]
                if text and any(parts):
                    parts.append(separator)
            else:
                parts.append("\n[Dokumentluecke]\n")
        parts.append(text)
        previous = index, end, original
    # Erst anhand der stabilen Chunkpositionen vereinigen, dann dieselbe
    # Koerperbereinigung wie beim Rendern anwenden. Ein an der Chunkgrenze
    # entferntes Steuer-Tag darf auch kein Benchmark-Beleg werden.
    return sanitize_with_report("".join(parts)).clean_text


def _join_evidence_chunks(chunks: list[dict]) -> str:
    if chunks and chunks[0]["metadata"].get("document_read"):
        return _join_read_pages(chunks)
    pieces, previous, seen = [], None, set()
    for chunk in sorted(chunks, key=_chunk_position):
        text = str(chunk.get("text", ""))
        position = _chunk_position(chunk)
        key = position, text
        if key in seen:
            continue
        seen.add(key)
        if previous is not None:
            old_position, old_text = previous
            adjacent = old_position[0] >= 0 and position[0] == old_position[0] + 1
            overlap, separator = (
                _join_adjacent(old_text, text, _chunk_join(chunk, raw=True))
                if adjacent
                else (0, "\n[Dokumentluecke]\n")
            )
            text = text[overlap:]
            pieces.append(separator)
        pieces.append(text)
        previous = position, str(chunk.get("text", ""))
    text = "".join(pieces)
    if any("context_spans" in chunk["metadata"] for chunk in chunks):
        text = sanitize_with_report(text).clean_text
    return text


def source_texts(chunks: list[dict]) -> dict[str, str]:
    """Rekonstruiert Belege aus Chunkpositionen und exakten Lesecursorn."""
    groups = defaultdict(list)
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        # Unterschiedliche Darstellungen/Versionen duerfen keine gemeinsame
        # Belegpassage bilden, auch wenn der Quellenpfad identisch ist.
        key = document_key(chunk), meta.get("document_ref"), bool(meta.get("document_read"))
        groups[key].append(chunk)
    texts = defaultdict(list)
    for (key, _, _), group in groups.items():
        texts[key[0]].append(_join_evidence_chunks(group))
    return {
        source: normalized("\n[Dokumentluecke]\n".join(parts)) for source, parts in texts.items()
    }


def evidence_metrics(case: dict, documents: dict, chunks: list[dict]) -> dict:
    texts = source_texts(chunks)
    hits, source_hits, required = [], [], []
    for evidence in case.get("evidence", []):
        doc_id = evidence["document_id"]
        doc = documents[doc_id]
        sources = [doc["source"], *doc.get("source_aliases", [])]
        matching = [texts[source] for source in sources if source in texts]
        if matching:
            source_hits.append(doc_id)
        for index, quote in enumerate(evidence["quotes"]):
            key = f"{doc_id}:{index}"
            required.append(key)
            if any(normalized(quote) in text for text in matching):
                hits.append(key)
    required_sources = {e["document_id"] for e in case.get("evidence", [])}
    return {
        "source_hits": sorted(set(source_hits)),
        "evidence_hits": hits,
        "required_evidence": required,
        "source_count": len(texts),
        "source_recall": len(set(source_hits)) / len(required_sources)
        if required_sources
        else None,
        "evidence_recall": len(hits) / len(required) if required else None,
        "all_evidence": len(hits) == len(required) if required else None,
    }


def answer_checks(case: dict, answer: str) -> dict:
    """Mechanische Vorpruefung; semantische Richtigkeit braucht eine Belegpruefung."""
    checks = case.get("answer_checks", [])
    text = normalized(answer)
    passed = [any(normalized(variant) in text for variant in variants) for variants in checks]
    return {
        "checks": passed,
        "all_passed": all(passed) if passed else None,
        "review_required": True,
    }


class ObservedRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.retriever, name)

    def retrieve(self, *args, **kwargs):
        record = {"args": args, "params": deepcopy(kwargs)}
        self.calls.append(record)
        started = time.monotonic()
        try:
            result = self.retriever.retrieve(*args, **kwargs)
            record["chunks"] = deepcopy(result)
            return result
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["latency_ms"] = round((time.monotonic() - started) * 1000)
            getter = getattr(self.retriever, "get_last_retrieval_diagnostics", None)
            record["diagnostics"] = deepcopy(getter()) if getter else None


class ObservedAgentChain(AgentChain):
    """Beobachtet die unveraenderte zentrale Kontextaufbereitung."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_records = []

    def _prepare_tool_result(self, result, tool_name, query, total_result_chars, tool_use_id=None):
        original = deepcopy(result)
        display, chunks, total = super()._prepare_tool_result(
            result, tool_name, query, total_result_chars, tool_use_id=tool_use_id
        )
        self.context_records.append(
            {
                "tool": tool_name,
                "query": query,
                "tool_result": original,
                "prepared_text": display,
                "prepared_chunks": deepcopy(chunks),
                "input_chars": sum(len(c.get("text", "")) for c in original.get("raw_chunks", [])),
                "output_chars": len(display),
            }
        )
        return display, chunks, total


class AgentRuntime:
    """Frische Tool-/Adapter-Zustaende, gleiche Bausteine und Prompts wie im Chat."""

    def __init__(self, backend_id: int, config_path: str | None = None):
        from kiara.backend_selector import get_backend_by_id
        from kiara.config import get_embedding_model, load_config
        from kiara.repositories.prompt import PromptTemplateRepository
        from kiara.retrieval.reranker import create_reranker
        from kiara.retrieval.retriever import HybridRetriever
        from kiara.retrieval.source_policy import resolve_chat_sources

        self.config, self.k = resolve_chat_sources(load_config(config_path), "agent", None)
        self.backend = get_backend_by_id(backend_id)
        if not self.backend or not self.backend.get("is_active"):
            raise ValueError("Aktives LLM-Backend erforderlich")
        if self.backend["provider"] != "ollama":
            raise ValueError("Dieser Agent-Benchmark unterstuetzt lokale Ollama-Backends")
        self.template = PromptTemplateRepository().get_by_name("system_agent")
        if not self.template or not self.template.get("content"):
            raise ValueError("Agent-Prompt fehlt")
        self.embedding_model = get_embedding_model(self.config)
        self.retriever = HybridRetriever(self.config, embedding_model=self.embedding_model)
        self.reranker = create_reranker(self.config)

    def search_tool(self, observed, references=None):
        from kiara.tools.builtin_tools import SearchDocumentsTool

        return SearchDocumentsTool(
            observed, self.reranker, self.config, default_k=self.k, document_references=references
        )

    def chain(self, observed):
        from kiara.generation.tool_adapter import create_tool_adapter
        from kiara.shared.constants import AGENT_MAX_ITERATIONS_OLLAMA
        from kiara.tools.builtin_tools import CalculateTool, ListDatasourcesTool
        from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool
        from kiara.tools.registry import ToolRegistry, load_config_tool_permissions

        registry = ToolRegistry(permission_provider=load_config_tool_permissions)
        references = DocumentReferences(str(observed._store.get_collection_name()))
        registry.register(self.search_tool(observed, references))
        registry.register(ReadDocumentTool(observed, self.config, references))
        registry.register(ListDatasourcesTool())
        registry.register(CalculateTool())
        adapter = create_tool_adapter(self.backend, self.config)
        if self.template.get("final_prompt"):
            adapter._final_prompt_override = self.template["final_prompt"]
        return ObservedAgentChain(
            self.config,
            registry,
            adapter=adapter,
            user_groups=["*"],
            max_iterations=AGENT_MAX_ITERATIONS_OLLAMA,
            system_prompt_override=self.template["content"],
        ), adapter

    def fingerprint(self) -> dict:
        import httpx

        response = httpx.get(self.backend["url"].rstrip("/") + "/api/tags", timeout=20)
        response.raise_for_status()
        models = [
            m for m in response.json().get("models", []) if m.get("name") == self.backend["model"]
        ]
        if len(models) != 1 or not models[0].get("digest"):
            raise ValueError("Ollama-Modellversion konnte nicht eindeutig bestimmt werden")
        return {
            "config_sha256": digest(self.config),
            "template_sha256": digest(self.template),
            "backend": {key: self.backend.get(key) for key in ("id", "provider", "model", "url")},
            "embedding_model": self.embedding_model,
            "model_digest": models[0]["digest"],
            "source_k": self.k,
            "agent_config": self.config.get("chat", {}).get("agent", {}),
            "retrieval_config": self.config.get("retrieval", {}),
            "scope": "all sources available to administrator; no reference-derived filters",
            "tools": ["search_documents", "read_document", "list_datasources", "calculate"],
        }


def run_case(runtime: AgentRuntime, case: dict, documents: dict, *, timeout: float) -> dict:
    started = time.monotonic()
    observed = ObservedRetriever(runtime.retriever)
    collector = ProcessingWarnings()
    chain, adapter = runtime.chain(observed)
    events, answer, error = [], [], None
    try:
        with deadline(timeout), processing_warning_scope(collector):
            for event in chain.query_stream(case["question"], think=False):
                events.append({"type": event.event_type, "data": event.data})
                if event.event_type == "chunk":
                    answer.append(str((event.data or {}).get("text", "")))
                if event.event_type == "error":
                    error = str(event.data)
    except (_DeadlineExceeded, Exception) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        client = getattr(adapter, "_client", None)
        http_client = getattr(client, "_client", None)
        if http_client is not None:
            http_client.close()
    raw = [c for r in chain.context_records for c in r["tool_result"].get("raw_chunks", [])]
    prepared = [c for r in chain.context_records for c in r["prepared_chunks"]]
    raw_metrics = evidence_metrics(case, documents, raw)
    prepared_metrics = evidence_metrics(case, documents, prepared)
    prompts = json.loads(adapter.last_conversation or "[]")
    final_prompts = [p["prompt"] for p in prompts if p.get("type") == "final"]
    delivered = [
        c
        for record in chain._document_context_results
        if record["text"] and any(record["text"] in prompt for prompt in final_prompts)
        for c in record["chunks"]
    ]
    joined = "".join(answer)
    return {
        "id": case["id"],
        "split": case["split"],
        "kind": case["kind"],
        "question": case["question"],
        "answer": joined,
        "error": error,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "warnings": collector.snapshot(),
        "events": events,
        "retrieval_calls": observed.calls,
        "context_records": chain.context_records,
        "active_context": deepcopy(chain._document_context_results),
        "sent_prompts": prompts,
        "usage": adapter.last_usage,
        "metrics": {
            "retrieved": raw_metrics,
            "prepared": prepared_metrics,
            "delivered": evidence_metrics(case, documents, delivered),
        },
        "lost_evidence": sorted(
            set(raw_metrics["evidence_hits"]) - set(prepared_metrics["evidence_hits"])
        ),
        "answer_check": answer_checks(case, joined),
    }


def run_direct(
    runtime: AgentRuntime, case: dict, documents: dict, *, timeout: float, budgets: list[int]
) -> dict:
    from kiara.benchmarks.line_selection_baseline import select_lines

    observed = ObservedRetriever(runtime.retriever)
    collector = ProcessingWarnings()
    raw, error, replay = [], None, []
    started = time.monotonic()
    try:
        with deadline(timeout), processing_warning_scope(collector):
            result = asyncio.run(runtime.search_tool(observed).execute({"query": case["question"]}))
            if not result.success:
                raise RuntimeError(result.error)
            raw = (result.data or {}).get("results", [])
            for budget in budgets:
                selected = prepare_context(
                    raw,
                    budget=budget,
                    query=case["question"],
                    max_sources=runtime.config["retrieval"]["max_context_chunks"],
                    source_diversity=runtime.config["chat"]["agent"].get(
                        "raw_chunk_compaction", True
                    ),
                    reranker=runtime.reranker
                    if runtime.config["chat"]["agent"].get("final_reranking")
                    else None,
                )
                replay.append(
                    {
                        "budget": budget,
                        "selector": "current",
                        "output_chars": len(selected.text),
                        "text": selected.text,
                        "chunks": selected.chunks,
                        "metrics": evidence_metrics(case, documents, selected.chunks),
                    }
                )
                baseline_text, baseline_chunks = select_lines(
                    raw, budget=budget, query=case["question"]
                )
                replay.append(
                    {
                        "budget": budget,
                        "selector": "lines_d37c2854",
                        "output_chars": len(baseline_text),
                        "text": baseline_text,
                        "chunks": baseline_chunks,
                        "metrics": evidence_metrics(case, documents, baseline_chunks),
                    }
                )
    except (_DeadlineExceeded, Exception) as exc:
        error = f"{type(exc).__name__}: {exc}"
    candidates = observed.calls[0].get("chunks", []) if observed.calls else []
    return {
        "id": case["id"],
        "split": case["split"],
        "kind": case["kind"],
        "error": error,
        "warnings": collector.snapshot(),
        "raw_chunks": raw,
        "retrieval_calls": observed.calls,
        "replay": replay,
        "latency_ms": round((time.monotonic() - started) * 1000),
        "metrics": evidence_metrics(case, documents, raw),
        "top_k": {str(k): evidence_metrics(case, documents, candidates[:k]) for k in (5, 10, 20)},
    }


def summarize(output: Path) -> dict:
    rows = [json.loads(path.read_text()) for path in sorted(output.glob("*-agent-*.json"))]
    direct = [json.loads(path.read_text()) for path in sorted(output.glob("*-direct.json"))]
    groups = {}
    for split in ("all", "development", "holdout"):
        selected = [r for r in rows if split == "all" or r["split"] == split]
        answerable = [r for r in selected if r["metrics"]["retrieved"]["all_evidence"] is not None]
        groups[split] = {
            "runs": len(selected),
            "answerable_runs": len(answerable),
            "errors": sum(bool(r["error"]) for r in selected),
            "degraded_runs": sum(bool(r["warnings"]) for r in selected),
            "healthy_runs": sum(not r["error"] and not r["warnings"] for r in selected),
            "all_retrieved": sum(r["metrics"]["retrieved"]["all_evidence"] for r in answerable),
            "all_prepared": sum(r["metrics"]["prepared"]["all_evidence"] for r in answerable),
            "all_delivered": sum(r["metrics"]["delivered"]["all_evidence"] for r in answerable),
            "runs_with_evidence_loss": sum(bool(r["lost_evidence"]) for r in selected),
            "answer_checks_passed": sum(
                r["answer_check"]["all_passed"] is True and not r["error"] for r in selected
            ),
        }
    summary = {
        "agent": groups,
        "direct_runs": len(direct),
        "direct_errors": sum(bool(r["error"]) for r in direct),
        "note": "Quote recall against fixed references; answer checks are not semantic accuracy. Review required.",
    }
    write_private_json(output / "summary.json", summary)
    return summary


def snapshot_index(runtime: AgentRuntime, suite: dict) -> dict:
    """Direkter Quellenabruf nur fuer den Evaluator, niemals fuer den Agenten."""
    from kiara.retrieval.vectorstore import VectorStore

    store = VectorStore(runtime.config, embedding_model=runtime.embedding_model)
    documents = {d["id"]: d for d in suite["documents"]}
    chunks = []
    for document in suite["documents"]:
        for source in [document["source"], *document.get("source_aliases", [])]:
            data = store.get_documents_by_source(source)
            chunks.extend(
                {"id": chunk_id, "text": text, "metadata": metadata}
                for chunk_id, text, metadata in zip(
                    data["ids"], data["documents"], data["metadatas"], strict=True
                )
            )
    return {
        "collection": store.get_collection_name(),
        "collections": VectorStore.list_collections(runtime.config),
        "chunks": chunks,
        "chunks_sha256": digest(chunks),
        "metrics": {
            case["id"]: evidence_metrics(case, documents, chunks) for case in suite["cases"]
        },
    }


def run_suite(
    suite_path: Path,
    output: Path,
    *,
    backend_id: int,
    config_path: str | None,
    split: str,
    repeats: int,
    timeout: float,
    budgets: list[int],
    resume: bool,
) -> dict:
    suite = load_suite(suite_path)
    documents = {d["id"]: d for d in suite["documents"]}
    runtime = AgentRuntime(backend_id, config_path)
    code_root = Path(__file__).resolve().parents[2]
    code_hash = digest(
        {
            str(path.relative_to(code_root)): path.read_text()
            for path in sorted((code_root / "kiara").rglob("*.py"))
        }
    )
    metadata = {
        "suite_sha256": suite["sha256"],
        "runtime": runtime.fingerprint(),
        "code_sha256": code_hash,
        "split": split,
        "repeats": repeats,
        "timeout": timeout,
        "budgets": budgets,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
        ).strip(),
    }
    if output.exists():
        if not resume or json.loads((output / "run.json").read_text())["fingerprint"] != digest(
            metadata
        ):
            raise ValueError("Lauf existiert bereits oder Konfiguration/Code wurde geaendert")
    else:
        output.mkdir(parents=True)
        write_private_json(
            output / "run.json",
            {
                **metadata,
                "fingerprint": digest(metadata),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    if not (output / "index.json").exists():
        write_private_json(output / "index.json", snapshot_index(runtime, suite))
    cases = [c for c in suite["cases"] if split == "all" or c["split"] == split]
    for index, case in enumerate(cases, 1):
        direct_path = output / f"{case['id']}-direct.json"
        if not direct_path.exists():
            result = run_direct(runtime, case, documents, timeout=timeout, budgets=budgets)
            write_private_json(direct_path, result)
            print(
                f"{index}/{len(cases)} {case['id']} direct error={bool(result['error'])}",
                flush=True,
            )
        for repeat in range(1, case.get("repetitions", repeats) + 1):
            agent_path = output / f"{case['id']}-agent-{repeat}.json"
            if agent_path.exists():
                continue
            result = run_case(runtime, case, documents, timeout=timeout)
            result["repeat"] = repeat
            write_private_json(agent_path, result)
            summarize(output)
            print(
                f"{index}/{len(cases)} {case['id']} agent={repeat} "
                f"evidence={result['metrics']['delivered']['evidence_recall']} "
                f"checks={result['answer_check']['all_passed']} error={bool(result['error'])}",
                flush=True,
            )
    return summarize(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--quotas", required=True, help="DS-ID=Anzahl,DS-ID=Anzahl")
    sample.add_argument("--seed", required=True)
    sample.add_argument("--embedding-model", required=True)
    sample.add_argument("--mount-map", type=Path)
    sample.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("draft", type=Path)
    freeze.add_argument("output", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--suite", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend-id", type=int, required=True)
    run.add_argument("--config")
    run.add_argument("--split", choices=("all", "development", "holdout"), default="development")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--budgets", default="16000,24000")
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "sample":
        result = sample_documents(
            quotas={
                int(k): int(v) for k, v in (item.split("=") for item in args.quotas.split(","))
            },
            seed=args.seed,
            embedding_model=args.embedding_model,
            output=args.output,
            mount_map=json.loads(args.mount_map.read_text()) if args.mount_map else None,
        )
        print(
            json.dumps({"documents": len(result["documents"]), "excluded": len(result["excluded"])})
        )
    elif args.command == "freeze":
        suite = freeze_suite(args.draft, args.output)
        print(json.dumps({"sha256": suite["sha256"], "cases": len(suite["cases"])}))
    else:
        budgets = [int(value) for value in args.budgets.split(",")]
        if args.repeats < 1 or args.timeout <= 0 or any(b < 1 for b in budgets):
            parser.error("Positive Wiederholungen, Timeout und Budgets erforderlich")
        logging.basicConfig(level=logging.WARNING)
        print(
            json.dumps(
                run_suite(
                    args.suite,
                    args.output,
                    backend_id=args.backend_id,
                    config_path=args.config,
                    split=args.split,
                    repeats=args.repeats,
                    timeout=args.timeout,
                    budgets=budgets,
                    resume=args.resume,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
