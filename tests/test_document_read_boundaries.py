"""Erzeugerpositionen bis zu echten SDK-Nachrichten und Benchmark-Belegen."""

import asyncio
import json
from pathlib import Path

import pytest
from test_agent_context_wire import _config, _prepare, _tool_contents, wire  # noqa: F401
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_read_anonymization import _anon
from test_document_version_completion import document_run  # noqa: F401

from kiara.benchmarks.agent_dataset import normalized
from kiara.benchmarks.agent_retrieval import source_texts
from kiara.exceptions import RetrievalError
from kiara.generation.agent_chain import AgentChain
from kiara.generation.context_selection import prepare_context, prepare_document_read
from kiara.ingestion import runtime
from kiara.tools.registry import ToolRegistry


def _ingest_chunks(path, text, size=100, overlap=0):
    path.write_text(text)
    options = {"chunk_size": size, "chunk_overlap": overlap}
    texts, metas, *_ = runtime._process_single_file(
        str(path), {"ingestion": options}, options, None, runtime.ProcessingContext(), {"errors": 0}
    )
    return [
        {"text": t, "metadata": {**m, "document_ref": "doc_test"}}
        for t, m in zip(texts, metas, strict=True)
    ]


@pytest.mark.parametrize("overlap", [0, 20, 40])
@pytest.mark.parametrize("max_chunks", [1, 8])
def test_actual_chunker_preserves_repetitions_and_proven_overlap(tmp_path, overlap, max_chunks):
    text = "Einleitung. " + "Wiederholung " * 36 + " Schluss."
    raw = _ingest_chunks(tmp_path / "source.txt", text, overlap=overlap)
    request = {
        "document_id": "doc_test",
        "start_chunk": 0,
        "start_offset": 0,
        "total_chunks": len(raw),
    }
    pages = []
    for _ in range(80):
        start = request["start_chunk"]
        prepared = prepare_document_read(
            raw[start : start + max_chunks], request=request, budget=750, source_numbers={}
        )
        assert prepared.chunks and len(prepared.text) <= 750
        pages.extend(prepared.chunks)
        status = pages[-1]["metadata"]["document_read"]
        if not status["has_more"]:
            break
        request.update(start_chunk=status["next_chunk"], start_offset=status["next_offset"])
    else:
        pytest.fail("Lesecursor bleibt stehen")
    assert source_texts(pages)[str(tmp_path / "source.txt")] == normalized(text)
    assert prepare_context(raw, budget=10000).chunks[0]["text"] == text
    # Auch Suchbelege nutzen ausschliesslich den belegten Ueberlapp.
    assert source_texts(raw)[str(tmp_path / "source.txt")] == normalized(text)


@pytest.mark.parametrize("provenance", [False, True])
def test_equal_edges_without_proven_overlap_remain_repeated(provenance):
    shared = "Wiederkehrende echte Originalpassage. "
    chunks = [
        {"text": "START " + shared, "metadata": {"source": "x", "chunk_index": 0}},
        {"text": shared + "ENDE", "metadata": {"source": "x", "chunk_index": 1}},
    ]
    if provenance:
        chunks[1]["metadata"].update(
            source_overlap_before=0, source_gap_before="", source_body_start=0
        )
    assert prepare_context(chunks, budget=2000).text.count(shared) == 2
    assert source_texts(chunks)["x"].count(normalized(shared)) == 2


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("max_chunks", [1, 8])
@pytest.mark.parametrize("variant", ["name", "long_name", "unicode", "regex", "enriched"])
def test_entity_fragments_are_protected_through_real_reader_and_sdk(
    whole_pipeline, wire, provider, max_chunks, variant
):
    case = whole_pipeline
    name = {
        "name": "Eva Mueller",
        "enriched": "Eva Mueller",
        "long_name": "Eva " + "Langname " * 27 + "Mueller",
        "unicode": "E\u0301va Mueller",
        "regex": "eva.mueller@" + "firma." * 20 + "example.test",
    }[variant]
    # Ein festes Praefix erzwingt eine Grenze innerhalb der bekannten Entitaet.
    source = " ".join(["Sachtext"] * 10) + " " + name + " genehmigt. " + "Folgetext " * 26
    Path(case.path).write_text(source)
    case.prefix = "Einordnung: " if variant == "enriched" else ""
    result = case.run_whole(case.ds, chunk_size=100)
    assert result["processed_files"] == 1 and result["errors"] == 0
    saved = case.snapshot()
    assert len(saved) >= 3 and not any(name in c["text"] for c in saved)
    config = _config(30000, 750)
    anon = _anon(config, None if variant == "regex" else name)
    adapter, captured = wire(provider, config, anon)
    chain = AgentChain(config, ToolRegistry(), adapter)
    reader, ref = case.reader()
    request = {"document_id": ref, "start_chunk": 0, "start_offset": 0, "max_chunks": max_chunks}
    total, pages, selected = 0, [], []
    for i in range(80):
        result = asyncio.run(reader.execute(request))
        assert result.success
        # Interner Kontext ist kein zusaetzliches serialisiertes Suchergebnis.
        assert len(json.loads(json.dumps(result.data))["results"]) <= max_chunks
        data = {"raw_chunks": result.data["results"], "document_read": result.data["document_read"]}
        display, chunks, total = _prepare(chain, data, "read_document", total, f"read-{i}")
        assert chunks and len(display) <= 750
        pages.append(display)
        selected.extend(chunks)
        adapter.step()
        status = data["document_read"]
        if not status["has_more"]:
            break
        request.update(start_chunk=status["next_chunk"], start_offset=status["next_offset"])
    else:
        pytest.fail("Kein Dokumentende")
    list(adapter.stream_final())
    assert _tool_contents(captured[-1], provider) == pages
    assert total == sum(map(len, pages)) <= 30000
    for body in captured:
        contents = "".join(_tool_contents(body, provider))
        assert "Eva" not in contents and "Mueller" not in contents and "E\u0301va" not in contents
        if variant == "regex":
            assert "eva.mueller" not in contents and "example.test" not in contents
    assert "‹ANON:" in "".join(pages)
    if variant not in {"enriched", "regex"}:
        # Eine einzige ueber Chunkgrenzen laufende Person bleibt pro Seite
        # eine Entitaet. Einzelne Folgeseiten koennen dasselbe Pseudonym tragen.
        assert all(page.count("‹ANON:Person_001›") <= 1 for page in pages)
    if variant != "enriched":
        assert source_texts(selected)[case.path] == normalized(source)


@pytest.mark.parametrize("fault", ["version", "membership", "point"])
def test_anonymization_context_checks_unrequested_chunks(whole_pipeline, wire, test_db, fault):
    case = whole_pipeline
    case.prefix = ""
    Path(case.path).write_text(" ".join(["Sachtext"] * 10) + " Eva Mueller " + "Folgetext " * 25)
    assert case.run_whole(case.ds)["processed_files"] == 1
    reader, ref = case.reader()
    result = asyncio.run(reader.execute({"document_id": ref, "max_chunks": 1}))
    missing = case.snapshot()[1]
    if fault == "membership":
        test_db.execute("DELETE FROM chunk_datasources WHERE chunk_id=%s", (missing["_doc_id"],))
        test_db.commit()
    elif fault == "point":
        case.store.delete_by_ids([missing["_doc_id"]])
    else:
        case.client.set_payload(
            case.store.get_collection_name(),
            {"processing_version": "changed"},
            [missing["_doc_id"]],
        )
    config = _config(5000, 750)
    adapter, captured = wire("openai", config, _anon(config, "Eva Mueller"))
    chain = AgentChain(config, ToolRegistry(), adapter)
    with pytest.raises(RetrievalError):
        _prepare(
            chain,
            {"raw_chunks": result.data["results"], "document_read": result.data["document_read"]},
            "read_document",
            0,
            "read-0",
        )
    assert captured == []
