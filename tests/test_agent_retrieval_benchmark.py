"""Unabhaengige Referenzen, Messgrenzen und echte Agent-Kontextbeobachtung."""

import hashlib
import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from kiara.benchmarks.agent_dataset import (
    extract_reference,
    freeze_suite,
    load_suite,
    validate_suite,
)
from kiara.benchmarks.agent_retrieval import (
    ObservedAgentChain,
    answer_checks,
    evidence_metrics,
    source_texts,
)
from kiara.benchmarks.line_selection_baseline import select_lines
from kiara.generation.context_selection import prepare_context, prepare_document_read
from kiara.tools.registry import ToolRegistry


def document(doc_id="D01", text="Zulaessig sind 8 bar. Nicht fuer Trinkwasser.", split="holdout"):
    return {
        "id": doc_id,
        "source": f"/{doc_id}.txt",
        "family": doc_id,
        "split": split,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def case():
    return {
        "id": "C01",
        "split": "holdout",
        "kind": "direct",
        "question": "Druck und Einschraenkung?",
        "reference_answer": "8 bar, kein Trinkwasser",
        "evidence": [{"document_id": "D01", "quotes": ["8 bar", "Nicht fuer Trinkwasser."]}],
        "answer_checks": [["8 bar"], ["nicht fuer Trinkwasser", "kein Trinkwasser"]],
    }


def chunk(text, index=0, source="/D01.txt"):
    return {"text": text, "metadata": {"source": source, "chunk_index": index}}


def test_freeze_rejects_later_changes_and_overwrite(tmp_path):
    draft, frozen = tmp_path / "draft.json", tmp_path / "frozen.json"
    draft.write_text(json.dumps({"documents": [document()], "cases": [case()]}))
    suite = freeze_suite(draft, frozen)
    assert load_suite(frozen)["sha256"] == suite["sha256"]
    with pytest.raises(FileExistsError):
        freeze_suite(draft, frozen)
    suite["cases"][0]["question"] = "Andere Frage"
    frozen.write_text(json.dumps(suite))
    with pytest.raises(ValueError, match="veraendert"):
        load_suite(frozen)


def test_family_split_and_false_references_are_rejected():
    suite = {"documents": [document()], "cases": [case()]}
    bad = deepcopy(suite)
    other = document("D02", text="Anderer Text", split="development")
    other["family"] = "D01"
    bad["documents"].append(other)
    with pytest.raises(ValueError, match="Dokumentfamilie"):
        validate_suite(bad)
    bad = deepcopy(suite)
    bad["cases"][0]["evidence"][0]["quotes"] = ["10 bar"]
    with pytest.raises(ValueError, match="Beleg fehlt"):
        validate_suite(bad)
    bad = deepcopy(suite)
    bad["cases"][0]["split"] = "development"
    with pytest.raises(ValueError, match="vermischt"):
        validate_suite(bad)


def test_identical_content_cannot_cross_split():
    suite = {"documents": [document(), document("D02", split="development")], "cases": [case()]}
    with pytest.raises(ValueError, match="identischer Inhalt"):
        validate_suite(suite)


def test_negative_cases_need_a_reason_and_no_positive_evidence():
    item = {**case(), "expect_abstention": True}
    with pytest.raises(ValueError, match="Negativfall"):
        validate_suite({"documents": [document()], "cases": [item]})


def test_evidence_requires_correct_source_and_all_conditions():
    docs = {"D01": document()}
    wrong = evidence_metrics(case(), docs, [chunk(docs["D01"]["text"], source="/other.txt")])
    assert wrong["source_recall"] == 0
    assert wrong["evidence_recall"] == 0
    partial = evidence_metrics(case(), docs, [chunk("Zulaessig sind 8 bar.")])
    assert partial["source_recall"] == 1
    assert partial["evidence_recall"] == 0.5
    assert not partial["all_evidence"]


def test_quotes_may_cross_adjacent_chunks_but_not_missing_regions():
    shared = "Gemeinsame Passage mit langem Originaltext "
    chunks = [chunk("Beginn " + shared, 0), chunk(shared + "Ende", 1)]
    chunks[1]["metadata"].update(
        source_overlap_before=len(shared),
        source_gap_before="",
        source_body_start=0,
    )
    assert source_texts(chunks)["/D01.txt"] == ("Beginn " + shared + "Ende").casefold()
    chunks[1]["metadata"]["chunk_index"] = 3
    assert "[dokumentluecke]" in source_texts(chunks)["/D01.txt"]


def _read_pages(texts, budget=650):
    chunks = [
        {
            "text": text,
            "metadata": {
                "source": "/D01.txt",
                "parser": "standard",
                "document_ref": "doc_a",
                "chunk_index": index,
                "chunk_total": len(texts),
            },
        }
        for index, text in enumerate(texts)
    ]
    start, offset = 0, 0
    pages = []
    while start < len(chunks):
        prepared = prepare_document_read(
            chunks[start:],
            request={
                "document_id": "doc_a",
                "start_chunk": start,
                "start_offset": offset,
                "total_chunks": len(texts),
            },
            budget=budget,
            source_numbers={},
        )
        assert prepared.chunks
        pages.extend(prepared.chunks)
        reading = prepared.chunks[0]["metadata"]["document_read"]
        if not reading["has_more"]:
            break
        start, offset = reading["next_chunk"], reading["next_offset"]
        assert len(pages) < 100
    return pages


@pytest.mark.parametrize("chunk_count", [1, 3])
def test_evidence_crossing_read_cursors_is_recovered_in_full(chunk_count):
    from kiara.benchmarks.agent_dataset import normalized

    texts = [
        " ".join(f"Messwert{index}-{i:03d}" for i in range(65)) for index in range(chunk_count)
    ]
    pages = _read_pages(texts)
    assert len(pages) > chunk_count
    first = pages[0]["metadata"]["document_read"]
    assert first["end_chunk"] == 0 and first["next_chunk"] == 0
    boundary = first["end_offset"]
    quote = texts[0][boundary - 18 : boundary + 18]
    item = {"evidence": [{"document_id": "D01", "quotes": [quote]}]}
    docs = {"D01": document(text="\n".join(texts))}
    assert evidence_metrics(item, docs, pages[:1])["evidence_recall"] == 0
    assert evidence_metrics(item, docs, pages)["evidence_recall"] == 1
    assert source_texts([*reversed(pages), pages[0]])["/D01.txt"] == normalized("\n".join(texts))


def test_evidence_does_not_bridge_missing_read_offsets():
    text = " ".join(f"Eintrag{i:03d}" for i in range(140))
    pages = _read_pages([text])
    assert len(pages) > 3
    omitted = pages[1]["metadata"]["document_read"]
    quote = text[omitted["start_offset"] - 10 : omitted["end_offset"] + 10]
    item = {"evidence": [{"document_id": "D01", "quotes": [quote]}]}
    remaining = [pages[0], *pages[2:]]
    assert "[dokumentluecke]" in source_texts(remaining)["/D01.txt"]
    assert evidence_metrics(item, {"D01": document(text=text)}, remaining)["evidence_recall"] == 0


def test_repeated_text_only_loses_the_cursor_proven_overlap():
    text = "abcdabcdabcdEF"
    pages = []
    for start, end in [(0, 8), (4, 12), (12, 14)]:
        page = chunk(text[start:end])
        page["metadata"]["document_read"] = {
            "start_chunk": 0,
            "start_offset": start,
            "end_chunk": 0,
            "end_offset": end,
            "next_chunk": 0 if end < len(text) else None,
            "next_offset": end if end < len(text) else None,
        }
        page["metadata"]["document_read_spans"] = [
            {
                "chunk_index": 0,
                "clean_start": start,
                "clean_end": end,
                "clean_length": len(text),
                "text_start": 0,
                "text_end": end - start,
            }
        ]
        pages.append(page)
    assert source_texts(pages)["/D01.txt"] == text.casefold()


@pytest.mark.parametrize("field, value", [("parser", "vlm"), ("document_ref", "doc_new")])
def test_evidence_does_not_join_different_representations_or_versions(field, value):
    text = " ".join(f"Eintrag{i:03d}" for i in range(80))
    pages = _read_pages([text])
    boundary = pages[0]["metadata"]["document_read"]["end_offset"]
    quote = text[boundary - 12 : boundary + 12]
    pages[1]["metadata"][field] = value
    item = {"evidence": [{"document_id": "D01", "quotes": [quote]}]}
    assert evidence_metrics(item, {"D01": document(text=text)}, pages)["evidence_recall"] == 0


def test_answer_checks_remain_explicitly_provisional():
    checks = answer_checks(case(), "8 bar, kein Trinkwasser.")
    assert checks["all_passed"] is True
    assert checks["review_required"] is True


def test_trace_does_not_inject_references_or_change_prepared_context():
    config = {
        "chat": {"agent": {"final_reranking": False}},
        "retrieval": {"max_context_chunks": 80},
    }
    adapter = MagicMock()
    adapter.prepare_tool_text.side_effect = str
    chain = ObservedAgentChain(
        config, ToolRegistry(), adapter=adapter, system_prompt_override="Test"
    )
    raw = [chunk(document()["text"])]
    result = {"raw_chunks": raw, "params": {"query": "Druck"}}
    expected = prepare_context(raw, budget=24000, query="Druck", max_sources=80)
    text, selected, _ = chain._prepare_tool_result(result, "search_documents", "Druck", 0)
    assert text == expected.text
    assert selected == expected.chunks
    assert chain.context_records[0]["tool_result"] == result
    raw[0]["text"] = "Changed afterwards"
    assert chain.context_records[0]["tool_result"]["raw_chunks"][0]["text"] == document()["text"]


@pytest.mark.parametrize("budget", [1, 100, 500, 16000])
def test_historical_baseline_includes_separators_in_budget(budget):
    chunks = [
        chunk("Rechnung\nZu zahlender Betrag 15,00 EUR\n" * 30, source=f"/{i}.txt")
        for i in range(10)
    ]
    text, _ = select_lines(chunks, budget=budget, query="Rechnung")
    assert len(text) <= budget


def test_reference_reader_handles_utf16_without_replacement_characters(tmp_path):
    path = tmp_path / "reference.txt"
    path.write_bytes("Geräte: 8 bar".encode("utf-16"))
    text, parser = extract_reference(path)
    assert text == "Geräte: 8 bar"
    assert parser == "text:utf-16"
