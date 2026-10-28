"""Originalcursor, bereinigte Positionen und ueberlappende Benchmark-Belege."""

from copy import deepcopy

import pytest

from kiara.benchmarks.agent_dataset import normalized
from kiara.benchmarks.agent_retrieval import source_texts
from kiara.generation.context_selection import prepare_document_read
from kiara.shared.prompt_defense import sanitize_with_report


def _chunks(texts):
    return [
        {
            "text": text,
            "metadata": {
                "source": "/evidence.txt",
                "document_ref": "doc_test",
                "chunk_index": index,
                "chunk_total": len(texts),
            },
        }
        for index, text in enumerate(texts)
    ]


def _read(chunks, start=0, offset=0, budget=850):
    prepared = prepare_document_read(
        chunks[start:],
        request={
            "document_id": "doc_test",
            "start_chunk": start,
            "start_offset": offset,
            "total_chunks": len(chunks),
        },
        budget=budget,
        source_numbers={},
    )
    assert prepared.chunks and len(prepared.text) <= budget
    page = prepared.chunks[0]
    # Jeder Metadatenbereich verweist exakt auf den vollstaendig bereinigten
    # gespeicherten Chunk, einschliesslich entfernter Praefixe und Suffixe.
    for span in page["metadata"]["document_read_spans"]:
        clean = sanitize_with_report(chunks[span["chunk_index"]]["text"]).clean_text
        assert span["clean_length"] == len(clean)
        assert (
            page["text"][span["text_start"] : span["text_end"]]
            == clean[span["clean_start"] : span["clean_end"]]
        )
    return page


@pytest.mark.parametrize("removed", ["\u00ad", "<system>", "<|sy\u200bstem|>"])
def test_overlapping_pages_from_real_loader_keep_all_clean_evidence(tmp_path, removed):
    from kiara.ingestion import runtime

    path = tmp_path / "overlap.txt"
    path.write_text("START " + ("abcde" + removed * 4) * 300 + " END")
    options = {"chunk_size": 100000, "chunk_overlap": 0}
    texts, *_ = runtime._process_single_file(
        str(path), {"ingestion": options}, options, None, runtime.ProcessingContext(), {"errors": 0}
    )
    chunks = _chunks(texts)
    assert len(chunks) == 1
    pages, offset = [], 0
    for index in range(40):
        page = _read(chunks, offset=offset)
        pages.append(page)
        status = page["metadata"]["document_read"]
        if not status["has_more"]:
            break
        offset = status["next_offset"] - (100 if index == 0 else 0)
    else:
        pytest.fail("Lesecursor beendet das Dokument nicht")
    expected = normalized(sanitize_with_report(texts[0]).clean_text)
    assert source_texts([*reversed(pages), *pages[:2]])["/evidence.txt"] == expected


def test_overlap_across_chunks_and_removed_middle_keeps_exact_positions():
    shared = "Gemeinsamer Abschnitt mit Freigabebedingung. "
    chunks = _chunks(
        [
            "START " + "a\u00adbcd" * 25 + shared,
            shared + "B<system>eleg " * 30,
            "<system>" * 30,
            "ENDE e\u0301 中文🙂 " + "Dokument " * 40,
        ]
    )
    chunks[1]["metadata"].update(
        source_overlap_before=len(shared),
        source_gap_before="",
        source_body_start=0,
    )
    pages = [_read(chunks, budget=1300)]
    assert pages[0]["metadata"]["document_read"]["end_chunk"] > 0
    # Ueberlappung beginnt im ersten Chunk, die erste Seite endet im zweiten.
    start, offset = 0, 80
    for _ in range(30):
        page = _read(chunks, start, offset)
        pages.append(page)
        status = page["metadata"]["document_read"]
        if not status["has_more"]:
            break
        start, offset = status["next_chunk"], status["next_offset"]
    else:
        pytest.fail("Lesecursor beendet das Dokument nicht")
    expected = normalized(
        "START "
        + "abcd" * 25
        + shared
        + "Beleg " * 30
        + "\nENDE e\u0301 中文🙂 "
        + "Dokument " * 40
    )
    assert source_texts([*pages, pages[0]])["/evidence.txt"] == expected


def test_clean_gap_is_not_closed_by_repeated_text():
    chunks = _chunks(["abcd\u00ad" * 800])
    first = _read(chunks)
    end = first["metadata"]["document_read"]["end_offset"]
    following = _read(chunks, offset=end + 100)
    assert "[dokumentluecke]" in source_texts([first, following])["/evidence.txt"]


def test_missing_or_conflicting_read_positions_fail_closed():
    page = _read(_chunks(["Ein Beleg mit unveraenderter Freigabe."]))
    missing = deepcopy(page)
    del missing["metadata"]["document_read_spans"]
    with pytest.raises(ValueError, match="ohne Zuordnung"):
        source_texts([missing])
    conflicting = deepcopy(page)
    conflicting["text"] = "X" + conflicting["text"][1:]
    with pytest.raises(ValueError, match="Widerspruechlicher Text"):
        source_texts([page, conflicting])
    conflicting = deepcopy(page)
    conflicting["metadata"]["document_read_spans"][0]["clean_end"] += 1
    with pytest.raises(ValueError, match="Positionsmetadaten"):
        source_texts([conflicting])
