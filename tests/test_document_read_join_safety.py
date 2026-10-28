"""Zusammengesetzte Belege: SDK-Schutz, Originalcursor und Unicode-Aufwand."""

import unicodedata

import pytest
from test_agent_context_wire import _config, _prepare, _tool_contents, wire  # noqa: F401
from test_document_read_anonymization import _anon
from test_document_read_boundaries import _ingest_chunks

from kiara.anonymization.anonymizer import Anonymizer
from kiara.anonymization.automaton import AnonymizationAutomaton
from kiara.benchmarks.agent_dataset import normalized
from kiara.benchmarks.agent_retrieval import source_texts
from kiara.generation.adapters import cloud_base
from kiara.generation.agent_chain import AgentChain
from kiara.generation.context_selection import (
    prepare_context,
    prepare_document_read,
    render_context,
)
from kiara.shared.prompt_defense import sanitize_with_report
from kiara.tools.document_tools import DocumentReadChunks
from kiara.tools.registry import ToolRegistry


@pytest.mark.parametrize("provider", ["ollama", "anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("kind", ["read_document", "search_documents"])
def test_rejoined_marker_stays_removed_in_iteration_final_and_evidence(
    tmp_path, wire, provider, kind
):
    marker = "<|im_start|>"
    original = "A" * 96 + marker + "B" * 220
    raw = _ingest_chunks(tmp_path / "source.txt", original)
    assert all(marker not in chunk["text"] for chunk in raw)
    config = _config(10000, 5000)
    adapter, captured = wire(
        provider, config, None if provider == "ollama" else _anon(config, None)
    )
    chain = AgentChain(config, ToolRegistry(), adapter)
    result = {"raw_chunks": DocumentReadChunks(raw, lambda: raw)}
    if kind == "read_document":
        result["document_read"] = {
            "document_id": "doc_test",
            "start_chunk": 0,
            "start_offset": 0,
            "total_chunks": len(raw),
        }
    display, selected, total = _prepare(chain, result, kind, 0, "joined")
    adapter.step()
    list(adapter.stream_final())
    expected = sanitize_with_report(original).clean_text
    assert expected in display and marker not in display
    assert total == len(display) <= 5000
    assert source_texts(selected)[str(tmp_path / "source.txt")] == normalized(expected)
    # Auch das spaetere Neurendern beim Ersetzen von Suchauszuegen bleibt sicher.
    assert expected in render_context(selected) and marker not in render_context(selected)
    assert len(captured) == 2
    for body in captured:
        contents = _tool_contents(body, provider)
        assert marker not in "".join(contents)
        assert display in "\n".join(contents)


@pytest.mark.parametrize("marker", ["<|im_start|>", "[INST]", "<start_of_turn>", "<system>"])
@pytest.mark.parametrize("max_chunks", [1, 8])
def test_removed_joins_keep_cursor_ranges_and_benchmark_evidence(tmp_path, marker, max_chunks):
    original = "A" * 98 + marker + "B" * 1300
    raw = _ingest_chunks(tmp_path / "source.txt", original)
    request = {
        "document_id": "doc_test",
        "start_chunk": 0,
        "start_offset": 0,
        "total_chunks": len(raw),
    }
    pages = []
    for _ in range(100):
        start = request["start_chunk"]
        page = prepare_document_read(
            raw[start : start + max_chunks], request=request, budget=750, source_numbers={}
        )
        assert page.chunks and len(page.text) <= 750 and marker not in page.text
        pages.extend(page.chunks)
        for span in page.chunks[0]["metadata"]["document_read_spans"]:
            clean = sanitize_with_report(raw[span["chunk_index"]]["text"]).clean_text
            assert span["clean_length"] == len(clean)
            assert (
                page.chunks[0]["text"][span["text_start"] : span["text_end"]]
                == clean[span["clean_start"] : span["clean_end"]]
            )
        status = page.chunks[0]["metadata"]["document_read"]
        if not status["has_more"]:
            assert status["end_chunk"] == len(raw) - 1
            assert status["end_offset"] == len(raw[-1]["text"])
            break
        assert (status["next_chunk"], status["next_offset"]) > (
            request["start_chunk"],
            request["start_offset"],
        )
        request.update(start_chunk=status["next_chunk"], start_offset=status["next_offset"])
    else:
        pytest.fail("Lesecursor beendet das Dokument nicht")
    expected = normalized(sanitize_with_report(original).clean_text)
    assert source_texts([*reversed(pages), *pages[:2]])[str(tmp_path / "source.txt")] == expected
    search = prepare_context(raw, budget=10000)
    assert marker not in search.text
    assert source_texts(search.chunks)[str(tmp_path / "source.txt")] == expected


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("offset", [0, 80])
def test_join_removal_preserves_entity_positions_and_actual_budget(
    tmp_path, wire, provider, offset
):
    # Der Name wird ebenfalls ueber echte Chunkgrenzen verteilt. Das zuvor
    # entfernte Tag verschiebt seine Position im gerenderten Quellenkoerper.
    name = "Eva " + "Langname " * 20 + "Mueller"
    original = "A" * 96 + "<|im_start|> " + name + " Ende. " + "Folgetext " * 30
    raw = _ingest_chunks(tmp_path / "source.txt", original)
    config = _config(20000, 750)
    adapter, captured = wire(provider, config, _anon(config, name))
    chain = AgentChain(config, ToolRegistry(), adapter)
    request = {
        "document_id": "doc_test",
        "start_chunk": 0,
        "start_offset": offset,
        "total_chunks": len(raw),
    }
    total, pages = 0, []
    for number in range(40):
        start = request["start_chunk"]
        data = {
            "raw_chunks": DocumentReadChunks(raw[start : start + 8], lambda: raw),
            "document_read": dict(request),
        }
        display, selected, total = _prepare(chain, data, "read_document", total, f"r{number}")
        assert selected and len(display) <= 750
        assert "<|im_start|>" not in display
        assert not any(part in display for part in ["Eva", "Langname", "Mueller"])
        pages.append(display)
        adapter.step()
        status = data["document_read"]
        if not status["has_more"]:
            break
        request.update(start_chunk=status["next_chunk"], start_offset=status["next_offset"])
    else:
        pytest.fail("Keine Lesefortsetzung")
    list(adapter.stream_final())
    assert total == sum(map(len, pages))
    assert any("‹ANON:Person_001›" in page for page in pages)
    for body in captured:
        payload = "".join(_tool_contents(body, provider))
        assert "<|im_start|>" not in payload
        assert not any(part in payload for part in ["Eva", "Langname", "Mueller"])


@pytest.mark.parametrize("repetitions", [100, 200, 400])
def test_unicode_mapping_is_linear_and_keeps_individual_replacements(
    wire, monkeypatch, repetitions
):
    names = ["Éva Mueller", "Änne", "각이름"]
    source = ("Cafe\u0301 E\u0301va Mueller, A\u0308nne, \u1100\u1161\u11a8이름. ") * repetitions
    automaton = AnonymizationAutomaton()
    automaton.add_entities(
        [{"value": name, "category": "employee", "id": str(i)} for i, name in enumerate(names)]
    )
    automaton.build()
    anon = Anonymizer(automaton, {})
    adapter, captured = wire("openai", {}, anon)
    normalize = unicodedata.normalize
    count = 0

    def counted(form, text):
        nonlocal count
        count += len(text)
        return normalize(form, text)

    with monkeypatch.context() as measure:
        measure.setattr(cloud_base.unicodedata, "normalize", counted)
        prepared = adapter.prepare_tool_text(source)
    assert count < 10 * len(source)
    assert len(prepared.replacements) == len(prepared.protected_spans) == 3 * repetitions
    assert prepared.count("‹ANON:") == 3 * repetitions
    for low, high, replacement in prepared.replacements:
        assert anon.last_result.mapping[replacement] == normalize("NFC", source[low:high])
    assert captured == []


def test_distinct_entities_in_one_unicode_group_keep_their_replacement_slots():
    # U+0F73 expandiert unter NFC in zwei Zeichen derselben Eingabegruppe.
    assert cloud_base._source_entity_spans("\u0f73", [(0, 1), (1, 2)]) == ((0, 1), (0, 1))
