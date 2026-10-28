"""Leseseiten: erkannte Entitaeten bleiben ganz, auch im echten SDK-Payload."""

import unicodedata

import pytest
from test_agent_context_wire import (
    _chunk,
    _config,
    _prepare,
    _tool_contents,
    wire,  # noqa: F401
)

from kiara.anonymization.anonymizer import Anonymizer
from kiara.anonymization.automaton import AnonymizationAutomaton
from kiara.benchmarks.agent_dataset import normalized
from kiara.benchmarks.agent_retrieval import source_texts
from kiara.exceptions import RetrievalError
from kiara.generation.agent_chain import AgentChain
from kiara.generation.context_selection import prepare_document_read
from kiara.shared.prompt_defense import sanitize_with_report
from kiara.tools.registry import ToolRegistry


def _anon(config, entity):
    automaton = AnonymizationAutomaton()
    if entity:
        automaton.add_entities([{"value": entity, "category": "employee", "id": "E1"}])
    automaton.build()
    return Anonymizer(automaton, config)


def _request(offset=0):
    return {"document_id": "doc_test", "start_chunk": 0, "start_offset": offset, "total_chunks": 1}


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize(
    "original_entity,known_entity",
    [
        ("Eva", "Eva"),
        ("E\u00adva", "Eva"),
        ("E\u0301va", "Éva"),
        ("Eva\nMueller", "Eva\nMueller"),
        ("eva@example.test", None),
    ],
)
def test_pages_keep_entities_whole_through_step_and_final(
    wire, provider, original_entity, known_entity
):
    prefix = "Cafe\u0301 Start " + "Sachtext " * 12 + "Verantwortlich: "
    original = prefix + original_entity + " genehmigt. " + "Folgetext " * 70
    entity_start, entity_end = len(prefix), len(prefix) + len(original_entity)
    raw = [_chunk(original, source="document.txt", ref="doc_test")]
    # Eine Budgetgrenze, die ohne Entitaetsschutz mitten in den Namen faellt.
    for budget in range(300, 900):
        plain = prepare_document_read(raw, request=_request(), budget=budget, source_numbers={})
        if (
            plain.chunks
            and entity_start
            < plain.chunks[0]["metadata"]["document_read"]["end_offset"]
            < entity_end
        ):
            break
    else:
        pytest.fail("Testaufbau erzeugt keinen Schnitt innerhalb der Entitaet")

    config = _config(16000, budget)
    anon = _anon(config, known_entity)
    clean = sanitize_with_report(original).clean_text
    full = anon.anonymize(clean)
    assert "‹ANON:" in full
    adapter, captured = wire(provider, config, anon)
    chain = AgentChain(config, ToolRegistry(), adapter)
    cursor, total, selected, displays, ranges = 0, 0, [], [], []
    for index in range(50):
        result = {"raw_chunks": raw, "document_read": _request(cursor)}
        display, chunks, total = _prepare(chain, result, "read_document", total, f"r{index}")
        assert chunks, "Sichtbarer Text muss mit gueltigem Fortsetzungscursor erreichbar bleiben"
        status = result["document_read"]
        assert status["start_offset"] == cursor < status["end_offset"]
        cursor = status["end_offset"]
        ranges.append((status["start_offset"], cursor))
        selected.extend(chunks)
        displays.append(display)
        assert len(display) <= budget
        assert total == sum(map(len, displays)) == chain._retrieval_context_chars <= 16000
        stats = anon.stats
        adapter.step()
        assert anon.stats == stats, (
            "Bereitete Ergebnisse werden beim Senden nicht erneut anonymisiert"
        )
        if not status["has_more"]:
            break
        assert (status["next_chunk"], status["next_offset"]) == (0, cursor)
    else:
        pytest.fail("Lesefortsetzung erreicht das Dokumentende nicht")
    list(adapter.stream_final())
    assert len(displays) > 1 and len(captured) == len(displays) + 1
    assert cursor == len(original)
    assert source_texts(selected)["document.txt"] == normalized(clean)
    # Jede echte SDK-Anfrage enthaelt exakt die bis dahin budgetierten Seiten.
    for index, body in enumerate(captured):
        expected = displays[: min(index + 1, len(displays))]
        assert _tool_contents(body, provider) == expected
        assert sum(map(len, _tool_contents(body, provider))) == sum(map(len, expected))
    assert any("‹ANON:" in text for text in displays)
    assert unicodedata.normalize(
        "NFC", sanitize_with_report(original_entity).clean_text
    ) not in "".join(displays)
    assert all(not entity_start < end < entity_end for _, end in ranges)
    assert any(start <= entity_start and end >= entity_end for start, end in ranges)


@pytest.mark.parametrize("entity", ["Eva", "E\u00adva", "E\u0301va"])
def test_explicit_cursor_inside_entity_fails_closed(wire, entity):
    config = _config(4000, 750)
    name = unicodedata.normalize("NFC", sanitize_with_report(entity).clean_text)
    adapter, captured = wire("openai", config, _anon(config, name))
    raw = [_chunk("Start " + entity + " Ende.", source="document.txt", ref="doc_test")]
    with pytest.raises(RetrievalError, match="Entitaet"):
        prepare_document_read(
            raw,
            request=_request(len("Start ") + 1),
            budget=750,
            source_numbers={},
            prepare_text=adapter.prepare_tool_text,
        )
    assert captured == []


def test_large_entity_at_page_start_fits_as_pseudonym(wire):
    config = _config(4000, 500)
    entity = "LangerName" * 100
    adapter, _ = wire("openai", config, _anon(config, entity))
    raw = [_chunk(entity + " Ende " * 200, source="document.txt", ref="doc_test")]
    page = prepare_document_read(
        raw,
        request=_request(),
        budget=500,
        source_numbers={},
        prepare_text=adapter.prepare_tool_text,
    )
    assert page.chunks and len(page.text) <= 500
    assert page.chunks[0]["metadata"]["document_read"]["end_offset"] >= len(entity)
    assert entity not in page.text and "‹ANON:Person_001›" in page.text
    empty = prepare_document_read(
        raw,
        request=_request(),
        budget=100,
        source_numbers={},
        prepare_text=adapter.prepare_tool_text,
    )
    assert not empty.chunks and not empty.text


@pytest.mark.parametrize(
    "raw,entity",
    [
        ("a\u0301\u0323 Eva", "Eva"),
        ("A\u0308nne spricht", "Änne"),
        ("\u1100\u1161\u11a8이름 Eva", "각이름"),
        ("\u0f73 Eva", "Eva"),
    ],
)
def test_entity_coordinates_refer_to_original_unicode(wire, raw, entity):
    config = _config(4000, 750)
    config["anonymization"] = {"min_term_length": 1}
    adapter, _ = wire("openai", config, _anon(config, entity))
    prepared = adapter.prepare_tool_text(raw)
    assert "‹ANON:" in prepared
    assert len(prepared.protected_spans) == 1
    start, end = prepared.protected_spans[0]
    assert unicodedata.normalize("NFC", raw[start:end]) == entity


def test_custom_anonymizer_without_positions_keeps_transformed_input_whole(wire):
    class AnonymizerWithoutPositions:
        def anonymize(self, text):
            return text.replace("Eva", "[Person]")

    adapter, _ = wire("openai", _config(4000, 500), AnonymizerWithoutPositions())
    raw = [_chunk("Eva " + "Text " * 1000, source="document.txt", ref="doc_test")]
    page = prepare_document_read(
        raw,
        request=_request(),
        budget=500,
        source_numbers={},
        prepare_text=adapter.prepare_tool_text,
    )
    assert not page.chunks and not page.text


def test_context_dependent_entity_cannot_be_exposed_after_safe_cursor(wire):
    config = _config(4000, 750)
    config["anonymization"] = {
        "custom_patterns": [
            {"name": "personalnummer", "pattern": r"(?<=Personalnummer: )\d{4}", "type": "CUSTOM"}
        ]
    }
    adapter, captured = wire("openai", config, _anon(config, None))
    text = "Personalnummer: 4729 genehmigt. " + "Text " * 100
    assert "4729" not in adapter.prepare_tool_text(text)
    # Der Cursor teilt die Entitaet nicht, entfernt aber den Regex-Kontext.
    with pytest.raises(RetrievalError, match="Anonymisierung"):
        prepare_document_read(
            [_chunk(text, source="document.txt", ref="doc_test")],
            request=_request(len("Personalnummer: ")),
            budget=750,
            source_numbers={},
            prepare_text=adapter.prepare_tool_text,
        )
    assert captured == []
