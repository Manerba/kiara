"""Regressionen fuer Fakten, Dokumentstruktur und gemeinsame Agent-Belege."""

import asyncio
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from kiara.generation.agent_chain import AgentChain
from kiara.generation.context_selection import prepare_context
from kiara.generation.tool_adapter import (
    AnthropicToolAdapter,
    OllamaToolAdapter,
    OpenAIToolAdapter,
)
from kiara.tools.registry import ToolRegistry


def _chunk(text, source="document.txt", index=0, **metadata):
    return {"text": text, "metadata": {"source": source, "chunk_index": index, **metadata}}


def _config(**agent):
    return {"chat": {"agent": {"final_reranking": False, **agent}}}


def _chain(**agent):
    adapter = MagicMock()
    adapter.prepare_tool_text.side_effect = str
    return AgentChain(_config(**agent), ToolRegistry(), adapter)


@pytest.mark.parametrize(
    "text",
    [
        "Apple Rechnung\niCloud+\n2,99 €",
        "Betriebsdruck\n10 bar\nNicht für Trinkwasser geeignet.",
        "Teilenummer\nAB-0098/3\nMaße\n12 × 9 mm",
        "Frist\n31.12.2026\nGilt nur bei schriftlichem Auftrag.\nKeine automatische Verlängerung.",
        "| Position | Anzahl | Preis |\n| --- | --- | --- |\n| Bauteil | 3 | 7,45 € |",
    ],
)
def test_preserves_full_original_when_it_fits(text):
    chunks = [_chunk(text), _chunk("Weitere Quelle", "other.txt")]
    original = deepcopy(chunks)
    prepared = prepare_context(chunks, budget=3200, query="Informationen")
    assert text in prepared.text
    assert not prepared.truncated
    assert chunks == original


def test_no_fixed_per_source_cap_or_invoice_grouping():
    text = "Rechnung 202500000001\n" + "Beschreibung mit Einzelheiten. " * 90
    prepared = prepare_context(
        [
            _chunk(text, "supplier-a.txt"),
            _chunk("Rechnung 202500000001\n5,03 €", "supplier-b.txt"),
        ],
        budget=5000,
    )
    assert text in prepared.text
    assert len(prepared.chunks) == 2


def test_same_paths_in_different_datasources_remain_separate():
    prepared = prepare_context(
        [
            _chunk("Druck\n10 bar", ds_ids=[1]),
            _chunk("Druck\n20 bar", ds_ids=[2]),
        ],
        budget=1000,
    )
    assert len(prepared.chunks) == 2
    assert "10 bar" in prepared.text and "20 bar" in prepared.text


def test_deduplicates_chunk_copies_and_exact_adjacent_overlap():
    shared = "Gemeinsamer Absatz mit einer erhaltenen Bedingung.\n"
    first = _chunk("Beginn\n" + shared, index=0)
    last = _chunk(
        shared + "Ende\n2,99 €",
        index=1,
        source_overlap_before=len(shared),
        source_gap_before="",
        source_body_start=0,
    )
    prepared = prepare_context([first, deepcopy(first), last], budget=1000)
    assert prepared.text.count(shared) == 1
    assert "Beginn\n" + shared + "Ende\n2,99 €" in prepared.text
    assert {span["chunk_index"] for span in prepared.chunks[0]["metadata"]["context_spans"]} == {
        0,
        1,
    }


def test_repeated_values_at_different_positions_are_not_deduplicated():
    text = "Wiederkehrender Betrag\n2,99 €"
    prepared = prepare_context([_chunk(text, index=0), _chunk(text, index=1)], budget=1000)
    assert prepared.text.count("2,99 €") == 2


def test_missing_document_ranges_remain_visible():
    prepared = prepare_context([_chunk("Start", index=0), _chunk("Ende", index=4)], budget=1000)
    assert "Nicht geladener Dokumentbereich" in prepared.text
    assert "Chunks: 0, 4" in prepared.text


def test_budget_selection_keeps_value_and_neighboring_condition():
    text = (
        "Einleitung " * 90
        + "\n\n## Technische Daten\n\nDruck\n10 bar\n\nNicht für Trinkwasser geeignet.\n\n"
        + "Anhang " * 90
    )
    prepared = prepare_context([_chunk(text)], budget=500, query="Druck")
    assert "## Technische Daten" in prepared.text
    assert "Druck\n10 bar" in prepared.text
    assert "Nicht für Trinkwasser geeignet." in prepared.text
    assert "ausgelassen" in prepared.text
    assert prepared.truncated


def test_table_selection_preserves_header_rows_and_condition():
    table = "| Artikel | Preis |\n| --- | --- |\n| AB-0098/3 | 7,45 € |\n| CD-123 | 0,00 € |"
    text = (
        "Einleitung " * 90
        + "\n\n## Preise\n\n"
        + table
        + "\n\nAlle Preise gelten nur bei Abnahme von 10 Stück.\n\n"
        + "Anhang " * 90
    )
    prepared = prepare_context([_chunk(text)], budget=600, query="AB-0098/3")
    assert table in prepared.text
    assert "Alle Preise gelten nur bei Abnahme von 10 Stück." in prepared.text
    assert prepared.truncated


def test_oversized_table_row_is_omitted_whole_and_reported():
    table = (
        "| Artikel | Bedingung |\n| --- | --- |\n| AB-0098/3 | "
        + "Nur bei Genehmigung " * 100
        + "|"
    )
    prepared = prepare_context([_chunk(table)], budget=350, query="AB-0098/3")
    assert "AB-0098/3" not in prepared.text
    assert "nachlesen" in prepared.text
    assert prepared.truncated
    assert not prepared.chunks


def test_fenced_code_with_blank_lines_is_not_split():
    code = "```python\nlimit = 10\n\nif not approved:\n    limit = 0\n```"
    prepared = prepare_context([_chunk("Titel\n\n" + code + "\n\nNur nach Freigabe.")], budget=400)
    assert code in prepared.text


def test_long_unstructured_paragraph_uses_adjacent_original_chunks():
    texts = [
        "Vorspann " * 100,
        "Weitere Angaben " * 60,
        "Hinweise " * 12,
        "BauteilAB-0098/3Druck10 barBedingungNur bei Freigabe.",
        "Nicht für Trinkwasser geeignet. " * 3,
        "Anhang " * 120,
        "Weitere Informationen " * 80,
    ]
    prepared = prepare_context(
        [_chunk(text, index=i) for i, text in enumerate(texts)], budget=850, query="AB-0098/3 Druck"
    )
    assert texts[3] in prepared.text
    assert texts[4] in prepared.text
    assert "Teilabschnitt eines langen Absatzes" in prepared.text
    assert prepared.chunks[0]["metadata"]["context_partial_paragraph"]
    assert len(prepared.text) <= 850


def test_table_spanning_several_chunks_stays_atomic():
    chunks = [
        _chunk("| Artikel | Bedingung |\n| --- | --- |\n", index=0),
        _chunk("| AB-0098/3 | " + "Nur bei Freigabe " * 100 + "|", index=1),
        _chunk("| EF-6 | Keine Freigabe |", index=2),
    ]
    prepared = prepare_context(chunks, budget=500, query="AB-0098/3")
    assert "AB-0098/3" not in prepared.text
    assert "nachlesen" in prepared.text


def test_short_paragraph_window_keeps_related_fields_together():
    text = (
        "Jahr 2025\n"
        + "Hintergrund " * 70
        + "\n\n"
        + "\n\n".join(
            [
                "Vertragsnummer 4711",
                "Leistungsjahr 2025",
                "Produkt: Speicher",
                "Monatliches Paket",
                "Verlängerung: 2026",
                "2,99 €",
                "Keine Erstattung.",
            ]
        )
        + "\n\n"
        + "Anhang " * 100
    )
    prepared = prepare_context([_chunk(text)], budget=700, query="Speicher 2025")
    assert "Produkt: Speicher" in prepared.text
    assert "2,99 €" in prepared.text
    assert "Keine Erstattung." in prepared.text


@pytest.mark.parametrize("budget", [0, 50, 120, 250, 500, 1000, 4000])
def test_includes_headers_gaps_and_wrapping_in_budget(budget):
    chunks = [_chunk((f"Absatz {i}.\n\n" * 30), f"source-{i}.txt") for i in range(5)]
    prepared = prepare_context(chunks, budget=budget, query="Absatz")
    assert len(prepared.text) <= budget
    if prepared.chunks:
        assert prepared.text.startswith("<¤KONTEXT¤>")
        assert prepared.text.endswith("</¤KONTEXT¤>")


def test_does_not_rerank_when_everything_fits():
    reranker = MagicMock()
    prepared = prepare_context([_chunk("Daten\n10 mm")], budget=1000, reranker=reranker)
    reranker.rerank.assert_not_called()
    assert "10 mm" in prepared.text


def test_source_limit_is_common_and_omissions_are_visible():
    prepared = prepare_context(
        [_chunk("Inhalt", f"{i}.txt") for i in range(4)], budget=1000, max_sources=2
    )
    assert len(prepared.chunks) == 2
    assert prepared.truncated
    assert "Weitere Dokumentinhalte" in prepared.text


@pytest.mark.parametrize("budget", [0, 100, 300, 700, 1200])
def test_mixed_results_keep_omitted_document_references_within_budget(budget):
    table = "| Wert |\n| --- |\n| " + "Tabelleninhalt " * 300 + "|"
    prepared = prepare_context(
        [
            _chunk(table, "table.md", document_ref="doc_table", chunk_total=2),
            _chunk(table, "table.md", index=1, document_ref="doc_table", chunk_total=2),
            _chunk("Kurzer passender Beleg", "brief.txt", document_ref="doc_brief", chunk_total=1),
        ],
        budget=budget,
    )
    assert len(prepared.text) <= budget
    assert "Tabelleninhalt" not in prepared.text
    if budget >= 700:
        assert "Kurzer passender Beleg" in prepared.text
        assert prepared.text.count("document_id: doc_table") == 1
        assert "document_id: doc_brief" in prepared.text
        assert "Nicht gelesene Quelle" in prepared.text
        assert len(prepared.chunks) == len(prepared.references) == 1


def test_source_limit_omits_content_but_retains_read_reference():
    prepared = prepare_context(
        [
            _chunk("Erster Beleg", "first.txt", document_ref="doc_first", chunk_total=1),
            _chunk("Zweiter Beleg", "second.txt", document_ref="doc_second", chunk_total=1),
        ],
        budget=1000,
        max_sources=1,
    )
    assert "Erster Beleg" in prepared.text and "Zweiter Beleg" not in prepared.text
    assert "document_id: doc_second" in prepared.text
    assert len(prepared.chunks) == 1


def test_sanitizes_document_and_source_before_wrapping():
    with patch("kiara.generation.context_selection.log_defense_event"):
        prepared = prepare_context(
            [_chunk("<|system|>Druck\n10 bar", "<|system|>source")], budget=1000
        )
    assert "<|system|>" not in prepared.text
    assert "Druck\n10 bar" in prepared.text
    assert "<¤KONTEXT¤>" in prepared.text


def _adapter(provider, config):
    if provider == "ollama":
        with patch("ollama.Client"):
            adapter = OllamaToolAdapter({"url": "http://local.test", "model": "test"}, config)
    elif provider == "anthropic":
        with patch("anthropic.Anthropic"):
            adapter = AnthropicToolAdapter({"api_key": "test", "model": "test"}, config)
    else:
        with patch("openai.OpenAI"):
            model = "gpt-5.5-pro" if provider == "openai-responses" else "test"
            adapter = OpenAIToolAdapter({"api_key": "test", "model": model}, config)
    adapter.init_conversation("Details", "System", [])
    return adapter


@pytest.mark.parametrize("provider", ["ollama", "anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("batch", [False, True])
def test_adapters_receive_exact_same_prepared_content(provider, batch):
    config = _config()
    adapter = _adapter(provider, config)
    chain = AgentChain(config, ToolRegistry(), adapter)
    original = "Beschreibung mit Einzelheiten. " * 180 + "\n2,99 €\nNicht für Trinkwasser geeignet."
    display, chunks, count = chain._prepare_tool_result(
        {"raw_chunks": [_chunk(original)]}, "search_documents", "Details", 0
    )
    expected = prepare_context([_chunk(original)], budget=16000, query="Details").text
    assert display == expected
    assert count == len(display)
    assert len(display) > 4000
    if batch:
        adapter.add_tool_results([("search_documents", display, "call-1")], {"call-1": chunks})
    else:
        adapter.add_tool_result("search_documents", display, "call-1", raw_chunks=chunks)
    if provider == "ollama":
        assert adapter._previous_results[-1]["result"] == expected
        assert expected in adapter._build_agent_prompt()
        assert expected in adapter._build_final_prompt()
    elif provider == "anthropic":
        assert adapter._messages[-1]["content"][0]["content"] == expected
    elif provider == "openai-responses":
        assert adapter._messages[-1]["output"] == expected
    else:
        assert adapter._messages[-1]["content"] == expected


def test_document_read_has_reserved_budget_after_search_exhaustion():
    from kiara.shared.processing_warnings import ProcessingWarnings, processing_warning_scope

    chain = _chain(
        context_budget_chars=3000, context_result_chars=2000, context_read_reserve_chars=1500
    )
    result = {"raw_chunks": [_chunk("Inhalt " * 170)]}
    collector = ProcessingWarnings()
    with processing_warning_scope(collector):
        display, _, total = chain._prepare_tool_result(result, "search_documents", "", 0)
        assert "Inhalt" in display
        assert collector.snapshot() == []
        display, _, total = chain._prepare_tool_result(result, "search_documents", "", total)
    assert "Inhalt " not in display
    (warning,) = collector.snapshot()
    assert warning["code"] == "context_incomplete"
    assert warning["reason"] == "budget_exhausted"
    assert warning["details"]["tool"] == "search_documents"
    read = {"raw_chunks": [_chunk("Gezielt nachgelesen. " * 60)]}
    display, _, _ = chain._prepare_tool_result(read, "read_document", "", total)
    assert "Gezielt nachgelesen. " * 60 in display
    assert len(display) > 500
    assert chain._retrieval_context_chars <= 3000


def test_parallel_ollama_results_keep_their_own_contents():
    config = _config()
    adapter = _adapter("ollama", config)
    chain = AgentChain(config, ToolRegistry(), adapter)
    executed = [
        (
            "search_documents",
            {"query": "A"},
            None,
            {"raw_chunks": [_chunk("A\n10 mm", "a.txt")], "result": "A"},
        ),
        (
            "search_documents",
            {"query": "B"},
            None,
            {"raw_chunks": [_chunk("B\n20 mm", "b.txt")], "result": "B"},
        ),
    ]
    _, results, raw_map, total = chain._build_executed_results(executed, 0, [], 0)
    adapter.add_tool_results(results, raw_map)
    assert "10 mm" in results[0][1] and "20 mm" not in results[0][1]
    assert "20 mm" in results[1][1] and "10 mm" not in results[1][1]
    assert total == sum(len(result[1]) for result in results)
    assert all(result[1] in adapter._build_final_prompt() for result in results)
    assert "[Quelle 1: a.txt" in results[0][1]
    assert "[Quelle 2: b.txt" in results[1][1]


def test_source_numbers_remain_stable_across_search_and_read():
    chain = _chain()
    display, _, total = chain._prepare_tool_result(
        {
            "raw_chunks": [
                _chunk("A", "a.txt"),
                _chunk("B", "b.txt"),
            ]
        },
        "search_documents",
        "",
        0,
    )
    assert "[Quelle 1: a.txt" in display
    assert "[Quelle 2: b.txt" in display
    display, _, total = chain._prepare_tool_result(
        {
            "raw_chunks": [
                _chunk("Weitere Details zu B", "b.txt"),
            ]
        },
        "read_document",
        "",
        total,
    )
    assert "[Quelle 2: b.txt" in display
    display, _, _ = chain._prepare_tool_result(
        {
            "raw_chunks": [
                _chunk("C", "c.txt"),
            ]
        },
        "search_documents",
        "",
        total,
    )
    assert "[Quelle 3: c.txt" in display


def _model_tool_contents(adapter, provider):
    if provider == "ollama":
        return [r["result"] for r in adapter._previous_results]
    if provider == "anthropic":
        return [
            block["content"]
            for message in adapter._messages
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
    key = "output" if provider == "openai-responses" else "content"
    return [
        message[key]
        for message in adapter._messages
        if message.get("type") == "function_call_output" or message.get("role") == "tool"
    ]


@pytest.mark.parametrize("provider", ["ollama", "anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("same_batch", [False, True])
@pytest.mark.parametrize("source", ["document.txt", "doc\u00adument.txt"])
def test_read_replaces_search_excerpt_and_reuses_budget_in_actual_model_messages(
    provider, same_batch, source
):
    config = _config(
        context_budget_chars=3000, context_result_chars=2200, context_read_reserve_chars=800
    )
    adapter = _adapter(provider, config)
    chain = AgentChain(config, ToolRegistry(), adapter)
    search = (
        "search_documents",
        {"query": "Druck"},
        "search-1",
        {
            "raw_chunks": [
                _chunk("ALTER AUSZUG " * 60, source, document_ref="doc_a", chunk_total=1),
                _chunk(
                    "Andere Quelle bleibt erhalten. " * 15,
                    "document.txt" if source != "document.txt" else "other.txt",
                    document_ref="doc_b",
                    chunk_total=1,
                ),
            ],
            "result": "Suchtreffer",
        },
    )
    original = "Bedingung:\nNur bei Freigabe.\n" * 50
    reading = (
        "read_document",
        {"document_id": "doc_a"},
        "read-1",
        {
            "raw_chunks": [_chunk(original, source, document_ref="doc_a", chunk_total=1)],
            "document_read": {
                "document_id": "doc_a",
                "start_chunk": 0,
                "start_offset": 0,
                "total_chunks": 1,
            },
        },
    )
    if same_batch:
        events, batch, raw_map, total = chain._build_executed_results([search, reading], 0, [], 0)
        adapter.add_tool_results(batch, raw_map)
    else:
        _, batch, raw_map, total = chain._build_executed_results([search], 0, [], 0)
        adapter.add_tool_results(batch, raw_map)
        events, batch, raw_map, total = chain._build_executed_results([reading], 1, [], total)
        adapter.add_tool_results(batch, raw_map)
    contents = _model_tool_contents(adapter, provider)
    assert len(contents) == 2
    assert "ALTER AUSZUG" not in "".join(contents)
    assert "Andere Quelle bleibt erhalten." in contents[0]
    assert original in contents[1]
    assert "[Quelle 1:" in contents[1]
    assert "[Quelle 2:" in contents[0]
    assert "Dokument vollstaendig gelesen" in contents[1]
    assert total == chain._retrieval_context_chars == sum(map(len, contents)) <= 3000
    assert events[-2].data["document_read"]["complete_document"]
    if provider == "ollama":
        assert "ALTER AUSZUG" not in adapter._build_final_prompt()
        assert original in adapter._build_final_prompt()


@pytest.mark.parametrize("provider", ["ollama", "anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("table_source", ["table.md", "ta\u00adble.md"])
def test_read_replacement_preserves_other_omitted_document_reference(provider, table_source):
    config = _config(
        context_budget_chars=4000, context_result_chars=1000, context_read_reserve_chars=500
    )
    adapter = _adapter(provider, config)
    chain = AgentChain(config, ToolRegistry(), adapter)
    table = _chunk(
        "| Wert |\n| --- |\n| " + "Langer Tabelleninhalt " * 300 + "|",
        table_source,
        document_ref="doc_table",
        chunk_total=1,
    )
    brief = _chunk("Kurzer Suchauszug", "brief.txt", document_ref="doc_brief", chunk_total=1)
    search = ("search_documents", {}, "search", {"raw_chunks": [table, brief]})
    _, batch, raw_map, total = chain._build_executed_results([search], 0, [], 0)
    adapter.add_tool_results(batch, raw_map)
    assert "document_id: doc_table" in _model_tool_contents(adapter, provider)[0]

    for iteration, chunk in enumerate([brief, table], 1):
        reference = chunk["metadata"]["document_ref"]
        result = {
            "raw_chunks": [chunk],
            "document_read": {
                "document_id": reference,
                "start_chunk": 0,
                "start_offset": 0,
                "total_chunks": 1,
            },
        }
        read = ("read_document", {"document_id": reference}, f"read-{iteration}", result)
        _, batch, raw_map, total = chain._build_executed_results([read], iteration, [], total)
        adapter.add_tool_results(batch, raw_map)
        contents = _model_tool_contents(adapter, provider)
        if iteration == 1:
            assert "document_id: doc_table" in contents[0]
            assert "Kurzer Suchauszug" not in contents[0]
            assert len(chain._document_context_results[0]["references"]) == 1
        else:
            assert "document_id: doc_table" not in contents[0]
            assert "Langer Tabelleninhalt" in contents[-1]
            assert chain._document_context_results[0]["references"] == []
        assert total == chain._retrieval_context_chars == sum(map(len, contents)) <= 4000


def test_failed_read_preparation_keeps_previous_search_evidence():
    config = _config(
        context_budget_chars=2000, context_result_chars=1500, context_read_reserve_chars=500
    )
    adapter = _adapter("ollama", config)
    chain = AgentChain(config, ToolRegistry(), adapter)
    chunk = _chunk("Vorhandener Beleg", document_ref="doc_a", chunk_total=1)
    display, raw, total = chain._prepare_tool_result(
        {"raw_chunks": [chunk]}, "search_documents", "", 0
    )
    adapter.add_tool_result("search_documents", display, raw_chunks=raw)
    config["chat"]["agent"]["context_result_chars"] = 10
    result = {
        "raw_chunks": [chunk],
        "document_read": {
            "document_id": "doc_a",
            "start_chunk": 0,
            "start_offset": 0,
            "total_chunks": 1,
        },
    }
    display, raw, _ = chain._prepare_tool_result(result, "read_document", "", total)
    assert not raw
    assert "Kontextbudget erschoepft" in display
    assert "Vorhandener Beleg" in adapter._previous_results[0]["result"]
    assert chain._document_read_chars == 0
    assert result["document_read"]["budget_exhausted"]


def test_exhausted_read_reports_failure_without_using_more_document_budget():
    chain = _chain(context_budget_chars=100, context_result_chars=100)
    chain._retrieval_context_chars = 100
    result = {
        "raw_chunks": [_chunk("Beleg", document_ref="doc_a", chunk_total=1)],
        "document_read": {
            "document_id": "doc_a",
            "start_chunk": 0,
            "start_offset": 0,
            "total_chunks": 1,
        },
    }
    events, batch, _, total = chain._build_executed_results(
        [("read_document", {"document_id": "doc_a"}, "read-1", result)], 0, [], 100
    )
    assert "Kontextbudget erschoepft" in batch[0][1]
    assert events[-1].data["success"] is False
    assert chain._retrieval_context_chars == 100
    assert total == 100 + len(batch[0][1])


@pytest.mark.parametrize("context_limit", [20, 80])
def test_sibling_details_survive_until_grouped_source_limit(context_limit):
    from kiara.retrieval.source_policy import resolve_chat_sources
    from kiara.tools.builtin_tools import SearchDocumentsTool

    config, k = resolve_chat_sources(
        {
            "chat": {
                "source_profiles": {"agent": {"max_context_chunks": context_limit}},
                "agent": {"final_reranking": False},
            }
        },
        "agent",
        30,
    )
    documents = {
        f"part-{i}.txt": [
            _chunk(text, f"part-{i}.txt", index, chunk_total=5)
            for index, text in enumerate(
                [
                    f"Bauteil AB-{i}",
                    "Technische Daten",
                    "Betriebsbedingungen",
                    "Nicht für Trinkwasser geeignet.",
                    f"Druck: {i + 1} bar",
                ]
            )
        ]
        for i in range(30)
    }
    selected = [chunks[0] for chunks in documents.values()]
    retriever = MagicMock()
    retriever.retrieve.return_value = selected
    retriever.load_chunks_by_source.side_effect = lambda source, allowed: documents[source]
    reranker = MagicMock(enabled=True)
    reranker.rerank.return_value = selected
    tool = SearchDocumentsTool(retriever, reranker, config, default_k=k)
    result = asyncio.run(tool.execute({"query": "Bauteile Druck"}))
    assert result.success
    assert len(result.data["results"]) == 150
    adapter = _adapter("ollama", config)
    chain = AgentChain(config, ToolRegistry(), adapter)
    display, chunks, _ = chain._prepare_tool_result(
        {"raw_chunks": result.data["results"]}, "search_documents", "Bauteile Druck", 0
    )
    assert len(chunks) == min(context_limit, 30)
    for chunk in chunks:
        assert documents[chunk["metadata"]["source"]][-1]["text"] in chunk["text"]
        assert "Nicht für Trinkwasser geeignet." in chunk["text"]
    adapter.add_tool_result("search_documents", display, raw_chunks=chunks)
    assert display in adapter._build_final_prompt()
