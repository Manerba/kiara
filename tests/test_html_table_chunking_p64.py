"""Tabellen aus HTML behalten Zeilenbezüge und echte Late-Chunking-Spans."""

import pytest
from kiara_ingestion_core.models import ProcessingContext


def _table(rows=40):
    header = "| Kunde | Position | Betrag |\n| --- | --- | --- |"
    data = [f"| Kunde {i // 2:02} | Teil {i:02} | {i + 10},50 EUR |" for i in range(rows)]
    return header, data


def _process(monkeypatch, text, size=240, overlap=30, *, html=True):
    from kiara.ingestion import runtime

    metadata = {"source": "/tmp/example.txt", "file_type": ".txt"}
    if html:
        metadata.update(content_format="markdown", source_mime_type="text/html")
    doc = {"text": text, "metadata": metadata}
    monkeypatch.setattr(runtime, "_load_content", lambda *a, **kw: (doc, False))
    monkeypatch.setattr(runtime, "_handle_vlm_content", lambda *a, **kw: None)
    return runtime._process_single_file(
        "/tmp/example.txt",
        {"ingestion": {"chunk_size": size, "chunk_overlap": overlap}},
        {"chunk_size": size},
        123,
        ProcessingContext(),
        {"errors": 0},
    )


def test_forty_rows_repeat_headers_and_keep_whole_rows(monkeypatch):
    header, rows = _table()
    text = "# Ersatzteile\n\n" + header + "\n" + "\n".join(rows) + "\n\nEnde der Liste."
    chunks, metadata, _, full_text, spans = _process(monkeypatch, text)
    table_chunks = [chunk for chunk in chunks if "| Teil " in chunk]
    assert len(table_chunks) >= 3
    for chunk in table_chunks:
        assert chunk.startswith(header + "\n")
        assert all(line in rows for line in chunk.splitlines()[2:])
    for row in rows:
        assert sum(row in chunk for chunk in table_chunks) == 1
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert full_text.count(header) == len(table_chunks)
    assert len(metadata) == len(spans) == len(chunks)
    for chunk, (start, end) in zip(chunks, spans, strict=True):
        assert full_text[start:end] == chunk
    assert [start for start, _ in spans] == sorted(start for start, _ in spans)


def test_very_large_row_has_labeled_continuations_and_row_identity(monkeypatch):
    value = "abcdefghij" * 75
    text = f"| Kunde | Beschreibung |\n| --- | --- |\n| Betrieb Z | {value} |"
    chunks, _, _, full_text, spans = _process(monkeypatch, text, size=180, overlap=20)
    assert len(chunks) > 2
    assert all("Fortsetzung" in chunk for chunk in chunks)
    assert all("Kunde: Betrieb Z" in chunk for chunk in chunks)
    assert all("Beschreibung: " in chunk for chunk in chunks)
    assert all(len(chunk) <= 180 for chunk in chunks)
    reconstructed = "".join(chunk.split("Beschreibung: ", 1)[1] for chunk in chunks)
    assert reconstructed == value
    assert all(
        full_text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True)
    )


def test_impossible_field_label_budget_fails_explicitly(monkeypatch):
    text = "| " + "Bezeichner" * 30 + " |\n| --- |\n| Wert |"
    with pytest.raises(ValueError, match="(?i)(budget|chunk|feld|kopf)"):
        _process(monkeypatch, text, size=80, overlap=10)


def test_prose_and_list_survive_between_tables_with_exact_spans(monkeypatch):
    header, rows = _table(4)
    prose = "## Hinweise\n\n- Anlage sperren.\n- Druck prüfen.\n\nEnde des Hinweises."
    text = header + "\n" + "\n".join(rows) + "\n\n" + prose + "\n\n" + header + "\n" + rows[0]
    chunks, _, _, full_text, spans = _process(monkeypatch, text, size=240)
    assert prose in full_text
    assert any("- Anlage sperren.\n- Druck prüfen." in chunk for chunk in chunks)
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert all(
        full_text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True)
    )


def test_plaintext_path_keeps_original_full_text_and_spans(monkeypatch):
    header, rows = _table()
    text = header + "\n" + "\n".join(rows)
    chunks, _, _, full_text, spans = _process(monkeypatch, text, html=False)
    assert full_text == text
    assert all(text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True))


def test_escaped_pipe_in_cell_does_not_create_another_column(monkeypatch):
    header = "| Typ | Bedeutung |\n| --- | --- |"
    rows = [f"| Code {i} | A\\|B ist die Freigabe |" for i in range(12)]
    chunks, _, _, full_text, spans = _process(
        monkeypatch, header + "\n" + "\n".join(rows), size=180
    )
    for chunk in chunks:
        assert chunk.startswith(header + "\n")
    assert all(row in full_text for row in rows)
    assert all(
        full_text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True)
    )


def test_table_example_inside_code_fence_is_not_rewritten(monkeypatch):
    header, rows = _table(12)
    text = "```markdown\n" + header + "\n" + "\n".join(rows) + "\n```"
    chunks, _, _, full_text, spans = _process(monkeypatch, text)
    assert full_text == text
    assert all(
        full_text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True)
    )


def test_single_small_table_needs_no_text_change(monkeypatch):
    header, rows = _table(2)
    text = header + "\n" + "\n".join(rows)
    chunks, metadata, _, full_text, spans = _process(monkeypatch, text)
    assert full_text == text
    assert chunks == [text]
    assert spans == [(0, len(text))]
    assert metadata[0]["html_chunking_revision"]


def test_malformed_column_count_is_not_silently_truncated(monkeypatch):
    text = "| Artikel | Betrag |\n| --- | --- |\n| A-27 | 17 EUR | nicht freigegeben |"
    with pytest.raises(ValueError, match="Spaltenzahl"):
        _process(monkeypatch, text)


def test_wide_table_keeps_each_field_and_row_within_the_same_budget(monkeypatch):
    headers = ["Messung", *[f"Monat {month:02} Temperatur" for month in range(1, 15)]]
    rows = [
        ["Maximum", *[str(20 + month) for month in range(1, 15)]],
        ["Minimum", *[str(month - 10) for month in range(1, 15)]],
    ]
    text = "\n".join(
        "| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows]
    )
    assert len(text.splitlines()[0]) > 240
    chunks, _, _, full_text, spans = _process(monkeypatch, text)
    for number, row in enumerate(rows, 1):
        for label, value in zip(headers[1:], row[1:], strict=True):
            matches = [chunk for chunk in chunks if f"{label}: {value}" in chunk.splitlines()]
            assert len(matches) == 1
            assert f"Tabelle 1, Zeile {number}" in matches[0]
            assert f"Messung: {row[0]}" in matches[0]
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert all(
        full_text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True)
    )


@pytest.mark.parametrize("second_columns", [2, 3])
def test_adjacent_tables_remain_separate_relations(monkeypatch, second_columns):
    first = "| Mannschaft | Punkte |\n| --- | --- |\n| Berlin | 91 |"
    second = "| Beginn | Ort" + (" | Runde" if second_columns == 3 else "") + " |\n"
    second += "| " + " | ".join(["---"] * second_columns) + " |\n"
    second += "| 19:00 | Arena" + (" | Finale" if second_columns == 3 else "") + " |"
    chunks, _, _, full_text, spans = _process(monkeypatch, first + "\n" + second)
    assert chunks == [first, second]
    assert full_text == first + "\n" + second
    assert all(
        full_text[start:end] == chunk for chunk, (start, end) in zip(chunks, spans, strict=True)
    )
