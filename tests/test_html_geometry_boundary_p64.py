"""R1: original geometry must survive conversion and actual canonical spans."""

import hashlib
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from kiara_document_loaders.html import HtmlConversionError, convert_html

from kiara.ingestion.chunker import chunk_document_with_full_text


def _canonical(source, size=768):
    result = convert_html(source)
    full, chunks = chunk_document_with_full_text(
        result.markdown,
        {"source": "/tmp/r1.html", "file_type": ".html", **result.metadata()},
        chunk_size=size,
        chunk_overlap=96,
    )
    assert chunks
    assert all(full[c["char_start"] : c["char_end"]] == c["text"] for c in chunks)
    assert all(len(c["text"]) <= size for c in chunks)
    return result, full, [c["text"] for c in chunks]


def test_real_dreamworks_keeps_all_four_explicit_casper_relations():
    raw = (Path(__file__).parent / "fixtures/p64-dreamworks-original.html").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "3f0afda8f919fb51fef0cbc3c597d116eaae934d5172a85c0e74173ab410cfd0"
    )
    soup = BeautifulSoup(raw.decode(), "html.parser")
    table = soup.find_all("table")[3]
    rows = table.find_all("tr")
    assert len(rows) == 6
    assert all(c.get("rowspan") == "2" for c in rows[1].find_all("td", recursive=False)[2:6])
    result, full, texts = _canonical(raw.decode())
    for header, value in [
        ("Director(s)", "Mark Gravas"),
        ("Writer(s)", "Kirk DeMicco"),
        ("Composer(s)", "Magnus Fiennes"),
        ("Studio(s)", "Kapow Pictures"),
    ]:
        assert any(all(x in t for x in ["Casper's Scare School", header, value]) for t in texts)
    assert "table_explicit_cell_records" in result.diagnostics
    # An explicit dangling span remains stated; no invented seventh row.
    assert "rowspan=2" in full and "Originalzeile 7;" not in full
    assert "Megamind vs." in full and "March 1, 2024" in full


def test_incomplete_td_row_preserves_cells_and_declared_extent_without_padding():
    source = '<table><tr><td colspan="2">A</td></tr><tr><td>B</td></tr></table>'
    result, full, texts = _canonical(source)
    assert "table_explicit_cell_records" in result.diagnostics
    assert any(
        "Originalzeile 1;" in t and "Spalten 1–2" in t and "colspan=2" in t and "A" in t
        for t in texts
    )
    assert any("Originalzeile 2;" in t and "Spalten 1–1" in t and "B" in t for t in texts)
    assert "Originalzeile 2; Spalten 2" not in full


def test_later_dangling_th_span_keeps_earlier_inheritance_and_raw_span():
    source = (
        "<table><tr><th>Film</th><th>Director</th></tr>"
        '<tr><td>Film A</td><td rowspan="2">Director AB</td></tr>'
        "<tr><td>Film B</td></tr>"
        '<tr><td>Film C</td><td rowspan="2">Director C</td></tr></table>'
    )
    result, full, texts = _canonical(source)
    assert any(
        all(x in t for x in ["Film B", "Director", "Director AB", "Ursprung Z2/S2", "rowspan=2"])
        for t in texts
    )
    assert any(all(x in t for x in ["Film C", "Director C", "rowspan=2"]) for t in texts)
    assert "Originalzeile 5;" not in full


@pytest.mark.parametrize(
    "source",
    [
        '<table><tr><td>A</td><td rowspan="2">B</td><td>C</td></tr><tr><td colspan="2">D</td><td>E</td></tr></table>',
        '<table><tr><th>A</th><th>B</th><th>C</th></tr><tr><td>V42</td><td rowspan="2">X</td><td>Y</td></tr><tr><td colspan="2">Kollision</td><td>Z</td></tr></table>',
        '<table><tr><td colspan="bad">A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
        '<table><tbody><tr><td rowspan="2">A</td><td>B</td></tr></tbody><tbody><tr><td>C</td><td>D</td></tr></tbody></table>',
    ],
)
def test_ambiguous_span_never_reaches_successful_native_conversion(source):
    with pytest.raises(HtmlConversionError, match="(?i)(span|geometry)"):
        _canonical(source)


def test_long_record_value_keeps_header_row_and_geometry_in_every_chunk():
    value = "original-value " * 140
    source = (
        "<table><tr><th>Film</th><th>Director</th></tr>"
        '<tr><td>Film A</td><td rowspan="2">' + value + "</td></tr></table>"
    )
    result, full, texts = _canonical(source)
    chunks = [t for t in texts if "original-value" in t]
    assert len(chunks) >= 3
    assert all(
        all(x in t for x in ["Film A", "Director", "Ursprung Z2/S2", "rowspan=2"]) for t in chunks
    )


def test_record_labels_preserve_br_inline_adjacency_and_empty_cells():
    source = (
        "<table><thead><tr><th>Item</th><th>Mean<br>°C</th></tr></thead>"
        '<tbody><tr><td>AB<strong>-</strong>42</td><td rowspan="2"><a href="/value">17</a></td></tr>'
        "<tr><td></td></tr><tr></tr></tbody></table>"
    )
    result, full, texts = _canonical(source)
    assert any(all(x in t for x in ["AB-42", "Mean °C", "[17](/value)"]) for t in texts)
    assert "Originalzeile 3; Spalten 1–1" in full
    assert "Originalzeile 4; keine Zellen im Original." in full


def test_record_zero_rowspan_keeps_raw_attribute_and_real_group_boundary():
    source = (
        '<table><tbody><tr><td rowspan="00">A</td><td>B</td></tr>'
        "<tr></tr></tbody><tbody><tr><td>C</td></tr></tbody></table>"
    )
    _, full, texts = _canonical(source)
    assert any(
        all(x in t for x in ["Originalzeile 2;", "rowspan=00", "Gruppe Z1–Z2", "A"]) for t in texts
    )
    assert not any("Originalzeile 3;" in t and "rowspan=00" in t for t in texts)


def test_rejected_geometry_is_a_failed_migration_preflight_not_a_completion(tmp_path, caplog):
    from kiara.ingestion.html_migration import build_selection, migration_completion, prepare_files

    path = tmp_path / "overlap.html"
    path.write_text(
        '<table><tr><td>A</td><td rowspan="2">B</td><td>C</td></tr>'
        '<tr><td colspan="2">D</td><td>E</td></tr></table>'
    )
    selection = build_selection(23, {"nomic-embed-text": [str(path)]})
    config = {
        "datasource": {},
        "_html_migration": {**selection, "paths": [str(path)], "errors": {}},
    }

    class Local:
        @staticmethod
        def effective_path(_config, _row):
            return str(tmp_path)

    result = prepare_files(config, Local, {}, {"id": 23}, "nomic-embed-text", None, None)
    assert result["files"] == [] and set(result["errors"]) == {str(path)}
    assert result["errors"][str(path)] == "HTML-Konvertierung liefert keinen indexierbaren Inhalt"
    assert "HTML conversion failed: overlapping spans" in caplog.text
    assert config["_html_migration"]["paths"] == []
    assert config["_html_migration"]["errors"] == result["errors"]
    completion = migration_completion(
        selection,
        [
            {
                "embedding_model": "nomic-embed-text",
                "status": "success",
                "stats_json": {
                    "html_migration": selection,
                    "html_migration_errors": result["errors"],
                    "html_migration_completed_paths": [],
                },
            }
        ],
    )
    assert completion == {"complete": False, "pending_models": ["nomic-embed-text"]}
