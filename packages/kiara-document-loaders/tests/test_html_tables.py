"""Development contracts for DOM tables, independent of retrieval acceptance."""

import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from kiara_document_loaders.html import convert_html
from kiara_document_loaders.html_tables import normalize_tables


def _prepare(source):
    soup = BeautifulSoup(source, "html.parser")
    return soup, normalize_tables(soup)


def _cells(row):
    return [cell.get_text() for cell in row.find_all(["th", "td"], recursive=False)]


def test_rowspan_inherits_customer_on_every_data_row():
    source = (
        "<table><tr><th>Kunde</th><th>Teil</th><th>Status</th></tr>"
        '<tr><td rowspan="2">Firma A</td><td>V42</td><td>freigegeben</td></tr>'
        "<tr><td>V43</td><td>nicht freigegeben</td></tr></table>"
    )
    soup, _ = _prepare(source)
    rows = soup.select("tr")
    assert _cells(rows[-1]) == ["Firma A", "V43", "nicht freigegeben"]
    assert not soup.select("[rowspan], [colspan]")


def test_multilevel_headers_have_unambiguous_column_names():
    source = (
        '<table><thead><tr><th rowspan="2">Artikel</th>'
        '<th colspan="2">Maße in mm</th></tr>'
        "<tr><th>Breite</th><th>Höhe</th></tr></thead>"
        "<tbody><tr><td>A-42</td><td>12,5</td><td>4,0</td></tr></tbody></table>"
    )
    soup, _ = _prepare(source)
    rows = soup.select("tr")
    assert len(rows) == 2
    assert _cells(rows[0]) == ["Artikel", "Maße in mm / Breite", "Maße in mm / Höhe"]
    assert _cells(rows[1]) == ["A-42", "12,5", "4,0"]


def test_data_colspan_keeps_value_for_each_covered_column():
    source = (
        "<table><tr><th>Teil</th><th>Innen</th><th>Außen</th></tr>"
        '<tr><td>V42</td><td colspan="2">nicht freigegeben</td></tr></table>'
    )
    soup, _ = _prepare(source)
    assert _cells(soup.select("tr")[-1]) == ["V42", "nicht freigegeben", "nicht freigegeben"]


def test_inline_identifiers_links_and_footnotes_survive_inheritance():
    source = (
        "<table><tr><th>Kunde</th><th>Artikel</th></tr>"
        '<tr><td id="customer" rowspan="2"><a id="customer-link" href="https://example.test/k">A<strong>B</strong>-42</a>'
        '<sup><a href="#fn1">1</a></sup></td><td>V42</td></tr>'
        '<tr><td>V43</td></tr></table><p id="fn1">1: Nicht für Trinkwasser.</p>'
    )
    soup, _ = _prepare(source)
    rows = soup.select("tr")
    assert _cells(rows[-1]) == ["AB-421", "V43"]
    assert len(soup.select('a[href="https://example.test/k"]')) == 2
    assert len(soup.select("#customer")) == 1
    assert len(soup.select("#customer-link")) == 1
    assert len(soup.select("#fn1")) == 1
    assert soup.get_text().count("Nicht für Trinkwasser.") == 1


def test_full_width_note_is_not_duplicated_as_column_values():
    source = (
        "<table><tr><th>Teil</th><th>Status</th></tr>"
        "<tr><td>V42</td><td>offen</td></tr>"
        '<tfoot><tr><td colspan="2">Nicht ohne Freigabe liefern.</td></tr></tfoot></table>'
    )
    soup, _ = _prepare(source)
    assert soup.get_text().count("Nicht ohne Freigabe liefern.") == 1


@pytest.mark.parametrize("role", [None, "presentation", "none"])
def test_nested_layout_keeps_content_once_and_normalizes_inner_table(role):
    attribute = f' role="{role}"' if role else ""
    source = (
        f"<table{attribute}><tr><td><h2>Beleg</h2><table>"
        "<tr><th>Kunde</th><th>Teil</th></tr>"
        '<tr><td rowspan="2">Firma A</td><td>V42</td></tr>'
        "<tr><td>V43</td></tr></table><p>Keine automatische Verlängerung.</p>"
        "</td></tr></table>"
    )
    soup, diagnostics = _prepare(source)
    assert len(soup.find_all("table")) == 1
    assert _cells(soup.select("table tr")[-1]) == ["Firma A", "V43"]
    assert soup.get_text().count("Beleg") == 1
    assert soup.get_text().count("Keine automatische Verlängerung.") == 1
    assert any("layout" in value for value in diagnostics)


def test_unclear_td_headers_remain_unclear_after_explicit_geometry_is_resolved():
    source = '<table><tr><td rowspan="2">Firma A</td><td>V42</td></tr><tr><td>V43</td></tr></table>'
    soup, diagnostics = _prepare(source)
    assert [_cells(row) for row in soup.select("tr")] == [["Firma A", "V42"], ["Firma A", "V43"]]
    assert not soup.select("th, [rowspan], [colspan]")
    assert "table_unclear_headers_preserved" in diagnostics
    assert "table_td_grid_normalized" in diagnostics


@pytest.mark.parametrize("span", ["-2", "invalid", "1000000000"])
def test_ambiguous_or_excessive_span_is_rejected_before_native_conversion(span):
    source = (
        "<table><tr><th>Kunde</th><th>Teil</th></tr>"
        f'<tr><td rowspan="{span}">Firma A</td><td>V42</td></tr></table>'
    )
    with pytest.raises(ValueError, match="span"):
        _prepare(source)
    with pytest.raises(ValueError, match="span"):
        convert_html(source)


def test_overlapping_span_geometry_is_rejected_before_native_conversion():
    source = (
        "<table><tr><th>A</th><th>B</th><th>C</th></tr>"
        '<tr><td>V42</td><td rowspan="2">X</td><td>Y</td></tr>'
        '<tr><td colspan="2">Kollision</td><td>Z</td></tr></table>'
    )
    with pytest.raises(ValueError, match="overlapping spans"):
        _prepare(source)
    with pytest.raises(ValueError, match="overlapping spans"):
        convert_html(source)


_DEVELOPMENT = json.loads(
    (Path(__file__).parent / "fixtures" / "html_conversion_development.json").read_text()
)


@pytest.mark.parametrize(
    "case", [case for case in _DEVELOPMENT if "<table" in case["html"]], ids=lambda case: case["id"]
)
def test_existing_table_development_cases_keep_literals_and_row_relationships(case):
    soup, _ = _prepare(case["html"])
    markdown = convert_html(str(soup)).markdown.replace("\\_", "_").replace("**", "")
    normalized = " ".join(markdown.split())
    for literal in case["required"]:
        assert " ".join(literal.split()) in normalized
    rows = [" ".join(row.split()) for row in markdown.splitlines() if "|" in row]
    for expected in case["rows"]:
        assert any(all(value in row for value in expected) for row in rows)
    if case["id"] == "long_table":
        assert len(soup.select("tr")) == 41
        for index in range(1, 41):
            assert markdown.count(f"TEIL-{index:03}") == 1
