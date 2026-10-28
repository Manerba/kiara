"""Explicit TD spans are geometry; they do not establish column headings."""

import pytest
from bs4 import BeautifulSoup
from kiara_document_loaders.html import convert_html
from kiara_document_loaders.html_tables import normalize_tables


def _texts(soup):
    return [
        [cell.get_text() for cell in row.find_all(["td", "th"], recursive=False)]
        for row in soup.select("tr")
    ]


def _prepare(source):
    soup = BeautifulSoup(source, "html.parser")
    return soup, normalize_tables(soup)


def test_td_rowspan_and_colspan_preserve_every_occupied_position():
    source = (
        '<table><tr><td rowspan="2">A</td><td colspan="2">B</td><td>C</td></tr>'
        '<tr><td>D</td><td colspan="2">E</td></tr>'
        "<tr><td>F</td><td>G</td><td>H</td><td>I</td></tr></table>"
    )
    soup, diagnostics = _prepare(source)
    assert _texts(soup) == [["A", "B", "B", "C"], ["A", "D", "E", "E"], ["F", "G", "H", "I"]]
    assert not soup.select("th, [colspan], [rowspan]")
    assert "table_td_grid_normalized" in diagnostics
    assert "table_unclear_headers_preserved" in diagnostics


@pytest.mark.parametrize(
    "decoration", ["", ' style="text-align:center;font-weight:bold;border-bottom:solid 1pt"']
)
def test_td_years_units_currency_and_parentheses_are_not_interpreted(decoration):
    source = (
        f'<table><tr><td></td><td colspan="3"{decoration}>Period</td><td rowspan="2">Change</td></tr>'
        f"<tr><td>in thousands</td><td></td><td{decoration}>2023</td><td></td></tr>"
        "<tr><td>Income</td><td>$</td><td>(17</td><td>)</td><td>5</td></tr></table>"
    )
    soup, diagnostics = _prepare(source)
    assert _texts(soup) == [
        ["", "Period", "Period", "Period", "Change"],
        ["in thousands", "", "2023", "", "Change"],
        ["Income", "$", "(17", ")", "5"],
    ]
    assert not soup.select("th")
    assert "table_td_headers_inferred" not in diagnostics


def test_native_markdown_has_exact_geometry_without_merging_cells():
    source = (
        '<table><tr><td></td><td colspan="3">Period</td><td rowspan="2">Change</td></tr>'
        "<tr><td>in thousands</td><td></td><td>2023</td><td></td></tr>"
        "<tr><td>Income</td><td>$</td><td>(17</td><td>)</td><td>5</td></tr></table>"
    )
    result = convert_html(source)
    lines = [line for line in result.markdown.splitlines() if line.startswith("|")]
    # Trim only the outer pipes: empty original edge cells are real cells.
    cells = [[cell.strip() for cell in line[1:-1].split("|")] for line in lines]
    assert cells[0] == ["", "Period", "Period", "Period", "Change"]
    assert cells[2] == ["in thousands", "", "2023", "", "Change"]
    assert cells[3] == ["Income", "$", "(17", ")", "5"]
    assert all(len(row) == 5 for row in cells)


def test_td_fullwidth_cells_retain_explicit_span_and_single_content_copy():
    source = (
        '<table><tr><td id="note" colspan="2"><a id="link" href="#fn">No delivery</a></td></tr>'
        '<tr><td>A</td><td>B</td></tr></table><p id="fn">Footnote</p>'
    )
    soup, _ = _prepare(source)
    assert _texts(soup) == [["No delivery"], ["A", "B"]]
    assert soup.select("tr")[0].td["colspan"] == "2"
    assert len(soup.select("#note")) == len(soup.select("#link")) == len(soup.select("#fn")) == 1
    assert len(soup.select('a[href="#fn"]')) == 1


def test_native_fullwidth_amount_is_kept_once_above_original_detail_cells():
    source = (
        '<table><tr><td colspan="3">2,99 €</td></tr>'
        "<tr><td>Tax included</td><td></td><td>0,48 €</td></tr></table>"
    )
    result = convert_html(source)
    assert result.markdown.count("2,99 €") == 1
    assert result.markdown.count("Tax included") == 1
    assert result.markdown.count("0,48 €") == 1


def test_td_zero_rowspan_ends_at_explicit_row_group():
    source = (
        '<table><tbody><tr><td rowspan="0">A</td><td>B</td></tr><tr><td>C</td></tr></tbody>'
        "<tbody><tr><td>D</td><td>E</td></tr></tbody></table>"
    )
    soup, _ = _prepare(source)
    assert _texts(soup) == [["A", "B"], ["A", "C"], ["D", "E"]]


@pytest.mark.parametrize(
    "source, records",
    [
        ('<table><tr><td colspan="2">A</td></tr><tr><td>B</td></tr></table>', True),
        (
            '<table><tr><td>A</td><td rowspan="2">B</td><td>C</td></tr><tr><td colspan="2">D</td><td>E</td></tr></table>',
            False,
        ),
        ('<table><tr><td rowspan="3">A</td><td>B</td></tr><tr><td>C</td></tr></table>', True),
        (
            '<table><tr><td colspan="bad">A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>',
            False,
        ),
        (
            '<table><tbody><tr><td rowspan="2">A</td><td>B</td></tr></tbody><tbody><tr><td>C</td><td>D</td></tr></tbody></table>',
            False,
        ),
    ],
)
def test_invalid_td_grid_uses_explicit_records_or_fails_without_padding(source, records):
    soup = BeautifulSoup(source, "html.parser")
    if not records:
        with pytest.raises(ValueError, match="span"):
            normalize_tables(soup)
        with pytest.raises(ValueError, match="span"):
            convert_html(source)
        return
    diagnostics = normalize_tables(soup)
    assert "table_explicit_cell_records" in diagnostics
    assert "table_td_grid_normalized" not in diagnostics
    assert "table_explicit_cell_records" in convert_html(source).diagnostics


@pytest.mark.parametrize(
    "source",
    [
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></table>",
        '<table><tr><td rowspan="2">A</td><th scope="row">B</th></tr><tr><td>C</td></tr></table>',
    ],
)
def test_unmarked_table_without_td_only_spans_is_not_reinterpreted(source):
    soup = BeautifulSoup(source, "html.parser")
    before = str(soup)
    assert "table_td_grid_normalized" not in normalize_tables(soup)
    assert str(soup) == before
