"""Original TD cell coordinates survive in canonical text without head inference."""

import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from kiara_document_loaders.html import convert_html

from kiara.ingestion.chunker import chunk_document_with_full_text

_SOURCE = Path(__file__).parent / "fixtures" / "p64-ftai-aviation-q3-2023.html"


def _plain(text):
    text = " ".join(text.replace("**", "").replace("*", "").split())
    return re.sub(r"\s+([,;)])", r"\1", text)


def _markdown_cells(line):
    # Drop exactly the boundary pipes, keeping empty first/last cells.
    stripped = line.strip()
    assert stripped.startswith("|") and stripped.endswith("|")
    return [_plain(cell) for cell in stripped[1:-1].split("|")]


def _source_cell_text(cell):
    copied = deepcopy(cell)
    for br in copied.find_all("br"):
        br.replace_with("\n")
    return _plain(copied.get_text())


@pytest.fixture(scope="module")
def original_ftai_table():
    soup = BeautifulSoup(_SOURCE.read_text(), "html.parser")
    table = soup.find_all("table")[5]
    rows = table.find_all("tr")
    cells = [row.find_all("td", recursive=False) for row in rows]
    assert len(rows) == 16 and not table.find("th")
    assert [len(row) for row in cells] == [8, 17] + [23] * 14
    text = [[_source_cell_text(cell) for cell in row] for row in cells]

    # Original span coordinates, independent of the normalizer's grid.
    assert [int(c.get("colspan", 1)) for c in cells[0]] == [1, 7, 1, 3, 1, 6, 1, 3]
    assert [int(c.get("rowspan", 1)) for c in cells[0]] == [1, 1, 1, 2, 1, 1, 1, 2]
    assert all(
        int(c.get("colspan", 1)) == int(c.get("rowspan", 1)) == 1 for row in cells[1:] for c in row
    )
    top = text[0]
    expected = [
        [
            top[0],
            *([top[1]] * 7),
            top[2],
            *([top[3]] * 3),
            top[4],
            *([top[5]] * 6),
            top[6],
            *([top[7]] * 3),
        ],
        [*text[1][:9], *([top[3]] * 3), *text[1][9:], *([top[7]] * 3)],
        *text[2:],
    ]
    assert all(len(row) == 23 for row in expected)
    return str(table), expected


def _prepare(html):
    converted = convert_html(html)
    full_text, chunks = chunk_document_with_full_text(
        converted.markdown,
        {"source": str(_SOURCE), "file_type": ".html", **converted.metadata()},
        chunk_size=1000,
        chunk_overlap=128,
    )
    return converted, full_text, chunks


def test_ftai_native_markdown_matches_every_original_grid_coordinate(original_ftai_table):
    html, expected = original_ftai_table
    converted = convert_html(html)
    lines = [line for line in converted.markdown.splitlines() if line.startswith("|")]
    actual = [_markdown_cells(line) for index, line in enumerate(lines) if index != 1]
    assert actual == expected
    assert "table_unclear_headers_preserved" in converted.diagnostics


def test_ftai_canonical_full_text_retains_cell_order_and_span_inheritance(original_ftai_table):
    html, expected = original_ftai_table
    _, full_text, chunks = _prepare(html)
    header, rows = expected[0], expected[1:]
    actual_rows = []
    continuations = defaultdict(list)
    first_continuation = {}
    for chunk in chunks:
        text = full_text[chunk["char_start"] : chunk["char_end"]]
        assert text == chunk["text"]
        assert len(text) <= 1000
        lines = text.splitlines()
        if lines[0].startswith("|"):
            assert _markdown_cells(lines[0]) == header
            for line in lines[2:]:
                actual_rows.append(_markdown_cells(line))
            continue
        marker = re.fullmatch(r"Tabelle 1, Zeile (\d+) \(Fortsetzung\)", lines[0])
        assert marker, text
        row_number = int(marker[1])
        key, separator, value = lines[1].partition(":")
        assert separator and _plain(key) == "Spalte 1"
        if row_number not in first_continuation:
            first_continuation[row_number] = len(actual_rows)
            actual_rows.append([_plain(value)])
        assert actual_rows[first_continuation[row_number]][0] == _plain(value)
        label, separator, value = " ".join(lines[2:]).partition(":")
        assert separator
        column = len(continuations[row_number]) + 1
        assert _plain(label) == (header[column] or f"Spalte {column + 1}")
        continuations[row_number].append(_plain(value))
        actual_rows[first_continuation[row_number]].append(_plain(value))
    assert actual_rows == rows
    for number, index in first_continuation.items():
        assert index == number - 1


def test_ftai_complete_original_has_exact_spans_and_unchanged_chunk_budget():
    _, full_text, chunks = _prepare(_SOURCE.read_text())
    assert chunks
    for chunk in chunks:
        assert len(chunk["text"]) <= 1000
        assert full_text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
    assert [c["char_start"] for c in chunks] == sorted(c["char_start"] for c in chunks)
