"""Generic reductions of failures found in the existing HTML migration stock."""

import pytest
from bs4 import BeautifulSoup
from kiara_document_loaders.html import convert_html


@pytest.mark.parametrize("tag", ["em", "strong"])
def test_json_ld_html_cannot_consume_the_actual_body(tag):
    source = (
        "<html><head><title>Beleg AB-314</title>"
        '<script type="application/ld+json">'
        '{"description":"<' + tag + ">Metadaten<\\/" + tag + '>"}'
        "</script></head><body><p>Keine Freigabe: AB-314, 7,45 €.</p></body></html>"
    )
    result = convert_html(source)
    assert result.markdown == "# Beleg AB-314\n\nKeine Freigabe: AB-314, 7,45 €.\n"
    assert "Metadaten" not in result.markdown


def test_noscript_fallback_keeps_text_links_and_page_title():
    result = convert_html(
        '<html><head><title>App</title><script src="app.js"></script></head>'
        "<body><noscript>JavaScript erforderlich. "
        '<a href="/hilfe">Keine Freigabe ohne Prüfung.</a></noscript>'
        '<div id="root"></div></body></html>'
    )
    assert result.markdown == (
        "# App\n\nJavaScript erforderlich. [Keine Freigabe ohne Prüfung.](/hilfe)\n"
    )
    assert "html:noscript_content_preserved" in result.diagnostics


def test_noscript_still_obeys_proven_css_visibility():
    result = convert_html(
        "<style>noscript {display:none}</style>"
        "<noscript>Nicht sichtbar</noscript><p>Freigabe bleibt offen.</p>"
    )
    assert result.markdown == "Freigabe bleibt offen.\n"


def _data_table(key, value):
    return (
        "<table><tr><th>Teil</th><th>Betrag</th></tr>"
        f"<tr><td>{key}</td><td>{value}</td></tr></table>"
    )


@pytest.mark.parametrize("role", ["", ' role="presentation"', ' role="none"'])
def test_side_by_side_tables_remain_tables_with_separate_rows(role):
    source = (
        f"<table{role}><tr><td><h2>Wartung</h2>{_data_table('AB-314', '7,45 €')}"
        "</td><td><h2>Ersatz</h2>" + _data_table("CD-271", "12,50 €") + "</td></tr></table>"
        "<p>Keine automatische Freigabe.</p>"
    )
    result = convert_html(source)
    assert "\\|" not in result.markdown
    lines = result.markdown.splitlines()
    assert "| AB-314 | 7,45 € |" in lines
    assert "| CD-271 | 12,50 € |" in lines
    assert sum(line == "| Teil | Betrag |" for line in lines) == 2
    for literal in ("Wartung", "Ersatz", "AB-314", "CD-271", "Keine automatische Freigabe."):
        assert result.markdown.count(literal) == 1


def test_presentation_rows_keep_lists_without_invented_column_headers():
    source = (
        '<table role="presentation"><tr><td><h3>Bestes Teil</h3>'
        "<ul><li>AB-314: nicht freigegeben</li><li>CD-271: offen</li></ul></td>"
        "<td><h3>Prüfer</h3><p>Firma A</p></td></tr>"
        "<tr><td><h3>Frist</h3><p>14 Tage</p></td><td>Keine Verlängerung.</td></tr></table>"
    )
    result = convert_html(source)
    assert "|" not in result.markdown
    assert "- AB-314: nicht freigegeben" in result.markdown
    assert result.markdown.index("Prüfer") < result.markdown.index("Frist")
    assert result.markdown.count("Keine Verlängerung.") == 1


def test_single_data_row_is_content_not_an_invented_header():
    result = convert_html(
        '<table><tr><td><a href="/beleg">Beleg AB-314</a></td>'
        "<td>7,45 €</td><td>Nicht freigegeben</td></tr></table>"
    )
    assert "|" not in result.markdown
    assert "[Beleg AB-314](/beleg)" in result.markdown
    assert result.markdown.count("7,45 €") == 1
    assert result.markdown.count("Nicht freigegeben") == 1


def test_header_only_colspan_does_not_multiply_its_content():
    result = convert_html(
        '<table><tr><th colspan="5"><a href="/vorschau">Vorschau</a> '
        "Prüfung am 29. September</th></tr></table>"
    )
    assert result.markdown.count("Vorschau") == 1
    assert result.markdown.count("Prüfung am 29. September") == 1


@pytest.mark.parametrize("wrapper", ["nav", 'div role="navigation"'])
def test_navigation_caption_is_one_block_and_keeps_all_links(wrapper):
    closing = wrapper.split()[0]
    source = (
        f'<{wrapper}><table><tr><th colspan="2">'
        '<a href="/view/very-long-reference">Ansicht</a> '
        '<a href="/edit/very-long-reference">Bearbeiten</a> Preisgeschichte</th></tr>'
        '<tr><th scope="row">2024</th><td>Firma A</td></tr>'
        f'<tr><th scope="row">2025</th><td>Firma B</td></tr></table></{closing}>'
    )
    result = convert_html(source)
    assert "|" not in result.markdown
    for literal in ("Preisgeschichte", "/view/very-long-reference", "/edit/very-long-reference"):
        assert result.markdown.count(literal) == 1
    assert result.markdown.index("2024") < result.markdown.index("Firma A")
    assert result.markdown.index("Firma A") < result.markdown.index("2025")
    assert result.markdown.index("2025") < result.markdown.index("Firma B")


def test_navigation_landmark_does_not_remove_real_column_headers():
    result = convert_html(
        '<nav><table><tr><th colspan="2">Preise</th></tr>'
        "<tr><th>Teil</th><th>Betrag</th></tr>"
        "<tr><td>AB-314</td><td>7,45 €</td></tr></table></nav>"
    )
    assert "| Preise / Teil | Preise / Betrag |" in result.markdown
    assert "| AB-314 | 7,45 € |" in result.markdown


@pytest.mark.parametrize("full_width_list", [False, True])
def test_navigation_row_groups_are_not_promoted_to_column_headers(full_width_list):
    opening = (
        '<tr><td colspan="2"><ul><li><a href="/a">AB-314</a></li><li>CD-271</li></ul></td></tr>'
        if full_width_list
        else '<tr><th scope="row">Teile</th><td><ul><li><a href="/a">AB-314</a></li>'
        "<li>CD-271</li></ul></td></tr>"
    )
    result = convert_html(
        '<div role="navigation"><table>'
        + opening
        + '<tr><th scope="row">Status</th><td>Keine Freigabe</td></tr></table></div>'
    )
    assert "|" not in result.markdown
    assert result.markdown.count("[AB-314](/a)") == 1
    assert result.markdown.count("CD-271") == 1
    assert result.markdown.index("Status") < result.markdown.index("Keine Freigabe")


def test_navigation_does_not_guess_layout_for_unlabelled_comparative_rows():
    from kiara_document_loaders.html_tables import normalize_tables

    source = (
        "<nav><table><tr><td>AB-314</td><td>7,45 €</td></tr>"
        "<tr><td>CD-271</td><td>12,50 €</td></tr></table></nav>"
    )
    soup = BeautifulSoup(source, "html.parser")
    before = str(soup)
    diagnostics = normalize_tables(soup)
    assert str(soup) == before
    assert "table_unclear_headers_preserved" in diagnostics


def test_layout_normalization_keeps_ids_and_nested_data_rowspan():
    from kiara_document_loaders.html_tables import normalize_tables

    source = (
        '<table id="layout"><tr id="group"><td id="cell">'
        '<h2 id="caption">Beleg</h2><table id="data">'
        "<tr><th>Kunde</th><th>Teil</th></tr>"
        '<tr><td rowspan="2">Firma A</td><td>AB-314</td></tr>'
        "<tr><td>CD-271</td></tr></table></td></tr></table>"
    )
    soup = BeautifulSoup(source, "html.parser")
    normalize_tables(soup)
    assert len(soup.find_all("table")) == 1
    assert soup.table["id"] == "data"
    for identifier in ("layout", "group", "cell", "caption", "data"):
        assert len(soup.select("#" + identifier)) == 1
    assert [c.get_text() for c in soup.table.find_all("tr")[-1].find_all("td")] == [
        "Firma A",
        "CD-271",
    ]


def test_outer_layout_remains_layout_after_inner_table_becomes_blocks():
    result = convert_html(
        "<table><tr><td><table><tr><td><h3>Untergruppe</h3>"
        "<p>AB-314: keine Freigabe</p></td></tr></table></td>"
        "<td>Firma A</td></tr><tr><td>Frist</td><td>14 Tage</td></tr></table>"
    )
    assert "|" not in result.markdown
    for value in ("Untergruppe", "AB-314: keine Freigabe", "Firma A", "Frist", "14 Tage"):
        assert result.markdown.count(value) == 1
