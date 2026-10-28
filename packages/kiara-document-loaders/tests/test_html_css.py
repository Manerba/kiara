"""Development cases for the bounded visibility policy in Tareas #328."""

import pytest
from bs4 import BeautifulSoup
from kiara_document_loaders.html import convert_html


def test_stylesheet_important_override_preserves_business_condition():
    result = convert_html(
        "<style>.shown{display:block!important}</style>"
        "<p class='shown' style='display:none'>Nicht für Trinkwasser.</p>"
        "<p>Ventil V-31</p>"
    )
    assert "Nicht für Trinkwasser." in result.markdown
    assert "Ventil V-31" in result.markdown


def test_stylesheet_hidden_class_is_removed():
    result = convert_html(
        "<style>.duplicate{display:none}</style>"
        "<p class='duplicate'>DOPPELTER_BETRAG</p><p>Gesamt 12,40 EUR</p>"
    )
    assert "DOPPELTER_BETRAG" not in result.markdown
    assert "Gesamt 12,40 EUR" in result.markdown


def test_unknown_media_visibility_is_preserved_through_native_cleanup():
    result = convert_html(
        "<style>@media (min-width:800px){.condition{display:block!important}}</style>"
        "<p class='condition' style='display:none'>Zahlbar binnen 14 Tagen.</p>"
        "<p>Rechnung R-31</p>"
    )
    assert "Zahlbar binnen 14 Tagen." in result.markdown
    assert any("conditional" in diagnostic for diagnostic in result.diagnostics)


@pytest.mark.parametrize(
    ("css", "attrs", "visible"),
    [
        ("p{display:none}.item{display:block}", "class='item'", True),
        ("#target{display:block}.item{display:none}", "class='item' id='target'", True),
        ("p.item#target{display:none}.item{display:block}", "class='item' id='target'", False),
        (".a.b{display:none}.a{display:block}", "class='a b'", False),
        (".a.b{display:none}", "class='a'", True),
        (".item{display:none}.item{display:block}", "class='item'", True),
        (".item{display:block}.item{display:none}", "class='item'", False),
        ("#target{display:block}.item{display:none!important}", "id='target' class='item'", False),
        ("#target{display:none!important}", "id='target' style='display:block!important'", True),
        ("#target{display:none}", "id='target' style='display:block'", True),
        ("", "style='display:none!important;display:block'", False),
        ("", "style='display:none;display:block'", True),
        ("", "hidden", False),
        ("", "hidden style='display:block'", True),
        ("p, .item{display:none}", "", False),
    ],
)
def test_supported_cascade(css, attrs, visible):
    result = convert_html(f"<style>{css}</style><p {attrs}>ZIELTEXT</p><div>Kontrolltext</div>")
    assert ("ZIELTEXT" in result.markdown) is visible
    assert "Kontrolltext" in result.markdown


@pytest.mark.parametrize(
    "html",
    [
        "<link rel='stylesheet' href='https://example.invalid/style.css'>"
        "<p style='display:none'>ZIELTEXT</p>",
        "<style>@import url('https://example.invalid/style.css');</style><p hidden>ZIELTEXT</p>",
        "<style>.wrapper p{display:block!important}</style>"
        "<div class='wrapper'><p style='display:none'>ZIELTEXT</p></div>",
        "<style>.item{display:var(--display)}</style><p class='item' hidden>ZIELTEXT</p>",
        "<style>.item{all:initial}</style><p class='item' hidden>ZIELTEXT</p>",
        "<p style='visibility:hidden;opacity:0' aria-hidden='true'>ZIELTEXT</p>",
        "<style media='print'>.item{display:block!important}</style>"
        "<p class='item' style='display:none'>ZIELTEXT</p>",
    ],
)
def test_uncertain_visibility_keeps_content_and_reports_it(html):
    result = convert_html(html)
    assert "ZIELTEXT" in result.markdown
    assert result.diagnostics


def test_media_rule_only_makes_matching_element_uncertain():
    result = convert_html(
        "<style>@media print{.conditional{display:block!important}}</style>"
        "<p class='conditional' style='display:none'>BEWAHREN</p>"
        "<p class='other' style='display:none'>ENTFERNEN</p>"
    )
    assert "BEWAHREN" in result.markdown
    assert "ENTFERNEN" not in result.markdown


def test_hidden_parent_removes_visible_child():
    result = convert_html("<div style='display:none'><p style='display:block'>VERSTECKT</p></div>")
    assert not result.markdown.strip()


def test_visibility_mutates_dom_without_leaving_native_hiding_attributes():
    from kiara_document_loaders.html_css import apply_visibility

    soup = BeautifulSoup(
        "<style>@media print{.x{display:block}}</style>"
        "<p class='x' style='display:none' aria-hidden='true'>BEWAHREN</p>",
        "html.parser",
    )
    diagnostics = apply_visibility(soup)
    assert isinstance(diagnostics, tuple)
    assert soup.p.get_text() == "BEWAHREN"
    assert not any(name in soup.p.attrs for name in ("style", "hidden", "aria-hidden"))
    assert not soup.find("style")
