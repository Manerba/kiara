"""Generic development cases; private original mails remain outside the repository."""

import hashlib
import json
import unicodedata
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import html_to_markdown
import pytest
from kiara_document_loaders.html import HtmlConversionError, convert_html

_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "html_conversion_development.json").read_text()
)


def _normalized(text):
    text = unicodedata.normalize("NFKC", text).replace("\\_", "_").replace("**", "")
    text = text.translate(str.maketrans({"–": "-", "—": "-", "‑": "-"}))
    return " ".join(text.split())


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case["id"],
        )
        for case in _CASES
    ],
)
def test_development_literals_and_rows(case):
    markdown = convert_html(case["html"]).markdown
    normalized = _normalized(markdown)
    assert all(_normalized(value) in normalized for value in case["required"])
    assert all(_normalized(value) not in normalized for value in case["forbidden"])
    rows = [_normalized(line) for line in markdown.splitlines() if "|" in line]
    for expected in case["rows"]:
        assert any(all(_normalized(value) in row for value in expected) for row in rows)


def test_hashes_and_metadata_describe_actual_input_and_output():
    source = "<h1>Prüfprotokoll</h1><p>Ventil V-24: 12,5 bar</p>"
    result = convert_html(source)
    assert result.source_hash == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert result.output_hash == hashlib.sha256(result.markdown.encode("utf-8")).hexdigest()
    metadata = result.metadata()
    assert metadata == {
        "parser": "html-to-markdown",
        "content_format": "markdown",
        "source_mime_type": "text/html",
        "html_parser_revision": result.parser_revision,
        "html_profile_revision": result.profile_revision,
        "html_source_hash": result.source_hash,
        "html_output_hash": result.output_hash,
        "html_diagnostics": [],
    }
    metadata["html_diagnostics"].append("changed by caller")
    assert result.diagnostics == ()
    with pytest.raises(FrozenInstanceError):
        result.markdown = "changed"


@pytest.mark.parametrize(
    "source",
    [
        "",
        " \n\t",
        "<p> </p>",
        "<!-- only a comment -->",
        "<head><title> </title><style>p {color:red}</style></head>",
        "<style>p {color:red}</style><script>track()</script>",
        "<div hidden>Unsichtbarer Text</div>",
        "<p style='display:none'>Unsichtbarer Text</p>",
        "<div hidden><br><img src='pixel.png'>Unsichtbarer Text</div>",
    ],
)
def test_empty_or_hidden_documents_are_distinguished(source):
    result = convert_html(source)
    assert not result.markdown.strip()
    assert result.diagnostics == ("empty_input" if not source.strip() else "no_visible_content",)


@pytest.mark.parametrize("body", ["<p>Lieferung am 29. September.</p>", ""])
def test_page_title_is_canonical_content_even_without_body(body):
    result = convert_html(
        f"<html><head><title>Reviewkennung AB-314</title></head><body>{body}</body></html>"
    )
    assert result.markdown.startswith("# Reviewkennung AB-314\n")
    assert "no_visible_content" not in result.diagnostics
    if body:
        assert "Lieferung am 29. September." in result.markdown


def test_title_only_fragment_is_not_empty():
    result = convert_html("<head><title>Internal title</title><style>p {color:red}</style></head>")
    assert result.markdown.strip() == "# Internal title"
    assert result.diagnostics == ()


def test_real_reddit_title_and_crawler_relationship_survive():
    source = (Path(__file__).parent / "fixtures" / "reddit_title.html").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == (
        "882042ad1d3dd008360077c99feafefb6cdb3aa868f2996a43341789f9864f1c"
    )
    result = convert_html(source)
    title = (
        "Requiem for a Dream: Ellen Burstyn should have took home the Oscar for "
        "Best Actress in 2000 over Julia Roberts. : r/movies"
    )
    assert result.markdown.startswith("# " + title + "\n")
    assert result.markdown.count(title) == 1
    assert "facebookexternalhit/1.1" in result.markdown
    assert "has been identified as a social crawler. If this is not the" in result.markdown
    assert "submit a bug report" in result.markdown
    assert "ticket_form_id=360000644872" in result.markdown


def test_title_does_not_duplicate_identical_visible_main_heading():
    result = convert_html(
        "<head><title>Google Play-Nutzungsbedingungen</title></head>"
        "<body><h1>Google Play-Nutzungsbedingungen</h1><p>Vorrang der Bedingungen.</p></body>"
    )
    assert result.markdown.count("Google Play-Nutzungsbedingungen") == 1


def test_title_is_literal_text_and_survives_hidden_body_heading():
    from bs4 import BeautifulSoup
    from markdown_it import MarkdownIt

    result = convert_html(
        "<head><title>  A &amp; B [Link](https://example.invalid) *Text* # </title></head>"
        "<body><h1 hidden>A &amp; B [Link](https://example.invalid) *Text* #</h1></body>"
    )
    rendered = BeautifulSoup(MarkdownIt().render(result.markdown), "html.parser")
    assert rendered.h1.get_text() == "A & B [Link](https://example.invalid) *Text* #"
    assert not rendered.find(["a", "em"])


def test_title_cannot_mask_unexpectedly_empty_body(monkeypatch):
    monkeypatch.setattr(
        html_to_markdown,
        "convert",
        lambda *_args, **_kwargs: SimpleNamespace(content="", warnings=[]),
    )
    with pytest.raises(HtmlConversionError, match="unexpectedly removed"):
        convert_html("<head><title>Titel</title></head><body><p>Pflichtinhalt.</p></body>")


@pytest.mark.parametrize(
    "source",
    [
        "<title>Seitentitel</title>",
        "<head hidden><title>Seitentitel</title></head>",
        "<head><style>title {display:none}</style><title>Seitentitel</title></head>",
    ],
)
def test_document_title_is_independent_of_body_visibility(source):
    assert convert_html(source).markdown.strip() == "# Seitentitel"


def test_title_entities_are_decoded_only_once():
    from bs4 import BeautifulSoup
    from markdown_it import MarkdownIt

    result = convert_html("<title>A &amp;lt;b&amp;gt; &amp;copy; \\ *x* `code`</title>")
    rendered = BeautifulSoup(MarkdownIt().render(result.markdown), "html.parser")
    assert rendered.h1.get_text() == "A &lt;b&gt; &copy; \\ *x* `code`"
    assert not rendered.find(["b", "em", "code"])


@pytest.mark.parametrize(
    "embedded",
    ["<svg><title>SVG-Titel</title></svg>", "<template><title>Vorlage</title></template>"],
)
def test_embedded_title_is_not_promoted_to_page_heading(embedded):
    result = convert_html(f"<body>{embedded}<p>Body bleibt.</p></body>")
    assert not result.markdown.startswith("# ")
    assert "Body bleibt." in result.markdown


@pytest.mark.parametrize(
    "source",
    [
        "<p>Wartung spätestens am 18.11.2026.</p>",
        "<img src='diagram.png' alt='Prüfdruck 8 bar'>",
        "<input value='Nicht freigegeben'>",
        "<p style='display:none;display:block'>Freigabe erforderlich.</p>",
        "<style>.shown{display:block!important}</style>"
        "<p class='shown' style='display:none'>Nicht für Trinkwasser.</p>",
    ],
)
def test_unexpected_empty_output_fails_closed(monkeypatch, source):
    monkeypatch.setattr(
        html_to_markdown,
        "convert",
        lambda *_args, **_kwargs: SimpleNamespace(content="", warnings=[]),
    )
    with pytest.raises(HtmlConversionError, match="unexpectedly removed"):
        convert_html(source)


def test_native_failure_is_reported(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("parser failed")

    monkeypatch.setattr(html_to_markdown, "convert", fail)
    with pytest.raises(HtmlConversionError, match="parser failed") as caught:
        convert_html("<p>Text</p>")
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize("warning", ["TRUNCATED_INPUT", "DEPTH_LIMIT_EXCEEDED"])
def test_content_loss_warnings_fail_even_for_nonempty_output(monkeypatch, warning):
    result = SimpleNamespace(
        content="Only a part of the source",
        warnings=[
            SimpleNamespace(kind=getattr(html_to_markdown.WarningKind, warning), message="limit")
        ],
    )
    monkeypatch.setattr(html_to_markdown, "convert", lambda *_args, **_kwargs: result)
    with pytest.raises(HtmlConversionError, match="truncated"):
        convert_html("<p>Complete source</p>")


def test_nonfatal_diagnostic_is_preserved(monkeypatch):
    result = SimpleNamespace(
        content="Repaired text",
        warnings=[
            SimpleNamespace(kind=html_to_markdown.WarningKind.MALFORMED_HTML, message="repaired")
        ],
    )
    monkeypatch.setattr(html_to_markdown, "convert", lambda *_args, **_kwargs: result)
    converted = convert_html("<p>Repaired text")
    assert len(converted.diagnostics) == 1
    assert "repaired" in converted.diagnostics[0]


@pytest.mark.parametrize("content", [None, b"undecoded"])
def test_invalid_native_result_fails(monkeypatch, content):
    monkeypatch.setattr(
        html_to_markdown,
        "convert",
        lambda *_args, **_kwargs: SimpleNamespace(content=content, warnings=[]),
    )
    with pytest.raises(HtmlConversionError, match="no text result"):
        convert_html("<p>Text</p>")


def test_input_must_be_decoded_text():
    with pytest.raises(HtmlConversionError, match="decoded text"):
        convert_html(b"<p>Text</p>")


def test_exact_dependency_version_is_required(monkeypatch):
    monkeypatch.setattr(html_to_markdown, "__version__", "99.0")
    with pytest.raises(HtmlConversionError, match="Unsupported"):
        convert_html("<p>Text</p>")


@pytest.mark.parametrize("tag", ["strong", "b", "em", "i"])
def test_partial_emphasis_keeps_identifier_contiguous_in_its_own_paragraph(tag):
    result = convert_html(
        f"<p>Artikel: Bauteil X<{tag}>Y</{tag}>-731, 24 V.</p><p>Andere Zeile: XY-731.</p>"
    )
    article = next(line for line in result.markdown.splitlines() if line.startswith("Artikel:"))
    assert article == "Artikel: Bauteil XY-731, 24 V."


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<p><strong>XY</strong>-731</p>", "XY-731"),
        ("<p>XY-<em>731</em></p>", "XY-731"),
        ("<p>X<em><strong>Y</strong></em>-731</p>", "XY-731"),
        ("<p>X<span><strong>Y</strong></span>-731</p>", "XY-731"),
    ],
)
def test_partial_emphasis_at_identifier_edges(html, expected):
    assert convert_html(html).markdown.strip() == expected


def test_whole_words_and_block_boundaries_keep_emphasis():
    result = convert_html("<p><strong>Wichtig</strong>.</p><p><em>Hinweis</em></p>")
    assert "**Wichtig**." in result.markdown
    assert "*Hinweis*" in result.markdown


def test_inline_link_target_survives_identifier_normalization():
    result = convert_html(
        "<p><a href='https://example.invalid/parts'>X<strong>Y</strong>-731</a></p>"
    )
    assert "[XY-731](https://example.invalid/parts)" in result.markdown
