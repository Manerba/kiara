"""Shared, local HTML conversion with explicit provenance and failure handling."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser

_PARSER_REVISION = "html-to-markdown/3.15.1"
_PROFILE_REVISION = "kiara-html-7"
_NON_CONTENT_TAGS = {"head", "script", "style", "template"}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def conversion_revision() -> tuple[str, str]:
    """Parser and profile versions used to invalidate completed HTML ingestion."""
    return _PARSER_REVISION, _PROFILE_REVISION


class HtmlConversionError(ValueError):
    """HTML could not be converted without a known loss of document content."""


@dataclass(frozen=True)
class HtmlConversion:
    markdown: str
    parser_revision: str
    profile_revision: str
    source_hash: str
    output_hash: str
    diagnostics: tuple[str, ...]

    def metadata(self) -> dict:
        """Return serializable provenance shared by file and mail ingestion."""
        return {
            "parser": "html-to-markdown",
            "content_format": "markdown",
            "source_mime_type": "text/html",
            "html_parser_revision": self.parser_revision,
            "html_profile_revision": self.profile_revision,
            "html_source_hash": self.source_hash,
            "html_output_hash": self.output_hash,
            "html_diagnostics": list(self.diagnostics),
        }


def _inline_hidden(style: str) -> bool:
    """Recognize only an unambiguous inline display:none for the empty guard."""
    if "/*" in style or "\\" in style:
        return False
    declarations = []
    for declaration in style.split(";"):
        key, separator, value = declaration.partition(":")
        if separator and key.strip().lower() == "display":
            value = value.strip().lower()
            important = bool(re.search(r"!\s*important\s*$", value))
            value = re.sub(r"!\s*important\s*$", "", value).strip()
            declarations.append((important, value))
    if not declarations:
        return False
    important = [value for priority, value in declarations if priority]
    return (important[-1] if important else declarations[-1][1]) == "none"


class _ContentProbe(HTMLParser):
    """Conservative empty-output check; this does not rewrite or render HTML."""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.has_content = False
        self._stack: list[tuple[str, bool]] = []
        # A stylesheet may override inline display. Its cascade is handled by
        # the separate HTML preparation stage, never guessed by this guard.
        self._has_stylesheet = bool(re.search(r"<(?:style|link)\b", html, re.IGNORECASE))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        suppressed = (
            bool(self._stack and self._stack[-1][1])
            or tag in _NON_CONTENT_TAGS
            or "hidden" in attributes
            or (not self._has_stylesheet and _inline_hidden(attributes.get("style") or ""))
        )
        if not suppressed and tag in {"img", "input", "area"}:
            self.has_content |= any(
                bool((attributes.get(name) or "").strip())
                for name in ("alt", "value", "src", "href")
            )
        if tag not in _VOID_TAGS:
            self._stack.append((tag, suppressed))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index][0] == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if data.strip() and not (self._stack and self._stack[-1][1]):
            self.has_content = True


def _has_content(html: str) -> bool:
    probe = _ContentProbe(html)
    probe.feed(html)
    probe.close()
    return probe.has_content


def _take_page_title(soup) -> str:
    """Read the document title before CSS; embedded SVG/template titles differ."""
    for element in soup.find_all("title"):
        if element.parent.name not in {"head", "html", "[document]"}:
            continue
        if element.find_parent(["body", "svg", "template"]):
            continue
        title = " ".join(element.get_text().split())
        # The title is added explicitly after checking body conversion. Remove
        # it here so a standalone title is not mistaken for a lost body.
        element.decompose()
        return title
    return ""


def _title_heading(title: str) -> str:
    # Keep literal Markdown syntax and already-decoded entity text literal.
    escaped = re.sub(r"([\\`*_[\]<>#|&])", r"\\\1", title)
    return f"# {escaped}\n"


def _prepare_embedded_content(soup) -> tuple[str, ...]:
    # Scripts never contribute document text. JSON-LD containing HTML can
    # otherwise make the native parser consume the following document body.
    for element in list(soup.find_all("script")):
        element.decompose()
    preserved = False
    for element in soup.find_all("noscript"):
        if element.find_parent(["head", "template"]):
            continue
        # Visibility was already resolved against the original tag and CSS.
        # With no JavaScript execution, the fallback is source content too.
        element.name = "div"
        preserved = True
    return ("html:noscript_content_preserved",) if preserved else ()


def convert_html(html: str) -> HtmlConversion:
    """Convert decoded HTML without network access or an LLM.

    An empty output is valid for an empty or explicitly hidden document. An
    unexpected empty output, native failure or known truncation is an error;
    callers must leave the source eligible for another processing attempt.
    """
    if not isinstance(html, str):
        raise HtmlConversionError("HTML input must be decoded text")
    try:
        from bs4 import BeautifulSoup
        from html_to_markdown import (
            ConversionOptions,
            PreprocessingOptions,
            WarningKind,
            __version__,
            convert,
        )

        if __version__ != "3.15.1":
            raise HtmlConversionError(f"Unsupported html-to-markdown version: {__version__}")
        from kiara_document_loaders.html_css import apply_visibility
        from kiara_document_loaders.html_inline import normalize_inline_emphasis
        from kiara_document_loaders.html_tables import normalize_tables

        soup = BeautifulSoup(html, "html.parser")
        title = _take_page_title(soup)
        preparation_diagnostics = (
            apply_visibility(soup)
            + _prepare_embedded_content(soup)
            + normalize_tables(soup)
            + normalize_inline_emphasis(soup)
        )
        title_in_body = bool(title) and any(
            " ".join(heading.get_text().split()) == title
            and not heading.find_parent(["head", "script", "style", "template", "svg"])
            for heading in soup.find_all("h1")
        )
        prepared_html = str(soup)
        result = convert(
            prepared_html,
            ConversionOptions(
                extract_metadata=False,
                compact_tables=True,
                preprocessing=PreprocessingOptions(
                    enabled=True, preset="minimal", remove_navigation=False, remove_forms=False
                ),
            ),
        )
        diagnostics = preparation_diagnostics + tuple(
            f"{warning.kind}: {warning.message}" for warning in result.warnings
        )
        if any(
            warning.kind in {WarningKind.TRUNCATED_INPUT, WarningKind.DEPTH_LIMIT_EXCEEDED}
            for warning in result.warnings
        ):
            raise HtmlConversionError(
                "HTML conversion truncated content: " + "; ".join(diagnostics)
            )
        markdown = result.content
        if not isinstance(markdown, str):
            raise HtmlConversionError("HTML conversion returned no text result")
        if not markdown.strip():
            if _has_content(prepared_html):
                raise HtmlConversionError(
                    "HTML conversion unexpectedly removed all document content"
                )
            if not title:
                diagnostics += ("empty_input" if not html.strip() else "no_visible_content",)
        if title and not title_in_body:
            markdown = _title_heading(title) + ("\n" + markdown if markdown.strip() else "")
        return HtmlConversion(
            markdown=markdown,
            parser_revision=_PARSER_REVISION,
            profile_revision=_PROFILE_REVISION,
            source_hash=hashlib.sha256(html.encode("utf-8")).hexdigest(),
            output_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            diagnostics=diagnostics,
        )
    except HtmlConversionError:
        raise
    except Exception as exc:
        raise HtmlConversionError(f"HTML conversion failed: {exc}") from exc
